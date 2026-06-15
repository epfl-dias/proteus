#define CUB_STDERR

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <iomanip>
#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "../src/include/utils/gpu_utils.h"
#include "../src/include/ssb_gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator g_allocator(true, false);

struct DeltaTileMetadata {
    uint32_t first_value;       // First value in the tile
    uint32_t total_count;       // Number of elements in tile
};

__forceinline__ __device__ int decodeElementDBin(int i, uint32_t* data_block, uint32_t first_value) {
    int for_reference = reinterpret_cast<int*>(data_block)[0];
    uint32_t miniblock_index = i / 32;
    uint32_t miniblock_bitwidths = data_block[1];

    uint32_t miniblock_word_offset = 0;
    uint32_t temp_bitwidths = miniblock_bitwidths;

    for (int j = 0; j < miniblock_index; j++) {
        uint32_t mb_bitwidth = temp_bitwidths & 255;
        uint32_t mb_total_bits = 32 * mb_bitwidth;
        miniblock_word_offset += (mb_total_bits + 31) / 32;
        temp_bitwidths >>= 8;
    }

    uint32_t bitwidth = (miniblock_bitwidths >> (miniblock_index * 8)) & 255;
    uint32_t index_into_miniblock = i & (32 - 1);
    uint32_t start_bitindex = (bitwidth * index_into_miniblock);
    uint32_t start_intindex = 2 + miniblock_word_offset + (start_bitindex >> 5);

    unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) |
                                      data_block[start_intindex];
    start_bitindex = start_bitindex & (32 - 1);
    uint32_t for_adjusted_value = (element_block >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);

    return for_reference + (int32_t)for_adjusted_value;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadDBinPack(
    DeltaTileMetadata* metadata,
    uint32_t* tile_offsets,
    uint32_t* compressed_data,
    uint32_t* shared_buffer,
    int (&items)[ITEMS_PER_THREAD],
    bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    const int BLOCKS_PER_TILE = 4;
    uint32_t first_value = metadata->first_value;

    // Load tile offsets into shared memory
    uint32_t *block_starts = &shared_buffer[0];
    if (threadIdx.x < 5) {
        block_starts[threadIdx.x] = tile_offsets[threadIdx.x];
    }
    __syncthreads();

    uint32_t* data_block = &shared_buffer[5];

    // Load compressed data into shared memory
    uint32_t start_offset = block_starts[0];
    uint32_t end_offset = block_starts[4];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint32_t index = start_offset + threadIdx.x + i * BLOCK_THREADS;
        if (index < end_offset) {
            data_block[threadIdx.x + i * BLOCK_THREADS] = compressed_data[index];
        }
    }
    __syncthreads();

    // Decode delta values for each block
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int element_idx = threadIdx.x + i * BLOCK_THREADS;
        if (is_last_tile && element_idx >= num_tile_items) {
            items[i] = 0;
        } else if (element_idx < BLOCK_THREADS * ITEMS_PER_THREAD) {
            // Find which block this element belongs to
            int block_idx = element_idx / 128;  // 128 elements per block
            int block_element_idx = element_idx % 128;

            if (block_idx < BLOCKS_PER_TILE) {
                uint32_t* block_data = data_block + (block_starts[block_idx] - start_offset);
                items[i] = decodeElementDBin(block_element_idx, block_data, first_value);
            } else {
                items[i] = 0;
            }
        } else {
            items[i] = 0;
        }
    }

    // Set first element to 0 for delta decoding
    if (threadIdx.x == 0) {
        items[0] = 0;
    }
    __syncthreads();

    // Use CUB for block operations
    typename BlockScan::TempStorage *temp_storage_scan =
        reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage *temp_storage_exchange =
        reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    // Convert from striped to blocked layout
    BlockExchange(*temp_storage_exchange).StripedToBlocked(items);
    __syncthreads();

    // Perform inclusive sum to reconstruct original values from deltas
    BlockScan(*temp_storage_scan).InclusiveSum(items, items);
    __syncthreads();

    // Convert back to striped layout for output
    BlockExchange(*temp_storage_exchange).BlockedToStriped(items);
    __syncthreads();

    // Add back the first value to complete reconstruction
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] += first_value;
    }
}

const int TILE_BLOCK_SIZE = 128;
const int ELEM_PER_THREAD = 4;
const int TILE_SIZE = TILE_BLOCK_SIZE * ELEM_PER_THREAD;
const int BLOCKS_PER_TILE = 4;
const size_t TILE_UNCOMPRESSED_BYTES = TILE_SIZE * sizeof(int);

//Compressed block
struct DeltaBlock {
    std::vector<std::vector<char>> compressed_chunks;
    size_t num_values;
};

size_t getFileSize(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filePath << std::endl;
        return 0;
    }
    size_t size = file.tellg();
    file.close();
    return size;
}

template<typename T>
T* loadColumn(const std::string& filePath, size_t num_elements) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filePath << std::endl;
        return nullptr;
    }

    T* data = new T[num_elements];
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));

    if (file.gcount() != static_cast<std::streamsize>(num_elements * sizeof(T))) {
        std::cerr << "Error: Failed to read expected amount of data from file" << std::endl;
        delete[] data;
        return nullptr;
    }

    file.close();
    return data;
}

uint deltaBinPackTile(int* tile_data, size_t tile_size, std::vector<char>& compressed_tile) {
    if (tile_size == 0) return 0;

    uint32_t block_size = 128;
    uint32_t miniblock_count = 4;
    uint32_t first_val = tile_data[0];

    // Create delta-encoded data
    std::vector<int> delta_data(tile_size);
    delta_data[0] = 0;
    for (size_t i = 1; i < tile_size; i++) {
        delta_data[i] = tile_data[i] - tile_data[i - 1];
    }

    // Create compressed data and tile offsets
    std::vector<uint32_t> compressed_data;
    std::vector<uint32_t> tile_offsets;
    uint32_t offset = 0;

    // Process each block (4 blocks per tile)
    for (int block_idx = 0; block_idx < BLOCKS_PER_TILE; block_idx++) {
        uint32_t block_start = block_idx * block_size;
        uint32_t block_end = std::min(block_start + block_size, static_cast<uint32_t>(tile_size));

        tile_offsets.push_back(offset);

        if (block_start >= tile_size) {
            continue;
        }

        int* current_block = &delta_data[block_start];
        uint32_t actual_block_size = block_end - block_start;

        // Find minimum value for FOR encoding
        int min_val = current_block[0];
        for (uint32_t i = 1; i < actual_block_size; i++) {
            if (current_block[i] < min_val) {
                min_val = current_block[i];
            }
        }

        compressed_data.push_back(min_val);  // Store reference value
        offset++;

        // Apply FOR encoding
        std::vector<uint32_t> for_adjusted_values(block_size);
        for (uint32_t i = 0; i < actual_block_size; i++) {
            for_adjusted_values[i] = static_cast<uint32_t>(current_block[i] - min_val);
        }
        // Pad remaining values
        for (uint32_t i = actual_block_size; i < block_size; i++) {
            if (actual_block_size > 0) {
                for_adjusted_values[i] = for_adjusted_values[actual_block_size - 1];
            } else {
                for_adjusted_values[i] = 0;
            }
        }

        // Calculate uniform bitwidth for the entire block (simplified approach)
        uint32_t max_for_value = 0;
        for (uint32_t i = 0; i < block_size; i++) {
            max_for_value = std::max(max_for_value, for_adjusted_values[i]);
        }

        uint32_t uniform_bitwidth;
        if (max_for_value == 0) {
            uniform_bitwidth = 1;
        } else {
            uniform_bitwidth = static_cast<uint32_t>(ceil(log2(max_for_value + 1)));
        }

        // Use uniform bitwidth for all miniblocks
        uint32_t miniblock_bitwidths[4] = {uniform_bitwidth, uniform_bitwidth, uniform_bitwidth, uniform_bitwidth};

        // Pack bitwidths
        uint32_t packed_bitwidths = miniblock_bitwidths[0] +
                                   (miniblock_bitwidths[1] << 8) +
                                   (miniblock_bitwidths[2] << 16) +
                                   (miniblock_bitwidths[3] << 24);
        compressed_data.push_back(packed_bitwidths);
        offset++;

        // Pack data for each miniblock
        uint32_t miniblock_size = block_size / miniblock_count;
        for (uint32_t miniblock = 0; miniblock < miniblock_count; miniblock++) {
            uint32_t bitwidth = uniform_bitwidth;
            uint32_t shift = 0;
            uint32_t current_word = 0;

            for (uint32_t i = 0; i < miniblock_size; i++) {
                uint32_t for_adjusted_val = for_adjusted_values[miniblock * miniblock_size + i];

                if (shift + bitwidth > 32) {
                    if (shift != 32) {
                        current_word += for_adjusted_val << shift;
                    }
                    compressed_data.push_back(current_word);
                    offset++;
                    current_word = 0;
                    shift = (shift + bitwidth) & (32 - 1);
                    current_word = for_adjusted_val >> (bitwidth - shift);
                } else {
                    current_word += for_adjusted_val << shift;
                    shift += bitwidth;
                }
            }

            if (shift > 0 || miniblock == miniblock_count - 1) {
                compressed_data.push_back(current_word);
                offset++;
                current_word = 0;
                shift = 0;
            }
        }
    }

    tile_offsets.push_back(offset);

    // Create metadata
    DeltaTileMetadata metadata;
    metadata.first_value = first_val;
    metadata.total_count = tile_size;

    // Create contiguous layout: [Metadata][Tile Offsets][Compressed Data]
    size_t metadata_size = sizeof(DeltaTileMetadata);
    size_t offsets_size = tile_offsets.size() * sizeof(uint32_t);
    size_t data_size = compressed_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + offsets_size + data_size;

    compressed_tile.resize(total_size);
    char* ptr = compressed_tile.data();

    // Copy metadata
    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;

    // Copy tile offsets
    std::memcpy(ptr, tile_offsets.data(), offsets_size);
    ptr += offsets_size;

    // Copy compressed data
    std::memcpy(ptr, compressed_data.data(), data_size);

    return offset;
}

// Create DBinPack-compressed blocks from input data, splitting it into tiles and compressing each tile
std::vector<DeltaBlock> createDeltaBlocks(int* h_data, size_t total_elements,
                                         size_t max_uncompressed_bytes_per_block, CachingDeviceAllocator& alloc) {

    std::vector<DeltaBlock> blocks;
    const size_t total_tiles = (total_elements + static_cast<size_t>(TILE_SIZE) - 1) / static_cast<size_t>(TILE_SIZE);
    const size_t max_tiles_per_block = max_uncompressed_bytes_per_block / TILE_UNCOMPRESSED_BYTES;

    size_t current_tile = 0;
    auto compression_start = std::chrono::high_resolution_clock::now();

    while (current_tile < total_tiles) {
        DeltaBlock block;
        size_t tiles_in_block = std::min(max_tiles_per_block, total_tiles - current_tile);

        for (size_t tile_idx = 0; tile_idx < tiles_in_block; tile_idx++) {
            size_t global_tile_idx = current_tile + tile_idx;
            size_t tile_start = global_tile_idx * static_cast<size_t>(TILE_SIZE);

            std::vector<int> tile_data(TILE_SIZE);
            for (size_t i = 0; i < static_cast<size_t>(TILE_SIZE); i++) {
                if (tile_start + i < total_elements) {
                    tile_data[i] = h_data[tile_start + i];
                } else {
                    tile_data[i] = h_data[total_elements - 1];
                }
            }

            std::vector<char> compressed_tile;
            deltaBinPackTile(tile_data.data(), TILE_SIZE, compressed_tile);
            block.compressed_chunks.push_back(std::move(compressed_tile));
        }

        block.num_values = std::min(tiles_in_block * static_cast<size_t>(TILE_SIZE),
                                   total_elements - (current_tile * static_cast<size_t>(TILE_SIZE)));
        blocks.push_back(std::move(block));
        current_tile += tiles_in_block;
    }

    auto compression_end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(compression_end - compression_start).count();

    return blocks;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDeltaKernel(
    DeltaTileMetadata** metadata_ptrs,
    uint32_t** tile_offset_ptrs,
    uint32_t** compressed_data_ptrs,
    int** output_ptrs,
    size_t total_tiles) {

    size_t tile_id = blockIdx.x;
    if (tile_id >= total_tiles) return;

    extern __shared__ uint32_t shared_buffer[];


    DeltaTileMetadata* metadata = metadata_ptrs[tile_id];
    uint32_t* tile_offsets = tile_offset_ptrs[tile_id];
    uint32_t* compressed_data = compressed_data_ptrs[tile_id];
    int* output = output_ptrs[tile_id];

    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];

    LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        metadata, tile_offsets, compressed_data, shared_buffer, items, false, tile_size);

    // Write output
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int output_idx = i * BLOCK_THREADS + threadIdx.x;
        if (output_idx < TILE_SIZE) {
            output[output_idx] = items[i];
        }
    }
}

float runDeltaBenchmark(
    std::vector<DeltaBlock>& blocks,
    int num_items,
    CachingDeviceAllocator& alloc,
    int* h_col_orig) {

    size_t total_uncompressed_bytes = static_cast<size_t>(num_items) * sizeof(int);
    size_t total_compressed_bytes = 0;
    size_t total_tiles = 0;

    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            total_compressed_bytes += chunk.size();
            total_tiles++;
        }
    }

    float compression_ratio = static_cast<float>(total_uncompressed_bytes) / static_cast<float>(total_compressed_bytes);

    // Prepare host-side data structures that will be reused across runs
    std::vector<char> host_buffer;
    std::vector<size_t> chunk_offsets;
    size_t total_size = 0;

    std::cout << "Preparing contiguous host buffer..." << std::endl;

    // Calculate total size for all compressed chunks
    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            total_size += chunk.size();
        }
    }

    // Create contiguous host buffer containing all compressed data
    host_buffer.resize(total_size);
    size_t current_offset = 0;

    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            std::memcpy(host_buffer.data() + current_offset, chunk.data(), chunk.size());
            chunk_offsets.push_back(current_offset);
            current_offset += chunk.size();
        }
    }

    std::cout << "Total compressed data size: " << std::fixed << std::setprecision(2)
              << (total_size / 1e6) << " MB" << std::endl;

    std::cout << "Running " << 11 << " complete allocation cycles (first run discarded)..." << std::endl;

    std::vector<float> all_run_times_ms;

    for (int run = 0; run < 11; ++run) {

        // New GPU allocation and deallocation for each run
        int* device_out = nullptr;
        char* d_buffer = nullptr;

        // Allocate output buffer
        alloc.DeviceAllocate((void**)&device_out, num_items * sizeof(int));

        // Allocate single contiguous buffer for compressed data
        alloc.DeviceAllocate((void**)&d_buffer, total_size);

        cudaMemcpy(d_buffer, host_buffer.data(), total_size, cudaMemcpyHostToDevice);

        // Build pointer arrays
        std::vector<DeltaTileMetadata*> h_metadata_ptrs;
        std::vector<uint32_t*> h_tile_offset_ptrs;
        std::vector<uint32_t*> h_compressed_data_ptrs;
        std::vector<int*> h_output_ptrs;

        h_metadata_ptrs.reserve(total_tiles);
        h_tile_offset_ptrs.reserve(total_tiles);
        h_compressed_data_ptrs.reserve(total_tiles);
        h_output_ptrs.reserve(total_tiles);

        // Build pointer arrays pointing into the single large buffer
        size_t global_output_offset = 0;

        for (size_t chunk_idx = 0; chunk_idx < chunk_offsets.size(); chunk_idx++) {
            char* chunk_ptr = d_buffer + chunk_offsets[chunk_idx];

            // Calculate pointers within chunk
            DeltaTileMetadata* metadata = reinterpret_cast<DeltaTileMetadata*>(chunk_ptr);
            uint32_t* tile_offsets = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(DeltaTileMetadata));
            uint32_t* compressed_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(DeltaTileMetadata) + (5 * sizeof(uint32_t)));
            int* output = device_out + global_output_offset;

            h_metadata_ptrs.push_back(metadata);
            h_tile_offset_ptrs.push_back(tile_offsets);
            h_compressed_data_ptrs.push_back(compressed_data);
            h_output_ptrs.push_back(output);

            global_output_offset += TILE_SIZE;
        }

        // Allocate GPU pointer arrays
        DeltaTileMetadata** d_metadata_ptrs;
        uint32_t** d_tile_offset_ptrs;
        uint32_t** d_compressed_data_ptrs;
        int** d_output_ptrs;

        alloc.DeviceAllocate((void**)&d_metadata_ptrs, total_tiles * sizeof(DeltaTileMetadata*));
        alloc.DeviceAllocate((void**)&d_tile_offset_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_compressed_data_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_output_ptrs, total_tiles * sizeof(int*));

        // Copy pointer arrays to GPU
        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(),
                  total_tiles * sizeof(DeltaTileMetadata*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tile_offset_ptrs, h_tile_offset_ptrs.data(),
                  total_tiles * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed_data_ptrs, h_compressed_data_ptrs.data(),
                  total_tiles * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_ptrs, h_output_ptrs.data(),
                  total_tiles * sizeof(int*), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        // Single kernel launch
        dim3 grid(total_tiles);
        dim3 block_dim(128);

        runDeltaKernel<128, 4><<<grid, block_dim, 4096, stream>>>(
            d_metadata_ptrs,
            d_tile_offset_ptrs,
            d_compressed_data_ptrs,
            d_output_ptrs,
            total_tiles
        );

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        // Only include runs after the first warmup run
        if (run > 0) {
            all_run_times_ms.push_back(kernel_time_ms);
            std::cout << "Run " << run << ": " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms" << std::endl;
        } else {
            std::cout << "Warmup run: " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms (discarded)" << std::endl;
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        // For the last run, copy results back to host for verification
        if (run == 10) {  // Last run
            std::cout << "Copying results back to host..." << std::endl;
            std::vector<int> h_decoded(num_items);
            cudaMemcpy(h_decoded.data(), device_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

            bool verification_passed = true;
            size_t error_count = 0;
            const size_t max_errors_to_show = 5;

            for (int i = 0; i < num_items; ++i) {
                if (h_decoded[i] != h_col_orig[i]) {
                    if (error_count < max_errors_to_show) {
                        std::cerr << "ERROR: idx " << i << " : expected=" << h_col_orig[i]
                                 << ", got=" << h_decoded[i] << std::endl;
                    }
                    verification_passed = false;
                    error_count++;
                    if (error_count >= max_errors_to_show) break;
                }
            }

            std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED") << std::endl;
        }

        // Cleanup all GPU allocations for this run
        alloc.DeviceFree(d_buffer);           // Single large buffer
        alloc.DeviceFree(d_metadata_ptrs);
        alloc.DeviceFree(d_tile_offset_ptrs);
        alloc.DeviceFree(d_compressed_data_ptrs);
        alloc.DeviceFree(d_output_ptrs);
        alloc.DeviceFree(device_out);
    }

    // Calculate metrics
    float avg_kernel_ms = std::accumulate(
        all_run_times_ms.begin(),
        all_run_times_ms.end(),
        0.0f
    ) / all_run_times_ms.size();


    double gpu_decompression_time_s = avg_kernel_ms * 1e-3;
    float overall_decompression_gbps = static_cast<float>(total_uncompressed_bytes) / (gpu_decompression_time_s * 1e9f);

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) << compression_ratio << ":1\n";
    std::cout << "GPU decompression speed: " << overall_decompression_gbps << " GB/s\n";

    return avg_kernel_ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <column_file_path> [max_uncompressed_mb]" << std::endl;
        return 1;
    }

    const std::string column_file = argv[1];
    size_t max_uncompressed_mb = 2;

    if (argc >= 3) {
        max_uncompressed_mb = std::stoul(argv[2]);
    }


    size_t max_uncompressed_bytes = max_uncompressed_mb << 20;

    size_t file_size = getFileSize(column_file);
    if (file_size == 0 || file_size % sizeof(int) != 0) {
        std::cerr << "Error: Invalid file size" << std::endl;
        return 1;
    }

    size_t num_elements = file_size / sizeof(int);

    int* h_col_orig = loadColumn<int>(column_file, num_elements);
    if (!h_col_orig) {
        std::cerr << "Error: Failed to load column data" << std::endl;
        return 1;
    }

    auto blocks = createDeltaBlocks(h_col_orig, num_elements, max_uncompressed_bytes, g_allocator);

    float total_ms = runDeltaBenchmark(blocks, num_elements, g_allocator, h_col_orig);

    delete[] h_col_orig;

    return 0;
}
