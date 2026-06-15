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
#include <iomanip>
#include <numeric>
#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "../src/include/utils/gpu_utils.h"
#include "../src/include/ssb_gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator g_allocator(true, false);

// Metadata
struct BinPackTileMetadata {
    uint32_t total_count;       // Number of elements in tile
};

// Decode a single element from a compressed miniblock
__forceinline__ __device__ int decodeElement(int i, uint miniblock_index, uint index_into_miniblock, uint* data_block, uint* bitwidths, uint* offsets) {
    int reference = reinterpret_cast<int*>(data_block)[0];

    uint miniblock_offset = offsets[miniblock_index];
    uint bitwidth = bitwidths[miniblock_index];

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 2 + (start_bitindex >> 5);

    start_bitindex = start_bitindex & (32-1);

    unsigned long long element_block = (((unsigned long long)data_block[miniblock_offset + start_intindex + 1]) << 32) | data_block[miniblock_offset + start_intindex];
    uint element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

    return reference + element;
}

// Load and decode a BinPack tile in shared memory
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack(
    uint* tile_offsets,
    uint* compressed_data,
    uint* shared_buffer,
    int (&items)[ITEMS_PER_THREAD],
    bool is_last_tile,
    int num_tile_items) {

    int threadId = threadIdx.x;

    // Part of shared memory stores offsets of miniblocks
    uint* block_starts = &shared_buffer[0];

    if (threadId < ITEMS_PER_THREAD + 1) {
        block_starts[threadId] = tile_offsets[threadId];
    }
    __syncthreads();

    // Use shared memory for data loading
    uint* data_block = &shared_buffer[ITEMS_PER_THREAD + 1];

    uint start_offset = block_starts[0];
    uint end_offset = block_starts[ITEMS_PER_THREAD];

    // Load compressed data into shared memory
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = start_offset + threadIdx.x + (i * BLOCK_THREADS);
        if (index < end_offset) {
            data_block[threadIdx.x + (i * BLOCK_THREADS)] = compressed_data[index];
        }
    }
    __syncthreads();

    for (int block_idx = 0; block_idx < ITEMS_PER_THREAD; block_idx++) {
        uint block_start_offset = block_starts[block_idx];
        uint* current_data_block = data_block + (block_start_offset - start_offset);

        int reference = reinterpret_cast<int*>(current_data_block)[0];
        uint miniblock_bitwidths = current_data_block[1];

        uint bitwidths[4];
        uint offsets[4];
        uint cumulative_offset = 0;

        for (int mb = 0; mb < 4; mb++) {
            bitwidths[mb] = (miniblock_bitwidths >> (mb << 3)) & 255; // unpack each miniblock bitwidth
            offsets[mb] = cumulative_offset;
            cumulative_offset += bitwidths[mb];
        }

        uint miniblock_index = threadId >> 5; // thread index / 32
        uint index_into_miniblock = threadId & (32 - 1); // thread index % 32

        if (miniblock_index < 4) {
            items[block_idx] = decodeElement(threadId, miniblock_index, index_into_miniblock,
                                           current_data_block, bitwidths, offsets);
        } else {
            items[block_idx] = reference;
        }

        if (is_last_tile) {
            int global_element_idx = block_idx * BLOCK_THREADS + threadId;
            if (global_element_idx >= num_tile_items) {
                items[block_idx] = 0;
            }
        }
    }
}

const int TILE_BLOCK_SIZE = 128;
const int ELEM_PER_THREAD = 4;
const int TILE_SIZE = TILE_BLOCK_SIZE * ELEM_PER_THREAD;
const int BLOCKS_PER_TILE = 4;
const size_t TILE_UNCOMPRESSED_BYTES = TILE_SIZE * sizeof(int);

// Block with binpacked tiles
struct BinPackBlock {
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

uint compressTileBinPack(int* tile_data, size_t tile_size, std::vector<char>& compressed_tile) {
    if (tile_size == 0) return 0;

    uint offset = 0;
    uint block_size = 128;
    uint miniblock_count = 4;
    uint miniblock_size = block_size / miniblock_count;

    // Create metadata
    BinPackTileMetadata metadata;
    metadata.total_count = tile_size;

    std::vector<uint> compressed_data;
    std::vector<uint> tile_offsets;

    for (int block_idx = 0; block_idx < BLOCKS_PER_TILE; block_idx++) {
        uint block_start = block_idx * block_size;
        uint block_end = std::min(block_start + block_size, static_cast<uint>(tile_size));

        tile_offsets.push_back(offset);

        if (block_start >= tile_size) {
            continue;
        }

        uint min_val = static_cast<uint>(tile_data[block_start]);
        for (uint i = block_start + 1; i < block_end; i++) {
            uint val = static_cast<uint>(tile_data[i]);
            if (val < min_val) min_val = val;
        }

        compressed_data.push_back(min_val);
        offset++;

        uint miniblock_bitwidths[4] = {0, 0, 0, 0};
        for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
            for (uint i = 0; i < miniblock_size; i++) {
                uint global_idx = block_start + miniblock * miniblock_size + i;
                if (global_idx < block_end) {
                    uint val = static_cast<uint>(tile_data[global_idx]);
                    uint for_adjusted = val - min_val;
                    uint bitwidth = (for_adjusted == 0) ? 1 : static_cast<uint>(ceil(log2(for_adjusted + 1)));
                    if (bitwidth > miniblock_bitwidths[miniblock]) {
                        miniblock_bitwidths[miniblock] = bitwidth;
                    }
                }
            }
        }

        // Use uniform bitwidth across all miniblocks (like legacy implementation)
        uint max_bitwidth = miniblock_bitwidths[0];
        for (int i = 1; i < miniblock_count; i++) {
            max_bitwidth = std::max(max_bitwidth, miniblock_bitwidths[i]);
        }
        for (int i = 0; i < miniblock_count; i++) {
            miniblock_bitwidths[i] = max_bitwidth;
        }

        uint packed_bitwidths = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) +
                               (miniblock_bitwidths[2] << 16) + (miniblock_bitwidths[3] << 24);
        compressed_data.push_back(packed_bitwidths);
        offset++;

        for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
            uint bitwidth = miniblock_bitwidths[miniblock];
            uint shift = 0;
            uint current_word = 0;

            for (uint i = 0; i < miniblock_size; i++) {
                uint global_idx = block_start + miniblock * miniblock_size + i;
                uint for_adjusted_val = 0;

                if (global_idx < block_end) {
                    uint val = static_cast<uint>(tile_data[global_idx]);
                    for_adjusted_val = val - min_val;
                } else {
                    if (block_end > block_start) {
                        uint val = static_cast<uint>(tile_data[block_end - 1]);
                        for_adjusted_val = val - min_val;
                    } else {
                        for_adjusted_val = 0;
                    }
                }

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

    // Create contiguous layout: [Metadata][Tile Offsets][Compressed Data]
    size_t metadata_size = sizeof(BinPackTileMetadata);
    size_t offsets_size = tile_offsets.size() * sizeof(uint);
    size_t data_size = compressed_data.size() * sizeof(uint);
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

// Create BinPack-compressed blocks from input data, splitting it into tiles and compressing each tile
std::vector<BinPackBlock> createBinPackBlocks(int* h_data, size_t total_elements,
                                              size_t max_uncompressed_bytes_per_block, CachingDeviceAllocator& alloc) {

    std::vector<BinPackBlock> blocks;
    const size_t total_tiles = (total_elements + static_cast<size_t>(TILE_SIZE) - 1) / static_cast<size_t>(TILE_SIZE);
    const size_t max_tiles_per_block = max_uncompressed_bytes_per_block / TILE_UNCOMPRESSED_BYTES;

    size_t current_tile = 0;
    auto compression_start = std::chrono::high_resolution_clock::now();

    while (current_tile < total_tiles) {
        BinPackBlock block;
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
            compressTileBinPack(tile_data.data(), TILE_SIZE, compressed_tile);
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
__global__ void runBinPackKernel(
    BinPackTileMetadata** metadata_ptrs,
    uint32_t** tile_offset_ptrs,
    uint32_t** compressed_data_ptrs,
    int** output_ptrs,
    size_t total_tiles) {

    size_t tile_id = blockIdx.x;
    if (tile_id >= total_tiles) return;

    extern __shared__ uint32_t shared_buffer[];

    BinPackTileMetadata* metadata = metadata_ptrs[tile_id];
    uint32_t* tile_offsets = tile_offset_ptrs[tile_id];
    uint32_t* compressed_data = compressed_data_ptrs[tile_id];
    int* output = output_ptrs[tile_id];

    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];

    LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        tile_offsets, compressed_data, shared_buffer, items, false, tile_size);

    // Write output
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int output_idx = i * BLOCK_THREADS + threadIdx.x;
        if (output_idx < TILE_SIZE) {
            output[output_idx] = items[i];
        }
    }
}

float runBinPackBenchmark(
    std::vector<BinPackBlock>& blocks,
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
        std::vector<BinPackTileMetadata*> h_metadata_ptrs;
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
            BinPackTileMetadata* metadata = reinterpret_cast<BinPackTileMetadata*>(chunk_ptr);
            uint32_t* tile_offsets = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(BinPackTileMetadata));
            uint32_t* compressed_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(BinPackTileMetadata) + (5 * sizeof(uint32_t)));
            int* output = device_out + global_output_offset;

            h_metadata_ptrs.push_back(metadata);
            h_tile_offset_ptrs.push_back(tile_offsets);
            h_compressed_data_ptrs.push_back(compressed_data);
            h_output_ptrs.push_back(output);

            global_output_offset += TILE_SIZE;
        }

        // Allocate GPU pointer arrays
        BinPackTileMetadata** d_metadata_ptrs;
        uint32_t** d_tile_offset_ptrs;
        uint32_t** d_compressed_data_ptrs;
        int** d_output_ptrs;

        alloc.DeviceAllocate((void**)&d_metadata_ptrs, total_tiles * sizeof(BinPackTileMetadata*));
        alloc.DeviceAllocate((void**)&d_tile_offset_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_compressed_data_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_output_ptrs, total_tiles * sizeof(int*));

        // Copy pointer arrays to GPU
        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(),
                  total_tiles * sizeof(BinPackTileMetadata*), cudaMemcpyHostToDevice);
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

        runBinPackKernel<128, 4><<<grid, block_dim, 4096, stream>>>(
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

    auto blocks = createBinPackBlocks(h_col_orig, num_elements, max_uncompressed_bytes, g_allocator);

    float total_ms = runBinPackBenchmark(blocks, num_elements, g_allocator, h_col_orig);

    delete[] h_col_orig;

    return 0;
}
