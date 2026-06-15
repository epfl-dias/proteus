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

#include "utils/gpu_utils.h"
#include "ssb_gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator g_allocator(true, false);

// Metadata structure for RLEBinPack
struct RLETileMetadata {
    uint32_t val_reference;
    uint32_t val_bitwidth;
    uint32_t val_count;
    uint32_t rl_reference;
    uint32_t rl_bitwidth;
    uint32_t rl_count;
};

__forceinline__ __device__ int decodeElementRBin(int i, uint* data_block, uint reference, uint bitwidth) {
    if (bitwidth == 0) return reference;

    uint start_bitindex = (bitwidth * i);
    uint start_intindex = (start_bitindex >> 5);
    start_bitindex = start_bitindex & (32-1);

    unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) | data_block[start_intindex];
    uint element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

    return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack(
    RLETileMetadata* metadata,
    uint32_t* val_data,
    uint32_t* rl_data,
    uint* shared_buffer,
    int (&items_value)[ITEMS_PER_THREAD],
    int (&items_run_length)[ITEMS_PER_THREAD],
    bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    uint count = metadata->val_count;
    uint num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
    uint data_offset = 0;

    // Decode values and run lengths in parallel
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && threadIdx.x + data_offset < count) {
            items_value[i] = decodeElementRBin(threadIdx.x + data_offset, val_data,
                                             metadata->val_reference, metadata->val_bitwidth);
            items_run_length[i] = decodeElementRBin(threadIdx.x + data_offset, rl_data,
                                                   metadata->rl_reference, metadata->rl_bitwidth);
        } else {
            items_value[i] = 0;
            items_run_length[i] = 0;
        }
        data_offset += num_decode;
    }

    __syncthreads();

    // Use shared memory for CUB operations
    typename BlockScan::TempStorage *temp_storage_scan =
        reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage *temp_storage_exchange =
        reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    // Working area for value placement
    uint32_t* working_area = shared_buffer + (BLOCK_THREADS * ITEMS_PER_THREAD);

    // Initialize working area
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
        }
    }
    __syncthreads();

    // Mark run boundaries
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

    // Convert boundaries to indices
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < BLOCK_THREADS * ITEMS_PER_THREAD) {
            items_run_length[i] = working_area[threadIdx.x * ITEMS_PER_THREAD + i];
        } else {
            items_run_length[i] = 0;
        }
    }
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();
    BlockExchange(*temp_storage_exchange).BlockedToStriped(items_run_length);
    __syncthreads();

    // Place values in working area
    uint rle_offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && threadIdx.x + rle_offset < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[threadIdx.x + rle_offset] = items_value[i];
        }
        rle_offset += num_decode;
    }
    __syncthreads();

    // Final value assignment
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < BLOCK_THREADS * ITEMS_PER_THREAD) {
            items_value[i] = working_area[items_run_length[i]];
        } else {
            items_value[i] = 0;
        }
    }
    __syncthreads();
}

const int TILE_BLOCK_SIZE = 128;
const int ELEM_PER_THREAD = 8;
const int TILE_SIZE = TILE_BLOCK_SIZE * ELEM_PER_THREAD;
const size_t TILE_UNCOMPRESSED_BYTES = TILE_SIZE * sizeof(int);

struct RLEBlock {
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

uint RLEBinPackTile(int* tile_data, size_t tile_size, std::vector<char>& compressed_tile) {
    if (tile_size == 0) return 0;

    std::vector<int> unique_values;
    std::vector<int> run_lengths;

    int current_value = tile_data[0];
    int current_run_length = 1;

    for (size_t i = 1; i < tile_size; i++) {
        if (tile_data[i] == current_value) {
            current_run_length++;
        } else {
            unique_values.push_back(current_value);
            run_lengths.push_back(current_run_length);
            current_value = tile_data[i];
            current_run_length = 1;
        }
    }

    unique_values.push_back(current_value);
    run_lengths.push_back(current_run_length);

    uint count = unique_values.size();

    // Compute val encoding
    int val_reference = unique_values[0];
    for (size_t i = 1; i < unique_values.size(); i++) {
        if (unique_values[i] < val_reference) {
            val_reference = unique_values[i];
        }
    }

    uint val_bitwidth = 1;
    for (size_t i = 0; i < unique_values.size(); i++) {
        uint for_adjusted = unique_values[i] - val_reference;
        if (for_adjusted > 0) {
            uint required_bits = static_cast<uint>(ceil(log2(for_adjusted + 1)));
            if (required_bits > val_bitwidth) {
                val_bitwidth = required_bits;
            }
        }
    }
    if (val_bitwidth == 0) val_bitwidth = 1;
    if (val_bitwidth > 32) val_bitwidth = 32;

    // Compute rl encoding
    int rl_reference = run_lengths[0];
    for (size_t i = 1; i < run_lengths.size(); i++) {
        if (run_lengths[i] < rl_reference) {
            rl_reference = run_lengths[i];
        }
    }

    uint rl_bitwidth = 1;
    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint for_adjusted = run_lengths[i] - rl_reference;
        if (for_adjusted > 0) {
            uint required_bits = static_cast<uint>(ceil(log2(for_adjusted + 1)));
            if (required_bits > rl_bitwidth) {
                rl_bitwidth = required_bits;
            }
        }
    }
    if (rl_bitwidth == 0) rl_bitwidth = 1;
    if (rl_bitwidth > 32) rl_bitwidth = 32;

    // Create metadata
    RLETileMetadata metadata;
    metadata.val_reference = static_cast<uint32_t>(val_reference);
    metadata.val_bitwidth = val_bitwidth;
    metadata.val_count = count;
    metadata.rl_reference = static_cast<uint32_t>(rl_reference);
    metadata.rl_bitwidth = rl_bitwidth;
    metadata.rl_count = count;

    // Pack values
    std::vector<uint32_t> val_data;
    uint current_word = 0;
    uint bits_used = 0;

    for (size_t i = 0; i < unique_values.size(); i++) {
        uint for_adjusted = unique_values[i] - val_reference;

        if (bits_used + val_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += val_bitwidth;

            if (bits_used == 32 || i == unique_values.size() - 1) {
                val_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint bits_in_current = 32 - bits_used;
            uint bits_in_next = val_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            val_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == unique_values.size() - 1 && bits_used > 0) {
                val_data.push_back(current_word);
            }
        }
    }

    // Pack run lengths
    std::vector<uint32_t> rl_data;
    current_word = 0;
    bits_used = 0;

    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint for_adjusted = run_lengths[i] - rl_reference;

        if (bits_used + rl_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += rl_bitwidth;

            if (bits_used == 32 || i == run_lengths.size() - 1) {
                rl_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint bits_in_current = 32 - bits_used;
            uint bits_in_next = rl_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            rl_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == run_lengths.size() - 1 && bits_used > 0) {
                rl_data.push_back(current_word);
            }
        }
    }

    // Create contiguous layout: [Metadata][Val Data][RL Data]
    size_t metadata_size = sizeof(RLETileMetadata);
    size_t val_data_size = val_data.size() * sizeof(uint32_t);
    size_t rl_data_size = rl_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + val_data_size + rl_data_size;

    compressed_tile.resize(total_size);
    char* ptr = compressed_tile.data();

    // Copy metadata
    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;

    // Copy val data
    std::memcpy(ptr, val_data.data(), val_data_size);
    ptr += val_data_size;

    // Copy rl data
    std::memcpy(ptr, rl_data.data(), rl_data_size);

    return count;
}
// Create RLEBinPack-compressed blocks from input data, splitting it into tiles and compressing each tile
std::vector<RLEBlock> createRLEBlocks(int* h_data, size_t total_elements,
                                     size_t max_uncompressed_bytes_per_block, CachingDeviceAllocator& alloc) {

    std::vector<RLEBlock> blocks;
    const size_t total_tiles = (total_elements + static_cast<size_t>(TILE_SIZE) - 1) / static_cast<size_t>(TILE_SIZE);

    const size_t max_tiles_per_block = max_uncompressed_bytes_per_block / TILE_UNCOMPRESSED_BYTES;

    size_t current_tile = 0;
    auto compression_start = std::chrono::high_resolution_clock::now();

    while (current_tile < total_tiles) {
        RLEBlock block;

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
            RLEBinPackTile(tile_data.data(), TILE_SIZE, compressed_tile);

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
__global__ void runRLEKernel(
    RLETileMetadata** metadata_ptrs,
    uint32_t** val_data_ptrs,
    uint32_t** rl_data_ptrs,
    int** output_ptrs,
    size_t total_tiles) {

    size_t tile_id = blockIdx.x;
    if (tile_id >= total_tiles) return;

    extern __shared__ uint shared_buffer[];

    RLETileMetadata* metadata = metadata_ptrs[tile_id];
    uint32_t* val_data = val_data_ptrs[tile_id];
    uint32_t* rl_data = rl_data_ptrs[tile_id];
    int* output = output_ptrs[tile_id];

    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int val_block[ITEMS_PER_THREAD];
    int rl_block[ITEMS_PER_THREAD];

    LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        metadata, val_data, rl_data, shared_buffer,
        val_block, rl_block, false, tile_size);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        output[i * BLOCK_THREADS + threadIdx.x] = val_block[i];
    }
}

float runRLEBenchmark(
    std::vector<RLEBlock>& blocks,
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
        std::vector<RLETileMetadata*> h_metadata_ptrs;
        std::vector<uint32_t*> h_val_data_ptrs;
        std::vector<uint32_t*> h_rl_data_ptrs;
        std::vector<int*> h_output_ptrs;

        h_metadata_ptrs.reserve(total_tiles);
        h_val_data_ptrs.reserve(total_tiles);
        h_rl_data_ptrs.reserve(total_tiles);
        h_output_ptrs.reserve(total_tiles);

        // Build pointer arrays pointing into the single large buffer
        size_t global_output_offset = 0;

        for (size_t chunk_idx = 0; chunk_idx < chunk_offsets.size(); chunk_idx++) {
            char* chunk_ptr = d_buffer + chunk_offsets[chunk_idx];

            // Calculate pointers within chunk
            RLETileMetadata* metadata = reinterpret_cast<RLETileMetadata*>(chunk_ptr);

            // Read metadata to calculate val section size
            RLETileMetadata temp_metadata;
            cudaMemcpy(&temp_metadata, metadata, sizeof(RLETileMetadata), cudaMemcpyDeviceToHost);

            uint32_t val_packed_words = ((temp_metadata.val_count * temp_metadata.val_bitwidth + 31) / 32);
            uint32_t val_section_size = sizeof(RLETileMetadata) + (val_packed_words * sizeof(uint32_t));

            uint32_t* val_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(RLETileMetadata));
            uint32_t* rl_data = reinterpret_cast<uint32_t*>(chunk_ptr + val_section_size);
            int* output = device_out + global_output_offset;

            h_metadata_ptrs.push_back(metadata);
            h_val_data_ptrs.push_back(val_data);
            h_rl_data_ptrs.push_back(rl_data);
            h_output_ptrs.push_back(output);

            global_output_offset += TILE_SIZE;
        }

        // Allocate GPU pointer arrays
        RLETileMetadata** d_metadata_ptrs;
        uint32_t** d_val_data_ptrs;
        uint32_t** d_rl_data_ptrs;
        int** d_output_ptrs;

        alloc.DeviceAllocate((void**)&d_metadata_ptrs, total_tiles * sizeof(RLETileMetadata*));
        alloc.DeviceAllocate((void**)&d_val_data_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_rl_data_ptrs, total_tiles * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_output_ptrs, total_tiles * sizeof(int*));

        // Copy pointer arrays to GPU
        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(),
                  total_tiles * sizeof(RLETileMetadata*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_ptrs, h_val_data_ptrs.data(),
                  total_tiles * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rl_data_ptrs, h_rl_data_ptrs.data(),
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

        runRLEKernel<128, 8><<<grid, block_dim, 8192, stream>>>(
            d_metadata_ptrs,
            d_val_data_ptrs,
            d_rl_data_ptrs,
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
        alloc.DeviceFree(d_val_data_ptrs);
        alloc.DeviceFree(d_rl_data_ptrs);
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

    auto blocks = createRLEBlocks(h_col_orig, num_elements, max_uncompressed_bytes, g_allocator);

    float total_ms = runRLEBenchmark(blocks, num_elements, g_allocator, h_col_orig);

    for (auto& block : blocks) {
        // No cleanup needed - compressed_chunks are automatically freed
    }

    delete[] h_col_orig;

    return 0;
}
