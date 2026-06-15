//
// Created by boffa on 08/08/25.
// Fixed version with proper shared memory layout
//

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <fstream>

using namespace std;
using namespace cub;

CachingDeviceAllocator g_allocator(true, false);

#define TILE_BLOCK_SIZE 128
#define ELEM_PER_THREAD 4
#define TILE_SIZE (TILE_BLOCK_SIZE * ELEM_PER_THREAD)
#define MAX_COLUMNS 3

enum CompressionType {
    RFOR,
    UNCOMPRESSED
};

struct Tuple {
    int values[MAX_COLUMNS];
};

struct RLETileMetadata {
    uint32_t val_reference;
    uint32_t val_bitwidth;
    uint32_t val_count;
    uint32_t rl_reference;
    uint32_t rl_bitwidth;
    uint32_t rl_count;
};

struct RLEColumn {
    std::vector<std::vector<char>> compressed_chunks;
    size_t num_values;
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
    int (&items_value)[ITEMS_PER_THREAD]) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    uint count = metadata->val_count;
    uint num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
    uint data_offset = 0;

    int items_run_length[ITEMS_PER_THREAD];

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

    typename BlockScan::TempStorage* temp_storage_scan =
        reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage* temp_storage_exchange =
        reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    uint32_t* working_area = shared_buffer + (BLOCK_THREADS * ITEMS_PER_THREAD);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
        }
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

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

    uint rle_offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && threadIdx.x + rle_offset < BLOCK_THREADS * ITEMS_PER_THREAD) {
            working_area[threadIdx.x + rle_offset] = items_value[i];
        }
        rle_offset += num_decode;
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < BLOCK_THREADS * ITEMS_PER_THREAD) {
            items_value[i] = working_area[items_run_length[i]];
        } else {
            items_value[i] = 0;
        }
    }
    __syncthreads();
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_tuple(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    Tuple* d_output,
    size_t tiles_per_column,
    size_t total_elements,
    int num_columns) {

    size_t tile_id = blockIdx.x;
    if (tile_id >= tiles_per_column) return;

    extern __shared__ uint shared_buffer[];

    const size_t cub_temp_offset = 4096 / sizeof(uint);
    int* column_elements = reinterpret_cast<int*>(shared_buffer + cub_temp_offset);

    // Process columns sequentially
    for (int col = 0; col < num_columns; col++) {
        size_t ptr_index = col * tiles_per_column + tile_id;

        RLETileMetadata* metadata = d_metadata_ptrs[ptr_index];
        uint32_t* val_data = d_val_data_ptrs[ptr_index];
        uint32_t* rl_data = d_rl_data_ptrs[ptr_index];

        // Decompress into thread-local registers
        int temp_elements[ITEMS_PER_THREAD];
        LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
            metadata, val_data, rl_data, shared_buffer, temp_elements);

        // Store decompressed data from registers to shared memory
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            size_t intermediate_idx = col * TILE_SIZE + threadIdx.x + i * BLOCK_THREADS;
            column_elements[intermediate_idx] = temp_elements[i];
        }

        __syncthreads();
    }

    // Form tuples from all decompressed columns
    size_t tile_base_offset = tile_id * TILE_SIZE;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t global_element_idx = tile_base_offset + i * BLOCK_THREADS + threadIdx.x;

        if (global_element_idx < total_elements) {
            Tuple tup;
            for (int col = 0; col < MAX_COLUMNS; col++) {
                tup.values[col] = 0;
            }

            for (int col = 0; col < num_columns; col++) {
                size_t intermediate_idx = col * TILE_SIZE + i * BLOCK_THREADS + threadIdx.x;
                tup.values[col] = column_elements[intermediate_idx];
            }

            d_output[global_element_idx] = tup;
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void uncompressed_tiles_to_tuple(
    int** d_column_data,
    Tuple* d_output,
    size_t total_elements,
    int num_columns) {

    size_t tile_id = blockIdx.x;
    size_t tile_base_offset = tile_id * TILE_SIZE;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t global_element_idx = tile_base_offset + i * BLOCK_THREADS + threadIdx.x;

        if (global_element_idx < total_elements) {
            Tuple tup;
            for (int col = 0; col < MAX_COLUMNS; col++) {
                tup.values[col] = 0;
            }

            for (int col = 0; col < num_columns; col++) {
                tup.values[col] = d_column_data[col][global_element_idx];
            }

            d_output[global_element_idx] = tup;
        }
    }
}

uint compressRFORTile(int* tile_data, size_t tile_size, std::vector<char>& compressed_tile) {
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

    int val_reference = *std::min_element(unique_values.begin(), unique_values.end());
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

    int rl_reference = *std::min_element(run_lengths.begin(), run_lengths.end());
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

    RLETileMetadata metadata;
    metadata.val_reference = static_cast<uint32_t>(val_reference);
    metadata.val_bitwidth = val_bitwidth;
    metadata.val_count = count;
    metadata.rl_reference = static_cast<uint32_t>(rl_reference);
    metadata.rl_bitwidth = rl_bitwidth;
    metadata.rl_count = count;

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

    size_t metadata_size = sizeof(RLETileMetadata);
    size_t val_data_size = val_data.size() * sizeof(uint32_t);
    size_t rl_data_size = rl_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + val_data_size + rl_data_size;

    compressed_tile.resize(total_size);
    char* ptr = compressed_tile.data();

    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;
    std::memcpy(ptr, val_data.data(), val_data_size);
    ptr += val_data_size;
    std::memcpy(ptr, rl_data.data(), rl_data_size);

    return count;
}

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

std::vector<RLEColumn> createRLEColumns(const std::vector<std::vector<int>>& column_data,
                                        size_t total_elements, int num_columns,
                                        size_t max_uncompressed_bytes_per_block) {

    std::cout << "Compressing " << num_columns << " columns..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<RLEColumn> columns(num_columns);
    const size_t total_tiles = (total_elements + TILE_SIZE - 1) / TILE_SIZE;
    const size_t max_tiles_per_block = max_uncompressed_bytes_per_block / (TILE_SIZE * sizeof(int));

    for (int col = 0; col < num_columns; col++) {
        size_t current_tile = 0;

        while (current_tile < total_tiles) {
            size_t tiles_in_block = std::min(max_tiles_per_block, total_tiles - current_tile);

            for (size_t tile_idx = 0; tile_idx < tiles_in_block; tile_idx++) {
                size_t global_tile_idx = current_tile + tile_idx;
                size_t tile_start = global_tile_idx * TILE_SIZE;

                std::vector<int> tile_data(TILE_SIZE);
                for (size_t i = 0; i < TILE_SIZE; i++) {
                    if (tile_start + i < total_elements) {
                        tile_data[i] = column_data[col][tile_start + i];
                    } else {
                        tile_data[i] = column_data[col][total_elements - 1];
                    }
                }

                std::vector<char> compressed_tile;
                compressRFORTile(tile_data.data(), TILE_SIZE, compressed_tile);
                columns[col].compressed_chunks.push_back(std::move(compressed_tile));
            }

            current_tile += tiles_in_block;
        }

        columns[col].num_values = total_elements;
        std::cout << "Column " << col << " compressed into " << columns[col].compressed_chunks.size() << " tiles" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(end - start).count();

    std::cout << "Compression completed in " << std::fixed << std::setprecision(1)
              << compression_time << " seconds" << std::endl;

    return columns;
}

float run_compressed_blocks_to_tuple(const std::vector<std::vector<int>>& column_data,
                                   size_t total_elements, int num_columns,
                                   std::vector<Tuple>& host_output) {

    const size_t max_block_size = 2 * 1024 * 1024;

    auto columns = createRLEColumns(column_data, total_elements, num_columns, max_block_size);

    size_t total_tiles = columns[0].compressed_chunks.size();
    size_t total_chunks = num_columns * total_tiles;

    // Prepare separate host buffers per column
    std::vector<std::vector<char>> h_col_buffers(num_columns);
    std::vector<std::vector<size_t>> chunk_offsets_per_col(num_columns);
    std::vector<size_t> col_total_sizes(num_columns);

    size_t total_compressed_bytes = 0;

    std::cout << "Preparing host buffers for " << num_columns << " columns..." << std::endl;

    for (int col = 0; col < num_columns; col++) {
        // Calculate total size for this column
        size_t col_total_size = 0;
        for (size_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            col_total_size += columns[col].compressed_chunks[tile_idx].size();
        }
        col_total_sizes[col] = col_total_size;

        // Create contiguous host buffer for this column
        h_col_buffers[col].resize(col_total_size);

        size_t current_offset = 0;
        for (size_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            const auto& chunk = columns[col].compressed_chunks[tile_idx];
            total_compressed_bytes += chunk.size();

            std::memcpy(h_col_buffers[col].data() + current_offset, chunk.data(), chunk.size());
            chunk_offsets_per_col[col].push_back(current_offset);
            current_offset += chunk.size();
        }

        std::cout << "Column " << col << " prepared: " << std::fixed << std::setprecision(2)
                  << (col_total_size / 1e6) << " MB" << std::endl;
    }

    std::cout << "Total compressed data size: "
         << fixed << setprecision(2) << (total_compressed_bytes / 1e9) << " GB" << endl;

    // Calculate shared memory requirements
    size_t cub_temp_size = 4096;
    size_t intermediate_layer_size = num_columns * TILE_SIZE * sizeof(int);
    size_t total_shared_mem_size = cub_temp_size + intermediate_layer_size;

    std::cout << "Shared memory per block: " << std::fixed << std::setprecision(1)
              << (total_shared_mem_size / 1024.0) << " KB" << std::endl;

    std::cout << "Running " << 11 << " complete allocation cycles (first run discarded)..." << std::endl;

    std::vector<float> all_run_times_ms;

    for (int run = 0; run < 11; ++run) {

        Tuple* d_output;
        std::vector<char*> d_column_buffers;
        std::vector<RLETileMetadata*> h_metadata_ptrs;
        std::vector<uint32_t*> h_val_data_ptrs;
        std::vector<uint32_t*> h_rl_data_ptrs;

        h_metadata_ptrs.resize(total_chunks);
        h_val_data_ptrs.resize(total_chunks);
        h_rl_data_ptrs.resize(total_chunks);

        cudaError_t result = g_allocator.DeviceAllocate((void**)&d_output, total_elements * sizeof(Tuple));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate output tuples in run " << run << ": " << cudaGetErrorString(result) << endl;
            return -1;
        }

        bool allocation_failed = false;

        // Allocate and upload separate buffer for each column
        for (int col = 0; col < num_columns && !allocation_failed; col++) {
            char* d_col_buffer;
            result = g_allocator.DeviceAllocate((void**)&d_col_buffer, col_total_sizes[col]);
            if (result != cudaSuccess) {
                cout << "[ERROR] Failed to allocate column buffer " << col << " in run " << run << ": " << cudaGetErrorString(result) << endl;
                allocation_failed = true;
                break;
            }

            cudaMemcpy(d_col_buffer, h_col_buffers[col].data(), col_total_sizes[col], cudaMemcpyHostToDevice);
            d_column_buffers.push_back(d_col_buffer);

            // Build pointer arrays for this column
            for (size_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
                char* chunk_ptr = d_col_buffer + chunk_offsets_per_col[col][tile_idx];

                RLETileMetadata* metadata = reinterpret_cast<RLETileMetadata*>(chunk_ptr);

                RLETileMetadata temp_metadata;
                cudaMemcpy(&temp_metadata, metadata, sizeof(RLETileMetadata), cudaMemcpyDeviceToHost);

                uint32_t val_packed_words = ((temp_metadata.val_count * temp_metadata.val_bitwidth + 31) / 32);
                uint32_t val_section_size = sizeof(RLETileMetadata) + (val_packed_words * sizeof(uint32_t));

                uint32_t* val_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(RLETileMetadata));
                uint32_t* rl_data = reinterpret_cast<uint32_t*>(chunk_ptr + val_section_size);

                size_t ptr_index = col * total_tiles + tile_idx;
                h_metadata_ptrs[ptr_index] = metadata;
                h_val_data_ptrs[ptr_index] = val_data;
                h_rl_data_ptrs[ptr_index] = rl_data;
            }
        }

        if (allocation_failed) {
            for (char* ptr : d_column_buffers) {
                g_allocator.DeviceFree(ptr);
            }
            g_allocator.DeviceFree(d_output);
            return -1;
        }

        RLETileMetadata** d_metadata_ptrs;
        uint32_t** d_val_data_ptrs;
        uint32_t** d_rl_data_ptrs;

        result = g_allocator.DeviceAllocate((void**)&d_metadata_ptrs, total_chunks * sizeof(RLETileMetadata*));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate metadata pointers in run " << run << ": " << cudaGetErrorString(result) << endl;
            for (char* ptr : d_column_buffers) g_allocator.DeviceFree(ptr);
            g_allocator.DeviceFree(d_output);
            return -1;
        }

        result = g_allocator.DeviceAllocate((void**)&d_val_data_ptrs, total_chunks * sizeof(uint32_t*));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate value data pointers in run " << run << ": " << cudaGetErrorString(result) << endl;
            for (char* ptr : d_column_buffers) g_allocator.DeviceFree(ptr);
            g_allocator.DeviceFree(d_output);
            g_allocator.DeviceFree(d_metadata_ptrs);
            return -1;
        }

        result = g_allocator.DeviceAllocate((void**)&d_rl_data_ptrs, total_chunks * sizeof(uint32_t*));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate run length pointers in run " << run << ": " << cudaGetErrorString(result) << endl;
            for (char* ptr : d_column_buffers) g_allocator.DeviceFree(ptr);
            g_allocator.DeviceFree(d_output);
            g_allocator.DeviceFree(d_metadata_ptrs);
            g_allocator.DeviceFree(d_val_data_ptrs);
            return -1;
        }

        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(), total_chunks * sizeof(RLETileMetadata*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_ptrs, h_val_data_ptrs.data(), total_chunks * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rl_data_ptrs, h_rl_data_ptrs.data(), total_chunks * sizeof(uint32_t*), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        dim3 grid(total_tiles);
        dim3 block_dim(TILE_BLOCK_SIZE);

        compressed_block_to_tuple<TILE_BLOCK_SIZE, ELEM_PER_THREAD><<<grid, block_dim, total_shared_mem_size, stream>>>(
            d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_output,
            total_tiles, total_elements, num_columns);

        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            cout << "[ERROR] Kernel launch failed in run " << run << ": " << cudaGetErrorString(kernel_error) << endl;
        }

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        if (run > 0) {
            all_run_times_ms.push_back(kernel_time_ms);
            std::cout << "Run " << run << ": " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms" << std::endl;
        } else {
            std::cout << "Warmup run: " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms (discarded)" << std::endl;
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        if (run == 10) {
            std::cout << "Copying results back to host..." << std::endl;
            host_output.resize(total_elements);
            cudaMemcpy(host_output.data(), d_output, total_elements * sizeof(Tuple), cudaMemcpyDeviceToHost);
        }

        // Cleanup
        for (char* ptr : d_column_buffers) {
            g_allocator.DeviceFree(ptr);
        }
        g_allocator.DeviceFree(d_metadata_ptrs);
        g_allocator.DeviceFree(d_val_data_ptrs);
        g_allocator.DeviceFree(d_rl_data_ptrs);
        g_allocator.DeviceFree(d_output);
    }

    float elapsed_ms = std::accumulate(all_run_times_ms.begin(), all_run_times_ms.end(), 0.0f) / all_run_times_ms.size();

    return elapsed_ms;
}

float run_uncompressed_blocks_to_tuple(const std::vector<std::vector<int>>& column_data,
                                     size_t total_elements, int num_columns,
                                     std::vector<Tuple>& host_output) {

    size_t total_tiles = (total_elements + TILE_SIZE - 1) / TILE_SIZE;

    std::cout << "Running " << 11 << " complete allocation cycles (first run discarded)..." << std::endl;

    std::vector<float> all_run_times_ms;

    for (int run = 0; run < 11; ++run) {

        Tuple* d_output;
        std::vector<int*> d_column_data(num_columns);
        int** d_column_ptrs;

        cudaError_t result = g_allocator.DeviceAllocate((void**)&d_output, total_elements * sizeof(Tuple));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate output tuples in run " << run << ": " << cudaGetErrorString(result) << endl;
            return -1;
        }

        bool allocation_failed = false;
        for (int col = 0; col < num_columns && !allocation_failed; col++) {
            result = g_allocator.DeviceAllocate((void**)&d_column_data[col], total_elements * sizeof(int));
            if (result != cudaSuccess) {
                cout << "[ERROR] Failed to allocate column data " << col << " in run " << run << ": " << cudaGetErrorString(result) << endl;
                allocation_failed = true;
                break;
            }

            cudaMemcpy(d_column_data[col], column_data[col].data(), total_elements * sizeof(int), cudaMemcpyHostToDevice);
        }

        if (allocation_failed) {
            for (int col = 0; col < num_columns; col++) {
                if (d_column_data[col]) g_allocator.DeviceFree(d_column_data[col]);
            }
            g_allocator.DeviceFree(d_output);
            return -1;
        }

        result = g_allocator.DeviceAllocate((void**)&d_column_ptrs, num_columns * sizeof(int*));
        if (result != cudaSuccess) {
            cout << "[ERROR] Failed to allocate column pointers in run " << run << ": " << cudaGetErrorString(result) << endl;
            for (int col = 0; col < num_columns; col++) {
                g_allocator.DeviceFree(d_column_data[col]);
            }
            g_allocator.DeviceFree(d_output);
            return -1;
        }

        cudaMemcpy(d_column_ptrs, d_column_data.data(), num_columns * sizeof(int*), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        dim3 grid(total_tiles);
        dim3 block_dim(TILE_BLOCK_SIZE);

        uncompressed_tiles_to_tuple<TILE_BLOCK_SIZE, ELEM_PER_THREAD><<<grid, block_dim, 0, stream>>>(
            d_column_ptrs, d_output, total_elements, num_columns);

        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            cout << "[ERROR] Kernel launch failed in run " << run << ": " << cudaGetErrorString(kernel_error) << endl;
        }

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        if (run > 0) {
            all_run_times_ms.push_back(kernel_time_ms);
            std::cout << "Run " << run << ": " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms" << std::endl;
        } else {
            std::cout << "Warmup run: " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms (discarded)" << std::endl;
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        if (run == 10) {
            std::cout << "Copying results back to host..." << std::endl;
            host_output.resize(total_elements);
            cudaMemcpy(host_output.data(), d_output, total_elements * sizeof(Tuple), cudaMemcpyDeviceToHost);
        }

        for (int col = 0; col < num_columns; col++) {
            g_allocator.DeviceFree(d_column_data[col]);
        }
        g_allocator.DeviceFree(d_column_ptrs);
        g_allocator.DeviceFree(d_output);
    }

    float elapsed_ms = std::accumulate(all_run_times_ms.begin(), all_run_times_ms.end(), 0.0f) / all_run_times_ms.size();

    return elapsed_ms;
}

CompressionType parseCompressionType(const std::string& type_str) {
    if (type_str == "RFOR") return RFOR;
    if (type_str == "UNCOMPRESSED") return UNCOMPRESSED;
    throw std::invalid_argument("Invalid compression type. Use RFOR or UNCOMPRESSED");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <compression_type> <column_file_1> <column_file_2> ... <column_file_N>" << std::endl;
        std::cout << "compression_type: RFOR or UNCOMPRESSED" << std::endl;
        return 1;
    }

    CompressionType compression_type;
    try {
        compression_type = parseCompressionType(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    int num_columns = argc - 2;
    if (num_columns > MAX_COLUMNS) {
        std::cerr << "Error: Maximum " << MAX_COLUMNS << " columns supported, got " << num_columns << std::endl;
        return 1;
    }

    std::vector<std::string> column_files;
    for (int i = 2; i < argc; i++) {
        column_files.push_back(argv[i]);
    }

    std::string compression_name = (compression_type == RFOR) ? "RFOR" : "UNCOMPRESSED";
    std::cout << compression_name << " Tuple Benchmark - " << num_columns << " columns" << std::endl;

    size_t total_elements = 0;
    for (int i = 0; i < num_columns; i++) {
        size_t file_size = getFileSize(column_files[i]);
        size_t num_elements = file_size / sizeof(int);

        if (i == 0) {
            total_elements = num_elements;
        } else if (num_elements != total_elements) {
            std::cerr << "Error: All files must have same element count" << std::endl;
            return 1;
        }
    }

    double total_data_gb = total_elements * num_columns * sizeof(int) / 1e9;
    std::cout << "Elements: " << total_elements << " (" << fixed << setprecision(2)
              << total_data_gb << " GB)" << std::endl;

    std::cout << "Loading " << num_columns << " columns..." << std::endl;
    std::vector<std::vector<int>> column_data(num_columns);
    for (int col = 0; col < num_columns; col++) {
        int* raw_data = loadColumn<int>(column_files[col], total_elements);
        if (!raw_data) {
            std::cerr << "Error: Failed to load column " << col << std::endl;
            return 1;
        }
        column_data[col].assign(raw_data, raw_data + total_elements);
        delete[] raw_data;
    }

    std::vector<Tuple> output;
    float elapsed_time;

    if (compression_type == RFOR) {
        elapsed_time = run_compressed_blocks_to_tuple(column_data, total_elements, num_columns, output);
    } else {
        elapsed_time = run_uncompressed_blocks_to_tuple(column_data, total_elements, num_columns, output);
    }

    if (elapsed_time < 0) {
        cout << "[ERROR] Pipeline failed due to memory allocation errors" << endl;
        return 1;
    }

    std::cout << "\n" << compression_name << " pipeline time: " << elapsed_time << " ms\n";

    std::cout << "Verifying results..." << std::endl;
    bool verification_passed = true;
    size_t error_count = 0;
    for (size_t i = 0; i < std::min(total_elements, size_t(100)); i++) {
        for (int col = 0; col < num_columns; col++) {
            if (output[i].values[col] != column_data[col][i]) {
                if (error_count < 3) {
                    std::cout << "ERROR at element " << i << ", column " << col
                             << ": expected " << column_data[col][i]
                             << ", got " << output[i].values[col] << std::endl;
                }
                verification_passed = false;
                error_count++;
                if (error_count >= 3) break;
            }
        }
        if (error_count >= 3) break;
    }

    std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED") << std::endl;

    double throughput_gb_s = (total_elements * num_columns * sizeof(int) / 1e9) / (elapsed_time * 1e-3);
    double tuples_per_second = total_elements / (elapsed_time * 1e-3);

    std::cout << "Throughput: " << std::fixed << std::setprecision(1) << throughput_gb_s << " GB/s" << std::endl;
    std::cout << "Tuples/s: " << std::fixed << std::setprecision(1) << tuples_per_second << " tuples/s" << std::endl;

    for (int i = 0; i < std::min(10, (int)total_elements); ++i) {
        std::cout << "Tuple[" << i << "] = (";
        for (int j = 0; j < num_columns; ++j)
            std::cout << output[i].values[j] << (j < num_columns - 1 ? ", " : "");
        std::cout << ")\n";
    }

    return 0;
}
