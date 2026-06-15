#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "ssb_utils.h"
#include "crystal_ssb_utils.h"
#include <cstring>

using namespace cub;

const size_t VEC_SIZE = 1024;


struct FastLanesVectorMetadata {
    uint32_t base_value;         // FOR base
    uint8_t bitwidth;           // Bitwidth
    uint32_t element_count;     // Number of elements in vector
};

struct FastLanesBlock {
    std::vector<std::vector<char>> compressed_chunks;  // Each chunk is one vector
    size_t num_values;
};

__global__ void runFastLanesKernel(
    FastLanesVectorMetadata** metadata_ptrs,
    uint32_t** compressed_data_ptrs,
    int** output_ptrs,
    size_t total_vectors) {

    static __shared__ uint32_t unpacked[1024];

    size_t vector_id = blockIdx.x;
    if (vector_id >= total_vectors) return;

    //metadata pointers
    FastLanesVectorMetadata* metadata = metadata_ptrs[vector_id];
    uint32_t* compressed_data = compressed_data_ptrs[vector_id];
    int* output = output_ptrs[vector_id];

    unpack_device(compressed_data, unpacked, metadata->bitwidth);

    int thread_id = threadIdx.x;
    for (int i = 0; i < 32; i++) {
        int idx = thread_id + i * 32;
        if (idx < metadata->element_count) {
            output[idx] = unpacked[idx] + metadata->base_value;
        }
    }
}

//Compress data into vectors, vectors in fastlanes are what tiles are in tile-based, chunks
uint fastLanesVectorCompress(uint32_t* vector_data, size_t vector_size, std::vector<char>& compressed_vector) {
    if (vector_size == 0) return 0;

    // Convert to int32
    auto* working_vec = new int32_t[VEC_SIZE];
    for (size_t i = 0; i < VEC_SIZE; i++) {
        if (i < vector_size) {
            working_vec[i] = static_cast<int32_t>(vector_data[i]);
        } else {
            working_vec[i] = 0;  // Pad with zeros
        }
    }

    // Apply FOR encoding
    int base = find_base<VEC_SIZE>(working_vec);
    subtract_base<VEC_SIZE>(working_vec, base);
    int bw = find_bw<VEC_SIZE>(working_vec);

    // Create metadata
    FastLanesVectorMetadata metadata;
    metadata.base_value = static_cast<uint32_t>(base);
    metadata.bitwidth = static_cast<uint8_t>(bw);
    metadata.element_count = vector_size;

    // Calculate compressed data size
    size_t compressed_data_words = (bw * VEC_SIZE + 31) / 32;
    std::vector<uint32_t> compressed_data(compressed_data_words);

    // Compress using FastLanes
    generated::pack::fallback::scalar::pack(
        reinterpret_cast<uint32_t*>(working_vec),
        compressed_data.data(),
        bw
    );

    // Create contiguous layout: [Metadata][Compressed Data]
    size_t metadata_size = sizeof(FastLanesVectorMetadata);
    size_t data_size = compressed_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + data_size;

    compressed_vector.resize(total_size);
    char* ptr = compressed_vector.data();

    // Copy metadata
    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;

    // Copy compressed data
    std::memcpy(ptr, compressed_data.data(), data_size);

    delete[] working_vec;
    return compressed_data_words;
}

// Build blocks, each block has multiple vectors inside
std::vector<FastLanesBlock> createBitPackedBlocks(uint32_t* h_data, size_t total_elements,
                                                   size_t max_uncompressed_bytes_per_block, CachingDeviceAllocator& alloc) {

    std::vector<FastLanesBlock> blocks;
    const size_t total_vectors = (total_elements + VEC_SIZE - 1) / VEC_SIZE;
    const size_t max_vectors_per_block = max_uncompressed_bytes_per_block / (VEC_SIZE * sizeof(uint32_t));

    size_t current_vector = 0;
    auto compression_start = std::chrono::high_resolution_clock::now();

    while (current_vector < total_vectors) {
        FastLanesBlock block;
        size_t vectors_in_block = std::min(max_vectors_per_block, total_vectors - current_vector);

        for (size_t vec_idx = 0; vec_idx < vectors_in_block; vec_idx++) {
            size_t global_vec_idx = current_vector + vec_idx;
            size_t vec_start = global_vec_idx * VEC_SIZE;

            std::vector<uint32_t> vector_data(VEC_SIZE);
            for (size_t i = 0; i < VEC_SIZE; i++) {
                if (vec_start + i < total_elements) {
                    vector_data[i] = h_data[vec_start + i];
                } else {
                    vector_data[i] = h_data[total_elements - 1];  // Pad with last value
                }
            }

            std::vector<char> compressed_chunk;
            fastLanesVectorCompress(vector_data.data(),
                                  std::min(VEC_SIZE, total_elements - vec_start),
                                  compressed_chunk);
            block.compressed_chunks.push_back(std::move(compressed_chunk));
        }

        block.num_values = std::min(vectors_in_block * VEC_SIZE,
                                   total_elements - (current_vector * VEC_SIZE));
        blocks.push_back(std::move(block));
        current_vector += vectors_in_block;
    }

    auto compression_end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(compression_end - compression_start).count();

    return blocks;
}

float runFastLanesBenchmark(
    std::vector<FastLanesBlock>& blocks,
    int num_items,
    CachingDeviceAllocator& alloc,
    uint32_t* h_col_orig) {

    size_t total_uncompressed_bytes = static_cast<size_t>(num_items) * sizeof(int);
    size_t total_compressed_bytes = 0;
    size_t total_vectors = 0;

    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            total_compressed_bytes += chunk.size();
            total_vectors++;
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
        std::vector<FastLanesVectorMetadata*> h_metadata_ptrs;
        std::vector<uint32_t*> h_compressed_data_ptrs;
        std::vector<int*> h_output_ptrs;

        h_metadata_ptrs.reserve(total_vectors);
        h_compressed_data_ptrs.reserve(total_vectors);
        h_output_ptrs.reserve(total_vectors);

        // Build pointer arrays pointing into the single large buffer
        size_t global_output_offset = 0;

        for (size_t chunk_idx = 0; chunk_idx < chunk_offsets.size(); chunk_idx++) {
            char* chunk_ptr = d_buffer + chunk_offsets[chunk_idx];

            FastLanesVectorMetadata* metadata = reinterpret_cast<FastLanesVectorMetadata*>(chunk_ptr);
            uint32_t* compressed_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(FastLanesVectorMetadata));
            int* output = device_out + global_output_offset;

            h_metadata_ptrs.push_back(metadata);
            h_compressed_data_ptrs.push_back(compressed_data);
            h_output_ptrs.push_back(output);

            global_output_offset += VEC_SIZE;
        }

        // Allocate GPU pointer arrays
        FastLanesVectorMetadata** d_metadata_ptrs;
        uint32_t** d_compressed_data_ptrs;
        int** d_output_ptrs;

        alloc.DeviceAllocate((void**)&d_metadata_ptrs, total_vectors * sizeof(FastLanesVectorMetadata*));
        alloc.DeviceAllocate((void**)&d_compressed_data_ptrs, total_vectors * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_output_ptrs, total_vectors * sizeof(int*));

        // Copy pointer arrays to GPU
        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(),
                  total_vectors * sizeof(FastLanesVectorMetadata*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed_data_ptrs, h_compressed_data_ptrs.data(),
                  total_vectors * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_ptrs, h_output_ptrs.data(),
                  total_vectors * sizeof(int*), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        // Single kernel launch
        dim3 grid(total_vectors);
        dim3 block_dim(32);  // FastLanes uses 32 threads per vector

        runFastLanesKernel<<<grid, block_dim, 0, stream>>>(
            d_metadata_ptrs,
            d_compressed_data_ptrs,
            d_output_ptrs,
            total_vectors
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
            std::vector<uint32_t> h_decoded(num_items);
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

// Utility methods
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

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <column_file_path> [max_block_size_mb]" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_discount" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_quantity 4" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_quantity 4 64" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string column_file = argv[1];
    size_t max_block_size_mb = 2;

    if (argc >= 3) {
        max_block_size_mb = std::stoul(argv[2]);
    }

    size_t max_block_size = max_block_size_mb << 20;

    CachingDeviceAllocator g_allocator(true);

    std::cout << "=== FastLanes Bitpack Benchmark ===\n";
    std::cout << "Column file: " << column_file << std::endl;
    std::cout << "Max block size: " << max_block_size_mb << " MB\n";

    size_t file_size = getFileSize(column_file);
    if (file_size == 0) {
        std::cerr << "Error: Failed to get file size or file is empty" << std::endl;
        return 1;
    }

    if (file_size % sizeof(uint32_t) != 0) {
        std::cerr << "Error: File size is not a multiple of 4 bytes" << std::endl;
        return 1;
    }

    size_t n_elements = file_size / sizeof(uint32_t);

    uint32_t* h_data = loadColumn<uint32_t>(column_file, n_elements);
    if (!h_data) {
        std::cerr << "Error: Failed to load column data" << std::endl;
        return 1;
    }

    auto blocks = createBitPackedBlocks(h_data, n_elements, max_block_size, g_allocator);

    float total_ms = runFastLanesBenchmark(blocks, n_elements, g_allocator, h_data);

    delete[] h_data;

    return 0;
}
