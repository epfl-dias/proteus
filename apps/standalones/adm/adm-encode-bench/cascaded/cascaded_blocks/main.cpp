#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cuda_runtime_api.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/lz4.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include "../utils.h"

// CUDA_CHECK macro is already defined in ../utils.h

enum class CompressionAlgorithm {
    CASCADED,
    GDEFLATE,
    LZ4
};

void run_benchmark_blocks_batch_cascaded(const std::vector<std::vector<char>>& uncompressed_blocks, size_t batch_size) {
    std::cout << "\n========== CASCADED COMPRESSION BENCHMARK ==========\n";
    
    //const values for chunk size and data type for nvcomp
    //64KiB
    const size_t chunk_size = 1024 * 64;
    const nvcompType_t data_type = NVCOMP_TYPE_INT;

    //check if chunk size is within the allowed range
    static_assert(chunk_size <= nvcompCascadedCompressionMaxAllowedChunkSize);    

    //metrics values
    double total_comp_time = 0;
    double total_decomp_time = 0;
    size_t total_uncompressed_bytes = 0;
    size_t total_compressed_bytes = 0;
    
    //cascaded options, encodings
    nvcompBatchedCascadedOpts_t cascaded_opts = {};
    cascaded_opts.type = data_type;
    cascaded_opts.num_RLEs = 0;
    cascaded_opts.num_deltas = 0;
    cascaded_opts.use_bp = 1;
    
    //iterate through blocks in batches
    for (size_t batch_start = 0; batch_start < uncompressed_blocks.size(); batch_start += batch_size){

        const size_t current_batch_size = std::min(batch_size, uncompressed_blocks.size() - batch_start);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream))

        //total number of chunks
        size_t total_chunks = 0;
        std::vector<size_t> block_chunk_sizes(current_batch_size);

        //compute total chunks and per block chunk sizes
        for (size_t i = 0; i < current_batch_size; ++i) {
            //iterate batch, set chunk size per block
            block_chunk_sizes[i] = (uncompressed_blocks[batch_start + i].size() + chunk_size - 1) / chunk_size;
            total_chunks += block_chunk_sizes[i];
        }

        //allocate GPU memory for uncompressed data and related pointers
        size_t* host_uncompressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * total_chunks))

        void** host_uncompressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * total_chunks))
        
        char* device_input_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_input_data, chunk_size * total_chunks, stream))

        size_t chunk_offset = 0;
        //iterate through current batch size
        for (size_t i = 0; i < current_batch_size; ++i){
            //current offset of block
            const auto& block = uncompressed_blocks[batch_start + i];
            size_t block_size = block.size();
            size_t num_chunks = block_chunk_sizes[i];
            

            //copy each block's data to GPU memory
            for (size_t j = 0; j < num_chunks; ++j){
                //set offset
                size_t offset = chunk_offset * chunk_size;    
                //set size to copy to GPU, last chunk, only copy what's left
                size_t copy_size = (j + 1 < num_chunks) ? chunk_size : block_size - j * chunk_size;
                
                CUDA_CHECK(cudaMemcpyAsync(device_input_data + offset, block.data() + j * chunk_size,
                            copy_size, cudaMemcpyHostToDevice, stream))

                //per chunk offset, the ptrs to uncompressed and to their sizes per chunk offest
                host_uncompressed_ptrs[chunk_offset] = device_input_data + offset;
                host_uncompressed_bytes[chunk_offset] = copy_size;
                chunk_offset++;
            }
        }

        //device uncompressed pointers and byte sizes
        void** device_uncompressed_ptrs = nullptr;
        size_t* device_uncompressed_bytes = nullptr;
        
        //allocaiton for pointers and bytes sizes
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * total_chunks, 
            cudaMemcpyHostToDevice, stream))

        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * total_chunks, 
        cudaMemcpyHostToDevice, stream))

        //temp bytes, as operational buffer for nvcomp
        //max out bytes is maximum possible size of compressed output for single chunk 
        size_t temp_bytes, max_out_bytes;
        nvcompBatchedCascadedCompressGetTempSize(total_chunks, chunk_size, cascaded_opts, &temp_bytes);
        nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_size, cascaded_opts, &max_out_bytes);

        //temp pointer for device
        void* device_comp_temp_ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_comp_temp_ptr, temp_bytes, stream))

        //ptrs for compressed chunks on host
        void** host_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * total_chunks))

        for (size_t i = 0; i < total_chunks; ++i){
            CUDA_CHECK(cudaMallocAsync(&host_compressed_ptrs[i], max_out_bytes, stream))
        }

        //ptrs for compressed chunks on device
        void** device_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        //allocate byte sizes on device
        size_t* device_compressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_bytes, sizeof(size_t) * total_chunks, stream))
    
        //start measuring compression time
        cudaEvent_t comp_start, comp_end;

        //measurement of compression speed, use in-gpu time
        CUDA_CHECK(cudaEventCreate(&comp_start))
        CUDA_CHECK(cudaEventCreate(&comp_end))
        CUDA_CHECK(cudaEventRecord(comp_start, stream))

        //compress
        nvcompStatus_t compression_result;
        compression_result = nvcompBatchedCascadedCompressAsync(device_uncompressed_ptrs, device_uncompressed_bytes,
            chunk_size, total_chunks, device_comp_temp_ptr, temp_bytes, device_compressed_ptrs, device_compressed_bytes,
            cascaded_opts, stream);

        if (compression_result != nvcompSuccess)
            throw std::runtime_error("Cascaded compression failed");

        CUDA_CHECK(cudaEventRecord(comp_end, stream))
        CUDA_CHECK(cudaStreamSynchronize(stream))

        float comp_time;
        CUDA_CHECK(cudaEventElapsedTime(&comp_time, comp_start, comp_end))
        
        total_comp_time += comp_time;

        //get sizes of compressed byte chunks
        std::vector<size_t>host_compressed_bytes(total_chunks);
        CUDA_CHECK(cudaMemcpyAsync(host_compressed_bytes.data(),
            device_compressed_bytes, total_chunks * sizeof(*device_compressed_bytes), cudaMemcpyDeviceToHost, stream))

        size_t comp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            comp_bytes += host_compressed_bytes[idx];
        }

        size_t uncomp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            uncomp_bytes += host_uncompressed_bytes[idx];
        }

        //increment total bytes sizes
        total_uncompressed_bytes += uncomp_bytes;
        total_compressed_bytes += comp_bytes;

        // === DECOMPRESSION SECTION ===

        // Allocate space for decompressed data on device
        char* device_decomp_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_data, uncomp_bytes, stream))

        // Allocate pointers for decomp device
        void** device_decomp_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_ptrs, sizeof(void*) * total_chunks, stream))

        // Allocate space for host decomp
        std::vector<void*> host_decomp_ptrs(total_chunks);
        size_t curr_offset = 0;
        for (size_t i = 0; i < total_chunks; ++i) {
            host_decomp_ptrs[i] = device_decomp_data + curr_offset;
            curr_offset += host_uncompressed_bytes[i];
        }
        CUDA_CHECK(cudaMemcpyAsync(device_decomp_ptrs, host_decomp_ptrs.data(),
            sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        // Uncompressed buffer bytes on device
        size_t* device_uncompressed_buffer_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_buffer_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_buffer_bytes, host_uncompressed_bytes,
            sizeof(size_t) * total_chunks, cudaMemcpyHostToDevice, stream))

        size_t* device_uncompressed_chunk_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_chunk_bytes, sizeof(size_t) * total_chunks, stream))

        // Get temp allowed size        
        size_t decomp_temp_bytes;
        nvcompStatus_t status = nvcompBatchedCascadedDecompressGetTempSize(
            total_chunks,
            chunk_size,
            &decomp_temp_bytes
        );
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get cascaded decompression temp size");
        }

        // Temp pointer for device decomp
        void* device_decomp_temp_ptr;
        CUDA_CHECK(cudaMalloc(&device_decomp_temp_ptr, decomp_temp_bytes))

        // Statuses per chunk
        nvcompStatus_t* statuses;
        CUDA_CHECK(cudaMallocAsync(&statuses, sizeof(nvcompStatus_t) * total_chunks, stream))

        // === Decompression benchmarking ===
        std::vector<float> decomp_times;

        for (int run = 0; run < 4; ++run) {
            cudaEvent_t decomp_start, decomp_end;
            CUDA_CHECK(cudaEventCreate(&decomp_start))
            CUDA_CHECK(cudaEventCreate(&decomp_end))

            CUDA_CHECK(cudaEventRecord(decomp_start, stream))

            status = nvcompBatchedCascadedDecompressAsync(
                device_compressed_ptrs,
                device_compressed_bytes,
                device_uncompressed_buffer_bytes,
                device_uncompressed_chunk_bytes,
                total_chunks,
                device_decomp_temp_ptr,
                decomp_temp_bytes,
                device_decomp_ptrs,
                statuses,
                stream
            );

            if (status != nvcompSuccess) {
                throw std::runtime_error("Failed to decompress cascaded data");
            }

            CUDA_CHECK(cudaEventRecord(decomp_end, stream))
            CUDA_CHECK(cudaStreamSynchronize(stream))

            float decomp_time = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&decomp_time, decomp_start, decomp_end))

            if (run > 0) {
                // skip first run
                decomp_times.push_back(decomp_time);
            }

            CUDA_CHECK(cudaEventDestroy(decomp_start))
            CUDA_CHECK(cudaEventDestroy(decomp_end))
        }

        // Compute average of last 3 decompression times
        float avg_decomp_time = std::accumulate(decomp_times.begin(), decomp_times.end(), 0.0f) / decomp_times.size();
        total_decomp_time += avg_decomp_time;

        std::vector<char> host_decomp_data(uncomp_bytes);

        CUDA_CHECK(cudaMemcpy(
            host_decomp_data.data(),
            device_decomp_data,
            uncomp_bytes,
            cudaMemcpyDeviceToHost))

        //reassemble decompressed data into blocks            
        std::vector<std::vector<char>> decompressed_blocks(current_batch_size);
        size_t cursor = 0;        
        size_t chunk_global = 0;     

        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t num_chunks = block_chunk_sizes[i];
          
            size_t block_size = 0;
            for (size_t c = 0; c < num_chunks; ++c) {
              block_size += host_uncompressed_bytes[chunk_global + c];
            }
          
            auto& block = decompressed_blocks[i];
            block.assign(
              host_decomp_data.begin()  + cursor,
              host_decomp_data.begin()  + cursor + block_size);
          
            cursor       += block_size;
            chunk_global += num_chunks;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream))

        //correctness check for input uncompressed and decompressed blocks of data
        bool is_valid = compare_data_blocks(
            {uncompressed_blocks.begin() + batch_start, uncompressed_blocks.begin() + batch_start + current_batch_size},
            decompressed_blocks
        );
        if (is_valid == false){
            std::cout << "Correctness check failed for batch: " << batch_start << " to " << batch_start + current_batch_size << std::endl;
        }

        //free allocated memory for the batch
        CUDA_CHECK(cudaEventDestroy(comp_start))
        CUDA_CHECK(cudaEventDestroy(comp_end))
        CUDA_CHECK(cudaStreamDestroy(stream))

        CUDA_CHECK(cudaFreeHost(host_uncompressed_bytes))
        CUDA_CHECK(cudaFreeHost(host_uncompressed_ptrs))

        CUDA_CHECK(cudaFree(device_input_data))
        CUDA_CHECK(cudaFree(device_uncompressed_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_bytes))
        CUDA_CHECK(cudaFree(device_comp_temp_ptr))
        
        for (size_t i = 0; i < total_chunks; ++i) {
            CUDA_CHECK(cudaFree(host_compressed_ptrs[i]))
        }

        CUDA_CHECK(cudaFreeHost(host_compressed_ptrs))

        CUDA_CHECK(cudaFree(device_compressed_ptrs))
        CUDA_CHECK(cudaFree(device_compressed_bytes))

        CUDA_CHECK(cudaFree(device_decomp_data))
        CUDA_CHECK(cudaFree(device_decomp_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_buffer_bytes))
        CUDA_CHECK(cudaFree(device_uncompressed_chunk_bytes))
        CUDA_CHECK(cudaFree(device_decomp_temp_ptr))
        CUDA_CHECK(cudaFree(statuses))
    }

    //calculate metrics results
    double total_comp_time_s   = total_comp_time   * 1e-3;
    double total_decomp_time_s = total_decomp_time * 1e-3;
    
    double compression_ratio = static_cast<double>(total_uncompressed_bytes) 
                             / static_cast<double>(total_compressed_bytes);
    double comp_speed_gibs   = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_comp_time_s;
    double decomp_speed_gibs = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_decomp_time_s;
    
    std::cout << "======== CASCADED Results ========\n";
    std::cout << "Total uncompressed bytes: " << total_uncompressed_bytes << "\n";
    std::cout << "Total   compressed bytes: " << total_compressed_bytes << "\n\n";
    std::cout << "=== Compression ===\n";
    std::cout << "Ratio  : " << std::fixed << std::setprecision(4) 
              << compression_ratio << "\n";
    std::cout << "Time   : " << total_comp_time_s << " s\n";
    std::cout << "Speed  : " << comp_speed_gibs << " GiB/s\n\n";
    std::cout << "=== Decompression ===\n";
    std::cout << "Time   : " << total_decomp_time_s << " s\n";
    std::cout << "Speed  : " << decomp_speed_gibs << " GiB/s\n";
    std::cout << "========================================\n\n";
}

void run_benchmark_blocks_batch_gdeflate(const std::vector<std::vector<char>>& uncompressed_blocks, size_t batch_size) {
    std::cout << "\n========== GDEFLATE COMPRESSION BENCHMARK ==========\n";
    
    //const values for chunk size
    //64KiB
    const size_t chunk_size = 1024 * 64;

    //metrics values
    double total_comp_time = 0;
    double total_decomp_time = 0;
    size_t total_uncompressed_bytes = 0;
    size_t total_compressed_bytes = 0;
    
    //iterate through blocks in batches
    for (size_t batch_start = 0; batch_start < uncompressed_blocks.size(); batch_start += batch_size){

        const size_t current_batch_size = std::min(batch_size, uncompressed_blocks.size() - batch_start);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream))

        //total number of chunks
        size_t total_chunks = 0;
        std::vector<size_t> block_chunk_sizes(current_batch_size);

        //compute total chunks and per block chunk sizes
        for (size_t i = 0; i < current_batch_size; ++i) {
            //iterate batch, set chunk size per block
            block_chunk_sizes[i] = (uncompressed_blocks[batch_start + i].size() + chunk_size - 1) / chunk_size;
            total_chunks += block_chunk_sizes[i];
        }

        //allocate GPU memory for uncompressed data and related pointers
        size_t* host_uncompressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * total_chunks))

        void** host_uncompressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * total_chunks))
        
        char* device_input_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_input_data, chunk_size * total_chunks, stream))

        size_t chunk_offset = 0;
        //iterate through current batch size
        for (size_t i = 0; i < current_batch_size; ++i){
            //current offset of block
            const auto& block = uncompressed_blocks[batch_start + i];
            size_t block_size = block.size();
            size_t num_chunks = block_chunk_sizes[i];

            //copy each block's data to GPU memory
            for (size_t j = 0; j < num_chunks; ++j){
                //set offset
                size_t offset = chunk_offset * chunk_size;    
                //set size to copy to GPU, last chunk, only copy what's left
                size_t copy_size = (j + 1 < num_chunks) ? chunk_size : block_size - j * chunk_size;
                
                CUDA_CHECK(cudaMemcpyAsync(device_input_data + offset, block.data() + j * chunk_size,
                            copy_size, cudaMemcpyHostToDevice, stream))

                //per chunk offset, the ptrs to uncompressed and to their sizes per chunk offest
                host_uncompressed_ptrs[chunk_offset] = device_input_data + offset;
                host_uncompressed_bytes[chunk_offset] = copy_size;
                chunk_offset++;
            }
        }

        //device uncompressed pointers and byte sizes
        void** device_uncompressed_ptrs = nullptr;
        size_t* device_uncompressed_bytes = nullptr;
        
        //allocaiton for pointers and bytes sizes
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * total_chunks, 
            cudaMemcpyHostToDevice, stream))

        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * total_chunks, 
        cudaMemcpyHostToDevice, stream))

        //temp bytes, as operational buffer for nvcomp
        //max out bytes is maximum possible size of compressed output for single chunk 
        size_t temp_bytes, max_out_bytes;
        nvcompBatchedGdeflateCompressGetTempSize(total_chunks, chunk_size, nvcompBatchedGdeflateDefaultOpts, &temp_bytes);
        nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedGdeflateDefaultOpts, &max_out_bytes);

        //temp pointer for device
        void* device_comp_temp_ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_comp_temp_ptr, temp_bytes, stream))

        //ptrs for compressed chunks on host
        void** host_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * total_chunks))

        for (size_t i = 0; i < total_chunks; ++i){
            CUDA_CHECK(cudaMallocAsync(&host_compressed_ptrs[i], max_out_bytes, stream))
        }

        //ptrs for compressed chunks on device
        void** device_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        //allocate byte sizes on device
        size_t* device_compressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_bytes, sizeof(size_t) * total_chunks, stream))
    
        //start measuring compression time
        cudaEvent_t comp_start, comp_end;

        //measurement of compression speed, use in-gpu time
        CUDA_CHECK(cudaEventCreate(&comp_start))
        CUDA_CHECK(cudaEventCreate(&comp_end))
        CUDA_CHECK(cudaEventRecord(comp_start, stream))

        //compress
        nvcompStatus_t compression_result;
        compression_result = nvcompBatchedGdeflateCompressAsync(device_uncompressed_ptrs, device_uncompressed_bytes,
            chunk_size, total_chunks, device_comp_temp_ptr, temp_bytes, device_compressed_ptrs, device_compressed_bytes,
            nvcompBatchedGdeflateDefaultOpts, stream);

        if (compression_result != nvcompSuccess)
            throw std::runtime_error("Gdeflate compression failed");

        CUDA_CHECK(cudaEventRecord(comp_end, stream))
        CUDA_CHECK(cudaStreamSynchronize(stream))

        float comp_time;
        CUDA_CHECK(cudaEventElapsedTime(&comp_time, comp_start, comp_end))
        
        total_comp_time += comp_time;

        //get sizes of compressed byte chunks
        std::vector<size_t>host_compressed_bytes(total_chunks);
        CUDA_CHECK(cudaMemcpyAsync(host_compressed_bytes.data(),
            device_compressed_bytes, total_chunks * sizeof(*device_compressed_bytes), cudaMemcpyDeviceToHost, stream))

        size_t comp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            comp_bytes += host_compressed_bytes[idx];
        }

        size_t uncomp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            uncomp_bytes += host_uncompressed_bytes[idx];
        }

        //increment total bytes sizes
        total_uncompressed_bytes += uncomp_bytes;
        total_compressed_bytes += comp_bytes;

        // === DECOMPRESSION SECTION ===

        // Allocate space for decompressed data on device
        char* device_decomp_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_data, uncomp_bytes, stream))

        // Allocate pointers for decomp device
        void** device_decomp_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_ptrs, sizeof(void*) * total_chunks, stream))

        // Allocate space for host decomp
        std::vector<void*> host_decomp_ptrs(total_chunks);
        size_t curr_offset = 0;
        for (size_t i = 0; i < total_chunks; ++i) {
            host_decomp_ptrs[i] = device_decomp_data + curr_offset;
            curr_offset += host_uncompressed_bytes[i];
        }
        CUDA_CHECK(cudaMemcpyAsync(device_decomp_ptrs, host_decomp_ptrs.data(),
            sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        // Uncompressed buffer bytes on device
        size_t* device_uncompressed_buffer_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_buffer_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_buffer_bytes, host_uncompressed_bytes,
            sizeof(size_t) * total_chunks, cudaMemcpyHostToDevice, stream))

        size_t* device_uncompressed_chunk_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_chunk_bytes, sizeof(size_t) * total_chunks, stream))

        // Get temp allowed size        
        size_t decomp_temp_bytes;
        nvcompStatus_t status = nvcompBatchedGdeflateDecompressGetTempSize(
            total_chunks,
            chunk_size,
            &decomp_temp_bytes
        );
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get gdeflate decompression temp size");
        }

        // Temp pointer for device decomp
        void* device_decomp_temp_ptr;
        CUDA_CHECK(cudaMalloc(&device_decomp_temp_ptr, decomp_temp_bytes))

        // Statuses per chunk
        nvcompStatus_t* statuses;
        CUDA_CHECK(cudaMallocAsync(&statuses, sizeof(nvcompStatus_t) * total_chunks, stream))

        // === Decompression benchmarking ===
        std::vector<float> decomp_times;

        for (int run = 0; run < 4; ++run) {
            cudaEvent_t decomp_start, decomp_end;
            CUDA_CHECK(cudaEventCreate(&decomp_start))
            CUDA_CHECK(cudaEventCreate(&decomp_end))

            CUDA_CHECK(cudaEventRecord(decomp_start, stream))

            status = nvcompBatchedGdeflateDecompressAsync(
                device_compressed_ptrs,
                device_compressed_bytes,
                device_uncompressed_buffer_bytes,
                device_uncompressed_chunk_bytes,
                total_chunks,
                device_decomp_temp_ptr,
                decomp_temp_bytes,
                device_decomp_ptrs,
                statuses,
                stream
            );

            if (status != nvcompSuccess) {
                throw std::runtime_error("Failed to decompress gdeflate data");
            }

            CUDA_CHECK(cudaEventRecord(decomp_end, stream))
            CUDA_CHECK(cudaStreamSynchronize(stream))

            float decomp_time = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&decomp_time, decomp_start, decomp_end))

            if (run > 0) {
                // skip first run
                decomp_times.push_back(decomp_time);
            }

            CUDA_CHECK(cudaEventDestroy(decomp_start))
            CUDA_CHECK(cudaEventDestroy(decomp_end))
        }

        // Compute average of last 3 decompression times
        float avg_decomp_time = std::accumulate(decomp_times.begin(), decomp_times.end(), 0.0f) / decomp_times.size();
        total_decomp_time += avg_decomp_time;

        std::vector<char> host_decomp_data(uncomp_bytes);

        CUDA_CHECK(cudaMemcpy(
            host_decomp_data.data(),
            device_decomp_data,
            uncomp_bytes,
            cudaMemcpyDeviceToHost))

        //reassemble decompressed data into blocks            
        std::vector<std::vector<char>> decompressed_blocks(current_batch_size);
        size_t cursor = 0;        
        size_t chunk_global = 0;     

        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t num_chunks = block_chunk_sizes[i];
          
            size_t block_size = 0;
            for (size_t c = 0; c < num_chunks; ++c) {
              block_size += host_uncompressed_bytes[chunk_global + c];
            }
          
            auto& block = decompressed_blocks[i];
            block.assign(
              host_decomp_data.begin()  + cursor,
              host_decomp_data.begin()  + cursor + block_size);
          
            cursor       += block_size;
            chunk_global += num_chunks;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream))

        //correctness check for input uncompressed and decompressed blocks of data
        bool is_valid = compare_data_blocks(
            {uncompressed_blocks.begin() + batch_start, uncompressed_blocks.begin() + batch_start + current_batch_size},
            decompressed_blocks
        );
        if (is_valid == false){
            std::cout << "Correctness check failed for batch: " << batch_start << " to " << batch_start + current_batch_size << std::endl;
        }

        //free allocated memory for the batch
        CUDA_CHECK(cudaEventDestroy(comp_start))
        CUDA_CHECK(cudaEventDestroy(comp_end))
        CUDA_CHECK(cudaStreamDestroy(stream))

        CUDA_CHECK(cudaFreeHost(host_uncompressed_bytes))
        CUDA_CHECK(cudaFreeHost(host_uncompressed_ptrs))

        CUDA_CHECK(cudaFree(device_input_data))
        CUDA_CHECK(cudaFree(device_uncompressed_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_bytes))
        CUDA_CHECK(cudaFree(device_comp_temp_ptr))
        
        for (size_t i = 0; i < total_chunks; ++i) {
            CUDA_CHECK(cudaFree(host_compressed_ptrs[i]))
        }

        CUDA_CHECK(cudaFreeHost(host_compressed_ptrs))

        CUDA_CHECK(cudaFree(device_compressed_ptrs))
        CUDA_CHECK(cudaFree(device_compressed_bytes))

        CUDA_CHECK(cudaFree(device_decomp_data))
        CUDA_CHECK(cudaFree(device_decomp_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_buffer_bytes))
        CUDA_CHECK(cudaFree(device_uncompressed_chunk_bytes))
        CUDA_CHECK(cudaFree(device_decomp_temp_ptr))
        CUDA_CHECK(cudaFree(statuses))
    }

    //calculate metrics results
    double total_comp_time_s   = total_comp_time   * 1e-3;
    double total_decomp_time_s = total_decomp_time * 1e-3;
    
    double compression_ratio = static_cast<double>(total_uncompressed_bytes) 
                             / static_cast<double>(total_compressed_bytes);
    double comp_speed_gibs   = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_comp_time_s;
    double decomp_speed_gibs = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_decomp_time_s;
    
    std::cout << "======== GDEFLATE Results ========\n";
    std::cout << "Total uncompressed bytes: " << total_uncompressed_bytes << "\n";
    std::cout << "Total   compressed bytes: " << total_compressed_bytes << "\n\n";
    std::cout << "=== Compression ===\n";
    std::cout << "Ratio  : " << std::fixed << std::setprecision(4) 
              << compression_ratio << "\n";
    std::cout << "Time   : " << total_comp_time_s << " s\n";
    std::cout << "Speed  : " << comp_speed_gibs << " GiB/s\n\n";
    std::cout << "=== Decompression ===\n";
    std::cout << "Time   : " << total_decomp_time_s << " s\n";
    std::cout << "Speed  : " << decomp_speed_gibs << " GiB/s\n";
    std::cout << "======================================\n\n";
}

void run_benchmark_blocks_batch_lz4(const std::vector<std::vector<char>>& uncompressed_blocks, size_t batch_size) {
    std::cout << "\n========== LZ4 COMPRESSION BENCHMARK ==========\n";
    
    //const values for chunk size
    //64KiB
    const size_t chunk_size = 1024 * 64;

    //metrics values
    double total_comp_time = 0;
    double total_decomp_time = 0;
    size_t total_uncompressed_bytes = 0;
    size_t total_compressed_bytes = 0;
    
    //iterate through blocks in batches
    for (size_t batch_start = 0; batch_start < uncompressed_blocks.size(); batch_start += batch_size){

        const size_t current_batch_size = std::min(batch_size, uncompressed_blocks.size() - batch_start);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream))

        //total number of chunks
        size_t total_chunks = 0;
        std::vector<size_t> block_chunk_sizes(current_batch_size);

        //compute total chunks and per block chunk sizes
        for (size_t i = 0; i < current_batch_size; ++i) {
            //iterate batch, set chunk size per block
            block_chunk_sizes[i] = (uncompressed_blocks[batch_start + i].size() + chunk_size - 1) / chunk_size;
            total_chunks += block_chunk_sizes[i];
        }

        //allocate GPU memory for uncompressed data and related pointers
        size_t* host_uncompressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * total_chunks))

        void** host_uncompressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * total_chunks))
        
        char* device_input_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_input_data, chunk_size * total_chunks, stream))

        size_t chunk_offset = 0;
        //iterate through current batch size
        for (size_t i = 0; i < current_batch_size; ++i){
            //current offset of block
            const auto& block = uncompressed_blocks[batch_start + i];
            size_t block_size = block.size();
            size_t num_chunks = block_chunk_sizes[i];

            //copy each block's data to GPU memory
            for (size_t j = 0; j < num_chunks; ++j){
                //set offset
                size_t offset = chunk_offset * chunk_size;    
                //set size to copy to GPU, last chunk, only copy what's left
                size_t copy_size = (j + 1 < num_chunks) ? chunk_size : block_size - j * chunk_size;
                
                CUDA_CHECK(cudaMemcpyAsync(device_input_data + offset, block.data() + j * chunk_size,
                            copy_size, cudaMemcpyHostToDevice, stream))

                //per chunk offset, the ptrs to uncompressed and to their sizes per chunk offest
                host_uncompressed_ptrs[chunk_offset] = device_input_data + offset;
                host_uncompressed_bytes[chunk_offset] = copy_size;
                chunk_offset++;
            }
        }

        //device uncompressed pointers and byte sizes
        void** device_uncompressed_ptrs = nullptr;
        size_t* device_uncompressed_bytes = nullptr;
        
        //allocaiton for pointers and bytes sizes
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * total_chunks, 
            cudaMemcpyHostToDevice, stream))

        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * total_chunks, 
        cudaMemcpyHostToDevice, stream))

        //temp bytes, as operational buffer for nvcomp
        //max out bytes is maximum possible size of compressed output for single chunk 
        size_t temp_bytes, max_out_bytes;
        nvcompBatchedLZ4CompressGetTempSize(total_chunks, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
        nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

        //temp pointer for device
        void* device_comp_temp_ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_comp_temp_ptr, temp_bytes, stream))

        //ptrs for compressed chunks on host
        void** host_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * total_chunks))

        for (size_t i = 0; i < total_chunks; ++i){
            CUDA_CHECK(cudaMallocAsync(&host_compressed_ptrs[i], max_out_bytes, stream))
        }

        //ptrs for compressed chunks on device
        void** device_compressed_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_ptrs, sizeof(void*) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        //allocate byte sizes on device
        size_t* device_compressed_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_compressed_bytes, sizeof(size_t) * total_chunks, stream))
    
        //start measuring compression time
        cudaEvent_t comp_start, comp_end;

        //measurement of compression speed, use in-gpu time
        CUDA_CHECK(cudaEventCreate(&comp_start))
        CUDA_CHECK(cudaEventCreate(&comp_end))
        CUDA_CHECK(cudaEventRecord(comp_start, stream))

        //compress
        nvcompStatus_t compression_result;
        compression_result = nvcompBatchedLZ4CompressAsync(device_uncompressed_ptrs, device_uncompressed_bytes,
            chunk_size, total_chunks, device_comp_temp_ptr, temp_bytes, device_compressed_ptrs, device_compressed_bytes,
            nvcompBatchedLZ4DefaultOpts, stream);

        if (compression_result != nvcompSuccess)
            throw std::runtime_error("LZ4 compression failed");

        CUDA_CHECK(cudaEventRecord(comp_end, stream))
        CUDA_CHECK(cudaStreamSynchronize(stream))

        float comp_time;
        CUDA_CHECK(cudaEventElapsedTime(&comp_time, comp_start, comp_end))
        
        total_comp_time += comp_time;

        //get sizes of compressed byte chunks
        std::vector<size_t>host_compressed_bytes(total_chunks);
        CUDA_CHECK(cudaMemcpyAsync(host_compressed_bytes.data(),
            device_compressed_bytes, total_chunks * sizeof(*device_compressed_bytes), cudaMemcpyDeviceToHost, stream))

        size_t comp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            comp_bytes += host_compressed_bytes[idx];
        }

        size_t uncomp_bytes = 0;
        for (size_t idx = 0; idx < total_chunks; ++idx){
            uncomp_bytes += host_uncompressed_bytes[idx];
        }

        //increment total bytes sizes
        total_uncompressed_bytes += uncomp_bytes;
        total_compressed_bytes += comp_bytes;

        // === DECOMPRESSION SECTION ===

        // Allocate space for decompressed data on device
        char* device_decomp_data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_data, uncomp_bytes, stream))

        // Allocate pointers for decomp device
        void** device_decomp_ptrs = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_decomp_ptrs, sizeof(void*) * total_chunks, stream))

        // Allocate space for host decomp
        std::vector<void*> host_decomp_ptrs(total_chunks);
        size_t curr_offset = 0;
        for (size_t i = 0; i < total_chunks; ++i) {
            host_decomp_ptrs[i] = device_decomp_data + curr_offset;
            curr_offset += host_uncompressed_bytes[i];
        }
        CUDA_CHECK(cudaMemcpyAsync(device_decomp_ptrs, host_decomp_ptrs.data(),
            sizeof(void*) * total_chunks, cudaMemcpyHostToDevice, stream))

        // Uncompressed buffer bytes on device
        size_t* device_uncompressed_buffer_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_buffer_bytes, sizeof(size_t) * total_chunks, stream))
        CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_buffer_bytes, host_uncompressed_bytes,
            sizeof(size_t) * total_chunks, cudaMemcpyHostToDevice, stream))

        size_t* device_uncompressed_chunk_bytes = nullptr;
        CUDA_CHECK(cudaMallocAsync(&device_uncompressed_chunk_bytes, sizeof(size_t) * total_chunks, stream))

        // Get temp allowed size        
        size_t decomp_temp_bytes;
        nvcompStatus_t status = nvcompBatchedLZ4DecompressGetTempSize(
            total_chunks,
            chunk_size,
            &decomp_temp_bytes
        );
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get LZ4 decompression temp size");
        }

        // Temp pointer for device decomp
        void* device_decomp_temp_ptr;
        CUDA_CHECK(cudaMalloc(&device_decomp_temp_ptr, decomp_temp_bytes))

        // Statuses per chunk
        nvcompStatus_t* statuses;
        CUDA_CHECK(cudaMallocAsync(&statuses, sizeof(nvcompStatus_t) * total_chunks, stream))

        // === Decompression benchmarking ===
        std::vector<float> decomp_times;

        for (int run = 0; run < 4; ++run) {
            cudaEvent_t decomp_start, decomp_end;
            CUDA_CHECK(cudaEventCreate(&decomp_start))
            CUDA_CHECK(cudaEventCreate(&decomp_end))

            CUDA_CHECK(cudaEventRecord(decomp_start, stream))

            status = nvcompBatchedLZ4DecompressAsync(
                device_compressed_ptrs,
                device_compressed_bytes,
                device_uncompressed_buffer_bytes,
                device_uncompressed_chunk_bytes,
                total_chunks,
                device_decomp_temp_ptr,
                decomp_temp_bytes,
                device_decomp_ptrs,
                statuses,
                stream
            );

            if (status != nvcompSuccess) {
                throw std::runtime_error("Failed to decompress LZ4 data");
            }

            CUDA_CHECK(cudaEventRecord(decomp_end, stream))
            CUDA_CHECK(cudaStreamSynchronize(stream))

            float decomp_time = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&decomp_time, decomp_start, decomp_end))

            if (run > 0) {
                // skip first run
                decomp_times.push_back(decomp_time);
            }

            CUDA_CHECK(cudaEventDestroy(decomp_start))
            CUDA_CHECK(cudaEventDestroy(decomp_end))
        }

        // Compute average of last 3 decompression times
        float avg_decomp_time = std::accumulate(decomp_times.begin(), decomp_times.end(), 0.0f) / decomp_times.size();
        total_decomp_time += avg_decomp_time;

        std::vector<char> host_decomp_data(uncomp_bytes);

        CUDA_CHECK(cudaMemcpy(
            host_decomp_data.data(),
            device_decomp_data,
            uncomp_bytes,
            cudaMemcpyDeviceToHost))

        //reassemble decompressed data into blocks            
        std::vector<std::vector<char>> decompressed_blocks(current_batch_size);
        size_t cursor = 0;        
        size_t chunk_global = 0;     

        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t num_chunks = block_chunk_sizes[i];
          
            size_t block_size = 0;
            for (size_t c = 0; c < num_chunks; ++c) {
              block_size += host_uncompressed_bytes[chunk_global + c];
            }
          
            auto& block = decompressed_blocks[i];
            block.assign(
              host_decomp_data.begin()  + cursor,
              host_decomp_data.begin()  + cursor + block_size);
          
            cursor       += block_size;
            chunk_global += num_chunks;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream))

        //correctness check for input uncompressed and decompressed blocks of data
        bool is_valid = compare_data_blocks(
            {uncompressed_blocks.begin() + batch_start, uncompressed_blocks.begin() + batch_start + current_batch_size},
            decompressed_blocks
        );
        if (is_valid == false){
            std::cout << "Correctness check failed for batch: " << batch_start << " to " << batch_start + current_batch_size << std::endl;
        }

        //free allocated memory for the batch
        CUDA_CHECK(cudaEventDestroy(comp_start))
        CUDA_CHECK(cudaEventDestroy(comp_end))
        CUDA_CHECK(cudaStreamDestroy(stream))

        CUDA_CHECK(cudaFreeHost(host_uncompressed_bytes))
        CUDA_CHECK(cudaFreeHost(host_uncompressed_ptrs))

        CUDA_CHECK(cudaFree(device_input_data))
        CUDA_CHECK(cudaFree(device_uncompressed_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_bytes))
        CUDA_CHECK(cudaFree(device_comp_temp_ptr))
        
        for (size_t i = 0; i < total_chunks; ++i) {
            CUDA_CHECK(cudaFree(host_compressed_ptrs[i]))
        }

        CUDA_CHECK(cudaFreeHost(host_compressed_ptrs))

        CUDA_CHECK(cudaFree(device_compressed_ptrs))
        CUDA_CHECK(cudaFree(device_compressed_bytes))

        CUDA_CHECK(cudaFree(device_decomp_data))
        CUDA_CHECK(cudaFree(device_decomp_ptrs))
        CUDA_CHECK(cudaFree(device_uncompressed_buffer_bytes))
        CUDA_CHECK(cudaFree(device_uncompressed_chunk_bytes))
        CUDA_CHECK(cudaFree(device_decomp_temp_ptr))
        CUDA_CHECK(cudaFree(statuses))
    }

    //calculate metrics results
    double total_comp_time_s   = total_comp_time   * 1e-3;
    double total_decomp_time_s = total_decomp_time * 1e-3;
    
    double compression_ratio = static_cast<double>(total_uncompressed_bytes) 
                             / static_cast<double>(total_compressed_bytes);
    double comp_speed_gibs   = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_comp_time_s;
    double decomp_speed_gibs = (static_cast<double>(total_uncompressed_bytes) / (1ull << 30))
                             / total_decomp_time_s;
    
    std::cout << "======== LZ4 Results ========\n";
    std::cout << "Total uncompressed bytes: " << total_uncompressed_bytes << "\n";
    std::cout << "Total   compressed bytes: " << total_compressed_bytes << "\n\n";
    std::cout << "=== Compression ===\n";
    std::cout << "Ratio  : " << std::fixed << std::setprecision(4) 
              << compression_ratio << "\n";
    std::cout << "Time   : " << total_comp_time_s << " s\n";
    std::cout << "Speed  : " << comp_speed_gibs << " GiB/s\n\n";
    std::cout << "=== Decompression ===\n";
    std::cout << "Time   : " << total_decomp_time_s << " s\n";
    std::cout << "Speed  : " << decomp_speed_gibs << " GiB/s\n";
    std::cout << "===================================\n\n";
}

void print_algorithm_comparison_summary(const std::string& datasetPath) {
    std::cout << "\n==================== SUMMARY ====================\n";
    std::cout << "Dataset: " << datasetPath << "\n";
    std::cout << "Block size: 2 MiB\n";
    std::cout << "Chunk size: 64 KiB\n";
    std::cout << "Batch size: 128\n\n";
    std::cout << "All three compression algorithms (Cascaded, gdeflate, LZ4) have been benchmarked.\n";
    std::cout << "==================================================\n";
}

//Example usage dataset paths:
// "/scratch-tmp/data/ssbm100/lineorder.csv.lo_orderdate",
// "/scratch-tmp/data/ssbm100/lineorder.csv.lo_quantity",
// "/scratch-tmp/data/ssbm100/lineorder.csv.lo_extendedprice",
// "/scratch-tmp/data/ssbm100/lineorder.csv.lo_discount",
// "/scratch-tmp/data/ssbm100/lineorder.csv.lo_partkey",

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }

    std::string datasetPath = argv[1];
    std::cout << "Processing dataset: " << datasetPath << "\n";

    // 2MiB block size
    const size_t blockSize = 1024 * 1024 * 2;
    std::cout << "Block size in MiB: " << blockSize / (1024 * 1024) << " MiB\n";

    size_t current_file_size = 0;

    std::filesystem::path fs_path(datasetPath);
    auto blocks = load_csv_column_into_blocks(fs_path, datasetPath, blockSize, current_file_size);

    std::cout << "\n==================== RUNNING ALL COMPRESSION BENCHMARKS ====================\n";
    std::cout << "Dataset size: " << current_file_size / (1024 * 1024) << " MB\n";
    std::cout << "Number of blocks: " << blocks.size() << "\n";
    std::cout << "Testing with batch size: 128\n";

    // Run all three benchmarks
    try {
        run_benchmark_blocks_batch_cascaded(blocks, 128);
        run_benchmark_blocks_batch_gdeflate(blocks, 128);
        run_benchmark_blocks_batch_lz4(blocks, 128);
        
        print_algorithm_comparison_summary(datasetPath);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmarking: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
