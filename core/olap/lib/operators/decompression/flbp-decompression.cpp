/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#include "flbp-decompression.hpp"

#include <iostream>
#include <platform/util/glog.hpp>

namespace {
constexpr uint32_t kFlbpWarpThreads = 32u;

__forceinline__ __device__ uint32_t decodeElementFLBPLaneStreamRuntime(
    const uint32_t vector_idx,
    const uint32_t* __restrict__ packed_data,
    const uint8_t bitwidth) {
    if (bitwidth == 0) return 0u;

    const uint32_t lane = vector_idx & (kFlbpWarpThreads - 1u);
    const uint32_t pos = vector_idx >> 5u;
    const uint32_t bit_pos = pos * static_cast<uint32_t>(bitwidth);
    const uint32_t word_idx = bit_pos >> 5u;
    const uint32_t bit_off = bit_pos & 31u;

    const uint64_t lo = static_cast<uint64_t>(
        packed_data[word_idx * kFlbpWarpThreads + lane]);
    uint64_t packed = lo;
    if (bit_off + bitwidth > 32u &&
        (word_idx + 1u) < static_cast<uint32_t>(bitwidth)) {
        const uint64_t hi = static_cast<uint64_t>(
            packed_data[(word_idx + 1u) * kFlbpWarpThreads + lane]);
        packed |= (hi << 32u);
    }

    const uint64_t mask =
        (bitwidth >= 32u) ? 0xFFFFFFFFull : ((1ull << bitwidth) - 1ull);
    return static_cast<uint32_t>((packed >> bit_off) & mask);
}

__global__ void runFLBPKernel(FLBPVectorMetadata** metadata_ptrs,
                              uint32_t** packed_data_ptrs, int** output_ptrs,
                              size_t total_vectors, int items_per_thread,
                              int tile_size) {
    const size_t vector_id = static_cast<size_t>(blockIdx.x);
    if (vector_id >= total_vectors) return;

    FLBPVectorMetadata* metadata = metadata_ptrs[vector_id];
    uint32_t* packed_data = packed_data_ptrs[vector_id];
    int* output = output_ptrs[vector_id];

    const uint32_t base_value = metadata->base_value;
    const uint8_t bitwidth = metadata->bitwidth;
    const uint32_t element_count = metadata->element_count;
    const uint32_t logical_tile_size = static_cast<uint32_t>(tile_size);
    const uint32_t worker_threads =
        blockDim.x == 0 ? 1u : static_cast<uint32_t>(blockDim.x);

    for (uint32_t tile_offset = 0;
         tile_offset < static_cast<uint32_t>(FLBP_VECTOR_SIZE);
         tile_offset += logical_tile_size) {
        for (int item = 0; item < items_per_thread; ++item) {
            const uint32_t local_idx =
                static_cast<uint32_t>(threadIdx.x) +
                static_cast<uint32_t>(item) * worker_threads;
            if (local_idx >= logical_tile_size) {
                break;
            }

            const uint32_t vector_idx = tile_offset + local_idx;
            if (vector_idx >= static_cast<uint32_t>(FLBP_VECTOR_SIZE)) {
                continue;
            }

            uint32_t decoded = base_value;
            if (vector_idx < element_count) {
                decoded += decodeElementFLBPLaneStreamRuntime(
                    vector_idx, packed_data, bitwidth);
            }
            output[vector_idx] = static_cast<int>(decoded);
        }
    }
}

__global__ void runFLBPKernelPBMD(const FLBPPBMDBlockDesc* block_descs,
                                  size_t num_blocks,
                                  size_t total_vectors,
                                  int items_per_thread,
                                  int tile_size) {
    const size_t vector_id = static_cast<size_t>(blockIdx.x);
    if (vector_id >= total_vectors || num_blocks == 0) return;

    size_t lo = 0;
    size_t hi = num_blocks;
    while (lo + 1 < hi) {
        const size_t mid = (lo + hi) >> 1;
        if (vector_id < static_cast<size_t>(block_descs[mid].vector_prefix)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    const auto desc = block_descs[lo];
    const uint32_t local_vector =
        static_cast<uint32_t>(vector_id -
                              static_cast<size_t>(desc.vector_prefix));
    if (local_vector >= desc.num_chunks) return;

    constexpr uint32_t kPbmdMagic = 0x50424D44;
    constexpr uint32_t kHeaderWords = 3;
    const auto* header_i32 =
        reinterpret_cast<const uint32_t*>(desc.compressed_block);
    if (header_i32[0] != kPbmdMagic) return;

    const uint32_t num_chunks = header_i32[2];
    if (local_vector >= num_chunks) return;

    // PBMD layout: [magic][algo][num_chunks][chunk_offsets...][payload chunks...]
    // Each chunk begins with its FLBPVectorMetadata header; the aux section is
    // exactly that header, so packed_data starts immediately after it.
    const uint32_t* chunk_offsets_or_sizes = header_i32 + kHeaderWords;

    uint32_t chunk_offset = 0;
    const bool pbmd_stores_offsets =
        (num_chunks > 0) && (chunk_offsets_or_sizes[0] == 0);
    if (pbmd_stores_offsets) {
        chunk_offset = chunk_offsets_or_sizes[local_vector];
    } else {
        for (uint32_t i = 0; i < local_vector; ++i) {
            chunk_offset += chunk_offsets_or_sizes[i];
        }
    }

    const size_t header_bytes =
        static_cast<size_t>(kHeaderWords) * sizeof(uint32_t) +
        static_cast<size_t>(num_chunks) * sizeof(uint32_t);
    const char* chunk_start = desc.compressed_block + header_bytes + chunk_offset;

    auto* metadata = reinterpret_cast<FLBPVectorMetadata*>(
        const_cast<char*>(chunk_start));
    auto* packed_data = reinterpret_cast<uint32_t*>(
        const_cast<char*>(chunk_start + sizeof(FLBPVectorMetadata)));
    auto* output = reinterpret_cast<int*>(
        reinterpret_cast<char*>(desc.output_block) +
        static_cast<size_t>(local_vector) * FLBP_VECTOR_SIZE * sizeof(int));

    const uint32_t base_value = metadata->base_value;
    const uint8_t bitwidth = metadata->bitwidth;
    const uint32_t element_count = metadata->element_count;
    const uint32_t logical_tile_size = static_cast<uint32_t>(tile_size);
    const uint32_t worker_threads =
        blockDim.x == 0 ? 1u : static_cast<uint32_t>(blockDim.x);

    for (uint32_t tile_offset = 0;
         tile_offset < static_cast<uint32_t>(FLBP_VECTOR_SIZE);
         tile_offset += logical_tile_size) {
        for (int item = 0; item < items_per_thread; ++item) {
            const uint32_t local_idx =
                static_cast<uint32_t>(threadIdx.x) +
                static_cast<uint32_t>(item) * worker_threads;
            if (local_idx >= logical_tile_size) {
                break;
            }

            const uint32_t vector_idx = tile_offset + local_idx;
            if (vector_idx >= static_cast<uint32_t>(FLBP_VECTOR_SIZE)) {
                continue;
            }

            uint32_t decoded = base_value;
            if (vector_idx < element_count) {
                decoded += decodeElementFLBPLaneStreamRuntime(
                    vector_idx, packed_data, bitwidth);
            }
            output[vector_idx] = static_cast<int>(decoded);
        }
    }
}

int launchFlbpPbmdKernelTyped(const FLBPPBMDBlockDesc* device_block_descs,
                              size_t num_blocks,
                              size_t total_vectors,
                              int block_threads,
                              int items_per_thread,
                              cudaStream_t stream) {
    dim3 grid(total_vectors);
    dim3 block(block_threads);

    runFLBPKernelPBMD<<<grid, block, 0, stream>>>(
        device_block_descs, num_blocks, total_vectors, items_per_thread,
        block_threads * items_per_thread);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cout << "FLBP PBMD kernel launch failed: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return -1;
    }

    if (stream == nullptr) {
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cout << "FLBP PBMD kernel execution failed: "
                      << cudaGetErrorString(sync_error) << std::endl;
            return -2;
        }
        return static_cast<int>(total_vectors * FLBP_VECTOR_SIZE *
                                sizeof(int));
    }

    return 0;
}
}  // namespace

int launchFLBPDecompressionKernel(
    FLBPVectorMetadata** device_metadata_ptrs,
    uint32_t** device_packed_data_ptrs,
    int** device_uncompressed_ptrs,
    size_t total_vectors,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream) {
    if (total_vectors == 0) {
        return 0;
    }

    const auto runtime_shape =
        proteus::flbp::predecompress::resolveRuntimeShape(
            block_threads, items_per_thread);
    if (runtime_shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported FLBP pre-decompress shape "
                   << block_threads << "x" << items_per_thread
                   << "; require positive values, block_threads <= 1024, "
                   << "and block_threads * items_per_thread dividing "
                   << FLBP_VECTOR_SIZE;
        return -1;
    }

    dim3 grid(total_vectors);
    dim3 block(runtime_shape.block_threads);

    runFLBPKernel<<<grid, block, 0, stream>>>(
        device_metadata_ptrs, device_packed_data_ptrs,
        device_uncompressed_ptrs, total_vectors,
        runtime_shape.items_per_thread, runtime_shape.tile_size);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cout << "FLBP kernel launch failed: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return -1;
    }

    if (stream == nullptr) {
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cout << "FLBP kernel execution failed: "
                      << cudaGetErrorString(sync_error) << std::endl;
            return -2;
        }
        return static_cast<int>(total_vectors * FLBP_VECTOR_SIZE *
                                sizeof(int));
    }

    return 0;
}

int launchFLBPDecompressionKernelPBMD(
    const FLBPPBMDBlockDesc* device_block_descs,
    size_t num_blocks,
    size_t total_vectors,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream) {
    if (total_vectors == 0) {
        return 0;
    }

    const auto runtime_shape =
        proteus::flbp::predecompress::resolveRuntimeShape(
            block_threads, items_per_thread);
    if (runtime_shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported FLBP pre-decompress PBMD shape "
                   << block_threads << "x" << items_per_thread
                   << "; require positive values, block_threads <= 1024, "
                   << "and block_threads * items_per_thread dividing "
                   << FLBP_VECTOR_SIZE;
        return -1;
    }

    return launchFlbpPbmdKernelTyped(device_block_descs, num_blocks,
                                     total_vectors, runtime_shape.block_threads,
                                     runtime_shape.items_per_thread, stream);
}

