/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include "rfor-decompression.hpp"
#include <platform/util/glog.hpp>
#include <cstring>
#include <cub/cub.cuh>

using namespace cub;

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRFORKernel(
    RFORTileMetadata** metadata_ptrs,
    uint32_t** val_data_ptrs,
    uint32_t** rl_data_ptrs,
    int** output_ptrs,
    size_t total_tiles);

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRFORKernelPBMD(
    const RFORPBMDBlockDesc* block_descs,
    size_t num_blocks,
    size_t total_tiles);

namespace {
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
constexpr size_t rforSharedMemBytes() {
    using BlockExchange =
        cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
    constexpr size_t kTempBytes =
        sizeof(typename BlockExchange::TempStorage) >
                sizeof(typename BlockScan::TempStorage)
            ? sizeof(typename BlockExchange::TempStorage)
            : sizeof(typename BlockScan::TempStorage);
    constexpr size_t kTempWords =
        (kTempBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    constexpr size_t kWorkingWords =
        static_cast<size_t>(BLOCK_THREADS) * ITEMS_PER_THREAD;
    return (kTempWords + kWorkingWords) * sizeof(uint32_t);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
int launchRforKernelTyped(
    RFORTileMetadata **device_metadata_ptrs,
    uint32_t **device_val_data_ptrs,
    uint32_t **device_rl_data_ptrs,
    int **device_uncompressed_ptrs,
    size_t total_tiles,
    cudaStream_t stream) {
    constexpr int kTileSize = BLOCK_THREADS * ITEMS_PER_THREAD;
    dim3 grid(total_tiles);
    dim3 block(BLOCK_THREADS);
    constexpr size_t kSharedMemBytes =
        rforSharedMemBytes<BLOCK_THREADS, ITEMS_PER_THREAD>();

    runRFORKernel<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<grid, block, kSharedMemBytes, stream>>>(
            device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
            device_uncompressed_ptrs, total_tiles);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cout << "RFOR kernel launch failed: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return -1;
    }

    if (stream == nullptr) {
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cout << "RFOR kernel execution failed: "
                      << cudaGetErrorString(sync_error) << std::endl;
            return -2;
        }
        return static_cast<int>(total_tiles * kTileSize * sizeof(int));
    }

    return 0;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
int launchRforPbmdKernelTyped(const RFORPBMDBlockDesc *device_block_descs,
                              size_t num_blocks, size_t total_tiles,
                              cudaStream_t stream) {
    constexpr int kTileSize = BLOCK_THREADS * ITEMS_PER_THREAD;
    dim3 grid(total_tiles);
    dim3 block(BLOCK_THREADS);
    constexpr size_t kSharedMemBytes =
        rforSharedMemBytes<BLOCK_THREADS, ITEMS_PER_THREAD>();

    runRFORKernelPBMD<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<grid, block, kSharedMemBytes, stream>>>(device_block_descs,
                                                   num_blocks, total_tiles);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cout << "RFOR PBMD kernel launch failed: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return -1;
    }

    if (stream == nullptr) {
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cout << "RFOR PBMD kernel execution failed: "
                      << cudaGetErrorString(sync_error) << std::endl;
            return -2;
        }
        return static_cast<int>(total_tiles * kTileSize * sizeof(int));
    }

    return 0;
}
}  // namespace

RforKernelShape resolveRforKernelShape(int tile_size) {
    return resolveRforKernelShape(tile_size, 0, 0);
}

RforKernelShape resolveRforKernelShape(int tile_size,
                                       int block_threads,
                                       int items_per_thread) {
    const auto runtime_shape =
        proteus::rfor::predecompress::resolveRuntimeShape(
            tile_size, block_threads, items_per_thread);
    return {
        runtime_shape.block_threads,
        runtime_shape.items_per_thread,
        runtime_shape.tile_size
    };
}

__forceinline__ __device__ uint32_t load_u32_unaligned_rfor(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

__forceinline__ __device__ int decodeElementRFOR(int i,
                                                 const uint8_t* data_block,
                                                 uint32_t packed_words,
                                                 uint32_t reference,
                                                 uint32_t bitwidth) {
    if (bitwidth == 0) return reference;

    uint32_t start_bitindex = (bitwidth * i);
    uint32_t start_intindex = (start_bitindex >> 5);
    start_bitindex = start_bitindex & (32-1);

    if (start_intindex >= packed_words) {
        return reference;
    }

    const auto* word_ptr =
        data_block + static_cast<size_t>(start_intindex) * sizeof(uint32_t);
    uint64_t element_block = load_u32_unaligned_rfor(word_ptr);
    if (start_bitindex + bitwidth > 32 && (start_intindex + 1) < packed_words) {
        const uint64_t hi =
            load_u32_unaligned_rfor(word_ptr + sizeof(uint32_t));
        element_block |= (hi << 32);
    }

    uint64_t mask = (bitwidth >= 32) ? 0xFFFFFFFFULL : ((1ULL << bitwidth) - 1ULL);
    uint32_t element = static_cast<uint32_t>((element_block >> start_bitindex) & mask);
    return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void decompressRFORTile(
    RFORTileMetadata* metadata,
    uint32_t* val_data,
    uint32_t* rl_data,
    uint32_t* shared_buffer,
    int (&items_value)[ITEMS_PER_THREAD],
    int (&items_run_length)[ITEMS_PER_THREAD]) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
    constexpr uint32_t kTileElems =
        static_cast<uint32_t>(BLOCK_THREADS * ITEMS_PER_THREAD);
    constexpr size_t kTempBytes =
        sizeof(typename BlockExchange::TempStorage) >
                sizeof(typename BlockScan::TempStorage)
            ? sizeof(typename BlockExchange::TempStorage)
            : sizeof(typename BlockScan::TempStorage);
    constexpr size_t kTempWords =
        (kTempBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);

    uint32_t count = metadata->val_count;
    uint32_t num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
    uint32_t data_offset = 0;

    const uint32_t val_word_count =
        static_cast<uint32_t>(((static_cast<uint64_t>(metadata->val_count) *
                                metadata->val_bitwidth) +
                               31ULL) >>
                              5U);
    const uint32_t rl_word_count =
        static_cast<uint32_t>(((static_cast<uint64_t>(metadata->rl_count) *
                                metadata->rl_bitwidth) +
                               31ULL) >>
                              5U);
    const auto* val_data_bytes = reinterpret_cast<const uint8_t*>(val_data);
    const auto* rl_data_bytes = reinterpret_cast<const uint8_t*>(rl_data);

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && threadIdx.x + data_offset < count) {
            items_value[i] = decodeElementRFOR(
                threadIdx.x + data_offset, val_data_bytes, val_word_count,
                metadata->val_reference, metadata->val_bitwidth);
            items_run_length[i] = decodeElementRFOR(
                threadIdx.x + data_offset, rl_data_bytes, rl_word_count,
                metadata->rl_reference, metadata->rl_bitwidth);
        } else {
            items_value[i] = 0;
            items_run_length[i] = 0;
        }
        data_offset += num_decode;
    }

    __syncthreads();

    typename BlockScan::TempStorage *temp_storage_scan =
        reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage *temp_storage_exchange =
        reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    uint32_t* working_area = shared_buffer + kTempWords;

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < kTileElems) {
            working_area[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < kTileElems) {
            working_area[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < kTileElems) {
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

    uint32_t rle_offset = 0;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode &&
            threadIdx.x + rle_offset < kTileElems) {
            working_area[threadIdx.x + rle_offset] = items_value[i];
        }
        rle_offset += num_decode;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < kTileElems) {
            items_value[i] = working_area[items_run_length[i]];
        } else {
            items_value[i] = 0;
        }
    }
    __syncthreads();
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRFORKernel(
    RFORTileMetadata** metadata_ptrs,
    uint32_t** val_data_ptrs,
    uint32_t** rl_data_ptrs,
    int** output_ptrs,
    size_t total_tiles) {

    int tile_id = blockIdx.x;
    if (tile_id >= total_tiles) return;

    RFORTileMetadata* metadata = metadata_ptrs[tile_id];
    uint32_t* val_data = val_data_ptrs[tile_id];
    uint32_t* rl_data = rl_data_ptrs[tile_id];
    int* output = output_ptrs[tile_id];

    // CUB TempStorage may require stronger-than-4-byte alignment.
    // Use explicitly aligned dynamic shared memory to avoid misaligned accesses.
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_buffer = reinterpret_cast<uint32_t*>(shared_raw);

    int items_value[ITEMS_PER_THREAD];
    int items_run_length[ITEMS_PER_THREAD];

    decompressRFORTile<BLOCK_THREADS, ITEMS_PER_THREAD>(
        metadata, val_data, rl_data, shared_buffer, items_value, items_run_length);

    __syncthreads();
    constexpr int kTileElems = BLOCK_THREADS * ITEMS_PER_THREAD;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int output_idx = i * BLOCK_THREADS + threadIdx.x;
        if (output_idx < kTileElems) {
            output[output_idx] = items_value[i];
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRFORKernelPBMD(
    const RFORPBMDBlockDesc* block_descs,
    size_t num_blocks,
    size_t total_tiles) {
    int tile_id = blockIdx.x;
    if (tile_id >= total_tiles || num_blocks == 0) return;

    size_t lo = 0;
    size_t hi = num_blocks;
    while (lo + 1 < hi) {
        const size_t mid = (lo + hi) >> 1;
        if (tile_id < static_cast<int>(block_descs[mid].tile_prefix)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    const auto desc = block_descs[lo];
    const uint32_t local_tile =
        static_cast<uint32_t>(tile_id - static_cast<int>(desc.tile_prefix));
    if (local_tile >= desc.num_chunks) return;

    constexpr uint32_t kPbmdMagic = 0x50424D44;  // "PBMD"
    constexpr uint32_t kHeaderWords = 3;         // magic, version, num_chunks
    const auto* header_i32 = reinterpret_cast<const uint32_t*>(desc.compressed_block);
    if (header_i32[0] != kPbmdMagic) return;

    const uint32_t num_chunks = header_i32[2];
    if (local_tile >= num_chunks) return;

    // PBMD layout: [magic][algo][num_chunks][chunk_offsets...][payload chunks...]
    // Each chunk begins with its RFORTileMetadata header; val_section_size is
    // derived from the header instead of being stored in PBMD.
    const uint32_t* chunk_offsets_or_sizes = header_i32 + kHeaderWords;

    uint32_t chunk_offset = 0;
    const bool pbmd_stores_offsets = (num_chunks > 0) && (chunk_offsets_or_sizes[0] == 0);
    if (pbmd_stores_offsets) {
        chunk_offset = chunk_offsets_or_sizes[local_tile];
    } else {
        for (uint32_t i = 0; i < local_tile; ++i) {
            chunk_offset += chunk_offsets_or_sizes[i];
        }
    }

    const size_t header_bytes =
        static_cast<size_t>(kHeaderWords) * sizeof(uint32_t) +
        static_cast<size_t>(num_chunks) * sizeof(uint32_t);
    const char* chunk_start = desc.compressed_block + header_bytes + chunk_offset;

    auto* metadata = reinterpret_cast<RFORTileMetadata*>(
        const_cast<char*>(chunk_start));
    const uint32_t val_packed_words =
        (metadata->val_count * metadata->val_bitwidth + 31u) >> 5u;
    const uint32_t val_section_size =
        static_cast<uint32_t>(sizeof(RFORTileMetadata)) +
        val_packed_words * static_cast<uint32_t>(sizeof(uint32_t));
    auto* val_data = reinterpret_cast<uint32_t*>(
        const_cast<char*>(chunk_start + sizeof(RFORTileMetadata)));
    auto* rl_data = reinterpret_cast<uint32_t*>(
        const_cast<char*>(chunk_start + val_section_size));
    auto* output_tile = reinterpret_cast<int*>(
        reinterpret_cast<char*>(desc.output_block) +
        static_cast<size_t>(local_tile) * BLOCK_THREADS * ITEMS_PER_THREAD *
            sizeof(int));

    // CUB TempStorage may require stronger-than-4-byte alignment.
    // Use explicitly aligned dynamic shared memory to avoid misaligned accesses.
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_buffer = reinterpret_cast<uint32_t*>(shared_raw);
    int items_value[ITEMS_PER_THREAD];
    int items_run_length[ITEMS_PER_THREAD];

    decompressRFORTile<BLOCK_THREADS, ITEMS_PER_THREAD>(
        metadata, val_data, rl_data, shared_buffer, items_value, items_run_length);

    __syncthreads();
    constexpr int kTileElems = BLOCK_THREADS * ITEMS_PER_THREAD;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int output_idx = i * BLOCK_THREADS + threadIdx.x;
        if (output_idx < kTileElems) {
            output_tile[output_idx] = items_value[i];
        }
    }
}

int decompressRFORTile(const RFORTileMetadata* metadata,
                      const uint32_t* val_data,
                      const uint32_t* rl_data,
                      int* output_tile,
                      int tile_size) {

    if (!metadata || !val_data || !rl_data || !output_tile) {
        LOG(ERROR) << "Invalid pointers in decompressRFORTile";
        return -1;
    }
    if (tile_size <= 0) {
        LOG(ERROR) << "Invalid RFOR tile_size=" << tile_size;
        return -1;
    }

    if (metadata->val_count != metadata->rl_count) {
        LOG(ERROR) << "Mismatched val_count and rl_count: " << metadata->val_count << " vs " << metadata->rl_count;
        return -1;
    }

    auto decode_host = [](uint32_t i, const uint32_t *data, uint32_t reference,
                          uint32_t bitwidth, uint32_t word_count) -> int {
      if (bitwidth == 0) return static_cast<int>(reference);
      uint32_t start_bitindex = bitwidth * i;
      uint32_t start_intindex = start_bitindex >> 5;
      start_bitindex &= 31U;
      uint64_t lo = (start_intindex < word_count) ? data[start_intindex] : 0U;
      uint64_t hi = (start_intindex + 1 < word_count) ? data[start_intindex + 1] : 0U;
      uint64_t element_block = (hi << 32) | lo;
      uint64_t mask =
          (bitwidth >= 32) ? 0xFFFFFFFFULL : ((1ULL << bitwidth) - 1ULL);
      return static_cast<int>(reference +
                              static_cast<uint32_t>((element_block >> start_bitindex) & mask));
    };

    const uint32_t val_word_count =
        static_cast<uint32_t>(((static_cast<uint64_t>(metadata->val_count) *
                                metadata->val_bitwidth) +
                               31ULL) >>
                              5U);
    const uint32_t rl_word_count =
        static_cast<uint32_t>(((static_cast<uint64_t>(metadata->rl_count) *
                                metadata->rl_bitwidth) +
                               31ULL) >>
                              5U);

    std::vector<int> values(metadata->val_count);
    for (uint32_t i = 0; i < metadata->val_count; ++i) {
      values[i] = decode_host(i, val_data, metadata->val_reference,
                              metadata->val_bitwidth, val_word_count);
    }

    std::vector<int> run_lengths(metadata->rl_count);
    for (uint32_t i = 0; i < metadata->rl_count; ++i) {
      run_lengths[i] = decode_host(i, rl_data, metadata->rl_reference,
                                   metadata->rl_bitwidth, rl_word_count);
    }

    int output_pos = 0;
    for (uint32_t i = 0;
         i < metadata->val_count && output_pos < tile_size; ++i) {
        int value = values[i];
        int run_length = run_lengths[i];

        for (int j = 0; j < run_length && output_pos < tile_size; ++j) {
            output_tile[output_pos] = value;
            output_pos++;
        }
    }

    return output_pos;
}

int decompressRFORChunks(const std::vector<uint32_t>& chunk_sizes,
                        std::span<const char> compressed_buffer,
                        std::span<char> output_buffer,
                        int max_decomp_chunk_size,
                        const std::vector<uint32_t>& val_section_sizes) {

    if (chunk_sizes.empty()) {
        LOG(ERROR) << "Empty chunk_sizes vector";
        return -1;
    }

    if (val_section_sizes.empty()) {
        LOG(ERROR) << "RFOR decompression requires val_section_sizes from JSON metadata";
        return -1;
    }

    if (chunk_sizes.size() != val_section_sizes.size()) {
        LOG(ERROR) << "Mismatch between chunk_sizes and val_section_sizes";
        return -1;
    }
    if (max_decomp_chunk_size <= 0 ||
        max_decomp_chunk_size % static_cast<int>(sizeof(int)) != 0) {
        LOG(ERROR) << "Invalid RFOR max_decomp_chunk_size=" << max_decomp_chunk_size;
        return -1;
    }
    const int tile_size = max_decomp_chunk_size / static_cast<int>(sizeof(int));
    const auto shape = resolveRforKernelShape(tile_size);
    if (shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported RFOR tile_size=" << tile_size
                   << " for CPU decompression. Supported: "
                   << RFOR_TILE_SIZE_512 << ", " << RFOR_TILE_SIZE_1024;
        return -1;
    }

    size_t buffer_offset = 0;
    size_t output_offset = 0;

    for (size_t i = 0; i < chunk_sizes.size(); ++i) {
        const char* chunk_start = compressed_buffer.data() + buffer_offset;
        const RFORTileMetadata* metadata = reinterpret_cast<const RFORTileMetadata*>(chunk_start);

        size_t val_section_size = val_section_sizes[i];
        size_t rl_data_offset = val_section_size;

        const uint32_t* val_data = reinterpret_cast<const uint32_t*>(chunk_start + sizeof(RFORTileMetadata));
        const uint32_t* rl_data = reinterpret_cast<const uint32_t*>(chunk_start + rl_data_offset);

        int* output_tile = reinterpret_cast<int*>(output_buffer.data() + output_offset);
        int result =
            decompressRFORTile(metadata, val_data, rl_data, output_tile, tile_size);

        if (result < 0) {
            LOG(ERROR) << "Failed to decompress RFOR tile " << i;
            return result;
        }

        buffer_offset += chunk_sizes[i];
        output_offset += static_cast<size_t>(tile_size) * sizeof(int);
    }

    return static_cast<int>(output_offset);
}

int decompress_rfor_block(const std::vector<uint32_t>& chunk_sizes,
                         std::span<const char> compressed_buffer,
                         std::span<char> output_buffer,
                         int max_decomp_chunk_size,
                         const std::vector<uint32_t>& val_section_sizes) {

    return decompressRFORChunks(chunk_sizes, compressed_buffer, output_buffer,
                                max_decomp_chunk_size, val_section_sizes);
}

int launchRFORDecompressionKernel(
    RFORTileMetadata** device_metadata_ptrs,
    uint32_t** device_val_data_ptrs,
    uint32_t** device_rl_data_ptrs,
    int** device_uncompressed_ptrs,
    size_t total_tiles,
    int tile_size,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream) {

    if (total_tiles == 0) {
        return 0;
    }
    const auto shape =
        resolveRforKernelShape(tile_size, block_threads, items_per_thread);
    if (shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported RFOR pre-decompress shape "
                   << block_threads << "x" << items_per_thread
                   << " for tile_size=" << tile_size
                   << ". Supported shapes: 64x8, 128x4, or 256x2 for tile_size=512; "
                   << "512x2, 256x4, or 128x8 for tile_size=1024.";
        return -1;
    }

    if (shape.tile_size == RFOR_TILE_SIZE_512) {
        if (shape.block_threads == 64 && shape.items_per_thread == 8) {
            return launchRforKernelTyped<64, 8>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
        if (shape.block_threads == 128 && shape.items_per_thread == 4) {
            return launchRforKernelTyped<128, 4>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
        if (shape.block_threads == 256 && shape.items_per_thread == 2) {
            return launchRforKernelTyped<256, 2>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
    } else if (shape.tile_size == RFOR_TILE_SIZE_1024) {
        if (shape.block_threads == 512 && shape.items_per_thread == 2) {
            return launchRforKernelTyped<512, 2>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
        if (shape.block_threads == 256 && shape.items_per_thread == 4) {
            return launchRforKernelTyped<256, 4>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
        if (shape.block_threads == 128 && shape.items_per_thread == 8) {
            return launchRforKernelTyped<128, 8>(
                device_metadata_ptrs, device_val_data_ptrs, device_rl_data_ptrs,
                device_uncompressed_ptrs, total_tiles, stream);
        }
    }

    LOG(ERROR) << "Unsupported RFOR pre-decompress dispatch "
               << shape.block_threads << "x" << shape.items_per_thread
               << " for tile_size=" << shape.tile_size;
    return -1;
}

int launchRFORDecompressionKernelPBMD(
    const RFORPBMDBlockDesc* device_block_descs,
    size_t num_blocks,
    size_t total_tiles,
    int tile_size,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream) {

    if (total_tiles == 0) {
        return 0;
    }
    const auto shape =
        resolveRforKernelShape(tile_size, block_threads, items_per_thread);
    if (shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported RFOR pre-decompress PBMD shape "
                   << block_threads << "x" << items_per_thread
                   << " for tile_size=" << tile_size
                   << ". Supported shapes: 64x8, 128x4, or 256x2 for tile_size=512; "
                   << "512x2, 256x4, or 128x8 for tile_size=1024.";
        return -1;
    }

    if (shape.tile_size == RFOR_TILE_SIZE_512) {
        if (shape.block_threads == 64 && shape.items_per_thread == 8) {
            return launchRforPbmdKernelTyped<64, 8>(device_block_descs,
                                                    num_blocks, total_tiles,
                                                    stream);
        }
        if (shape.block_threads == 128 && shape.items_per_thread == 4) {
            return launchRforPbmdKernelTyped<128, 4>(device_block_descs,
                                                     num_blocks, total_tiles,
                                                     stream);
        }
        if (shape.block_threads == 256 && shape.items_per_thread == 2) {
            return launchRforPbmdKernelTyped<256, 2>(device_block_descs,
                                                     num_blocks, total_tiles,
                                                     stream);
        }
    } else if (shape.tile_size == RFOR_TILE_SIZE_1024) {
        if (shape.block_threads == 512 && shape.items_per_thread == 2) {
            return launchRforPbmdKernelTyped<512, 2>(device_block_descs,
                                                     num_blocks, total_tiles,
                                                     stream);
        }
        if (shape.block_threads == 256 && shape.items_per_thread == 4) {
            return launchRforPbmdKernelTyped<256, 4>(device_block_descs,
                                                     num_blocks, total_tiles,
                                                     stream);
        }
        if (shape.block_threads == 128 && shape.items_per_thread == 8) {
            return launchRforPbmdKernelTyped<128, 8>(device_block_descs,
                                                     num_blocks, total_tiles,
                                                     stream);
        }
    }

    LOG(ERROR) << "Unsupported RFOR pre-decompress PBMD dispatch "
               << shape.block_threads << "x" << shape.items_per_thread
               << " for tile_size=" << shape.tile_size;
    return -1;
}


//
// // Explicit template instantiation for common configurations
// template __global__ void runRFORKernel<128, 4>(
//     RFORTileMetadata** metadata_ptrs,
//     uint32_t** val_data_ptrs,
//     uint32_t** rl_data_ptrs,
//     int** output_ptrs,
//     size_t total_tiles);
