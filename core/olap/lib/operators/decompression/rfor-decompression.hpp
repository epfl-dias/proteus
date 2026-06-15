/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#ifndef RFOR_DECOMPRESSION_HPP
#define RFOR_DECOMPRESSION_HPP

#include <cuda_runtime.h>
#include <vector>
#include <span>
#include <cstdint>
#include <olap/operators/decompression-config.hpp>

// Constants for RFOR decompression.
// Uses the pre-decompression decode strategy contract.
static constexpr int RFOR_BLOCK_THREADS =
    proteus::rfor::predecompress::kBlockThreads;
static constexpr int RFOR_ITEMS_PER_THREAD =
    proteus::rfor::predecompress::kItemsPerThread;
static constexpr int RFOR_TILE_SIZE = proteus::rfor::predecompress::kTileSize;
static constexpr int RFOR_TILE_SIZE_512 = 512;
static constexpr int RFOR_TILE_SIZE_1024 = 1024;
struct RforKernelShape {
    int block_threads;
    int items_per_thread;
    int tile_size;
};

// RFOR Tile Metadata - all metadata at the beginning of chunk
struct RFORTileMetadata {
    uint32_t val_reference;    // FOR reference for values
    uint32_t val_bitwidth;     // Bit width for values
    uint32_t val_count;        // Number of unique values
    uint32_t rl_reference;     // FOR reference for run lengths
    uint32_t rl_bitwidth;      // Bit width for run lengths
    uint32_t rl_count;         // Number of run lengths (should equal val_count)
};

static constexpr size_t RFOR_METADATA_SIZE = sizeof(RFORTileMetadata); // 24 bytes

struct RFORPBMDBlockDesc {
    const char* compressed_block;
    int* output_block;
    uint32_t num_chunks;
    uint32_t tile_prefix;
};

// GPU device functions for RFOR decompression (original benchmark implementations)
__forceinline__ __device__ int decodeElementRFOR(int i,
                                                 const uint8_t* data_block,
                                                 uint32_t packed_words,
                                                 uint32_t reference,
                                                 uint32_t bitwidth);

// GPU decompression function for RFOR chunks
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void decompressRFORTile(
    RFORTileMetadata* metadata,          // Pointer to metadata section
    uint32_t* val_data,                  // Pointer to values data section
    uint32_t* rl_data,                   // Pointer to RL data section
    uint32_t* shared_buffer,
    int (&items_value)[ITEMS_PER_THREAD],
    int (&items_run_length)[ITEMS_PER_THREAD]);

// RFOR GPU kernel
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRFORKernel(
    RFORTileMetadata** metadata_ptrs,    // Pointers to metadata sections
    uint32_t** val_data_ptrs,            // Pointers to values data sections
    uint32_t** rl_data_ptrs,             // Pointers to RL data sections
    int** output_ptrs,                   // Direct pointers to output locations
    size_t total_tiles);


/**
 * CPU decompression for single RFOR tile
 * @param metadata Pointer to tile's metadata section
 * @param val_data Pointer to tile's values data section
 * @param rl_data Pointer to tile's run-length data section
 * @param output_tile Output buffer for decompressed tile data
 * @return Number of elements decompressed, or negative on error
 */
int decompressRFORTile(const RFORTileMetadata* metadata,
                      const uint32_t* val_data,
                      const uint32_t* rl_data,
                      int* output_tile,
                      int tile_size);

/**
 * CPU decompression for RFOR chunks
 * @param chunk_sizes Vector containing sizes of chunks (each chunk = one tile)
 * @param compressed_buffer Buffer containing RFOR chunks
 * @param output_buffer Output buffer for decompressed data
 * @param max_decomp_chunk_size Decompressed chunk size (bytes), used to infer tile_size
 * @return Number of bytes decompressed, or negative on error
 */
int decompressRFORChunks(const std::vector<uint32_t>& chunk_sizes,
                        std::span<const char> compressed_buffer,
                        std::span<char> output_buffer,
                        int max_decomp_chunk_size,
                        const std::vector<uint32_t>& val_section_sizes);


/**
 * Launch RFOR GPU decompression kernel
 * @param device_metadata_ptrs Array of pointers to tile metadata on GPU
 * @param device_val_data_ptrs Array of pointers to values data on GPU
 * @param device_rl_data_ptrs Array of pointers to run-length data on GPU
 * @param device_uncompressed_ptrs Array of pointers to output buffers on GPU
 * @param total_tiles Number of tiles to decompress
 * @param stream CUDA stream (nullptr for synchronous)
 * @return Number of bytes decompressed on success, negative on error
 */
int launchRFORDecompressionKernel(
    RFORTileMetadata** device_metadata_ptrs,
    uint32_t** device_val_data_ptrs,
    uint32_t** device_rl_data_ptrs,
    int** device_uncompressed_ptrs,
    size_t total_tiles,
    int tile_size,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream);

int launchRFORDecompressionKernelPBMD(
    const RFORPBMDBlockDesc* device_block_descs,
    size_t num_blocks,
    size_t total_tiles,
    int tile_size,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream);

[[nodiscard]] RforKernelShape resolveRforKernelShape(int tile_size);
[[nodiscard]] RforKernelShape resolveRforKernelShape(
    int tile_size,
    int block_threads,
    int items_per_thread);

/**
 * RFOR-specific decompression function for the memove operator
 * @param chunk_sizes Vector containing sizes of chunks
 * @param compressed_buffer Buffer containing RFOR chunks
 * @param output_buffer Output buffer for decompressed data
 * @param max_decomp_chunk_size Decompressed chunk size (bytes), used to infer tile_size
 * @return Number of bytes decompressed, or negative on error
 */
[[nodiscard]] int decompress_rfor_block(
    const std::vector<uint32_t>& chunk_sizes,
    std::span<const char> compressed_buffer,
    std::span<char> output_buffer,
    int max_decomp_chunk_size,
    const std::vector<uint32_t>& val_section_sizes);

#endif //RFOR_DECOMPRESSION_HPP

