/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#ifndef PROTEUS_FLBP_DECOMPRESSION_HPP
#define PROTEUS_FLBP_DECOMPRESSION_HPP

#include <cuda_runtime.h>
#include <cstdint>
#include <olap/operators/decompression-config.hpp>

static constexpr int FLBP_VECTOR_SIZE =
    proteus::flbp::predecompress::kVectorSize;
static constexpr int FLBP_BLOCK_THREADS =
    proteus::flbp::predecompress::kBlockThreads;
static constexpr int FLBP_ITEMS_PER_THREAD =
    proteus::flbp::predecompress::kItemsPerThread;
static constexpr int FLBP_TILE_SIZE =
    proteus::flbp::predecompress::kTileSize;

struct FLBPPBMDBlockDesc {
    const char* compressed_block;
    int* output_block;
    uint32_t num_chunks;
    uint32_t vector_prefix;
};

struct FLBPVectorMetadata {
    uint32_t base_value;
    uint8_t bitwidth;
    uint32_t element_count;
};
static_assert(sizeof(FLBPVectorMetadata) ==
                  proteus::flbp::predecompress::kMetadataBytes,
              "FLBPVectorMetadata size must match FLBP config");

int launchFLBPDecompressionKernel(
    FLBPVectorMetadata** device_metadata_ptrs,
    uint32_t** device_packed_data_ptrs,
    int** device_uncompressed_ptrs,
    size_t total_vectors,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream);

int launchFLBPDecompressionKernelPBMD(
    const FLBPPBMDBlockDesc* device_block_descs,
    size_t num_blocks,
    size_t total_vectors,
    int block_threads,
    int items_per_thread,
    cudaStream_t stream);

#endif  // PROTEUS_FLBP_DECOMPRESSION_HPP

