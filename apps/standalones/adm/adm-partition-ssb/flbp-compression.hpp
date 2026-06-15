/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#ifndef PROTEUS_FLBP_COMPRESSION_HPP
#define PROTEUS_FLBP_COMPRESSION_HPP

#include <cstdint>
#include <vector>

#include "rfor-compression.hpp"

// Default FastLanes bitpack vector width (matches benchmark/codec contract).
static constexpr uint32_t FLBP_VECTOR_SIZE = 1024;

// Keep field layout aligned with the FastLanes benchmark metadata.
struct FLBPVectorMetadata {
  uint32_t base_value;
  uint8_t bitwidth;
  uint32_t element_count;
};
static_assert(sizeof(FLBPVectorMetadata) == 12,
              "FLBPVectorMetadata must match benchmark layout");

/**
 * Compresses uncompressed blocks with FastLanes-style FOR + bitpack vectors.
 * Chunk layout is: [FLBPVectorMetadata][packed_words]
 */
std::vector<CompressedBlock> compressBlocksFLBP(
    std::vector<std::vector<char>>& uncompressed_blocks,
    size_t vector_size_elems = FLBP_VECTOR_SIZE);

#endif  // PROTEUS_FLBP_COMPRESSION_HPP
