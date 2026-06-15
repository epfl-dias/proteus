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

#ifndef PROTEUS_RFOR_COMPRESSION_HPP
#define PROTEUS_RFOR_COMPRESSION_HPP

#include <vector>
#include <span>
#include <cmath>
#include <cstdint>
#include <string>

#include <olap/operators/decompression-config.hpp>

// Forward declaration for CompressedBlock
struct CompressedBlock {
    std::vector<std::vector<char>> compressed_chunks;
    size_t num_values;
};


// Dataset-generation RFOR tile size (in elements).
// Default to CB2T decode contract so generated datasets work with in-GPU
// CompressedBlockToTuples without additional flags.
static constexpr int RFOR_TILE_SIZE = proteus::rfor::cb2t::kTileSize;


struct RFORTileMetadata {
    uint32_t val_reference;    // FOR reference for values
    uint32_t val_bitwidth;     // Bit width for values
    uint32_t val_count;        // Number of unique values
    uint32_t rl_reference;     // FOR reference for run lengths
    uint32_t rl_bitwidth;      // Bit width for run lengths
    uint32_t rl_count;         // Number of run lengths (should equal val_count)
};

static constexpr size_t RFOR_METADATA_SIZE = sizeof(RFORTileMetadata); // 24 bytes

/**

 * Compresses a single tile using RLE + Frame-of-Reference + Bit Packing
 * Returns separated metadata and data sections
 * @param tile_data Input data for one tile
 * @param tile_size Size of the tile in elements
 * @param metadata Output metadata structure
 * @param val_data Output vector for compressed values (data only, no metadata)
 * @param rl_data Output vector for compressed run lengths (data only, no metadata)
 * @return Number of unique value/run-length pairs
 */
uint32_t RLEBinPackTile(int* tile_data, size_t tile_size,
                        RFORTileMetadata& metadata,
                        std::vector<uint32_t>& val_data,
                        std::vector<uint32_t>& rl_data,
                        bool precompute_cumulative_rl = false);

/**

 * Each chunk contains: [metadata][val_data][rl_data]
 * This exactly matches the working microbenchmark structure
 * @param uncompressed_blocks Vector of uncompressed data blocks
 * @param tile_size_elems RFOR tile size in elements
 * @return Vector of compressed blocks, each chunk has 3 sections
 */
std::vector<CompressedBlock> compressBlocksRFOR(
    std::vector<std::vector<char>>& uncompressed_blocks,
    size_t tile_size_elems = RFOR_TILE_SIZE,
    bool precompute_cumulative_rl = false);

#endif // PROTEUS_RFOR_COMPRESSION_HPP
