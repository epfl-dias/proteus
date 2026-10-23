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

#include "rfor-compression.hpp"
#include <platform/util/glog.hpp>
#include <cstring>
#include <algorithm>


uint32_t RLEBinPackTile(int* tile_data, size_t tile_size,
                        RFORTileMetadata& metadata,
                        std::vector<uint32_t>& val_data,
                        std::vector<uint32_t>& rl_data,
                        bool precompute_cumulative_rl) {
    if (tile_size == 0) return 0;

    std::vector<int> unique_values;
    std::vector<int> run_lengths;

    int current_value = tile_data[0];
    int current_run_length = 1;

    // Extract RLE pairs
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

    uint32_t count = unique_values.size();

    // Verification of counts of val and rl counts
    CHECK_EQ(unique_values.size(), run_lengths.size())
      << "RLE extraction produced mismatched counts: val_count=" << unique_values.size()
      << " rl_count=" << run_lengths.size();

    if (precompute_cumulative_rl) {
        // Convert run lengths to cumulative run ends once during dataset
        // generation so runtime predecode can binary-search directly.
        for (size_t i = 1; i < run_lengths.size(); ++i) {
            run_lengths[i] += run_lengths[i - 1];
        }
        CHECK_EQ(run_lengths.back(), static_cast<int>(tile_size))
            << "Cumulative RL generation mismatch: expected tile_size="
            << tile_size << " got last_run_end=" << run_lengths.back();
    }


    int val_reference = unique_values[0];
    for (size_t i = 1; i < unique_values.size(); i++) {
        if (unique_values[i] < val_reference) {
            val_reference = unique_values[i];
        }
    }

    uint32_t val_bitwidth = 1;
    for (size_t i = 0; i < unique_values.size(); i++) {
        uint32_t for_adjusted = unique_values[i] - val_reference;
        if (for_adjusted > 0) {
            uint32_t required_bits = static_cast<uint32_t>(ceil(log2(for_adjusted + 1)));
            if (required_bits > val_bitwidth) {
                val_bitwidth = required_bits;
            }
        }
    }

    if (val_bitwidth == 0) val_bitwidth = 1;
    if (val_bitwidth > 32) val_bitwidth = 32;

    val_data.clear();
    uint32_t current_word = 0;
    uint32_t bits_used = 0;

    for (size_t i = 0; i < unique_values.size(); i++) {
        uint32_t for_adjusted = unique_values[i] - val_reference;

        if (bits_used + val_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += val_bitwidth;

            if (bits_used == 32 || i == unique_values.size() - 1) {
                val_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint32_t bits_in_current = 32 - bits_used;
            uint32_t bits_in_next = val_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            val_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == unique_values.size() - 1 && bits_used > 0) {
                val_data.push_back(current_word);
            }
        }
    }


    int rl_reference = run_lengths[0];
    for (size_t i = 1; i < run_lengths.size(); i++) {
        if (run_lengths[i] < rl_reference) {
            rl_reference = run_lengths[i];
        }
    }

    uint32_t rl_bitwidth = 1;
    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint32_t for_adjusted = run_lengths[i] - rl_reference;
        if (for_adjusted > 0) {
            uint32_t required_bits = static_cast<uint32_t>(ceil(log2(for_adjusted + 1)));
            if (required_bits > rl_bitwidth) {
                rl_bitwidth = required_bits;
            }
        }
    }

    if (rl_bitwidth == 0) rl_bitwidth = 1;
    if (rl_bitwidth > 32) rl_bitwidth = 32;

    rl_data.clear();
    current_word = 0;
    bits_used = 0;

    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint32_t for_adjusted = run_lengths[i] - rl_reference;

        if (bits_used + rl_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += rl_bitwidth;

            if (bits_used == 32 || i == run_lengths.size() - 1) {
                rl_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint32_t bits_in_current = 32 - bits_used;
            uint32_t bits_in_next = rl_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            rl_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == run_lengths.size() - 1 && bits_used > 0) {
                rl_data.push_back(current_word);
            }
        }
    }


    metadata.val_reference = static_cast<uint32_t>(val_reference);
    metadata.val_bitwidth = val_bitwidth;
    metadata.val_count = count;
    metadata.rl_reference = static_cast<uint32_t>(rl_reference);
    metadata.rl_bitwidth = rl_bitwidth;
    metadata.rl_count = count;

    return count;
}

std::vector<CompressedBlock> compressBlocksRFOR(
    std::vector<std::vector<char>>& uncompressed_blocks,
    size_t tile_size_elems,
    bool precompute_cumulative_rl) {

    std::vector<CompressedBlock> compressed_blocks;
    compressed_blocks.reserve(uncompressed_blocks.size());

    CHECK_GT(tile_size_elems, 0) << "RFOR tile size must be > 0";

    for (auto& block : uncompressed_blocks) {
        CompressedBlock compressed_block;

        // Convert char data to int data
        CHECK_EQ(block.size() % sizeof(int), 0) << "Block size must be multiple of int size";
        size_t num_ints = block.size() / sizeof(int);
        const int* int_data = reinterpret_cast<const int*>(block.data());

        // Process block in tiles
        size_t num_tiles = (num_ints + tile_size_elems - 1) / tile_size_elems;

        for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            size_t tile_start = tile_idx * tile_size_elems;

            // Pad tile if necessary
            std::vector<int> tile_data(tile_size_elems);
            for (size_t i = 0; i < tile_size_elems; i++) {
                if (tile_start + i < num_ints) {
                    tile_data[i] = int_data[tile_start + i];
                } else {
                    tile_data[i] = int_data[num_ints - 1]; // Pad with last value
                }
            }


            RFORTileMetadata metadata;
            std::vector<uint32_t> val_data, rl_data;
            RLEBinPackTile(tile_data.data(), tile_size_elems, metadata, val_data, rl_data,
                           precompute_cumulative_rl);

            size_t metadata_size = sizeof(RFORTileMetadata);
            size_t val_data_size = val_data.size() * sizeof(uint32_t);
            size_t rl_data_size = rl_data.size() * sizeof(uint32_t);
            size_t total_size = metadata_size + val_data_size + rl_data_size;

            std::vector<char> chunk;
            chunk.resize(total_size);

            char* ptr = chunk.data();


            std::memcpy(ptr, &metadata, metadata_size);
            ptr += metadata_size;

            std::memcpy(ptr, val_data.data(), val_data_size);
            ptr += val_data_size;

            std::memcpy(ptr, rl_data.data(), rl_data_size);

            compressed_block.compressed_chunks.push_back(std::move(chunk));
        }

        compressed_block.num_values = num_ints;
        compressed_blocks.push_back(std::move(compressed_block));

        // Clear original block to save memory
        block.clear();
        block.shrink_to_fit();
    }

    return compressed_blocks;
}
