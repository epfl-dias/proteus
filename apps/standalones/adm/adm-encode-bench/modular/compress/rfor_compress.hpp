#pragma once

// ---------------------------------------------------------------------------
// Host-side RLE+FOR+BinPack compression and GPU upload helpers.
//
// Extracted from hardcoded-SSB1.1-compressed-block-to-tuple.cpp:
//   compressRFORTile  (lines 1520-1681)
//   createRLEColumns  (lines 1683-1728)
//   + upload / free helpers (new in modular design)
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include "encoding_concept.cuh"

// One column's complete compressed data, ready for cudaMemcpy.
struct CompressedColumnData {
    std::vector<char>   h_buffer;         // contiguous: [chunk0][chunk1]...
    std::vector<size_t> chunk_offsets;    // chunk_offsets[tile_idx] = byte offset in h_buffer
    size_t              num_tiles;
};

// Per-tile metadata written at the head of every compressed chunk.
// Layout in the GPU buffer: [RLETileMetadata][val_data_words...][rl_data_words...]
struct RLETileMetadata {
  uint32_t val_reference;
  uint32_t val_bitwidth;
  uint32_t val_count;
  uint32_t rl_reference;
  uint32_t rl_bitwidth;
  uint32_t rl_count;
};

struct RLEColumnGPUData {
  RLETileMetadata** metadata_ptrs;
  uint32_t**        val_data_ptrs;
  uint32_t**        rl_data_ptrs;
  size_t            tiles_per_column;
  size_t            total_tuples;
};

// ---------------------------------------------------------------------------
// RFORCompressedData — RAII bundle owning everything needed to run an RFOR kernel.
//
// Holds compressed host data, device payload buffers, and device pointer arrays.
// The destructor frees all GPU resources via the allocator.
//
// Non-copyable (GPU resources have no cheap copy semantics); movable.
// ---------------------------------------------------------------------------
struct RFORCompressedData {
    std::vector<CompressedColumnData> per_col;       // host compressed columns
    std::vector<char*>                d_col_bufs;    // device payload buffers (one per col)
    RLEColumnGPUData                  gpu_data = {}; // device pointer arrays + view for kernels
    int32_t*                          d_tuples = nullptr; // global-memory decompression buffer
                                                          // (RFOR_GLOBAL only; nullptr otherwise)
    cub::CachingDeviceAllocator*      allocator = nullptr; // non-owning; must outlive this struct

    RFORCompressedData() = default;
    RFORCompressedData(const RFORCompressedData&) = delete;
    RFORCompressedData& operator=(const RFORCompressedData&) = delete;
    RFORCompressedData(RFORCompressedData&&) = default;
    RFORCompressedData& operator=(RFORCompressedData&&) = default;

    // Verified below (after full definition) via static_assert.
    ~RFORCompressedData() {
        if (!allocator) return;
        // Free global-memory decompression buffer (RFOR_GLOBAL only)
        if (d_tuples) { allocator->DeviceFree(d_tuples); d_tuples = nullptr; }
        // Free pointer arrays
        if (gpu_data.metadata_ptrs) { allocator->DeviceFree(gpu_data.metadata_ptrs); gpu_data.metadata_ptrs = nullptr; }
        if (gpu_data.val_data_ptrs) { allocator->DeviceFree(gpu_data.val_data_ptrs); gpu_data.val_data_ptrs = nullptr; }
        if (gpu_data.rl_data_ptrs)  { allocator->DeviceFree(gpu_data.rl_data_ptrs);  gpu_data.rl_data_ptrs  = nullptr; }
        // Free device payload buffers
        for (char* buf : d_col_bufs) {
            if (buf) allocator->DeviceFree(buf);
        }
    }
};

// ---------------------------------------------------------------------------
// Internal: compress one tile of raw int32 data into the binary layout:
//   [RLETileMetadata][val_data_words...][rl_data_words...]
// cumulative_rl: if true, store cumulative prefix-sums of run-lengths
//   (used by RFOR_PREDECODE / RFOR_WARP modes; not needed for GLOBAL/SHARED).
// ---------------------------------------------------------------------------
static inline uint32_t compressRFORTile(
    const int*         tile_data,
    size_t             elems_per_tile,
    std::vector<char>& compressed_tile,
    bool               cumulative_rl = false) {

    if (elems_per_tile == 0) return 0;

    std::vector<int> unique_values;
    std::vector<int> run_lengths;

    int current_value      = tile_data[0];
    int current_run_length = 1;

    for (size_t i = 1; i < elems_per_tile; i++) {
        if (tile_data[i] == current_value) {
            current_run_length++;
        } else {
            unique_values.push_back(current_value);
            run_lengths.push_back(current_run_length);
            current_value      = tile_data[i];
            current_run_length = 1;
        }
    }
    unique_values.push_back(current_value);
    run_lengths.push_back(current_run_length);

    if (cumulative_rl) {
        int cumulative = 0;
        for (auto& rl : run_lengths) {
            cumulative += rl;
            rl = cumulative;
        }
    }

    const uint32_t count = static_cast<uint32_t>(unique_values.size());

    // Compute FOR base + bitwidth for values
    int val_reference = *std::min_element(unique_values.begin(), unique_values.end());
    uint32_t val_bitwidth = 1;
    for (int v : unique_values) {
        uint32_t for_adjusted = static_cast<uint32_t>(v - val_reference);
        if (for_adjusted > 0) {
            uint32_t required_bits = static_cast<uint32_t>(
                std::ceil(std::log2(static_cast<double>(for_adjusted + 1))));
            if (required_bits > val_bitwidth) val_bitwidth = required_bits;
        }
    }
    if (val_bitwidth == 0)  val_bitwidth = 1;
    if (val_bitwidth > 32)  val_bitwidth = 32;

    // Compute FOR base + bitwidth for run-lengths
    int rl_reference = *std::min_element(run_lengths.begin(), run_lengths.end());
    uint32_t rl_bitwidth = 1;
    for (int rl : run_lengths) {
        uint32_t for_adjusted = static_cast<uint32_t>(rl - rl_reference);
        if (for_adjusted > 0) {
            uint32_t required_bits = static_cast<uint32_t>(
                std::ceil(std::log2(static_cast<double>(for_adjusted + 1))));
            if (required_bits > rl_bitwidth) rl_bitwidth = required_bits;
        }
    }
    if (rl_bitwidth == 0)  rl_bitwidth = 1;
    if (rl_bitwidth > 32)  rl_bitwidth = 32;

    RLETileMetadata metadata;
    metadata.val_reference = static_cast<uint32_t>(val_reference);
    metadata.val_bitwidth  = val_bitwidth;
    metadata.val_count     = count;
    metadata.rl_reference  = static_cast<uint32_t>(rl_reference);
    metadata.rl_bitwidth   = rl_bitwidth;
    metadata.rl_count      = count;

    // Pack values
    std::vector<uint32_t> val_data;
    {
        uint32_t current_word = 0;
        uint32_t bits_used    = 0;
        for (size_t i = 0; i < unique_values.size(); i++) {
            uint32_t for_adjusted = static_cast<uint32_t>(unique_values[i] - val_reference);
            if (bits_used + val_bitwidth <= 32) {
                current_word |= (for_adjusted << bits_used);
                bits_used    += val_bitwidth;
                if (bits_used == 32 || i == unique_values.size() - 1) {
                    val_data.push_back(current_word);
                    current_word = 0;
                    bits_used    = 0;
                }
            } else {
                uint32_t bits_in_current = 32 - bits_used;
                uint32_t bits_in_next    = val_bitwidth - bits_in_current;
                current_word |= (for_adjusted << bits_used);
                val_data.push_back(current_word);
                current_word = for_adjusted >> bits_in_current;
                bits_used    = bits_in_next;
                if (i == unique_values.size() - 1 && bits_used > 0) {
                    val_data.push_back(current_word);
                }
            }
        }
    }

    // Pack run-lengths
    std::vector<uint32_t> rl_data;
    {
        uint32_t current_word = 0;
        uint32_t bits_used    = 0;
        for (size_t i = 0; i < run_lengths.size(); i++) {
            uint32_t for_adjusted = static_cast<uint32_t>(run_lengths[i] - rl_reference);
            if (bits_used + rl_bitwidth <= 32) {
                current_word |= (for_adjusted << bits_used);
                bits_used    += rl_bitwidth;
                if (bits_used == 32 || i == run_lengths.size() - 1) {
                    rl_data.push_back(current_word);
                    current_word = 0;
                    bits_used    = 0;
                }
            } else {
                uint32_t bits_in_current = 32 - bits_used;
                uint32_t bits_in_next    = rl_bitwidth - bits_in_current;
                current_word |= (for_adjusted << bits_used);
                rl_data.push_back(current_word);
                current_word = for_adjusted >> bits_in_current;
                bits_used    = bits_in_next;
                if (i == run_lengths.size() - 1 && bits_used > 0) {
                    rl_data.push_back(current_word);
                }
            }
        }
    }

    const size_t metadata_size = sizeof(RLETileMetadata);
    const size_t val_size      = val_data.size() * sizeof(uint32_t);
    const size_t rl_size       = rl_data.size()  * sizeof(uint32_t);
    compressed_tile.resize(metadata_size + val_size + rl_size);

    char* ptr = compressed_tile.data();
    std::memcpy(ptr, &metadata, metadata_size);  ptr += metadata_size;
    std::memcpy(ptr, val_data.data(), val_size);  ptr += val_size;
    std::memcpy(ptr, rl_data.data(), rl_size);

    return count;
}

// ---------------------------------------------------------------------------
// Compress all columns into per-tile chunks.
// elems_per_tile must equal BLOCK_THREADS * ITEMS_PER_THREAD used in the kernel.
// Returns one CompressedColumnData per column.
// ---------------------------------------------------------------------------
static inline std::vector<CompressedColumnData> compressColumns(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements,
    size_t                               elems_per_tile,
    bool                                 cumulative_rl = false) {

    const int num_cols = static_cast<int>(raw_columns.size());
    const size_t total_tiles = (total_elements + elems_per_tile - 1) / elems_per_tile;

    std::vector<CompressedColumnData> result(num_cols);

    for (int col = 0; col < num_cols; col++) {
        CompressedColumnData& ccd = result[col];
        ccd.num_tiles = total_tiles;
        ccd.chunk_offsets.reserve(total_tiles);

        size_t current_offset = 0;
        for (size_t tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
            const size_t tile_start = tile_idx * elems_per_tile;

            // Pad the last (potentially partial) tile with the last valid element.
            std::vector<int> tile_data(elems_per_tile);
            for (size_t i = 0; i < elems_per_tile; i++) {
                const size_t src = (tile_start + i < total_elements)
                                 ? tile_start + i
                                 : total_elements - 1;
                tile_data[i] = raw_columns[col][src];
            }

            std::vector<char> compressed_chunk;
            compressRFORTile(tile_data.data(), elems_per_tile,
                             compressed_chunk, cumulative_rl);

            ccd.chunk_offsets.push_back(current_offset);
            ccd.h_buffer.insert(ccd.h_buffer.end(),
                                compressed_chunk.begin(), compressed_chunk.end());
            current_offset += compressed_chunk.size();
        }
    }

    return result;
}

// Internal: build GPU pointer arrays from pre-uploaded device column buffers.
// Only called by compressAndUploadRFOR.
static inline RLEColumnGPUData buildPointerArrays(
    const std::vector<CompressedColumnData>& cols,
    const std::vector<char*>&                d_col_bufs,
    size_t                                   total_tuples,
    cub::CachingDeviceAllocator&             allocator) {

    const int    num_cols  = static_cast<int>(cols.size());
    const size_t num_tiles = cols[0].num_tiles;
    const size_t num_ptrs  = static_cast<size_t>(num_cols) * num_tiles;

    // Build host-side pointer arrays by reading tile metadata from host buffer
    // (avoids a cudaMemcpyDeviceToHost per tile).
    std::vector<RLETileMetadata*> h_meta(num_ptrs);
    std::vector<uint32_t*>        h_val(num_ptrs);
    std::vector<uint32_t*>        h_rl(num_ptrs);

    for (int col = 0; col < num_cols; col++) {
        for (size_t tile = 0; tile < num_tiles; tile++) {
            // Use the host buffer to read metadata (no device round-trip needed).
            const char* h_chunk = cols[col].h_buffer.data() + cols[col].chunk_offsets[tile];
            RLETileMetadata meta_tmp;
            std::memcpy(&meta_tmp, h_chunk, sizeof(RLETileMetadata));

            const uint32_t val_words =
                (meta_tmp.val_count * meta_tmp.val_bitwidth + 31u) / 32u;
            const size_t val_sec =
                sizeof(RLETileMetadata) + val_words * sizeof(uint32_t);

            char* d_chunk = d_col_bufs[col] + cols[col].chunk_offsets[tile];
            const size_t idx = static_cast<size_t>(col) * num_tiles + tile;
            h_meta[idx] = reinterpret_cast<RLETileMetadata*>(d_chunk);
            h_val [idx] = reinterpret_cast<uint32_t*>(d_chunk + sizeof(RLETileMetadata));
            h_rl  [idx] = reinterpret_cast<uint32_t*>(d_chunk + val_sec);
        }
    }

    RLETileMetadata** d_meta;
    uint32_t**        d_val;
    uint32_t**        d_rl;
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_meta),
                             num_ptrs * sizeof(RLETileMetadata*));
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_val),
                             num_ptrs * sizeof(uint32_t*));
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_rl),
                             num_ptrs * sizeof(uint32_t*));

    cudaMemcpy(d_meta, h_meta.data(),
               num_ptrs * sizeof(RLETileMetadata*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,  h_val.data(),
               num_ptrs * sizeof(uint32_t*),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl,   h_rl.data(),
               num_ptrs * sizeof(uint32_t*),        cudaMemcpyHostToDevice);

    RLEColumnGPUData out;
    out.metadata_ptrs    = d_meta;
    out.val_data_ptrs    = d_val;
    out.rl_data_ptrs     = d_rl;
    out.tiles_per_column = num_tiles;
    out.total_tuples     = total_tuples;
    return out;
}

// ---------------------------------------------------------------------------
// Factory: compress, upload, and wire up pointer arrays in one call.
// Returns an RFORCompressedData whose destructor handles all cleanup.
// ---------------------------------------------------------------------------
inline RFORCompressedData compressAndUploadRFOR(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements,
    size_t                               elems_per_tile,
    cub::CachingDeviceAllocator&         allocator,
    bool                                 cumulative_rl = false)
{
    RFORCompressedData bundle;
    bundle.allocator = &allocator;
    bundle.per_col   = compressColumns(raw_columns, total_elements, elems_per_tile, cumulative_rl);

    const int num_cols = static_cast<int>(bundle.per_col.size());
    bundle.d_col_bufs.resize(num_cols, nullptr);
    for (int c = 0; c < num_cols; c++) {
        allocator.DeviceAllocate(reinterpret_cast<void**>(&bundle.d_col_bufs[c]),
                                 bundle.per_col[c].h_buffer.size());
        cudaMemcpy(bundle.d_col_bufs[c],
                   bundle.per_col[c].h_buffer.data(),
                   bundle.per_col[c].h_buffer.size(),
                   cudaMemcpyHostToDevice);
    }

    bundle.gpu_data = buildPointerArrays(
        bundle.per_col, bundle.d_col_bufs, total_elements, allocator);
    return bundle;
}

static_assert(CompressedBundle<RFORCompressedData>,
    "RFORCompressedData must satisfy CompressedBundle");
