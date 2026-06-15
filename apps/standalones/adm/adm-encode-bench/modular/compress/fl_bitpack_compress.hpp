#pragma once

// ---------------------------------------------------------------------------
// fl_bitpack_compress.hpp — Host-side FastLanes compression + GPU upload.
//
// Supports two FastLanes encoding families:
//
//   FL_BITPACK  (FOR + bitpack)
//     Each 1024-element vector is reduced by a global FOR base, then
//     bit-packed to `bitwidth` bits/element using the FastLanes scalar packer.
//     Metadata: FastLanesVectorMetadata { base_value, bitwidth, element_count }
//
//   FL_DELTA  (per-lane delta + sign-magnitude + bitpack)
//     The 1024 elements are split into 32 lanes of 32 elements each.
//     Within each lane, consecutive deltas are sign-magnitude encoded and
//     bit-packed.  Each lane has its own base value.
//     Metadata: FastLanesDeltaVectorMetadata { bitwidth, base_values[32], element_count }
//
// Compressed format per vector (on device):
//   [Metadata struct][packed uint32_t words...]
//
// Public API (called by FLBitPackConfig / FLDeltaConfig in the mode headers):
//   compressAndUploadFLBitPack(raw_cols, total_elements, allocator)
//     → FLBitPackCompressedData
//   compressAndUploadFLDelta(raw_cols, total_elements, allocator)
//     → FLDeltaCompressedData
//
// Both return RAII bundles; the destructor frees all GPU resources.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>

#include "fls_gen/pack/pack.hpp"
#include "ssb_utils.h"

#include "encoding_concept.cuh"

static constexpr size_t FLBP_VEC_SIZE = 1024;
// constexpr int FLBP_THREADS          = 32;    // one warp per block
// constexpr int FLBP_ELEMS_PER_THREAD = FLBP_VEC_SIZE / FLBP_THREADS;  // 32

// ---------------------------------------------------------------------------
// Per-column compressed data (host side)
//
// Each column is split into ceil(total_elements / VEC_SIZE) vectors.
// All vectors are stored contiguously in h_buffer; chunk_offsets[vec_idx]
// gives the byte offset of vector vec_idx within h_buffer.
// ---------------------------------------------------------------------------

struct FLBitPackColumnData {
    std::vector<char>   h_buffer;        // contiguous: [chunk0][chunk1]...
    std::vector<size_t> chunk_offsets;   // chunk_offsets[vec_idx] = byte offset in h_buffer
    size_t              num_vecs;
};

// ---------------------------------------------------------------------------
// Metadata structs — stored at the head of every compressed vector on device.
// ---------------------------------------------------------------------------

struct FastLanesVectorMetadata {
    uint32_t base_value;       // FOR base subtracted from all elements before packing
    uint8_t  bitwidth;         // bits per packed element
    uint32_t element_count;    // actual elements in this vector (≤ 1024; last vector may be partial)
};

// ---------------------------------------------------------------------------
// GPU column view structs — pointer arrays indexed as:
//   [col * vecs_per_column + vec_idx]
// These are passed directly to CUDA kernels.
// ---------------------------------------------------------------------------

// Bitpack (FOR + bitpack) column on device.
struct FLBitPackGPUData {
    FastLanesVectorMetadata** metadata_ptrs;    // per-vector metadata
    uint32_t**                compressed_ptrs;  // per-vector packed data (first word)
    size_t                    vecs_per_column;
    size_t                    total_tuples;
};

// ---------------------------------------------------------------------------
// RAII bundles — own compressed host data, device buffers, and pointer arrays.
// Non-copyable (GPU resources have no cheap copy semantics); movable.
// Destructor frees all GPU allocations via the stored allocator pointer.
// ---------------------------------------------------------------------------

struct FLBitPackCompressedData {
    std::vector<FLBitPackColumnData> per_col;           // host compressed columns
    std::vector<char*>               d_col_bufs;        // device payload buffers (one per col)
    FLBitPackGPUData                 gpu_data = {};     // device pointer arrays + kernel view
    cub::CachingDeviceAllocator*     allocator = nullptr; // non-owning; must outlive this struct

    FLBitPackCompressedData() = default;
    FLBitPackCompressedData(const FLBitPackCompressedData&) = delete;
    FLBitPackCompressedData& operator=(const FLBitPackCompressedData&) = delete;
    FLBitPackCompressedData(FLBitPackCompressedData&&) = default;
    FLBitPackCompressedData& operator=(FLBitPackCompressedData&&) = default;

    ~FLBitPackCompressedData() {
        if (!allocator) return;
        if (gpu_data.metadata_ptrs)   { allocator->DeviceFree(gpu_data.metadata_ptrs);   gpu_data.metadata_ptrs   = nullptr; }
        if (gpu_data.compressed_ptrs) { allocator->DeviceFree(gpu_data.compressed_ptrs); gpu_data.compressed_ptrs = nullptr; }
        for (char* buf : d_col_bufs) {
            if (buf) allocator->DeviceFree(buf);
        }
    }
};

inline uint fastLanesVectorCompress(
    uint32_t* vector_data,
    size_t vector_size,
    std::vector<char>& compressed_vector) {

  if (vector_size == 0) return 0;

  auto* working_vec = new int32_t[FLBP_VEC_SIZE];
  for (size_t i = 0; i < FLBP_VEC_SIZE; i++) {
    if (i < vector_size) {
      working_vec[i] = static_cast<int32_t>(vector_data[i]);
    } else {
      working_vec[i] = static_cast<int32_t>(vector_data[vector_size - 1]);
    }
  }

  int base = find_base<FLBP_VEC_SIZE>(working_vec);
  subtract_base<FLBP_VEC_SIZE>(working_vec, base);
  int bw = find_bw<FLBP_VEC_SIZE>(working_vec);
  if (bw%4 != 0) std::cout << "######bw####### = " << bw << "\n";

  FastLanesVectorMetadata metadata;
  metadata.base_value = static_cast<uint32_t>(base);
  metadata.bitwidth = static_cast<uint8_t>(bw);
  metadata.element_count = vector_size;

  size_t compressed_data_words = (bw * FLBP_VEC_SIZE + 31) / 32;
  std::vector<uint32_t> compressed_data(compressed_data_words);

  generated::pack::fallback::scalar::pack(
      reinterpret_cast<uint32_t*>(working_vec),
      compressed_data.data(),
      bw);

  size_t metadata_size = sizeof(FastLanesVectorMetadata);
  size_t data_size = compressed_data.size() * sizeof(uint32_t);
  size_t total_size = metadata_size + data_size;

  compressed_vector.resize(total_size);
  char* ptr = compressed_vector.data();
  std::memcpy(ptr, &metadata, metadata_size);
  ptr += metadata_size;
  std::memcpy(ptr, compressed_data.data(), data_size);

  delete[] working_vec;
  return compressed_data_words;
}

// ---------------------------------------------------------------------------
// Internal: compress all columns into per-vector chunks (host side).
// Each vector is padded to VEC_SIZE by repeating the last element, so the
// decompressor always works on full 1024-element vectors.
// ---------------------------------------------------------------------------

static inline std::vector<FLBitPackColumnData> flBitPackCompressColumns(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements)
{
    const int    num_cols   = static_cast<int>(raw_columns.size());
    const size_t total_vecs = (total_elements + FLBP_VEC_SIZE - 1) / FLBP_VEC_SIZE;

    std::vector<FLBitPackColumnData> result(num_cols);

    for (int col = 0; col < num_cols; col++) {
        FLBitPackColumnData& cd = result[col];
        cd.num_vecs = total_vecs;
        cd.chunk_offsets.reserve(total_vecs);

        size_t current_offset = 0;
        for (size_t vec_idx = 0; vec_idx < total_vecs; ++vec_idx) {
            const size_t vec_start = vec_idx * FLBP_VEC_SIZE;
            const size_t vec_size  = std::min(FLBP_VEC_SIZE,
                                              total_elements - vec_start);

            // Pad partial last vector with the last valid element.
            std::vector<uint32_t> vector_data(FLBP_VEC_SIZE);
            for (size_t i = 0; i < FLBP_VEC_SIZE; i++) {
                vector_data[i] = (i < vec_size)
                    ? static_cast<uint32_t>(raw_columns[col][vec_start + i])
                    : static_cast<uint32_t>(raw_columns[col][total_elements - 1]);
            }

            std::vector<char> chunk;
            fastLanesVectorCompress(vector_data.data(), vec_size, chunk);

            cd.chunk_offsets.push_back(current_offset);
            cd.h_buffer.insert(cd.h_buffer.end(), chunk.begin(), chunk.end());
            current_offset += chunk.size();
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Internal: build GPU pointer arrays from pre-uploaded device column buffers.
//
// For each (col, vec_idx) pair, computes the device pointer to the metadata
// and to the packed data words within the contiguous device buffer.
// The pointer arrays are uploaded to device as flat [col * num_vecs + vec_idx]
// arrays, matching the kernel's indexing scheme.
// ---------------------------------------------------------------------------

static inline FLBitPackGPUData buildFLBitPackPointerArrays(
    const std::vector<FLBitPackColumnData>& cols,
    const std::vector<char*>&               d_col_bufs,
    size_t                                  total_tuples,
    cub::CachingDeviceAllocator&            allocator)
{
    const int    num_cols = static_cast<int>(cols.size());
    const size_t num_vecs = cols[0].num_vecs;
    const size_t num_ptrs = static_cast<size_t>(num_cols) * num_vecs;

    // Build host-side pointer arrays by reading chunk offsets (no device round-trip).
    std::vector<FastLanesVectorMetadata*> h_meta(num_ptrs);
    std::vector<uint32_t*>               h_comp(num_ptrs);

    for (int col = 0; col < num_cols; col++) {
        for (size_t vec = 0; vec < num_vecs; vec++) {
            char* d_chunk = d_col_bufs[col] + cols[col].chunk_offsets[vec];
            const size_t idx = static_cast<size_t>(col) * num_vecs + vec;
            // Metadata is at the start of each chunk.
            h_meta[idx] = reinterpret_cast<FastLanesVectorMetadata*>(d_chunk);
            // Packed data immediately follows the metadata struct.
            h_comp[idx] = reinterpret_cast<uint32_t*>(d_chunk + sizeof(FastLanesVectorMetadata));
        }
    }

    FastLanesVectorMetadata** d_meta;
    uint32_t**                d_comp;
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_meta),
                             num_ptrs * sizeof(FastLanesVectorMetadata*));
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_comp),
                             num_ptrs * sizeof(uint32_t*));

    cudaMemcpy(d_meta, h_meta.data(),
               num_ptrs * sizeof(FastLanesVectorMetadata*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp, h_comp.data(),
               num_ptrs * sizeof(uint32_t*), cudaMemcpyHostToDevice);

    FLBitPackGPUData out;
    out.metadata_ptrs   = d_meta;
    out.compressed_ptrs = d_comp;
    out.vecs_per_column = num_vecs;
    out.total_tuples    = total_tuples;
    return out;
}

// ---------------------------------------------------------------------------
// Public factory functions — compress, upload, and wire up pointer arrays.
//
// Steps:
//   1. Compress raw int32 columns into per-vector chunks on the host.
//   2. Upload each column's contiguous buffer to a device allocation.
//   3. Build device pointer arrays (metadata_ptrs, compressed_ptrs) and
//      upload them; kernels use these to index into the payload buffers.
//   4. Return an RAII bundle; caller does not manage GPU memory directly.
// ---------------------------------------------------------------------------

inline FLBitPackCompressedData compressAndUploadFLBitPack(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements,
    cub::CachingDeviceAllocator&         allocator)
{
    FLBitPackCompressedData bundle;
    bundle.allocator = &allocator;
    bundle.per_col   = flBitPackCompressColumns(raw_columns, total_elements);

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

    bundle.gpu_data = buildFLBitPackPointerArrays(
        bundle.per_col, bundle.d_col_bufs, total_elements, allocator);
    return bundle;
}

static_assert(CompressedBundle<FLBitPackCompressedData>,
    "FLBitPackCompressedData must satisfy CompressedBundle");