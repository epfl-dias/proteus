#pragma once

// ---------------------------------------------------------------------------
// Host-side FastLanes compression + GPU upload helpers.
//
// Mirrors rfor_compress.hpp but for the FastLanes encoding family.
// Compression delegates to fastLanesVectorCompress / fastLanesDeltaVectorCompress
// from the shared headers in kernel-fused/fastlanes/src/include/.
//
// Two encoding families are supported:
//   FLBitPackColumnData  — FOR + bitpack (fastlanes_bitpack.cuh)
//   FLDeltaColumnData    — per-lane delta + sign-magnitude + bitpack (fastlanes_delta.cuh)
//
// GPU data structures (FLBitPackGPUData / FLDeltaGPUData) are built from
// pre-uploaded device buffers by build*PointerArrays().  The caller owns the
// device column payload buffers; only the pointer arrays are freed here.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>

#include "fls_gen/pack/pack.hpp"
#include "fls_gen/transpose/transpose.hpp"

#include "encoding_concept.cuh"

static constexpr size_t FLBD_VEC_SIZE = 1024;

// ---------------------------------------------------------------------------
// Per-column compressed data (host side)
// ---------------------------------------------------------------------------
struct FLDeltaColumnData {
    std::vector<char>   h_buffer;
    std::vector<size_t> chunk_offsets;
    size_t              num_vecs;
};

// ---------------------------------------------------------------------------
// Metadata structs — stored at the head of every compressed vector on device.
// ---------------------------------------------------------------------------
struct FastLanesDeltaVectorMetadata {
  uint8_t bitwidth;                // Bitwidth for delta encoding
  uint32_t base_values[32];        // Base values for each lane (32 lanes)
  uint32_t element_count;          // Number of elements in this vector
};

// ---------------------------------------------------------------------------
// GPU column view structs — pointer arrays indexed as:
//   [col * vecs_per_column + vec_idx]
// These are passed directly to CUDA kernels.
// Delta (per-lane delta + sign-magnitude + bitpack) column on device.
// ---------------------------------------------------------------------------
struct FLDeltaGPUData {
  FastLanesDeltaVectorMetadata** metadata_ptrs;
  uint32_t**                     compressed_ptrs;
  size_t                         vecs_per_column;
  size_t                         total_tuples;
};

// ---------------------------------------------------------------------------
// RAII bundles — own compressed host data, device buffers, and pointer arrays.
// Non-copyable; movable. Destructor frees all GPU resources via allocator.
// ---------------------------------------------------------------------------

struct FLDeltaCompressedData {
  std::vector<FLDeltaColumnData>   per_col;
  std::vector<char*>               d_col_bufs;
  FLDeltaGPUData                   gpu_data = {};
  cub::CachingDeviceAllocator*     allocator = nullptr;

  FLDeltaCompressedData() = default;
  FLDeltaCompressedData(const FLDeltaCompressedData&) = delete;
  FLDeltaCompressedData& operator=(const FLDeltaCompressedData&) = delete;
  FLDeltaCompressedData(FLDeltaCompressedData&&) = default;
  FLDeltaCompressedData& operator=(FLDeltaCompressedData&&) = default;

  ~FLDeltaCompressedData() {
    if (!allocator) return;
    if (gpu_data.metadata_ptrs)   { allocator->DeviceFree(gpu_data.metadata_ptrs);   gpu_data.metadata_ptrs   = nullptr; }
    if (gpu_data.compressed_ptrs) { allocator->DeviceFree(gpu_data.compressed_ptrs); gpu_data.compressed_ptrs = nullptr; }
    for (char* buf : d_col_bufs) {
      if (buf) allocator->DeviceFree(buf);
    }
  }
};

// ---------------------------------------------------------------------------
// Bitwidth calculation for sign-magnitude deltas
// Verbatim from bench_delta_batch.cpp::find_bw_sign_mag
// ---------------------------------------------------------------------------

template <int SIZE>
uint8_t find_bw_sign_mag(int32_t* input_data) {
    const uint32_t* transposed_data = reinterpret_cast<const uint32_t*>(input_data);

    uint32_t max_abs = 0;

    for (int i = 0; i < 32; ++i) {
        uint32_t register_0_0;
        uint32_t register_0_1;
        int32_t tmp_0;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 0)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 1)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        uint32_t abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 1)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 2)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 2)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 3)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 3)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 4)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 4)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 5)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 5)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 6)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 6)];
        register_0_1 = transposed_data[(0 * 32) + (i * 1) + (128 * 7)];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[(0 * 32) + (i * 1) + (128 * 7)];
        register_0_1 = transposed_data[64 + (128 * 0) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 0) + i];
        register_0_1 = transposed_data[64 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 1) + i];
        register_0_1 = transposed_data[64 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 2) + i];
        register_0_1 = transposed_data[64 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 3) + i];
        register_0_1 = transposed_data[64 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 4) + i];
        register_0_1 = transposed_data[64 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 5) + i];
        register_0_1 = transposed_data[64 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 6) + i];
        register_0_1 = transposed_data[64 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[64 + (128 * 7) + i];
        register_0_1 = transposed_data[32 + (128 * 0) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 0) + i];
        register_0_1 = transposed_data[32 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 1) + i];
        register_0_1 = transposed_data[32 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 2) + i];
        register_0_1 = transposed_data[32 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 3) + i];
        register_0_1 = transposed_data[32 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 4) + i];
        register_0_1 = transposed_data[32 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 5) + i];
        register_0_1 = transposed_data[32 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 6) + i];
        register_0_1 = transposed_data[32 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[32 + (128 * 7) + i];
        register_0_1 = transposed_data[96 + (128 * 0) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 0) + i];
        register_0_1 = transposed_data[96 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 1) + i];
        register_0_1 = transposed_data[96 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 2) + i];
        register_0_1 = transposed_data[96 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 3) + i];
        register_0_1 = transposed_data[96 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 4) + i];
        register_0_1 = transposed_data[96 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 5) + i];
        register_0_1 = transposed_data[96 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;

        register_0_0 = transposed_data[96 + (128 * 6) + i];
        register_0_1 = transposed_data[96 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        abs_delta = (tmp_0 < 0) ? (uint32_t)(-tmp_0) : (uint32_t)tmp_0;
        if (abs_delta > max_abs) max_abs = abs_delta;
    }

    uint8_t standard_bitwidths[] = {4, 8, 12, 16, 20, 24, 28, 32};
    for (int i = 0; i < 8; i++) {
        uint8_t total_bits = standard_bitwidths[i];
        uint8_t mag_bits = total_bits - 1;
        if (mag_bits >= 32) return total_bits;
        uint64_t max_representable = (1ULL << mag_bits) - 1;
        if (max_abs <= max_representable) return total_bits;
    }
    return 32;
}

__host__ __device__ __forceinline__ uint32_t encode_sign_mag(int32_t n, uint8_t total_bits) {
  uint32_t mag_mask = (1U << (total_bits - 1)) - 1;
  uint32_t sign_bit = 1U << (total_bits - 1);

  if (n >= 0) {
    uint32_t magnitude = (uint32_t)n;
    if (magnitude > mag_mask) {
      return magnitude | 0x80000000;
    }
    return magnitude;
  } else {
    uint32_t magnitude = (uint32_t)(-n);
    if (magnitude > mag_mask) {
      return sign_bit | magnitude | 0x40000000;
    }
    return sign_bit | magnitude;
  }
}

// ---------------------------------------------------------------------------
// CPU delta encoding
// Verbatim from bench_delta_batch.cpp::unrsum_sign_mag
// ---------------------------------------------------------------------------

inline void unrsum_sign_mag(const uint32_t* a_in_p, uint32_t* a_out_p, uint8_t bitwidth) {
    const auto in = reinterpret_cast<const uint32_t*>(a_in_p);
    auto out = reinterpret_cast<uint32_t*>(a_out_p);
    uint32_t register_0_0;
    uint32_t register_0_1;
    int32_t tmp_0;

    for (int i = 0; i < 32; ++i) {
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 0)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 1)];
        out[(0 * 32) + (i * 1) + (0)] = 0;

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 1)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 2)];
        out[(i * 1) + (0 * 32) + 128] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 2)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 3)];
        out[(i * 1) + (0 * 32) + 256] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 3)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 4)];
        out[(i * 1) + (0 * 32) + 384] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 4)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 5)];
        out[(i * 1) + (0 * 32) + 512] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 5)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 6)];
        out[(i * 1) + (0 * 32) + 640] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 6)];
        register_0_1 = in[(0 * 32) + (i * 1) + (128 * 7)];
        out[(i * 1) + (0 * 32) + 768] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[(0 * 32) + (i * 1) + (128 * 7)];
        register_0_1 = in[64 + (128 * 0) + i];
        out[(i * 1) + (0 * 32) + 896] = encode_sign_mag(tmp_0, bitwidth);

        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 0) + i];
        a_out_p[64 + (128 * 0) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 1) + i];
        a_out_p[64 + (128 * 1) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 2) + i];
        a_out_p[64 + (128 * 2) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 3) + i];
        a_out_p[64 + (128 * 3) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 4) + i];
        a_out_p[64 + (128 * 4) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 5) + i];
        a_out_p[64 + (128 * 5) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 6) + i];
        a_out_p[64 + (128 * 6) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[64 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[64 + (128 * 7) + i];
        a_out_p[64 + (128 * 7) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 0) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 0) + i];
        a_out_p[32 + (128 * 0) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 1) + i];
        a_out_p[32 + (128 * 1) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 2) + i];
        a_out_p[32 + (128 * 2) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 3) + i];
        a_out_p[32 + (128 * 3) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 4) + i];
        a_out_p[32 + (128 * 4) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 5) + i];
        a_out_p[32 + (128 * 5) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 6) + i];
        a_out_p[32 + (128 * 6) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[32 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[32 + (128 * 7) + i];
        a_out_p[32 + (128 * 7) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 0) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 0) + i];
        a_out_p[96 + (128 * 0) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 1) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 1) + i];
        a_out_p[96 + (128 * 1) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 2) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 2) + i];
        a_out_p[96 + (128 * 2) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 3) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 3) + i];
        a_out_p[96 + (128 * 3) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 4) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 4) + i];
        a_out_p[96 + (128 * 4) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 5) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 5) + i];
        a_out_p[96 + (128 * 5) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 6) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 6) + i];
        a_out_p[96 + (128 * 6) + i] = encode_sign_mag(tmp_0, bitwidth);

        register_0_1 = in[96 + (128 * 7) + i];
        tmp_0 = (int32_t)register_0_1 - (int32_t)register_0_0;
        register_0_0 = in[96 + (128 * 7) + i];
        a_out_p[96 + (128 * 7) + i] = encode_sign_mag(tmp_0, bitwidth);
    }
}

__host__ __device__ __forceinline__ int32_t decode_sign_mag(uint32_t n, uint8_t total_bits) {
  uint32_t sign_mask = 1U << (total_bits - 1);
  uint32_t magnitude = n & ~sign_mask;

  if (n & sign_mask) {
    return -(int32_t)magnitude;
  } else {
    return (int32_t)magnitude;
  }
}

// ---------------------------------------------------------------------------
// Host: compress one 1024-element vector with delta encoding
// Verbatim from bench_delta_batch.cpp::fastLanesDeltaVectorCompress
// ---------------------------------------------------------------------------

inline uint fastLanesDeltaVectorCompress(
    uint32_t* vector_data,
    size_t vector_size,
    std::vector<char>& compressed_vector) {

    if (vector_size == 0) return 0;

    auto* temp_vector    = new uint32_t[FLBD_VEC_SIZE];
    auto* transposed_arr = new uint32_t[FLBD_VEC_SIZE];
    auto* delta_encoded  = new uint32_t[FLBD_VEC_SIZE];

    for (size_t i = 0; i < FLBD_VEC_SIZE; i++) {
        temp_vector[i] = (i < vector_size) ? vector_data[i] : 0;
    }

    generated::transpose::fallback::scalar::transpose_i(temp_vector, transposed_arr);

    auto* delta_array = new int32_t[FLBD_VEC_SIZE];
    memcpy(delta_array, transposed_arr, FLBD_VEC_SIZE * sizeof(uint32_t));

    uint8_t bw = find_bw_sign_mag<FLBD_VEC_SIZE>(delta_array);
    if (bw < 4) bw = 4;

    unrsum_sign_mag(transposed_arr, delta_encoded, bw);

    FastLanesDeltaVectorMetadata metadata;
    metadata.bitwidth = bw;
    metadata.element_count = vector_size;
    for (int i = 0; i < 32; i++) {
        metadata.base_values[i] = transposed_arr[i];
    }

    size_t compressed_data_words = (bw * FLBD_VEC_SIZE + 31) / 32;
    std::vector<uint32_t> compressed_data(compressed_data_words);
    generated::pack::fallback::scalar::pack(delta_encoded, compressed_data.data(), bw);

    size_t metadata_size = sizeof(FastLanesDeltaVectorMetadata);
    size_t data_size     = compressed_data.size() * sizeof(uint32_t);
    size_t total_size    = metadata_size + data_size;

    compressed_vector.resize(total_size);
    char* ptr = compressed_vector.data();
    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;
    std::memcpy(ptr, compressed_data.data(), data_size);

    delete[] temp_vector;
    delete[] transposed_arr;
    delete[] delta_encoded;
    delete[] delta_array;

    return compressed_data_words;
}

static inline std::vector<FLDeltaColumnData> flDeltaCompressColumns(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements)
{
    const int    num_cols   = static_cast<int>(raw_columns.size());
    const size_t total_vecs = (total_elements + FLBD_VEC_SIZE - 1) / FLBD_VEC_SIZE;

    std::vector<FLDeltaColumnData> result(num_cols);

    for (int col = 0; col < num_cols; col++) {
        FLDeltaColumnData& cd = result[col];
        cd.num_vecs = total_vecs;
        cd.chunk_offsets.reserve(total_vecs);

        size_t current_offset = 0;
        for (size_t vec_idx = 0; vec_idx < total_vecs; ++vec_idx) {
            const size_t vec_start = vec_idx * FLBD_VEC_SIZE;
            const size_t vec_size  = std::min(FLBD_VEC_SIZE,
                                              total_elements - vec_start);

            std::vector<uint32_t> vector_data(FLBD_VEC_SIZE);
            for (size_t i = 0; i < FLBD_VEC_SIZE; i++) {
                vector_data[i] = (i < vec_size)
                    ? static_cast<uint32_t>(raw_columns[col][vec_start + i])
                    : static_cast<uint32_t>(raw_columns[col][total_elements - 1]);
            }

            std::vector<char> chunk;
            fastLanesDeltaVectorCompress(vector_data.data(), vec_size, chunk);

            cd.chunk_offsets.push_back(current_offset);
            cd.h_buffer.insert(cd.h_buffer.end(), chunk.begin(), chunk.end());
            current_offset += chunk.size();
        }
    }

    return result;
}

static inline FLDeltaGPUData buildFLDeltaPointerArrays(
    const std::vector<FLDeltaColumnData>& cols,
    const std::vector<char*>&             d_col_bufs,
    size_t                                total_tuples,
    cub::CachingDeviceAllocator&          allocator)
{
    const int    num_cols = static_cast<int>(cols.size());
    const size_t num_vecs = cols[0].num_vecs;
    const size_t num_ptrs = static_cast<size_t>(num_cols) * num_vecs;

    std::vector<FastLanesDeltaVectorMetadata*> h_meta(num_ptrs);
    std::vector<uint32_t*>                     h_comp(num_ptrs);

    for (int col = 0; col < num_cols; col++) {
        for (size_t vec = 0; vec < num_vecs; vec++) {
            char* d_chunk = d_col_bufs[col] + cols[col].chunk_offsets[vec];
            const size_t idx = static_cast<size_t>(col) * num_vecs + vec;
            h_meta[idx] = reinterpret_cast<FastLanesDeltaVectorMetadata*>(d_chunk);
            h_comp[idx] = reinterpret_cast<uint32_t*>(d_chunk + sizeof(FastLanesDeltaVectorMetadata));
        }
    }

    FastLanesDeltaVectorMetadata** d_meta;
    uint32_t**                     d_comp;
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_meta),
                             num_ptrs * sizeof(FastLanesDeltaVectorMetadata*));
    allocator.DeviceAllocate(reinterpret_cast<void**>(&d_comp),
                             num_ptrs * sizeof(uint32_t*));

    cudaMemcpy(d_meta, h_meta.data(),
               num_ptrs * sizeof(FastLanesDeltaVectorMetadata*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp, h_comp.data(),
               num_ptrs * sizeof(uint32_t*), cudaMemcpyHostToDevice);

    FLDeltaGPUData out;
    out.metadata_ptrs   = d_meta;
    out.compressed_ptrs = d_comp;
    out.vecs_per_column = num_vecs;
    out.total_tuples    = total_tuples;
    return out;
}

// ---------------------------------------------------------------------------
// Factory: compress, upload, and wire up pointer arrays in one call.
// Returns an RAII bundle whose destructor handles all cleanup.
// ---------------------------------------------------------------------------
inline FLDeltaCompressedData compressAndUploadFLDelta(
    const std::vector<std::vector<int>>& raw_columns,
    size_t                               total_elements,
    cub::CachingDeviceAllocator&         allocator)
{
    FLDeltaCompressedData bundle;
    bundle.allocator = &allocator;
    bundle.per_col   = flDeltaCompressColumns(raw_columns, total_elements);

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

    bundle.gpu_data = buildFLDeltaPointerArrays(
        bundle.per_col, bundle.d_col_bufs, total_elements, allocator);
    return bundle;
}

static_assert(CompressedBundle<FLDeltaCompressedData>,
    "FLDeltaCompressedData must satisfy CompressedBundle");
