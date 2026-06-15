#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "fls_gen/transpose/transpose.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "ssb_utils.h"
#include "crystal_ssb_utils.h"
#include <cstring>

using namespace cub;

const size_t VEC_SIZE = 1024;

struct FastLanesDeltaVectorMetadata {
    uint8_t bitwidth;                // Bitwidth for delta encoding
    uint32_t base_values[32];        // Base values for each lane (32 lanes)
    uint32_t element_count;          // Number of elements in this vector
};

struct FastLanesDeltaBlock {
    std::vector<std::vector<char>> compressed_chunks;  // Each chunk is one vector
    size_t num_values;
};

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

__host__ __device__ __forceinline__ int32_t decode_sign_mag(uint32_t n, uint8_t total_bits) {
    uint32_t sign_mask = 1U << (total_bits - 1);
    uint32_t magnitude = n & ~sign_mask;

    if (n & sign_mask) {
        return -(int32_t)magnitude;
    } else {
        return (int32_t)magnitude;
    }
}

// Bitwidth calculation for sign-magnitude deltas
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

        if (mag_bits >= 32) {
            return total_bits;
        }

        uint64_t max_representable = (1ULL << mag_bits) - 1;

        if (max_abs <= max_representable) {
            return total_bits;
        }
    }

    return 32;
}

// CPU delta encoding method
void unrsum_sign_mag(const uint32_t* a_in_p, uint32_t* a_out_p, uint8_t bitwidth) {
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

// GPU delta reconstruction with sign-magnitude
__device__ __forceinline__ void d_rsum_32_sign_mag(
    const uint32_t* in, uint32_t* out, const uint32_t* base, uint8_t bitwidth) {

    uint32_t trd_idx = threadIdx.x % 32;
    int32_t r_1 = (int32_t)base[trd_idx];

    uint32_t r_0 = *(in + (0 * 32) + (trd_idx * 1) + 0);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 0] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 128);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 128] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 256);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 256] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 384);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 384] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 512);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 512] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 640);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 640] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 768);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 768] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 896);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 896] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 64);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 64] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 192);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 192] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 320);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 320] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 448);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 448] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 576);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 576] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 704);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 704] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 832);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 832] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 960);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 960] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 32);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 32] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 160);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 160] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 288);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 288] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 416);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 416] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 544);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 544] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 672);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 672] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 800);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 800] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 928);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 928] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 96);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 96] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 224);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 224] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 352);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 352] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 480);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 480] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 608);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 608] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 736);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 736] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 864);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 864] = (uint32_t)r_1;

    r_0 = *(in + (0 * 32) + (trd_idx * 1) + 992);
    r_1 = r_1 + decode_sign_mag(r_0, bitwidth);
    out[(trd_idx * 1) + (0 * 32) + 992] = (uint32_t)r_1;
}


__global__ void runFastLanesDeltaKernel(
    FastLanesDeltaVectorMetadata** metadata_ptrs,
    uint32_t** compressed_data_ptrs,
    int** output_ptrs,
    size_t total_vectors) {

    static __shared__ uint32_t unpacked[1024];
    static __shared__ uint32_t reconstructed[1024];

    size_t vector_id = blockIdx.x;
    if (vector_id >= total_vectors) return;

    FastLanesDeltaVectorMetadata* metadata = metadata_ptrs[vector_id];
    uint32_t* compressed_data = compressed_data_ptrs[vector_id];
    int* output = output_ptrs[vector_id];

    unpack_device(compressed_data, unpacked, metadata->bitwidth);

    // Reconstruct original values using delta decompression
    d_rsum_32_sign_mag(unpacked, reconstructed, metadata->base_values, metadata->bitwidth);

    int thread_id = threadIdx.x;
    for (int i = 0; i < 32; i++) {
        int idx = thread_id + i * 32;
        if (idx < metadata->element_count) {
            output[idx] = (int)reconstructed[idx];
        }
    }
}

// Compression function that creates contiguous chunks
uint fastLanesDeltaVectorCompress(uint32_t* vector_data, size_t vector_size, std::vector<char>& compressed_vector) {
    if (vector_size == 0) return 0;

    // Prepare working arrays
    auto* temp_vector = new uint32_t[VEC_SIZE];
    auto* transposed_arr = new uint32_t[VEC_SIZE];
    auto* delta_encoded = new uint32_t[VEC_SIZE];

    // Pad vector to full size
    for (size_t i = 0; i < VEC_SIZE; i++) {
        if (i < vector_size) {
            temp_vector[i] = vector_data[i];
        } else {
            temp_vector[i] = 0;
        }
    }

    // Transpose to lane format
    generated::transpose::fallback::scalar::transpose_i(temp_vector, transposed_arr);

    // Create delta array for bitwidth calculation
    auto* delta_array = new int32_t[VEC_SIZE];
    memcpy(delta_array, transposed_arr, VEC_SIZE * sizeof(uint32_t));

    // Find bitwidth for sign-magnitude encoding
    uint8_t bw = find_bw_sign_mag<VEC_SIZE>(delta_array);
    if (bw < 4) bw = 4;  // Minimum 4 bits

    // Apply delta encoding
    unrsum_sign_mag(transposed_arr, delta_encoded, bw);

    // Create metadata
    FastLanesDeltaVectorMetadata metadata;
    metadata.bitwidth = bw;
    metadata.element_count = vector_size;

    // Store base values (first element of each lane)
    for (int i = 0; i < 32; i++) {
        metadata.base_values[i] = transposed_arr[i];  // FIXED: was transposed_arr[i * 32]
    }

    // Calculate compressed data size
    size_t compressed_data_words = (bw * VEC_SIZE + 31) / 32;
    std::vector<uint32_t> compressed_data(compressed_data_words);


    generated::pack::fallback::scalar::pack(delta_encoded, compressed_data.data(), bw);

    // Create contiguous layout: [Metadata][Compressed Data]
    size_t metadata_size = sizeof(FastLanesDeltaVectorMetadata);
    size_t data_size = compressed_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + data_size;

    compressed_vector.resize(total_size);
    char* ptr = compressed_vector.data();

    // Copy metadata
    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;

    // Copy compressed data
    std::memcpy(ptr, compressed_data.data(), data_size);

    // Cleanup
    delete[] temp_vector;
    delete[] transposed_arr;
    delete[] delta_encoded;
    delete[] delta_array;

    return compressed_data_words;
}

// Build blocks
std::vector<FastLanesDeltaBlock> createFastLanesDeltaBlocks(uint32_t* h_data, size_t total_elements,
                                                            size_t max_uncompressed_bytes_per_block, CachingDeviceAllocator& alloc) {

    std::vector<FastLanesDeltaBlock> blocks;
    const size_t total_vectors = (total_elements + VEC_SIZE - 1) / VEC_SIZE;
    const size_t max_vectors_per_block = max_uncompressed_bytes_per_block / (VEC_SIZE * sizeof(uint32_t));

    size_t current_vector = 0;
    auto compression_start = std::chrono::high_resolution_clock::now();

    while (current_vector < total_vectors) {
        FastLanesDeltaBlock block;
        size_t vectors_in_block = std::min(max_vectors_per_block, total_vectors - current_vector);

        for (size_t vec_idx = 0; vec_idx < vectors_in_block; vec_idx++) {
            size_t global_vec_idx = current_vector + vec_idx;
            size_t vec_start = global_vec_idx * VEC_SIZE;

            std::vector<uint32_t> vector_data(VEC_SIZE);
            for (size_t i = 0; i < VEC_SIZE; i++) {
                if (vec_start + i < total_elements) {
                    vector_data[i] = h_data[vec_start + i];
                } else {
                    vector_data[i] = h_data[total_elements - 1];  // Pad with last value
                }
            }

            std::vector<char> compressed_chunk;
            fastLanesDeltaVectorCompress(vector_data.data(),
                                       std::min(VEC_SIZE, total_elements - vec_start),
                                       compressed_chunk);
            block.compressed_chunks.push_back(std::move(compressed_chunk));
        }

        block.num_values = std::min(vectors_in_block * VEC_SIZE,
                                   total_elements - (current_vector * VEC_SIZE));
        blocks.push_back(std::move(block));
        current_vector += vectors_in_block;
    }

    auto compression_end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(compression_end - compression_start).count();

    return blocks;
}

float runFastLanesDeltaBenchmark(
    std::vector<FastLanesDeltaBlock>& blocks,
    int num_items,
    CachingDeviceAllocator& alloc,
    uint32_t* h_col_orig) {

    size_t total_uncompressed_bytes = static_cast<size_t>(num_items) * sizeof(int);
    size_t total_compressed_bytes = 0;
    size_t total_vectors = 0;

    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            total_compressed_bytes += chunk.size();
            total_vectors++;
        }
    }

    float compression_ratio = static_cast<float>(total_uncompressed_bytes) / static_cast<float>(total_compressed_bytes);

    // Prepare host-side data structures that will be reused across runs
    std::vector<char> host_buffer;
    std::vector<size_t> chunk_offsets;
    size_t total_size = 0;

    std::cout << "Preparing contiguous host buffer..." << std::endl;

    // Calculate total size for all compressed chunks
    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            total_size += chunk.size();
        }
    }

    // Create contiguous host buffer containing all compressed data
    host_buffer.resize(total_size);
    size_t current_offset = 0;

    for (const auto& block : blocks) {
        for (const auto& chunk : block.compressed_chunks) {
            std::memcpy(host_buffer.data() + current_offset, chunk.data(), chunk.size());
            chunk_offsets.push_back(current_offset);
            current_offset += chunk.size();
        }
    }

    std::cout << "Total compressed data size: " << std::fixed << std::setprecision(2)
              << (total_size / 1e6) << " MB" << std::endl;

    std::cout << "Running " << 11 << " complete allocation cycles (first run discarded)..." << std::endl;

    std::vector<float> all_run_times_ms;

    for (int run = 0; run < 11; ++run) {

        // New GPU allocation and deallocation for each run
        int* device_out = nullptr;
        char* d_buffer = nullptr;

        // Allocate output buffer
        alloc.DeviceAllocate((void**)&device_out, num_items * sizeof(int));

        // Allocate single contiguous buffer for compressed data
        alloc.DeviceAllocate((void**)&d_buffer, total_size);

        cudaMemcpy(d_buffer, host_buffer.data(), total_size, cudaMemcpyHostToDevice);

        // Build pointer arrays
        std::vector<FastLanesDeltaVectorMetadata*> h_metadata_ptrs;
        std::vector<uint32_t*> h_compressed_data_ptrs;
        std::vector<int*> h_output_ptrs;

        h_metadata_ptrs.reserve(total_vectors);
        h_compressed_data_ptrs.reserve(total_vectors);
        h_output_ptrs.reserve(total_vectors);

        // Build pointer arrays pointing into the single large buffer
        size_t global_output_offset = 0;

        for (size_t chunk_idx = 0; chunk_idx < chunk_offsets.size(); chunk_idx++) {
            char* chunk_ptr = d_buffer + chunk_offsets[chunk_idx];

            // Calculate pointers within chunk
            FastLanesDeltaVectorMetadata* metadata = reinterpret_cast<FastLanesDeltaVectorMetadata*>(chunk_ptr);
            uint32_t* compressed_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(FastLanesDeltaVectorMetadata));
            int* output = device_out + global_output_offset;

            h_metadata_ptrs.push_back(metadata);
            h_compressed_data_ptrs.push_back(compressed_data);
            h_output_ptrs.push_back(output);

            global_output_offset += VEC_SIZE;
        }

        // Allocate GPU pointer arrays
        FastLanesDeltaVectorMetadata** d_metadata_ptrs;
        uint32_t** d_compressed_data_ptrs;
        int** d_output_ptrs;

        alloc.DeviceAllocate((void**)&d_metadata_ptrs, total_vectors * sizeof(FastLanesDeltaVectorMetadata*));
        alloc.DeviceAllocate((void**)&d_compressed_data_ptrs, total_vectors * sizeof(uint32_t*));
        alloc.DeviceAllocate((void**)&d_output_ptrs, total_vectors * sizeof(int*));

        // Copy pointer arrays to GPU
        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(),
                  total_vectors * sizeof(FastLanesDeltaVectorMetadata*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_compressed_data_ptrs, h_compressed_data_ptrs.data(),
                  total_vectors * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_ptrs, h_output_ptrs.data(),
                  total_vectors * sizeof(int*), cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        // Single kernel launch
        dim3 grid(total_vectors);
        dim3 block_dim(32);  // FastLanes uses 32 threads per vector

        runFastLanesDeltaKernel<<<grid, block_dim, 0, stream>>>(
            d_metadata_ptrs,
            d_compressed_data_ptrs,
            d_output_ptrs,
            total_vectors
        );

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        // Only include runs after the first warmup run
        if (run > 0) {
            all_run_times_ms.push_back(kernel_time_ms);
            std::cout << "Run " << run << ": " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms" << std::endl;
        } else {
            std::cout << "Warmup run: " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms (discarded)" << std::endl;
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        // For the last run, copy results back to host for verification
        if (run == 10) {  // Last run
            std::cout << "Copying results back to host..." << std::endl;
            uint32_t* h_decoded = new uint32_t[num_items];
            cudaMemcpy(h_decoded, device_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);

            // Generate reference data using same transposition as compression
            uint32_t* reference_data = new uint32_t[num_items];
            uint32_t* temp_transposed = new uint32_t[1024];

            size_t total_vectors_ref = (num_items + VEC_SIZE - 1) / VEC_SIZE;
            for (uint64_t vec_idx = 0; vec_idx < total_vectors_ref; vec_idx++) {
                size_t vec_start = vec_idx * VEC_SIZE;
                size_t vec_size = std::min(VEC_SIZE, static_cast<size_t>(num_items) - vec_start);

                uint32_t* temp_vector = new uint32_t[VEC_SIZE];
                for (size_t i = 0; i < VEC_SIZE; i++) {
                    temp_vector[i] = (i < vec_size) ? h_col_orig[vec_start + i] : 0;
                }

                generated::transpose::fallback::scalar::transpose_i(temp_vector, temp_transposed);
                memcpy(reference_data + (vec_idx * VEC_SIZE), temp_transposed, VEC_SIZE * sizeof(uint32_t));
                delete[] temp_vector;
            }

            bool verification_passed = true;
            size_t error_count = 0;
            const size_t max_errors_to_show = 5;

            for (int i = 0; i < num_items; ++i) {
                if (h_decoded[i] != reference_data[i]) {
                    if (error_count < max_errors_to_show) {
                        std::cerr << "ERROR: idx " << i << " : expected=" << reference_data[i]
                                 << ", got=" << h_decoded[i] << std::endl;
                    }
                    verification_passed = false;
                    error_count++;
                    if (error_count >= max_errors_to_show) break;
                }
            }

            std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED") << std::endl;

        }

        // Cleanup all GPU allocations for this run
        alloc.DeviceFree(d_buffer);           // Single large buffer
        alloc.DeviceFree(d_metadata_ptrs);
        alloc.DeviceFree(d_compressed_data_ptrs);
        alloc.DeviceFree(d_output_ptrs);
        alloc.DeviceFree(device_out);
    }

    // Calculate metrics
    float avg_kernel_ms = std::accumulate(
        all_run_times_ms.begin(),
        all_run_times_ms.end(),
        0.0f
    ) / all_run_times_ms.size();

    double gpu_decompression_time_s = avg_kernel_ms * 1e-3;
    float overall_decompression_gbps = static_cast<float>(total_uncompressed_bytes) / (gpu_decompression_time_s * 1e9f);

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) << compression_ratio << ":1\n";
    std::cout << "GPU decompression speed: " << overall_decompression_gbps << " GB/s\n";

    return avg_kernel_ms;
}

// Utility functions
size_t getFileSize(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filePath << std::endl;
        return 0;
    }
    size_t size = file.tellg();
    file.close();
    return size;
}

template<typename T>
T* loadColumn(const std::string& filePath, size_t num_elements) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filePath << std::endl;
        return nullptr;
    }

    T* data = new T[num_elements];
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));

    if (file.gcount() != static_cast<std::streamsize>(num_elements * sizeof(T))) {
        std::cerr << "Error: Failed to read expected amount of data from file" << std::endl;
        delete[] data;
        return nullptr;
    }

    file.close();
    return data;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <column_file_path> [max_block_size_mb]" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_discount" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_quantity 4" << std::endl;
    std::cout << "  " << program_name << " ../data/ssbm100/lineorder.csv.lo_quantity 4 64" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string column_file = argv[1];
    size_t max_block_size_mb = 2;

    if (argc >= 3) {
        max_block_size_mb = std::stoul(argv[2]);
    }

    size_t max_block_size = max_block_size_mb << 20;

    CachingDeviceAllocator g_allocator(true);

    std::cout << "=== FastLanes Delta Query Engine Style Benchmark ===\n";
    std::cout << "Column file: " << column_file << std::endl;
    std::cout << "Max block size: " << max_block_size_mb << " MB\n";

    size_t file_size = getFileSize(column_file);
    if (file_size == 0) {
        std::cerr << "Error: Failed to get file size or file is empty" << std::endl;
        return 1;
    }

    if (file_size % sizeof(uint32_t) != 0) {
        std::cerr << "Error: File size is not a multiple of 4 bytes" << std::endl;
        return 1;
    }

    size_t n_elements = file_size / sizeof(uint32_t);

    uint32_t* h_data = loadColumn<uint32_t>(column_file, n_elements);
    if (!h_data) {
        std::cerr << "Error: Failed to load column data" << std::endl;
        return 1;
    }

    auto blocks = createFastLanesDeltaBlocks(h_data, n_elements, max_block_size, g_allocator);

    float total_ms = runFastLanesDeltaBenchmark(blocks, n_elements, g_allocator, h_data);

    delete[] h_data;

    return 0;
}
