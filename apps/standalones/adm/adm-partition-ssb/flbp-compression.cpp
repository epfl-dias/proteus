/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#include "flbp-compression.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>

#include <platform/util/glog.hpp>

#include "fls_gen/pack/pack.hpp"

namespace {

uint8_t bitwidthForMaxResidual(uint32_t max_residual) {
  if (max_residual == 0) {
    return 0;
  }
  return static_cast<uint8_t>(32 - __builtin_clz(max_residual));
}

std::vector<char> compressFLBPVector(const int32_t* data,
                                     size_t element_count_in_vector,
                                     size_t vector_size_elems) {
  CHECK_GT(vector_size_elems, 0) << "FLBP vector size must be > 0";
  CHECK_LE(element_count_in_vector, vector_size_elems)
      << "FLBP element_count exceeds vector size";

  std::vector<int32_t> working(vector_size_elems, 0);
  if (element_count_in_vector > 0) {
    std::memcpy(working.data(), data,
                element_count_in_vector * sizeof(int32_t));
    const int32_t pad_val = working[element_count_in_vector - 1];
    for (size_t i = element_count_in_vector; i < vector_size_elems; ++i) {
      working[i] = pad_val;
    }
  }

  int32_t base = std::numeric_limits<int32_t>::max();
  for (const int32_t v : working) {
    base = std::min(base, v);
  }

  std::vector<uint32_t> residuals(vector_size_elems, 0);
  uint32_t max_residual = 0;
  for (size_t i = 0; i < vector_size_elems; ++i) {
    const int64_t diff = static_cast<int64_t>(working[i]) -
                         static_cast<int64_t>(base);
    CHECK_GE(diff, 0) << "FLBP residual must be non-negative";
    CHECK_LE(diff, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        << "FLBP residual exceeds uint32_t";
    const uint32_t r = static_cast<uint32_t>(diff);
    residuals[i] = r;
    max_residual = std::max(max_residual, r);
  }

  const uint8_t bw = bitwidthForMaxResidual(max_residual);
  const size_t packed_words =
      ((static_cast<size_t>(bw) * vector_size_elems) + 31) / 32;

  std::vector<uint32_t> packed_data(packed_words);
  if (bw > 0 && !packed_data.empty()) {
    generated::pack::fallback::scalar::pack(
        residuals.data(), packed_data.data(), bw);
  }

  FLBPVectorMetadata md{};
  md.base_value = static_cast<uint32_t>(base);
  md.bitwidth = bw;
  md.element_count = static_cast<uint32_t>(element_count_in_vector);

  const size_t total_bytes =
      sizeof(FLBPVectorMetadata) + packed_data.size() * sizeof(uint32_t);
  std::vector<char> chunk(total_bytes);
  std::memcpy(chunk.data(), &md, sizeof(FLBPVectorMetadata));
  if (!packed_data.empty()) {
    std::memcpy(chunk.data() + sizeof(FLBPVectorMetadata), packed_data.data(),
                packed_data.size() * sizeof(uint32_t));
  }
  return chunk;
}

}  // namespace

std::vector<CompressedBlock> compressBlocksFLBP(
    std::vector<std::vector<char>>& uncompressed_blocks,
    size_t vector_size_elems) {
  CHECK_GT(vector_size_elems, 0) << "FLBP vector size must be > 0";
  CHECK_EQ(vector_size_elems, static_cast<size_t>(FLBP_VECTOR_SIZE))
      << "Only FLBP vector_size=1024 is currently supported";

  std::vector<CompressedBlock> compressed_blocks;
  compressed_blocks.reserve(uncompressed_blocks.size());

  for (auto& block : uncompressed_blocks) {
    CHECK_EQ(block.size() % sizeof(int32_t), 0)
        << "Block size must be a multiple of int32_t";

    const size_t num_ints = block.size() / sizeof(int32_t);
    const auto* int_data = reinterpret_cast<const int32_t*>(block.data());

    CompressedBlock compressed_block;
    const size_t num_vectors =
        (num_ints + vector_size_elems - 1) / vector_size_elems;
    compressed_block.compressed_chunks.reserve(num_vectors);

    for (size_t vec_idx = 0; vec_idx < num_vectors; ++vec_idx) {
      const size_t vector_start = vec_idx * vector_size_elems;
      const size_t remaining = num_ints - vector_start;
      const size_t element_count = std::min(vector_size_elems, remaining);
      auto chunk =
          compressFLBPVector(int_data + vector_start, element_count, vector_size_elems);
      compressed_block.compressed_chunks.emplace_back(std::move(chunk));
    }

    compressed_block.num_values = num_ints;
    compressed_blocks.emplace_back(std::move(compressed_block));

    block.clear();
    block.shrink_to_fit();
  }

  return compressed_blocks;
}
