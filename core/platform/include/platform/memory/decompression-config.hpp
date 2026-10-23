/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                Ecole Polytechnique Federale de Lausanne

                            All Rights Reserved.
*/

#ifndef PROTEUS_PLATFORM_MEMORY_DECOMPRESSION_CONFIG_HPP
#define PROTEUS_PLATFORM_MEMORY_DECOMPRESSION_CONFIG_HPP

#include <cstdint>

namespace proteus::rfor {

namespace cb2t {
inline constexpr int kBlockThreads = 128;
inline constexpr int kItemsPerThread = 4;
inline constexpr int kTileSize = kBlockThreads * kItemsPerThread;
static_assert(
    kBlockThreads * kItemsPerThread == kTileSize,
    "RFOR CB2T defaults must satisfy block_threads * items_per_thread == tile_size");
}  // namespace cb2t

namespace predecompress {
inline constexpr int kBlockThreads = 256;
inline constexpr int kItemsPerThread = 4;
inline constexpr int kTileSize = kBlockThreads * kItemsPerThread;
static_assert(
    kBlockThreads * kItemsPerThread == kTileSize,
    "RFOR predecompress defaults must satisfy block_threads * items_per_thread == tile_size");

struct RuntimeShape {
  int block_threads;
  int items_per_thread;
  int tile_size;
};

[[nodiscard]] inline constexpr RuntimeShape resolveRuntimeShape(
    const int tile_size,
    const int block_threads_override,
    const int items_per_thread_override) {
  if ((block_threads_override > 0) != (items_per_thread_override > 0)) {
    return {0, 0, 0};
  }

  int default_block_threads = 0;
  int default_items_per_thread = 0;
  if (tile_size == 512) {
    default_block_threads = 128;
    default_items_per_thread = 4;
  } else if (tile_size == 1024) {
    default_block_threads = kBlockThreads;
    default_items_per_thread = kItemsPerThread;
  } else {
    return {0, 0, 0};
  }

  const int block_threads =
      block_threads_override > 0 ? block_threads_override : default_block_threads;
  const int items_per_thread =
      items_per_thread_override > 0 ? items_per_thread_override
                                    : default_items_per_thread;

  if (block_threads <= 0 || items_per_thread <= 0 ||
      block_threads * items_per_thread != tile_size) {
    return {0, 0, 0};
  }

  const bool supported_512 =
      tile_size == 512 &&
      ((block_threads == 64 && items_per_thread == 8) ||
       (block_threads == 128 && items_per_thread == 4) ||
       (block_threads == 256 && items_per_thread == 2));
  const bool supported_1024 =
      tile_size == 1024 &&
      ((block_threads == 512 && items_per_thread == 2) ||
       (block_threads == 256 && items_per_thread == 4) ||
       (block_threads == 128 && items_per_thread == 8));

  if (!(supported_512 || supported_1024)) {
    return {0, 0, 0};
  }

  return {block_threads, items_per_thread, tile_size};
}
}  // namespace predecompress

inline constexpr int kCb2tBlockThreads = cb2t::kBlockThreads;
inline constexpr int kCb2tItemsPerThread = cb2t::kItemsPerThread;
inline constexpr int kCb2tTileSize = cb2t::kTileSize;

}  // namespace proteus::rfor

namespace proteus::flbp {

namespace cb2t {
inline constexpr int kBlockThreads = 256;
inline constexpr int kItemsPerThread = 4;
inline constexpr int kTileSize = kBlockThreads * kItemsPerThread;
static_assert(
    kBlockThreads * kItemsPerThread == kTileSize,
    "FLBP CB2T defaults must satisfy block_threads * items_per_thread == tile_size");
inline constexpr uint32_t kVectorSize = 1024;
inline constexpr uint32_t kMetadataBytes = 12;
}  // namespace cb2t

namespace predecompress {
inline constexpr uint32_t kVectorSize = 1024;
inline constexpr int kBlockThreads = 32;
inline constexpr int kItemsPerThread =
    static_cast<int>(kVectorSize) / kBlockThreads;
inline constexpr int kTileSize = kBlockThreads * kItemsPerThread;
static_assert(
    kBlockThreads * kItemsPerThread == kTileSize,
    "FLBP predecompress defaults must satisfy block_threads * items_per_thread == tile_size");
static_assert(
    static_cast<uint32_t>(kTileSize) == kVectorSize,
    "FLBP predecompress default tile size must cover one full vector");
inline constexpr uint32_t kMetadataBytes = 12;

struct RuntimeShape {
  int block_threads;
  int items_per_thread;
  int tile_size;
};

[[nodiscard]] inline constexpr RuntimeShape resolveRuntimeShape(
    const int block_threads_override,
    const int items_per_thread_override) {
  const int block_threads =
      block_threads_override > 0 ? block_threads_override : kBlockThreads;
  const int items_per_thread =
      items_per_thread_override > 0 ? items_per_thread_override
                                    : kItemsPerThread;
  if (block_threads <= 0 || items_per_thread <= 0 || block_threads > 1024) {
    return {0, 0, 0};
  }

  const int tile_size = block_threads * items_per_thread;
  if (tile_size <= 0 ||
      (kVectorSize % static_cast<uint32_t>(tile_size)) != 0u) {
    return {0, 0, 0};
  }

  return {block_threads, items_per_thread, tile_size};
}
}  // namespace predecompress

inline constexpr uint32_t kCb2tVectorSize = cb2t::kVectorSize;
inline constexpr uint32_t kCb2tMetadataBytes = cb2t::kMetadataBytes;
inline constexpr int kCb2tBlockThreads = cb2t::kBlockThreads;
inline constexpr int kCb2tItemsPerThread = cb2t::kItemsPerThread;
inline constexpr int kCb2tTileSize = cb2t::kTileSize;
inline constexpr uint32_t kPredecompressVectorSize = predecompress::kVectorSize;
inline constexpr int kPredecompressBlockThreads = predecompress::kBlockThreads;
inline constexpr int kPredecompressItemsPerThread =
    predecompress::kItemsPerThread;
inline constexpr int kPredecompressTileSize = predecompress::kTileSize;
inline constexpr uint32_t kPredecompressMetadataBytes =
    predecompress::kMetadataBytes;

}  // namespace proteus::flbp

namespace proteus::decompression {

enum class Cb2tCodec : uint8_t {
  kRfor,
  kFlbp,
};

struct Cb2tRuntimeDefaults {
  int block_threads;
  int items_per_thread;
  int tile_size;
};

inline constexpr Cb2tRuntimeDefaults getCb2tRuntimeDefaults(
    const Cb2tCodec codec) {
  switch (codec) {
    case Cb2tCodec::kRfor:
      return {
          proteus::rfor::cb2t::kBlockThreads,
          proteus::rfor::cb2t::kItemsPerThread,
          proteus::rfor::cb2t::kTileSize,
      };
    case Cb2tCodec::kFlbp:
      return {
          proteus::flbp::cb2t::kBlockThreads,
          proteus::flbp::cb2t::kItemsPerThread,
          proteus::flbp::cb2t::kTileSize,
      };
  }

  return {
      proteus::rfor::cb2t::kBlockThreads,
      proteus::rfor::cb2t::kItemsPerThread,
      proteus::rfor::cb2t::kTileSize,
  };
}

}  // namespace proteus::decompression

#endif  // PROTEUS_PLATFORM_MEMORY_DECOMPRESSION_CONFIG_HPP
