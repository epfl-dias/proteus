/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

// Based on: https://github.com/facebook/folly/blob/master/folly/AtomicBitSet.h

#ifndef UTILS_ATOMIC_BIT_SET_HPP_
#define UTILS_ATOMIC_BIT_SET_HPP_

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>

namespace utils {

/**
 * An atomic bitset of fixed size (specified at compile time).
 */
template <size_t N>
class AtomicBitSet {
 public:
  /**
   * Construct an AtomicBitSet; all bits are initially false.
   */
  AtomicBitSet();

  AtomicBitSet(const AtomicBitSet&) = delete;
  AtomicBitSet& operator=(const AtomicBitSet&) = delete;

  /**
   * Set bit idx to true, using the given memory order. Returns the
   * previous value of the bit.
   *
   * Note that the operation is a read-modify-write operation due to the use
   * of fetch_or.
   */
  bool set(size_t idx, std::memory_order order = std::memory_order_seq_cst);

  /**
   * Set bit idx to false, using the given memory order. Returns the
   * previous value of the bit.
   *
   * Note that the operation is a read-modify-write operation due to the use
   * of fetch_and.
   */
  bool reset(size_t idx, std::memory_order order = std::memory_order_seq_cst);

  /**
   * Set bit idx to the given value, using the given memory order. Returns
   * the previous value of the bit.
   *
   * Note that the operation is a read-modify-write operation due to the use
   * of fetch_and or fetch_or.
   *
   * Yes, this is an overload of set(), to keep as close to std::bitset's
   * interface as possible.
   */
  bool set(size_t idx, bool value,
           std::memory_order order = std::memory_order_seq_cst);

  /**
   * Read bit idx.
   */
  bool test(size_t idx,
            std::memory_order order = std::memory_order_seq_cst) const;

  /**
   * Same as test() with the default memory order.
   */
  bool operator[](size_t idx) const;

  // Additions:

  size_t count(std::memory_order order = std::memory_order_seq_cst) const;
  void reset(std::memory_order order = std::memory_order_seq_cst);
  bool all(std::memory_order order = std::memory_order_seq_cst) const;
  bool any(std::memory_order order = std::memory_order_seq_cst) const;

  /**
   * Return the size of the bitset.
   */
  constexpr size_t size() const { return N; }

 private:
  // Pick the largest lock-free type available
#if (ATOMIC_LLONG_LOCK_FREE == 2)
  typedef unsigned long long BlockType;
#elif (ATOMIC_LONG_LOCK_FREE == 2)
  typedef unsigned long BlockType;
#else
  // Even if not lock free, what can we do?
  typedef unsigned int BlockType;
#endif
  typedef std::atomic<BlockType> AtomicBlockType;

  static constexpr size_t kBitsPerBlock =
      std::numeric_limits<BlockType>::digits;

  static constexpr size_t allBitsSet = std::numeric_limits<BlockType>::max();

  static constexpr size_t blockIndex(size_t bit) { return bit / kBitsPerBlock; }

  static constexpr size_t bitOffset(size_t bit) { return bit % kBitsPerBlock; }

  // avoid casts
  static constexpr BlockType kOne = 1;

  static constexpr size_t numberOfBlocks = N / kBitsPerBlock;

  static_assert((N / kBitsPerBlock) % sizeof(BlockType) == 0,
                "N should be in log2");

  std::array<AtomicBlockType, numberOfBlocks> data_;
};

// value-initialize to zero
template <size_t N>
inline AtomicBitSet<N>::AtomicBitSet() : data_() {}

template <size_t N>
inline bool AtomicBitSet<N>::set(size_t idx, std::memory_order order) {
  assert(idx < N * kBitsPerBlock);
  BlockType mask = kOne << bitOffset(idx);
  return data_[blockIndex(idx)].fetch_or(mask, order) & mask;
}

template <size_t N>
inline bool AtomicBitSet<N>::reset(size_t idx, std::memory_order order) {
  assert(idx < N * kBitsPerBlock);
  BlockType mask = kOne << bitOffset(idx);
  return data_[blockIndex(idx)].fetch_and(~mask, order) & mask;
}

template <size_t N>
inline bool AtomicBitSet<N>::set(size_t idx, bool value,
                                 std::memory_order order) {
  return value ? set(idx, order) : reset(idx, order);
}

template <size_t N>
inline bool AtomicBitSet<N>::test(size_t idx, std::memory_order order) const {
  assert(idx < N * kBitsPerBlock);
  BlockType mask = kOne << bitOffset(idx);
  return data_[blockIndex(idx)].load(order) & mask;
}

template <size_t N>
inline bool AtomicBitSet<N>::operator[](size_t idx) const {
  return test(idx);
}

// More functions to match std::bitset interface
template <size_t N>
inline bool AtomicBitSet<N>::any(std::memory_order order) const {
  for (size_t __i = 0; __i < numberOfBlocks; __i++)
    if (__builtin_popcountl(data_[__i].load(order)) > 0) return true;
  return false;
}

template <size_t N>
inline bool AtomicBitSet<N>::all(std::memory_order order) const {
  // return value ? set(idx, order) : reset(idx, order);

  for (const auto& block : data_) {
    if (block.load(order) != allBitsSet) return false;
  }

  return true;
}

template <size_t N>
inline void AtomicBitSet<N>::reset(std::memory_order order) {
  for (auto& block : data_) {
    block.store(0, order);
  }
}

template <size_t N>
inline size_t AtomicBitSet<N>::count(std::memory_order order) const {
  size_t __result = 0;

  for (size_t __i = 0; __i < numberOfBlocks; __i++)
    __result += __builtin_popcountl(data_[__i].load(order));

  return __result;
}

}  // namespace utils

#endif /* UTILS_ATOMIC_BIT_SET_HPP_ */
