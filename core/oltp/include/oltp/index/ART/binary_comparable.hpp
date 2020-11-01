/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_BINARY_COMPARABLE_HPP
#define PROTEUS_BINARY_COMPARABLE_HPP

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <platform/common/common.hpp>
#include <utility>
#include <vector>

template <typename K = uint64_t>
class BinaryComparableKey;

template <typename K>
std::ostream &operator<<(std::ostream &out, const BinaryComparableKey<K> &r);

template <typename K>
class BinaryComparableKey {
  // FIXME: add a static_assert that key should be size of power of 2.

 public:
  BinaryComparableKey(const BinaryComparableKey &) = default;
  BinaryComparableKey() { BinaryComparableKey(0); }

  explicit BinaryComparableKey(K val) {
    if constexpr (sizeof(K) == 1) {
      byte_arr[0] = val;
    }
    if constexpr (sizeof(K) == 2) {
      *((uint16_t *)byte_arr) = val;  //__builtin_bswap16(val);
    }

    if constexpr (sizeof(K) == 4) {
      *((uint32_t *)byte_arr) = val;  //__builtin_bswap32(val);
    }
    if constexpr (sizeof(K) == 8) {
      *((uint64_t *)byte_arr) = val;  //__builtin_bswap64(val);
    }
    if constexpr (sizeof(K) > 8) {
      static_assert(sizeof(K) > 8, "unimplemented");
    }
  }

  constexpr size_t size() const { return sizeof(K); }

  uint8_t &operator[](std::size_t idx) { return byte_arr[idx]; }
  const uint8_t &operator[](std::size_t idx) const { return byte_arr[idx]; }

  bool operator==(const BinaryComparableKey &other) const {
    if constexpr (sizeof(K) == 1) {
      return other.byte_arr[0] == this->byte_arr[0];
    }
    if constexpr (sizeof(K) == 2) {
      return (*((uint16_t *)other.byte_arr) == *((uint16_t *)this->byte_arr));
    }
    if constexpr (sizeof(K) == 4) {
      return (*((uint32_t *)other.byte_arr) == *((uint32_t *)this->byte_arr));
    }
    if constexpr (sizeof(K) == 8) {
      return (*((uint64_t *)other.byte_arr) == *((uint64_t *)this->byte_arr));
    }
    if constexpr (sizeof(K) > 8) {
      // FIXME: unoptimized. should be simd, or highest-to-lowest size
      for (auto i = 0; i < this->size(); i++) {
        if (other.byte_arr[i] != this->byte_arr[i]) {
          return false;
        }
      }
      return true;
    }
  }
  K getRawValue() const {
    if constexpr (sizeof(K) == 1) {
      return byte_arr[0];
    }
    if constexpr (sizeof(K) == 2) {
      return *(
          (uint16_t *)byte_arr);  //__builtin_bswap16(*((uint16_t *)byte_arr));
    }

    if constexpr (sizeof(K) == 4) {
      return *(
          (uint32_t *)byte_arr);  //__builtin_bswap32(*((uint32_t *)byte_arr));
    }
    if constexpr (sizeof(K) == 8) {
      return *(
          (uint64_t *)byte_arr);  //__builtin_bswap64(*((uint64_t *)byte_arr));
    }
    if constexpr (sizeof(K) > 8) {
      static_assert(sizeof(K) > 8, "unimplemented");
    }
  }

  size_t key_length() {
    // we have to count trailing zeros as we already have byte-swapped.
    if constexpr (sizeof(K) == 1) {
      if (byte_arr[0] == 0)
        return 0;
      else
        return 1;
    }
    if constexpr (sizeof(K) == 2) {
      uint16_t *tmp = ((uint16_t *)byte_arr);
      if (*tmp == 0) return 0;
      if (*tmp < 256)
        return 1;
      else
        return 2;
    }

    if constexpr (sizeof(K) == 4) {
      if (*((uint32_t *)byte_arr) == 0) return 0;
      return 4 - (__builtin_clz(*((uint32_t *)byte_arr)) / 8);
    }
    if constexpr (sizeof(K) == 8) {
      // due to undefined result of ctz when x=0;
      if (*((uint64_t *)byte_arr) == 0) return 0;
      return 8 - (__builtin_clzl(*((uint64_t *)byte_arr)) / 8);
    }
    if constexpr (sizeof(K) > 8) {
      // FIXME: unoptimized. should be simd, or highest-to-lowest size
      size_t tmp = 0;
      for (auto i = 0; i < this->size(); i++) {
        if (byte_arr[i] != 0) tmp += 1;
      }
      return tmp;
    }
  }

  char *data() { return (char *)byte_arr; }

 private:
  uint8_t byte_arr[sizeof(K)];
  friend std::ostream &operator<<<>(std::ostream &out,
                                    const BinaryComparableKey<K> &r);
};

template <typename K>
std::ostream &operator<<(std::ostream &out, const BinaryComparableKey<K> &r) {
  out << r.getRawValue();
  return out;
}

#endif  // PROTEUS_BINARY_COMPARABLE_HPP
