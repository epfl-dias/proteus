/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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
#pragma once

#ifndef PROTEUS_ART_KEY_HPP
#define PROTEUS_ART_KEY_HPP

#include <array>
#include <bit>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <platform/memory/allocator.hpp>
#include <utility>
#include <vector>

#include "platform/memory/memory-manager.hpp"

namespace art_olc {

typedef uint8_t key_unit_t;
typedef size_t len_t;

template <size_t N>
class ARTKeyFixedWidth {
 public:
  key_unit_t _data[N]{};

 public:
  key_unit_t *data() { return _data; }

  explicit ARTKeyFixedWidth(key_unit_t *data) {
    switch (N) {
      case 8:
        *((uint64_t *)(_data)) = *((uint64_t *)(data));
        break;
      case 4:
        *((uint32_t *)(_data)) = *((uint32_t *)(data));
        break;
      case 2:
        *((uint16_t *)(_data)) = *((uint16_t *)(data));
        break;
      default:
        memcpy(_data, data, N);
    }
  }

  key_unit_t &operator[](std::size_t i) { return _data[i]; }
  const key_unit_t &operator[](std::size_t i) const { return _data[i]; }
};

class ARTKeyVariableWidth {
  key_unit_t *_data;

 public:
  key_unit_t &operator[](std::size_t i) { return _data[i]; }
  const key_unit_t &operator[](std::size_t i) const { return _data[i]; }
};

template <typename K>
class ARTKey {
 public:
  len_t _key_len;
  typename std::conditional<std::is_integral<K>::value,
                            ARTKeyFixedWidth<sizeof(K)>,
                            ARTKeyVariableWidth>::type _key_holder;

  explicit ARTKey(key_unit_t *data, len_t key_len)
      : _key_len(key_len), _key_holder(data) {}

  ARTKey(ARTKey const &artKey)
      : _key_len(artKey._key_len), _key_holder(artKey._key_holder) {}
  ARTKey(ARTKey const &&artKey)
      : _key_len(std::move(artKey._key_len)),
        _key_holder(std::move(artKey._key_holder)) {}

  ARTKey &operator=(ARTKey const &artKey) {
    _key_len = artKey._key_len;
    _key_holder = artKey._key_holder;
    return *this;
  }

  ARTKey &operator=(ARTKey const &&artKey) {
    _key_len = std::move(artKey._key_len);
    _key_holder = std::move(artKey._key_holder);
    return *this;
  }

  key_unit_t *getData() { return _key_holder.data(); }

  static void destroy(ARTKey *k) {}
};

}  // namespace art_olc

#endif  // PROTEUS_ART_KEY_HPP
