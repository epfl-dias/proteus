/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef INDEXES_HASH_INDEX_HPP_
#define INDEXES_HASH_INDEX_HPP_

#include <iostream>
#include <libcuckoo/cuckoohash_map.hh>

#include "oltp/index/index.hpp"

namespace indexes {

template <class K, class V = void*>
class HashIndex : public libcuckoo::cuckoohash_map<K, V>, public Index<K, V> {
 public:
  // HashIndex(std::string name ): cuckoohash_map<K, V>(), Index<K,V>(name) {}

  HashIndex(std::string name, rowid_t reserved_capacity)
      : libcuckoo::cuckoohash_map<K, V>(),
        Index<K, V>(name, reserved_capacity) {
    if (reserved_capacity > 0) this->reserve(reserved_capacity);
  }
  HashIndex(std::string name, size_t capacity_per_partition,
            rowid_t reserved_capacity)
      : HashIndex(name, reserved_capacity) {}
};

};  // namespace indexes

#endif /* INDEXES_HASH_INDEX_HPP_ */
