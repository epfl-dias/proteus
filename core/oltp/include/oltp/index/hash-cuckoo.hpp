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

template <class K, class V = void *>
class HashCuckoo : public HashIndex<K, V>,
                   public libcuckoo::cuckoohash_map<K, V> {
 public:
  HashCuckoo(std::string name, uint64_t reserved_capacity)
      : libcuckoo::cuckoohash_map<K, V>(), HashIndex<K, V>(name) {
    if (reserved_capacity > 0) this->reserve(reserved_capacity);
  }
  HashCuckoo(std::string name, size_t capacity_per_partition,
             uint64_t reserved_capacity)
      : HashCuckoo(name, reserved_capacity) {}

  bool update(const K &key, V &value) override {
    return libcuckoo::cuckoohash_map<K, V>::update(key, value);
  }

  V find(K key) override { return libcuckoo::cuckoohash_map<K, V>::find(key); }
  bool find(K key, V &value) override {
    return libcuckoo::cuckoohash_map<K, V>::find(key, value);
  }
  bool insert(K key, V &value) override {
    return libcuckoo::cuckoohash_map<K, V>::insert(key, value);
  }
};

};  // namespace indexes

#endif /* INDEXES_HASH_INDEX_HPP_ */
