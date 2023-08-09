/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_HASH_CUCKOO_PARTITIONED_HPP
#define PROTEUS_HASH_CUCKOO_PARTITIONED_HPP

#include <atomic>
#include <cassert>
#include <iostream>
#include <libcuckoo/cuckoohash_map.hh>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

#include "oltp/common/common.hpp"
#include "oltp/index/index.hpp"

namespace indexes {

template <class K = uint64_t, class V = void *>
class CuckooPartitioned : public HashIndex<K, V> {
 public:
  uint64_t capacity;
  uint64_t capacity_per_partition{};
  partition_id_t partitions;

  libcuckoo::cuckoohash_map<K, V> **indexPartitions;

  CuckooPartitioned(std::string name, uint64_t reserved_capacity);
  CuckooPartitioned(std::string name, size_t capacity_per_partition,
                    uint64_t reserved_capacity);
  ~CuckooPartitioned() override;

  bool update(const K &key, V &value) override {
    ushort pid = getPid(key);
    uint64_t idx = getIdx(key);

    if (__likely(pid < partitions && idx < capacity_per_partition)) {
      return indexPartitions[pid]->update(key, value);
    } else {
      LOG(FATAL) << "Faulty key[" << this->name << "]: " << key
                 << " | pid: " << pid << " | idx: " << idx;
    }
  }

  V find(K key) override {
    ushort pid = getPid(key);
    uint64_t idx = getIdx(key);

    if (__likely(pid < partitions && idx < capacity_per_partition)) {
      return indexPartitions[pid]->find(key);
    } else {
      LOG(FATAL) << "Faulty key[" << this->name << "]: " << key
                 << " | pid: " << pid << " | idx: " << idx;
    }
  }
  inline bool find(K key, V &value) override {
    ushort pid = getPid(key);
    uint64_t idx = getIdx(key);

    if (__likely(pid < partitions && idx < capacity_per_partition)) {
      return indexPartitions[pid]->find(key, value);
    } else {
      // assert(false);
      return false;
    }
  }

  inline bool insert(K key, V &value) override {
    ushort pid = getPid(key);
    uint64_t idx = getIdx(key);

    if (__likely(pid < partitions && idx < capacity_per_partition)) {
      return indexPartitions[pid]->insert(key, value);

    } else {
      LOG(FATAL) << "key: " << key << " | pid: " << pid << " | idx: " << idx
                 << " | capacity_per_partition: " << capacity_per_partition
                 << " | partitions: " << (uint32_t)partitions
                 << " | table: " << this->name;

      return false;
    }
  }

 private:
  inline auto getPid(K key) {
    // ushort pid = key / capacity_per_partition;
    return key / capacity_per_partition;
  }
  inline auto getIdx(K key) {
    // uint64_t idx = key % capacity_per_partition;
    return key % capacity_per_partition;
  }
};

extern template class CuckooPartitioned<uint64_t, void *>;

};  // namespace indexes

#endif  // PROTEUS_HASH_CUCKOO_PARTITIONED_HPP
