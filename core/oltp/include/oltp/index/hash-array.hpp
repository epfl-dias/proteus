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

// NOTE: THIS IS BROKEN. AS WE DON'T KNOW WHICH KEY TO PLACE WHERE.

#ifndef INDEXES_HASH_ARRAY_HPP_
#define INDEXES_HASH_ARRAY_HPP_

#include <atomic>
#include <cassert>
#include <iostream>

#include "oltp/common/common.hpp"
#include "oltp/index/index.hpp"

namespace indexes {

#define PARTITIONED_INDEX true
#define debug_idx false

template <class K = uint64_t, class V = void *>
class HashArray : public HashIndex<K, V> {
 public:
  char ***arr;
  uint64_t capacity;
  uint64_t capacity_per_partition{};
  partition_id_t partitions;
  //  std::atomic<uint64_t> filler[4];
  uint64_t filler[4]{};

  HashArray(std::string name, uint64_t reserved_capacity);
  HashArray(std::string name, size_t capacity_per_partition,
            uint64_t reserved_capacity);
  ~HashArray() override;

  void report() {
#if debug_idx
    std::cout << "Index: " << name << std::endl;
    std::cout << "PID0: " << filler[0] << std::endl;
    std::cout << "PID1: " << filler[1] << std::endl;
    std::cout << "PID2: " << filler[2] << std::endl;
    std::cout << "PID3: " << filler[3] << std::endl;
#endif
  }

  V find(K key) override {
#if PARTITIONED_INDEX

    ushort pid = key / capacity_per_partition;
    uint64_t idx = key % capacity_per_partition;

    if (__builtin_expect((pid < partitions && idx < capacity_per_partition),
                         1)) {
      return (void *)arr[pid][idx];
    } else {
      LOG(INFO) << "Faulty key: " << key << " | pid: " << pid
                << " | idx: " << idx;
      assert(false);
    }

#else
    assert(key <= capacity);
    return (void *)arr[0][key];
#endif
  }
  inline bool find(K key, V &value) override {
#if PARTITIONED_INDEX

    ushort pid = key / capacity_per_partition;
    uint64_t idx = key % capacity_per_partition;

    if (__builtin_expect((pid < partitions && idx < capacity_per_partition),
                         1)) {
      value = (void *)arr[pid][idx];
      return true;
    } else {
      assert(false);
      return false;
    }

#else

    if (key >= capacity) {
      assert(false);
      return false;
    } else {
      value = (void *)arr[0][key];
      return true;
    }
#endif
  }

  inline bool insert(K key, V &value) override {
#if PARTITIONED_INDEX

    ushort pid = key / capacity_per_partition;
    uint64_t idx = key % capacity_per_partition;
#if debug_idx
    filler[pid]++;
#endif

    if (__builtin_expect((pid < partitions && idx < capacity_per_partition),
                         1)) {
      arr[pid][idx] = (char *)value;
      return true;
    } else {
      LOG(FATAL) << "key: " << key << " | pid: " << pid << " | idx: " << idx
                 << " | capacity_per_partition: " << capacity_per_partition
                 << " | partitions: " << (uint32_t)partitions
                 << " | table: " << this->name;

      return false;
    }

#else

    if (key >= capacity) {
      // assert(false);
      return false;
    } else {
      arr[0][key] = (char *)value;
      return true;
    }
#endif
  }
};

extern template class HashArray<uint64_t, void *>;

};  // namespace indexes

#endif /* INDEXES_HASH_ARRAY_HPP_ */
