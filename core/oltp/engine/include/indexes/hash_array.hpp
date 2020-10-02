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

// THIS IS BROKEN. AS WE DONT KNOW WHICH KEY TO PLACE WHERE.

#ifndef INDEXES_HASH_ARRAY_HPP_
#define INDEXES_HASH_ARRAY_HPP_

#include <atomic>
#include <cassert>
#include <iostream>

#include "storage/memory_manager.hpp"

namespace indexes {

#define IDX_SLACK 0
#define PARTITIONED_INDEX true
#define debug_idx false

// DECLARE_uint64(num_partitions);

template <class K = uint64_t, class V = void *>
class HashArray {
 public:
  char ***arr;
  uint64_t capacity;
  size_t capacity_per_partition;
  uint partitions;
  std::atomic<uint64_t> filler[4];

  std::string name;

  HashArray(std::string name = "", uint64_t num_obj = 72000000);
  ~HashArray();

  //~HashArray() { storage::MemoryManager::free(arr, capacity * sizeof(V)); }

  void report() {
#if debug_idx
    std::cout << "Index: " << name << std::endl;
    std::cout << "PID0: " << filler[0] << std::endl;
    std::cout << "PID1: " << filler[1] << std::endl;
    std::cout << "PID2: " << filler[2] << std::endl;
    std::cout << "PID3: " << filler[3] << std::endl;
#endif
  }

  V find(K key) {
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
  inline bool find(K key, V &value) {
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

  inline bool insert(K key, V &value) {
#if PARTITIONED_INDEX

    ushort pid = key / capacity_per_partition;
    uint64_t idx = key % capacity_per_partition;
#if debug_idx
    filler[pid]++;
#endif

    if (__builtin_expect((pid < partitions && idx < capacity_per_partition),
                         1)) {
      // std::cout << "key: " << key << std::endl;
      // std::cout << "pid: " << pid << std::endl;
      // std::cout << "idx: " << idx << std::endl;
      // std::cout << "capacity_per_partition: " << capacity_per_partition
      //           << std::endl;
      // std::cout << "partitions: " << partitions << std::endl;
      arr[pid][idx] = (char *)value;
      return true;
    } else {
      std::cout << "key: " << key << std::endl;
      std::cout << "pid: " << pid << std::endl;
      std::cout << "idx: " << idx << std::endl;
      std::cout << "capacity_per_partition: " << capacity_per_partition
                << std::endl;
      std::cout << "partitions: " << partitions << std::endl;
      std::cout << name << std::endl;
      assert(false);
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
// template <class K>
// class HashIndex : public cuckoohash_map<K, void*> {
// p_index->find(op.key, val

// void* find(const K &key){

//}

/*template <key, hash_val>
bool delete_fn(const K &key, F fn) {
  const hash_value hv = hashed_key(key);
  const auto b = snapshot_and_lock_two<normal_mode>(hv);
  const table_position pos = cuckoo_find(key, hv.partial, b.i1, b.i2);
  if (pos.status == ok) {
    fn(buckets_[pos.index].mapped(pos.slot));
    return true;
  } else {
    return false;
  }*/
//};

};  // namespace indexes

#endif /* INDEXES_HASH_ARRAY_HPP_ */
