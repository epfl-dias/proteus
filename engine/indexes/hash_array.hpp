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

#ifndef INDEXES_HASH_ARRAY_HPP_
#define INDEXES_HASH_ARRAY_HPP_

#include <iostream>

#include "storage/memory_manager.hpp"

namespace indexes {

// TODO: Partition the index.
#define PARTITIONED_INDEX true

// typedef cuckoohash_map<std::string, std::string> HashIndex;

// template <class key, class hash_val>
// using HashIndex = cuckoohash_map<key, hash_val>;

template <class K = uint64_t, class V = void *>
class HashArray {
 public:
  char ***arr;
  uint64_t capacity;
  size_t capacity_per_partition;
  uint partitions;

  HashArray(uint partitions = 1, uint64_t num_obj = 72000000)
      : capacity(num_obj), partitions(partitions) {
    std::cout << "Creating a hashindex of size: " << num_obj << std::endl;

    size_t size = num_obj * sizeof(char *);
    arr = (char ***)malloc(sizeof(char *) * partitions);

#if PARTITIONED_INDEX
    size_t size_per_partition = ((size / partitions) + 1);
    capacity_per_partition = num_obj / partitions;

    for (int i = 0; i < partitions; i++) {
      arr[i] = (char **)storage::MemoryManager::alloc(size_per_partition, i);
    }
#else

    arr[0] = (char **)storage::MemoryManager::alloc(size, 0);
    capacity_per_partition = capacity;
#endif

    for (int i = 0; i < partitions; i++) {
      uint64_t *pt = (uint64_t *)arr[i];
      int warmup_max =
          (capacity_per_partition * sizeof(char *)) / sizeof(uint64_t);
      for (int i = 0; i < warmup_max; i++) pt[i] = 0;
    }
  }
  //~HashArray() { storage::MemoryManager::free(arr, capacity * sizeof(V)); }
  V find(K key) {
#if PARTITIONED_INDEX

    ushort pid = key / partitions;
    K idx = key % capacity_per_partition;

    if (pid < partitions && idx < capacity_per_partition) {
      return (void *)arr[pid][idx];
    } else {
      assert(false);
    }

#else
    assert(key <= capacity);
    return (void *)arr[0][key];
#endif
  }
  inline bool find(K key, V &value) {
#if PARTITIONED_INDEX

    ushort pid = key / partitions;
    K idx = key % capacity_per_partition;

    if (pid < partitions && idx < capacity_per_partition) {
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

    ushort pid = key / partitions;
    K idx = key % capacity_per_partition;

    if (pid < partitions && idx < capacity_per_partition) {
      arr[pid][idx] = (char *)value;
      return true;
    } else {
      std::cout << "key: " << key << std::endl;
      std::cout << "pid: " << pid << std::endl;
      std::cout << "idx: " << idx << std::endl;
      std::cout << "capacity_per_partition: " << capacity_per_partition
                << std::endl;
      std::cout << "partitions: " << partitions << std::endl;
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
