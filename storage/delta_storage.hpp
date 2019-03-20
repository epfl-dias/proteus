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

#ifndef DELTA_STORAGE_HPP_
#define DELTA_STORAGE_HPP_

#include <sys/mman.h>
#include <cstdlib>
#include <iostream>
#include "glo.hpp"
//#include "indexes/hash_index.hpp"
#include "storage/memory_manager.hpp"
//#include "transactions/transaction_manager.hpp"
namespace storage {
/*
    TODO:
      - Memory management sucks
      - Delta size is not resizeable
      - same locl for all 3 types of data-struc, parition and have fine-grained
          locking
      - templated so that it can have multiple types of indexing
*/

/* Currently DeltaStore is not resizeable*/
class DeltaStore {
 public:
  DeltaStore(size_t rec_size, uint64_t initial_num_objs,
             int num_partitions = -1) {
    if (num_partitions == -1) {
      num_partitions = 4;  // std::thread::hardware_concurrency();
    }

    uint64_t total_ver_capacity =
        initial_num_objs * global_conf::max_ver_factor_per_thread;
    uint64_t total_tuple_capacity = initial_num_objs;

    size_t ver_list_mem_req =
        sizeof(global_conf::mv_version_list) * total_tuple_capacity;

    size_t ver_data_mem_req =
        (rec_size * total_ver_capacity) +
        (sizeof(global_conf::mv_version) * total_ver_capacity);

    int list_numa_id = global_conf::delta_list_numa_id;
    int data_numa_id = global_conf::delta_ver_numa_id;

    for (int i = 0; i < num_partitions; i++) {
      /* TODO: if numa memset is undefined, then allocate memory on the local
               node.
        if(list_numa_id == -1 ){}
        if(data_numa_id == -1 ){}
      */

      // THIS IS PURE HACKED FOR DIAS33

      list_numa_id = i % 4;
      data_numa_id = i % 4;

      std::cout << "PID-" << i << " - memset: " << data_numa_id << std::endl;

      void* mem_list = MemoryManager::alloc(ver_list_mem_req, list_numa_id);
      void* mem_data = MemoryManager::alloc(ver_data_mem_req, data_numa_id);

      // assert(mlock(mem_list, ver_list_mem_req));
      // assert(mlock(mem_data, ver_data_mem_req));

      assert(mem_list != nullptr);
      assert(mem_data != nullptr);

      partitions.push_back(new DeltaPartition(
          (char*)mem_list, mem_chunk(mem_list, ver_list_mem_req, list_numa_id),
          (char*)mem_data, mem_chunk(mem_data, ver_data_mem_req, data_numa_id),
          rec_size, i));
    }

    // void* mem_data = malloc(ver_data_mem_req);
    // void* mem_list = malloc(ver_list_mem_req);

    // // warm-up mem-list
    // std::cout << "\t warming up delta storage" << std::endl;
    // uint64_t* pt = (uint64_t*)mem_list;
    // int warmup_size = ver_list_mem_req / sizeof(uint64_t);
    // for (int i = 0; i < warmup_size; i++) pt[i] = i * 2;

    // // warm-up mem-data
    // pt = (uint64_t*)mem_data;
    // warmup_size = ver_data_mem_req / sizeof(uint64_t);
    // pt[0] = 1;
    // for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

    std::cout << "\tDelta size: "
              << ((double)(ver_list_mem_req + ver_data_mem_req) /
                  (1024 * 1024 * 1024))
              << " GB * " << num_partitions << " Partitions" << std::endl;

    this->rec_size = rec_size;

    // std::cout << "Rec size: " << rec_size << std::endl;

    // reserve hash-capacity before hand
    vid_version_map.reserve(total_tuple_capacity * num_partitions);
  }

  void* insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax, int pid) {
    // void* cnk = getVersionDataChunk();
    char* cnk = (char*)partitions[pid]->getVersionDataChunk();
    global_conf::mv_version* val = (global_conf::mv_version*)cnk;
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = cnk + sizeof(global_conf::mv_version);

    global_conf::mv_version_list* vlst = nullptr;

    if (vid_version_map.find(vid, vlst))
      vlst->insert(val);
    else {
      vlst = (global_conf::mv_version_list*)partitions[pid]->getListChunk();
      vlst->insert(val);
      vid_version_map.insert(vid, vlst);
    }

    return val->data;
  }

  bool getVersionList(uint64_t vid, global_conf::mv_version_list*& vlst) {
    if (vid_version_map.find(vid, vlst))
      return true;
    else
      assert(false);
    return false;
  }

  global_conf::mv_version_list* getVersionList(uint64_t vid) {
    global_conf::mv_version_list* vlst = nullptr;
    vid_version_map.find(vid, vlst);
    assert(vlst != nullptr);
    return vlst;
  }

  void reset() {
    vid_version_map.clear();
    for (auto& p : partitions) {
      p->reset();
    }
  }

 private:
  size_t rec_size;
  indexes::HashIndex<uint64_t, global_conf::mv_version_list*> vid_version_map;

  class DeltaPartition {
    mem_chunk ver_list_mem;
    mem_chunk ver_data_mem;
    std::atomic<char*> ver_list_cursor;
    std::atomic<char*> ver_data_cursor;

    size_t rec_size;
    int pid;

   public:
    DeltaPartition(char* ver_list_cursor, mem_chunk ver_list_mem,
                   char* ver_data_cursor, mem_chunk ver_data_mem,
                   size_t rec_size, int pid)
        : ver_list_mem(ver_list_mem),
          ver_data_mem(ver_data_mem),
          ver_list_cursor(ver_list_cursor),
          ver_data_cursor(ver_data_cursor),
          rec_size(rec_size),
          pid(pid) {
      // WARMUP MAYBE?

      // // warm-up mem-list
      std::cout << "\t warming up delta storage P" << pid << std::endl;
      uint64_t* pt = (uint64_t*)ver_list_cursor;
      int warmup_size = ver_list_mem.size / sizeof(uint64_t);
      pt[0] = 3;
      for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

      // warm-up mem-data
      pt = (uint64_t*)ver_data_cursor;
      warmup_size = ver_data_mem.size / sizeof(uint64_t);
      pt[0] = 1;
      for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;
    }

    ~DeltaPartition() {
      std::cout << "Clearing Delta Parition-" << pid << std::endl;
      // int munlock(const void *addr, size_t len);
      MemoryManager::free(ver_list_mem.data, ver_list_mem.size);
      MemoryManager::free(ver_data_mem.data, ver_data_mem.size);
    }

    void reset() {
      ver_list_cursor = (char*)ver_list_mem.data;
      ver_data_cursor = (char*)ver_data_mem.data;
    }
    void* getListChunk() {
      // void* tmp = nullptr;
      // tmp =
      // ver_list_cursor//.fetch_add(sizeof(global_conf::mv_version_list));
      // --
      // void* tmp = (void*)ver_list_cursor;
      // ver_list_cursor += sizeof(global_conf::mv_version_list);
      void* tmp = (void*)ver_list_cursor.fetch_add(
          sizeof(global_conf::mv_version_list));

      return tmp;
    }

    void* getVersionDataChunk() {
      // void* tmp = nullptr;
      // tmp = ver_data_cursor.fetch_add(rec_size +
      // sizeof(global_conf::mv_version));

      // void* tmp = (void*)ver_data_cursor;
      // ver_data_cursor += rec_size + sizeof(global_conf::mv_version);

      void* tmp = (void*)ver_list_cursor.fetch_add(
          rec_size + sizeof(global_conf::mv_version));
      return tmp;
    }

    friend class DeltaStore;
  };

  std::vector<DeltaPartition*> partitions;
};

};  // namespace storage

#endif /* DELTA_STORAGE_HPP_ */
