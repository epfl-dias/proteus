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

#ifndef STORAGE_DELTA_STORAGE_HPP_
#define STORAGE_DELTA_STORAGE_HPP_

#include <sys/mman.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>

#include "glo.hpp"
#include "storage/memory_manager.hpp"

#define DELTA_DEBUG 1
#define GC_ATOMIC 1
#define GC_CAPCITY_MIN_PERCENT 75
#define VER_CLEAR_ITER 25000

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

/*
Optimization: for hash table zeroing, put a tag inside the list so that we know
that if this list is the valid version or not.
*/
class DeltaStore {
 public:
  DeltaStore(uint delta_id, uint64_t ver_list_capacity = DELTA_SIZE,
             uint64_t ver_data_capacity = DELTA_SIZE, int num_partitions = -1);
  ~DeltaStore();

  void print_info();

  void* insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax,
                       ushort rec_size, int parition_id);

  bool getVersionList(uint64_t vid, global_conf::mv_version_list*& vlst);

  global_conf::mv_version_list* getVersionList(uint64_t vid);

  void gc();
  void gc_with_counter_arr(int wrk_id);

  inline bool should_gc() {
#if (GC_CAPCITY_MIN_PERCENT > 0)
    for (auto& p : partitions) {
      // std::cout << "usage: " << p->usage() << std::endl;
      if (p->usage() > ((double)GC_CAPCITY_MIN_PERCENT) / 100) {
        // std::cout << "usage: " << p->usage() << std::endl;
        return true;
      }
    }
    return false;
#else
    return true;
#endif
  }

  inline void __attribute__((always_inline))
  increment_reader(uint64_t epoch, uint8_t worker_id) {
    // for (int tries = 0; true; ++tries) {
    //   if (gc_lock == 0) break;
    //   if (tries == 1000) {
    //     tries = 0;
    //     sched_yield();
    //   }

    // }

    // FIXME: add a condition: if the usage of paritions is more than 80%, dont
    // admit new workers until we clear everything to enfore cleaning.
    while (gc_lock < 0 && !should_gc())
      ;

    if (max_active_epoch < epoch) {
      max_active_epoch = epoch;
    }

#if GC_ATOMIC
    this->readers++;
#else
    read_ctr[worker_id] = 1;
#endif
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, uint8_t worker_id) {
#if GC_ATOMIC
    // this->readers--;
    if (readers.fetch_sub(1) <= 1 && touched) {
      gc();
      ;
    }
#else
    read_ctr[id] = 0;
    gc_with_counter_arr(worker_id);
#endif
  }

 private:
  // indexes::HashIndex<uint64_t, global_conf::mv_version_list*>
  // vid_version_map;
  char read_ctr[MAX_WORKERS];

  // hash table optimizations

  std::atomic<uint> tag;
  indexes::HashIndex<uint64_t, std::pair<int, global_conf::mv_version_list*>>
      vid_version_map;

  uint64_t max_active_epoch;
  uint delta_id;

  class DeltaPartition {
    mem_chunk ver_list_mem;
    mem_chunk ver_data_mem;
    std::atomic<char*> ver_list_cursor;
    std::atomic<char*> ver_data_cursor;
    bool touched;

    int pid;

   public:
    DeltaPartition(char* ver_list_cursor, mem_chunk ver_list_mem,
                   char* ver_data_cursor, mem_chunk ver_data_mem, int pid)
        : ver_list_mem(ver_list_mem),
          ver_data_mem(ver_data_mem),
          ver_list_cursor(ver_list_cursor),
          ver_data_cursor(ver_data_cursor),
          touched(false),
          pid(pid) {
      // warm-up mem-list
      if (DELTA_DEBUG)
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
      if (DELTA_DEBUG)
        std::cout << "Clearing Delta Parition-" << pid << std::endl;
      // int munlock(const void *addr, size_t len);
      // MemoryManager::free(ver_list_mem.data, ver_list_mem.size);
      // MemoryManager::free(ver_data_mem.data, ver_data_mem.size);
    }

    void reset() {
      if (touched) {
        ver_list_cursor = (char*)ver_list_mem.data;
        ver_data_cursor = (char*)ver_data_mem.data;
      }
      touched = false;
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

      assert((((int*)tmp - (int*)this->ver_list_mem.data) * sizeof(int)) <=
             this->ver_list_mem.size);
      touched = true;
      return tmp;
    }

    void* getVersionDataChunk(size_t rec_size) {
      // void* tmp = nullptr;
      // tmp = ver_data_cursor.fetch_add(rec_size +
      // sizeof(global_conf::mv_version));

      // void* tmp = (void*)ver_data_cursor;
      // ver_data_cursor += rec_size + sizeof(global_conf::mv_version);
      void* tmp = (void*)ver_data_cursor.fetch_add(
          rec_size + sizeof(global_conf::mv_version));

      assert((((int*)tmp - (int*)this->ver_data_mem.data) * sizeof(int)) <=
             this->ver_data_mem.size);
      touched = true;
      return tmp;
    }

    inline double usage() {
      return ((double)(((char*)ver_data_cursor.load() -
                        (char*)this->ver_data_mem.data))) /
             this->ver_data_mem.size;
    }

    void report() {
      /* data memory only */
      char* curr_ptr = (char*)ver_data_cursor.load();
      char* start_ptr = (char*)ver_data_mem.data;

      auto diff = curr_ptr - start_ptr;
      double percent = ((double)diff / ver_data_mem.size) * 100;
      std::cout.precision(2);
      std::cout << "\tMemory Consumption: "
                << ((double)diff) / (1024 * 1024 * 1024) << "GB/"
                << (double)ver_data_mem.size / (1024 * 1024 * 1024)
                << "GB  --  " << percent << "%%" << std::endl;
    }

    friend class DeltaStore;
  };

  std::vector<DeltaPartition*> partitions;
  std::atomic<uint> readers;
  // std::mutex gc_lock;

  std::atomic<short> gc_lock;
  bool touched;

  std::atomic<uint> gc_reset_success;
  std::atomic<uint> gc_requests;
  std::atomic<uint> ops;

 public:
  uint64_t total_mem_reserved;
};

};  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
