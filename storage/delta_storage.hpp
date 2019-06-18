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
#include <limits>
#include <mutex>
#include "glo.hpp"
//#include "indexes/hash_index.hpp"
#include "scheduler/worker.hpp"
#include "storage/memory_manager.hpp"
//#include "transactions/transaction_manager.hpp"

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
             uint64_t ver_data_capacity = DELTA_SIZE, int num_partitions = -1)
      : touched(false) {
    this->delta_id = delta_id;
    if (num_partitions == -1) {
      num_partitions = NUM_SOCKETS;
    }

    ver_list_capacity = ver_list_capacity * (1024 * 1024 * 1024);  // GB
    ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB
    for (int i = 0; i < num_partitions; i++) {
      uint list_numa_id = i % NUM_SOCKETS;
      uint data_numa_id = i % NUM_SOCKETS;

      // std::cout << "PID-" << i << " - memset: " << data_numa_id << std::endl;

      void* mem_list = MemoryManager::alloc(ver_list_capacity, list_numa_id);
      void* mem_data = MemoryManager::alloc(ver_data_capacity, data_numa_id);
      assert(mem_list != NULL);
      assert(mem_data != NULL);

      // assert(mlock(mem_list, ver_list_mem_req));
      // assert(mlock(mem_data, ver_data_mem_req));

      assert(mem_list != nullptr);
      assert(mem_data != nullptr);

      partitions.push_back(new DeltaPartition(
          (char*)mem_list, mem_chunk(mem_list, ver_list_capacity, list_numa_id),
          (char*)mem_data, mem_chunk(mem_data, ver_data_capacity, data_numa_id),
          i));
    }

    if (DELTA_DEBUG) {
      std::cout << "\tDelta size: "
                << ((double)(ver_list_capacity + ver_data_capacity) /
                    (1024 * 1024 * 1024))
                << " GB * " << num_partitions << " Partitions" << std::endl;
      std::cout << "\tDelta size: "
                << ((double)(ver_list_capacity + ver_data_capacity) *
                    num_partitions / (1024 * 1024 * 1024))
                << " GB" << std::endl;
    }
    this->total_mem_reserved =
        (ver_list_capacity + ver_data_capacity) * num_partitions;

    for (int i = 0; i < MAX_WORKERS; i++) {
      read_ctr[i] = 0;
    }

    // reserve hash-capacity before hand
    vid_version_map.reserve(10000000);
    this->readers.store(0);
    this->gc_reset_success.store(0);
    this->gc_requests.store(0);
    this->ops.store(0);
    this->gc_lock.store(0);
    this->tag = 0;
    this->max_active_epoch = 0;
    // this->min_active_epoch = std::numeric_limits<uint64_t>::max();
  }

  void print_info() {
    static int i = 0;
    std::cout << "[DeltaStore # " << i
              << "] Number of GC Requests: " << this->gc_requests.load()
              << std::endl;

    std::cout << "[DeltaStore # " << i << "] Number of Successful GC Resets: "
              << this->gc_reset_success.load() << std::endl;
    std::cout << "[DeltaStore # " << i
              << "] Number of Operations: " << this->ops.load() << std::endl;

    for (auto& p : partitions) {
      p->report();
    }
    i++;
    if (i >= partitions.size()) i = 0;
  }
  ~DeltaStore() { print_info(); }

  void* insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax,
                       ushort rec_size, int parition_id) {
    // void* cnk = getVersionDataChunk();

    // while (gc_lock != 0) dont need a gc lock, if someone is here means the
    // read_counter is already +1 so never gonna gc
    // std::cout << "--" << parition_id << "--" << std::endl;
    char* cnk = (char*)partitions[parition_id]->getVersionDataChunk(rec_size);
    global_conf::mv_version* val = (global_conf::mv_version*)cnk;
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = cnk + sizeof(global_conf::mv_version);

    // global_conf::mv_version_list* vlst = nullptr;

    std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);

    if (vid_version_map.find(vid, v_pair)) {
      if (v_pair.first == this->tag) {
        // valid list
        v_pair.second->insert(val);
      } else {
        // invalid list
        // int tmp = v_pair.first;
        v_pair.first = tag;
        v_pair.second = (global_conf::mv_version_list*)partitions[parition_id]
                            ->getListChunk();
        v_pair.second->insert(val);
        vid_version_map.update(vid, v_pair);
      }

    } else {
      // new record overall
      v_pair.first = tag;
      v_pair.second = (global_conf::mv_version_list*)partitions[parition_id]
                          ->getListChunk();
      v_pair.second->insert(val);
      vid_version_map.insert(vid, v_pair);
    }
    // ops++;
    if (!touched) touched = true;
    return val->data;
  }

  bool getVersionList(uint64_t vid, global_conf::mv_version_list*& vlst) {
    std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);
    if (vid_version_map.find(vid, v_pair)) {
      if (v_pair.first == tag) {
        vlst = v_pair.second;
        return true;
      }
    }
    assert(false);

    return false;

    // if (vid_version_map.find(vid, vlst))
    //   return true;
    // else
    //   assert(false);
    // return false;
  }

  global_conf::mv_version_list* getVersionList(uint64_t vid) {
    std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);
    vid_version_map.find(vid, v_pair);

    // if (v_pair.first != tag) {
    //   std::cout << "first: " << v_pair.first << std::endl;
    //   std::cout << "tag: " << tag << std::endl;
    // }

    assert(v_pair.first == tag);
    return v_pair.second;

    // global_conf::mv_version_list* vlst = nullptr;
    // vid_version_map.find(vid, vlst);
    // assert(vlst != nullptr);
    // return vlst;
  }

  // void reset() {
  //   vid_version_map.clear();
  //   for (auto& p : partitions) {
  //     p->reset();
  //   }
  // }

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

  void gc() {
    // std::cout << "." << std::endl;
    short e = 0;
    if (gc_lock.compare_exchange_strong(e, -1)) {
      gc_requests++;

      uint64_t last_alive_txn =
          scheduler::WorkerPool::getInstance().get_min_active_txn();

      // missing condition: or space > 90%
      if (this->readers == 0 && should_gc() &&
          last_alive_txn > max_active_epoch) {
        // std::cout << "delta_id#: " << delta_id << std::endl;
        // std::cout << "request#: " << gc_requests << std::endl;
        // std::cout << "last_alive_txn: " << last_alive_txn << std::endl;
        // std::cout << "max_active_epoch: " << max_active_epoch << std::endl;

        // std::chrono::time_point<std::chrono::system_clock,
        //                         std::chrono::nanoseconds>
        //     start_time;

        // vid_version_map.clear();

        // std::chrono::duration<double> diff =
        //     std::chrono::system_clock::now() - start_time;

        // std::cout << "version clear time: " << diff.count() << std::endl;
        for (auto& p : partitions) {
          p->reset();
        }
        tag++;

        if (tag % VER_CLEAR_ITER == 0) {
          vid_version_map.clear();
        }
        // gc_lock.unlock();
        gc_lock.store(0);
        touched = false;
        gc_reset_success++;
      } else {
        // gc_lock.unlock();
        gc_lock.store(0);
      }
    }
  }

  void gc_with_counter_arr(int wrk_id) {
    // optimization: start with your own socket and then look for readers on
    // other scoket. second optimization, keep a read counter per partition but
    // atomic/volatile maybe.

    short e = 0;
    gc_requests++;

    if (gc_lock.compare_exchange_strong(e, -1)) {
      bool go = true;

// for (int i = 0; i < MAX_WORKERS / 8; i += 8) {
//   uint64_t* t = (uint64_t*)(read_ctr + (i * 8));
//   if (*t != 0) {
//     go = false;
//     // break;
//   }
// }
#pragma clang loop vectorize(enable)
      for (int i = 0; i < MAX_WORKERS; i++) {
        if (read_ctr[i] != 0) {
          go = false;
          // break;
        }
      }
      uint64_t last_alive_txn =
          scheduler::WorkerPool::getInstance().get_min_active_txn();
      if (go && last_alive_txn > max_active_epoch) {
        // vid_version_map.clear();
        tag += 1;
        if (tag % VER_CLEAR_ITER == 0) {
          vid_version_map.clear();
        }
        for (auto& p : partitions) {
          p->reset();
        }
        gc_reset_success++;
      }
      gc_lock.store(0);
    }
  }

  /*
std::chrono::time_point<std::chrono::system_clock,
                                std::chrono::nanoseconds>
            start_time;

        vid_version_map.clear();

        std::chrono::duration<double> diff =
            std::chrono::system_clock::now() - start_time;

        std::cout << "version clear time: " << diff.count() << std::endl;

  */

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

    double usage() {
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

#endif /* DELTA_STORAGE_HPP_ */
