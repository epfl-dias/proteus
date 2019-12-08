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
#define GC_CAPCITY_MIN_PERCENT 75

namespace storage {

/* Currently DeltaStore is not resizeable*/

class alignas(4096) DeltaStore {
 public:
  DeltaStore(uint delta_id, uint64_t ver_list_capacity = g_delta_size,
             uint64_t ver_data_capacity = g_delta_size,
             int num_partitions = g_num_partitions);
  ~DeltaStore();

  void print_info();
  void *insert_version(global_conf::IndexVal *idx_ptr, uint rec_size,
                       ushort parition_id);
  void gc();
  void gc_with_counter_arr(int wrk_id);

  inline bool should_gc() {
#if (GC_CAPCITY_MIN_PERCENT > 0)
    for (auto &p : partitions) {
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
    while (gc_lock < 0 && !should_gc())
      ;

    if (max_active_epoch < epoch) {
      max_active_epoch = epoch;
    }
    this->readers++;
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, uint8_t worker_id) {
    if (readers.fetch_sub(1) <= 1 && touched) {
      gc();
    }
  }

 private:
  class alignas(4096) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    int pid;
    mem_chunk ver_list_mem;
    mem_chunk ver_data_mem;
    bool touched;
    std::mutex print_lock;
    bool printed;
    const char *list_cursor_max;
    const char *data_cursor_max;

    // std::vector<bool> reset_listeners;
    bool reset_listeners[NUM_CORE_PER_SOCKET];

   public:
    DeltaPartition(char *ver_list_cursor, mem_chunk ver_list_mem,
                   char *ver_data_cursor, mem_chunk ver_data_mem, int pid)
        : ver_list_mem(ver_list_mem),
          ver_data_mem(ver_data_mem),
          ver_list_cursor(ver_list_cursor),
          ver_data_cursor(ver_data_cursor),
          list_cursor_max(ver_list_cursor + ver_list_mem.size),
          data_cursor_max(ver_list_cursor + ver_data_mem.size),
          touched(false),
          pid(pid) {
      printed = false;
      // warm-up mem-list
      if (DELTA_DEBUG)
        std::cout << "\t warming up delta storage P" << pid << std::endl;

      uint64_t *pt = (uint64_t *)ver_list_cursor;
      int warmup_size = ver_list_mem.size / sizeof(uint64_t);
      pt[0] = 3;
      for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

      // warm-up mem-data
      pt = (uint64_t *)ver_data_cursor;
      warmup_size = ver_data_mem.size / sizeof(uint64_t);
      pt[0] = 1;
      for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

      // reset_listeners.reserve(NUM_CORE_PER_SOCKET);
      for (int i = 0; i < NUM_CORE_PER_SOCKET; i++) {
        // reset_listeners.push_back(false);
        reset_listeners[i] = false;
      }
    }

    ~DeltaPartition() {
      if (DELTA_DEBUG)
        std::cout << "Clearing Delta Parition-" << pid << std::endl;
      // MemoryManager::free(ver_list_mem.data, ver_list_mem.size);
      // MemoryManager::free(ver_data_mem.data, ver_data_mem.size);
    }

    void reset() {
      if (__likely(touched)) {
        ver_list_cursor = (char *)ver_list_mem.data;
        ver_data_cursor = (char *)ver_data_mem.data;

        for (uint i = 0; i < NUM_CORE_PER_SOCKET; i++) {
          reset_listeners[i] = true;
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      touched = false;
    }
    inline void *getListChunk() {
      char *tmp = ver_list_cursor.fetch_add(
          sizeof(global_conf::mv_version_list), std::memory_order_relaxed);

      assert((tmp + sizeof(global_conf::mv_version_list)) <= list_cursor_max);
      touched = true;
      return tmp;
    }

    inline void *getVersionDataChunk(size_t rec_size) {
      constexpr uint slack_size = 8192;

      static thread_local uint remaining_slack = 0;
      static thread_local char *ptr = nullptr;
      static int thread_counter = 0;

      static thread_local uint tid = thread_counter++;

      size_t req = rec_size + sizeof(global_conf::mv_version);

      // works.
      if (reset_listeners[tid] == true) {
        remaining_slack = 0;
        reset_listeners[tid] = false;
      }

      if (__unlikely(req > remaining_slack)) {
        ptr = ver_data_cursor.fetch_add(slack_size, std::memory_order_relaxed);
        remaining_slack = slack_size;

        if (__unlikely((ptr + remaining_slack) > data_cursor_max)) {
          // FIXME: if delta-storage is full, there should be a manual trigger
          // to initiate a detailed/granular GC algorithm, not just crash the
          // engine.

          std::unique_lock<std::mutex> lk(print_lock);
          if (!printed) {
            printed = true;
            std::cout << "#######" << std::endl;
            std::cout << "PID: " << pid << std::endl;
            report();
            std::cout << "#######" << std::endl;
            assert(false);
          }
        }
      }

      char *tmp = ptr;
      ptr += req;
      remaining_slack -= req;

      // char *tmp = ver_data_cursor.fetch_add(req, std::memory_order_relaxed);

      touched = true;
      return tmp;
    }

    inline double usage() {
      return ((double)(((char *)ver_data_cursor.load() -
                        (char *)this->ver_data_mem.data))) /
             this->ver_data_mem.size;
    }

    void report() {
      /* data memory only */
      char *curr_ptr = (char *)ver_data_cursor.load();
      char *start_ptr = (char *)ver_data_mem.data;

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

  std::atomic<uint> tag;
  uint64_t max_active_epoch;
  uint delta_id;
  std::vector<DeltaPartition *> partitions;
  std::atomic<uint> readers;
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
