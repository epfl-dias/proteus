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
  DeltaStore(uint32_t delta_id, uint64_t ver_list_capacity = g_delta_size,
             uint64_t ver_data_capacity = g_delta_size,
             int num_partitions = g_num_partitions);
  ~DeltaStore();

  void print_info();
  void *insert_version(global_conf::IndexVal *idx_ptr, uint rec_size,
                       ushort parition_id);
  void *validate_or_create_list(void *list_ptr, size_t &delta_ver_tag,
                                ushort partition_id);
  void *create_version(size_t size, ushort partition_id);
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

  inline auto getTag() { return tag.load(); }

  [[maybe_unused]] inline auto getFullTag() {
    return create_delta_tag(this->delta_id, tag.load());
  }

  [[maybe_unused]] inline bool verifyTag(uint64_t &d_tag_ver) {
    return (create_delta_tag(this->delta_id, tag.load()) == d_tag_ver);
  }

  static inline uint64_t __attribute__((always_inline))
  create_delta_tag(uint64_t delta_idx, uint32_t delta_tag) {
    // 2 byte delta_idx | 4-byte delta-tag
    return (delta_idx << 32) | (delta_tag);
  }

 private:
  class alignas(4096) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    int pid;
    const storage::memory::mem_chunk ver_list_mem;
    const storage::memory::mem_chunk ver_data_mem;
    bool touched;
    std::mutex print_lock;
    bool printed;
    const char *list_cursor_max;
    const char *data_cursor_max;

    std::vector<bool> reset_listeners;

   public:
    DeltaPartition(char *ver_list_cursor,
                   storage::memory::mem_chunk ver_list_mem,
                   char *ver_data_cursor,
                   storage::memory::mem_chunk ver_data_mem, int pid);

    ~DeltaPartition() {
      if (DELTA_DEBUG) {
        LOG(INFO) << "Clearing DeltaPartition-" << pid;
      }
      storage::memory::MemoryManager::free(ver_list_mem.data);
      storage::memory::MemoryManager::free(ver_data_mem.data);
    }

    void reset() {
      if (__likely(touched)) {
        ver_list_cursor = (char *)ver_list_mem.data;
        ver_data_cursor = (char *)ver_data_mem.data;

        for (uint i = 0; i < reset_listeners.size(); i++) {
          reset_listeners[i] = true;
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      touched = false;
    }
    void *getListChunk();

    void *getVersionDataChunk(size_t rec_size);

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

  std::atomic<size_t> tag;
  uint64_t max_active_epoch;
  uint32_t delta_id;
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
