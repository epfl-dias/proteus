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
#include <forward_list>
#include <iostream>
#include <limits>
#include <mutex>
#include <platform/memory/memory-manager.hpp>

#include "oltp/common/common.hpp"
#include "oltp/common/memory-chunk.hpp"

#define DELTA_DEBUG 1
#define GC_CAPACITY_MIN_PERCENT 0

namespace storage {

/* Currently DeltaStore is not resizeable*/

class DeltaList;
class DeltaDataPtr;

class alignas(4096) DeltaStore {
 public:
  explicit DeltaStore(delta_id_t delta_id, uint64_t ver_list_capacity = 4,
                      uint64_t ver_data_capacity = 4,
                      partition_id_t num_partitions = 1);
  ~DeltaStore();

  void print_info();
  // for single-list
  void *insert_version(DeltaDataPtr &delta_list, xid_t t_min, xid_t t_max,
                       size_t rec_size, partition_id_t partition_id);
  // for list-per-attribute
  std::pair<void *, void *> insert_version(DeltaList &delta_list, xid_t t_min,
                                           xid_t t_max, size_t rec_size,
                                           partition_id_t partition_id);
  //  void *validate_or_create_list(void *list_ptr, size_t &delta_ver_tag,
  //                                ushort partition_id);
  void *validate_or_create_list(DeltaList &delta_list,
                                partition_id_t partition_id);
  void *getTransientChunk(DeltaList &delta_list, size_t size,
                          partition_id_t partition_id);
  void *create_version(size_t size, partition_id_t partition_id);
  void gc();

  inline bool should_gc() {
#if (GC_CAPACITY_MIN_PERCENT > 0)
    for (auto &p : partitions) {
      // LOG(INFO) << "usage: " << p->usage() ;
      if (p->usage() > ((double)GC_CAPACITY_MIN_PERCENT) / 100) {
        // LOG(INFO) << "usage: " << p->usage()*100;
        return true;
      }
    }
    return false;
#else
    return true;
#endif
  }

  inline void update_active_epoch(xid_t epoch, worker_id_t worker_id) {
    //    xid_t e = max_active_epoch;
    //    while(max_active_epoch < epoch &&
    //    !(max_active_epoch.compare_exchange_weak(e, epoch,
    //    std::memory_order_relaxed))){
    //      e = max_active_epoch;
    //    }

    while (deltaMeta.max_active_epoch < epoch) {
      deltaMeta.max_active_epoch = epoch;
    }
  }

  inline void __attribute__((always_inline))
  increment_reader(xid_t epoch, worker_id_t worker_id) {
    while (deltaMeta.readers < 0)
      ;

    deltaMeta.readers++;

    while (deltaMeta.max_active_epoch < epoch) {
      deltaMeta.max_active_epoch = epoch;
    }
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    if (deltaMeta.readers.fetch_sub(1, std::memory_order_relaxed) == 1 &&
        touched) {
      gc();
    }
  }

 private:
  class alignas(hardware_destructive_interference_size) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    const delta_id_t delta_id;
    const partition_id_t pid;
    const uint16_t delta_uuid;
    const oltp::common::mem_chunk ver_list_mem;
    const oltp::common::mem_chunk ver_data_mem;
    bool touched;
    const char *list_cursor_max;
    const char *data_cursor_max;

    std::vector<bool> reset_listeners;

    class threadLocalSlack {
     public:
      char *ptr{};
      int remaining_slack{};
      threadLocalSlack() : ptr(nullptr), remaining_slack(0) {}
    };

    static_assert(sizeof(delta_id_t) == 1);
    static_assert(sizeof(partition_id_t) == 1);

   public:
    DeltaPartition(char *ver_list_cursor, oltp::common::mem_chunk ver_list_mem,
                   char *ver_data_cursor, oltp::common::mem_chunk ver_data_mem,
                   partition_id_t pid, delta_id_t delta_id);

    static inline uint16_t create_delta_uuid(delta_id_t d_id,
                                             partition_id_t partitionId) {
      uint16_t tmp = 0;
      tmp = (tmp | d_id) << 8u;
      tmp |= partitionId;
      return tmp;
    }
    static inline auto get_delta_id(uint16_t delta_uuid) {
      return (size_t)(delta_uuid >> 8u);
    }
    static inline auto get_delta_pid(uint16_t delta_uuid) {
      return (size_t)(delta_uuid & 0x00FF);
    }

    ~DeltaPartition() {
      if (DELTA_DEBUG) {
        LOG(INFO) << "Clearing DeltaPartition-" << (int)pid;
      }
      MemoryManager::freePinned(ver_list_mem.data);
      MemoryManager::freePinned(ver_data_mem.data);
    }

    void reset() {
      if (__builtin_expect(touched, 1)) {
        ver_list_cursor = (char *)ver_list_mem.data;
        ver_data_cursor = (char *)ver_data_mem.data;

        for (auto &&reset_listener : reset_listeners) {
          reset_listener = true;
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      touched = false;
    }
    void *getListChunk();

    void *getChunk(size_t size);

    void *getVersionDataChunk(size_t rec_size);
    void *getVersionDataChunk_ThreadLocal(size_t rec_size);

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

 public:
  const delta_id_t delta_id;

 private:
  bool touched;
  std::atomic<uint32_t> tag{};
  std::vector<DeltaPartition *> partitions{};

  struct meta {
    volatile xid_t max_active_epoch;
    std::atomic<int64_t> readers{};
  };

  alignas(hardware_destructive_interference_size) meta deltaMeta;

  volatile uint gc_reset_success{};
  volatile uint gc_requests{};

 public:
  uint64_t total_mem_reserved;

  friend class DeltaMemoryPtr;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
