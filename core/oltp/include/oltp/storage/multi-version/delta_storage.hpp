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

class DeltaDataPtr;
class ClassicPtrWrapper;
class CircularDeltaStore;
class DeltaStoreMalloc;

using DeltaStore = std::conditional<GcMechanism == GcTypes::OneShot,
                                    CircularDeltaStore, DeltaStoreMalloc>::type;

class DeltaStoreMalloc {
 public:
  explicit DeltaStoreMalloc(delta_id_t delta_id, uint64_t ver_data_capacity = 4,
                            partition_id_t num_partitions = 1);

  ~DeltaStoreMalloc();

  ClassicPtrWrapper allocate(size_t sz, partition_id_t partition_id);
  void release(ClassicPtrWrapper &ptr);

  inline void update_active_epoch(xid_t epoch, worker_id_t worker_id) {
    return;
  }

  inline void increment_reader(xid_t epoch, worker_id_t worker_id) { return; }

  inline void decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    return;
  }

 public:
  const uint64_t total_mem_reserved{};
  std::mutex lk;

 public:
  static ClassicPtrWrapper ptrType;
};

class alignas(4096) CircularDeltaStore {
 public:
  explicit CircularDeltaStore(delta_id_t delta_id,
                              uint64_t ver_data_capacity = 4,
                              partition_id_t num_partitions = 1);
  ~CircularDeltaStore();

  DeltaDataPtr allocate(size_t sz, partition_id_t partition_id);
  void release(DeltaDataPtr &ptr);

  inline void update_active_epoch(xid_t epoch, worker_id_t worker_id) {
    //    xid_t e = max_active_epoch;
    //    while(max_active_epoch < epoch &&
    //    !(max_active_epoch.compare_exchange_weak(e, epoch,
    //    std::memory_order_relaxed))){
    //      e = max_active_epoch;
    //    }
    assert(deltaMeta.readers > 0);
    while (deltaMeta.max_active_epoch < epoch) {
      deltaMeta.max_active_epoch = epoch;
    }
  }

  inline void __attribute__((always_inline))
  increment_reader(xid_t epoch, worker_id_t worker_id) {
    while (deltaMeta.readers < 0)
      ;

    auto x = deltaMeta.readers++;

    // safety-check
    while (x < 0) {
      while (deltaMeta.readers < 0)
        ;
      x = deltaMeta.readers++;
    }

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

  static DeltaDataPtr ptrType;

 private:
  void print_info();
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

 private:
  class alignas(hardware_destructive_interference_size) DeltaPartition {
    std::atomic<uintptr_t> ver_data_cursor;
    const delta_id_t delta_id;
    const partition_id_t pid;
    const uint16_t delta_uuid;
    const oltp::common::mem_chunk ver_data_mem;
    bool touched;
    const uintptr_t data_cursor_max;

    std::vector<bool> reset_listeners;

    class threadLocalSlack {
     public:
      uintptr_t ptr{};
      int remaining_slack{};
      threadLocalSlack()
          : ptr(reinterpret_cast<uintptr_t>(nullptr)), remaining_slack(0) {}
    };

    static_assert(sizeof(delta_id_t) == 1);
    static_assert(sizeof(partition_id_t) == 1);

   public:
    DeltaPartition(uintptr_t ver_data_cursor,
                   oltp::common::mem_chunk ver_data_mem, partition_id_t pid,
                   delta_id_t delta_id);

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
      MemoryManager::freePinned(ver_data_mem.data);
    }

    void reset() {
      if (__builtin_expect(touched, 1)) {
        ver_data_cursor = reinterpret_cast<uintptr_t>(ver_data_mem.data);

        for (auto &&reset_listener : reset_listeners) {
          reset_listener = true;
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      touched = false;
    }

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

    friend class CircularDeltaStore;
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

  friend class DeltaDataPtr;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
