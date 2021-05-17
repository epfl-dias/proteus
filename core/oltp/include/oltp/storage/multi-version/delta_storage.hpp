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
#include <platform/memory/memory-manager.hpp>

#include "oltp/common/common.hpp"
#include "oltp/common/memory-chunk.hpp"

#define DELTA_DEBUG 1
#define GC_CAPACITY_MIN_PERCENT 75

namespace storage {

/* Currently DeltaStore is not resizeable*/

class DeltaList;
class DeltaDataPtr;

class alignas(4096) DeltaStore {
 public:
  DeltaStore(delta_id_t delta_id, uint64_t ver_list_capacity = 4,
             uint64_t ver_data_capacity = 4, partition_id_t num_partitions = 1);
  ~DeltaStore();

  void print_info();
  void *insert_version(DeltaDataPtr &delta_list, xid_t t_min, xid_t t_max,
                       size_t rec_size, partition_id_t partition_id);
  //  void *validate_or_create_list(void *list_ptr, size_t &delta_ver_tag,
  //                                ushort partition_id);
  void *validate_or_create_list(DeltaList &delta_list,
                                partition_id_t partition_id);
  void *getTransientChunk(DeltaList &delta_list, size_t size,
                          partition_id_t partition_id);
  void *create_version(size_t size, partition_id_t partition_id);
  void gc();
  // void gc_with_counter_arr(int wrk_id);

  inline bool should_gc() {
#if (GC_CAPACITY_MIN_PERCENT > 0)
    for (auto &p : partitions) {
      // std::cout << "usage: " << p->usage() << std::endl;
      if (p->usage() > ((double)GC_CAPACITY_MIN_PERCENT) / 100) {
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
  increment_reader(uint64_t epoch, worker_id_t worker_id) {
    while (gc_lock < 0 && !should_gc())
      ;

    if (max_active_epoch < epoch) {
      max_active_epoch = epoch;
    }
    this->readers++;
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    if (readers.fetch_sub(1) <= 1 && touched) {
      gc();
    }
  }

 private:
  class alignas(256) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    const delta_id_t delta_id;
    const partition_id_t pid;
    const uint16_t delta_uuid;
    const oltp::common::mem_chunk ver_list_mem;
    const oltp::common::mem_chunk ver_data_mem;
    bool touched;
    std::mutex print_lock;
    bool printed;
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

    inline uint16_t create_delta_uuid(delta_id_t delta_id, partition_id_t pid) {
      uint16_t tmp = 0;
      tmp = (tmp | delta_id) << 8u;
      tmp |= pid;
      return tmp;
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

  std::atomic<uint32_t> tag{};
  xid_t max_active_epoch;

  std::vector<DeltaPartition *> partitions{};
  std::atomic<uint> readers{};
  std::atomic<short> gc_lock{};
  bool touched;
  std::atomic<uint> gc_reset_success{};
  std::atomic<uint> gc_requests{};

 public:
  const delta_id_t delta_id;
  uint64_t total_mem_reserved;

  friend class DeltaMemoryPtr;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
