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
#include <libcuckoo/cuckoohash_map.hh>
#include <limits>
#include <mutex>
#include <platform/memory/memory-manager.hpp>
#include <thread>

#include "oltp/common/common.hpp"
#include "oltp/common/memory-chunk.hpp"
#include "oltp/storage/memory-pool.hpp"

#define DELTA_DEBUG 1
#define GC_CAPACITY_MIN_PERCENT 0

namespace storage {

class DeltaDataPtr;
class ClassicPtrWrapper;
class CircularDeltaStore;
class DeltaStoreMalloc;

using DeltaStore = std::conditional<GcMechanism == GcTypes::OneShot ||
                                        GcMechanism == GcTypes::NoGC,
                                    CircularDeltaStore, DeltaStoreMalloc>::type;

class DeltaStoreMalloc {
  constexpr static size_t pool_chunk_sz = 832;  // 100;  // tpcc 832
  constexpr static bool memory_pool_per_part = false;

 public:
  const size_t total_mem_reserved{};
  static ClassicPtrWrapper ptrType;

 public:
  explicit DeltaStoreMalloc(delta_id_t delta_id, uint64_t ver_data_capacity = 4,
                            partition_id_t num_partitions = 1);

  ~DeltaStoreMalloc();

  ClassicPtrWrapper allocate(size_t sz, partition_id_t partition_id);
  ClassicPtrWrapper allocate(size_t sz, partition_id_t partition_id,
                             xid_t version_ts);

  void release(ClassicPtrWrapper &ptr);

  inline void update_active_epoch(xid_t epoch, worker_id_t worker_id) {
    return;
  }

  inline void increment_reader(xid_t epoch, worker_id_t worker_id) { return; }
  inline void decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    return;
  }

  inline void __attribute__((always_inline)) try_gc(xid_t xid) { return; }
  void GC() { return; }

  void initThreadLocalPools(partition_id_t partition_id);

 private:
  std::unordered_map<
      std::thread::id,
      oltp::mv::memorypool::BucketMemoryPool_threadLocal<pool_chunk_sz>>
      _memPools;

  std::unordered_map<
      partition_id_t,
      oltp::mv::memorypool::BucketMemoryPool_threadLocal<pool_chunk_sz>>
      _memPoolsPart;
};

class alignas(4096) CircularDeltaStore {
 public:
  explicit CircularDeltaStore(delta_id_t delta_id, double ver_data_capacity = 4,
                              partition_id_t num_partitions = 1);
  ~CircularDeltaStore();

  DeltaDataPtr allocate(size_t sz, partition_id_t partition_id);
  DeltaDataPtr allocate(size_t sz, partition_id_t partition_id,
                        xid_t version_ts);

  void release(DeltaDataPtr &ptr);

  void saveInstanceCrossingPtr(row_uuid_t row_uuid, DeltaDataPtr ptr);
  void initThreadLocalPools(partition_id_t partition_id);

  inline void update_active_epoch(xid_t epoch, worker_id_t worker_id) {
    getMeta_threadLocal()->max_active_epoch = epoch;
  }

  inline void __attribute__((always_inline))
  increment_reader(xid_t epoch, worker_id_t worker_id) {
    auto x = readers++;

    // safety-check
    while (x < 0) {
      while (readers < 0)
        ;
      x = readers++;
    }

    while (getMeta_threadLocal()->max_active_epoch < epoch) {
      getMeta_threadLocal()->max_active_epoch = epoch;
    }
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    if (readers.fetch_sub(1, std::memory_order_relaxed) == 1) {
      GC();
    }
  }

  void GC();

  const auto &getConsolidateHT(partition_id_t pid) {
    return this->partitions[pid]->consolidateHT;
  }

  static DeltaDataPtr ptrType;

 private:
  void print_info();

  inline bool impl_gc_withConsolidation();
  inline bool impl_gc_simple();

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
    static_assert(sizeof(delta_id_t) == 1);
    static_assert(sizeof(partition_id_t) == 1);

    class alignas(64) DeltaSlackCache {
     public:
      static constexpr uint slack_size = 8192;
      uintptr_t ptr{};
      int remaining_slack{};
      DeltaSlackCache()
          : ptr(reinterpret_cast<uintptr_t>(nullptr)), remaining_slack(0) {}

      DeltaSlackCache(DeltaSlackCache &&) = delete;
      DeltaSlackCache &operator=(DeltaSlackCache &&) = delete;
      DeltaSlackCache(const DeltaSlackCache &) = delete;
      DeltaSlackCache &operator=(const DeltaSlackCache &) = delete;
    };

    std::unordered_map<std::thread::id, DeltaSlackCache,
                       std::hash<std::thread::id>, std::equal_to<>,
                       proteus::memory::PinnedMemoryAllocator<
                           std::pair<const std::thread::id, DeltaSlackCache>>>
        _threadLocal_cache;

    std::atomic<uintptr_t> ver_data_cursor;
    [[maybe_unused]] const delta_id_t delta_id;
    const partition_id_t pid;
    const uint16_t delta_uuid;
    const oltp::common::mem_chunk ver_data_mem;
    bool touched;
    const uintptr_t data_cursor_max;

    // FIXME: change it to explicit socket allocator as cuckoo pre-initializes.
    libcuckoo::cuckoohash_map<row_uuid_t, std::vector<DeltaDataPtr>,
                              std::hash<row_uuid_t>, std::equal_to<>,
                              proteus::memory::PinnedMemoryAllocator<std::pair<
                                  const row_uuid_t, std::vector<DeltaDataPtr>>>,
                              1>
        consolidateHT;

    void initSlackCache();

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
      if constexpr (DELTA_DEBUG) {
        LOG(INFO) << "Clearing DeltaPartition-" << (int)pid;
      }
      MemoryManager::freePinned(ver_data_mem.data);
    }

    void reset() {
      if (__builtin_expect(touched, 1)) {
        ver_data_cursor = reinterpret_cast<uintptr_t>(ver_data_mem.data);

        for (auto &[thread_id, slack_cache] : _threadLocal_cache) {
          slack_cache.remaining_slack = 0;
        }

        // std::atomic_thread_fence(std::memory_order_seq_cst);

        if (!(this->consolidateHT.empty())) {
          consolidateHT.clear();
        }
      }
      touched = false;
    }

    void *getVersionDataChunk(size_t rec_size);
    void *getVersionDataChunk_ThreadLocal(size_t rec_size);
    void *getVersionDataChunk_ThreadLocal2(size_t rec_size);

    inline double usage() {
      if (touched) {
        return ((double)(((char *)ver_data_cursor.load() -
                          (char *)this->ver_data_mem.data))) /
               this->ver_data_mem.size;
      } else {
        return 0;
      }
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
  //  bool touched;
  std::atomic<uint32_t> tag{};
  std::vector<DeltaPartition *> partitions{};

  class meta2 {
   public:
    alignas(hardware_destructive_interference_size) volatile xid_t
        max_active_epoch = 0;
    volatile xid_t min_version_ts = UINT64_MAX;
  };

  alignas(hardware_destructive_interference_size)
      std::vector<meta2 *> meta_per_thread{};

  std::unordered_map<std::thread::id, meta2 *> _threadlocal_meta{};

  inline meta2 *getMeta_threadLocal() {
    return this->_threadlocal_meta[std::this_thread::get_id()];
  }

  inline xid_t getMaxActiveEpoch() {
    xid_t max = 0;
    for (const auto &m : meta_per_thread) {
      if (m && m->max_active_epoch > max) max = m->max_active_epoch;
    }
    return max;
  }
  inline void resetMinVersionTs() {
    for (auto &m : meta_per_thread) {
      if (m) m->min_version_ts = UINT64_MAX;
    }
  }
  inline xid_t getMinVersionTs() {
    xid_t min = UINT64_MAX;
    for (const auto &m : meta_per_thread) {
      if (m && m->min_version_ts < min) min = m->min_version_ts;
    }
    return min;
  }

  std::atomic<int64_t> readers{};

  volatile uint gc_requests{};
  volatile uint gc_reset_success{};
  volatile uint gc_consolidate_success{};

 public:
  uint64_t total_mem_reserved;

  friend class DeltaDataPtr;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
