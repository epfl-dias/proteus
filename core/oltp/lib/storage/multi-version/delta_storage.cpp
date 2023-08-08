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

#include "oltp/storage/multi-version/delta_storage.hpp"

#include <platform/common/error-handling.hpp>

#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/common/utils.hpp"
#include "oltp/storage/multi-version/delta-memory-ptr.hpp"
#include "oltp/storage/multi-version/mv.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace storage {

alignas(64)
    std::unordered_map<delta_id_t, uintptr_t> DeltaDataPtr::data_memory_base;
alignas(64) std::unordered_map<
    delta_id_t, CircularDeltaStore*> DeltaDataPtr::deltaStore_map;

DeltaStoreMalloc* ClassicPtrWrapper::deltaStore = nullptr;

DeltaStoreMalloc::DeltaStoreMalloc(delta_id_t delta_id,
                                   uint64_t ver_data_capacity,
                                   partition_id_t num_partitions) {
  LOG(INFO) << "DeltaStoreMalloc: " << (uint)delta_id;
  ClassicPtrWrapper::deltaStore = this;
}

DeltaStoreMalloc::~DeltaStoreMalloc() = default;

void DeltaStoreMalloc::initThreadLocalPools(partition_id_t partition_id) {
  static std::mutex initLk;
  if constexpr (memory_pool_per_part) {
    std::unique_lock<std::mutex> lk(initLk);
    if (!_memPoolsPart.contains(partition_id)) {
      LOG(INFO) << "Init Memory pool for PID: " << (uint)partition_id;
      auto& pool = _memPoolsPart[partition_id];
      pool.init();
      auto* x = pool.allocate();
      assert(x);
      pool.free(x);
    }
  } else {
    LOG(INFO) << "Init Memory pool for TID: " << std::this_thread::get_id();
    {
      std::unique_lock<std::mutex> lk(initLk);
      auto& y = _memPools[std::this_thread::get_id()];
    }
    auto& pool = _memPools[std::this_thread::get_id()];
    pool.init();
    auto* x = pool.allocate();
    assert(x);
    pool.free(x);
  }
}

ClassicPtrWrapper DeltaStoreMalloc::allocate(size_t sz,
                                             partition_id_t partition_id) {
  // FIXME: Memory pool chunks are hardcoded to avoid the overhead of malloc for
  //  baseline in a paper, and hence, should be tuned or set to a big enough
  //  number to cater for all version request size for the given workload.
  LOG_IF(FATAL, sz > pool_chunk_sz) << "Alloc requested: " << sz;

  void* cnk;
  if constexpr (memory_pool_per_part) {
    cnk = _memPoolsPart[partition_id].allocate();
  } else {
    cnk = _memPools[std::this_thread::get_id()].allocate();
  }
  assert(cnk);
  return ClassicPtrWrapper{reinterpret_cast<uintptr_t>(cnk), partition_id};
}

ClassicPtrWrapper DeltaStoreMalloc::allocate(size_t sz,
                                             partition_id_t partition_id,
                                             xid_t) {
  return allocate(sz, partition_id);
}
void DeltaStoreMalloc::release(ClassicPtrWrapper& ptr) {
  if constexpr (memory_pool_per_part) {
    _memPoolsPart[ptr.getPid()].free(ptr.get_ptr());
  } else {
    _memPools[std::this_thread::get_id()].free(ptr.get_ptr());
  }
}

CircularDeltaStore::CircularDeltaStore(delta_id_t delta_id,
                                       double ver_data_capacity,
                                       partition_id_t num_partitions)
    : delta_id(delta_id) {
  ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB

  assert(ver_data_capacity < std::pow(2, DeltaDataPtr::offset_bits));

  DeltaDataPtr::deltaStore_map.emplace(delta_id, this);

  for (int i = 0; i < num_partitions; i++) {
    const auto& numa_idx = storage::NUMAPartitionPolicy::getInstance()
                               .getPartitionInfo(i)
                               .numa_idx;
    void* mem_data =
        MemoryManager::mallocPinnedOnNode(ver_data_capacity, numa_idx);
    assert(mem_data != nullptr);

    void* obj_data =
        MemoryManager::mallocPinnedOnNode(sizeof(DeltaPartition), numa_idx);

    partitions.emplace_back(new (obj_data) DeltaPartition(
        reinterpret_cast<uintptr_t>(mem_data),
        oltp::common::mem_chunk(mem_data, ver_data_capacity, numa_idx), i,
        delta_id));

    // Insert references into delta-chunk.
    auto idx = DeltaDataPtr::create_delta_idx_pid_pair(delta_id, i);
    DeltaDataPtr::data_memory_base.emplace(
        idx, reinterpret_cast<uintptr_t>(mem_data));
  }

  LOG(INFO) << "Delta ID: " << (uint)(this->delta_id);
  LOG(INFO) << "\tDelta size: "
            << ((double)(ver_data_capacity) / (1024 * 1024 * 1024)) << " GB * "
            << (uint)num_partitions << " Partitions" << std::endl;
  LOG(INFO) << "\tDelta size: "
            << ((double)(ver_data_capacity) * (uint)num_partitions /
                (1024 * 1024 * 1024))
            << " GB" << std::endl;

  this->total_mem_reserved = (ver_data_capacity)*num_partitions;
  this->readers = 0;
  this->gc_reset_success = 0;
  this->gc_requests = 0;
  this->tag = 1;

  //   proteus::utils::timed_func::interval_runner(
  //       [this](){
  //         for (auto &p : partitions) {
  //           if (p->usage() > ((double)GC_CAPACITY_MIN_PERCENT) / 100) {
  //             if(this->delta_id != 0)
  //              LOG(INFO) << "\t\tDeltaID: " << (uint)(this->delta_id) << " |
  //              usage: " << p->usage()*100;
  //             else
  //               LOG(INFO) << "DeltaID: " << (uint)(this->delta_id) << " |
  //               usage: " << p->usage()*100;
  //           }
  //         }
  //
  //      }, (1000)); // print delta-usage every 1 second
}

CircularDeltaStore::~CircularDeltaStore() {
  readers = 0;
  print_info();
  LOG(INFO) << "[" << (int)(this->delta_id)
            << "] Delta Partitions: " << partitions.size();

  for (auto& p : partitions) {
    p->~DeltaPartition();
    MemoryManager::freePinned(p);
  }
}

void CircularDeltaStore::print_info() {
  LOG(INFO) << "[CircularDeltaStore # " << (int)(this->delta_id)
            << "] Number of Successful GC Resets: " << this->gc_reset_success;
  LOG(INFO) << "[CircularDeltaStore # " << (int)(this->delta_id)
            << "] Number of Successful GC Consolidations: "
            << this->gc_consolidate_success;

  if constexpr (DELTA_DEBUG) {
    LOG(INFO) << "[CircularDeltaStore # " << (int)(this->delta_id)
              << "] Number of GC Requests: " << this->gc_requests;
  }

  for (auto& p : partitions) {
    p->report();
  }
}

void CircularDeltaStore::initThreadLocalPools(partition_id_t partition_id) {
  // Assumed to be called for each worker thread initially, thereby,
  // initializing and registering thread-local pool for that specific worker.

  static std::mutex m;
  {
    std::unique_lock<std::mutex> lk;
    auto* o = new meta2;
    meta_per_thread.emplace_back(o);
    this->_threadlocal_meta.emplace(std::this_thread::get_id(), o);
    //    if constexpr (DELTA_DEBUG) {
    //      LOG(INFO) << "DeltaId: " << (uint)(this->delta_id)
    //                << " meta2 thread_size: " << _threadlocal_meta.size() << "
    //                | "
    //                << std::this_thread::get_id();
    //    }
  }
  for (auto& p : partitions) {
    p->initSlackCache();
  }
}
DeltaDataPtr CircularDeltaStore::allocate(size_t sz,
                                          partition_id_t partition_id,
                                          xid_t version_ts) {
  if constexpr (OneShot_CONSOLIDATION) {
    if (version_ts < getMeta_threadLocal()->min_version_ts) {
      getMeta_threadLocal()->min_version_ts = version_ts;
    }
    //    while (version_ts < deltaMeta.min_version_ts) {
    //      deltaMeta.min_version_ts = version_ts;
    //    }
  }
  return allocate(sz, partition_id);
}

DeltaDataPtr CircularDeltaStore::allocate(size_t sz,
                                          partition_id_t partition_id) {
  char* cnk =
      (char*)partitions[partition_id]->getVersionDataChunk_ThreadLocal2(sz);

  //  if (!touched) touched = true;
  return DeltaDataPtr{cnk, tag.load(), this->delta_id, partition_id};
}

void CircularDeltaStore::release(DeltaDataPtr& ptr) { ptr._val = 0; }

void CircularDeltaStore::saveInstanceCrossingPtr(row_uuid_t row_uuid,
                                                 DeltaDataPtr ptr) {
  // FIXME: Temporary fix, bu we need to remove cuckoo from here.

  auto pid = storage::StorageUtils::get_pid_from_rowUuid(row_uuid);

  auto found = this->partitions[pid]->consolidateHT.update_fn(
      row_uuid, [&](std::vector<DeltaDataPtr>& v) { v.push_back(ptr); });

  if (!found) {
    this->partitions[pid]->consolidateHT.insert(row_uuid,
                                                std::vector<DeltaDataPtr>{ptr});
  }
}

bool CircularDeltaStore::impl_gc_withConsolidation() {
  xid_t min = UINT64_MAX;
  auto activeTxns =
      txn::TransactionManager::getInstance().get_all_CurrentActiveTxn(min);

  //      auto min =
  //          std::min_element(activeTxns.begin(),
  //          activeTxns.end()).operator*();

  //      LOG(INFO) << "REPORT_GC " << (uint)(this->delta_id) << " | min: " <<
  //      min
  //                << " && max_active_epoch: " << deltaMeta.max_active_epoch
  //                << " && min_verTS: " << deltaMeta.min_version_ts;

  auto max_active_epoch = getMaxActiveEpoch();  // deltaMeta.max_active_epoch

  if (min > max_active_epoch) {
    return true;
  } else {
    // TRY TO CONSOLIDATE:
    // All txn should hold:
    //  ( txTs > deltaMeta.max_active_epoch ||
    //            txTs < deltaMeta.min_version_ts)

    bool can_consolidate = true;

    auto min_version_ts = getMinVersionTs();  // deltaMeta.min_version_ts
    for (auto txTs : activeTxns) {
      if (txTs <= max_active_epoch && txTs >= min_version_ts) {
        can_consolidate = false;
        break;
      }
    }

    if constexpr (DELTA_DEBUG) {
      if (can_consolidate) {
        gc_consolidate_success++;
        LOG(INFO) << "Consolidate " << (uint)(this->delta_id)
                  << " | min: " << min;
      }
    }
    return can_consolidate;
  }
}

bool CircularDeltaStore::impl_gc_simple() {
  if (should_gc()) {
    auto min = txn::TransactionManager::getInstance().get_min_activeTxn();
    auto max_active_epoch = getMaxActiveEpoch();  // deltaMeta.max_active_epoch
    if (min > max_active_epoch) {
      return true;
    }
  }
  return false;
}

void CircularDeltaStore::GC() {
  static thread_local auto& txnManager = txn::TransactionManager::getInstance();
  int64_t e = 0;
  bool gc_allowed = false;
  if (readers.compare_exchange_weak(e, INT64_MIN)) {
    // #if DELTA_DEBUG
    //     gc_requests++;
    //     // eventlogger.log(this, OLTP_GC_REQUEST);
    // #endif

    if constexpr (OneShot_CONSOLIDATION) {
      gc_allowed = impl_gc_withConsolidation();
    } else {
      gc_allowed = impl_gc_simple();
    }

    if (gc_allowed) {
      tag++;
      for (auto& p : partitions) {
        p->reset();
      }
      // touched = false;
      gc_reset_success++;

      resetMinVersionTs();
    }

    readers = 0;
  }
}

CircularDeltaStore::DeltaPartition::DeltaPartition(
    uintptr_t ver_data_cursor, oltp::common::mem_chunk ver_data_mem,
    partition_id_t pid, delta_id_t delta_id)
    : ver_data_mem(ver_data_mem),
      ver_data_cursor(ver_data_cursor),
      data_cursor_max(ver_data_cursor + ver_data_mem.size),
      touched(false),
      pid(pid),
      delta_id(delta_id),
      delta_uuid(create_delta_uuid(delta_id, pid)),
      consolidateHT(512) {
  // warm-up mem-list
  LOG_IF(INFO, DELTA_DEBUG)
      << "\t warming up delta storage-" << static_cast<uint>(delta_id) << " P"
      << (uint32_t)pid << std::endl;

  // warm-up mem-data
  auto* pt = (uint64_t*)ver_data_cursor;
  uint64_t warmup_size = ver_data_mem.size / sizeof(uint64_t);
  pt[0] = 1;
  for (auto i = 1; i < warmup_size; i++) pt[i] = i * 2;

  auto max_workers_in_partition =
      topology::getInstance()
          .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                 .getPartitionInfo(pid)
                                 .numa_idx]
          .local_cores.size();
}

void* CircularDeltaStore::DeltaPartition::getVersionDataChunk(size_t rec_size) {
  auto sz = rec_size;
  auto tmp = ver_data_cursor.fetch_add(sz, std::memory_order_relaxed);

  LOG_IF(FATAL, (tmp + sz) > data_cursor_max)
      << "DeltaFull: " << (size_t)this->delta_id;
  assert((tmp + sz) <= data_cursor_max);
  touched = true;
  return reinterpret_cast<void*>(tmp);
}

void CircularDeltaStore::DeltaPartition::initSlackCache() {
  static std::mutex initLk;
  {
    std::unique_lock<std::mutex> lk(initLk);
    auto& x = this->_threadLocal_cache[std::this_thread::get_id()];
    // NOTE: is the following necessary?
    this->getVersionDataChunk_ThreadLocal2(8);
    touched = true;
  }
}

void* CircularDeltaStore::DeltaPartition::getVersionDataChunk_ThreadLocal2(
    size_t rec_size) {
  auto& slack_cache = this->_threadLocal_cache.at(std::this_thread::get_id());

  if (rec_size > slack_cache.remaining_slack) {
    slack_cache.ptr = ver_data_cursor.fetch_add(DeltaSlackCache::slack_size);
    slack_cache.remaining_slack = DeltaSlackCache::slack_size;

    // FIXME: this means delta is out-of-memory. three options to implement:
    //  1) Either go granular-GC, start from the first object in this delta, and
    //  move along, 2) horizontal expansion: add a new delta, or 3) vertical
    //  expansion: increase the size of the delta
    assert(slack_cache.ptr >= reinterpret_cast<uintptr_t>(ver_data_mem.data) &&
           slack_cache.ptr < data_cursor_max);
  }

  auto tmp = slack_cache.ptr;
  slack_cache.ptr += rec_size;
  slack_cache.remaining_slack -= rec_size;

  if (!touched) {
    touched = true;
  }

  return reinterpret_cast<void*>(tmp);
}

}  // namespace storage
