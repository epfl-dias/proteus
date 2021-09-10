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
#include "oltp/storage/memory-pool.hpp"
#include "oltp/storage/multi-version/delta-memory-ptr.hpp"
#include "oltp/storage/multi-version/mv.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace storage {

std::unordered_map<delta_id_t, CircularDeltaStore*>
    DeltaDataPtr::deltaStore_map;
std::unordered_map<delta_id_t, uintptr_t> DeltaDataPtr::data_memory_base;

constexpr size_t pool_chunk_sz = 384;  // tpcc

DeltaStoreMalloc* ClassicPtrWrapper::deltaStore = nullptr;

DeltaStoreMalloc::DeltaStoreMalloc(delta_id_t delta_id,
                                   uint64_t ver_data_capacity,
                                   partition_id_t num_partitions) {
  LOG(INFO) << "DeltaStoreMalloc: " << (uint)delta_id;
  ClassicPtrWrapper::deltaStore = this;
}

DeltaStoreMalloc::~DeltaStoreMalloc() {
  oltp::mv::memorypool::BucketMemoryPool<pool_chunk_sz>::getInstance()
      .destruct();
}

inline void* alloc_or_free(bool alloc, void* ptr = nullptr) {
  static thread_local oltp::mv::memorypool::BucketMemoryPool_threadLocal<
      pool_chunk_sz>
      memPool;

  if (alloc) {
    return memPool.allocate();
  } else {
    memPool.free(ptr);
    return nullptr;
  }
}

ClassicPtrWrapper DeltaStoreMalloc::allocate(size_t sz,
                                             partition_id_t partition_id) {
  LOG_IF(FATAL, sz > pool_chunk_sz) << "ALloc requested: " << sz;
  auto* cnk = alloc_or_free(true);
  assert(cnk);
  return ClassicPtrWrapper{reinterpret_cast<uintptr_t>(cnk)};
}

void DeltaStoreMalloc::release(ClassicPtrWrapper& ptr) {
  alloc_or_free(false, ptr.get_ptr());
}

CircularDeltaStore::CircularDeltaStore(delta_id_t delta_id,
                                       uint64_t ver_data_capacity,
                                       partition_id_t num_partitions)
    : touched(false), delta_id(delta_id) {
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

    //    LOG(INFO) << "[DATA] Delta-id: " << delta_id << ", PID: " << i
    //              << " | pair: " << idx << " | maxOffset: "
    //              << reinterpret_cast<uintptr_t>(((char*)mem_data) +
    //                                             ver_data_capacity)
    //              << " | Base: " <<
    //              reinterpret_cast<uintptr_t>(((char*)mem_data));

    // DeltaDataPtr::max_offset = reinterpret_cast<uintptr_t>(((char*)mem_data)
    // +ver_data_capacity);

    //    auto offset = reinterpret_cast<uintptr_t>(data_ptr -
    //                                              data_memory_base[create_delta_idx_pid_pair(delta_idx,
    //                                              pid)]);
  }

  if (DELTA_DEBUG) {
    LOG(INFO) << "Delta ID: " << (uint)(this->delta_id);
    LOG(INFO) << "\tDelta size: "
              << ((double)(ver_data_capacity) / (1024 * 1024 * 1024))
              << " GB * " << (uint)num_partitions << " Partitions" << std::endl;
    LOG(INFO) << "\tDelta size: "
              << ((double)(ver_data_capacity) * (uint)num_partitions /
                  (1024 * 1024 * 1024))
              << " GB" << std::endl;
  }
  this->total_mem_reserved = (ver_data_capacity)*num_partitions;

  this->deltaMeta.readers.store(0);
  this->gc_reset_success = 0;
  this->gc_requests = 0;
  // this->gc_lock.store(0);
  this->tag = 1;
  this->deltaMeta.max_active_epoch = 0;
  // this->min_active_epoch = std::numeric_limits<uint64_t>::max();

  //  LOG(INFO) << "Sizeof(char*): " << sizeof(char*);
  //  LOG(INFO) << "Sizeof(uintptr_t): " << sizeof(uintptr_t);
  //  LOG(INFO) << "sizeof(TaggedDeltaDataPtr<storage::mv::mv_version>): "
  //            << sizeof(TaggedDeltaDataPtr<storage::mv::mv_version>);
  //  LOG(INFO) << "sizeof(DeltaList): " << sizeof(DeltaList);
  //  LOG(INFO) << "sizeof(DeltaDataPtr): " << sizeof(DeltaDataPtr);
  //  LOG(INFO) << "sizeof(DeltaPartition): " << sizeof(DeltaPartition);

  //   timed_func::interval_runner(
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
  //      }, (500)); // 500ms
}

CircularDeltaStore::~CircularDeltaStore() {
  deltaMeta.readers = 0;
  deltaMeta.max_active_epoch =
      std::numeric_limits<decltype(deltaMeta.max_active_epoch)>::max();
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

  if constexpr (DELTA_DEBUG) {
    LOG(INFO) << "[CircularDeltaStore # " << (int)(this->delta_id)
              << "] Number of GC Requests: " << this->gc_requests;
  }

  for (auto& p : partitions) {
    p->report();
  }
}

DeltaDataPtr CircularDeltaStore::allocate(size_t sz,
                                          partition_id_t partition_id) {
  char* cnk = (char*)partitions[partition_id]->getVersionDataChunk(sz);

  if (!touched) touched = true;
  return DeltaDataPtr{cnk, tag.load(), this->delta_id, partition_id};
}

void CircularDeltaStore::release(DeltaDataPtr& ptr) { ptr._val = 0; }

void CircularDeltaStore::gc() {
  static thread_local auto& txnManager = txn::TransactionManager::getInstance();
  int64_t e = 0;
  if (deltaMeta.readers.compare_exchange_weak(e, -10000)) {
#if DELTA_DEBUG
    gc_requests++;
    // eventlogger.log(this, OLTP_GC_REQUEST);
#endif

    if (txnManager.get_min_activeTxn() > deltaMeta.max_active_epoch) {
      tag++;
      for (auto& p : partitions) {
        p->reset();
      }
      touched = false;
      gc_reset_success++;
      // eventlogger.log(this, OLTP_GC_SUCCESS);
    }
    deltaMeta.readers = 0;
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
      delta_uuid(create_delta_uuid(delta_id, pid)) {
  // warm-up mem-list
  LOG(INFO) << "\t warming up delta storage P" << (uint32_t)pid << std::endl;

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
  for (auto i = 0; i < max_workers_in_partition; i++) {
    reset_listeners.push_back(false);
  }
}

void* CircularDeltaStore::DeltaPartition::getVersionDataChunk(size_t rec_size) {
  auto sz = rec_size;
  auto tmp = ver_data_cursor.fetch_add(sz, std::memory_order_relaxed);

  assert(tmp >= reinterpret_cast<uintptr_t>(ver_data_mem.data) &&
         (tmp + sz) <= data_cursor_max);
  touched = true;
  return reinterpret_cast<void*>(tmp);
}

void* CircularDeltaStore::DeltaPartition::getVersionDataChunk_ThreadLocal(
    size_t rec_size) {
  constexpr uint slack_size = 8192;

  static int thread_counter = 0;
  static thread_local uint tid = thread_counter++;

  // there can be multiple delta-storages, so this function needs to maintain
  // slack per delta-storage per thread.

  // n_delta_storage * n_threads

  static thread_local std::map<uint16_t, threadLocalSlack, std::less<>,
                               proteus::memory::PinnedMemoryAllocator<
                                   std::pair<const uint16_t, threadLocalSlack>>>
      local_slack_map{};

  // static thread_local std::map<uint16_t, threadLocalSlack> local_slack_map;

  if (__unlikely(!local_slack_map.contains(this->delta_uuid))) {
    // local_slack_map.insert(this->delta_id);
    local_slack_map.insert({this->delta_uuid, threadLocalSlack()});
    // LOG(INFO) << "TiD: " << tid << " Pushing a newDelta: " <<
    // (uint)(this->delta_id) << " Pid: " <<(uint)(this->pid);
  }

  threadLocalSlack& slackRef = local_slack_map[delta_uuid];
  size_t request_size = rec_size;

  if (reset_listeners[tid]) {
    slackRef.remaining_slack = 0;
    reset_listeners[tid] = false;
  }

  if (__unlikely(request_size > slackRef.remaining_slack)) {
    slackRef.ptr = ver_data_cursor.fetch_add(slack_size);
    slackRef.remaining_slack = slack_size;

    LOG_IF(FATAL,
           (slackRef.ptr < reinterpret_cast<uintptr_t>(ver_data_mem.data) ||
            slackRef.ptr >= data_cursor_max))
        << "Issue: id: " << get_delta_id(this->delta_uuid)
        << " | pid: " << get_delta_pid(delta_uuid);

    assert(slackRef.ptr >= reinterpret_cast<uintptr_t>(ver_data_mem.data) &&
           slackRef.ptr < data_cursor_max);

    LOG_IF(FATAL, (slackRef.ptr + slackRef.remaining_slack) > data_cursor_max)
        << "############## DeltaMemory Full\n"
        << "DeltaID: " << (uint)delta_id << "\t\tPID: " << (uint)pid
        << "##############";

    // if(  (slackRef.ptr + slackRef.remaining_slack)  > data_cursor_max ){}
  }

  assert(request_size <= slackRef.remaining_slack);
  assert((slackRef.ptr >= reinterpret_cast<uintptr_t>(ver_data_mem.data)));
  assert((slackRef.ptr < data_cursor_max));

  auto tmp = slackRef.ptr;
  slackRef.ptr += request_size;
  slackRef.remaining_slack -= request_size;

  assert(slackRef.ptr >= reinterpret_cast<uintptr_t>(ver_data_mem.data) &&
         slackRef.ptr < data_cursor_max);
  assert(tmp >= reinterpret_cast<uintptr_t>(ver_data_mem.data) &&
         tmp < data_cursor_max);

  if (!touched) {
    touched = true;
  }

  return reinterpret_cast<void*>(tmp);
}

}  // namespace storage
