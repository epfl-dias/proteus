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

#include <sys/mman.h>

#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/storage/multi-version/mv.hpp"
#include "platform/common/error-handling.hpp"

namespace storage {

std::map<delta_id_t, DeltaStore*> DeltaMemoryPtr::deltaStore_map;
std::map<delta_id_t, uintptr_t> DeltaList::list_memory_base;
std::map<delta_id_t, uintptr_t> DeltaDataPtr::data_memory_base;

DeltaStore::DeltaStore(delta_id_t delta_id, uint64_t ver_list_capacity,
                       uint64_t ver_data_capacity,
                       partition_id_t num_partitions)
    : touched(false), delta_id(delta_id) {
  DeltaList::deltaStore_map.emplace(delta_id, this);

  ver_list_capacity = ver_list_capacity * (1024 * 1024 * 1024);  // GB
  ver_list_capacity = ver_list_capacity / 2;
  ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB

  assert(ver_data_capacity < std::pow(2, DeltaList::offset_bits));

  for (int i = 0; i < num_partitions; i++) {
    const auto& numa_idx = storage::NUMAPartitionPolicy::getInstance()
                               .getPartitionInfo(i)
                               .numa_idx;

    void* mem_list =
        MemoryManager::mallocPinnedOnNode(ver_list_capacity, numa_idx);
    void* mem_data =
        MemoryManager::mallocPinnedOnNode(ver_data_capacity, numa_idx);

    assert(mem_list != nullptr);
    assert(mem_data != nullptr);

    void* obj_data =
        MemoryManager::mallocPinnedOnNode(sizeof(DeltaPartition), numa_idx);

    partitions.emplace_back(new (obj_data) DeltaPartition(
        (char*)mem_list,
        oltp::common::mem_chunk(mem_list, ver_list_capacity, numa_idx),
        (char*)mem_data,
        oltp::common::mem_chunk(mem_data, ver_data_capacity, numa_idx), i,
        delta_id));

    // Insert references into delta-chunk.
    auto idx = DeltaMemoryPtr::create_delta_idx_pid_pair(delta_id, i);
    DeltaList::list_memory_base.emplace(idx,
                                        reinterpret_cast<uintptr_t>(mem_list));
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

  this->readers.store(0);
  this->gc_reset_success.store(0);
  this->gc_requests.store(0);
  this->gc_lock.store(0);
  this->tag = 1;
  this->max_active_epoch = 0;
  // this->min_active_epoch = std::numeric_limits<uint64_t>::max();

  LOG(INFO) << "Sizeof(char*): " << sizeof(char*);
  LOG(INFO) << "Sizeof(uintptr_t): " << sizeof(uintptr_t);
  LOG(INFO) << "sizeof(TaggedDeltaDataPtr<storage::mv::mv_version>): "
            << sizeof(TaggedDeltaDataPtr<storage::mv::mv_version>);
  LOG(INFO) << "sizeof(DeltaList): " << sizeof(DeltaList);
  LOG(INFO) << "sizeof(DeltaDataPtr): " << sizeof(DeltaDataPtr);
  LOG(INFO) << "sizeof(DeltaPartition): " << sizeof(DeltaPartition);
}

DeltaStore::~DeltaStore() {
  print_info();
  LOG(INFO) << "[" << (int)(this->delta_id)
            << "] Delta Partitions: " << partitions.size();

  for (auto& p : partitions) {
    p->~DeltaPartition();
    MemoryManager::freePinned(p);
  }
}

void DeltaStore::print_info() {
  LOG(INFO) << "[DeltaStore # " << (int)(this->delta_id)
            << "] Number of Successful GC Resets: "
            << this->gc_reset_success.load();

#if DELTA_DEBUG
  LOG(INFO) << "[DeltaStore # " << (int)(this->delta_id)
            << "] Number of GC Requests: " << this->gc_requests.load();
#endif

  for (auto& p : partitions) {
    p->report();
  }
}

void* DeltaStore::getTransientChunk(DeltaList& delta_list, size_t size,
                                    partition_id_t partition_id) {
  throw std::runtime_error("deprecated after delta-list-bug fix");
  auto* ptr = partitions[partition_id]->getChunk(size);

  delta_list.update(reinterpret_cast<const char*>(ptr),
                    tag.load(std::memory_order_acquire), this->delta_id,
                    partition_id);

  if (!touched) touched = true;
  return ptr;
}

void* DeltaStore::validate_or_create_list(DeltaList& delta_list,
                                          partition_id_t partition_id) {
  throw std::runtime_error("deprecated after delta-list-bug fix");

  auto* delta_ptr = (storage::mv::mv_version_chain*)(delta_list.ptr());

  if (delta_ptr == nullptr) {
    // none/stale list
    auto* list_ptr = (storage::mv::mv_version_chain*)new (
        partitions[partition_id]->getListChunk())
        storage::mv::mv_version_chain();
    delta_list.update(reinterpret_cast<const char*>(list_ptr),
                      tag.load(std::memory_order_acquire), this->delta_id,
                      partition_id);

    // logic for transient timestamps instead of persistent.
    list_ptr->last_updated_tmin =
        scheduler::WorkerPool::getInstance().get_min_active_txn();

    if (!touched) touched = true;

    return list_ptr;
  }

  return delta_ptr;
}

void* DeltaStore::create_version(size_t size, partition_id_t partition_id) {
  throw std::runtime_error("deprecated after delta-list-bug fix");
  char* cnk = (char*)partitions[partition_id]->getVersionDataChunk(size);
  if (!touched) touched = true;
  return cnk;
}

void* DeltaStore::insert_version(DeltaDataPtr& delta_list, xid_t t_min,
                                 xid_t t_max, size_t rec_size,
                                 partition_id_t partition_id) {
  assert(!storage::mv::mv_type::isPerAttributeMVList);

  char* cnk = (char*)partitions[partition_id]->getVersionDataChunk_ThreadLocal(
      rec_size);

  auto* version_ptr = new ((void*)cnk) storage::mv::mv_version(
      t_min, t_max, cnk + sizeof(storage::mv::mv_version));

  // point next version to previous head, regardless it is valid or not.
  if (delta_list.get_ptr() != nullptr) {
    version_ptr->next._val = delta_list._val;
  }

  // update pointer on delta_list (which is inside index) to point to latest
  // version.
  delta_list.update(reinterpret_cast<const char*>(version_ptr),
                    tag.load(std::memory_order_acquire), this->delta_id,
                    partition_id);

  if (!touched) touched = true;
  return version_ptr;
}

void DeltaStore::gc() {
  short e = 0;
  if (gc_lock.compare_exchange_strong(e, -1)) {
#if DELTA_DEBUG
    gc_requests++;
#endif
    xid_t last_alive_txn =
        scheduler::WorkerPool::getInstance().get_min_active_txn();
    if (this->readers == 0 && should_gc() &&
        last_alive_txn > max_active_epoch) {
      // LOG(INFO) << "ACTUAL GC START";
      tag++;
      for (auto& p : partitions) {
        p->reset();
      }
      gc_lock.store(0);
      touched = false;
      gc_reset_success.fetch_add(1, std::memory_order_relaxed);
      // LOG(INFO) << "GC HAPPENED";
    } else {
      // gc_lock.unlock();
      gc_lock.store(0);
    }
  }
  // LOG(INFO) << "GC END";
}

DeltaStore::DeltaPartition::DeltaPartition(char* ver_list_cursor,
                                           oltp::common::mem_chunk ver_list_mem,
                                           char* ver_data_cursor,
                                           oltp::common::mem_chunk ver_data_mem,
                                           partition_id_t pid,
                                           delta_id_t delta_id)
    : ver_list_mem(ver_list_mem),
      ver_data_mem(ver_data_mem),
      ver_list_cursor(ver_list_cursor),
      ver_data_cursor(ver_data_cursor),
      list_cursor_max(ver_list_cursor + ver_list_mem.size),
      data_cursor_max(ver_data_cursor + ver_data_mem.size),
      touched(false),
      pid(pid),
      delta_id(delta_id),
      delta_uuid(create_delta_uuid(delta_id, pid)) {
  printed = false;
  // warm-up mem-list
  LOG(INFO) << "\t warming up delta storage P" << (uint32_t)pid << std::endl;

  auto* pt = (uint64_t*)ver_list_cursor;
  uint64_t warmup_size = ver_list_mem.size / sizeof(uint64_t);
  pt[0] = 3;
  for (auto i = 1; i < warmup_size; i++) pt[i] = i * 2;

  // warm-up mem-data
  pt = (uint64_t*)ver_data_cursor;
  warmup_size = ver_data_mem.size / sizeof(uint64_t);
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

void* DeltaStore::DeltaPartition::getListChunk() {
  char* tmp = ver_list_cursor.fetch_add(sizeof(storage::mv::mv_version_chain),
                                        std::memory_order_relaxed);

  assert((tmp + sizeof(storage::mv::mv_version_chain)) <= list_cursor_max);
  touched = true;
  return tmp;
}

void* DeltaStore::DeltaPartition::getChunk(size_t size) {
  char* tmp = ver_list_cursor.fetch_add(size, std::memory_order_relaxed);

  assert((tmp + size) <= list_cursor_max);
  touched = true;
  return tmp;
}

void* DeltaStore::DeltaPartition::getVersionDataChunk(size_t rec_size) {
  auto sz = rec_size + sizeof(storage::mv::mv_version);
  char* tmp = ver_data_cursor.fetch_add(sz, std::memory_order_relaxed);

  assert(tmp >= (char*)(ver_data_mem.data) && (tmp + sz) <= data_cursor_max);
  touched = true;
  return tmp;
}

void* DeltaStore::DeltaPartition::getVersionDataChunk_ThreadLocal(
    size_t rec_size) {
  constexpr uint slack_size = 8192;

  static int thread_counter = 0;
  static thread_local uint tid = thread_counter++;

  // there can be multiple delta-storages, so this function needs to maintain
  // slack per delta-storage per thread.

  // n_delta_storage * n_threads

  static thread_local std::map<uint16_t, threadLocalSlack, std::less<uint16_t>,
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
  size_t request_size = rec_size + sizeof(storage::mv::mv_version);

  if (reset_listeners[tid]) {
    slackRef.remaining_slack = 0;
    reset_listeners[tid] = false;
  }

  if (__unlikely(request_size > slackRef.remaining_slack)) {
    slackRef.ptr = ver_data_cursor.fetch_add(slack_size);
    slackRef.remaining_slack = slack_size;

    assert(slackRef.ptr >= (char*)(ver_data_mem.data) &&
           slackRef.ptr < data_cursor_max);

    LOG_IF(FATAL, (slackRef.ptr + slackRef.remaining_slack) > data_cursor_max)
        << "############## DeltaMemory Full\n"
        << "DeltaID: " << (uint)delta_id << "\t\tPID: " << (uint)pid
        << "##############";

    // if(  (slackRef.ptr + slackRef.remaining_slack)  > data_cursor_max ){}
  }

  assert(request_size <= slackRef.remaining_slack);
  assert((slackRef.ptr >= (char*)(ver_data_mem.data)));
  assert((slackRef.ptr < data_cursor_max));

  char* tmp = slackRef.ptr;
  slackRef.ptr += request_size;
  slackRef.remaining_slack -= request_size;

  assert(slackRef.ptr >= (char*)(ver_data_mem.data) &&
         slackRef.ptr < data_cursor_max);
  assert(tmp >= (char*)(ver_data_mem.data) && tmp < data_cursor_max);

  if (!touched) {
    touched = true;
  }

  return tmp;
}

}  // namespace storage
