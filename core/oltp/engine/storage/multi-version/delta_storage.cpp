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

#include "storage/multi-version/delta_storage.hpp"

#include <sys/mman.h>

#include "scheduler/worker.hpp"
#include "storage/multi-version/mv.hpp"
#include "storage/table.hpp"

namespace storage {

std::map<uint8_t, DeltaStore*> DeltaList::deltaStore_map;
std::map<uint8_t, char*> DeltaList::list_memory_base;

DeltaStore::DeltaStore(uint8_t delta_id, uint64_t ver_list_capacity,
                       uint64_t ver_data_capacity, int num_partitions)
    : touched(false) {
  this->delta_id = delta_id;
  DeltaList::deltaStore_map.emplace(delta_id, this);

  ver_list_capacity = ver_list_capacity * (1024 * 1024 * 1024);  // GB
  ver_list_capacity = ver_list_capacity / 2;
  ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB

  assert(ver_data_capacity < std::pow(2, DeltaList::offset_bits));

  for (int i = 0; i < num_partitions; i++) {
    const auto& numa_idx = storage::NUMAPartitionPolicy::getInstance()
                               .getPartitionInfo(i)
                               .numa_idx;

    void* mem_list = storage::memory::MemoryManager::alloc(
        ver_list_capacity, numa_idx, MADV_DONTFORK | MADV_HUGEPAGE);
    void* mem_data = storage::memory::MemoryManager::alloc(
        ver_data_capacity, numa_idx, MADV_DONTFORK | MADV_HUGEPAGE);
    assert(mem_list != NULL);
    assert(mem_data != NULL);

    assert(mem_list != nullptr);
    assert(mem_data != nullptr);

    void* obj_data = storage::memory::MemoryManager::alloc(
        sizeof(DeltaPartition), numa_idx, MADV_DONTFORK);

    partitions.emplace_back(new (obj_data) DeltaPartition(
        (char*)mem_list,
        storage::memory::mem_chunk(mem_list, ver_list_capacity, numa_idx),
        (char*)mem_data,
        storage::memory::mem_chunk(mem_data, ver_data_capacity, numa_idx), i));

    // Insert references into delta-chunk.
    auto idx = DeltaList::create_delta_idx_pid_pair(delta_id, i);
    DeltaList::list_memory_base.emplace(idx, (char*)mem_list);
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
}

DeltaStore::~DeltaStore() {
  print_info();
  LOG(INFO) << "[" << this->delta_id
            << "] Delta Partitions: " << partitions.size();

  for (auto& p : partitions) {
    p->~DeltaPartition();
    storage::memory::MemoryManager::free(p);
  }
}

void DeltaStore::print_info() {
  LOG(INFO) << "[DeltaStore # " << this->delta_id
            << "] Number of Successful GC Resets: "
            << this->gc_reset_success.load();

#if DELTA_DEBUG
  LOG(INFO) << "[DeltaStore # " << this->delta_id
            << "] Number of GC Requests: " << this->gc_requests.load();
#endif

  for (auto& p : partitions) {
    p->report();
  }
}

void* DeltaStore::getTransientChunk(DeltaList& delta_chunk, uint size,
                                    ushort partition_id) {
  auto* ptr = partitions[partition_id]->getChunk(size);

  delta_chunk.update(reinterpret_cast<const char*>(ptr),
                     tag.load(std::memory_order_acquire), this->delta_id,
                     partition_id);

  if (!touched) touched = true;
  return ptr;
}

void* DeltaStore::validate_or_create_list(DeltaList& delta_chunk,
                                          ushort partition_id) {
  auto* delta_ptr = (storage::mv::mv_version_chain*)(delta_chunk.ptr());

  if (delta_ptr == nullptr) {
    // none/stale list
    auto* list_ptr = (storage::mv::mv_version_chain*)new (
        partitions[partition_id]->getListChunk())
        storage::mv::mv_version_chain();
    delta_chunk.update(reinterpret_cast<const char*>(list_ptr),
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

void* DeltaStore::create_version(size_t size, ushort partition_id) {
  char* cnk = (char*)partitions[partition_id]->getVersionDataChunk(size);
  if (!touched) touched = true;
  return cnk;
}

void* DeltaStore::insert_version(DeltaList& delta_chunk, uint64_t t_min,
                                 uint64_t t_max, uint rec_size,
                                 ushort partition_id) {
  assert(!storage::mv::mv_type::isPerAttributeMVList);

  char* cnk = (char*)partitions[partition_id]->getVersionDataChunk(rec_size);

  auto* version_ptr = new ((void*)cnk) storage::mv::mv_version(
      t_min, t_max, cnk + sizeof(storage::mv::mv_version));

  auto* delta_ptr = (storage::mv::mv_version_chain*)(delta_chunk.ptr());

  if (delta_ptr == nullptr) {
    // none/stale list
    auto* list_ptr = (storage::mv::mv_version_chain*)new (
        partitions[partition_id]->getListChunk())
        storage::mv::mv_version_chain();

    list_ptr->head = version_ptr;
    delta_chunk.update(reinterpret_cast<const char*>(list_ptr),
                       tag.load(std::memory_order_acquire), this->delta_id,
                       partition_id);

  } else {
    // valid list
    delta_ptr->insert(version_ptr);
  }

  if (!touched) touched = true;
  return version_ptr;
}

void DeltaStore::gc() {
  short e = 0;
  if (gc_lock.compare_exchange_strong(e, -1)) {
#if DELTA_DEBUG
    gc_requests++;
#endif
    uint64_t last_alive_txn =
        scheduler::WorkerPool::getInstance().get_min_active_txn();
    if (this->readers == 0 && should_gc() &&
        last_alive_txn > max_active_epoch) {
      for (auto& p : partitions) {
        p->reset();
      }
      tag++;
      gc_lock.store(0);
      touched = false;
      gc_reset_success.fetch_add(1, std::memory_order_relaxed);
    } else {
      // gc_lock.unlock();
      gc_lock.store(0);
    }
  }
}

DeltaStore::DeltaPartition::DeltaPartition(
    char* ver_list_cursor, storage::memory::mem_chunk ver_list_mem,
    char* ver_data_cursor, storage::memory::mem_chunk ver_data_mem, int pid)
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

  uint64_t* pt = (uint64_t*)ver_list_cursor;
  uint64_t warmup_size = ver_list_mem.size / sizeof(uint64_t);
  pt[0] = 3;
  for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

  // warm-up mem-data
  pt = (uint64_t*)ver_data_cursor;
  warmup_size = ver_data_mem.size / sizeof(uint64_t);
  pt[0] = 1;
  for (int i = 1; i < warmup_size; i++) pt[i] = i * 2;

  size_t max_workers_in_partition =
      topology::getInstance()
          .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                 .getPartitionInfo(pid)
                                 .numa_idx]
          .local_cores.size();
  for (int i = 0; i < max_workers_in_partition; i++) {
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
  constexpr uint slack_size = 8192;

  static thread_local uint remaining_slack = 0;
  static thread_local char* ptr = nullptr;
  static int thread_counter = 0;

  static thread_local uint tid = thread_counter++;

  size_t req = rec_size + sizeof(storage::mv::mv_version);

  // works.
  if (reset_listeners[tid]) {
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

  char* tmp = ptr;
  ptr += req;
  remaining_slack -= req;

  // char *tmp = ver_data_cursor.fetch_add(req, std::memory_order_relaxed);

  touched = true;
  return tmp;
}

}  // namespace storage
