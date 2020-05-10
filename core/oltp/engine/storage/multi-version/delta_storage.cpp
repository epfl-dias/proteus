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

#include "storage/delta_storage.hpp"

#include <sys/mman.h>

#include "scheduler/worker.hpp"
#include "storage/table.hpp"

namespace storage {

DeltaStore::DeltaStore(uint32_t delta_id, uint64_t ver_list_capacity,
                       uint64_t ver_data_capacity, int num_partitions)
    : touched(false) {
  this->delta_id = delta_id;

  ver_list_capacity = ver_list_capacity * (1024 * 1024 * 1024);  // GB
  ver_list_capacity = ver_list_capacity / 2;
  ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB
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
  this->ops.store(0);
  this->gc_lock.store(0);
  this->tag = 1;
  this->max_active_epoch = 0;
  // this->min_active_epoch = std::numeric_limits<uint64_t>::max();
}

DeltaStore::~DeltaStore() {
  print_info();
  for (auto& p : partitions) {
    p->~DeltaPartition();
    storage::memory::MemoryManager::free(p);
  }
}

void DeltaStore::print_info() {
  static int i = 0;
  std::cout << "[DeltaStore # " << i
            << "] Number of GC Requests: " << this->gc_requests.load()
            << std::endl;

  std::cout << "[DeltaStore # " << i << "] Number of Successful GC Resets: "
            << this->gc_reset_success.load() << std::endl;
  std::cout << "[DeltaStore # " << i
            << "] Number of Operations: " << this->ops.load() << std::endl;

  for (auto& p : partitions) {
    p->report();
  }
  i++;
  if (i >= partitions.size()) i = 0;
}

void* DeltaStore::insert_version(global_conf::IndexVal* idx_ptr, uint rec_size,
                                 ushort parition_id, const ushort* col_idx,
                                 ushort num_cols) {
  char* cnk = (char*)partitions[parition_id]->getVersionDataChunk(rec_size);

  global_conf::mv_version* val = new ((void*)cnk) global_conf::mv_version(
      idx_ptr->t_min, 0, cnk + sizeof(global_conf::mv_version));

  if (num_cols > 0) {
    for (auto i = 0; i < num_cols; i++) {
      val->col_ids.emplace_back(col_idx[i]);
    }
  }

  auto curr_tag = create_delta_tag(this->delta_id, tag);
  if (idx_ptr->delta_ver_tag != curr_tag) {
    // none/stale list
    idx_ptr->delta_ver_tag = curr_tag;
    idx_ptr->delta_ver =
        (global_conf::mv_version_list*)partitions[parition_id]->getListChunk();
    idx_ptr->delta_ver->head = val;
  } else {
    // valid list
    idx_ptr->delta_ver->insert(val);
  }

  if (!touched) touched = true;
  return val->data;
}

void DeltaStore::gc() {
  short e = 0;
  if (gc_lock.compare_exchange_strong(e, -1)) {
    // gc_requests++;

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
      gc_reset_success++;
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
  int warmup_size = ver_list_mem.size / sizeof(uint64_t);
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

}  // namespace storage
