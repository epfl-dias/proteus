/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <cassert>
#include <cstring>
#include <iostream>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/thread.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <string>
#include <utility>

#include "oltp/common/constants.hpp"
#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/snapshot/snapshot_manager.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/storage/table.hpp"

namespace storage {

template bool Column::UpdateInPlace<std::vector<oltp::common::mem_chunk>>(
    std::vector<oltp::common::mem_chunk>& data_vector, size_t offset,
    size_t unit_size, void* data);

template bool Column::UpdateInPlace<std::deque<oltp::common::mem_chunk>>(
    std::deque<oltp::common::mem_chunk>& data_vector, size_t offset,
    size_t unit_size, void* data);

//-----------------------------------------------------------------
// Column
//-----------------------------------------------------------------

// Constructor to be called from derived classes only.
Column::Column(SnapshotTypes snapshotType, column_id_t column_id,
               std::string name, data_type type, size_t unit_size,
               size_t offset_inRecord, bool numa_partitioned)
    : column_id(column_id),
      name(name),
      unit_size(unit_size),
      byteOffset_record(offset_inRecord),
      type(type),
      n_partitions(numa_partitioned ? g_num_partitions : 1),
      single_version_only(false),
      snapshotType(snapshotType) {
  // oltp::snapshot::SnapshotMaster::getInstance().turnOnMechanism(snapshotType);
}

Column::~Column() {
  for (auto j = 0; j < g_num_partitions; j++) {
    for (auto& chunk : primaryData[j]) {
      MemoryManager::freePinned(chunk.data);
    }
    primaryData[j].clear();
  }
}

Column::Column(column_id_t column_id, std::string name, data_type type,
               size_t unit_size, size_t offset_inRecord, bool numa_partitioned,
               size_t reserved_capacity, int numa_idx,
               SnapshotTypes snapshotType)
    : column_id(column_id),
      name(std::move(name)),
      unit_size(unit_size),
      byteOffset_record(offset_inRecord),
      type(type),
      n_partitions(numa_partitioned ? g_num_partitions : 1),
      single_version_only(true),
      snapshotType(snapshotType) {
  assert(g_num_partitions <= topology::getInstance().getCpuNumaNodeCount());

  this->capacity = reserved_capacity;
  this->capacity_per_partition =
      (reserved_capacity / n_partitions) + (reserved_capacity % n_partitions);

  this->total_size_per_partition = capacity_per_partition * unit_size;
  this->total_size = this->total_size_per_partition * this->n_partitions;

  std::vector<proteus::thread> loaders;

  for (auto j = 0; j < this->n_partitions; j++) {
    void* mem = MemoryManager::mallocPinnedOnNode(
        this->total_size_per_partition,
        storage::NUMAPartitionPolicy::getInstance()
            .getPartitionInfo(j)
            .numa_idx);
    assert(mem != nullptr);
    loaders.emplace_back([mem, this, j]() {
      set_exec_location_on_scope d{
          topology::getInstance()
              .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                     .getPartitionInfo(j)
                                     .numa_idx]};

      auto* pt = (uint64_t*)mem;
      uint64_t warmup_max = this->total_size_per_partition / sizeof(uint64_t);
#pragma clang loop vectorize(enable)
      for (uint64_t k = 0; k < warmup_max; k++) pt[k] = 0;
    });

    primaryData[j].emplace_back(mem, this->total_size_per_partition,
                                storage::NUMAPartitionPolicy::getInstance()
                                    .getPartitionInfo(j)
                                    .numa_idx);
  }

  for (auto& th : loaders) {
    th.join();
  }
}

void Column::initializeMetaColumn() const {
  assert(this->type == META);
  assert(this->single_version_only);

  std::vector<proteus::thread> loaders;
  for (auto j = 0; j < this->n_partitions; j++) {
    for (const auto& chunk : primaryData[j]) {
      char* ptr = (char*)chunk.data;
      assert(chunk.size % this->unit_size == 0);
      loaders.emplace_back([this, chunk, j, ptr]() {
        for (uint64_t i = 0; i < (chunk.size / this->unit_size); i++) {
          void* c = new (ptr + (i * this->unit_size))
              global_conf::IndexVal(0, StorageUtils::create_vid(i, j));
        }
      });
    }
  }
  for (auto& th : loaders) {
    th.join();
  }
}

/*  DML Functions
 *
 */

void* Column::getElem(rowid_t vid) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t data_idx = StorageUtils::get_offset(vid) * unit_size;

  assert(StorageUtils::get_m_version(vid) == 0 &&
         "shouldn't be used for attribute columns");
  assert(!primaryData[pid].empty());

  for (const auto& chunk : primaryData[pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + unit_size))) {
      return ((char*)chunk.data) + data_idx;
    }
    data_idx -= chunk.size;
  }

  assert(false && "Out-of-Bound-Access");
  return nullptr;
}

void Column::getElem(rowid_t vid, void* copy_location) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t data_idx = StorageUtils::get_offset(vid) * unit_size;

  for (const auto& chunk : primaryData[pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + unit_size))) {
      std::memcpy(copy_location, ((char*)chunk.data) + data_idx,
                  this->unit_size);
      return;
    }
    data_idx -= chunk.size;
  }
  assert(false && "Out-of-Bound-Access");
}

void Column::updateElem(rowid_t vid, void* elem) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t offset = StorageUtils::get_offset(vid);

  assert(pid < n_partitions);
  assert(offset < capacity_per_partition);

  if (!UpdateInPlace(primaryData[pid], offset, unit_size, elem)) {
    assert(false && "Out Of Memory Error");
  }
}

void* Column::insertElem(rowid_t vid) {
  assert(this->type == META);
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t data_idx = StorageUtils::get_offset(vid) * unit_size;

  assert(pid < g_num_partitions);
  assert((data_idx / unit_size) < capacity_per_partition);
  assert(data_idx < total_size_per_partition);

  for (const auto& chunk : primaryData[pid]) {
    if (__likely(chunk.size >= (data_idx + unit_size))) {
      return (void*)(((char*)chunk.data) + data_idx);
    }
    data_idx -= chunk.size;
  }

  assert(false && "Out Of Memory Error");
  return nullptr;
}

void Column::insertElem(rowid_t vid, void* elem) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t offset = StorageUtils::get_offset(vid);

  assert(pid < n_partitions);
  assert(offset < this->capacity_per_partition);

  if (!UpdateInPlace(primaryData[pid], offset, unit_size, elem)) {
    LOG(INFO) << "ALLOCATE MORE MEMORY:\t" << this->name << ",vid: " << vid
              << ", idx:" << offset << ", pid: " << pid;

    assert(false && "Out Of Memory Error");
  }

  /*
  for (const auto& chunk : primaryData[pid]) {
    if (__likely(chunk.size >= (data_idx + unit_size))) {
      void* dst = (void*)(((char*)chunk.data) + data_idx);
      if (__unlikely(elem == nullptr)) {
        uint64_t* tptr = (uint64_t*)dst;
        (*tptr)++;
      } else {
        char* src_t = (char*)chunk.data;
        char* dst_t = (char*)dst;

        assert(src_t <= dst_t);
        assert((src_t + chunk.size) >= (dst_t + this->unit_size));

        std::memcpy(dst, elem, this->unit_size);
      }

      return;
    }
  }

  LOG(INFO) << "(1) ALLOCATE MORE MEMORY:\t" << this->name << ",vid: " << vid
            << ", idx:" << (data_idx / unit_size) << ", pid: " << pid;

  assert(false && "Out Of Memory Error");
   */
}

void* Column::insertElemBatch(rowid_t vid, uint16_t num_elem) {
  assert(this->type == META);

  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t data_idx_st = StorageUtils::get_offset(vid) * unit_size;
  size_t data_idx_en = data_idx_st + (num_elem * unit_size);

  assert(pid < g_num_partitions);
  assert((data_idx_en / unit_size) < capacity_per_partition);
  assert(data_idx_en < total_size_per_partition);

  bool ins = false;
  for (const auto& chunk : primaryData[pid]) {
    if (__likely(chunk.size >= (data_idx_en + unit_size))) {
      return (void*)(((char*)chunk.data) + data_idx_st);
    }
  }

  assert(false && "Out Of Memory Error");
  return nullptr;
}

void Column::insertElemBatch(rowid_t vid, uint16_t num_elem,
                             void* source_data) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t offset = StorageUtils::get_offset(vid);
  size_t data_idx_st = offset * unit_size;
  size_t copy_size = num_elem * this->unit_size;
  size_t data_idx_en = data_idx_st + copy_size;

  assert(pid < g_num_partitions);
  assert((data_idx_en / unit_size) < capacity_per_partition);
  assert(data_idx_en < total_size_per_partition);

  for (const auto& chunk : this->primaryData[pid]) {
    //      assert(pid == chunk.numa_id);

    if (__likely(chunk.size >= (data_idx_en + unit_size))) {
      void* dst = (void*)(((char*)chunk.data) + data_idx_st);
      std::memcpy(dst, source_data, copy_size);
      return;
    }
  }

  assert(false && "Out Of Memory Error");
}

template <typename T>
inline bool Column::UpdateInPlace(T& data_vector, size_t offset,
                                  size_t unit_size, void* data) {
  offset = offset * unit_size;
  // bool second_idx = false;

  for (const auto& chunk : data_vector) {
    //    if(second_idx){
    //      LOG(INFO) << "Offset: " << offset;
    //      LOG(INFO) << "Chunk Size: " << chunk.size;
    //      LOG(INFO) << "X: " << offset + unit_size;
    //    }
    if (__likely(chunk.size >= (offset + unit_size))) {
      void* dst = (void*)(((char*)chunk.data) + offset);

      char* src_t = (char*)chunk.data;
      char* dst_t = (char*)dst;

      assert(src_t <= dst_t);
      assert((src_t + chunk.size) >= (dst_t + unit_size));

      // assert(elem != nullptr);
      if (__unlikely(data == nullptr)) {
        // YCSB update hack.
        (*((uint64_t*)dst_t))++;
      } else {
        std::memcpy(dst, data, unit_size);
      }

      return true;
    }
    //    LOG(INFO) << "__Offset: " << offset;
    //    LOG(INFO) << "__(chunk.size): " << chunk.size;
    offset -= chunk.size;
    //    LOG(INFO) << "__Offset2: " << offset;
    LOG(INFO) << "Going to second index";
    //    second_idx = true;
  }
  return false;
}

}  // namespace storage
