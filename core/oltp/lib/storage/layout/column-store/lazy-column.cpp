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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <olap/values/expressionTypes.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/thread.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <string>
#include <utility>
#include <vector>

#include "oltp/common/constants.hpp"
#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/storage/table.hpp"

namespace storage {

LazyColumn::~LazyColumn() {
  // TODO:

  for (auto i = 0; i < MAX_PARTITIONS; i++) {
    // clean-up primary.
    // primary would be cleaned by the base-class.

    // clean-up secondaries.
    for (auto& sec : secondary[i]) {
      sec.~LazySecondary();
    }
    secondary[i].clear();
  }
}

LazyColumn::LazyColumn(column_id_t column_id, std::string name, data_type type,
                       size_t unit_size, size_t offset_inRecord,
                       bool numa_partitioned, size_t reserved_capacity,
                       int numa_idx)
    : Column(column_id, name, type, unit_size, offset_inRecord,
             numa_partitioned, reserved_capacity, numa_idx,
             SnapshotTypes::LazyMaster) {
  // the above constructor intializes the primaryData.
  // for LazyColumn, extra initialization requires just the bitmasks only.

  size_t num_bit_packs = (capacity_per_partition / BIT_PACK_SIZE) +
                         (capacity_per_partition % BIT_PACK_SIZE);

  std::vector<proteus::thread> loaders;

  for (auto j = 0; j < this->n_partitions; j++) {
    loaders.emplace_back([this, j, num_bit_packs]() {
      set_exec_location_on_scope d{
          topology::getInstance()
              .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                     .getPartitionInfo(j)
                                     .numa_idx]};

      for (auto bb = 0; bb < num_bit_packs; bb++) {
        dirty_upd_mask[j].emplace_back();
        dirty_delete_mask[j].emplace_back();

        dirty_upd_mask[j][bb].reset(std::memory_order::relaxed);
        dirty_delete_mask[j][bb].reset(std::memory_order::relaxed);
      }
    });
  }

  for (auto& th : loaders) {
    th.join();
  }

  for (auto i = 0; i < this->n_partitions; i++) this->touched[i] = false;

  this->active_snapshot_idx.store(0);
}

// Ops

static inline void* getElemByOffset(std::deque<oltp::common::mem_chunk>& data,
                                    size_t unit_size, size_t offset) {
  size_t data_idx = offset * unit_size;
  for (const auto& chunk : data) {
    if (__likely(chunk.size >= ((size_t)data_idx + unit_size))) {
      return ((char*)chunk.data) + data_idx;
    }
  }
  assert(false && "Out-of-Bound-Access");
}

void* LazyColumn::getElem(rowid_t vid) {
  auto snap_id = rid_snapshot_map.getValue(vid);
  partition_id_t pid = StorageUtils::get_pid(vid);

  if (__likely(snap_id == 0)) {
    // primary-ver
    return Column::getElem(vid);
  } else {
    // different snapshot
    // snap_id - 1 to account for primary
    auto& sec = secondary[pid][snap_id - 1];
    auto internal_vid = sec.sec_idx.find(vid);
    // get element from secondary.
    return getElemByOffset(sec.data, unit_size, internal_vid);
  }
}

void LazyColumn::getElem(rowid_t vid, void* copy_destination) {
  auto* data_ptr = getElem(vid);
  memcpy(copy_destination, data_ptr, unit_size);
}

void LazyColumn::updateElem(rowid_t vid, void* data) {
  partition_id_t pid = StorageUtils::get_pid(vid);

  snapshot_version_t snapIdx =
      active_snapshot_idx.load(std::memory_order_acquire);
  if (snapIdx == 0) {
    // primary snapshot

    // primary.
    //    1) update like normal-column

    size_t offset = StorageUtils::get_offset(vid);
    if (!UpdateInPlace(this->primaryData[pid], offset, unit_size, data)) {
      // FIXME: this should trigger the call to increase capacity.
      assert(false && "couldn't insert into primary due to space.");
    }

    //    2) set dirty-bitmask for this index.
    dirty_upd_mask[pid][offset / BIT_PACK_SIZE].set(offset % BIT_PACK_SIZE,
                                                    std::memory_order_release);
    if (!this->touched[pid]) this->touched[pid] = true;

  } else {
    // secondary snapshot

    auto& sec = secondary[pid][snapIdx - 1];
    assert(sec.is_locked && "How come accessing locked secondary.");

    LazySecondary::secondary_vid_t internal_vid;
    if (sec.sec_idx.find(vid, internal_vid) == false) {
      // updating this value for the first-time.
      internal_vid = sec.secondary_vid.fetch_add(1);
      sec.sec_idx.insert(vid, internal_vid);
      if (sec.touched == false) {
        sec.touched = true;
      }
      sec.dirty_upd_mask[internal_vid / BIT_PACK_SIZE].set(
          internal_vid % BIT_PACK_SIZE, std::memory_order_release);
    }

    if (!UpdateInPlace(sec.data, internal_vid, unit_size, data)) {
      // FIXME: this should trigger the call to increase capacity.
      LOG(INFO) << "Increasing capacity of secondary.";

      {
        time_block t("T_increase_secondary_capacity");
        sec.increaseCapacity(2);
      }
      if (!UpdateInPlace(sec.data, internal_vid, unit_size, data)) {
        assert(false && "couldn't insert even after increasing capacity.");
      }
    }
  }

  // update the vid->snap mapping in interval map.
  rid_snapshot_map.upsert(vid, snapIdx);
}

// LazySecondary

LazyColumn::LazySecondary::LazySecondary(size_t capacity, size_t unit_size,
                                         partition_id_t pid)
    : is_locked(false),
      initial_capacity(capacity),
      capacity(capacity),
      unit_size(unit_size),
      pid(pid),
      touched(false) {
  // std::vector<proteus::thread> loaders;
  auto numa_idx = storage::NUMAPartitionPolicy::getInstance()
                      .getPartitionInfo(pid)
                      .numa_idx;

  // appropriate stack-variable allocation through affinity control.
  set_exec_location_on_scope d{
      topology::getInstance().getCpuNumaNodes()[numa_idx]};

  // FIXME: sec_idx might-be going to wrong-socket.
  sec_idx.reserve(capacity);
  secondary_vid.store(0);

  auto sz = capacity * unit_size;
  auto num_bit_packs = (sz / BIT_PACK_SIZE) + (sz % BIT_PACK_SIZE);

  void* mem = MemoryManager::mallocPinnedOnNode(sz, numa_idx);
  assert(mem != nullptr);

  // FIXME: Skipping warm-up!

  data.emplace_back(mem, sz, numa_idx);

  for (auto bb = 0; bb < num_bit_packs; bb++) {
    dirty_upd_mask.emplace_back();
    dirty_delete_mask.emplace_back();

    dirty_upd_mask[bb].reset(std::memory_order::relaxed);
    dirty_delete_mask[bb].reset(std::memory_order::relaxed);
  }
}

LazyColumn::LazySecondary::~LazySecondary() {
  for (auto& dmem : data) {
    MemoryManager::freePinned(dmem.data);
  }
}

void LazyColumn::LazySecondary::increaseCapacity(double factor) {
  size_t incSize = capacity * factor;
  incSize += incSize % unit_size;

  // FIXME: make sure capacity doesnt go beyond the size of primary.

  auto numa_idx = storage::NUMAPartitionPolicy::getInstance()
                      .getPartitionInfo(pid)
                      .numa_idx;

  set_exec_location_on_scope d{
      topology::getInstance().getCpuNumaNodes()[numa_idx]};

  sec_idx.reserve(capacity);

  auto n_bit_packs = (incSize / BIT_PACK_SIZE) + (incSize % BIT_PACK_SIZE);

  void* mem = MemoryManager::mallocPinnedOnNode(incSize, numa_idx);
  assert(mem != nullptr);

  data.emplace_back(mem, incSize, numa_idx);

  for (auto bb = 0; bb < n_bit_packs; bb++) {
    dirty_upd_mask.emplace_back();
    dirty_delete_mask.emplace_back();

    dirty_upd_mask[bb].reset(std::memory_order::relaxed);
    dirty_delete_mask[bb].reset(std::memory_order::relaxed);
  }

  capacity += initial_capacity;
}

}  // namespace storage
