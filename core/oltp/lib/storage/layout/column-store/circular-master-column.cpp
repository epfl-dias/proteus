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
#include <cmath>
#include <cstring>
#include <iostream>
#include <olap/values/expressionTypes.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/thread.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <string>
#include <utility>

#include "oltp/common/constants.hpp"
#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace storage {

#define HTAP_UPD_BIT_ON_INSERT false

//-----------------------------------------------------------------
// Column
//-----------------------------------------------------------------

CircularMasterColumn::~CircularMasterColumn() {
  for (auto& master_version : master_versions) {
    for (auto j = 0; j < g_num_partitions; j++) {
      for (auto& chunk : master_version[j]) {
        MemoryManager::freePinned(chunk.data);
      }
      master_version[j].clear();
    }
  }

#if HTAP_ETL
  for (auto j = 0; j < this->n_partitions; j++) {
    for (auto& r : readonly_etl_snapshot[j]) {
      MemoryManager::freePinned(r.data);
    }
    readonly_etl_snapshot[j].clear();
  }
#endif
}

CircularMasterColumn::CircularMasterColumn(
    column_id_t column_id, std::string name, data_type type, size_t unit_size,
    size_t offset_inRecord, bool numa_partitioned, size_t reserved_capacity,
    int numa_idx)
    : Column(SnapshotTypes::CircularMaster, column_id, name, type, unit_size,
             offset_inRecord, numa_partitioned) {
  this->capacity = reserved_capacity;
  this->capacity_per_partition =
      (reserved_capacity / n_partitions) + (reserved_capacity % n_partitions);

  this->total_size_per_partition = capacity_per_partition * unit_size;
  this->total_size = this->total_size_per_partition * this->n_partitions;
  this->total_size *= global_conf::num_master_versions;

  // snapshot arenas
  for (uint j = 0; j < g_num_partitions; j++) {
    snapshot_arenas[j].emplace_back(
        std::make_unique<aeolus::snapshot::CircularMasterArenaV2>());

    snapshot_arenas[j][0]->create_snapshot(
        {0, 0, 0, static_cast<uint8_t>(j), false});

#if HTAP_ETL
    etl_arenas[j].emplace_back(
        std::make_unique<aeolus::snapshot::CircularMasterArenaV2>());
    etl_arenas[j][0]->create_snapshot({0, 0, 0, static_cast<uint8_t>(j), true});
#endif
  }

#if HTAP_ETL
  // essentially this is memory allocation for OLAP where it will do the ETL
  // and update its snapshots.
  for (auto j = 0; j < this->n_partitions; j++) {
    if (g_num_partitions == 1) {
      this->readonly_etl_snapshot[j].emplace_back(
          MemoryManager::mallocPinnedOnNode(this->total_size_per_partition,
                                            global_conf::DEFAULT_OLAP_SOCKET),
          this->total_size_per_partition, global_conf::DEFAULT_OLAP_SOCKET);
      // assert((total_numa_nodes - j - 1) == 1);
      // MemoryManager::alloc_shm(name + "__" + std::to_string(j),
      //                          size_per_partition, total_numa_nodes - j - 1);
      assert(this->readonly_etl_snapshot[j][0].data != nullptr);
    }
  }
#endif

  std::vector<proteus::thread> loaders;

  for (auto i = 0; i < global_conf::num_master_versions; i++) {
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

      master_versions[i][j].emplace_back(
          mem, this->total_size_per_partition,
          storage::NUMAPartitionPolicy::getInstance()
              .getPartitionInfo(j)
              .numa_idx);

      size_t num_bit_packs = (capacity_per_partition / BIT_PACK_SIZE) +
                             (capacity_per_partition % BIT_PACK_SIZE);

      loaders.emplace_back([this, i, j, num_bit_packs]() {
        set_exec_location_on_scope d{
            topology::getInstance()
                .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                       .getPartitionInfo(j)
                                       .numa_idx]};

        for (uint64_t bb = 0; bb < num_bit_packs; bb++) {
          upd_bit_masks[i][j].emplace_back();
        }

        for (auto& bb : this->upd_bit_masks[i][j]) {
          bb.reset();
        }
      });
    }
  }

  for (auto& th : loaders) {
    th.join();
  }

  for (auto i = 0; i < this->n_partitions; i++) this->touched[i] = false;
}

void CircularMasterColumn::initializeMetaColumn() const {
  throw std::runtime_error("Column type META should not be HTAP Column. ");
}

/*  DML Functions
 *
 */

void CircularMasterColumn::getElem(rowid_t vid, void* copy_location) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  master_version_t m_ver = StorageUtils::get_m_version(vid);
  size_t data_idx = StorageUtils::get_offset(vid) * unit_size;

  assert(!(master_versions[m_ver][pid].empty()));

  for (const auto& chunk : master_versions[m_ver][pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + unit_size))) {
      std::memcpy(copy_location, ((char*)chunk.data) + data_idx,
                  this->unit_size);
      return;
    }
  }
  assert(false && "Out-of-Bound-Access");
}

void CircularMasterColumn::updateElem(rowid_t vid, void* elem) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  master_version_t mver = StorageUtils::get_m_version(vid);
  size_t offset = StorageUtils::get_offset(vid);

  assert(pid < n_partitions);
  assert(offset < capacity_per_partition);

  if (__likely(
          UpdateInPlace(master_versions[mver][pid], offset, unit_size, elem))) {
    upd_bit_masks[mver][pid][offset / BIT_PACK_SIZE].set(offset %
                                                         BIT_PACK_SIZE);

    if (!this->touched[pid]) this->touched[pid] = true;

  } else {
    assert(false && "Out Of Memory Error");
  }
}

void CircularMasterColumn::insertElem(rowid_t vid, void* elem) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  master_version_t mver = StorageUtils::get_m_version(vid);
  size_t offset = StorageUtils::get_offset(vid);

  assert(pid < n_partitions);
  assert(offset < capacity_per_partition);

  for (auto& master_version : master_versions) {
    if (__unlikely(UpdateInPlace(master_version[pid], offset, unit_size,
                                 elem) == false)) {
      LOG(INFO) << "(1) ALLOCATE MORE MEMORY:\t" << this->name
                << ",vid: " << vid << ", idx:" << offset << ", pid: " << pid;

      assert(false && "Out Of Memory Error");
    }

#if HTAP_UPD_BIT_ON_INSERT
    if (__likely(i == mver)) {
      upd_bit_masks[mver][pid][offset / BIT_PACK_SIZE].set(offset %
                                                           BIT_PACK_SIZE);
      if (!this->touched[pid]) this->touched[pid] = true;
    }
#endif
  }
}

void CircularMasterColumn::insertElemBatch(rowid_t vid, uint16_t num_elem,
                                           void* data) {
  partition_id_t pid = StorageUtils::get_pid(vid);
  size_t offset = StorageUtils::get_offset(vid);
  size_t data_idx_st = offset * unit_size;
  size_t copy_size = num_elem * this->unit_size;
  size_t data_idx_en = data_idx_st + copy_size;

  assert(pid < g_num_partitions);
  assert((data_idx_en / unit_size) < capacity_per_partition);
  assert(data_idx_en < total_size_per_partition);

  for (auto& master_version : master_versions) {
    bool ins = false;
    for (const auto& chunk : master_version[pid]) {
      //      assert(pid == chunk.numa_id);

      if (__likely(chunk.size >= (data_idx_en + unit_size))) {
        void* dst = (void*)(((char*)chunk.data) + data_idx_st);
        std::memcpy(dst, data, copy_size);

        ins = true;
        break;
      }
    }

#if HTAP_UPD_BIT_ON_INSERT
    if (__likely(i == mver)) {
      for (auto st = 0; st < num_elem; st++) {
        upd_bit_masks[mver][pid][(offset + st) / BIT_PACK_SIZE].set(
            (offset + st) % BIT_PACK_SIZE);
      }

      if (!this->touched[pid]) this->touched[pid] = true;
    }
#endif

    if (__unlikely(ins == false)) {
      assert(false && "Out Of Memory Error");
    }
  }
}

/*  Snapshotting Functions
 *
 */

size_t CircularMasterColumn::num_upd_tuples(master_version_t master_ver,
                                            const size_t* num_records,
                                            bool print) {
  size_t counter = 0;
  for (int j = 0; j < g_num_partitions; j++) {
    if (touched[j] == false) continue;

    if (__likely(num_records != nullptr)) {
      size_t recs_scanned = 0;
      for (auto& chunk : upd_bit_masks[master_ver][j]) {
        counter += chunk.count(std::memory_order_acquire);
        recs_scanned += BIT_PACK_SIZE;
        if (recs_scanned >= num_records[j]) {
          break;
        }
      }

    } else {
      for (auto& chunk : upd_bit_masks[master_ver][j]) {
        counter += chunk.count();
      }
    }
  }

  if (__unlikely(print) && counter > 0) {
    LOG(INFO) << "UPDATED[" << master_ver << "]: COL:" << this->name
              << " | #num_upd: " << counter;
  }
  return counter;
}

std::vector<std::pair<oltp::common::mem_chunk, size_t>>
CircularMasterColumn::snapshot_get_data(bool olap_local,
                                        bool elastic_scan) const {
  std::vector<std::pair<oltp::common::mem_chunk, size_t>> ret;

  for (uint i = 0; i < n_partitions; i++) {
    assert(master_versions[0][i].size() == 1);
    if (olap_local) {
      LOG(INFO) << "OLAP_LOCAL Requested: ";
      assert(HTAP_ETL && "OLAP local mode is not turned on");
      const auto& olap_arena = etl_arenas[i][0]->getMetadata();

      LOG(INFO) << "[SnapshotData][" << this->name << "][P:" << i
                << "] Mode: LOCAL "
                << ((double)(olap_arena.numOfRecords * this->unit_size)) /
                       (1024 * 1024 * 1024)
                << " GB";

      ret.emplace_back(std::make_pair(
          oltp::common::mem_chunk(this->readonly_etl_snapshot[i][0].data,
                                  olap_arena.numOfRecords * this->unit_size,
                                  -1),
          olap_arena.numOfRecords));

    } else {
      const auto& snap_arena = snapshot_arenas[i][0]->getMetadata();
      assert(master_versions[snap_arena.master_ver][i].size() == 1 &&
             "Memory expansion not supported yet.");

      LOG(INFO) << "[SnapshotData][" << this->name << "][P:" << i
                << "] Mode: REMOTE2 "
                << ((double)(snap_arena.numOfRecords * this->unit_size)) /
                       (1024 * 1024 * 1024)
                << " GB";

      ret.emplace_back(
          std::make_pair(master_versions[snap_arena.master_ver][i][0],
                         snap_arena.numOfRecords));
    }

    // for (const auto& chunk : master_versions[snap_arena.master_ver][i]) {
    //   if (olap_local) {
    //     assert(HTAP_ETL && "OLAP local mode is not turned on");
    //     ret.emplace_back(std::make_pair(
    //         mem_chunk(this->etl_mem[i],
    //                   snap_arena.numOfRecords * this->unit_size, -1),
    //         snap_arena.numOfRecords));
    //   } else {
    //     ret.emplace_back(std::make_pair(chunk, snap_arena.numOfRecords));
    //   }
    // }
  }

  return ret;
}

std::vector<std::pair<oltp::common::mem_chunk, size_t>>
CircularMasterColumn::elastic_partition(uint pid,
                                        std::set<size_t>& segment_boundaries) {
  // tuple: <mem_chunk, num_records>, offset

  std::vector<std::pair<oltp::common::mem_chunk, size_t>> ret;

  assert(master_versions[0][pid].size() == 1);
  assert(g_num_partitions == 1);

  const auto& snap_arena = snapshot_arenas[pid][0]->getMetadata();
  const auto& olap_arena = etl_arenas[pid][0]->getMetadata();

  if (snap_arena.upd_since_last_snapshot ||
      olap_arena.upd_since_last_snapshot) {
    // update-elastic-case
    // second-cond: txn-snapshot was updated as somepoint so not safe to
    // read from local storage In this case, all pointers should be txn.

    assert(master_versions[snap_arena.master_ver][pid].size() == 1 &&
           "Memory expansion not supported yet.");

    LOG(INFO) << "[SnapshotData][" << this->name << "][P:" << pid
              << "] Mode: ELASTIC-REMOTE "
              << ((double)(snap_arena.numOfRecords * this->unit_size)) /
                     (1024 * 1024 * 1024);

    ret.emplace_back(
        std::make_pair(master_versions[snap_arena.master_ver][pid][0],
                       snap_arena.numOfRecords));

  } else {
    if (snap_arena.numOfRecords == olap_arena.numOfRecords) {
      time_block t("Tcs:");
      // safe to read from local storage
      assert(HTAP_ETL && "OLAP local mode is not turned on");

      LOG(INFO) << "[SnapshotData][" << this->name << "][P:" << pid
                << "] Mode: ELASTIC-LOCAL "
                << ((double)(olap_arena.numOfRecords * this->unit_size)) /
                       (1024 * 1024 * 1024);
      ret.emplace_back(std::make_pair(
          oltp::common::mem_chunk(this->readonly_etl_snapshot[pid][0].data,
                                  olap_arena.numOfRecords * this->unit_size,
                                  -1),
          olap_arena.numOfRecords));

    } else if (snap_arena.numOfRecords > olap_arena.numOfRecords) {
      LOG(INFO) << "[SnapshotData][" << this->name << "][P:" << pid
                << "] Mode: HYBRID";

      assert(HTAP_ETL && "OLAP local mode is not turned on");
      // new records, safe to do local + tail

      size_t diff = snap_arena.numOfRecords - olap_arena.numOfRecords;
      // local-part
      ret.emplace_back(std::make_pair(
          oltp::common::mem_chunk(this->readonly_etl_snapshot[pid][0].data,
                                  olap_arena.numOfRecords * this->unit_size,
                                  -1),
          olap_arena.numOfRecords));

      segment_boundaries.insert(olap_arena.numOfRecords);

      // tail-part
      assert(diff <= master_versions[snap_arena.master_ver][pid][0].size);

      char* oltp_mem =
          (char*)master_versions[snap_arena.master_ver][pid][0].data;
      oltp_mem += olap_arena.numOfRecords * this->unit_size;

      ret.emplace_back(std::make_pair(
          oltp::common::mem_chunk(oltp_mem, diff * this->unit_size, -1), diff));

    } else {
      assert(false && "Delete now supported, how it can be here??");
    }
  }

  return ret;
}

void CircularMasterColumn::ETL(uint numa_affinity_idx) {
  // TODO: ETL with respect to the bit-mask.
  set_exec_location_on_scope d{
      topology::getInstance().getCpuNumaNodes()[numa_affinity_idx]};

  for (uint i = 0; i < this->n_partitions; i++) {
    // zero assume no runtime column expansion
    const auto& snap_arena = snapshot_arenas[i][0]->getMetadata();
    const auto& olap_arena = etl_arenas[i][0]->getMetadata();
    const auto olap_num_rec = olap_arena.numOfRecords;
    // book-keeping for etl-data
    etl_arenas[i][0]->create_snapshot(
        {snap_arena.numOfRecords, snap_arena.epoch_id, snap_arena.master_ver,
         snap_arena.partition_id, false});

    const auto& chunk = master_versions[snap_arena.master_ver][i][0];

    // this shouldnt be snap_arena as it may not be updted since last snapshot
    // in oltp snap but since last etl, yes.
    if (snap_arena.upd_since_last_snapshot) {
      for (size_t msk = 0; msk < upd_bit_masks[snap_arena.master_ver][i].size();
           msk++) {
        if (msk * BIT_PACK_SIZE >= olap_num_rec) break;

        if (upd_bit_masks[snap_arena.master_ver][i][msk].any(
                std::memory_order_acquire)) {
          size_t to_cpy = BIT_PACK_SIZE * this->unit_size;
          size_t st = msk * to_cpy;

          if (__likely(st + to_cpy <= chunk.size)) {
            memcpy((char*)(readonly_etl_snapshot[i][0].data) + st,
                   (char*)chunk.data + st, to_cpy);
          } else {
            memcpy((char*)(readonly_etl_snapshot[i][0].data) + st,
                   (char*)chunk.data + st, chunk.size - st);
          }

          upd_bit_masks[snap_arena.master_ver][i][msk].reset(
              std::memory_order_release);
        }
      }
    }

    if (__likely(snap_arena.numOfRecords > olap_num_rec)) {
      //      LOG(INFO) << "ETL-" << this->name << " | inserted records: "
      //                << (snap_arena.numOfRecords - olap_num_rec) << ", Size:
      //                "
      //                << (double)((snap_arena.numOfRecords - olap_num_rec) *
      //                            this->unit_size) /
      //                       (1024 * 1024 * 1024);
      size_t st = olap_num_rec * this->unit_size;
      size_t to_cpy =
          (snap_arena.numOfRecords - olap_num_rec) * this->unit_size;
      memcpy(((char*)(readonly_etl_snapshot[i][0].data)) + st,
             ((char*)chunk.data) + st, to_cpy);
    }

    // for (const auto& chunk : master_versions[snap_arena.master_ver][i]) {
    //   if (snap_arena.upd_since_last_snapshot) {
    //     memcpy(etl_mem[i], chunk.data,
    //            snap_arena.numOfRecords * this->unit_size);
    //   }
    // }
  }
}

void CircularMasterColumn::snapshot(const rowid_t* num_rec_per_part,
                                    xid_t epoch) {
  master_version_t snapshot_master_ver =
      txn::TransactionManager::getInstance().get_snapshot_masterVersion(epoch);
  for (auto i = 0; i < g_num_partitions; i++) {
    assert(snapshot_arenas[i].size() == 1);

    snapshot_arenas[i][0]->create_snapshot(
        {num_rec_per_part[i], epoch, snapshot_master_ver,
         static_cast<uint8_t>(i), this->touched[i]});

#if HTAP_ETL
    if (this->touched[i]) etl_arenas[i][0]->setUpdated();
#endif

    this->touched[i] = false;
  }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

void CircularMasterColumn::syncSnapshot(master_version_t inactive_master_idx) {
  assert(global_conf::num_master_versions > 1);

  for (auto i = 0; i < global_conf::num_master_versions; i++) {
    if (i == inactive_master_idx) continue;
    for (auto j = 0; j < this->n_partitions; j++) {
      // std::cout << "sync: p_id: " << j << std::endl;
      assert(master_versions[inactive_master_idx][j].size() ==
             master_versions[i][j].size());

      // assert(upd_bit_masks[master_ver_idx][j].size() ==
      //        upd_bit_masks[i][j].size());

      if (master_versions[i][j].size() != 1) {
        LOG(INFO) << this->name;
        LOG(INFO) << master_versions[i][j].size();
        LOG(INFO) << "i: " << i;
        LOG(INFO) << "j: " << j;
      }

      assert(master_versions[i][j].size() == 1 &&
             "Expandable memory not supported");
      // assert(snapshot_arenas[i][j].size() == 1 &&
      //        "Expandable memory not supported");

      const auto& dst = master_versions[i][j][0];
      const auto& src = master_versions[inactive_master_idx][j][0];
      assert(dst.size == src.size);

      const auto& snap_arena = snapshot_arenas[j][0]->getMetadata();

      if (snap_arena.numOfRecords == 0 || !snap_arena.upd_since_last_snapshot)
        continue;

      const uint8_t* actv_ptr = (uint8_t*)dst.data;
      const uint8_t* src_ptr = (uint8_t*)src.data;

      // std::cout << "Total masks: " << upd_bit_masks[master_ver_idx][j].size()
      //           << std::endl;
      // std::cout << "Total SnapRec: " << snap_arena.numOfRecords << std::endl;

      for (auto msk = 0; msk < upd_bit_masks[inactive_master_idx][j].size();
           msk++) {
        // std::cout << "msk: " << msk << std::endl;
        const auto& src_msk = upd_bit_masks[inactive_master_idx][j][msk];
        const auto& actv_msk = upd_bit_masks[i][j][msk];

        if ((msk * BIT_PACK_SIZE) > snap_arena.numOfRecords) break;

        if (!src_msk.any() || actv_msk.all()) continue;

        for (auto bb = 0; bb < BIT_PACK_SIZE; bb++) {
          size_t data_idx = (msk * BIT_PACK_SIZE) + bb;

          // scan only the records snapshotted, not everything.
          if (data_idx > snap_arena.numOfRecords) break;

          if (src_msk.test(bb) && !actv_msk.test(bb)) {
            // do the sync

            size_t mem_idx = data_idx * unit_size;
            assert(mem_idx < dst.size);
            assert(mem_idx < src.size);

            switch (this->unit_size) {
              case 1: {  // uint8_t
                uint8_t old_val = (*(actv_ptr + mem_idx));
                uint8_t new_val = (*(src_ptr + mem_idx));
                uint8_t* dst_ptr = (uint8_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst_ptr, old_val, new_val);
                break;
              }
              case 2: {  // uint16_t
                uint16_t old_val = *((uint16_t*)(actv_ptr + mem_idx));
                uint16_t new_val = *((uint16_t*)(src_ptr + mem_idx));
                uint16_t* dst_ptr = (uint16_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst_ptr, old_val, new_val);
                break;
              }
              case 4: {  // uint32_t
                uint32_t old_val = *((uint32_t*)(actv_ptr + mem_idx));
                uint32_t new_val = *((uint32_t*)(src_ptr + mem_idx));
                uint32_t* dst_ptr = (uint32_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst_ptr, old_val, new_val);
                break;
              }
              case 8: {  // uint64_t
                uint64_t old_val = *((uint64_t*)(actv_ptr + mem_idx));
                uint64_t new_val = *((uint64_t*)(src_ptr + mem_idx));
                uint64_t* dst_ptr = (uint64_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst_ptr, old_val, new_val);
                break;
              }
              default: {
                LOG(INFO) << "Unsupported column width for snapshotting: "
                          << this->name;
                throw std::runtime_error(
                    "Unsupported column width for snapshotting: " + this->name);
              }
            }
          }
        }
      }
    }
  }
}

#pragma clang diagnostic pop

}  // namespace storage
