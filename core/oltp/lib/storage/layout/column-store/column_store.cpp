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

#include "oltp/storage/layout/column_store.hpp"

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
#include "oltp/storage/multi-version/delta_storage.hpp"
#include "oltp/storage/multi-version/mv.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction.hpp"

namespace storage {

static std::mutex print_mutex;

//-----------------------------------------------------------------
// ColumnStore
//-----------------------------------------------------------------

ColumnStore::~ColumnStore() {
  if (metaColumn) {
    metaColumn->~Column();
    MemoryManager::freePinned(metaColumn);
  }
  delete p_index;
}

ColumnStore::ColumnStore(table_id_t table_id, std::string name,
                         TableDef columns, bool indexed, bool numa_partitioned,
                         size_t reserved_capacity, int numa_idx,
                         size_t max_partition_size)
    : Table(table_id, name, COLUMN_STORE, columns)
//      ,columns(
//          proteus::memory::ExplicitSocketPinnedMemoryAllocator<std::unique_ptr<storage::Column>>(
//              storage::NUMAPartitionPolicy::getInstance()
//                  .getDefaultPartition()))
{
  this->indexed = indexed;
  this->deltaStore = storage::Schema::getInstance().deltaStore;
  this->n_columns = columns.size();
  this->n_partitions = (numa_partitioned ? g_num_partitions : 1);

  assert(g_num_partitions <= topology::getInstance().getCpuNumaNodeCount());

  std::vector<proteus::thread> loaders;

  // MetaColumn.
  void* obj_ptr = MemoryManager::mallocPinnedOnNode(
      sizeof(Column),
      storage::NUMAPartitionPolicy::getInstance().getDefaultPartition());

  metaColumn = new (obj_ptr)
      Column(0, name + "_meta", META, sizeof(global_conf::IndexVal), 0,
             numa_partitioned, reserved_capacity, numa_idx);

  assert(metaColumn);
  loaders.emplace_back([this]() { this->metaColumn->initializeMetaColumn(); });

  // If Indexed, register index.
  if (indexed) {
    if (max_partition_size > 0) {
      this->p_index = new global_conf::PrimaryIndex<uint64_t>(
          name, max_partition_size, reserved_capacity);
    } else {
      this->p_index =
          new global_conf::PrimaryIndex<uint64_t>(name, reserved_capacity);
    }

    // TODO: register index w/ IndexManager
  }

  // create columns
  column_id_t col_id_ctr = 0;
  size_t col_offset = 0;
  size_t rec_size = 0;
  this->columns.reserve(columns.getColumns().size());

  for (const auto& t : columns.getColumns()) {
    auto col_width = t.getSize();

    switch (t.getSnapshotType()) {
      case SnapshotTypes::None:
        //        this->columns.emplace_back(std::make_unique<storage::Column>(
        //            col_id_ctr++, t.getName(), t.getType(), col_width,
        //            col_offset, numa_partitioned, reserved_capacity,
        //            numa_idx));

        this->columns.emplace_back(new storage::Column(
            col_id_ctr++, t.getName(), t.getType(), col_width, col_offset,
            numa_partitioned, reserved_capacity, numa_idx));
        break;
      case SnapshotTypes::CircularMaster:
        //        this->columns.emplace_back(
        //            std::make_unique<storage::CircularMasterColumn>(
        //                col_id_ctr++, t.getName(), t.getType(), col_width,
        //                col_offset, numa_partitioned, reserved_capacity,
        //                numa_idx));

        this->columns.emplace_back(new storage::CircularMasterColumn(
            col_id_ctr++, t.getName(), t.getType(), col_width, col_offset,
            numa_partitioned, reserved_capacity, numa_idx));
        break;
      case SnapshotTypes::LazyMaster:
        //        this->columns.emplace_back(std::make_unique<storage::LazyColumn>(
        //            col_id_ctr++, t.getName(), t.getType(), col_width,
        //            col_offset, numa_partitioned, reserved_capacity,
        //            numa_idx));

        this->columns.emplace_back(new storage::LazyColumn(
            col_id_ctr++, t.getName(), t.getType(), col_width, col_offset,
            numa_partitioned, reserved_capacity, numa_idx));
        break;
      default:
        throw std::runtime_error(
            "Unknown snapshotting mechanism in ColumnDef for column: " +
            t.getName());
    }

    column_size.push_back(col_width);
    column_size_offsets.push_back(col_offset);
    column_size_offset_pairs.emplace_back(col_width, col_offset);
    col_offset += col_width;
    rec_size += col_width;
  }

  for (const auto& t : this->columns) {
    total_memory_reserved += t->total_size;
  }

  this->record_size = rec_size;
  assert(rec_size == col_offset);
  this->record_capacity = reserved_capacity;

  for (auto& th : loaders) {
    th.join();
  }

  if (indexed) total_memory_reserved += metaColumn->total_size;

  LOG(INFO) << "\n\tTable: " << name << "\n\ttable_id: " << (uint)this->table_id
            << "\n\trecord size: " << rec_size << " bytes"
            << "\n\tnum_records: " << reserved_capacity << "\n\tMem reserved: "
            << (double)total_memory_reserved / (1024 * 1024 * 1024) << " GB";

  elastic_mappings.reserve(columns.size());
}

void ColumnStore::steamGC(const std::set<vid_t>& row_vector,
                          txn::TxnTs globalMin) {
  for (const auto& vid : row_vector) {
    auto idxPtr = (global_conf::IndexVal*)(this->metaColumn->getElem(vid));
    mv::mv_type::gc(idxPtr, globalMin);
  }
}

/* ColumnStore::insertRecordBatch assumes that the  void* rec has columns in the
 * same order as the actual columns
 */
global_conf::IndexVal* ColumnStore::insertRecordBatch(
    const void* data, size_t num_records, size_t max_capacity,
    xid_t transaction_id, partition_id_t partition_id,
    master_version_t master_ver) {
  partition_id = partition_id % this->n_partitions;
  uint64_t idx_st = vid[partition_id].vid_part.fetch_add(num_records);
  // get batch from meta column

  uint64_t st_vid = StorageUtils::create_vid(idx_st, partition_id, master_ver);
  uint64_t st_vid_meta = StorageUtils::create_vid(idx_st, partition_id, 0);

  global_conf::IndexVal* hash_ptr = nullptr;
  if (this->indexed) {
    // meta stuff
    hash_ptr = (global_conf::IndexVal*)this->metaColumn->insertElemBatch(
        st_vid_meta, num_records);
    assert(hash_ptr != nullptr);

    for (uint i = 0; i < num_records; i++) {
      hash_ptr[i].t_min = transaction_id;
      hash_ptr[i].VID =
          StorageUtils::create_vid(idx_st + i, partition_id, master_ver);
    }
  }

  // for loop to copy all columns.
  for (auto& col : columns) {
    col->insertElemBatch(
        st_vid, num_records,
        ((char*)data) + (col->byteOffset_record *
                         (max_capacity == 0 ? num_records : max_capacity)));
  }

  // return starting address of batch meta.
  return hash_ptr;
}

global_conf::IndexVal* ColumnStore::insertRecord(const void* data,
                                                 xid_t transaction_id,
                                                 partition_id_t partition_id,
                                                 master_version_t master_ver) {
#if INSTRUMENTATION
  static thread_local proteus::utils::threadLocal_percentile ins_cdf(
      "columnStore-InsertRecord");
  proteus::utils::percentile_point cd(ins_cdf);
#endif

  partition_id = partition_id % this->n_partitions;
  uint64_t idx = vid[partition_id].vid_part.fetch_add(1);
  uint64_t curr_vid = StorageUtils::create_vid(idx, partition_id, master_ver);

  global_conf::IndexVal* hash_ptr = nullptr;

  if (indexed) {
    auto indexed_cc_vid = StorageUtils::create_vid(idx, partition_id, 0);
    hash_ptr =
        (global_conf::IndexVal*)this->metaColumn->getElem(indexed_cc_vid);
    assert(hash_ptr != nullptr);
    hash_ptr->t_min = transaction_id;
    hash_ptr->VID = curr_vid;
  }

  char* rec_ptr = (char*)data;
  for (auto& col : columns) {
    col->insertElem(curr_vid, rec_ptr + col->byteOffset_record);
  }

  return hash_ptr;
}

void ColumnStore::getRecord(const txn::TxnTs& txnTs, rowid_t rowid,
                            void* destination, const column_id_t* col_idx,
                            const short num_cols) {
  auto* metaPtr = (global_conf::IndexVal*)metaColumn->getElem(rowid);
  return getIndexedRecord(txnTs, *metaPtr, destination, col_idx, num_cols);
}

void ColumnStore::getIndexedRecord(const txn::TxnTs& txnTs,
                                   const global_conf::IndexVal& index_ptr,
                                   void* destination,
                                   const column_id_t* col_idx, short num_cols) {
#if INSTRUMENTATION
  static thread_local proteus::utils::threadLocal_percentile rd_cdf("read_cdf");
  static thread_local proteus::utils::threadLocal_percentile rd_mv_cdf(
      "read_mv_cdf");
  proteus::utils::percentile_point cd(rd_cdf);
  // proteus::utils::percentile_point mv_cd(rd_mv_cdf);
#endif

  char* write_loc = static_cast<char*>(destination);

  if (txn::CC_MV2PL::is_readable(index_ptr, txnTs)) {
    if (__unlikely(col_idx == nullptr)) {
      for (auto& col : columns) {
        col->getElem(index_ptr.VID, write_loc);
        write_loc += col->unit_size;
      }
    } else {
      for (auto i = 0; i < num_cols; i++) {
        auto& col = columns.at(col_idx[i]);
        col->getElem(index_ptr.VID, write_loc);
        write_loc += col->unit_size;
      }
    }
  } else {
#if INSTRUMENTATION
    proteus::utils::percentile_point mv_cd(rd_mv_cdf);
#endif
    auto done_mask = mv::mv_type::get_readable_version(
        index_ptr, txnTs, write_loc, *this, col_idx, num_cols);

    if (!done_mask.all()) {
      // LOG(INFO) << "reading from main";

      if constexpr (!storage::mv::mv_type::isPerAttributeMVList &&
                    !storage::mv::mv_type::isAttributeLevelMV) {
        assert(false && "Impossible for full-record versioning!");
      }

      // write_offset is the cumulative offset in the write_location.
      if (__unlikely(col_idx == nullptr)) {
        for (auto i = 0; i < this->n_columns; i++) {
          if (!done_mask[i]) {
            columns[i]->getElem(index_ptr.VID,
                                (write_loc + this->column_size_offsets[i]));
          }
        }
      } else {
        for (auto i = 0, write_offset = 0; i < num_cols; i++) {
          write_offset += column_size[col_idx[i]];
          if (!done_mask[i]) {
            columns[i]->getElem(index_ptr.VID, (write_loc + write_offset));
          }
        }
      }
    }
  }
}

/*
  FIXME: [Maybe] Update records create a delta version not in the local
   partition to the worker but local to record-master partition. this create
   version creation over QPI.
*/

void ColumnStore::updateRecord(const txn::Txn& txn,
                               global_conf::IndexVal* index_ptr, void* data,
                               const column_id_t* col_idx,
                               const short num_columns) {
#if INSTRUMENTATION
  static thread_local proteus::utils::threadLocal_percentile update_cdf(
      "update_cdf");
  proteus::utils::percentile_point cd(update_cdf);

#endif

  assert((num_columns > 0 && col_idx != nullptr) || num_columns <= 0);

  partition_id_t pid = StorageUtils::get_pid(index_ptr->VID);
  char* cursor = static_cast<char*>(data);

  auto old_vid = index_ptr->VID;
  if constexpr (global_conf::num_master_versions > 1) {
    index_ptr->VID =
        StorageUtils::update_mVer(index_ptr->VID, txn.master_version);
  }

  auto version_ptr =
      mv::mv_type::create_versions(txn.txnTs.txn_start_time, index_ptr, *this,
                                   *(this->deltaStore[txn.delta_version]),
                                   txn.partition_id, col_idx, num_columns);

  assert(index_ptr->delta_list.isValid());

  auto n_cols = (num_columns > 0 ? num_columns : columns.size());
  uint idx = 0;

  if constexpr (storage::mv::mv_type::isPerAttributeMVList) {
    // multiple version pointers.

    for (auto i = 0; i < n_cols; i++) {
      if (__likely(num_columns > 0)) {
        idx = col_idx[i];
      } else {
        idx = i;
      }

      auto& col = columns.at(idx);
      col->getElem(old_vid, version_ptr.at(i)->data);

      // update column
      col->updateElem(index_ptr->VID,
                      (data == nullptr ? nullptr : (void*)cursor));
      if (__likely(data != nullptr)) {
        cursor += col->unit_size;
      }
    }
  }

  if constexpr (!storage::mv::mv_type::isPerAttributeMVList &&
                storage::mv::mv_type::isAttributeLevelMV) {
    // single data pointer, but only copy specific attributes
    char* version_data_ptr = (char*)(version_ptr.at(0)->data);
    assert(version_data_ptr != nullptr);

    // LOG(INFO) << "ColumStore! " << version_ptr[0]->attribute_mask;

    for (auto i = 0; i < n_cols; i++) {
      if (__likely(num_columns > 0)) {
        idx = col_idx[i];
      } else {
        idx = i;
      }
      auto& col = columns.at(idx);

      col->getElem(old_vid, version_data_ptr);
      version_data_ptr += col->unit_size;

      // update column
      col->updateElem(index_ptr->VID,
                      (data == nullptr ? nullptr : (void*)cursor));
      if (__likely(data != nullptr)) {
        cursor += col->unit_size;
      }
    }
  }
  if constexpr (!storage::mv::mv_type::isPerAttributeMVList &&
                !storage::mv::mv_type::isAttributeLevelMV) {
    // full record version.

    // first copy entire record.
    char* version_data_ptr = (char*)(version_ptr[0]->data);
    assert(version_data_ptr != nullptr);
    // std::atomic_thread_fence(std::memory_order_seq_cst);
    LOG_IF(FATAL, version_ptr.at(0)->size < this->record_size)
        << "Tbl: " << this->name
        << " | Failed: ver-size: " << version_ptr.at(0)->size
        << " | rec: " << this->record_size;

    for (const auto& col : columns) {
      col->getElem(old_vid, version_data_ptr + col->byteOffset_record);
    }

    // do actual update
    for (auto i = 0; i < n_cols; i++) {
      if (__likely(num_columns > 0)) {
        idx = col_idx[i];
      } else {
        idx = i;
      }
      auto& col = columns[idx];
      // update column
      col->updateElem(index_ptr->VID,
                      (data == nullptr ? nullptr : (void*)cursor));
      if (__likely(data != nullptr)) {
        cursor += col->unit_size;
      }
    }
  }

  // FIXME: this should be the job of txn, not the storage.
  index_ptr->t_min = txn.txnTs.txn_start_time;
}

/*  Snapshotting Functions
 *
 */

void ColumnStore::num_upd_tuples() {
  for (uint i = 0; i < global_conf::num_master_versions; i++) {
    for (auto& col : this->columns) {
      col->num_upd_tuples(i, nullptr, true);
    }
  }
}

void ColumnStore::ETL(uint numa_affinity_idx) {
  std::vector<proteus::thread> workers;

  for (auto& col : this->columns) {
    workers.emplace_back(
        [&col, numa_affinity_idx]() { col->ETL(numa_affinity_idx); });
  }

  for (auto& th : workers) {
    th.join();
  }
}

void ColumnStore::snapshot(xid_t epoch) {
  uint64_t partitions_n_recs[global_conf::MAX_PARTITIONS];

  for (uint i = 0; i < g_num_partitions; i++) {
    partitions_n_recs[i] = this->vid[i].vid_part.load();
    LOG(INFO) << "Snapshot " << this->name << " : Records in P[" << i
              << "]: " << partitions_n_recs[i];
  }

  for (auto& col : this->columns) {
    col->snapshot(partitions_n_recs, epoch);
  }
}

void ColumnStore::snapshot(xid_t epoch, column_id_t columnId) {
  uint64_t partitions_n_recs[global_conf::MAX_PARTITIONS];

  for (uint i = 0; i < g_num_partitions; i++) {
    partitions_n_recs[i] = this->vid[i].vid_part.load();
    LOG(INFO) << "Snapshot " << this->name << " : Records in P[" << i
              << "]: " << partitions_n_recs[i];
  }

  this->columns[columnId]->snapshot(partitions_n_recs, epoch);
}

// void ColumnStore::twinColumn_snapshot(
//    xid_t epoch, master_version_t snapshot_master_version) {
//  uint64_t partitions_n_recs[global_conf::MAX_PARTITIONS];
//
//  for (uint i = 0; i < g_num_partitions; i++) {
//    partitions_n_recs[i] = this->vid[i].load();
//    LOG(INFO) << "Snapshot " << this->name << " : Records in P[" << i
//              << "]: " << partitions_n_recs[i];
//  }
//
//  for (auto& col : this->columns) {
//    col->snapshot(partitions_n_recs, epoch, snapshot_master_version);
//  }
//}

int64_t* ColumnStore::snapshot_get_number_tuples(bool olap_snapshot,
                                                 bool elastic_scan) {
  if (elastic_scan) {
    assert(g_num_partitions == 1 &&
           "cannot do it for more as of now due to static nParts");

    const auto& totalNumRecords =
        columns[0]->getSnapshotArena(0)[0]->getMetadata().numOfRecords;

    auto* arr = (int64_t*)malloc(sizeof(int64_t*) * nParts);

    size_t i = 0;
    size_t sum = 0;
    for (const auto& eoffs : elastic_offsets) {
      arr[i] = eoffs;
      sum += eoffs;
      i++;
    }

    arr[i] = totalNumRecords - sum;
    i++;

    while (i < nParts) {
      arr[i] = 0;
      i++;
    }

    // for (uint cc = 0; cc < nParts; cc++) {
    //   LOG(INFO) << this->name << " " << cc << " - " << arr[cc];
    // }

    return arr;
  } else {
    const uint num_parts = this->columns[0]->n_partitions;
    auto* arr = (int64_t*)malloc(sizeof(int64_t*) * num_parts);

    for (uint i = 0; i < num_parts; i++) {
      if (__unlikely(olap_snapshot)) {
        arr[i] =
            this->columns[0]->getETLArena(i)[0]->getMetadata().numOfRecords;
        //        LOG(INFO) << this->name
        //                  << " -- [OLAP-snapshot] NumberOfRecords:" << arr[i];
      } else {
        arr[i] = this->columns[0]
                     ->getSnapshotArena(i)[0]
                     ->getMetadata()
                     .numOfRecords;

        //        LOG(INFO) << this->name
        //                  << " -- [OLTP-snapshot] NumberOfRecords:" << arr[i];
      }
    }
    return arr;
  }
}

std::vector<std::pair<oltp::common::mem_chunk, size_t>>
ColumnStore::snapshot_get_data(size_t scan_idx,
                               std::vector<RecordAttribute*>& wantedFields,
                               bool olap_local, bool elastic_scan) {
  if (elastic_scan) {
    assert(g_num_partitions == 1 &&
           "cannot do it for more as of now due to static nParts");
    this->nParts = wantedFields.size() * wantedFields.size();
    // std::pow(wantedFields.size(), wantedFields.size());

    const auto& totalNumRecords =
        columns[0]->getSnapshotArena(0)[0]->getMetadata().numOfRecords;

    if (scan_idx == 0) {
      elastic_mappings.clear();
      elastic_offsets.clear();

      // restart

      for (size_t j = 0; j < wantedFields.size(); j++) {
        for (auto& cl : this->columns) {
          if (cl->name.compare(wantedFields[j]->getAttrName()) == 0) {
            // 0 partition id
            elastic_mappings.emplace_back(
                cl->elastic_partition(0, elastic_offsets));
          }
        }
      }

      // assert(elastic_mappings.size() > scan_idx);
      // return elastic_mappings[scan_idx];
    }

    assert(elastic_mappings.size() > scan_idx);

    auto& getit = elastic_mappings[scan_idx];

    size_t num_to_return = wantedFields.size();

    std::vector<std::pair<oltp::common::mem_chunk, size_t>> ret;

    // for (uint xd = 0; xd < getit.size(); xd++) {
    //   LOG(INFO) << "Part: " << xd << ": numa_loc: "
    //             << topology::getInstance()
    //                    .getCpuNumaNodeAddressed(getit[xd].first.data)
    //                    ->id;
    // }

    if (elastic_offsets.size() == 0) {
      // add the remaining
      ret.emplace_back(std::make_pair(getit[0].first, totalNumRecords));

      for (size_t i = 1; i < num_to_return; i++) {
        ret.emplace_back(std::make_pair(getit[0].first, 0));
      }

    } else if (getit.size() == elastic_offsets.size() + 1) {
      size_t partitions_in_place = getit.size();
      // just add others

      for (size_t i = 0; i < num_to_return; i++) {
        if (i < partitions_in_place) {
          ret.emplace_back(std::make_pair(getit[i].first, getit[i].second));
        } else {
          ret.emplace_back(std::make_pair(getit[0].first, 0));
        }
      }

    } else {
      size_t unit_size = 0;
      for (size_t j = 0; j < wantedFields.size(); j++) {
        for (const auto& cl : this->columns) {
          if (cl->name.compare(wantedFields[j]->getAttrName()) == 0) {
            unit_size = cl->unit_size;
          }
        }
      }

      assert(unit_size != 0);
      for (const auto& ofs : elastic_offsets) {
        bool added = false;
        for (auto& sy : getit) {
          if (((char*)sy.first.data + (ofs * unit_size)) <=
              ((char*)sy.first.data + sy.first.size)) {
            ret.emplace_back(std::make_pair(sy.first, ofs));
            added = true;
          }
        }
        assert(added == true);
      }
      // added all offsets, now add extra
      for (size_t i = ret.size() - 1; i < num_to_return; i++) {
        ret.emplace_back(std::make_pair(getit[0].first, 0));
      }
    }
    assert(ret.size() == num_to_return);
    return ret;
  }

  else {
    auto n = wantedFields[scan_idx]->getAttrName().find("_bitmask");
    if (n == std::string::npos) {
      for (auto& cl : this->columns) {
        if (cl->name.compare(wantedFields[scan_idx]->getAttrName()) == 0) {
          return cl->snapshot_get_data(olap_local, false);
        }
      }
    } else {
      // extract the name
      auto strSize = wantedFields[scan_idx]->getAttrName().size();
      auto colName =
          wantedFields[scan_idx]->getAttrName().substr(0, strSize - 8);
      LOG(INFO) << "Bitmask Column name extraction::::: " << colName;

      for (auto& cl : this->columns) {
        if (cl->name.compare(colName) == 0) {
          // for lazy, elastic scan means return bitmask for now.
          return cl->snapshot_get_data(olap_local, true);
        }
      }
    }

    assert(false && "Snapshot -- Unknown Column.");
  }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

void ColumnStore::twinColumn_syncMasters(master_version_t master_idx) {
  assert(global_conf::num_master_versions > 1);
  for (auto& col : this->columns) {
    if (col->snapshotType == SnapshotTypes::CircularMaster) {
      if (!(col->type == STRING || col->type == VARCHAR)) {
        col->syncSnapshot(master_idx);
      }
    }
  }
}

#pragma clang diagnostic pop

};  // namespace storage
