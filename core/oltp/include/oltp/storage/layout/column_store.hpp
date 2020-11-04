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

#ifndef STORAGE_COLUMN_STORE_HPP_
#define STORAGE_COLUMN_STORE_HPP_

#include <cassert>
#include <deque>
#include <iostream>
#include <map>
#include <platform/memory/allocator.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "oltp/common/atomic_bit_set.hpp"
#include "oltp/common/common.hpp"
#include "oltp/common/constants.hpp"
#include "oltp/common/memory-chunk.hpp"
#include "oltp/storage/table.hpp"

#define BIT_PACK_SIZE 8192

class RecordAttribute;

namespace storage {

class Column;

class alignas(BlockManager::block_size) ColumnStore : public Table {
 public:
  ColumnStore(table_id_t table_id, std::string name, TableDef columns,
              bool indexed = true, bool numa_partitioned = true,
              size_t reserved_capacity = 1000000, int numa_idx = -1);

  ~ColumnStore() override;

  global_conf::IndexVal *insertRecord(const void *data, xid_t transaction_id,
                                      partition_id_t partition_id,
                                      master_version_t master_ver = 0) override;
  global_conf::IndexVal *insertRecordBatch(
      const void *data, size_t num_records, size_t max_capacity,
      xid_t transaction_id, partition_id_t partition_id,
      master_version_t master_ver = 0) override;

  void updateRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                    void *data, delta_id_t current_delta_id,
                    const column_id_t *col_idx = nullptr,
                    short num_columns = -1,
                    master_version_t master_ver = 0) override;

  [[noreturn]] void updateRecordBatch(
      xid_t transaction_id, global_conf::IndexVal *index_ptr, void *data,
      size_t num_records, delta_id_t current_delta_id,
      const column_id_t *col_idx = nullptr, short num_columns = -1,
      master_version_t master_ver = 0) override {
    throw std::runtime_error("Unimplemented");
  }

  void deleteRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                    master_version_t master_ver = 0) override {
    throw std::runtime_error("Unimplemented");
  }

  void getIndexedRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                        void *destination, const column_id_t *col_idx = nullptr,
                        short num_cols = -1) override;

  void getRecord(xid_t transaction_id, rowid_t rowid, void *destination,
                 const column_id_t *col_idx = nullptr,
                 short num_cols = -1) override;

  //------------------TwinColumn
  // TwinColumn snapshotting (TwinColumn is misleading as in theory,
  //  we can have N copies where N-1 are snapshots.
  void twinColumn_snapshot(xid_t epoch,
                           master_version_t snapshot_master_version) override;
  void twinColumn_syncMasters(master_version_t master_idx) override;

  //------------------ETL
  void ETL(uint numa_affinity_idx) override;

  // OLAP-plugin interfaces
  int64_t *snapshot_get_number_tuples(bool olap_snapshot = false,
                                      bool elastic_scan = false);
  std::vector<std::pair<oltp::common::mem_chunk, size_t>> snapshot_get_data(
      size_t scan_idx, std::vector<RecordAttribute *> &wantedFields,
      bool olap_local, bool elastic_scan);

  //-----------------Utilities
  void num_upd_tuples();

  const auto &getColumns() { return columns; }

 private:
  ColumnVector columns;
  Column *metaColumn{};

  // OLAP-TwinColumn Snapshot
  size_t nParts{};
  std::vector<std::vector<std::pair<oltp::common::mem_chunk, size_t>>>
      elastic_mappings{};
  std::set<size_t> elastic_offsets{};

  friend class Column;
};

class alignas(BlockManager::block_size) Column {
 public:
  ~Column();
  Column(column_id_t column_id, std::string name, data_type type,
         size_t unit_size, size_t offset_inRecord, bool numa_partitioned = true,
         size_t reserved_capacity = 1000000, bool single_version = false,
         int numa_idx = -1);

  Column(const Column &) = delete;
  Column(Column &&) = default;

 private:
  void *getElem(rowid_t vid);
  void getElem(rowid_t vid, void *copy_destination);
  void updateElem(rowid_t vid, void *data);
  void insertElem(rowid_t vid, void *data);
  void *insertElem(rowid_t vid);
  void *insertElemBatch(rowid_t vid, uint16_t num_elem);
  void insertElemBatch(rowid_t vid, uint16_t num_elem, void *data);
  void initializeMetaColumn() const;

 public:
  //-------Snapshotting

  //------------------TwinColumn
  // TwinColumn snapshotting (TwinColumn is misleading as in theory,
  //  we can have N copies where N-1 are snapshots.
  void twinColumn_snapshot(const rowid_t *num_rec_per_part, xid_t epoch,
                           master_version_t snapshot_master_ver);
  void twinColumn_syncMasters(master_version_t inactive_master_idx);

  //------------------ETL
  void ETL(uint numa_affinity_idx = 0);

  //------------------OLAP-plugin utilities
  [[nodiscard]] std::vector<std::pair<oltp::common::mem_chunk, size_t>>
  snapshot_get_data(bool olap_local = false, bool elastic_scan = false) const;

  std::vector<std::pair<oltp::common::mem_chunk, size_t>> elastic_partition(
      uint pid, std::set<size_t> &segment_boundaries);

  //------------------Utilities

  size_t num_upd_tuples(master_version_t master_ver, const size_t *num_records,
                        bool print);

 private:
  volatile bool touched[global_conf::MAX_PARTITIONS]{};

  std::vector<oltp::common::mem_chunk>
      master_versions[global_conf::num_master_versions]
                     [global_conf::MAX_PARTITIONS];

  // bit-mask for dirty-records.
  std::deque<utils::AtomicBitSet<BIT_PACK_SIZE>>
      upd_bit_masks[global_conf::num_master_versions]
                   [global_conf::MAX_PARTITIONS];

 public:
  // meta-data
  const std::string name;
  const column_id_t column_id;
  const size_t unit_size;
  const data_type type;
  const partition_id_t n_partitions;
  const size_t byteOffset_record;
  const bool single_version_only;

 protected:
  // stats
  size_t total_size;
  size_t total_size_per_partition;
  //  size_t total_num_blocks;
  //  size_t total_num_blocks_per_partition;
  size_t capacity;
  size_t capacity_per_partition;

  // Snapshotting Utils
  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      snapshot_arenas[global_conf::MAX_PARTITIONS];

  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      etl_arenas[global_conf::MAX_PARTITIONS];

  // FIXME: ETL memory should be owned by SnapshotManager.
  std::vector<oltp::common::mem_chunk>
      readonly_etl_snapshot[global_conf::MAX_PARTITIONS];

  friend class ColumnStore;
};

};  // namespace storage

#endif /* STORAGE_COLUMN_STORE_HPP_ */
