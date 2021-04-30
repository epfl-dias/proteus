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

#ifndef STORAGE_TABLE_HPP_
#define STORAGE_TABLE_HPP_

#include <cassert>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <olap/values/expressionTypes.hpp>
#include <platform/util/percentile.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "oltp/common/constants.hpp"
#include "oltp/snapshot/snapshot_manager.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"
#include "oltp/storage/schema.hpp"

namespace storage {

class Table {
 public:
  virtual global_conf::IndexVal *insertRecord(const void *data,
                                              xid_t transaction_id,
                                              partition_id_t partition_id,
                                              master_version_t master_ver) = 0;
  virtual global_conf::IndexVal *insertRecordBatch(
      const void *data, size_t num_records, size_t max_capacity,
      xid_t transaction_id, partition_id_t partition_id,
      master_version_t master_ver) = 0;

  virtual void updateRecord(xid_t transaction_id,
                            global_conf::IndexVal *index_ptr, void *data,
                            delta_id_t current_delta_id,
                            const column_id_t *col_idx, short num_columns,
                            master_version_t master_ver) = 0;

  virtual void updateRollback(const txn::TxnTs &txnTs,
                              global_conf::IndexVal *index_ptr,
                              const column_id_t *col_idx,
                              const short num_columns) = 0;

  virtual void updateRecordBatch(xid_t transaction_id,
                                 global_conf::IndexVal *index_ptr, void *data,
                                 size_t num_records,
                                 delta_id_t current_delta_id,
                                 const column_id_t *col_idx, short num_columns,
                                 master_version_t master_ver) = 0;

  virtual void deleteRecord(xid_t transaction_id,
                            global_conf::IndexVal *index_ptr,
                            master_version_t master_ver) = 0;

  virtual void getIndexedRecord(const txn::TxnTs &txnTs,
                                const global_conf::IndexVal &index_ptr,
                                void *destination, const column_id_t *col_idx,
                                short num_cols) = 0;

  virtual void getRecord(const txn::TxnTs &txnTs, rowid_t rowid,
                         void *destination, const column_id_t *col_idx,
                         short num_cols) = 0;

  virtual void createVersion(xid_t transaction_id,
                             global_conf::IndexVal *index_ptr,
                             delta_id_t current_delta_id,
                             const column_id_t *col_idx,
                             const short num_columns) = 0;

  virtual void updateRecordWithoutVersion(
      xid_t transaction_id, global_conf::IndexVal *index_ptr, void *data,
      delta_id_t current_delta_id, const column_id_t *col_idx,
      const short num_columns, master_version_t master_ver) = 0;

  // virtual std::vector<vid_t> getNumberOfRecords(xid_t epoch) = 0;

  // Snapshot / HTAP
  //  virtual void twinColumn_snapshot(
  //      xid_t epoch, master_version_t snapshot_master_version) = 0;
  virtual void twinColumn_syncMasters(master_version_t master_idx) = 0;

  virtual void ETL(uint numa_affinity_idx) = 0;

  virtual void snapshot(xid_t epoch, column_id_t columnId) = 0;
  virtual void snapshot(xid_t epoch) = 0;

 protected:
  Table(table_id_t table_id, std::string &name, layout_type storage_layout,
        TableDef &columns);
  virtual ~Table();

 public:
  const std::string name;
  const table_id_t table_id;
  const layout_type storage_layout;

  global_conf::PrimaryIndex<uint64_t> *p_index{};

  // global_conf::PrimaryIndex<uint64_t> **s_index;

 protected:
  std::deque<std::atomic<rowid_t>> vid{};
  DeltaStore **deltaStore{};

  uint64_t total_memory_reserved;
  size_t record_size{};
  uint64_t record_capacity{};

  partition_id_t n_partitions{};
  column_id_t n_columns{};
  bool indexed{};

  std::vector<std::pair<uint16_t, uint16_t>> column_size_offset_pairs{};
  std::vector<uint16_t> column_size_offsets{};
  std::vector<uint16_t> column_size{};

 public:
  // Utilities
  void reportUsage();
  static ExpressionType *getProteusType(const ColumnDef &column);
  [[nodiscard]] inline auto numColumns() const { return this->n_columns; }
  [[nodiscard]] inline auto numPartitions() const { return this->n_partitions; }
  [[nodiscard]] inline auto isIndexed() const { return indexed; }

  friend class Schema;
};

};  // namespace storage

#endif /* STORAGE_TABLE_HPP_ */
