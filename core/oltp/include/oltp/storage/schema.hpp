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

#ifndef PROTEUS_SCHEMA_HPP
#define PROTEUS_SCHEMA_HPP

#include <cassert>
#include <future>
#include <iostream>
#include <map>
#include <platform/util/percentile.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "oltp/common/constants.hpp"
#include "oltp/snapshot/snapshot_manager.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage {

class Table;
class RowStore;
class ColumnStore;
class Column;
class DeltaStore;

enum layout_type { ROW_STORE, COLUMN_STORE };

enum data_type { META, MV, INTEGER, STRING, FLOAT, VARCHAR, DATE, DSTRING };

// using ColumnVector = std::vector<
//    storage::Column,
//    proteus::memory::ExplicitSocketPinnedMemoryAllocator<storage::Column>>;

// using ColumnVector = std::vector<std::unique_ptr<storage::Column>,
//    proteus::memory::ExplicitSocketPinnedMemoryAllocator<std::unique_ptr<storage::Column>>>;

using ColumnVector = std::vector<std::unique_ptr<storage::Column>>;

class ColumnDef {
 public:
  inline auto getName() const { return std::get<0>(col); }
  inline auto getType() const { return std::get<1>(col); }
  inline auto getWidth() const { return std::get<2>(col); }
  inline auto getSize() const { return getWidth(); }
  inline auto getSnapshotType() const { return std::get<3>(col); }
  inline auto getDict() const { return std::get<4>(col); }
  inline auto getColumnDef() const { return col; }

  explicit ColumnDef(std::string name, data_type dType, size_t width,
                     SnapshotTypes snapshotType = DefaultSnapshotMechanism,
                     dict_dstring_t *dict = nullptr)
      : col(name, dType, width, snapshotType, dict) {}

 private:
  std::tuple<std::string, storage::data_type, size_t, SnapshotTypes,
             dict_dstring_t *>
      col;
};

class TableDef {
 public:
  void emplace_back(std::string name, data_type dt, size_t width,
                    SnapshotTypes snapshotType = DefaultSnapshotMechanism,
                    dict_dstring_t *dict = nullptr) {
    columns.emplace_back(name, dt, width, snapshotType, dict);
  }

  size_t size() { return columns.size(); }

  std::vector<ColumnDef> getColumns() { return columns; }

 private:
  std::vector<ColumnDef> columns{};
};

class Schema {
 public:
  // Singleton
  static inline Schema &getInstance() {
    static Schema instance;
    return instance;
  }
  Schema(Schema const &) = delete;          // Don't Implement
  void operator=(Schema const &) = delete;  // Don't implement

  /* returns pointer to the table */
  Table *create_table(std::string name, layout_type layout, TableDef columns,
                      uint64_t initial_num_records = 10000000,
                      bool indexed = true, bool partitioned = true,
                      int numa_idx = -1);

  storage::Table *getTable(table_id_t tableId);
  Table *getTable(const std::string &name);
  const std::vector<Table *> &getTables() { return tables; }
  void drop_table(const std::string &name);

  void teardown(const std::string &cdf_out_path = "");

  // delta-based multi-versioning
  inline void add_active_txn(delta_id_t ver, xid_t epoch,
                             worker_id_t worker_id) {
    this->deltaStore[ver]->increment_reader(epoch, worker_id);
  }
  inline void remove_active_txn(delta_id_t ver, xid_t epoch,
                                worker_id_t worker_id) {
    this->deltaStore[ver]->decrement_reader(epoch, worker_id);
  }
  inline void switch_delta(delta_id_t prev, delta_id_t curr, xid_t epoch,
                           worker_id_t worker_id) {
    this->deltaStore[prev]->decrement_reader(epoch, worker_id);
    this->deltaStore[curr]->increment_reader(epoch, worker_id);
  }

  // twin-column/ HTAP snapshotting
  void twinColumn_snapshot(xid_t epoch,
                           master_version_t snapshot_master_version);
  void ETL(uint numa_affinity_idx);
  bool is_sync_in_progress() { return snapshot_sync_in_progress.load(); }

  void snapshot(xid_t epoch,
                std::vector<column_uuid_t> *snapshot_columns = nullptr);

  // utility functions
  void report();
  void memoryReport() const;
  static void save_cdf(const std::string &out_path);

 private:
  std::vector<Table *> tables{};
  DeltaStore *deltaStore[global_conf::num_delta_storages]{};
  std::map<std::string, Table *> table_name_map{};

  // serialization-lock
  std::mutex schema_lock;

  // stats
  table_id_t num_tables;
  uint64_t total_mem_reserved;
  uint64_t total_delta_mem_reserved;

  // snapshotting
  std::future<bool> snapshot_sync;
  std::atomic<bool> snapshot_sync_in_progress;

 private:
  void drop_table(Table *);
  bool sync_master_ver_tbl(storage::Table *tbl,
                           master_version_t snapshot_master_ver);
  bool sync_master_ver_schema(master_version_t snapshot_master_ver);

  Schema()
      : total_mem_reserved(0),
        total_delta_mem_reserved(0),
        snapshot_sync_in_progress(false),
        num_tables(0) {
    aeolus::snapshot::SnapshotManager::init();

    for (int i = 0; i < global_conf::num_delta_storages; i++) {
      deltaStore[i] =
          new DeltaStore(i, g_delta_size, g_delta_size, g_num_partitions);
      this->total_delta_mem_reserved += deltaStore[i]->total_mem_reserved;
    }
  }

  friend class Table;
  friend class ColumnStore;
};

};  // namespace storage

#endif  // PROTEUS_SCHEMA_HPP
