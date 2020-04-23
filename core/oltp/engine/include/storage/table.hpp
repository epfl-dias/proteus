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

#include <assert.h>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "snapshot/snapshot_manager.hpp"
#include "storage/delta_storage.hpp"
#include "storage/memory_manager.hpp"

typedef std::map<uint64_t, std::string> dict_dstring_t;

namespace storage {

class Schema;
class Table;
class ColumnStore;
class RowStore;
class Column;
class DeltaStore;

enum layout_type { ROW_STORE, COLUMN_STORE };

enum data_type { META, INTEGER, STRING, FLOAT, VARCHAR, DATE, DSTRING };

class ColumnDef {
  std::vector<
      std::tuple<std::string, storage::data_type, size_t, dict_dstring_t *>>
      columns;


 public:
  void emplace_back(std::string name, data_type dt, size_t width,
                    dict_dstring_t *dict = nullptr) {
    columns.emplace_back(name, dt, width, dict);
  }

  size_t size() { return columns.size(); }

  std::vector<
      std::tuple<std::string, storage::data_type, size_t, dict_dstring_t *>>

  getColumns() {
    return columns;
  }
};

class NUMAPartitionPolicy {
 public:
  // Singleton
  static inline NUMAPartitionPolicy &getInstance() {
    static NUMAPartitionPolicy instance;
    return instance;
  }
  NUMAPartitionPolicy(NUMAPartitionPolicy const &) = delete;  // Don't Implement
  void operator=(NUMAPartitionPolicy const &) = delete;       // Don't implement

  class TablePartition {
   public:
    const uint pid;
    const uint numa_idx;

   public:
    explicit TablePartition(uint pid, uint numa_idx)
        : pid(pid), numa_idx(numa_idx) {}

    inline bool operator==(const TablePartition &o) {
      return (pid == o.pid && numa_idx == o.numa_idx);
    }
  };

  const TablePartition &getPartitionInfo(uint pid) {
    assert(pid < PartitionVector.size());
    return PartitionVector[pid];
  }

  uint getDefaultPartition() {
    assert(PartitionVector.size() > 0);
    return PartitionVector[0].numa_idx;
  }

 private:
  std::vector<TablePartition> PartitionVector;

  NUMAPartitionPolicy() {
    auto num_numa_nodes = topology::getInstance().getCpuNumaNodeCount();
    for (uint i = 0; i < num_numa_nodes; i++) {
      if (global_conf::reverse_partition_numa_mapping)
        PartitionVector.emplace_back(TablePartition{i, num_numa_nodes - i - 1});
      else
        PartitionVector.emplace_back(TablePartition{i, i});
    }
  }
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

  Table *getTable(int idx);
  Table *getTable(std::string name);
  std::vector<Table *> getAllTables();

  /* returns pointer to the table */
  Table *create_table(std::string name, layout_type layout, ColumnDef columns,
                      uint64_t initial_num_records = 10000000,
                      bool indexed = true, bool partitioned = true,
                      int numa_idx = -1);
  void destroy_table(Table *);

  [[noreturn]] void drop_table(std::string name);
  [[noreturn]] void drop_table(int idx);

  void initiate_gc(ushort ver);
  void add_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void remove_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void switch_delta(ushort prev, ushort curr, uint64_t epoch,
                    uint8_t worker_id);

  void teardown();
  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver);
  void ETL(uint numa_node_idx);

  bool is_sync_in_progress(){
    return snapshot_sync_in_progress.load();
  }

  void report();

  std::vector<Table *> getTables() { return tables; }
  uint64_t total_mem_reserved;
  uint64_t total_delta_mem_reserved;

  DeltaStore *deltaStore[global_conf::num_delta_storages];

  // volatile std::atomic<uint64_t> rid;
  // inline uint64_t __attribute__((always_inline)) get_next_rid() {
  //   return rid.fetch_add(1);
  // }

 private:
  uint8_t num_tables;
  std::vector<Table *> tables;

  // snapshotting
  std::future<bool> snapshot_sync;
  std::atomic<bool> snapshot_sync_in_progress;
  bool sync_master_ver_tbl(const storage::Table *tbl,
                           const uint8_t snapshot_master_ver);
  bool sync_master_ver_schema(const uint8_t snapshot_master_ver);

  Schema()
      : total_mem_reserved(0),
        total_delta_mem_reserved(0),
        snapshot_sync_in_progress(0) {
    aeolus::snapshot::SnapshotManager::init();

    if (global_conf::cc_ismv) {
      for (int i = 0; i < global_conf::num_delta_storages; i++) {
        deltaStore[i] = new DeltaStore(i);
        this->total_delta_mem_reserved += deltaStore[i]->total_mem_reserved;
      }
    }
  }

  friend class Table;
};

class Table {
 public:
  virtual uint64_t insertRecord(void *rec, ushort partition_id,
                                ushort master_ver) = 0;
  virtual void *insertRecord(void *rec, uint64_t xid, ushort partition_id,
                             ushort master_ver) = 0;
  virtual void *insertRecordBatch(void *rec_batch, uint recs_to_ins,
                                  uint capacity_offset, uint64_t xid,
                                  ushort partition_id, ushort master_ver) = 0;

  virtual void updateRecord(global_conf::IndexVal *hash_ptr, const void *rec,
                            ushort curr_master, ushort curr_delta,
                            const ushort *col_idx = nullptr,
                            short num_cols = -1) = 0;

  // virtual void updateRecord(ushort pid, uint64_t &vid, const void *rec,
  //                           ushort curr_master, ushort curr_delta,
  //                           uint64_t tmin, const ushort *col_idx = nullptr,
  //                           short num_cols = -1) = 0;
  virtual void deleteRecord(uint64_t vid, ushort master_ver) = 0;
  virtual std::vector<const void *> getRecordByKey(uint64_t rec_vid,
                                                   const ushort *col_idx,
                                                   ushort num_cols) {
    assert(false && "Not implemented");
  }

  virtual void getRecordByKey(uint64_t vid, const ushort *col_idx,
                              ushort num_cols, void *loc) = 0;

  virtual void touchRecordByKey(uint64_t vid) = 0;

  // hack for loading binary files
  virtual void insertIndexRecord(uint64_t rid, uint64_t xid,
                                 ushort partition_id, ushort master_ver) = 0;

  virtual void snapshot(uint64_t epoch, uint8_t snapshot_master_ver) = 0;

  virtual void ETL(uint numa_node_idx) = 0;

  // uint64_t getNumRecords() { return (vid.load() - 1); }

  void reportUsage();
  Table(std::string name, uint8_t table_id, layout_type storage_layout,
        ColumnDef columns);
  virtual ~Table();

  global_conf::PrimaryIndex<uint64_t> *p_index;
  global_conf::PrimaryIndex<uint64_t> **s_index;
  uint64_t total_mem_reserved;
  //volatile std::atomic<uint64_t> vid[NUM_SOCKETS];
  std::deque<std::atomic<uint64_t>> vid;
  const std::string name;
  const uint8_t table_id;

  const layout_type storage_layout;

 protected:
  int num_columns;
  DeltaStore **deltaStore;

  size_t rec_size;
  uint64_t initial_num_recs;
  bool indexed;

  friend class Schema;
};

};  // namespace storage

#endif /* STORAGE_TABLE_HPP_ */
