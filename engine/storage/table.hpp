/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef STORAGE_TABLE_HPP_
#define STORAGE_TABLE_HPP_

#include <assert.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "storage/delta_storage.hpp"
#include "storage/memory_manager.hpp"

namespace storage {

class Schema;
class Table;
class ColumnStore;
class RowStore;
class Column;
class DeltaStore;

enum layout_type { ROW_STORE, COLUMN_STORE };

enum data_type { META, INTEGER, STRING, FLOAT, VARCHAR, DATE };

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
  Table *create_table(
      std::string name, layout_type layout,
      std::vector<std::tuple<std::string, data_type, size_t>> columns,
      uint64_t initial_num_records = 10000000, bool indexed = true,
      bool partitioned = true, int numa_idx = -1);

  void drop_table(std::string name);
  void drop_table(int idx);

  void initiate_gc(ushort ver);
  void add_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void remove_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void switch_delta(ushort prev, ushort curr, uint64_t epoch,
                    uint8_t worker_id);

  void teardown();
  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver);

  void report();

  std::vector<Table *> getTables() { return tables; }
  uint64_t total_mem_reserved;
  uint64_t total_delta_mem_reserved;

  DeltaStore *deltaStore[global_conf::num_delta_storages];

  volatile std::atomic<uint64_t> rid;
  inline uint64_t __attribute__((always_inline)) get_next_rid() {
    return rid.fetch_add(1);
  }

 private:
  uint8_t num_tables;
  std::vector<Table *> tables;

  // MultiVersioning

  Schema() {
    global_conf::SnapshotManager::init();

    total_mem_reserved = 0;
    total_delta_mem_reserved = 0;
    rid = 0;

    if (global_conf::cc_ismv) {
      // init delta store

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
  virtual std::vector<const void *> getRecordByKey(uint64_t vid,
                                                   const ushort *col_idx,
                                                   ushort num_cols) = 0;

  virtual void getRecordByKey(uint64_t vid, const ushort *col_idx,
                              ushort num_cols, void *loc) = 0;

  virtual void touchRecordByKey(uint64_t vid) = 0;

  // hack for loading binary files
  virtual void insertIndexRecord(uint64_t rid, uint64_t xid,
                                 ushort partition_id, ushort master_ver) = 0;

  void reportUsage() {
    std::cout << "Table: " << this->name << std::endl;
    for (int i = 0; i < NUM_SOCKETS; i++) {
      std::cout << "P" << i << ": " << vid[i].load() << std::endl;
    }
  }

  virtual void snapshot(uint64_t epoch, uint8_t snapshot_master_ver) = 0;
  void printDetails() {
    std::cout << "Number of Columns:\t" << num_columns << std::endl;
  }

  // uint64_t getNumRecords() { return (vid.load() - 1); }

  Table(std::string name, uint8_t table_id)
      : name(name), table_id(table_id), total_mem_reserved(0) {
    for (int i = 0; i < NUM_SOCKETS; i++) vid[i] = 0;
  }
  virtual ~Table();

  global_conf::PrimaryIndex<uint64_t> *p_index;
  global_conf::PrimaryIndex<uint64_t> **s_index;
  uint64_t total_mem_reserved;
  volatile std::atomic<uint64_t> vid[NUM_SOCKETS];
  const std::string name;
  const uint8_t table_id;

 protected:
  int num_columns;
  DeltaStore **deltaStore;

  // int primary_index_col_idx;

  friend class Schema;
};

};  // namespace storage

#endif /* STORAGE_TABLE_HPP_ */
