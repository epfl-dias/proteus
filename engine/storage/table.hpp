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
  std::vector<Table *> getAllTable();

  /* returns pointer to the table */
  Table *create_table(
      std::string name, layout_type layout,
      std::vector<std::tuple<std::string, data_type, size_t>> columns,
      uint64_t initial_num_records = 10000000);

  void drop_table(std::string name);
  void drop_table(int idx);

  void initiate_gc(ushort ver);
  void add_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void remove_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id);
  void switch_delta(ushort prev, ushort curr, uint64_t epoch,
                    uint8_t worker_id);

  void teardown();
  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver);
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
  virtual uint64_t insertRecord(void *rec, ushort master_ver) = 0;
  virtual void *insertRecord(void *rec, uint64_t xid, ushort master_ver) = 0;

  virtual void updateRecord(uint64_t vid, const void *data,
                            ushort ins_master_ver, ushort prev_master_ver,
                            ushort delta_ver, uint64_t tmin, uint64_t tmax,
                            ushort pid) = 0;
  virtual void updateRecord(uint64_t vid, const void *rec,
                            ushort ins_master_ver, ushort prev_master_ver,
                            ushort delta_ver, uint64_t tmin, uint64_t tmax,
                            ushort pid, std::vector<ushort> *col_idx) = 0;
  virtual void deleteRecord(uint64_t vid, ushort master_ver) = 0;
  virtual std::vector<const void *> getRecordByKey(
      uint64_t vid, ushort master_ver,
      const std::vector<ushort> *col_idx = nullptr) = 0;

  virtual void getRecordByKey(uint64_t vid, ushort master_ver,
                              const std::vector<ushort> *col_idx,
                              void *loc) = 0;
  virtual void touchRecordByKey(uint64_t vid, ushort master_ver) = 0;

  virtual void insertIndexRecord(
      uint64_t xid, ushort master_ver) = 0;  // hack for loading binary files

  virtual global_conf::mv_version_list *getVersions(uint64_t vid,
                                                    ushort delta_ver) = 0;

  virtual void snapshot(uint64_t epoch, uint8_t snapshot_master_ver) = 0;
  void printDetails() {
    std::cout << "Number of Columns:\t" << num_columns << std::endl;
  }

  uint64_t getNumRecords() { return (vid.load() - 1); }

  Table(std::string name, uint8_t table_id)
      : name(name), table_id(table_id), vid(0), total_mem_reserved(0) {}
  virtual ~Table();

  global_conf::PrimaryIndex<uint64_t> *p_index;
  global_conf::PrimaryIndex<uint64_t> **s_index;
  uint64_t total_mem_reserved;
  volatile std::atomic<uint64_t> vid;
  const std::string name;
  const uint8_t table_id;

 protected:
  int num_columns;
  DeltaStore **deltaStore;

  // int primary_index_col_idx;

  friend class Schema;
};

/*  DATA LAYOUT -- COLUMN STORE
 */

class ColumnStore : public Table {
 public:
  ColumnStore(uint8_t table_id, std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns,
              uint64_t initial_num_records = 10000000);
  uint64_t insertRecord(void *rec, ushort master_ver);
  void *insertRecord(void *rec, uint64_t xid, ushort master_ver);
  void updateRecord(uint64_t vid, const void *data, ushort ins_master_ver,
                    ushort prev_master_ver, ushort delta_ver, uint64_t tmin,
                    uint64_t tmax, ushort pid);
  void updateRecord(uint64_t vid, const void *rec, ushort ins_master_ver,
                    ushort prev_master_ver, ushort delta_ver, uint64_t tmin,
                    uint64_t tmax, ushort pid, std::vector<ushort> *col_idx);
  void deleteRecord(uint64_t vid, ushort master_ver);
  std::vector<const void *> getRecordByKey(
      uint64_t vid, ushort master_ver,
      const std::vector<ushort> *col_idx = nullptr);

  void insertIndexRecord(uint64_t xid,
                         ushort master_ver);  // hack for loading binary files

  uint64_t load_data_from_binary(std::string col_name, std::string file_path);

  void getRecordByKey(uint64_t vid, ushort master_ver,
                      const std::vector<ushort> *col_idx, void *loc);
  void touchRecordByKey(uint64_t vid, ushort master_ver);

  global_conf::mv_version_list *getVersions(uint64_t vid, ushort delta_ver);

  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver);
  void num_upd_tuples();
  const std::vector<Column *> &getColumns() { return columns; }

  /*
    No secondary indexes supported as of yet so dont need the following
    void createIndex(int col_idx);
    void createIndex(std::string col_name);
  */

  ~ColumnStore();

 private:
  std::vector<Column *> columns;
  Column *meta_column;
  // Column **secondary_index_vals;
  size_t rec_size;
};

class Column {
 public:
  Column(std::string name, uint64_t initial_num_records,
         data_type type = INTEGER, size_t unit_size = sizeof(uint64_t),
         bool build_index = false, bool single_version_only = false);
  ~Column();

  void buildIndex();
  void *getElem(uint64_t idx, ushort master_ver);
  void touchElem(uint64_t idx, ushort master_ver);
  void getElem(uint64_t idx, ushort master_ver, void *copy_location);

  void insertElem(uint64_t offset, void *elem, ushort master_ver);
  void *insertElem(uint64_t offset);
  void updateElem(uint64_t offset, void *elem, ushort master_ver);
  void deleteElem(uint64_t offset, ushort master_ver);

  void snapshot(uint64_t num_records, uint64_t epoch,
                uint8_t snapshot_master_ver);

  uint64_t load_from_binary(std::string file_path);

  void num_upd_tuples();

  size_t getSize() { return this->total_mem_reserved; }

  // const std::vector<mem_chunk *> get_data(ushort master_version = 0) {
  //   assert(master_version <= global_conf::num_master_versions);
  //   return master_versions[master_version];
  // }

  // snapshot stuff
  std::vector<std::pair<mem_chunk, uint64_t>> snapshot_get_data();
  uint64_t snapshot_get_num_records();

  const std::string name;
  const size_t elem_size;
  const data_type type;

 private:
  size_t total_mem_reserved;
  bool is_indexed;

  std::vector<mem_chunk> master_versions[global_conf::num_master_versions]
                                        [NUM_SOCKETS];

  // Insert snapshotting manager here.
  std::vector<decltype(global_conf::SnapshotManager::create(0))> arena;

  friend class ColumnStore;
};

};  // namespace storage

#endif /* STORAGE_TABLE_HPP_ */
