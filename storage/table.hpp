/*
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

#ifndef TABLE_HPP_
#define TABLE_HPP_

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
class Row;
class Column;
class DeltaStore;

enum layout_type { ROW_STORE, COLUMN_STORE };

enum data_type { META, INTEGER };

class Schema {
 public:
  Schema(std::string name) : name(name) {
    std::cout << "Create Schema:\t" << name << std::endl;
  }

  Table* getTable(int idx);
  Table* getTable(std::string name);

  /* returns pointer to the table */
  Table* create_table(
      std::string name, layout_type layout,
      std::vector<std::tuple<std::string, data_type, size_t>> columns,
      uint64_t initial_num_records = 10000000);

  void drop_table(std::string name);
  void drop_table(int idx);

 private:
  std::string name;
  int num_tables;
  std::vector<Table*> tables;
};

class Table {
 public:
  virtual uint64_t insertRecord(void* rec, short master_ver) = 0;
  virtual void* insertRecord(void* rec, uint64_t xid, short master_ver) = 0;

  virtual void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                            short prev_master_ver, uint64_t tmin,
                            uint64_t tmax) = 0;
  virtual void deleteRecord(uint64_t vid, short master_ver) = 0;
  virtual std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, short master_ver, std::vector<int>* col_idx = nullptr) = 0;

  void clearDelta(short ver);
  virtual global_conf::mv_version_list* getVersions(uint64_t vid,
                                                    short master_ver) = 0;

  void printDetails() {
    std::cout << "Number of Columns:\t" << num_columns << std::endl;
  }
  Table() {}
  virtual ~Table(){};

  global_conf::PrimaryIndex<uint64_t>* p_index;

 protected:
  std::string name;
  int num_columns;
  volatile std::atomic<uint64_t> vid;

  // MultiVersioning
  DeltaStore* deltaStore[global_conf::num_master_versions];

  // int primary_index_col_idx;

  friend class Schema;
};

/*  DATA LAYOUT -- ROW STORE
 */

class Row {
  size_t elem_size;
  std::vector<mem_chunk*> data_ptr;

  void* getRow(int idx);
  void* getRange(int start_idx, int end_idx);
};

class rowStore : public Table {
 public:
  uint64_t insertRecord(void* rec, short master_ver) { return -1; };
  void* insertRecord(void* rec, uint64_t xid, short master_ver) {
    return nullptr;
  };

  void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                    short prev_master_ver, uint64_t tmin, uint64_t tmax) {}
  void deleteRecord(uint64_t vid, short master_ver) {}
  void clearDelta(short ver) {}
  global_conf::mv_version_list* getVersions(uint64_t vid, short master_ver) {
    return nullptr;
  }

 private:
};

/*  DATA LAYOUT -- COLUMN STORE
 */

class ColumnStore : public Table {
 public:
  ColumnStore(std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns,
              uint64_t initial_num_records = 10000000);
  uint64_t insertRecord(void* rec, short master_ver);
  void* insertRecord(void* rec, uint64_t xid, short master_ver);
  void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                    short prev_master_ver, uint64_t tmin, uint64_t tmax);
  void deleteRecord(uint64_t vid, short master_ver);
  std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, short master_ver, std::vector<int>* col_idx = nullptr);

  global_conf::mv_version_list* getVersions(uint64_t vid, short master_ver);

  /*
    No secondary indexes supported as of yet so dont need the following
    void createIndex(int col_idx);
    void createIndex(std::string col_name);
  */

  ~ColumnStore() {
    std::cout << "COLUMNSTORE DESSTRUCTOR SCALLED!" << std::endl;
  }

 private:
  std::vector<Column*> columns;
  Column* meta_column;
  size_t rec_size;
};

class Column {
 public:
  Column(std::string name, uint64_t initial_num_records,
         data_type type = INTEGER, size_t unit_size = sizeof(uint64_t),
         bool build_index = false);
  ~Column();

  void buildIndex();
  void* getRange(uint64_t start_idx, uint64_t end_idx, short master_ver);
  void* getElem(uint64_t idx, short master_ver);

  void* insertElem(uint64_t offset, void* elem, short master_ver);
  void updateElem(uint64_t offset, void* elem, short master_ver);
  void deleteElem(uint64_t offset, short master_ver);

 private:
  std::string name;
  size_t elem_size;
  bool is_indexed;
  data_type type;
  // indexes::Index* index_ptr;
  // we need data structure for # number of master versions and delta storage
  // for MVCC.
  std::vector<mem_chunk*> master_versions[global_conf::num_master_versions];

  // std::vector<std::pair<int, std::vector<mem_chunk*>>> master_ver;
  // std::vector<mem_chunk*> data_ptr;

  friend class ColumnStore;
};

};  // namespace storage

#endif /* TABLE_HPP_ */
