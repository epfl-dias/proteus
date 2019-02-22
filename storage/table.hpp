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
#include "storage/memory_manager.hpp"

namespace storage {

class Schema;
class Table;
class ColumnStore;
class RowStore;
class Row;
class Column;

enum layout_type { ROW_STORE, COLUMN_STORE };

enum data_type { INTEGER };

class Schema {
 public:
  Schema(std::string name) : name(name) {
    std::cout << "Schema Constructor\n";
  };

  /*
        some functionality to start the transaction.
  */

  Table* getTable(int idx);
  Table* getTable(std::string name);

  /* returns pointer to the table */
  Table* create_table(
      std::string name, layout_type layout,
      std::vector<std::tuple<std::string, data_type, size_t>> columns);

  void drop_table(std::string name);
  void drop_table(int idx);

 private:
  std::string name;
  int num_tables;
  std::vector<Table*> tables;
};

class Table {
 public:
  // virtual void deleteAllTuples() = 0;
  virtual uint64_t insertRecord(void* rec) = 0;
  virtual void updateRecord(void* key, void* data) = 0;
  virtual void deleteRecord(void* key) = 0;
  virtual std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, std::vector<int>* col_idx = nullptr) = 0;

  void printDetails() {
    std::cout << "Number of Columns:\t" << num_columns << std::endl;
    // std::cout << "Primary Index on Column # " << primary_index_col_idx
    //         << std::endl;
  }
  Table() { std::cout << "TABLE CONSTRUCTOR CALLED!" << std::endl; }
  virtual ~Table() { std::cout << "TABLE DESSTRUCTOR SCALLED!" << std::endl; };

  global_conf::PrimaryIndex<uint64_t>* p_index;

 protected:
  std::string name;
  int num_columns;
  std::atomic<uint64_t> vid;

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
  uint64_t insertRecord(void* rec) { return -1; }
  void updateRecord(void* key, void* data) {}
  void deleteRecord(void* key) {}

 private:
};

/*  DATA LAYOUT -- COLUMN STORE
 */

class ColumnStore : public Table {
 public:
  ColumnStore(std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns);
  uint64_t insertRecord(void* rec);
  void updateRecord(void* key, void* data);
  void deleteRecord(void* key);
  std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, std::vector<int>* col_idx = nullptr);

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
};

class Column {
 public:
  Column(std::string name, data_type type = INTEGER,
         size_t unit_size = sizeof(int), bool build_index = false,
         int initial_num_records = 100000000);
  ~Column();

  void buildIndex();
  void* getRange(uint64_t start_idx, uint64_t end_idx);
  void* getElem(uint64_t idx); /*{

           assert(data_ptr != NULL);

           int data_loc = idx * elem_size;

           for (const auto &chunk : data_ptr) {
                   if(chunk->size <= (data_loc+elem_size) ){
                           return chunk->data+data_loc;
                   }
           }

   }*/

  void insertElem(uint64_t offset, void* elem);
  void updateElem(uint64_t offset, void* elem);
  void deleteElem(uint64_t offset);

 private:
  std::string name;
  size_t elem_size;
  std::vector<mem_chunk*> data_ptr;
  bool is_indexed;
  data_type type;
  // indexes::Index* index_ptr;

  friend class ColumnStore;
};

};  // namespace storage

#endif /* TABLE_HPP_ */
