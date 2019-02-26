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
class DeltaStore;

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
  virtual uint64_t insertRecord(void* rec, short master_ver) = 0;
  virtual void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                            short prev_master_ver, uint64_t tmin,
                            uint64_t tmax) = 0;
  virtual void deleteRecord(uint64_t vid, short master_ver) = 0;
  virtual std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, short master_ver, std::vector<int>* col_idx = nullptr) = 0;

  void clearDelta(short ver);
  virtual bool getVersions(uint64_t vid, short master_ver,
                           global_conf::mv_version_list& vlst) = 0;

  void printDetails() {
    std::cout << "Number of Columns:\t" << num_columns << std::endl;
  }
  Table() { std::cout << "TABLE CONSTRUCTOR CALLED!" << std::endl; }
  virtual ~Table() { std::cout << "TABLE DESSTRUCTOR SCALLED!" << std::endl; };

  global_conf::PrimaryIndex<uint64_t>* p_index;

 protected:
  std::string name;
  int num_columns;
  std::atomic<uint64_t> vid;

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
  uint64_t insertRecord(void* rec, short master_ver) { return -1; }
  void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                    short prev_master_ver, uint64_t tmin, uint64_t tmax) {}
  void deleteRecord(uint64_t vid, short master_ver) {}
  void clearDelta(short ver) {}
  bool getVersions(uint64_t vid, short master_ver,
                   global_conf::mv_version_list& vlst) {
    return false;
  }

 private:
};

/*  DATA LAYOUT -- COLUMN STORE
 */

/* Currently DeltaStore is not resizeable*/
class DeltaStore {
 public:
  DeltaStore(size_t rec_size, uint64_t initial_num_objs) {
    size_t mem_req = (rec_size * initial_num_objs) +
                     (rec_size * sizeof(global_conf::mv_version));
    int numa_id = 0;
    void* mem = MemoryManager::alloc(mem_req, numa_id);

    // warm-up mem
    int* pt = (int*)mem;
    for (int i = 0; i < initial_num_objs; i++) pt[i] = 0;

    // init object vars
    // this->data_ptr.emplace_back(new mem_chunk(mem, mem_req, numa_id));
    data_ptr = new mem_chunk(mem, mem_req, numa_id);
    this->rec_size = rec_size;
    this->cursor = (char*)mem;
    this->total_rec_capacity = initial_num_objs;
    this->used_recs_capacity = 0;

    vid_version_map.reserve(initial_num_objs);
  }
  ~DeltaStore();

  void insert_version(uint64_t vid, void* rec, uint64_t tmin, uint64_t tmax) {
    assert(used_recs_capacity < total_rec_capacity);
    global_conf::mv_version* val = (global_conf::mv_version*)getVersionChunk();
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = getDataChunk();
    memcpy(val->data, rec, rec_size);
    used_recs_capacity++;

    // template <typename K> bool find(const K &key, mapped_type &val)
    global_conf::mv_version_list vlst;
    vid_version_map.find(vid, vlst);
    vlst.insert(val);
    vid_version_map.insert_or_assign(vid, vlst);
  }

  void* insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax) {
    assert(used_recs_capacity < total_rec_capacity);
    global_conf::mv_version* val = (global_conf::mv_version*)getVersionChunk();
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = getDataChunk();
    used_recs_capacity++;

    // template <typename K> bool find(const K &key, mapped_type &val)
    global_conf::mv_version_list vlst;
    vid_version_map.find(vid, vlst);
    vlst.insert(val);
    vid_version_map.insert_or_assign(vid, vlst);

    return val->data;
  }

  bool getVersionList(uint64_t vid, global_conf::mv_version_list& vlst) {
    if (vid_version_map.find(vid, vlst))
      return true;
    else
      return false;
  }

  double getUtilPercentage() {
    return ((double)used_recs_capacity.load() / (double)total_rec_capacity) *
           100;
  };
  void reset() {
    std::unique_lock<std::mutex> lock(this->m);
    vid_version_map.clear();
    cursor = (char*)data_ptr->data;
    used_recs_capacity = 0;
  }

 private:
  inline void* getVersionChunk() {
    void* tmp = nullptr;
    {
      std::unique_lock<std::mutex> lock(this->m);
      tmp = (void*)cursor;
      cursor += sizeof(global_conf::mv_version);
    }
    return tmp;
  }

  inline void* getDataChunk() {
    void* tmp = nullptr;
    {
      std::unique_lock<std::mutex> lock(this->m);
      tmp = (void*)cursor;
      cursor += rec_size;
    }
    return tmp;
  }

  std::mutex m;
  char* cursor;
  size_t rec_size;
  mem_chunk* data_ptr;
  uint64_t total_rec_capacity;
  std::atomic<uint64_t> used_recs_capacity;

  indexes::HashIndex<uint64_t, global_conf::mv_version_list> vid_version_map;

  /* VID -> List Mapping*/
};

class ColumnStore : public Table {
 public:
  ColumnStore(std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns,
              uint64_t initial_num_records = 10000000);
  uint64_t insertRecord(void* rec, short master_ver);
  void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                    short prev_master_ver, uint64_t tmin, uint64_t tmax);
  void deleteRecord(uint64_t vid, short master_ver);
  std::vector<std::tuple<const void*, data_type>> getRecordByKey(
      uint64_t vid, short master_ver, std::vector<int>* col_idx = nullptr);

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
  Column(std::string name, int initial_num_records, data_type type = INTEGER,
         size_t unit_size = sizeof(int), bool build_index = false);
  ~Column();

  void buildIndex();
  void* getRange(uint64_t start_idx, uint64_t end_idx, short master_ver);
  void* getElem(uint64_t idx, short master_ver); /*{

           assert(data_ptr != NULL);

           int data_loc = idx * elem_size;

           for (const auto &chunk : data_ptr) {
                   if(chunk->size <= (data_loc+elem_size) ){
                           return chunk->data+data_loc;
                   }
           }

   }*/

  void insertElem(uint64_t offset, void* elem, short master_ver);
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
