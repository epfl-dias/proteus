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

#include "storage/table.hpp"
#include <cassert>
#include <iostream>
#include <string>

#include "glo.hpp"
#include "indexes/hash_index.hpp"

//#include <codegen/memory/memory-manager.hpp>

namespace storage {

Table* Schema::getTable(const int idx) { return tables.at(idx); }

Table* Schema::getTable(std::string name) {
  // TODO: a better way would be to store table-idx mapping in a hashmap from
  // STL.

  for (const auto& t : tables) {
    if (name.compare(t->name) == 0) return t;
  }
  return nullptr;
}

/* returns pointer to the table */
Table* Schema::create_table(
    std::string name, layout_type layout,
    std::vector<std::tuple<std::string, data_type, size_t>> columns) {
  Table* tbl = nullptr;

  if (layout == COLUMN_STORE) {
    tbl = new ColumnStore(name, columns);

  } else if (layout == ROW_STORE) {
    throw new std::runtime_error("ROW STORE NOT IMPLEMENTED");
  } else {
    throw new std::runtime_error("Unknown layout type");
  }

  this->num_tables++;
  return tbl;
}

void Schema::drop_table(std::string name) {
  int index = -1;
  /*for (const auto &t : tables) {
          if(name.compare(t->name) == 0) {
                  index = std::distance(tables.begin(), &t);
          }
  }*/

  if (index != -1) this->drop_table(index);
}
void Schema::drop_table(int idx) {
  std::cout << "[Schema][drop_table] Not Implemented" << std::endl;
}
void Table::clearDelta(short ver) { deltaStore[ver]->reset(); };

ColumnStore::ColumnStore(
    std::string name,
    std::vector<std::tuple<std::string, data_type, size_t>> columns,
    uint64_t initial_num_records) {
  /*
          TODO: take an argument for column_index or maybe a flag in the tuple
     for the indexed columns. Currently, by default, the first column would be
     considered as key and will be indexed.
  */

  // assert column type for zeroth column should ne int64 as we only support
  // int64 index for now.

  // create columns
  for (const auto& t : columns) {
    std::cout << "Creating Column: " << std::get<0>(t) << std::endl;
    this->columns.emplace_back(new Column(std::get<0>(t), initial_num_records));
  }
  this->num_columns = columns.size();
  this->name = name;
  this->vid = 0;

  // build index over the first column
  this->p_index = new global_conf::PrimaryIndex<uint64_t>();
  this->columns.at(0)->buildIndex();

  // init delta store
  size_t rec_size = 0;
  for (auto& co : this->columns) rec_size += co->elem_size;
  for (int i = 0; i < global_conf::num_master_versions; i++) {
    deltaStore[i] = new DeltaStore(rec_size, initial_num_records);
  }
}

/* Following function assumes that the  void* rec has columns in the same order
 * as the actual columns
 */
uint64_t ColumnStore::insertRecord(void* rec, short master_ver) {
  // std::vector<uint64_t> tmp(num_fields, key_gen++);
  // ycsb_tbl->insertRecord(&tmp);
  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(vid, rec_ptr, master_ver);
    rec_ptr += col->elem_size;
  }
  return this->vid++;
}

void ColumnStore::deleteRecord(uint64_t vid, short master_ver) {}

std::vector<std::tuple<const void*, data_type>> ColumnStore::getRecordByKey(
    uint64_t vid, short master_ver, std::vector<int>* col_idx) {
  std::vector<std::tuple<const void*, data_type>> record;

  // int num_cols = col_idx->size();
  // TODO: return col_name too maybe for projections?
  if (col_idx == nullptr) {
    for (auto& col : columns) {
      record.push_back(std::tuple<const void*, data_type>(
          (const void*)(col->getElem(vid, master_ver)), col->type));
    }
  } else {
    for (auto& c_idx : *col_idx) {
      Column* col = columns.at(c_idx);
      record.push_back(std::tuple<const void*, data_type>(
          (const void*)(col->getElem(vid, master_ver)), col->type));
    }
  }
  return record;
}

bool ColumnStore::getVersions(uint64_t vid, short master_ver,
                              global_conf::mv_version_list& vlst) {
  return this->deltaStore[master_ver]->getVersionList(vid, vlst);
}

void ColumnStore::updateRecord(uint64_t vid, void* rec, short ins_master_ver,
                               short prev_master_ver, uint64_t tmin,
                               uint64_t tmax) {
  // char* cursor = (char*)rec;

  // if(ins_master_ver == prev_master_ver) need version ELSE update the master
  // one.
  if (ins_master_ver == prev_master_ver) {
    // create_version
    char* ver = (char*)this->deltaStore[ins_master_ver]->insert_version(
        vid, tmin, tmax);
    for (auto& col : columns) {
      size_t elem_size = col->elem_size;
      memcpy((void*)ver, col->getElem(vid, prev_master_ver), elem_size);
      ver += elem_size;
    }
  }

  char* cursor = (char*)rec;
  for (auto& col : columns) {
    // skip indexed or let says primary column to update
    // if (!col->is_indexed) {
    col->insertElem(vid, (rec == nullptr ? nullptr : (void*)cursor),
                    ins_master_ver);
    //}
    cursor += col->elem_size;
  }
}

void Column::buildIndex() {
  // TODO: build column index here.

  this->is_indexed = true;
}

Column::Column(std::string name, int initial_num_records, data_type type,
               size_t unit_size, bool build_index)
    : name(name), elem_size(unit_size), type(type) {
  /*
  ALGO:
          - Allocate memory for that column with default start number of records
          - Create index on the column
*/

  // TODO: Allocating memory in the current socket.
  // size: initial_num_recs * unit_size
  // std::cout << "INITIAL NUM REC: " << initial_num_records << std::endl;
  int numa_id = 0;
  size_t size = initial_num_records * unit_size;

  for (short i = 0; i < global_conf::num_master_versions; i++) {
    void* mem = MemoryManager::alloc(size, numa_id);
    int* pt = (int*)mem;
    for (int i = 0; i < initial_num_records; i++) pt[i] = 0;
    master_versions[i].emplace_back(new mem_chunk(mem, size, numa_id));
  }

  // TODO: Allocate column memory and pointers

  if (build_index) this->buildIndex();
}

void* Column::getElem(uint64_t vid, short master_ver) {
  // master_versions[master_ver];

  assert(master_versions[master_ver].size() != 0);

  int data_idx = vid * elem_size;
  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size <= (data_idx + elem_size)) {
      return ((char*)chunk->data) + data_idx - elem_size;
    }
  }
  return nullptr;
}

void Column::insertElem(uint64_t offset, void* elem, short master_ver) {
  uint64_t data_idx = offset * this->elem_size;
  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= (data_idx + elem_size)) {
      // insert elem here
      void* dst = (void*)(((char*)chunk->data) + data_idx);
      if (elem == nullptr) {
        uint64_t* tptr = (uint64_t*)dst;
        (*tptr)++;
      } else {
        std::memcpy(dst, elem, this->elem_size);
      }

      return;
    }
  }

  std::cout << "FUCK. ALLOCATE MOTE MEMORY" << std::endl;
  // exit(-1);
}

Column::~Column() {
  for (short i = 0; i < global_conf::num_master_versions; i++) {
    for (auto& chunk : master_versions[i]) {
      MemoryManager::free(chunk->data, chunk->size);
    }
  }
}

};  // namespace storage
