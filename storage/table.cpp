/*
                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

ColumnStore::ColumnStore(
    std::string name,
    std::vector<std::tuple<std::string, data_type, size_t>> columns) {
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
    this->columns.emplace_back(new Column(std::get<0>(t)));
  }
  this->num_columns = columns.size();
  this->name = name;

  // build index over the first column

  this->p_index = new global_conf::PrimaryIndex<uint64_t>();

  this->columns.at(0)->buildIndex();
}

void ColumnStore::insertRecord(void* rec) {}
void ColumnStore::updateRecord(void* key, void* data) {}
void ColumnStore::deleteRecord(void* key) {}

std::vector<std::tuple<const void*, data_type>> ColumnStore::getRecordByKey(
    uint64_t vid, std::vector<int>* col_idx) {
  std::vector<std::tuple<const void*, data_type>> record;

  // int num_cols = col_idx->size();
  // TODO: return col_name too maybe for projections?
  if (col_idx == nullptr) {
    for (auto& col : columns) {
      record.push_back(std::tuple<const void*, data_type>(
          (const void*)(col->getElem(vid)), col->type));
    }
  } else {
    for (auto& c_idx : *col_idx) {
      Column* col = columns.at(c_idx);
      record.push_back(std::tuple<const void*, data_type>(
          (const void*)(col->getElem(vid)), col->type));
    }
  }
  return record;
}

void Column::buildIndex() {
  // TODO: build column index here.

  this->is_indexed = true;
}

Column::Column(std::string name, data_type type, size_t unit_size,
               bool build_index, int initial_num_records)
    : name(name), elem_size(unit_size), type(type) {
  /*
  ALGO:
          - Allocate memory for that column with default start number of records
          - Create index on the column
*/

  // TODO: Allocating memory in the current socket.
  // size: initial_num_recs * unit_size
  std::cout << "INITIAL NUM REC: " << initial_num_records << std::endl;
  int numa_id = 0;
  size_t size = initial_num_records * unit_size;
  void* mem = MemoryManager::alloc(size, numa_id);
  int* pt = (int*)mem;
  for (int i = 0; i < initial_num_records; i++) pt[i] = 0;
  data_ptr.emplace_back(new mem_chunk(mem, size, numa_id));

  // TODO: Allocate column memory and pointers

  if (build_index) this->buildIndex();
}

void* Column::getElem(uint64_t vid) {
  assert(data_ptr.size() != 0);

  int data_idx = vid * elem_size;
  for (const auto& chunk : data_ptr) {
    if (chunk->size <= (data_idx + elem_size)) {
      return ((char*)chunk->data) + data_idx - elem_size;
      ;
    }
  }
  assert(false);
}

Column::~Column() {
  for (auto& chunk : this->data_ptr) {
    MemoryManager::free(chunk->data, chunk->size);
  }
}

};  // namespace storage
