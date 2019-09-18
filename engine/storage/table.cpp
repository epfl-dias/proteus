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

#include "storage/table.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/delta_storage.hpp"
#include "storage/row_store.hpp"

#if HTAP_DOUBLE_MASTER
#include "codegen/memory/memory-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/topology/topology.hpp"
#endif

/*

  TODO:
    - resizeable columns
    - batch insert/upd
*/

namespace storage {

void Schema::snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
  for (auto& tbl : tables) {
    tbl->snapshot(epoch, snapshot_master_ver);
  }
}

void Schema::report() {
  for (auto& tbl : tables) {
    tbl->p_index->report();
  }
}

void Schema::initiate_gc(ushort ver) {  // deltaStore[ver]->try_reset_gc();
}

void Schema::add_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id) {
  deltaStore[ver]->increment_reader(epoch, worker_id);
}

void Schema::remove_active_txn(ushort ver, uint64_t epoch, uint8_t worker_id) {
  deltaStore[ver]->decrement_reader(epoch, worker_id);
}

void Schema::switch_delta(ushort prev, ushort curr, uint64_t epoch,
                          uint8_t worker_id) {
  deltaStore[prev]->decrement_reader(epoch, worker_id);
  // either add a barrier here or inside delta storage.

  deltaStore[curr]->increment_reader(epoch, worker_id);
}

void Schema::teardown() {
  for (auto& tbl : tables) {
    tbl->~Table();
  }
  if (global_conf::cc_ismv) {
    // init delta store

    for (int i = 0; i < global_conf::num_delta_storages; i++) {
      deltaStore[i]->~DeltaStore();
    }
  }
}

std::vector<Table*> Schema::getAllTables() { return tables; }

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
    std::vector<std::tuple<std::string, data_type, size_t>> columns,
    uint64_t initial_num_records, bool indexed, bool partitioned,
    int numa_idx) {
  Table* tbl = nullptr;

  if (layout == COLUMN_STORE) {
    // void* obj_ptr =
    //     MemoryManager::alloc(sizeof(ColumnStore), DEFAULT_MEM_NUMA_SOCKET);

    tbl = new ColumnStore((this->num_tables + 1), name, columns,
                          initial_num_records, indexed, partitioned, numa_idx);

  } else if (layout == ROW_STORE) {
    // void* obj_ptr =
    //     MemoryManager::alloc(sizeof(RowStore), DEFAULT_MEM_NUMA_SOCKET);

    tbl = new RowStore((this->num_tables + 1), name, columns,
                       initial_num_records, indexed, partitioned, numa_idx);
  } else {
    throw new std::runtime_error("Unknown layout type");
  }
  tables.push_back(tbl);
  this->num_tables++;
  this->total_mem_reserved += tbl->total_mem_reserved;

  return tbl;
}

void Schema::drop_table(std::string name) {
  assert(false && "Not Implemented");
  // int index = -1;
  // for (const auto& t : tables) {
  //   if (name.compare(t->name) == 0) {
  //     index = std::distance(tables.begin(), &t);
  //   }
  // }

  // if (index != -1) this->drop_table(index);
}

void Schema::drop_table(int idx) {
  // TODO: drop table impl
  assert(false && "Not Implemented");
}

void Table::reportUsage() {
  std::cout << "Table: " << this->name << std::endl;
  for (int i = 0; i < NUM_SOCKETS; i++) {
    std::cout << "P" << i << ": " << vid[i].load() << std::endl;
  }
}

Table::~Table() {}

};  // namespace storage
