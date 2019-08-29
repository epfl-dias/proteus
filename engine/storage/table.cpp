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
#include <iostream>
#include <string>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/worker.hpp"
#include "storage/delta_storage.hpp"

#define MEMORY_SLACK 1000
#define CIDR_HACK false

// Proteus includes

#if HTAP_DOUBLE_MASTER
#include "codegen/memory/memory-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/topology/topology.hpp"
#endif

/*

  TODO:
    - resizeable columns
    - partitionable columns
*/

namespace storage {

static inline void __attribute__((always_inline)) set_upd_bit(char* data) {
  *data = *data | (1 << 7);
}

static inline uint64_t __attribute__((always_inline))
merge_vid_parition_id(uint64_t vid, uint8_t partition_id) {
  return (vid & 0x00FFFFFFFFFFFFFF) | (((uint64_t)partition_id) << 56);
}

static inline uint64_t __attribute__((always_inline))
extract_vid(uint64_t vid_pid) {
  return (vid_pid & 0x00FFFFFFFFFFFFFF);
}

static inline ushort __attribute__((always_inline))
extract_pid(uint64_t vid_pid) {
  return (ushort)(vid_pid >> 56);
}

inline uint64_t __attribute__((always_inline))
vid_to_uuid(uint8_t tbl_id, uint64_t vid) {
  return (vid & 0x00FFFFFFFFFFFFFF) | (((uint64_t)tbl_id) << 56);
}

void Schema::snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
  for (auto& tbl : tables) {
    tbl->snapshot(epoch, snapshot_master_ver);
  }
}
inline void ColumnStore::snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
  assert(false);
  // uint64_t num_records = this->vid.load();

  // std::cout << this->name << ":: " << num_records << std::endl;
  // std::cout << "MasterVer:: " << (int)snapshot_master_ver << std::endl;

  // // if (this->name.compare("ssbm_part") == 0) {
  // uint64_t partition_recs = (num_records / NUM_SOCKETS);  //+ (num_records %
  // i); for (int c = 0; c < global_conf::num_master_versions; c++) {
  //   for (int j = 0; j < NUM_SOCKETS; j++) {
  //     if (plugin_ptr[c][j] != nullptr) {
  //       std::cout << "WRITE DONE!!!!!::" << j
  //                 << ", part_recs: " << partition_recs << std::endl;

  //       *plugin_ptr[c][j] = partition_recs;
  //     }
  //   }
  // }
  // //  }

  // for (auto& col : columns) {
  //   col->snapshot(num_records, epoch, snapshot_master_ver);
  // }
}

inline void Column::snapshot(uint64_t num_records, uint64_t epoch,
                             uint8_t snapshot_master_ver) {
  // arena->destroy_snapshot();
  // arena->create_snapshot({num_records, epoch, snapshot_master_ver});

  uint i = 0;
  // FIXME:
  for (auto& ar : arena) {
    uint64_t partition_recs =
        (num_records / NUM_SOCKETS);  //+ (num_records % i);
    // std::cout << "a_" << this->name << ", rec: " << (num_records /
    // NUM_SOCKETS)
    //           << std::endl;
    // std::cout << "b_" << this->name << ", rec: " << (num_records % i)
    //           << std::endl;
    ar->destroy_snapshot();
    ar->create_snapshot({partition_recs, epoch, snapshot_master_ver});

    i++;
  }
  assert(i == NUM_SOCKETS);

#if HTAP_COW
  this->master_versions[0][0]->data = arena->oltp();
#endif

  // for (auto& ar : arena) {
  //   ar->destroy_snapshot();
  //   ar->create_snapshot({num_records, epoch});
  // }
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
    uint64_t initial_num_records, bool indexed, bool partitioned) {
  Table* tbl = nullptr;

  if (layout == COLUMN_STORE) {
    tbl = new ColumnStore((this->num_tables + 1), name, columns,
                          initial_num_records, indexed, partitioned);

  } else if (layout == ROW_STORE) {
    throw new std::runtime_error("ROW STORE NOT IMPLEMENTED");
  } else {
    throw new std::runtime_error("Unknown layout type");
  }
  tables.push_back(tbl);
  this->num_tables++;
  this->total_mem_reserved += tbl->total_mem_reserved;

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
  // TODO: drop table impl
  std::cout << "[Schema][drop_table] Not Implemented" << std::endl;
}

Table::~Table() {}

ColumnStore::~ColumnStore() {
  for (auto& col : columns) {
    delete col;
  }
  delete meta_column;
}
uint64_t ColumnStore::load_data_from_binary(std::string col_name,
                                            std::string file_path) {
  for (auto& c : this->columns) {
    if (c->name.compare(col_name) == 0) {
      return c->load_from_binary(file_path);
    }
  }
  assert(false && "Column not found: ");
}

ColumnStore::ColumnStore(
    uint8_t table_id, std::string name,
    std::vector<std::tuple<std::string, data_type, size_t>> columns,
    uint64_t initial_num_records, bool indexed, bool partitioned)
    : Table(name, table_id), indexed(indexed) {
  /*
          TODO: take an argument for column_index or maybe a flag in the tuple
     for the indexed columns. Currently, by default, the first column would be
     considered as key and will be indexed.
  */

  this->total_mem_reserved = 0;
  this->deltaStore = storage::Schema::getInstance().deltaStore;

  for (int i = 0; i < FLAGS_num_partitions; i++) this->vid[i] = 0;

  if (indexed) {
    meta_column = new Column(name + "_meta", initial_num_records, this, META,
                             sizeof(global_conf::IndexVal));

    // if (partitioned)
    //   this->p_index =
    //       new global_conf::PrimaryIndex<uint64_t>(name, NUM_SOCKETS);
    // else
    //   this->p_index = new global_conf::PrimaryIndex<uint64_t>(name);
    this->p_index = new global_conf::PrimaryIndex<uint64_t>();

    std::cout << "Index done" << std::endl;
  }

  // create columns
  for (const auto& t : columns) {
    this->columns.emplace_back(new Column(std::get<0>(t), initial_num_records,
                                          this, std::get<1>(t), std::get<2>(t),
                                          false));
  }
  for (const auto& t : this->columns) {
    total_mem_reserved += t->total_mem_reserved;
  }

  this->num_columns = columns.size();

  size_t rec_size = 0;
  for (auto& co : this->columns) {
    rec_size += co->elem_size;
  }
  this->rec_size = rec_size;
  this->offset = 0;
#if CIDR_HACK
  this->initial_num_recs = initial_num_records - (NUM_SOCKETS * MEMORY_SLACK);
#else
  this->initial_num_recs = initial_num_records;

#endif
  std::cout << "Table: " << name << std::endl;
  std::cout << "\trecord size: " << rec_size << " bytes" << std::endl;
  std::cout << "\tnum_records: " << initial_num_records << std::endl;

  if (indexed) total_mem_reserved += meta_column->total_mem_reserved;

  std::cout << "\tMem reserved: "
            << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
            << std::endl;

  for (int i = 0; i < global_conf::num_master_versions; i++) {
    for (int j = 0; j < NUM_SOCKETS; j++) {
      plugin_ptr[i][j] = nullptr;
    }
  }
}

// void* ColumnStore::insertMeta(uint64_t vid, global_conf::IndexVal& hash_val)
// {}

void ColumnStore::offsetVID(uint64_t offset) {
  for (int i = 0; i < NUM_SOCKETS; i++) vid[i].store(offset);
  this->offset = offset;
}

void ColumnStore::insertIndexRecord(uint64_t rid, uint64_t xid,
                                    ushort partition_id, ushort master_ver) {
  assert(this->indexed);
  uint64_t curr_vid = vid[partition_id].fetch_add(1);

  void* pano = this->meta_column->insertElem(
      merge_vid_parition_id(curr_vid, partition_id));

  void* hash_ptr = new (pano) global_conf::IndexVal(
      xid, merge_vid_parition_id(curr_vid, partition_id), master_ver);

  this->p_index->insert(rid, hash_ptr);
}

/* Following function assumes that the  void* rec has columns in the same order
 * as the actual columns
 */
void* ColumnStore::insertRecord(void* rec, uint64_t xid, ushort partition_id,
                                ushort master_ver) {
  uint64_t curr_vid = vid[partition_id].fetch_add(1);
  // std::cout << "vID: " << curr_vid << std::endl;
  global_conf::IndexVal* hash_ptr = nullptr;

#if CIDR_HACK
  if (curr_vid >= (initial_num_recs / NUM_SOCKETS)) {
    scheduler::WorkerPool::getInstance().shutdown_manual();
  }

#endif

  if (indexed) {
    void* pano = this->meta_column->insertElem(
        merge_vid_parition_id(curr_vid, partition_id));
    hash_ptr = new (pano) global_conf::IndexVal(
        xid, merge_vid_parition_id(curr_vid, partition_id), master_ver);
  }

  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(merge_vid_parition_id(curr_vid, partition_id), rec_ptr,
                    master_ver);
    rec_ptr += col->elem_size;
  }
  return (void*)hash_ptr;
}

uint64_t ColumnStore::insertRecord(void* rec, ushort partition_id,
                                   ushort master_ver) {
  uint64_t curr_vid = vid[partition_id].fetch_add(1);

  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(merge_vid_parition_id(curr_vid, partition_id), rec_ptr,
                    master_ver);
    rec_ptr += col->elem_size;
  }
  return curr_vid;
}

void ColumnStore::deleteRecord(uint64_t vid, ushort master_ver) {
  assert(false && "Not implemented");
}

void ColumnStore::touchRecordByKey(uint64_t vid, ushort master_ver) {
  for (auto& col : columns) {
    col->touchElem(vid, master_ver);
  }
}

void ColumnStore::getRecordByKey(uint64_t vid, ushort master_ver,
                                 const std::vector<ushort>* col_idx,
                                 void* loc) {
  char* write_loc = (char*)loc;
  if (col_idx == nullptr) {
    for (auto& col : columns) {
      col->getElem(vid, master_ver, write_loc);
      write_loc += col->elem_size;
    }
  } else {
    for (auto& c_idx : *col_idx) {
      Column* col = columns.at(c_idx);
      // std::cout << "\t Reading col: " << col->name << std::endl;
      col->getElem(vid, master_ver, write_loc);
      write_loc += col->elem_size;
    }
  }
}

std::vector<const void*> ColumnStore::getRecordByKey(
    uint64_t vid, ushort master_ver, const std::vector<ushort>* col_idx) {
  if (col_idx == nullptr) {
    std::vector<const void*> record(columns.size());
    for (auto& col : columns) {
      record.push_back((const void*)(col->getElem(vid, master_ver)));
    }
    return record;
  } else {
    std::vector<const void*> record(col_idx->size());
    for (auto& c_idx : *col_idx) {
      Column* col = columns.at(c_idx);
      record.push_back((const void*)(col->getElem(vid, master_ver)));
    }
    return record;
  }
}

global_conf::mv_version_list* ColumnStore::getVersions(uint64_t vid,
                                                       ushort delta_ver) {
  assert(global_conf::cc_ismv);
  return this->deltaStore[delta_ver]->getVersionList(
      vid_to_uuid(this->table_id, vid));
}

void ColumnStore::updateRecord(uint64_t vid, const void* rec,
                               ushort ins_master_ver, ushort prev_master_ver,
                               ushort delta_ver, uint64_t tmin, uint64_t tmax,
                               ushort pid) {
  // if(ins_master_ver == prev_master_ver) need version ELSE update the master
  // one.
  // uint64_t c = 4;

  if (global_conf::cc_ismv) {
    // std::cout << "same master, create ver" << std::endl;
    // create_version
    char* ver = (char*)this->deltaStore[delta_ver]->insert_version(
        vid_to_uuid(this->table_id, vid), tmin, tmax, this->rec_size, pid);
    assert(ver != nullptr);
    // std::cout << "inserted into delta" << std::endl;
    size_t total_rec_size = 0;
    for (auto& col : columns) {
      size_t elem_size = col->elem_size;
      total_rec_size += elem_size;
      // std::cout << "attempting memcpy" << std::endl;
      // void* tcc = col->getElem(vid, prev_master_ver);
      assert(ver != nullptr);

      memcpy((void*)ver, col->getElem(vid, prev_master_ver), elem_size);
      // std::cout << "vid:" << vid << std::endl;
      // std::cout << "ptr:" << ((uint64_t*)col->getElem(vid, prev_master_ver))
      //          << std::endl;
      // std::cout << "val:" << *((uint64_t*)col->getElem(vid, prev_master_ver))
      //          << std::endl;
      // memcpy((void*)ver, &c, elem_size);
      ver += elem_size;
    }
    assert(this->rec_size == total_rec_size);
    // std::cout << "updated column" << std::endl;
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

void ColumnStore::updateRecord(uint64_t vid, const void* rec,
                               ushort ins_master_ver, ushort prev_master_ver,
                               ushort delta_ver, uint64_t tmin, uint64_t tmax,
                               ushort pid, std::vector<ushort>* col_idx) {
  if (global_conf::cc_ismv) {
    // create_version
    // std::cout << "UPD TBL: " << this->name << std::endl;
    char* ver = (char*)this->deltaStore[delta_ver]->insert_version(
        vid_to_uuid(this->table_id, vid), tmin, tmax, this->rec_size, pid);
    assert(ver != nullptr);
    size_t total_rec_size = 0;
    for (auto& col : columns) {
      size_t elem_size = col->elem_size;
      total_rec_size += elem_size;
      assert(ver != nullptr);
      memcpy((void*)ver, col->getElem(vid, prev_master_ver), elem_size);
      ver += elem_size;
    }
    assert(this->rec_size == total_rec_size);
    // std::cout << "updated column" << std::endl;
  }

  char* cursor = (char*)rec;
  for (auto& c_idx : *col_idx) {
    Column* col = columns.at(c_idx);
    col->insertElem(vid, (rec == nullptr ? nullptr : (void*)cursor),
                    ins_master_ver);
    cursor += col->elem_size;
  }
}

void Column::buildIndex() {
  // TODO: build column index here.

  this->is_indexed = true;
}

std::vector<std::pair<mem_chunk, uint64_t>> Column::snapshot_get_data(
    uint64_t* save_the_ptr) {
  // assert(master_version <= global_conf::num_master_versions);
  // return master_versions[this->arena->getMetadata().master_ver];

  std::vector<std::pair<mem_chunk, uint64_t>> ret;
  uint i = 0;
  for (auto& ar : arena) {
    for (const auto& chunk : master_versions[ar->getMetadata().master_ver][i]) {
#if HTAP_DOUBLE_MASTER
      std::cout << "SNAPD COL: " << this->name << std::endl;
      std::cout << "AR:" << ar->getMetadata().numOfRecords << std::endl;
      std::cout << "AR:MA: " << (uint)ar->getMetadata().master_ver << std::endl;
      ret.emplace_back(
          std::make_pair(mem_chunk(chunk.data, chunk.size, chunk.numa_id),
                         ar->getMetadata().numOfRecords));

      for (int j = 0; j < NUM_SOCKETS; j++) {
        this->parent->plugin_ptr[(int)ar->getMetadata().master_ver][j] =
            (save_the_ptr + j);
      }

#elif HTAP_COW

      ret.emplace_back(std::make_pair(
          mem_chunk(
              ar->olap(),
              (this->total_mem_reserved / global_conf::num_master_versions), 0),
          ar->getMetadata().numOfRecords));

#else
      assert(false && "Undefined snapshot mechanism");
#endif
    }
    i++;
  }
  assert(i == NUM_SOCKETS);
  // for (const auto& chunk :
  //      master_versions[this->arena->getMetadata().master_ver]) {
  //   ret.push_back(chunk);
  // }
  return ret;
}

// uint64_t Column::snapshot_get_num_records() {
//   return this->arena->getMetadata().numOfRecords;
// }

Column::Column(std::string name, uint64_t initial_num_records,
               ColumnStore* parent, data_type type, size_t unit_size,
               bool build_index, bool single_version_only)
    : name(name), parent(parent), elem_size(unit_size), type(type) {
  /*
  ALGO:
          - Allocate memory for that column with default start number of records
          - Create index on the column
*/

  // TODO: Allocating memory in the current socket.
  // size: initial_num_recs * unit_size
  // std::cout << "INITIAL NUM REC: " << initial_num_records << std::endl;
  // int numa_id = global_conf::master_col_numa_id;
  // std::cout << "Creating column: " << name << ", size:" << unit_size
  //          << std::endl;

  assert(FLAGS_num_partitions <= NUM_SOCKETS);
  size_t size = initial_num_records * unit_size;
  size_t size_per_partition =
      (((initial_num_records * unit_size) / FLAGS_num_partitions) + 1);
  this->total_mem_reserved = size * global_conf::num_master_versions;

  std::cout << "Col:" << name
            << ", Size required:" << ((double)size / (1024 * 1024 * 1024))
            << ", total: "
            << ((double)total_mem_reserved / (1024 * 1024 * 1024)) << std::endl;

  for (uint i = 0; i < FLAGS_num_partitions; i++) {
    arena.emplace_back(
        global_conf::SnapshotManager::create(size_per_partition));
  }

#if HTAP_COW

  for (ushort i = 0; i < NUM_SOCKETS, i++) {
    ar[i]->create_snapshot({0, 0});
    void* mem = ar->oltp();
    uint64_t* pt = (uint64_t*)mem;
    int warmup_max = size_per_partition / sizeof(uint64_t);
    for (int j = 0; j < warmup_max; j++) pt[j] = 0;
    master_versions[0].emplace_back(mem, size_per_partition, 0);
  }

  // for (ushort i = 0; i < NUM_SOCKETS; i++) {
  //   auto tmp_arena =
  //       global_conf::SnapshotManager::create(initial_num_records *
  //       unit_size);
  //   tmp_arena->create_snapshot({0, 0});
  //   arena.push_back(tmp_arena);
  //   void* mem = arena->oltp();

  //   uint64_t* pt = (uint64_t*)mem;
  //   int warmup_max = size / sizeof(uint64_t);
  //   for (int j = 0; j < warmup_max; i++) pt[j] = 0;
  //   master_versions[0].emplace_back(new mem_chunk(mem, size, i));
  // }

  // arena->create_snapshot({0, 0});

#else

  // std::cout << "Column--" << name << "| size: " << size
  //          << "| num_r: " << initial_num_records << std::endl;

#if HTAP_DOUBLE_MASTER
  // auto &topo = topology::getInstance();
  // auto &nodes = topo.getCpuNumaNodes();
  // exec_location{numa_node}.activate();
  auto& cpunumanodes = ::topology::getInstance().getCpuNumaNodes();
#endif

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (ushort j = 0; j < FLAGS_num_partitions; j++) {

#if HTAP_RM_SERVER
      std::cout << "HTAP REMOTE ALLOCATION: "
                << (std::to_string(i) + "__" + name) << std::endl;
      std::cout << "TABLE UNIT SIZE: " << unit_size << std::endl;
      void* mem = MemoryManager::alloc_shm_htap(
          std::to_string(i) + "__" + std::to_string(j) + "__" + name,
          size_per_partition, unit_size, j);

#elif SHARED_MEMORY
      void* mem = MemoryManager::alloc_shm(
          std::to_string(i) + "__" + std::to_string(j) + "__" + name,
          size_per_partition, j);
#elif HTAP_DOUBLE_MASTER

      set_exec_location_on_scope d{cpunumanodes[j]};
      void* mem = ::MemoryManager::mallocPinned(size_per_partition);

#else
      void* mem = MemoryManager::alloc(size_per_partition, j);
#endif

      uint64_t* pt = (uint64_t*)mem;
      int warmup_max = size_per_partition / sizeof(uint64_t);
      for (int j = 0; j < warmup_max; j++) pt[j] = 0;

      master_versions[i][j].emplace_back(mem, size_per_partition, j);
    }
    if (single_version_only) break;
  }

#endif

  if (build_index) this->buildIndex();
}

void* Column::getElem(uint64_t vid, ushort master_ver) {
  // ushort pid = vid % NUM_SOCKETS;
  // uint64_t idx = vid / NUM_SOCKETS;

  ushort pid = extract_pid(vid);
  uint64_t idx = extract_vid(vid);
  assert(master_versions[master_ver][pid].size() != 0);

  int data_idx = idx * elem_size;
  for (const auto& chunk : master_versions[master_ver][pid]) {
    if (chunk.size >= ((size_t)data_idx + elem_size)) {
      return ((char*)chunk.data) + data_idx;
    }
  }
  return nullptr;
}
void Column::touchElem(uint64_t vid, ushort master_ver) {
  // uint pid = vid % NUM_SOCKETS;
  // uint64_t idx = vid / NUM_SOCKETS;
  ushort pid = extract_pid(vid);
  uint64_t idx = extract_vid(vid);
  assert(master_versions[master_ver][pid].size() != 0);

  int data_idx = idx * elem_size;
  for (const auto& chunk : master_versions[master_ver][pid]) {
    if (chunk.size >= ((size_t)data_idx + elem_size)) {
      char* loc = ((char*)chunk.data) + data_idx;
#if HTAP_DOUBLE_MASTER
      set_upd_bit(loc);
#endif
      volatile char tmp = 'a';
      for (int i = 0; i < elem_size; i++) {
        tmp += *loc;
      }
    }
  }
}

void Column::getElem(uint64_t vid, ushort master_ver, void* copy_location) {
  // uint pid = vid % NUM_SOCKETS;
  // uint64_t idx = vid / NUM_SOCKETS;
  ushort pid = extract_pid(vid);
  uint64_t idx = extract_vid(vid);
  assert(master_versions[master_ver][pid].size() != 0);

  int data_idx = idx * elem_size;
  // std::cout << "GetElem-" << this->name << std::endl;
  for (const auto& chunk : master_versions[master_ver][pid]) {
    if (chunk.size >= ((size_t)data_idx + elem_size)) {
      std::memcpy(copy_location, ((char*)chunk.data) + data_idx,
                  this->elem_size);
      return;
    }
  }
  assert(false);  // as control should never reach here.
}

void Column::insertElem(uint64_t vid, void* elem, ushort master_ver) {
  // TODO: insert in both masters but set upd bit only in curr master.

  // uint pid = vid % NUM_SOCKETS;
  // uint64_t idx = vid / NUM_SOCKETS;
  ushort pid = extract_pid(vid);
  uint64_t idx = extract_vid(vid);

  uint64_t data_idx = idx * this->elem_size;
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    bool ins = false;
    for (const auto& chunk : master_versions[i][pid]) {
      if (chunk.size >= (data_idx + elem_size)) {
        // insert elem here
        void* dst = (void*)(((char*)chunk.data) + data_idx);
        if (elem == nullptr) {
          uint64_t* tptr = (uint64_t*)dst;
          (*tptr)++;

          *tptr = (*tptr) | ((uint64_t)1 << 63);
          // std::cout << "old:" << *((uint64_t*)dst) << "|new:" << *tptr
          //         << std::endl;
        } else {
          std::memcpy(dst, elem, this->elem_size);
#if HTAP_DOUBLE_MASTER
          if (i == master_ver) set_upd_bit((char*)dst);
#endif
        }
        ins = true;
        break;
      }
    }
    if (ins == false) {
      std::cout << "FUCK. ALLOCATE MORE MEMORY:\t" << this->name << std::endl;
    }
  }
  // exit(-1);
}

// For index/meta column
void* Column::insertElem(uint64_t vid) {
  // uint pid = vid % NUM_SOCKETS;
  // uint64_t idx = vid / NUM_SOCKETS;

  ushort pid = extract_pid(vid);
  uint64_t idx = extract_vid(vid);

  uint64_t data_idx = idx * this->elem_size;

  bool ins = false;
  for (const auto& chunk : master_versions[0][pid]) {
    // std::cout << "chunksize: " << chunk.size << std::endl;
    // std::cout << "dataidx: " << data_idx << std::endl;
    // std::cout << "elemsize: " << elem_size << std::endl;
    if (chunk.size >= (data_idx + elem_size)) {
      // insert elem here
      return (void*)(((char*)chunk.data) + data_idx);
    }
  }

  if (ins == false) {
    // std::cout << "res: " << this->total_mem_reserved << std::endl;
    // std::cout << "VID: " << vid << std::endl;
    // std::cout << "pid: " << pid << ", idx: " << idx << std::endl;
    std::cout << "FUCK. ALLOCATE MOTE MEMORY:\t" << this->name
              << ",vid: " << vid << ", idx:" << idx << ", pid: " << pid
              << std::endl;
  }

  assert(false && "Out Of Memory Error");

  return nullptr;
}

Column::~Column() {
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (ushort j = 0; j < FLAGS_num_partitions; j++) {
      for (auto& chunk : master_versions[i][j]) {
#if HTAP_DOUBLE_MASTER
        ::MemoryManager::freePinned(chunk.data);
#else
        MemoryManager::free(chunk.data, chunk.size);
#endif
      }
      master_versions[i][j].clear();
    }
  }
}

void ColumnStore::num_upd_tuples() {
  for (auto& col : this->columns) {
    col->num_upd_tuples();
  }
}

void Column::num_upd_tuples() {
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    uint64_t counter = 0;
    for (ushort j = 0; j < FLAGS_num_partitions; j++) {
      for (auto& chunk : master_versions[i][j]) {
        for (uint i = 0; i < (chunk.size / elem_size); i++) {
          uint8_t* p = ((uint8_t*)chunk.data) + i;
          if (*p >> 7 == 1) {
            counter++;
          }
        }
      }
    }
    std::cout << "UPD[" << i << "]: COL:" << this->name
              << " | #num_upd: " << counter << std::endl;
    counter = 0;
  }
}

uint64_t Column::load_from_binary(std::string file_path) {
  std::ifstream binFile(file_path.c_str(), std::ifstream::binary);
  // std::cout << "Loading binary file: " << file_path << std::endl;
  if (binFile) {
    // get length of file
    binFile.seekg(0, binFile.end);
    size_t length = static_cast<size_t>(binFile.tellg());
    // std::cout << "\tContains " << (length / this->elem_size) << " elements."
    //           << std::endl;

    for (ushort i = 0; i < global_conf::num_master_versions; i++) {
      binFile.seekg(0, binFile.beg);

      for (ushort j = 0; j < FLAGS_num_partitions; j++) {
        size_t part_size = length / FLAGS_num_partitions;
        if (j == (FLAGS_num_partitions - 1)) {
          // remaining shit.
          part_size += part_size % FLAGS_num_partitions;
        }

        // assumes first memory chunk is big enough.
        if (master_versions[i][j][0].size <= part_size) {
          std::cout << "Failed loading binary file: " << file_path << std::endl;
          std::cout << "\tpart_size " << part_size << std::endl;
          std::cout << "\tchunk_size: " << master_versions[i][j][0].size
                    << std::endl;
        }

        assert(master_versions[i][j][0].size > part_size);
        char* tmp = (char*)master_versions[i][j][0].data;
        binFile.read(tmp, part_size);
      }
    }

    binFile.close();
    return (length / this->elem_size);
  }
  assert(false);
}

};  // namespace storage
