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
#include "storage/delta_storage.hpp"

/*

  TODO:
    - resizeable columns
    - partitionable columns
    - dont create meta for each master version. uneecary
*/

namespace storage {

static inline void set_upd_bit(const void* data) {
  uint8_t* p = (uint8_t*)data;
  *p = (*p) | (1 << 7);
}

inline void Schema::snapshot(uint64_t epoch) {
  for (auto& tbl : tables) {
    tbl->snapshot(epoch);
  }
}
inline void ColumnStore::snapshot(uint64_t epoch) {
  uint64_t num_records = this->vid.load() - 1;
  for (auto& col : columns) {
    col->snapshot(epoch, num_records);
  }
}

inline void Column::snapshot(uint64_t num_records, uint64_t epoch) {
  arena->destroy_snapshot();
  arena->create_snapshot({num_records, epoch});
}

std::vector<Table*> Schema::getAllTable() { return tables; }

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
    uint64_t initial_num_records) {
  Table* tbl = nullptr;

  if (layout == COLUMN_STORE) {
    tbl = new ColumnStore((this->num_tables + 1), name, columns,
                          initial_num_records);

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
  std::cout << "[Schema][drop_table] Not Implemented" << std::endl;
}

Table::~Table() {}

ColumnStore::~ColumnStore() {
  for (auto& col : columns) {
    delete col;
  }
  delete meta_column;
}

ColumnStore::ColumnStore(
    uint8_t table_id, std::string name,
    std::vector<std::tuple<std::string, data_type, size_t>> columns,
    uint64_t initial_num_records)
    : Table(name, table_id) {
  /*
          TODO: take an argument for column_index or maybe a flag in the tuple
     for the indexed columns. Currently, by default, the first column would be
     considered as key and will be indexed.
  */

  this->total_mem_reserved = 0;
  this->vid = 0;
  this->deltaStore = storage::Schema::getInstance().deltaStore;

  // create meta_data column.
  // std::cout << "Create meta-column" << std::endl;
  // std::cout << "size of hash_val: " << sizeof(global_conf::IndexVal)
  //         << std::endl;
  meta_column = new Column(name + "_meta", initial_num_records, META,
                           sizeof(global_conf::IndexVal));
  // create columns
  for (const auto& t : columns) {
    // std::cout << "Creating Column: " << std::get<0>(t) << std::endl;
    this->columns.emplace_back(new Column(std::get<0>(t), initial_num_records,
                                          std::get<1>(t), std::get<2>(t),
                                          false));
  }
  for (const auto& t : this->columns) {
    total_mem_reserved += t->total_mem_reserved;
  }

  this->num_columns = columns.size();
  // build index over the first column
  this->p_index =
      new global_conf::PrimaryIndex<uint64_t>(initial_num_records + 1);
  this->columns.at(0)->buildIndex();

  // Secondary Indexes
  // this->s_index = new
  // global_conf::PrimaryIndex<uint64_t>()[num_secondary_indexes];
  size_t rec_size = 0;
  for (auto& co : this->columns) {
    rec_size += co->elem_size;
  }
  this->rec_size = rec_size;

  std::cout << "Table: " << name << std::endl;
  std::cout << "\trecord size: " << rec_size << " bytes" << std::endl;
  std::cout << "\tnum_records: " << initial_num_records << std::endl;
  total_mem_reserved += meta_column->total_mem_reserved;
  std::cout << "\tMem reserved: "
            << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
            << std::endl;
}

// void* ColumnStore::insertMeta(uint64_t vid, global_conf::IndexVal& hash_val)
// {}

/* Following function assumes that the  void* rec has columns in the same order
 * as the actual columns
 */
void* ColumnStore::insertRecord(void* rec, uint64_t xid, ushort master_ver) {
  uint64_t curr_vid = vid.fetch_add(1);

  void* pano = this->meta_column->insertElem(curr_vid);

  global_conf::IndexVal* hash_ptr =
      new (pano) global_conf::IndexVal(xid, curr_vid, master_ver);

  // s = new (pano) std::atomic<uint64_t>;
  // void *insertElem(uint64_t offset);

  // global_conf::IndexVal hash_val(xid, curr_vid, master_ver);
  // void* hash_ptr =
  //    this->meta_column->insertElem(curr_vid, &hash_val, master_ver);

  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(curr_vid, rec_ptr, master_ver);
    rec_ptr += col->elem_size;
  }
  return (void*)hash_ptr;
}

uint64_t ColumnStore::insertRecord(void* rec, ushort master_ver) {
  uint64_t curr_vid = vid.fetch_add(1);

  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(curr_vid, rec_ptr, master_ver);
    rec_ptr += col->elem_size;
  }
  return curr_vid;
}

void ColumnStore::deleteRecord(uint64_t vid, ushort master_ver) {}

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

inline uint64_t __attribute__((always_inline))
vid_to_uuid(uint8_t tbl_id, uint64_t vid) {
  return (vid & 0x00FFFFFFFFFFFFFF) | (tbl_id < 56);
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

Column::Column(std::string name, uint64_t initial_num_records, data_type type,
               size_t unit_size, bool build_index, bool single_version_only)
    : name(name),
      elem_size(unit_size),
      type(type),
      arena(global_conf::SnapshotManager::create(initial_num_records *
                                                 unit_size)) {
  /*
  ALGO:
          - Allocate memory for that column with default start number of records
          - Create index on the column
*/

  // TODO: Allocating memory in the current socket.
  // size: initial_num_recs * unit_size
  // std::cout << "INITIAL NUM REC: " << initial_num_records << std::endl;
  int numa_id = global_conf::master_col_numa_id;
  size_t size = initial_num_records * unit_size;
  this->total_mem_reserved = size * global_conf::num_master_versions;
  arena->create_snapshot({0, 0});
  // std::cout << "Column--" << name << "| size: " << size
  //          << "| num_r: " << initial_num_records << std::endl;

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
#if HTAP_COW
    void* mem = arena->oltp();

#elif HTAP
    std::cout << "HTAP REMOTE ALLOCATION: " << (std::to_string(i) + "__" + name)
              << std::endl;
    std::cout << "TABLE UNIT SIZE: " << unit_size << std::endl;
    void* mem = MemoryManager::alloc_shm_htap(std::to_string(i) + "__" + name,
                                              size, unit_size, i);

#elif SHARED_MEMORY
    void* mem =
        MemoryManager::alloc_shm(std::to_string(i) + "__" + name, size, i);

#else
    // void* mem = nullptr;
    // if (name.find("_meta") == std::string::npos) {
    //   std::cout << "Special Column: " << name << std::endl;
    //   mem = MemoryManager::alloc(size, 18);
    // } else
    //   mem = MemoryManager::alloc(size, i);

    void* mem = MemoryManager::alloc(size, i);

#endif

    uint64_t* pt = (uint64_t*)mem;
    int warmup_max = size / sizeof(uint64_t);
    for (int i = 0; i < warmup_max; i++) pt[i] = 0;
    master_versions[i].emplace_back(new mem_chunk(mem, size, numa_id));

    if (single_version_only) break;
  }

  if (build_index) this->buildIndex();
}

void* Column::getElem(uint64_t vid, ushort master_ver) {
  // master_versions[master_ver];
  assert(master_versions[master_ver].size() != 0);
  // std::cout << "getElem -> " << vid << " | master:" << master_ver <<
  // std::endl;
  int data_idx = vid * elem_size;
  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= ((size_t)data_idx + elem_size)) {
      return ((char*)chunk->data) + data_idx;
    }
  }
  return nullptr;
}
void Column::touchElem(uint64_t vid, ushort master_ver) {
  // master_versions[master_ver];
  assert(master_versions[master_ver].size() != 0);
  // std::cout << "getElem -> " << vid << " | master:" << master_ver <<
  // std::endl;
  int data_idx = vid * elem_size;
  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= ((size_t)data_idx + elem_size)) {
      char* loc = ((char*)chunk->data) + data_idx;
#if HTAP_UPD_BIT_MASK
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
  // master_versions[master_ver];
  assert(master_versions[master_ver].size() != 0);
  int data_idx = vid * elem_size;
  // std::cout << "GetElem-" << this->name << std::endl;
  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= ((size_t)data_idx + elem_size)) {
      std::memcpy(copy_location, ((char*)chunk->data) + data_idx,
                  this->elem_size);
      return;
    }
  }
  assert(false);  // as control should never reach here.
}

void Column::insertElem(uint64_t offset, void* elem, ushort master_ver) {
  // TODO: insert in both masters but set upd bit only in curr master.
  uint64_t data_idx = offset * this->elem_size;
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    bool ins = false;
    for (const auto& chunk : master_versions[i]) {
      if (chunk->size >= (data_idx + elem_size)) {
        // insert elem here
        void* dst = (void*)(((char*)chunk->data) + data_idx);
        if (elem == nullptr) {
          uint64_t* tptr = (uint64_t*)dst;
          (*tptr)++;

          *tptr = (*tptr) | ((uint64_t)1 << 63);
          // std::cout << "old:" << *((uint64_t*)dst) << "|new:" << *tptr
          //         << std::endl;
        } else {
          std::memcpy(dst, elem, this->elem_size);
#if HTAP_UPD_BIT_MASK
          if (i == master_ver) {
            char* tt = (char*)dst;
            *tt = *tt | (1 << 7);
          }
#endif
        }
        ins = true;
        break;
      }
    }
    if (ins == false) {
      std::cout << "FUCK. ALLOCATE MOTE MEMORY:\t" << this->name << std::endl;
    }
  }
  // exit(-1);
}

// For index/meta column
void* Column::insertElem(uint64_t offset) {
  uint64_t data_idx = offset * this->elem_size;

  bool ins = false;
  for (const auto& chunk : master_versions[0]) {
    if (chunk->size >= (data_idx + elem_size)) {
      // insert elem here
      return (void*)(((char*)chunk->data) + data_idx);
    }
  }

  if (ins == false) {
    std::cout << "FUCK. ALLOCATE MOTE MEMORY:\t" << this->name << std::endl;
  }

  assert(false && "Out Of Memory Error");

  return nullptr;
}

Column::~Column() {
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (auto& chunk : master_versions[i]) {
      MemoryManager::free(chunk->data, chunk->size);
    }
  }
}

void ColumnStore::num_upd_tuples() {
  for (auto& col : this->columns) {
    col->num_upd_tuples();
  }
}

void Column::num_upd_tuples() {
  for (uint master_ver = 0; master_ver < global_conf::num_master_versions;
       master_ver++) {
    uint64_t counter = 0;
    for (const auto& chunk : master_versions[master_ver]) {
      for (uint i = 0; i < (chunk->size / elem_size); i++) {
        uint8_t* p = ((uint8_t*)chunk->data) + i;
        if (*p >> 7 == 1) {
          counter++;
        }
      }
    }

    std::cout << "UPD[" << master_ver << "]: COL:" << this->name
              << " | #num_upd: " << counter << std::endl;
    counter = 0;
  }
}

};  // namespace storage
