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

#include <cassert>
#include <iostream>
#include <string>
#include "storage/table.hpp"

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "storage/delta_storage.hpp"
#include "storage/row_store.hpp"

/*

  TODO:
    - resizeable storage
    - partitionable storage
*/

namespace storage {

inline uint64_t __attribute__((always_inline))
vid_to_uuid(uint8_t tbl_id, uint64_t vid) {
  return (vid & 0x00FFFFFFFFFFFFFF) | (tbl_id < 56);
}

std::vector<const void*> rowStore::getRecordByKey(
    uint64_t vid, ushort master_ver, const std::vector<ushort>* col_idx) {
  

  std::vector<const void*> ret;

  char* rw = (char*)this->getRow(vid, master_ver);

  if (col_idx == nullptr) {
    for (auto& cw : this->column_width) {
      ret.emplace_back(rw + cw.second);
    }

  } else {
    for (auto& c_idx : *col_idx) {
      std::pair<size_t, size_t> sz = column_width.at(c_idx);
      ret.emplace_back(rw + sz.second);
    }
  }

  return ret;
}

void rowStore::getRecordByKey(uint64_t vid, ushort master_ver,
                              const std::vector<ushort>* col_idx, void* loc) {
  char* rw = (char*)this->getRow(vid, master_ver);
  char* wloc = (char*)loc;

  if (col_idx == nullptr) {
    memcpy(loc, rw, this->rec_size);
  } else {
    size_t offset = 0;
    for (auto& c_idx : *col_idx) {
      std::pair<size_t, size_t> sz = column_width.at(c_idx);
      memcpy(wloc + offset, rw + sz.second, sz.first);
      offset += sz.first;
    }
  }
}

void rowStore::touchRecordByKey(uint64_t vid, ushort master_ver) {
  uint64_t data_idx = vid * this->rec_size;

  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= (data_idx + this->rec_size)) {
      // insert elem here
      void* dst = (void*)(((char*)chunk->data) + data_idx);

      uint64_t* tptr = (uint64_t*)dst;
      for (ushort i = 0; i < this->num_columns; i++) {
        tptr += i;
        (*tptr)++;
      }

      return;
    }
  }
  assert(false && "Record does not exists");
}

global_conf::mv_version_list* rowStore::getVersions(uint64_t vid,
                                                    ushort delta_ver) {
  assert(global_conf::cc_ismv);
  return this->deltaStore[delta_ver]->getVersionList(
      vid_to_uuid(this->table_id, vid));
}

void rowStore::updateRecord(uint64_t vid, const void* data,
                            ushort ins_master_ver, ushort prev_master_ver,
                            ushort delta_ver, uint64_t tmin, uint64_t tmax,
                            ushort pid) {
  if (global_conf::cc_ismv) {
    // create_version
    char* ver = (char*)this->deltaStore[delta_ver]->insert_version(
        vid_to_uuid(this->table_id, vid), tmin, tmax, this->rec_size, pid);
    assert(ver != nullptr);

    memcpy((void*)ver, this->getRow(vid, prev_master_ver), this->rec_size);
  }

  this->insert_or_update(vid, data, ins_master_ver);
}
void rowStore::updateRecord(uint64_t vid, const void* data,
                            ushort ins_master_ver, ushort prev_master_ver,
                            ushort delta_ver, uint64_t tmin, uint64_t tmax,
                            ushort pid, std::vector<ushort>* col_idx) {
  if (global_conf::cc_ismv) {
    // create_version
    char* ver = (char*)this->deltaStore[delta_ver]->insert_version(
        vid_to_uuid(this->table_id, vid), tmin, tmax, this->rec_size, pid);
    assert(ver != nullptr);

    memcpy((void*)ver, this->getRow(vid, prev_master_ver), this->rec_size);
  }

  this->update_partial(vid, data, ins_master_ver, col_idx);
}

void rowStore::update_partial(uint64_t vid, const void* data, ushort master_ver,
                              const std::vector<ushort>* col_idx) {
  uint64_t data_idx = vid * this->rec_size;

  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= (data_idx + this->rec_size)) {
      // insert elem here
      char* dst = ((char*)chunk->data) + data_idx;
      if (data == nullptr) {
        size_t offset = 0;
        for (auto& c_idx : *col_idx) {
          std::pair<size_t, size_t> sz = column_width.at(c_idx);

          uint64_t* tptr = (uint64_t*)((char*)data + offset);
          (*tptr)++;
          offset += sz.first;
        }
      } else {
        size_t offset = 0;
        for (auto& c_idx : *col_idx) {
          std::pair<size_t, size_t> sz = column_width.at(c_idx);
          memcpy(dst + sz.second, (char*)data + offset, sz.first);
          offset += sz.first;
        }
      }
      return;
    }
  }
  assert(false && "Memory limit exceeded, allocate more memory for storage");
}

void rowStore::insert_or_update(uint64_t vid, const void* rec,
                                ushort master_ver) {
  uint64_t data_idx = vid * this->rec_size;

  for (const auto& chunk : master_versions[master_ver]) {
    if (chunk->size >= (data_idx + this->rec_size)) {
      // insert elem here
      void* dst = (void*)(((char*)chunk->data) + data_idx);
      if (rec == nullptr) {
        uint64_t* tptr = (uint64_t*)dst;
        for (ushort i = 0; i < this->num_columns; i++) {
          tptr += i;
          (*tptr)++;
        }
      } else {
        std::memcpy(dst, rec, this->rec_size);
      }
      return;
    }
  }
  assert(false && "Memory limit exceeded, allocate more memory for storage");
}

uint64_t rowStore::insertRecord(void* rec, ushort master_ver) {
  // TODO: update bit for snapshotting
  uint64_t curr_vid = vid.fetch_add(1);
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    this->insert_or_update(curr_vid, rec, i);
  }

  return curr_vid;
}

void* rowStore::insertRecord(void* rec, uint64_t xid, ushort master_ver) {
  uint64_t curr_vid = vid.fetch_add(1);

  global_conf::IndexVal hash_val(xid, curr_vid, master_ver);
  void* hash_ptr =
      this->meta_column->insertElem(curr_vid, &hash_val, master_ver);

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    this->insert_or_update(curr_vid, rec, i);
  }
  return hash_ptr;
}

rowStore::rowStore(
    uint8_t table_id, std::string name,
    std::vector<std::tuple<std::string, data_type, size_t>> columns,
    uint64_t initial_num_records) {
  this->table_id = table_id;
  this->total_mem_reserved = 0;
  this->vid = 0;
  this->deltaStore = storage::Schema::getInstance().deltaStore;
  this->rec_size = 0;
  int numa_id = global_conf::master_col_numa_id;

  meta_column = new Column("meta", initial_num_records, META,
                           sizeof(global_conf::IndexVal));
  this->total_mem_reserved += meta_column->getSize();
  for (const auto& t : columns) {
    this->columns.emplace_back(std::get<0>(t));
    this->column_data_types.emplace_back(std::get<1>(t));

    this->column_width.emplace_back(
        std::pair<size_t, size_t>(std::get<2>(t), this->rec_size));

    this->rec_size += std::get<2>(t);
  }

  this->num_columns = columns.size();
  this->name = name;
  size_t size = this->rec_size * initial_num_records;

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    // void* mem = MemoryManager::alloc(size, numa_id);
    void* mem = MemoryManager::alloc_shm(name, size, numa_id);

    uint64_t* pt = (uint64_t*)mem;
    int warmup_max = size / sizeof(uint64_t);
    for (uint i = 0; i < warmup_max; i++) pt[i] = 0;
    master_versions[i].emplace_back(new mem_chunk(mem, size, numa_id));

    this->total_mem_reserved += size;
  }

  // build index over the first column
  this->p_index =
      new global_conf::PrimaryIndex<uint64_t>(initial_num_records + 1);

  std::cout << "Table: " << name << std::endl;
  std::cout << "\trecord size: " << rec_size << " bytes" << std::endl;
  std::cout << "\tnum_records: " << initial_num_records << std::endl;
  std::cout << "\tMem reserved: "
            << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
            << std::endl;
}

};  // namespace storage
