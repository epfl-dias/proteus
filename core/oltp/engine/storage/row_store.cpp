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

#include "storage/row_store.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "storage/delta_storage.hpp"
#include "storage/table.hpp"

#if PROTEUS_MEM_MANAGER
#include "memory/memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#endif

namespace storage {

std::mutex row_store_print_mutex;

static inline void __attribute__((always_inline)) set_upd_bit(char* data) {
  *data = *data | (1 << 7);
}

static inline uint64_t __attribute__((always_inline))
CC_gen_vid(uint64_t vid, ushort partition_id, ushort master_ver,
           ushort delta_version) {
  return ((vid & 0x000000FFFFFFFFFF) |
          ((uint64_t)(partition_id & 0x00FF) << 40) |
          ((uint64_t)(master_ver & 0x00FF) << 48) |
          ((uint64_t)(delta_version & 0x00FF) << 56));
}

static inline uint64_t __attribute__((always_inline))
CC_upd_vid(uint64_t vid, ushort master_ver, ushort delta_version) {
  return ((vid & 0x0000FFFFFFFFFFFF) | ((uint64_t)(master_ver & 0x00FF) << 48) |
          ((uint64_t)(delta_version & 0x00FF) << 56));
}

void RowStore::updateRecord(global_conf::IndexVal* hash_ptr, const void* rec,
                            ushort curr_master, ushort curr_delta,
                            const ushort* col_idx, short num_cols) {
  ushort pid = CC_extract_pid(hash_ptr->VID);
  ushort m_ver = CC_extract_m_ver(hash_ptr->VID);
  uint64_t data_idx = CC_extract_offset(hash_ptr->VID) * this->rec_size;

  assert(data_idx <= this->data[m_ver][pid][0].size);
  // assert(CC_extract_offset(hash_ptr->VID) < initial_num_records_per_part);
  // assert(m_ver < global_conf::num_master_versions);
  // assert(curr_delta < global_conf::num_delta_storages);

  // char* limit =
  //     (char*)(this->data[m_ver][pid][0].data) +
  //     this->data[m_ver][pid][0].size;

  char* rec_ptr = ((char*)(this->data[m_ver][pid][0].data)) + data_idx;

  // delta versioning

  char* ver = (char*)this->deltaStore[curr_delta]->insert_version(
      hash_ptr, this->rec_size - HTAP_UPD_BIT_COUNT, pid);  // tmax==0

  // char* ver = (char*)this->deltaStore[curr_delta]->insert_version(
  //     vid_to_uuid(this->table_id, hash_ptr->VID), hash_ptr->t_min, 0,
  //     this->rec_size,
  //     pid);  // tmax==0
  assert(ver != nullptr);

  memcpy(ver, rec_ptr + HTAP_UPD_BIT_COUNT,
         this->rec_size - HTAP_UPD_BIT_COUNT);
  // end--delta versioning

  hash_ptr->VID = CC_upd_vid(hash_ptr->VID, curr_master, curr_delta);
  // assert(CC_extract_offset(hash_ptr->VID) < initial_num_records_per_part);

#if HTAP_DOUBLE_MASTER
  set_upd_bit(rec_ptr);
#endif

  if (__unlikely(num_cols <= 0)) {
    if (__likely(rec == nullptr || rec == NULL)) {
      for (ushort i = 0; i < columns.size(); i++) {
        uint64_t* tptr = (uint64_t*)(rec_ptr + column_width.at(i).second);
        (*tptr)++;
      }

    } else {
      memcpy(rec_ptr + HTAP_UPD_BIT_COUNT, rec,
             this->rec_size - HTAP_UPD_BIT_COUNT);
    }

  } else {
    char* write_loc = (char*)rec_ptr;
    char* read_loc = (char*)rec;

    for (int i = 0; i < num_cols; i++) {
      memcpy(write_loc, read_loc + column_width.at(col_idx[i]).second,
             column_width.at(col_idx[i]).first);

      write_loc += column_width.at(col_idx[i]).first;
      // assert(write_loc <= limit);
    }
  }
}

void RowStore::getRecordByKey(uint64_t vid, const ushort* col_idx,
                              ushort num_cols, void* loc) {
  ushort pid = CC_extract_pid(vid);
  ushort m_ver = CC_extract_m_ver(vid);
  uint64_t data_idx = CC_extract_offset(vid) * this->rec_size;
  // std::cout << "TBL: " << name << " -- " << num_cols << std::endl;

  char* src = ((char*)(this->data[m_ver][pid][0].data) + data_idx);

  if (__unlikely(col_idx == nullptr)) {
    memcpy(loc, src + HTAP_UPD_BIT_COUNT, this->rec_size - HTAP_UPD_BIT_COUNT);

  } else {
    // std::cout << name << " : " << this->columns.size() << std::endl;

    char* write_loc = (char*)loc;
    for (ushort i = 0; i < num_cols; i++) {
      // std::cout << i << " :+: " << col_idx[i] << std::endl;
      // std::cout << columns.at(i) << " ::: " <<
      // column_width.at(col_idx[i]).first
      //           << " zz " << column_width.at(col_idx[i]).second << std::endl;
      // // std::cout << columns.at(col_idx[i])
      // //           << " w: " << column_width.at(col_idx[i]).first <<
      // std::endl;
      memcpy(write_loc, src + column_width.at(col_idx[i]).second,
             column_width.at(col_idx[i]).first);
      write_loc += column_width.at(col_idx[i]).first;
    }
  }
}

void RowStore::touchRecordByKey(uint64_t vid) {
  ushort pid = CC_extract_pid(vid);
  ushort m_ver = CC_extract_m_ver(vid);
  uint64_t data_idx = CC_extract_offset(vid) * this->rec_size;

  assert(data[m_ver][pid].size() != 0);

  for (const auto& chunk : data[m_ver][pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + this->rec_size))) {
      char* loc = ((char*)chunk.data) + data_idx;

#if HTAP_DOUBLE_MASTER
      set_upd_bit(loc);
#endif
      loc += HTAP_UPD_BIT_COUNT;

      volatile char tmp = 'a';
      for (int i = 1; i < this->rec_size; i++) {
        tmp += *loc;
      }
    }
  }
}

void* RowStore::insertRecordBatch(void* rec_batch, uint recs_to_ins,
                                  uint capacity_offset, uint64_t xid,
                                  ushort partition_id, ushort master_ver) {
  uint64_t idx_st = vid[partition_id].fetch_add(recs_to_ins);
  // get batch from meta column
  uint64_t st_vid = CC_gen_vid(idx_st, partition_id, master_ver, 0);

  // meta stuff
  global_conf::IndexVal* hash_ptr =
      (global_conf::IndexVal*)((char*)(this->metadata[partition_id][0].data) +
                               (idx_st * sizeof(global_conf::IndexVal)));
  assert(hash_ptr != nullptr);
  for (uint i = 0; i < recs_to_ins; i++) {
    hash_ptr->t_min = xid;
    hash_ptr->VID = CC_gen_vid(idx_st + i, partition_id, master_ver, 0);
    hash_ptr += 1;
  }

  for (uint i = 0; i < recs_to_ins; i++) {
    char* dst = ((char*)(this->data[master_ver][partition_id][0].data) +
                 (idx_st + (i * this->rec_size)) + HTAP_UPD_BIT_COUNT);

#if HTAP_DOUBLE_MASTER
    set_upd_bit(dst - HTAP_UPD_BIT_COUNT);
#endif

    void* src = (char*)rec_batch + (i * this->rec_size);

    memcpy(dst, src, this->rec_size - HTAP_UPD_BIT_COUNT);
  }

  return (void*)hash_ptr;

  // return starting address of batch meta.
}

void* RowStore::insertRecord(void* rec, uint64_t xid, ushort partition_id,
                             ushort master_ver) {
  uint64_t idx = vid[partition_id].fetch_add(1);

  uint64_t curr_vid = CC_gen_vid(idx, partition_id, master_ver, 0);

  global_conf::IndexVal* hash_ptr = nullptr;

  assert(idx <= this->initial_num_records_per_part);

  // copy meta
  if (indexed) {
    // void* pano = ((char*)(this->metadata[partition_id][0].data) +
    //               (idx * sizeof(global_conf::IndexVal)));
    // hash_ptr = new (pano) global_conf::IndexVal(xid, curr_vid);

    hash_ptr =
        (global_conf::IndexVal*)((char*)(this->metadata[partition_id][0].data) +
                                 (idx * sizeof(global_conf::IndexVal)));

    assert(hash_ptr != nullptr);
    hash_ptr->t_min = xid;
    hash_ptr->VID = curr_vid;
  }

  // Copy data
  char* dst = ((char*)(this->data[master_ver][partition_id][0].data) +
               (idx * this->rec_size) + HTAP_UPD_BIT_COUNT);
#if HTAP_DOUBLE_MASTER
  set_upd_bit(dst - HTAP_UPD_BIT_COUNT);
#endif
  memcpy(dst, rec, this->rec_size - HTAP_UPD_BIT_COUNT);

  return (void*)hash_ptr;
}

uint64_t RowStore::insertRecord(void* rec, ushort partition_id,
                                ushort master_ver) {
  uint64_t idx = vid[partition_id].fetch_add(1);

  uint64_t curr_vid = CC_gen_vid(idx, partition_id, master_ver, 0);

  // Copy data
  char* dst = ((char*)(this->data[master_ver][partition_id][0].data) +
               (idx * this->rec_size) + HTAP_UPD_BIT_COUNT);

#if HTAP_DOUBLE_MASTER
  set_upd_bit(dst - HTAP_UPD_BIT_COUNT);
#endif

  memcpy(dst, rec, this->rec_size - HTAP_UPD_BIT_COUNT);

  return curr_vid;
}

// global_conf::mv_version_list* RowStore::getVersions(uint64_t vid) {
//   assert(CC_extract_delta_id(vid) < global_conf::num_delta_storages);
//   return this->deltaStore[CC_extract_delta_id(vid)]->getVersionList(
//       vid_to_uuid(this->table_id, vid));
// }

void RowStore::initializeMetaColumn() {
  std::vector<std::thread> loaders;
  size_t elem_size = sizeof(global_conf::IndexVal);

  for (ushort j = 0; j < this->num_partitions; j++) {
    for (const auto& chunk : metadata[j]) {
      char* ptr = (char*)chunk.data;
      assert(chunk.size % elem_size == 0);
      loaders.emplace_back([chunk, j, ptr, elem_size]() {
        for (uint64_t i = 0; i < (chunk.size / elem_size); i++) {
          void* c = new (ptr + (i * elem_size))
              global_conf::IndexVal(0, CC_gen_vid(i, j, 0, 0));
        }
      });
    }
  }
  for (auto& th : loaders) {
    th.join();
  }
}

RowStore::RowStore(uint8_t table_id, std::string name, ColumnDef columns,
                   uint64_t initial_num_records, bool indexed, bool partitioned,
                   int numa_idx)
    : Table(name, table_id, ROW_STORE, columns),
      indexed(indexed),
      initial_num_records(initial_num_records) {
  this->rec_size = HTAP_UPD_BIT_COUNT;  // upd bit
  this->total_mem_reserved = 0;
  this->deltaStore = storage::Schema::getInstance().deltaStore;
  this->vid_offset = 0;

  for (int i = 0; i < g_num_partitions; i++) this->vid[i] = 0;

  if (indexed) {
    // this->rec_size += sizeof(global_conf::IndexVal);
    // meta_column = new Column(name + "_meta", initial_num_records, this, META,
    //                          sizeof(global_conf::IndexVal), 0, true,
    //                          partitioned, numa_idx);

    void* obj_data = MemoryManager::alloc(
        sizeof(global_conf::PrimaryIndex<uint64_t>),
        storage::NUMAPartitionPolicy::getInstance().getDefaultPartition(),
        MADV_DONTFORK);

    this->p_index = new (obj_data)
        global_conf::PrimaryIndex<uint64_t>(name, initial_num_records);
  }

  // create columns
  size_t col_offset = 0;
  for (const auto& t : columns.getColumns()) {
    this->columns.emplace_back(std::get<0>(t));
    this->column_width.emplace_back(
        std::make_pair(std::get<2>(t), this->rec_size));
    // std::cout << "C: " << std::get<0>(t) << " -- " << std::get<2>(t)
    //           << " == " << this->rec_size << std::endl;

    this->rec_size += std::get<2>(t);
    this->column_data_types.emplace_back(std::get<1>(t));
  }
  this->num_columns = columns.size();

  if (partitioned)
    this->num_partitions = g_num_partitions;
  else
    this->num_partitions = 1;

  assert(g_num_partitions <= NUM_SOCKETS);
  size_t tsize = initial_num_records * this->rec_size;
  this->size_per_part = tsize / this->num_partitions;
  this->initial_num_records_per_part =
      initial_num_records / this->num_partitions;

#if PROTEUS_MEM_MANAGER
  const auto& cpunumanodes = ::topology::getInstance().getCpuNumaNodes();
#endif

  // Memory Allocation
  std::vector<std::thread> loaders;

  // meta-data

  size_t meta_tsize = initial_num_records * sizeof(global_conf::IndexVal);
  size_t meta_size_per_part = meta_tsize / num_partitions;

  for (ushort j = 0; j < this->num_partitions; j++) {
#if PROTEUS_MEM_MANAGER

    set_exec_location_on_scope d{cpunumanodes[j]};
    void* mem = ::MemoryManager::mallocPinned(meta_size_per_part);

#else

    void* mem = MemoryManager::alloc(meta_size_per_part, j);

#endif

    assert(mem != nullptr || mem != NULL);
    //     loaders.emplace_back([mem, meta_size_per_part]() {
    //       uint64_t* pt = (uint64_t*)mem;
    //       uint64_t warmup_max = meta_size_per_part / sizeof(uint64_t);
    // #pragma clang loop vectorize(enable)
    //       for (uint64_t j = 0; j < warmup_max; j++) pt[j] = 0;
    //     });

    metadata[j].emplace_back(mem, meta_size_per_part, j);
  }

  loaders.emplace_back([this]() { this->initializeMetaColumn(); });

  // data
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (ushort j = 0; j < this->num_partitions; j++) {
#if PROTEUS_MEM_MANAGER

      set_exec_location_on_scope d{cpunumanodes[j]};
      void* mem = ::MemoryManager::mallocPinned(size_per_part);

#else

      void* mem = MemoryManager::alloc(size_per_part, j);

#endif

      assert(mem != nullptr || mem != NULL);
      loaders.emplace_back([mem, this]() {
        uint64_t* pt = (uint64_t*)mem;
        uint64_t warmup_max = size_per_part / sizeof(uint64_t);
#pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < warmup_max; j++) pt[j] = 0;
      });

      data[i][j].emplace_back(mem, size_per_part, j);
    }
  }

  for (auto& th : loaders) {
    th.join();
  }

  total_mem_reserved += tsize;

  {
    std::unique_lock<std::mutex> lk(row_store_print_mutex);
    std::cout << "Table: " << name << std::endl;
    std::cout << "\trecord size: " << rec_size << " bytes" << std::endl;
    std::cout << "\tnum_records: " << this->initial_num_records << std::endl;
    std::cout << "\tMem reserved (Data): "
              << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
              << std::endl;
    std::cout << "\tMem reserved (Meta): "
              << (double)meta_tsize / (1024 * 1024 * 1024) << "GB" << std::endl;

    total_mem_reserved += meta_tsize;
    std::cout << "\tMem reserved (Total): "
              << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
              << std::endl;
  }
}

};  // namespace storage
