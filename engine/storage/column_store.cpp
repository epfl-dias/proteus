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

#include "storage/column_store.hpp"

#include <sched.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/topology.hpp"
#include "storage/delta_storage.hpp"
#include "storage/table.hpp"
#include "util/timing.hpp"

#if PROTEUS_MEM_MANAGER
#include "codegen/memory/memory-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/topology/topology.hpp"
#endif

#define MEMORY_SLACK 1000
#define CIDR_HACK false
#define PARTITIONED_WORKLOAD false

#define HTAP_UPD_BIT_ON_INSERT false

namespace storage {

std::mutex print_mutex;

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

void ColumnStore::sync_master_snapshots(ushort master_ver_idx) {
  assert(global_conf::num_master_versions > 1);
  for (auto& col : this->columns) {
    if (!(col->type == STRING || col->type == VARCHAR)) {
      col->sync_master_snapshots(master_ver_idx);
    }
  }
}

// master_ver_idx is the inactive master, that is the snapshot.
void Column::sync_master_snapshots(ushort master_ver_idx) {
  assert(global_conf::num_master_versions > 1);

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    if (i == master_ver_idx) continue;
    for (ushort j = 0; j < g_num_partitions; j++) {
      // std::cout << "sync: p_id: " << j << std::endl;
      assert(master_versions[master_ver_idx][j].size() ==
             master_versions[i][j].size());

      // assert(upd_bit_masks[master_ver_idx][j].size() ==
      //        upd_bit_masks[i][j].size());

      assert(master_versions[i][j].size() == 1 &&
             "Expandable memory not supported");
      // assert(snapshot_arenas[i][j].size() == 1 &&
      //        "Expandable memory not supported");

      const auto& dst = master_versions[i][j][0];
      const auto& src = master_versions[master_ver_idx][j][0];
      assert(dst.size == src.size);

      const auto& snap_arena = snapshot_arenas[j][0]->getMetadata();

      if (snap_arena.numOfRecords == 0 || !snap_arena.upd_since_last_snapshot)
        continue;

      const uint8_t* actv_ptr = (uint8_t*)dst.data;
      const uint8_t* src_ptr = (uint8_t*)src.data;

      // std::cout << "Total masks: " << upd_bit_masks[master_ver_idx][j].size()
      //           << std::endl;
      // std::cout << "Total SnapRec: " << snap_arena.numOfRecords << std::endl;

      for (size_t msk = 0; msk < upd_bit_masks[master_ver_idx][j].size();
           msk++) {
        // std::cout << "msk: " << msk << std::endl;
        const auto& src_msk = upd_bit_masks[master_ver_idx][j][msk];
        const auto& actv_msk = upd_bit_masks[i][j][msk];

        if ((msk * BIT_PACK_SIZE) > snap_arena.numOfRecords) break;

        if (!src_msk.any() || actv_msk.all()) continue;

        for (ushort bb = 0; bb < BIT_PACK_SIZE; bb++) {
          size_t data_idx = (msk * BIT_PACK_SIZE) + bb;

          // scan only the records snapshotted, not everything.
          if (data_idx > snap_arena.numOfRecords) break;

          if (src_msk.test(bb) && !actv_msk.test(bb)) {
            // do the sync

            size_t mem_idx = data_idx * elem_size;
            assert(mem_idx < dst.size);
            assert(mem_idx < src.size);

            switch (this->elem_size) {
              case 1: {  // uint8_t
                uint8_t old_val = (*(actv_ptr + mem_idx));
                uint8_t new_val = (*(src_ptr + mem_idx));
                uint8_t* dst = (uint8_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst, old_val, new_val);
                break;
              }
              case 2: {  // uint16_t
                uint16_t old_val = *((uint16_t*)(actv_ptr + mem_idx));
                uint16_t new_val = *((uint16_t*)(src_ptr + mem_idx));
                uint16_t* dst = (uint16_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst, old_val, new_val);
                break;
              }
              case 4: {  // uint32_t
                uint32_t old_val = *((uint32_t*)(actv_ptr + mem_idx));
                uint32_t new_val = *((uint32_t*)(src_ptr + mem_idx));
                uint32_t* dst = (uint32_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst, old_val, new_val);
                break;
              }
              case 8: {  // uint64_t
                uint64_t old_val = *((uint64_t*)(actv_ptr + mem_idx));
                uint64_t new_val = *((uint64_t*)(src_ptr + mem_idx));
                uint64_t* dst = (uint64_t*)(actv_ptr + mem_idx);
                __sync_bool_compare_and_swap(dst, old_val, new_val);
                break;
              }
              default: {
                // std::unique_lock<std::mutex> lk(print_mutex);
                std::cout << "col fucked: " << this->name << std::endl;
                assert(false && "unsupported for now");
              }
            }
          }
        }
      }
    }
  }
}

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
    uint64_t initial_num_records, bool indexed, bool partitioned, int numa_idx)
    : Table(name, table_id, COLUMN_STORE, columns) {
  this->total_mem_reserved = 0;
  this->indexed = indexed;
  this->deltaStore = storage::Schema::getInstance().deltaStore;

  for (int i = 0; i < g_num_partitions; i++) this->vid[i] = 0;

  if (indexed) {
    meta_column = new Column(name + "_meta", initial_num_records, this, META,
                             sizeof(global_conf::IndexVal), 0, true,
                             partitioned, numa_idx);
    meta_column->initializeMetaColumn();

    this->p_index =
        new global_conf::PrimaryIndex<uint64_t>(name, initial_num_records);
    // if (partitioned)
    //   this->p_index =
    //       new global_conf::PrimaryIndex<uint64_t>(name, NUM_SOCKETS);
    // else
    //   this->p_index = new global_conf::PrimaryIndex<uint64_t>(name);
    // this->p_index = new global_conf::PrimaryIndex<uint64_t>();
    // this->p_index->max_num_worker_threads(MAX_WORKERS);
    // if (initial_num_records < 50000000)
    //   this->p_index->reserve(initial_num_records);
    // else
    //   this->p_index->reserve(50000000);

    // this->p_index = new global_conf::PrimaryIndex<uint64_t>(name);
  }

  // create columns
  size_t col_offset = 0;
  for (const auto& t : columns) {
    this->columns.emplace_back(
        new Column(std::get<0>(t), initial_num_records, this, std::get<1>(t),
                   std::get<2>(t), col_offset, false, partitioned, numa_idx));
    col_offset += std::get<2>(t);
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
  assert(rec_size == col_offset);
  this->offset = 0;

#if CIDR_HACK
  this->initial_num_recs = initial_num_records - (NUM_SOCKETS * MEMORY_SLACK);
#else
  this->initial_num_recs = initial_num_records;

#endif

  {
    std::unique_lock<std::mutex> lk(print_mutex);
    std::cout << "Table: " << name << std::endl;
    std::cout << "\trecord size: " << rec_size << " bytes" << std::endl;
    std::cout << "\tnum_records: " << initial_num_records << std::endl;

    if (indexed) total_mem_reserved += meta_column->total_mem_reserved;

    std::cout << "\tMem reserved: "
              << (double)total_mem_reserved / (1024 * 1024 * 1024) << "GB"
              << std::endl;
  }
}

void ColumnStore::offsetVID(uint64_t offset) {
  for (int i = 0; i < NUM_SOCKETS; i++) vid[i].store(offset);
  this->offset = offset;
}

void ColumnStore::insertIndexRecord(uint64_t rid, uint64_t xid,
                                    ushort partition_id, ushort master_ver) {
  assert(this->indexed);
  uint64_t curr_vid = vid[partition_id].fetch_add(1);

  global_conf::IndexVal* hash_ptr =
      (global_conf::IndexVal*)this->meta_column->getElem(
          CC_gen_vid(curr_vid, partition_id, 0, 0));
  hash_ptr->t_min = xid;
  hash_ptr->VID = CC_gen_vid(curr_vid, partition_id, master_ver, 0);
  void* pano = (void*)hash_ptr;
  this->p_index->insert(rid, pano);
}

/* Following function assumes that the  void* rec has columns in the same order
 * as the actual columns
 */

void* ColumnStore::insertRecordBatch(void* rec_batch, uint recs_to_ins,
                                     uint capacity_offset, uint64_t xid,
                                     ushort partition_id, ushort master_ver) {
  uint64_t idx_st = vid[partition_id].fetch_add(recs_to_ins);
  // get batch from meta column
  uint64_t st_vid = CC_gen_vid(idx_st, partition_id, master_ver, 0);
  uint64_t st_vid_meta = CC_gen_vid(idx_st, partition_id, 0, 0);

  global_conf::IndexVal* hash_ptr = nullptr;
  if (this->indexed) {
    // meta stuff
    hash_ptr = (global_conf::IndexVal*)this->meta_column->insertElemBatch(
        st_vid_meta, recs_to_ins);
    assert(hash_ptr != nullptr);

    for (uint i = 0; i < recs_to_ins; i++) {
      hash_ptr->t_min = xid;
      hash_ptr->VID = CC_gen_vid(idx_st + i, partition_id, master_ver, 0);
      hash_ptr += 1;
    }
  }

  // for loop to copy all columns.
  for (auto& col : columns) {
    col->insertElemBatch(
        st_vid, recs_to_ins,
        ((char*)rec_batch) + (col->cummulative_offset * capacity_offset));
  }

  return (void*)hash_ptr;

  // return starting address of batch meta.
}

void* ColumnStore::insertRecord(void* rec, uint64_t xid, ushort partition_id,
                                ushort master_ver) {
  uint64_t idx = vid[partition_id].fetch_add(1);
  uint64_t curr_vid = CC_gen_vid(idx, partition_id, master_ver, 0);

  // if (this->name.compare("tpcc_district") == 0) {
  //   std::cout << "----------------" << this->name << std::endl;
  //   std::cout << "idx: " << idx << std::endl;
  // }

  global_conf::IndexVal* hash_ptr = nullptr;

#if CIDR_HACK
  if (curr_vid >= (initial_num_recs / NUM_SOCKETS)) {
    scheduler::WorkerPool::getInstance().shutdown_manual();
  }

#endif

  if (indexed) {
    hash_ptr = (global_conf::IndexVal*)this->meta_column->getElem(
        CC_gen_vid(idx, partition_id, 0, 0));
    assert(hash_ptr != nullptr);
    hash_ptr->t_min = xid;
    hash_ptr->VID = curr_vid;
  }

  char* rec_ptr = (char*)rec;

  for (auto& col : columns) {
    col->insertElem(curr_vid, rec_ptr + col->cummulative_offset);
  }

  return (void*)hash_ptr;
}

uint64_t ColumnStore::insertRecord(void* rec, ushort partition_id,
                                   ushort master_ver) {
  uint64_t curr_vid =
      CC_gen_vid(vid[partition_id].fetch_add(1), partition_id, master_ver, 0);

  char* rec_ptr = (char*)rec;
  for (auto& col : columns) {
    col->insertElem(curr_vid, rec_ptr + col->cummulative_offset);
  }
  return curr_vid;
}

void ColumnStore::touchRecordByKey(uint64_t vid) {
  for (auto& col : columns) {
    col->touchElem(vid);
  }
}

void ColumnStore::getRecordByKey(uint64_t vid, const ushort* col_idx,
                                 ushort num_cols, void* loc) {
  char* write_loc = (char*)loc;
  if (__unlikely(col_idx == nullptr)) {
    for (auto& col : columns) {
      col->getElem(vid, write_loc);
      write_loc += col->elem_size;
    }
  } else {
    for (ushort i = 0; i < num_cols; i++) {
      Column* col = columns.at(col_idx[i]);
      col->getElem(vid, write_loc);
      write_loc += col->elem_size;
    }
  }
}

std::vector<const void*> ColumnStore::getRecordByKey(uint64_t vid,
                                                     const ushort* col_idx,
                                                     ushort num_cols) {
  // uint64_t vid, ushort master_ver, const std::vector<ushort>* col_idx)
  // if (col_idx == nullptr) {
  //   std::vector<const void*> record(columns.size());
  //   for (auto& col : columns) {
  //     record.push_back((const void*)(col->getElem(vid)));
  //   }
  //   return record;
  // } else {
  //   std::vector<const void*> record(col_idx->size());
  //   for (auto& c_idx : *col_idx) {
  //     Column* col = columns.at(c_idx);
  //     record.push_back((const void*)(col->getElem(vid)));
  //   }
  //   return record;
  // }

  // std::vector<const void*> tmp;
  assert(false && "Not implemented");
  // return tmp;
}

/*
  Possible Bug: Update records create a delta version not in the local parition
  to the worker but local to record-master partition. this create version
  creation over QPI.
*/
void ColumnStore::updateRecord(global_conf::IndexVal* hash_ptr, const void* rec,
                               ushort curr_master, ushort curr_delta,
                               const ushort* col_idx, short num_cols) {
  ushort pid = CC_extract_pid(hash_ptr->VID);
  ushort m_ver = CC_extract_m_ver(hash_ptr->VID);

#if PARTITIONED_WORKLOAD
  if (ppid != pid) {
    std::unique_lock<std::mutex> ll(print_mutex);
    std::cout << this->name << std::endl;
    std::cout << ppid << std::endl;
    std::cout << pid << std::endl;
  }
  assert(ppid == pid);
#endif

  char* ver = (char*)this->deltaStore[curr_delta]->insert_version(
      hash_ptr, this->rec_size, pid);
  assert(ver != nullptr);

  for (auto& col : columns) {
    memcpy(ver + col->cummulative_offset, col->getElem(hash_ptr->VID),
           col->elem_size);
  }

  hash_ptr->VID = CC_upd_vid(hash_ptr->VID, curr_master, curr_delta);
  char* cursor = (char*)rec;

  if (__unlikely(num_cols <= 0)) {
    for (auto& col : columns) {
      col->updateElem(
          hash_ptr->VID,
          (rec == nullptr ? nullptr : cursor + col->cummulative_offset));
    }
  } else {
    for (int i = 0; i < num_cols; i++) {
      Column* col = columns.at(col_idx[i]);
      col->updateElem(hash_ptr->VID,
                      (rec == nullptr ? nullptr : (void*)cursor));
      cursor += col->elem_size;
    }
  }
}

Column::Column(std::string name, uint64_t initial_num_records,
               ColumnStore* parent, data_type type, size_t unit_size,
               size_t cummulative_offset, bool single_version_only,
               bool partitioned, int numa_idx)
    : name(name),
      parent(parent),
      elem_size(unit_size),
      cummulative_offset(cummulative_offset),
      type(type) {
  this->parent = parent;

  if (partitioned)
    this->num_partitions = g_num_partitions;
  else
    this->num_partitions = 1;

  assert(g_num_partitions <= NUM_SOCKETS);

  this->initial_num_records = initial_num_records;
  this->initial_num_records_per_part =
      (initial_num_records / this->num_partitions) +
      initial_num_records % this->num_partitions;

  size_t size_per_partition = initial_num_records_per_part * unit_size;
  size_t size = size_per_partition * this->num_partitions;
  this->total_mem_reserved = size * global_conf::num_master_versions;
  this->size_per_part = size_per_partition;

  // snapshot arenas
  for (uint j = 0; j < g_num_partitions; j++) {
    snapshot_arenas[j].emplace_back(
        global_conf::SnapshotManager::create(size_per_partition));
  }

#if HTAP_ETL
  // essentially this is memory allocation for proteus where it will do the ETL
  // and update its snapshots.

  // FIXME: hack for expr to make memory in proteus sockets
  for (ushort j = 0; j < this->num_partitions; j++) {
    this->etl_mem[j] = MemoryManager::alloc_shm(name + "__" + std::to_string(j),
                                                size_per_partition, j + 2);
    assert(this->etl_mem[j] != nullptr || this->etl_mem[j] != NULL);
  }

#endif

#if PROTEUS_MEM_MANAGER
  const auto& cpunumanodes = ::topology::getInstance().getCpuNumaNodes();
#endif

  std::vector<std::thread> loaders;

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (ushort j = 0; j < this->num_partitions; j++) {
#if HTAP_RM_SERVER
      void* mem = MemoryManager::alloc_shm_htap(
          std::to_string(i) + "__" + std::to_string(j) + "__" + name,
          size_per_partition, unit_size, j);

#elif SHARED_MEMORY
      void* mem = MemoryManager::alloc_shm(
          std::to_string(i) + "__" + std::to_string(j) + "__" + name,
          size_per_partition, j);

#elif PROTEUS_MEM_MANAGER
      set_exec_location_on_scope d{cpunumanodes[j]};
      void* mem = ::MemoryManager::mallocPinned(size_per_partition);
#else
      void* mem = MemoryManager::alloc(size_per_partition, j);

#endif
      assert(mem != nullptr || mem != NULL);
      loaders.emplace_back([mem, size_per_partition, j]() {
        const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
        scheduler::AffinityManager::getInstance().set(&vec[j]);

        uint64_t* pt = (uint64_t*)mem;
        uint64_t warmup_max = size_per_partition / sizeof(uint64_t);
#pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < warmup_max; j++) pt[j] = 0;
      });

      master_versions[i][j].emplace_back(mem, size_per_partition, j);

      if (!single_version_only) {
        uint num_bit_packs = (initial_num_records_per_part / BIT_PACK_SIZE) +
                             (initial_num_records_per_part % BIT_PACK_SIZE);

        loaders.emplace_back([this, i, j, num_bit_packs]() {
          // set affinity to fault in the correct numa-memset
          const auto& vec =
              scheduler::Topology::getInstance().getCpuNumaNodes();
          scheduler::AffinityManager::getInstance().set(&vec[j]);

          for (uint64_t bb = 0; bb < num_bit_packs; bb++) {
            upd_bit_masks[i][j].emplace_back();
          }
          for (auto& bb : this->upd_bit_masks[i][j]) {
            bb.reset();
          }
        });
      }
    }
    if (single_version_only) break;
  }

  for (auto& th : loaders) {
    th.join();
  }

  for (uint i = 0; i < this->num_partitions; i++) this->touched[i] = false;
}

void Column::touchElem(uint64_t vid) {
  ushort pid = CC_extract_pid(vid);
  ushort m_ver = CC_extract_m_ver(vid);
  uint64_t offset = CC_extract_offset(vid);
  uint64_t data_idx = offset * elem_size;

  assert(master_versions[m_ver][pid].size() != 0);

  for (const auto& chunk : master_versions[m_ver][pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + elem_size))) {
      char* loc = ((char*)chunk.data) + data_idx;

      // set_upd_bit(loc);
      upd_bit_masks[m_ver][pid][offset / BIT_PACK_SIZE].set(offset %
                                                            BIT_PACK_SIZE);
      this->touched[pid] = true;
      char tmp = 'a';
      for (int i = 0; i < elem_size; i++) {
        tmp += *loc;
      }
    }
  }
}

void Column::getElem(uint64_t vid, void* copy_location) {
  ushort pid = CC_extract_pid(vid);
  ushort m_ver = CC_extract_m_ver(vid);
  uint64_t data_idx = CC_extract_offset(vid) * elem_size;

  assert(master_versions[m_ver][pid].size() != 0);

  for (const auto& chunk : master_versions[m_ver][pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + elem_size))) {
      std::memcpy(copy_location, ((char*)chunk.data) + data_idx,
                  this->elem_size);
      return;
    }
  }
  assert(false);  // as control should never reach here.
}

void Column::initializeMetaColumn() {
  std::vector<std::thread> loaders;
  for (ushort j = 0; j < this->num_partitions; j++) {
    for (const auto& chunk : master_versions[0][j]) {
      char* ptr = (char*)chunk.data;
      assert(chunk.size % this->elem_size == 0);
      loaders.emplace_back([this, chunk, j, ptr]() {
        for (uint64_t i = 0; i < (chunk.size / this->elem_size); i++) {
          void* c = new (ptr + (i * this->elem_size))
              global_conf::IndexVal(0, CC_gen_vid(i, j, 0, 0));
        }
      });

      // for (uint64_t i = 0; i < (chunk.size / this->elem_size); i++) {
      //   void* c = new (ptr + (i * this->elem_size))
      //       global_conf::IndexVal(0, CC_gen_vid(i, j, 0, 0));
      //   assert(c != nullptr && c != NULL);
      // }
    }
  }
  for (auto& th : loaders) {
    th.join();
  }
}

void* Column::getElem(uint64_t vid) {
  ushort pid = CC_extract_pid(vid);
  ushort m_ver = CC_extract_m_ver(vid);
  uint64_t data_idx = CC_extract_offset(vid) * elem_size;

  assert(master_versions[m_ver][pid].size() != 0);

  for (const auto& chunk : master_versions[m_ver][pid]) {
    if (__likely(chunk.size >= ((size_t)data_idx + elem_size))) {
      return ((char*)chunk.data) + data_idx;
    }
  }

  assert(false && "Out of Memory");
  return nullptr;
}

void Column::insertElem(uint64_t vid, void* elem) {
  ushort pid = CC_extract_pid(vid);
  ushort mver = CC_extract_m_ver(vid);
  uint64_t offset = CC_extract_offset(vid);
  uint64_t data_idx = offset * elem_size;

  assert(pid < g_num_partitions);
  // assert(idx < initial_num_records_per_part);

  if (data_idx >= size_per_part) {
    std::cout << "-----------" << std::endl;
    std::cout << this->name << std::endl;
    std::cout << "pid: " << (uint)pid << std::endl;
    std::cout << "mver: " << (uint)mver << std::endl;
    std::cout << "offset: " << offset << std::endl;
  }

  assert(data_idx < size_per_part);

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    bool ins = false;
    for (const auto& chunk : master_versions[i][pid]) {
      assert(pid == chunk.numa_id);

      if (__likely(chunk.size >= (data_idx + elem_size))) {
        void* dst = (void*)(((char*)chunk.data) + data_idx);
        if (__unlikely(elem == nullptr)) {
          uint64_t* tptr = (uint64_t*)dst;
          (*tptr)++;
        } else {
          char* src_t = (char*)chunk.data;
          char* dst_t = (char*)dst;

          assert(src_t <= dst_t);
          assert((src_t + chunk.size) >= (dst_t + this->elem_size));

          std::memcpy(dst, elem, this->elem_size);
        }

        ins = true;
        break;
      }
    }
#if HTAP_UPD_BIT_ON_INSERT
    if (__likely(i == mver)) {
      // if (this->name[0] == 'd') {
      //   std::cout << "bitpack: " << (offset / BIT_PACK_SIZE)
      //             << "| bit-offset: " << (offset % BIT_PACK_SIZE) <<
      //             std::endl;
      // }
      upd_bit_masks[mver][pid][offset / BIT_PACK_SIZE].set(offset %
                                                           BIT_PACK_SIZE);
      if (!this->touched[pid]) this->touched[pid] = true;
    }
#endif
    if (__unlikely(ins == false)) {
      std::cout << "(1) ALLOCATE MORE MEMORY:\t" << this->name
                << ",vid: " << vid << ", idx:" << (data_idx / elem_size)
                << ", pid: " << pid << std::endl;

      assert(false && "Out Of Memory Error");
    }
  }
}

void Column::updateElem(uint64_t vid, void* elem) {
  ushort pid = CC_extract_pid(vid);

  uint64_t offset = CC_extract_offset(vid);
  uint64_t data_idx = offset * elem_size;
  uint8_t mver = CC_extract_m_ver(vid);

  assert(pid < g_num_partitions);
  // assert(idx < initial_num_records_per_part);
  assert(data_idx < size_per_part);

  for (const auto& chunk : master_versions[mver][pid]) {
    assert(pid == chunk.numa_id);

    if (__likely(chunk.size >= (data_idx + elem_size))) {
      void* dst = (void*)(((char*)chunk.data) + data_idx);
      if (__unlikely(elem == nullptr)) {
        uint64_t* tptr = (uint64_t*)dst;
        (*tptr)++;
      } else {
        char* src_t = (char*)chunk.data;
        char* dst_t = (char*)dst;

        assert(src_t <= dst_t);
        assert((src_t + chunk.size) >= (dst_t + this->elem_size));

        std::memcpy(dst, elem, this->elem_size);
      }

      upd_bit_masks[mver][pid][offset / BIT_PACK_SIZE].set(offset %
                                                           BIT_PACK_SIZE);
      if (!this->touched[pid]) this->touched[pid] = true;
      return;
    }
  }

  assert(false && "Out Of Memory Error");
}

void* Column::insertElem(uint64_t vid) {
  assert(this->type == META);
  ushort pid = CC_extract_pid(vid);
  uint64_t data_idx = CC_extract_offset(vid) * elem_size;
  ushort mver = CC_extract_m_ver(vid);

  assert(pid < g_num_partitions);
  assert((data_idx / elem_size) < initial_num_records_per_part);
  assert(data_idx < size_per_part);

  bool ins = false;
  for (const auto& chunk : master_versions[mver][pid]) {
    if (__likely(chunk.size >= (data_idx + elem_size))) {
      return (void*)(((char*)chunk.data) + data_idx);
    }
  }

  assert(false && "Out Of Memory Error");
  return nullptr;
}

void* Column::insertElemBatch(uint64_t vid, uint64_t num_elem) {
  assert(this->type == META);

  ushort pid = CC_extract_pid(vid);
  uint64_t data_idx_st = CC_extract_offset(vid) * elem_size;
  uint64_t data_idx_en = data_idx_st + (num_elem * elem_size);
  ushort mver = CC_extract_m_ver(vid);

  assert(pid < g_num_partitions);
  assert((data_idx_en / elem_size) < initial_num_records_per_part);
  assert(data_idx_en < size_per_part);

  bool ins = false;
  for (const auto& chunk : master_versions[mver][pid]) {
    if (__likely(chunk.size >= (data_idx_en + elem_size))) {
      return (void*)(((char*)chunk.data) + data_idx_st);
    }
  }

  assert(false && "Out Of Memory Error");
  return nullptr;
}

void Column::insertElemBatch(uint64_t vid, uint64_t num_elem, void* data) {
  ushort pid = CC_extract_pid(vid);
  uint64_t offset = CC_extract_offset(vid);

  uint64_t data_idx_st = offset * elem_size;
  size_t copy_size = num_elem * this->elem_size;
  uint64_t data_idx_en = data_idx_st + copy_size;
  ushort mver = CC_extract_m_ver(vid);

  // uint64_t data_idx = CC_extract_offset(vid) * elem_size;

  assert(pid < g_num_partitions);
  assert((data_idx_en / elem_size) < initial_num_records_per_part);
  assert(data_idx_en < size_per_part);

  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    bool ins = false;
    for (const auto& chunk : master_versions[i][pid]) {
      assert(pid == chunk.numa_id);

      if (__likely(chunk.size >= (data_idx_en + elem_size))) {
        void* dst = (void*)(((char*)chunk.data) + data_idx_st);
        std::memcpy(dst, data, copy_size);
        ins = true;
        break;
      }
    }

#if HTAP_UPD_BIT_ON_INSERT
    if (__likely(i == mver)) {
      for (size_t st = 0; st < num_elem; st++) {
        upd_bit_masks[mver][pid][(offset + st) / BIT_PACK_SIZE].set(
            (offset + st) % BIT_PACK_SIZE);
      }

      if (!this->touched[pid]) this->touched[pid] = true;
    }
#endif

    if (__unlikely(ins == false)) {
      assert(false && "Out Of Memory Error");
    }
  }
}

Column::~Column() {
  for (ushort i = 0; i < global_conf::num_master_versions; i++) {
    for (ushort j = 0; j < g_num_partitions; j++) {
      for (auto& chunk : master_versions[i][j]) {
#if PROTEUS_MEM_MANAGER
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
  for (uint i = 0; i < global_conf::num_master_versions; i++) {
    for (auto& col : this->columns) {
      col->num_upd_tuples(i, nullptr, true);
    }
  }
}

uint64_t Column::num_upd_tuples(const ushort master_ver,
                                const uint64_t* num_records, bool print) {
  uint64_t counter = 0;
  bool printed_first = false;
  for (uint j = 0; j < g_num_partitions; j++) {
    if (touched[j] == false) continue;

    if (__likely(num_records != nullptr)) {
      uint64_t recs_scanned = 0;
      for (auto& chunk : upd_bit_masks[master_ver][j]) {
        counter += chunk.count(std::memory_order::memory_order_acquire);
        recs_scanned += BIT_PACK_SIZE;
        if (recs_scanned >= num_records[j]) {
          break;
        }
      }

    } else {
      for (auto& chunk : upd_bit_masks[master_ver][j]) {
        counter += chunk.count();
      }
    }
  }

  if (__unlikely(print)) {
    if (counter > 0) {
      std::cout << "UPD[" << master_ver << "]: COL:" << this->name
                << " | #num_upd: " << counter << std::endl;
    }
  }
  return counter;
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

      for (ushort j = 0; j < g_num_partitions; j++) {
        size_t part_size = length / g_num_partitions;
        if (j == (g_num_partitions - 1)) {
          // remaining shit.
          part_size += part_size % g_num_partitions;
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

void ColumnStore::snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
  uint64_t partitions_n_recs[NUM_SOCKETS];

  //{
  // std::unique_lock<std::mutex> lk(print_mutex);
  std::cout << "SnapTable: " << this->name << std::endl;
  //}
  for (uint i = 0; i < g_num_partitions; i++) {
    partitions_n_recs[i] = this->vid[i].load();
    std::cout << "NumRecords[" << i << "]" << this->vid[i].load() << std::endl;
  }

  for (auto& col : this->columns) {
    col->snapshot(partitions_n_recs, epoch, snapshot_master_ver);
  }
}

void Column::snapshot(const uint64_t* n_recs_part, uint64_t epoch,
                      uint8_t snapshot_master_ver) {
  if (this->name.compare("d_next_o_id") == 0 ||
      this->name.compare("s_quantity") == 0 ||
      this->name.compare("s_order_cnt") == 0 ||
      this->name.compare("s_ytd") == 0)
    num_upd_tuples(snapshot_master_ver, n_recs_part, true);
  for (uint i = 0; i < g_num_partitions; i++) {
    assert(snapshot_arenas[i].size() == 1);

    snapshot_arenas[i][0]->create_snapshot(
        {n_recs_part[i], snapshot_arenas[i][0]->getMetadata().numOfRecords,
         epoch, snapshot_master_ver, static_cast<uint8_t>(i),
         this->touched[i]});
    this->touched[i] = false;
  }

  // for (uint j = 0; j < g_num_partitions; j++) {
  //   const auto& snap_arena = snapshot_arenas[j][0]->getMetadata();
  //   if (snap_arena.upd_since_last_snapshot == false) continue;

  //   uint64_t recs_scanned = 0;
  //   for (auto& chunk : upd_bit_masks[snapshot_master_ver][j]) {
  //     chunk.reset();
  //     recs_scanned += BIT_PACK_SIZE;
  //     if (recs_scanned >= snap_arena.numOfRecords) {
  //       break;
  //     }
  //   }
  // }
}

std::vector<std::pair<mem_chunk, uint64_t>> Column::snapshot_get_data() {
  std::vector<std::pair<mem_chunk, uint64_t>> ret;

  for (uint i = 0; i < num_partitions; i++) {
    assert(master_versions[0][i].size() == 1);
    auto& snap_arena = snapshot_arenas[i][0]->getMetadata();

    for (const auto& chunk : master_versions[snap_arena.master_ver][i]) {
#if HTAP_ETL

      ret.emplace_back(std::make_pair(
          mem_chunk(this->etl_mem[i], snap_arena.numOfRecords * this->elem_size,
                    -1),
          snap_arena.numOfRecords));
#else
      ret.emplace_back(std::make_pair(chunk, snap_arena.numOfRecords));
#endif
    }
  }

  return ret;
}

void ColumnStore::ETL(uint numa_node_idx) {
  std::vector<std::thread> workers;

  for (const auto& col : this->columns) {
    workers.emplace_back([col, numa_node_idx]() { col->ETL(numa_node_idx); });
  }

  for (auto& th : workers) {
    th.join();
  }
}

void Column::ETL(uint numa_node_index) {
  // TODO: ETL with respect to the bit-mask.
  const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
  scheduler::AffinityManager::getInstance().set(&vec[numa_node_index]);

  for (uint i = 0; i < this->num_partitions; i++) {
    // zero assume no runtime column expansion
    const auto& snap_arena = snapshot_arenas[i][0]->getMetadata();
    const auto& chunk = master_versions[snap_arena.master_ver][i][0];

    if (snap_arena.upd_since_last_snapshot) {
      for (size_t msk = 0; msk < upd_bit_masks[snap_arena.master_ver][i].size();
           msk++) {
        if (msk * BIT_PACK_SIZE >= snap_arena.prev_numOfRecords) break;

        if (upd_bit_masks[snap_arena.master_ver][i][msk].any(
                std::memory_order::memory_order_acquire)) {
          size_t to_cpy = BIT_PACK_SIZE * this->elem_size;
          size_t st = msk * to_cpy;

          if (__likely(st + to_cpy <= chunk.size)) {
            memcpy((char*)(etl_mem[i]) + st, (char*)chunk.data + st, to_cpy);
          } else {
            memcpy((char*)(etl_mem[i]) + st, (char*)chunk.data + st,
                   chunk.size - st);
          }

          upd_bit_masks[snap_arena.master_ver][i][msk].reset(
              std::memory_order::memory_order_release);
        }
      }
    }

    if (__likely(snap_arena.numOfRecords > snap_arena.prev_numOfRecords)) {
      size_t st = snap_arena.prev_numOfRecords * this->elem_size;
      size_t to_cpy = (snap_arena.numOfRecords - snap_arena.prev_numOfRecords) *
                      this->elem_size;
      memcpy(((char*)(etl_mem[i])) + st, ((char*)chunk.data) + st, to_cpy);
    }

    // for (const auto& chunk : master_versions[snap_arena.master_ver][i]) {
    //   if (snap_arena.upd_since_last_snapshot) {
    //     memcpy(etl_mem[i], chunk.data,
    //            snap_arena.numOfRecords * this->elem_size);
    //   }
    // }
  }
}
};  // namespace storage
