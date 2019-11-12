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

#ifndef STORAGE_COLUMN_STORE_HPP_
#define STORAGE_COLUMN_STORE_HPP_

#include <assert.h>

#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "codegen/plan/plan-parser.hpp"
#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "storage/delta_storage.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "utils/atomic_bit_set.hpp"

#define BIT_PACK_SIZE 8192

namespace storage {

class Column;

class ColumnStore : public Table {
 public:
  ColumnStore(uint8_t table_id, std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns,
              uint64_t initial_num_records = 10000000, bool indexed = true,
              bool partitioned = true, int numa_idx = -1);
  uint64_t insertRecord(void *rec, ushort partition_id, ushort master_ver);
  void *insertRecord(void *rec, uint64_t xid, ushort partition_id,
                     ushort master_ver);
  void *insertRecordBatch(void *rec_batch, uint recs_to_ins,
                          uint capacity_offset, uint64_t xid,
                          ushort partition_id, ushort master_ver);

  void updateRecord(global_conf::IndexVal *hash_ptr, const void *rec,
                    ushort curr_master, ushort curr_delta,
                    const ushort *col_idx = nullptr, short num_cols = -1);
  // void updateRecord(ushort pid, uint64_t &vid, const void *rec,
  //                   ushort curr_master, ushort curr_delta, uint64_t tmin,
  //                   const ushort *col_idx = nullptr, short num_cols =
  //                   -1);

  void deleteRecord(uint64_t vid, ushort master_ver) {
    assert(false && "Not implemented");
  }

  void insertIndexRecord(uint64_t rid, uint64_t xid, ushort partition_id,
                         ushort master_ver);  // hack for loading binary files
  void offsetVID(uint64_t offset);

  uint64_t load_data_from_binary(std::string col_name, std::string file_path);

  void touchRecordByKey(uint64_t vid);
  void getRecordByKey(uint64_t vid, const ushort *col_idx, ushort num_cols,
                      void *loc);

  [[noreturn]] std::vector<const void *> getRecordByKey(uint64_t vid,
                                                        const ushort *col_idx,
                                                        ushort num_cols);

  // global_conf::mv_version_list *getVersions(uint64_t vid);

  void sync_master_snapshots(ushort master_ver_idx);
  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver);
  void ETL(uint numa_node_idx);
  void num_upd_tuples();
  int64_t *snapshot_get_number_tuples(bool olap_snapshot = false,
                                      bool elastic_scan = false);
  const std::vector<Column *> &getColumns() { return columns; }

  std::vector<std::pair<mem_chunk, size_t>> snapshot_get_data(
      size_t scan_idx, std::vector<RecordAttribute *> &wantedFields,
      bool olap_local, bool elastic_scan);

  /*
    No secondary indexes supported as of yet so dont need the following
    void createIndex(int col_idx);
    void createIndex(std::string col_name);
  */

  ~ColumnStore();
  // uint64_t *plugin_ptr[global_conf::num_master_versions][NUM_SOCKETS];
  std::vector<std::vector<std::pair<mem_chunk, size_t>>> elastic_mappings;
  std::set<size_t> elastic_offsets;

 private:
  std::vector<Column *> columns;
  Column *meta_column;
  // Column **secondary_index_vals;
  uint64_t offset;

  size_t nParts;
};

class Column {
 public:
  Column(std::string name, uint64_t initial_num_records, ColumnStore *parent,
         data_type type, size_t unit_size, size_t cummulative_offset,
         bool single_version_only = false, bool partitioned = true,
         int numa_idx = -1);
  ~Column();

  void *getElem(uint64_t vid);
  void touchElem(uint64_t vid);
  void getElem(uint64_t vid, void *copy_location);
  void updateElem(uint64_t vid, void *elem);
  void insertElem(uint64_t vid, void *elem);
  void *insertElem(uint64_t vid);
  void *insertElemBatch(uint64_t vid, uint64_t num_elem);
  void insertElemBatch(uint64_t vid, uint64_t num_elem, void *data);
  void initializeMetaColumn();

  // void updateElem(uint64_t offset, void *elem, ushort master_ver);
  // void deleteElem(uint64_t offset, ushort master_ver);

  void sync_master_snapshots(ushort master_ver_idx);
  void snapshot(const uint64_t *n_recs_part, uint64_t epoch,
                uint8_t snapshot_master_ver);

  void ETL(uint numa_node_idx);

  uint64_t load_from_binary(std::string file_path);

  uint64_t num_upd_tuples(const ushort master_ver = 0,
                          const uint64_t *num_records = nullptr,
                          bool print = false);

  size_t getSize() { return this->total_mem_reserved; }

  // const std::vector<mem_chunk *> get_data(ushort master_version = 0) {
  //   assert(master_version <= global_conf::num_master_versions);
  //   return master_versions[master_version];
  // }

  // snapshot stuff
  std::vector<std::pair<mem_chunk, size_t>> snapshot_get_data(
      bool olap_local = false, bool elastic_scan = false);

  std::vector<std::pair<mem_chunk, size_t>> elastic_partition(
      uint pid, std::set<size_t> &segment_boundaries);

  const std::string name;
  const size_t elem_size;
  const size_t cummulative_offset;
  const data_type type;

  ColumnStore *parent;

 private:
  uint num_partitions;
  volatile bool touched[NUM_SOCKETS];

  size_t total_mem_reserved;
  size_t size_per_part;
  uint64_t initial_num_records;
  uint64_t initial_num_records_per_part;

  std::vector<mem_chunk> master_versions[global_conf::num_master_versions]
                                        [NUM_SOCKETS];

  std::deque<utils::AtomicBitSet<BIT_PACK_SIZE>>
      upd_bit_masks[global_conf::num_master_versions][NUM_SOCKETS];

  // Snapshotting Utils
  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      snapshot_arenas[NUM_SOCKETS];
  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      etl_arenas[NUM_SOCKETS];

  void *etl_mem[NUM_SOCKETS];

  friend class ColumnStore;
};

};  // namespace storage

#endif /* STORAGE_COLUMN_STORE_HPP_ */
