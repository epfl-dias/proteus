/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#ifndef STORAGE_COLUMN_STORE_HPP_
#define STORAGE_COLUMN_STORE_HPP_

#include <cassert>
#include <deque>
#include <iostream>
#include <map>
#include <platform/memory/allocator.hpp>
#include <platform/memory/memory-manager.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "oltp/common/atomic_bit_set.hpp"
#include "oltp/common/constants.hpp"
#include "oltp/common/memory-chunk.hpp"
#include "oltp/storage/table.hpp"

#define BIT_PACK_SIZE 8192

class RecordAttribute;

namespace storage {

class Column;

using ColumnVector =
    std::vector<storage::Column,
                proteus::memory::ExplicitSocketPinnedMemoryAllocator<Column>>;

class alignas(4096) ColumnStore : public Table {
 public:
  ColumnStore(uint8_t table_id, const std::string &name, ColumnDef columns,
              uint64_t initial_num_records = 10000000, bool indexed = true,
              bool partitioned = true, int numa_idx = -1);
  ~ColumnStore() override;

  uint64_t insertRecord(void *rec, ushort partition_id,
                        ushort master_ver) override;
  void *insertRecord(void *rec, uint64_t xid, ushort partition_id,
                     ushort master_ver) override;
  void *insertRecordBatch(void *rec_batch, uint recs_to_ins,
                          uint capacity_offset, uint64_t xid,
                          ushort partition_id, ushort master_ver) override;

  void updateRecord(uint64_t xid, global_conf::IndexVal *hash_ptr,
                    const void *rec, ushort curr_master, ushort curr_delta,
                    const ushort *col_idx, short num_cols) override;

  void deleteRecord(uint64_t vid, ushort master_ver) override {
    assert(false && "Not implemented");
  }

  /*  Utils for loading data from binary or offseting datasets
   * */
  void insertIndexRecord(
      uint64_t rid, uint64_t xid, ushort partition_id,
      ushort master_ver) override;  // hack for loading binary files
  void offsetVID(uint64_t offset);
  uint64_t load_data_from_binary(std::string col_name, std::string file_path);

  void touchRecordByKey(uint64_t vid) override;

  void getRecordByKey(global_conf::IndexVal *idx_ptr, uint64_t txn_id,
                      const ushort *col_idx, ushort num_cols,
                      void *loc) override;

  // HTAP / Snapshotting Methods
  void sync_master_snapshots(ushort master_ver_idx);
  void snapshot(uint64_t epoch, uint8_t snapshot_master_ver) override;
  void ETL(uint numa_node_idx) override;
  void num_upd_tuples();
  int64_t *snapshot_get_number_tuples(bool olap_snapshot = false,
                                      bool elastic_scan = false);

  std::vector<std::pair<oltp::common::mem_chunk, size_t>> snapshot_get_data(
      size_t scan_idx, std::vector<RecordAttribute *> &wantedFields,
      bool olap_local, bool elastic_scan);

 private:
  ColumnVector columns;
  Column *meta_column;
  uint64_t offset;
  ushort num_data_partitions;
  size_t nParts{};
  std::vector<std::pair<uint16_t, uint16_t>> column_size_offset_pairs;
  std::vector<uint16_t> column_size_offsets;
  std::vector<uint16_t> column_size;

  std::vector<std::vector<std::pair<oltp::common::mem_chunk, size_t>>>
      elastic_mappings;
  std::set<size_t> elastic_offsets;

 public:
  const decltype(columns) &getColumns() { return columns; }
};

class alignas(4096) Column {
 public:
  ~Column();
  Column(std::string name, uint64_t initial_num_records, data_type type,
         size_t unit_size, size_t cumulative_offset,
         bool single_version_only = false, bool partitioned = true,
         int numa_idx = -1);

  Column(const Column &) = delete;
  Column(Column &&) = default;

 private:
  void *getElem(uint64_t vid);
  void touchElem(uint64_t vid);
  void getElem(uint64_t vid, void *copy_location);
  void updateElem(uint64_t vid, void *elem);
  void insertElem(uint64_t vid, void *elem);
  void *insertElem(uint64_t vid);
  void *insertElemBatch(uint64_t vid, uint64_t num_elem);
  void insertElemBatch(uint64_t vid, uint64_t num_elem, void *data);
  void initializeMetaColumn();

 public:
  [[nodiscard]] size_t getSize() const { return this->total_mem_reserved; }

  void sync_master_snapshots(ushort master_ver_idx);
  void snapshot(const uint64_t *n_recs_part, uint64_t epoch,
                uint8_t snapshot_master_ver);
  void ETL(uint numa_node_idx);
  uint64_t num_upd_tuples(ushort master_ver = 0,
                          const uint64_t *num_records = nullptr,
                          bool print = false);
  [[nodiscard]] std::vector<std::pair<oltp::common::mem_chunk, size_t>>
  snapshot_get_data(bool olap_local = false, bool elastic_scan = false) const;

  std::vector<std::pair<oltp::common::mem_chunk, size_t>> elastic_partition(
      uint pid, std::set<size_t> &segment_boundaries);

 private:
  uint num_partitions;
  volatile bool touched[global_conf::MAX_PARTITIONS];

  const std::string name;
  const size_t elem_size;
  const size_t cumulative_offset;
  const data_type type;

  size_t total_mem_reserved;
  size_t size_per_part;
  uint64_t initial_num_records;
  uint64_t initial_num_records_per_part;

  std::vector<oltp::common::mem_chunk>
      master_versions[global_conf::num_master_versions]
                     [global_conf::MAX_PARTITIONS];

  std::deque<utils::AtomicBitSet<BIT_PACK_SIZE>>
      upd_bit_masks[global_conf::num_master_versions]
                   [global_conf::MAX_PARTITIONS];

  // Snapshotting Utils
  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      snapshot_arenas[global_conf::MAX_PARTITIONS];
  std::vector<decltype(global_conf::SnapshotManager::create(0))>
      etl_arenas[global_conf::MAX_PARTITIONS];

  void *etl_mem[global_conf::MAX_PARTITIONS];

  friend class ColumnStore;
};

};  // namespace storage

#endif /* STORAGE_COLUMN_STORE_HPP_ */
