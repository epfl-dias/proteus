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

#ifndef STORAGE_ROW_STORE_HPP_
#define STORAGE_ROW_STORE_HPP_

#include <assert.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "storage/delta_storage.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

namespace storage {

class RowStore;

/*  DATA LAYOUT -- ROW STORE
 */

class RowStore : public Table {
 public:
  RowStore(uint8_t table_id, std::string name, ColumnDef columns,
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

  void getRecordByKey(uint64_t vid, const ushort *col_idx, ushort num_cols,
                      void *loc);

  void touchRecordByKey(uint64_t vid);

  global_conf::mv_version_list *getVersions(uint64_t vid);

  void deleteRecord(uint64_t vid, ushort master_ver) {
    assert(false && "Not implemented");
  }

  [[noreturn]] std::vector<const void *> getRecordByKey(uint64_t vid,
                                                        const ushort *col_idx,
                                                        ushort num_cols) {
    assert(false && "Not implemented");
  }

  [[noreturn]] void getRecordByKey(uint64_t vid, ushort master_ver,
                                   const std::vector<ushort> *col_idx,
                                   void *loc) {
    assert(false && "Not implemented");
  }

  [[noreturn]] void insertIndexRecord(uint64_t rid, uint64_t xid,
                                      ushort partition_id, ushort master_ver) {
    assert(false && "Not implemented");
  }

  [[noreturn]] void snapshot(uint64_t epoch, uint8_t snapshot_master_ver) {
    assert(false && "Not implemented");
  }

 private:
  size_t rec_size;
  std::vector<std::string> columns;
  // 1-size, 2-cumm size until that col
  std::vector<std::pair<size_t, size_t>> column_width;
  std::vector<data_type> column_data_types;
  std::vector<storage::memory::mem_chunk> data[global_conf::num_master_versions]
                                              [NUM_SOCKETS];

  std::vector<storage::memory::mem_chunk> metadata[NUM_SOCKETS];

  bool indexed;
  uint64_t vid_offset;
  uint num_partitions;
  size_t total_mem_reserved;
  size_t size_per_part;
  uint64_t initial_num_records;
  uint64_t initial_num_records_per_part;

  void initializeMetaColumn();

  // void *getRow(uint64_t idx, ushort master_ver);
  // void *getRange(int start_idx, int end_idx);

  // void insert_or_update(uint64_t vid, const void *rec, ushort master_ver);
  // void update_partial(uint64_t vid, const void *data, ushort master_ver,
  //                     const std::vector<ushort> *col_idx);
};

};  // namespace storage

#endif /* STORAGE_ROW_STORE_HPP_ */
