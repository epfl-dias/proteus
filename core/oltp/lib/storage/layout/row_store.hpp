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

#ifndef STORAGE_ROW_STORE_HPP_
#define STORAGE_ROW_STORE_HPP_

#include <cassert>
#include <deque>
#include <iostream>
#include <map>
#include <platform/memory/allocator.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "oltp/common/atomic_bit_set.hpp"
#include "oltp/common/common.hpp"
#include "oltp/common/constants.hpp"
#include "oltp/common/memory-chunk.hpp"
#include "oltp/storage/table.hpp"

namespace storage {

class RowStore;

/*  DATA LAYOUT -- ROW STORE
 */

class RowStore : public Table {
 public:
  RowStore(table_id_t table_id, std::string name, TableDef columns,
           bool indexed = true, bool numa_partitioned = true,
           size_t reserved_capacity = 1000000, int numa_idx = -1);

  ~RowStore() override;

  global_conf::IndexVal *insertRecord(const void *data, xid_t transaction_id,
                                      partition_id_t partition_id,
                                      master_version_t master_ver = 0) override;
  global_conf::IndexVal *insertRecordBatch(
      const void *data, size_t num_records, size_t max_capacity,
      xid_t transaction_id, partition_id_t partition_id,
      master_version_t master_ver = 0) override;

  void updateRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                    void *data, delta_id_t current_delta_id,
                    const column_id_t *col_idx = nullptr,
                    short num_columns = -1,
                    master_version_t master_ver = 0) override;

  [[noreturn]] void updateRecordBatch(
      xid_t transaction_id, global_conf::IndexVal *index_ptr, void *data,
      size_t num_records, delta_id_t current_delta_id,
      const column_id_t *col_idx = nullptr, short num_columns = -1,
      master_version_t master_ver = 0) override {
    throw std::runtime_error("Unimplemented");
  }

  void deleteRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                    master_version_t master_ver = 0) override {
    throw std::runtime_error("Unimplemented");
  }

  void getIndexedRecord(xid_t transaction_id, global_conf::IndexVal *index_ptr,
                        void *destination, const column_id_t *col_idx = nullptr,
                        short num_cols = -1) override;

  void getRecord(xid_t transaction_id, rowid_t rowid, void *destination,
                 const column_id_t *col_idx = nullptr,
                 short num_cols = -1) override;

 private:
  size_t rec_size;
  std::vector<std::string> columns;
  // 1-size, 2-cumm size until that col
  std::vector<std::pair<size_t, size_t>> column_width;
  std::vector<data_type> column_data_types;
  // vector of partitions.
  std::vector<std::vector<oltp::common::mem_chunk>>
      data_[global_conf::num_master_versions];

  std::vector<std::vector<oltp::common::mem_chunk>> metadata;

  bool indexed;
  uint64_t vid_offset;
  uint num_partitions;
  size_t total_mem_reserved;
  size_t size_per_part;
  uint64_t initial_num_records;
  uint64_t initial_num_records_per_part;

 private:
  void initializeMetaColumn();

  // void *getRow(uint64_t idx, ushort master_ver);
  // void *getRange(int start_idx, int end_idx);

  // void insert_or_update(uint64_t vid, const void *rec, ushort master_ver);
  // void update_partial(uint64_t vid, const void *data, ushort master_ver,
  //                     const std::vector<ushort> *col_idx);
};

};  // namespace storage

#endif /* STORAGE_ROW_STORE_HPP_ */
