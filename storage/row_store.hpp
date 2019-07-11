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

#ifndef ROW_STORE_HPP_
#define ROW_STORE_HPP_

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

class rowStore : public Table {
  rowStore(uint8_t table_id, std::string name,
           std::vector<std::tuple<std::string, data_type, size_t>> columns,
           uint64_t initial_num_records = 10000000);
  /*
  public:
  ColumnStore(uint8_t table_id, std::string name,
              std::vector<std::tuple<std::string, data_type, size_t>> columns,
              uint64_t initial_num_records = 10000000);
  uint64_t insertRecord(void* rec, short master_ver);
  void* insertRecord(void* rec, uint64_t xid, short master_ver);
  void updateRecord(uint64_t vid, void* data, short ins_master_ver,
                    short prev_master_ver, short delta_ver, uint64_t tmin,
                    uint64_t tmax, int pid);
  void updateRecord(uint64_t vid, void* rec, short ins_master_ver,
                    short prev_master_ver, short delta_ver, uint64_t tmin,
                    uint64_t tmax, int pid, std::vector<int>* col_idx);
  void deleteRecord(uint64_t vid, short master_ver);
  std::vector<const void*> getRecordByKey(uint64_t vid, short master_ver,
                                          std::vector<int>* col_idx = nullptr);

  void getRecordByKey(uint64_t vid, short master_ver, std::vector<int>* col_idx,
                      void* loc);
  void touchRecordByKey(uint64_t vid, short master_ver);

  global_conf::mv_version_list* getVersions(uint64_t vid, short master_ver);


  ~ColumnStore();

 private:
  std::vector<Column*> columns;
  Column* meta_column;
  Column** secondary_index_vals;
  size_t rec_size;
  */
 public:
  uint64_t insertRecord(void* rec, ushort master_ver);
  void* insertRecord(void* rec, uint64_t xid, ushort master_ver);

  void updateRecord(uint64_t vid, const void* data, ushort ins_master_ver,
                    ushort prev_master_ver, ushort delta_ver, uint64_t tmin,
                    uint64_t tmax, ushort pid);

  void updateRecord(uint64_t vid, const void* data, ushort ins_master_ver,
                    ushort prev_master_ver, ushort delta_ver, uint64_t tmin,
                    uint64_t tmax, ushort pid, std::vector<ushort>* col_idx);

  void deleteRecord(uint64_t vid, ushort master_ver) {}

  global_conf::mv_version_list* getVersions(uint64_t vid, ushort delta_ver);

  void touchRecordByKey(uint64_t vid, ushort master_ver);

  std::vector<const void*> getRecordByKey(uint64_t vid, ushort master_ver,
                                          const std::vector<ushort>* col_idx);

  void getRecordByKey(uint64_t vid, ushort master_ver,
                      const std::vector<ushort>* col_idx, void* loc);

 private:
  size_t rec_size;
  Column* meta_column;
  std::vector<std::string> columns;
  std::vector<std::pair<size_t, size_t>>
      column_width;  // 1-size, 2-cumm size until that col
  std::vector<data_type> column_data_types;
  std::vector<mem_chunk*> master_versions[global_conf::num_master_versions];

  void* getRow(uint64_t idx, ushort master_ver);
  void* getRange(int start_idx, int end_idx);

  void insert_or_update(uint64_t vid, const void* rec, ushort master_ver);
  void update_partial(uint64_t vid, const void* data, ushort master_ver,
                      const std::vector<ushort>* col_idx);
};

};  // namespace storage

#endif /* ROW_STORE_HPP_ */
