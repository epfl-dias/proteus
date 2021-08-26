/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_MV_RECORD_LIST_HPP
#define PROTEUS_MV_RECORD_LIST_HPP

#include <bitset>
#include <cassert>
#include <iostream>
#include <utility>

#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/mv-versions.hpp"

namespace storage::mv {

/* Class: MV_RecordList_Full
 * Description:
 *
 * Layout:
 *
 *
 * Traversal Algo:
 *
 *
 * */

class MV_RecordList_Full {
 public:
  static constexpr bool isAttributeLevelMV = false;
  static constexpr bool isPerAttributeMVList = false;

  using version_t = VersionSingle;
  using version_chain_t = size_t;  // VersionChain<MV_RecordList_Full>;

  static std::bitset<1> get_readable_version(
      const DeltaPtr& delta_list, const txn::TxnTs& txTs, char* write_loc,
      const std::vector<std::pair<uint16_t, uint16_t>>&
          column_size_offset_pairs,
      const column_id_t* col_idx = nullptr, short num_cols = 0);

  static std::vector<MV_RecordList_Full::version_t*> create_versions(
      xid_t xid, global_conf::IndexVal* idx_ptr,
      std::vector<uint16_t>& attribute_widths, storage::DeltaStore& deltaStore,
      partition_id_t partition_id, const column_id_t* col_idx, short num_cols);

  static void gc(global_conf::IndexVal* idx_ptr, txn::TxnTs minTxnTs);
};

/* Class: MV_RecordList_Partial
 * Description: single: one list per-relation.
 *             However, each version contains updated-attributed only.
 * */

class MV_RecordList_Partial {
 public:
  static constexpr bool isAttributeLevelMV = true;
  static constexpr bool isPerAttributeMVList = false;

  using version_t = VersionMultiAttr;
  using version_chain_t = VersionChain<MV_RecordList_Partial>;

  static std::bitset<64> get_readable_version(
      const DeltaPtr& delta_list, const txn::TxnTs& txTs, char* write_loc,
      const std::vector<std::pair<uint16_t, uint16_t>>&
          column_size_offset_pairs,
      const column_id_t* col_idx = nullptr, short num_cols = 0);

  static std::vector<MV_RecordList_Partial::version_t*> create_versions(
      xid_t xid, global_conf::IndexVal* idx_ptr,
      std::vector<uint16_t>& attribute_widths, storage::DeltaStore& deltaStore,
      partition_id_t partition_id, const column_id_t* col_idx, short num_cols);

  //  static void rollback(const txn::TxnTs& txTs, global_conf::IndexVal*
  //  idx_ptr,
  //                       ColumnVector& columns,
  //                       const column_id_t* col_idx = nullptr,
  //                       short num_cols = 0);
};

}  // namespace storage::mv

#endif  // PROTEUS_MV_RECORD_LIST_HPP
