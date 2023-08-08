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

#ifndef PROTEUS_OLTP_MV_HPP
#define PROTEUS_OLTP_MV_HPP

#include "oltp/storage/multi-version/mv-attribute-list.hpp"
#include "oltp/storage/multi-version/mv-record-list.hpp"
#include "oltp/storage/table.hpp"

namespace storage::mv {

template <typename MvType = MV_RecordList_Full>
class MultiVersionStorage_impl {
 public:
  static constexpr bool isPerAttributeMVList = MvType::isPerAttributeMVList;
  static constexpr bool isAttributeLevelMV = MvType::isAttributeLevelMV;

  using version_t = typename MvType::version_t;
  using version_chain_t = typename MvType::version_chain_t;

  MultiVersionStorage_impl<MvType> getType() const { return {}; }

  static inline auto create_versions(
      uint64_t xid, global_conf::IndexVal* idx_ptr, const Table& tableRef,
      storage::DeltaStore& deltaStore, partition_id_t partition_id,
      const column_id_t* col_idx, short num_cols) {
    return MvType::create_versions(xid, idx_ptr, tableRef, deltaStore,
                                   partition_id, col_idx, num_cols);
  }

  static inline auto get_readable_version(
      const global_conf::IndexVal& index_ptr, const txn::TxnTs& txTs,
      char* write_loc, const Table& tableRef,
      const column_id_t* col_idx = nullptr, short num_cols = 0) {
    return MvType::get_readable_version(index_ptr, txTs, write_loc, tableRef,
                                        col_idx, num_cols);
  }

  static void gc(global_conf::IndexVal* idx_ptr, txn::TxnTs minTxnTs) {
    MvType::gc(idx_ptr, minTxnTs);
  }
};

/*
 *  MV_RecordList_Full: (Working)
 *      copy and stores the full record version
 *  MV_RecordList_Partial: (Partially working)
 *      stores only the changed attributes (change delta)
 *
 *  MV_perAttribute<MV_attributeList>: (Broken)
 *      version list-per-attribute(column)
 *  MultiVersionStorage_impl<MV_perAttribute<MV_DAG>>: (Broken)
 *      version list-per-attribute but in a DAG to have a better traversal
 * algorithm when accessing multiple attributes from version storage.
 * */

using mv_type = MultiVersionStorage_impl<MV_RecordList_Full>;
// using mv_type = MultiVersionStorage_impl<MV_RecordList_Partial>;
// using mv_type = MultiVersionStorage_impl<MV_perAttribute<MV_attributeList>>;
// using mv_type = MultiVersionStorage_impl<MV_perAttribute<MV_DAG>>;

}  // namespace storage::mv

#endif  // PROTEUS_OLTP_MV_HPP
