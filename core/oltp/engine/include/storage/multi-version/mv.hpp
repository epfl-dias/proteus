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

// MV Types

#include "storage/multi-version/mv-attribute-list.hpp"
#include "storage/multi-version/mv-record-list.hpp"

namespace storage::mv {

using mv_type = MV_RecordList_Full;
// using mv_type = MV_RecordList_Partial;

// using mv_type = MV_perAttribute<MV_attributeList>;
// using mv_type = MV_perAttribute<MV_DAG>;

using mv_version_chain = mv_type::version_chain_t;
using mv_version = mv_type::version_t;

template <typename MvType = mv_type>
class MultiVersionStorage_impl {
 public:
  static constexpr bool isPerAttributeMVList = MvType::isPerAttributeMVList;
  static constexpr bool isAttributeLevelMV = MvType::isAttributeLevelMV;

  typedef typename MvType::version_t version_t;
  typedef typename MvType::version_chain_t version_chain_t;

  MultiVersionStorage_impl<MvType> getType() const { return {}; }
  static auto create_versions(uint64_t xid, global_conf::IndexVal *idx_ptr,
                              std::vector<size_t> &attribute_widths,
                              storage::DeltaStore &deltaStore,
                              ushort partition_id, const ushort *col_idx,
                              short num_cols) {
    return MvType::create_versions(xid, idx_ptr, attribute_widths, deltaStore,
                                   partition_id, col_idx, num_cols);
  }

  static auto get_readable_version(
      global_conf::IndexVal *idx_ptr, void *list_ptr, uint64_t xid,
      char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      storage::DeltaStore **deltaStore, const ushort *col_idx = nullptr,
      ushort num_cols = 0) {
    return MvType::get_readable_version(idx_ptr, list_ptr, xid, write_loc,
                                        column_size_offset_pairs, deltaStore,
                                        col_idx, num_cols);
  }
};

}  // namespace storage::mv

#endif  // PROTEUS_OLTP_MV_HPP
