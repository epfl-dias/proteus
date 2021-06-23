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

#ifndef PROTEUS_MV_ATTRIBUTE_LIST_HPP
#define PROTEUS_MV_ATTRIBUTE_LIST_HPP

#include "oltp/common/constants.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/multi-version/mv-versions.hpp"

namespace storage::mv {

class MV_attributeList;
class MV_DAG;

template <typename T>
class MV_perAttribute {
 public:
  static constexpr bool isPerAttributeMVList = true;
  static constexpr bool isAttributeLevelMV = true;

  using version_t = typename T::version_t;
  using version_chain_t = typename T::version_chain_t;

  static auto create_versions(xid_t xid, global_conf::IndexVal *idx_ptr,
                              const std::vector<uint16_t> &attribute_widths,
                              storage::DeltaStore &deltaStore,
                              partition_id_t partition_id,
                              const column_id_t *col_idx, short num_cols) {
    return T::create_versions(xid, idx_ptr, attribute_widths, deltaStore,
                              partition_id, col_idx, num_cols);
  }

  static auto get_readable_version(
      const DeltaMemoryPtr &delta_list, const txn::TxnTs &txTs, char *write_loc,
      const std::vector<std::pair<uint16_t, uint16_t>>
          &column_size_offset_pairs,
      const column_id_t *col_idx = nullptr, short num_cols = 0) {
    return T::get_readable_version(delta_list, txTs, write_loc,
                                   column_size_offset_pairs, col_idx, num_cols);
  }

  static void rollback(const txn::TxnTs &txTs, global_conf::IndexVal *idx_ptr,
                       ColumnVector &columns,
                       const column_id_t *col_idx = nullptr,
                       short num_cols = 0) {
    assert(false);
  }
};

class MV_attributeList {
 public:
  static constexpr bool isPerAttributeMVList = true;
  static constexpr bool isAttributeLevelMV = true;
  // typedef MVattributeListCol<VERSION_CHAIN> MVattributeLists;

  // Single-version as each version contains a single-attribute only.
  using version_t = VersionSingle;
  using version_chain_t = VersionChain<MV_attributeList>;
  using attributeVerList_t = MVattributeListCol;

  static std::vector<MV_attributeList::version_t *> create_versions(
      xid_t xid, global_conf::IndexVal *idx_ptr,
      const std::vector<uint16_t> &attribute_widths,
      storage::DeltaStore &deltaStore, partition_id_t partition_id,
      const column_id_t *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      const DeltaMemoryPtr &delta_list, const txn::TxnTs &txTs, char *write_loc,
      const std::vector<std::pair<uint16_t, uint16_t>>
          &column_size_offset_pairs,
      const column_id_t *col_idx = nullptr, short num_cols = 0);

  // friend class MV_perAttribute<MV_attributeList>;
};

/* Class: MV_DAG
 * Description:
 *
 * Layout:
 *
 *
 * Traversal Algo:
 *
 *
 * */

class MV_DAG {
 public:
  static constexpr bool isPerAttributeMVList = true;
  static constexpr bool isAttributeLevelMV = true;

  // Multi-version as each version contains a multiple-attribute.
  using version_t = VersionMultiAttr;
  using version_chain_t = VersionChain<MV_DAG>;
  using attributeVerList_t = MVattributeListCol;

  static std::vector<MV_DAG::version_t *> create_versions(
      xid_t xid, global_conf::IndexVal *idx_ptr,
      const std::vector<uint16_t> &attribute_widths,
      storage::DeltaStore &deltaStore, partition_id_t partition_id,
      const column_id_t *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      const DeltaMemoryPtr &delta_list, xid_t xid, char *write_loc,
      const std::vector<std::pair<uint16_t, uint16_t>>
          &column_size_offset_pairs,
      const column_id_t *col_idx = nullptr, short num_cols = 0);
};

}  // namespace storage::mv

#endif  // PROTEUS_MV_ATTRIBUTE_LIST_HPP
