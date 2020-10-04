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

#include "glo.hpp"
#include "mv-versions.hpp"

namespace storage::mv {

class MV_attributeList;
class MV_DAG;

template <typename T>
class MV_perAttribute {
 public:
  static constexpr bool isPerAttributeMVList = true;
  static constexpr bool isAttributeLevelMV = true;

  typedef typename T::version_t version_t;
  typedef typename T::version_chain_t version_chain_t;

  static auto create_versions(uint64_t xid, global_conf::IndexVal *idx_ptr,
                              const std::vector<size_t> &attribute_widths,
                              storage::DeltaStore &deltaStore,
                              ushort partition_id, const ushort *col_idx,
                              short num_cols) {
    return T::create_versions(xid, idx_ptr, attribute_widths, deltaStore,
                              partition_id, col_idx, num_cols);
  }

  static auto get_readable_version(
      const DeltaList &delta_list, uint64_t xid, char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      const ushort *col_idx = nullptr, ushort num_cols = 0) {
    return T::get_readable_version(delta_list, xid, write_loc,
                                   column_size_offset_pairs, col_idx, num_cols);
  }
};

class MV_attributeList {
 public:
  static constexpr bool isPerAttributeMVList = true;
  static constexpr bool isAttributeLevelMV = true;
  // typedef MVattributeListCol<VERSION_CHAIN> MVattributeLists;

  // Single-version as each version contains a single-attribute only.
  typedef VersionSingle version_t;
  typedef VersionChain<MV_attributeList> version_chain_t;
  typedef MVattributeListCol attributeVerList_t;

  static std::vector<MV_attributeList::version_t *> create_versions(
      uint64_t xid, global_conf::IndexVal *idx_ptr,
      const std::vector<size_t> &attribute_widths,
      storage::DeltaStore &deltaStore, ushort partition_id,
      const ushort *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      const DeltaList &delta_list, uint64_t xid, char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      const ushort *col_idx = nullptr, ushort num_cols = 0);

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
  typedef VersionMultiAttr version_t;
  typedef VersionChain<MV_DAG> version_chain_t;
  typedef MVattributeListCol attributeVerList_t;

  static std::vector<MV_DAG::version_t *> create_versions(
      uint64_t xid, global_conf::IndexVal *idx_ptr,
      const std::vector<size_t> &attribute_widths,
      storage::DeltaStore &deltaStore, ushort partition_id,
      const ushort *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      const DeltaList &delta_list, uint64_t xid, char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      const ushort *col_idx = nullptr, ushort num_cols = 0);
};

}  // namespace storage::mv

#endif  // PROTEUS_MV_ATTRIBUTE_LIST_HPP
