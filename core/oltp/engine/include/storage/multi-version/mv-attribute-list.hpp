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

  static auto create_versions(global_conf::IndexVal *idx_ptr, void *list_ptr,
                              std::vector<size_t> &attribute_widths,
                              storage::DeltaStore &deltaStore,
                              ushort partition_id, const ushort *col_idx,
                              short num_cols) {
    // static_cast<T&>(*this).destroy_snapshot_();
    // T::create_versions(list_ptr, attribute_widths, col_idx, num_cols);
    auto *tmp_list_ptr = (typename T::attributeVerList_t *)list_ptr;

    return T::create_versions(idx_ptr, tmp_list_ptr, attribute_widths,
                              deltaStore, partition_id, col_idx, num_cols);
  }

  static auto get_readable_version(
      void *list_ptr, uint64_t xid, char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      const ushort *col_idx = nullptr, ushort num_cols = 0) {
    auto *tmp_list_ptr = (typename T::attributeVerList_t *)list_ptr;

    return T::get_readable_version(tmp_list_ptr, xid, write_loc,
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
  typedef MVattributeListCol<version_chain_t> attributeVerList_t;

  // required stuff to create versions
  // - pointer to mv-lists
  // - column_indexes
  // - column_sizes (attribute_sizes) .. this is somewhere already saved i
  // guess. (column_size)

  // return: pointers (ordered with col_idxes) where columnstore can create
  // versions directly. diff in dag vs multi: nothing here, but in list
  // connections!
  //  static std::vector<MV_attributeList::version_t*> create_versions(
  //      void *list_ptr, std::vector<size_t> &attribute_widths,
  //      const ushort *col_idx, short num_cols);

  static std::vector<MV_attributeList::version_t *> create_versions(
      global_conf::IndexVal *idx_ptr,
      MV_attributeList::attributeVerList_t *list_ptr,
      std::vector<size_t> &attribute_widths, storage::DeltaStore &deltaStore,
      ushort partition_id, const ushort *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      MV_attributeList::attributeVerList_t *list_ptr, uint64_t xid,
      char *write_loc,
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
  typedef MVattributeListCol<version_chain_t> attributeVerList_t;

  static std::vector<MV_DAG::version_t *> create_versions(
      void *list_ptr, std::vector<size_t> &attribute_widths,
      const ushort *col_idx, short num_cols);

  static std::vector<MV_DAG::version_t *> create_versions(
      global_conf::IndexVal *idx_ptr, MV_DAG::attributeVerList_t *list_ptr,
      std::vector<size_t> &attribute_widths, storage::DeltaStore &deltaStore,
      ushort partition_id, const ushort *col_idx, short num_cols);

  static std::bitset<64> get_readable_version(
      version_t *head, uint64_t xid, char *write_loc,
      const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
      const ushort *col_idx = nullptr, ushort num_cols = 0);
};

}  // namespace storage::mv

#endif  // PROTEUS_MV_ATTRIBUTE_LIST_HPP
