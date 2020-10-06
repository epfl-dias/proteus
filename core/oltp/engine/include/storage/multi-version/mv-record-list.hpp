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

#include "mv-versions.hpp"

namespace storage {
namespace mv {

class MV_RecordList_Full;
class MV_RecordList_Partial;

/* Class: MV_RecordList_Full
 * FIXME:
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
  using version_chain_t = VersionChain<MV_RecordList_Full>;

  static std::bitset<1> get_readable_version(
      const DeltaList& delta_list, uint64_t tid_self, char* write_loc,
      const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
      const ushort* col_idx = nullptr, ushort num_cols = 0);

  static std::vector<MV_RecordList_Full::version_t*> create_versions(
      uint64_t xid, global_conf::IndexVal* idx_ptr,
      std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
      ushort partition_id, const ushort* col_idx, short num_cols);

 private:
  static void* get_readable_version(version_t* head, uint64_t tid_self);
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
      const DeltaList& delta_list, uint64_t tid_self, char* write_loc,
      const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
      const ushort* col_idx = nullptr, ushort num_cols = 0);

  static std::vector<MV_RecordList_Partial::version_t*> create_versions(
      uint64_t xid, global_conf::IndexVal* idx_ptr,
      std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
      ushort partition_id, const ushort* col_idx, short num_cols);
};

}  // namespace mv
}  // namespace storage

#endif  // PROTEUS_MV_RECORD_LIST_HPP
