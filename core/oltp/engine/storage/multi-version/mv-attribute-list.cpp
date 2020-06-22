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

#include "storage/multi-version/mv-attribute-list.hpp"

#include "glo.hpp"
#include "storage/multi-version/delta_storage.hpp"

namespace storage::mv {

/* MV_attributeList
 *
 * */
std::vector<MV_attributeList::version_t*> MV_attributeList::create_versions(
    global_conf::IndexVal* idx_ptr,
    MV_attributeList::attributeVerList_t* mv_list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols) {
  std::vector<MV_attributeList::version_t*> version_pointers;
  version_pointers.reserve(num_cols > 0 ? num_cols : attribute_widths.size());

  // first check if the list-ptr is valid (revisit the init logic)

  // the for each updated column, create a version for it.
  if (__likely(num_cols > 0 && col_idx != nullptr)) {
    for (auto i = 0; i < num_cols; i++) {
      void* ver_chunk = deltaStore.create_version(
          attribute_widths.at(col_idx[i]), partition_id);
      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
          idx_ptr->t_min, 0,
          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));

      // Check if the list if valid, if not, then do the list thing,
      mv_list_ptr->version_lists[i] =
          static_cast<MV_attributeList::version_chain_t*>(
              deltaStore.validate_or_create_list(mv_list_ptr->version_lists[i],
                                                 mv_list_ptr->delta_tags[i],
                                                 partition_id));

      mv_list_ptr->version_lists[i]->insert(tmp);
      version_pointers.emplace_back(tmp);
    }
    assert(version_pointers.size() == num_cols);
  } else {
    uint i = 0;
    for (auto& col_width : attribute_widths) {
      void* ver_chunk = deltaStore.create_version(
          attribute_widths.at(col_idx[i]), partition_id);
      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
          idx_ptr->t_min, 0,
          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));

      mv_list_ptr->version_lists[i] =
          static_cast<MV_attributeList::version_chain_t*>(
              deltaStore.validate_or_create_list(mv_list_ptr->version_lists[i],
                                                 mv_list_ptr->delta_tags[i],
                                                 partition_id));

      mv_list_ptr->version_lists[i]->insert(tmp);
      version_pointers.emplace_back(tmp);
      i++;
    }

    assert(version_pointers.size() == attribute_widths.size());
  }

  return version_pointers;
}

std::bitset<64> MV_attributeList::get_readable_version(
    MV_attributeList::attributeVerList_t* list_ptr, uint64_t xid,
    char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    const ushort* col_idx, ushort num_cols) {
  std::bitset<64> done_mask;

  if (__unlikely(num_cols == 0 || col_idx == nullptr)) {
    uint i = 0;
    for (auto& col_so_pair : column_size_offset_pairs) {
      // verify delta-tag.
      // FIXME: get and verify delta-tag (assert). extra delta-id from the tag
      //        and then verify against that delta-store if the tag is valid.

      auto version = list_ptr->version_lists[i]->get_readable_version(xid);
      if (version != nullptr) {
        memcpy(write_loc + col_so_pair.second, version->data,
               col_so_pair.first);
        done_mask.set(i);
      }
      i++;
    }
  } else {
    for (auto i = 0; i < num_cols; i++) {
      auto c_idx = col_idx[i];

      // verify delta-tag.
      // FIXME: get and verify delta-tag (assert). extra delta-id from the tag
      //        and then verify against that delta-store if the tag is valid.
      // list_ptr->delta_tags[c_idx];

      auto col_width = column_size_offset_pairs.at(c_idx).first;
      auto version = list_ptr->version_lists[c_idx]->get_readable_version(xid);
      if (version != nullptr) {
        memcpy(write_loc, version->data, col_width);
        done_mask.set(i);
      }
      write_loc += col_width;
    }
  }

  return done_mask;
}

/* MV_DAG
 *
 * */

std::vector<MV_DAG::version_t*> MV_DAG::create_versions(
    global_conf::IndexVal* idx_ptr, MV_DAG::attributeVerList_t* mv_list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols) {
  return {};
}

std::bitset<64> MV_DAG::get_readable_version(
    version_t* head, uint64_t xid, char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    const ushort* col_idx, ushort num_cols) {
  std::bitset<64> tmp;
  return tmp;
}

}  // namespace storage::mv
