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

#include "oltp/storage/multi-version/mv-attribute-list.hpp"

#include "oltp/common/constants.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage::mv {

/* MV_attributeList
 *
 * */

//
// std::vector<MV_attributeList::version_t*> MV_attributeList::create_versions(
//    xid_t xid, global_conf::IndexVal* idx_ptr,
//    const std::vector<uint16_t>& attribute_widths,
//    storage::DeltaStore& deltaStore, partition_id_t partition_id,
//    const column_id_t* col_idx, short num_cols) {
//
//  std::vector<MV_attributeList::version_t*> version_pointers;
//
//  version_pointers.reserve(num_cols > 0 ? num_cols : attribute_widths.size());
//
//  // first check if the list-ptr is valid (revisit the init logic)
//
//  auto* mv_list_ptr = (attributeVerList_t*)idx_ptr->delta_list.ptr();
//  if (mv_list_ptr == nullptr) {
//    // deltaStore.validate_or_create_list();
//    // basically we dont want a regular version_list which contains head and
//    // all.
//    // we just want a chunk. so have a interface in delta-storage, from
//    // list-partition, not data, which will give you the chunk of your
//    required
//    // size.
//
//    auto size = attributeVerList_t::getSize(attribute_widths.size());
//    mv_list_ptr = (attributeVerList_t*)deltaStore.getTransientChunk(
//        idx_ptr->delta_list, size, partition_id);
//
//    attributeVerList_t::create(mv_list_ptr, attribute_widths.size());
//  }
//
//  // the for each updated column, create a version for it.
//  if (__likely(num_cols > 0 && col_idx != nullptr)) {
//    for (auto i = 0; i < num_cols; i++) {
//      // check if the old-list is valid, if not then create-new.
//      //
//      // 1) if the current list is valid, then used the last-upd-tmin from the
//      // list
//      //
//      // 2) if the list is new, then get the minimum active txn and use
//      // that.
//
//      auto c_idx = col_idx[i];
//
//      auto* attr_ver_list_ptr =
//          (MV_attributeList::version_chain_t*)
//              deltaStore.validate_or_create_list(
//                  mv_list_ptr->version_list[c_idx], partition_id);
//
//      void* ver_chunk =
//          deltaStore.create_version(attribute_widths.at(c_idx), partition_id);
//
//      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
//          attr_ver_list_ptr->last_updated_tmin, 0,
//          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));
//
//      attr_ver_list_ptr->insert(tmp);
//      version_pointers.emplace_back(tmp);
//
//      // in the end, update the list-last-upd-tmin to current xid.
//      attr_ver_list_ptr->last_updated_tmin = xid;
//    }
//    assert(version_pointers.size() == num_cols);
//  } else {
//    column_id_t i = 0;
//    for (auto& col_width : attribute_widths) {
//      auto* attr_ver_list_ptr =
//          (MV_attributeList::version_chain_t*)
//              deltaStore.validate_or_create_list(mv_list_ptr->version_list[i],
//                                                 partition_id);
//
//      void* ver_chunk = deltaStore.create_version(col_width, partition_id);
//
//      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
//          attr_ver_list_ptr->last_updated_tmin, 0,
//          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));
//
//      attr_ver_list_ptr->insert(tmp);
//
//      version_pointers.emplace_back(tmp);
//
//      // in the end, update the list-last-upd-tmin to current xid.
//      attr_ver_list_ptr->last_updated_tmin = xid;
//      i++;
//    }
//
//    assert(version_pointers.size() == attribute_widths.size());
//  }
//
//  return version_pointers;
//}

// std::bitset<64> MV_attributeList::get_readable_version(
//    const DeltaList& delta_list, xid_t xid, char* write_loc,
//    const std::vector<std::pair<uint16_t, uint16_t>>&
//    column_size_offset_pairs, const column_id_t* col_idx, short num_cols) {
//  std::bitset<64> done_mask(0xffffffffffffffff);
//
//  auto* mv_col_list = (MV_attributeList::attributeVerList_t*)delta_list.ptr();
//
//  if (__unlikely(num_cols <= 0 || col_idx == nullptr)) {
//    column_id_t i = 0;
//
//    if (mv_col_list == nullptr) {
//      done_mask.reset();
//      return done_mask;
//    }
//
//    for (auto& col_so_pair : column_size_offset_pairs) {
//      auto* col_ver_list =
//          (MV_attributeList::version_chain_t*)mv_col_list->version_list[i]
//              .ptr();
//      if (col_ver_list == nullptr || xid >= col_ver_list->last_updated_tmin) {
//        i++;
//        done_mask.reset(i);
//        continue;
//      }
//
//      auto* version = col_ver_list->get_readable_version(xid);
//
//      assert(version != nullptr);
//      memcpy(write_loc + col_so_pair.second, version->data,
//      col_so_pair.first); i++;
//    }
//  } else {
//    if (mv_col_list == nullptr) {
//      for (auto j = 0; j < num_cols; j++) {
//        done_mask.reset(col_idx[j]);
//      }
//      return done_mask;
//    }
//
//    for (auto j = 0; j < num_cols; j++) {
//      auto c_idx = col_idx[j];
//
//      auto* col_ver_list =
//          (MV_attributeList::version_chain_t*)mv_col_list->version_list[c_idx]
//              .ptr();
//
//      if (col_ver_list == nullptr || xid >= col_ver_list->last_updated_tmin) {
//        done_mask.reset(c_idx);
//        continue;
//      }
//
//      auto* version = col_ver_list->get_readable_version(xid);
//
//      assert(version != nullptr);
//      auto col_width = column_size_offset_pairs.at(c_idx).first;
//      memcpy(write_loc, version->data, col_width);
//
//      // done_mask.set(c_idx);
//      write_loc += col_width;
//    }
//  }
//
//  return done_mask;
//}

/* MV_DAG
 *
 * */

// std::vector<MV_DAG::version_t*> MV_DAG::create_versions(
//    uint64_t xid, global_conf::IndexVal* idx_ptr,
//    const std::vector<uint16_t>& attribute_widths, storage::DeltaStore&
//    deltaStore, ushort partition_id, const ushort* col_idx, short num_cols) {
//
//  // the main thing here is join the version into single one. and then connect
//  // appropriately. tmin thing here would be tricky here.
//
//  return {};
//}
//
//
// std::bitset<64> MV_DAG::get_readable_version(
//    global_conf::IndexVal* idx_ptr,
//    uint64_t xid,
//    char* write_loc,
//    const std::vector<std::pair<uint16_t, uint16_t>>&
//    column_size_offset_pairs, storage::DeltaStore** deltaStore, const ushort*
//    col_idx, ushort num_cols)
//    {
//  std::bitset<64> tmp;
//  return tmp;
//}

}  // namespace storage::mv
