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

#include <storage/multi-version/mv.hpp>

#include "glo.hpp"
#include "storage/multi-version/delta_storage.hpp"
#include "storage/table.hpp"

namespace storage::mv {

/* MV_attributeList
 *
 * */
std::vector<MV_attributeList::version_t*> MV_attributeList::create_versions(
    uint64_t xid, global_conf::IndexVal* idx_ptr,
    MV_attributeList::attributeVerList_t* mv_list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols) {
  std::vector<MV_attributeList::version_t*> version_pointers;
  version_pointers.reserve(num_cols > 0 ? num_cols : attribute_widths.size());

  // first check if the list-ptr is valid (revisit the init logic)

  // the for each updated column, create a version for it.
  if (__likely(num_cols > 0 && col_idx != nullptr)) {
    for (auto i = 0; i < num_cols; i++) {
      // check if the old-list is valid, if not then create-new.
      // 1) if the current list is valid, then used the last-upd-tmin from the
      // list 2) if the list is new, then get the minimum active txn and use
      // that.

      auto c_idx = col_idx[i];

      mv_list_ptr->attr_lists[c_idx].versions =
          (MV_attributeList::
               version_chain_t*)(deltaStore.validate_or_create_list(
              mv_list_ptr->attr_lists[c_idx].versions,
              mv_list_ptr->attr_lists[c_idx].delta_tag, partition_id));

      void* ver_chunk =
          deltaStore.create_version(attribute_widths.at(c_idx), partition_id);

      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
          mv_list_ptr->attr_lists[c_idx].versions->last_updated_tmin, 0,
          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));

      mv_list_ptr->attr_lists[c_idx].versions->insert(tmp);
      version_pointers.emplace_back(tmp);

      // in the end, update the list-last-upd-tmin to current xid.
      mv_list_ptr->attr_lists[c_idx].versions->last_updated_tmin = xid;
    }
    assert(version_pointers.size() == num_cols);
  } else {
    uint i = 0;
    for (auto& col_width : attribute_widths) {
      mv_list_ptr->attr_lists[i].versions =
          (MV_attributeList::
               version_chain_t*)(deltaStore.validate_or_create_list(
              mv_list_ptr->attr_lists[i].versions,
              mv_list_ptr->attr_lists[i].delta_tag, partition_id));

      void* ver_chunk = deltaStore.create_version(col_width, partition_id);

      auto* tmp = new (ver_chunk) MV_attributeList::version_t(
          mv_list_ptr->attr_lists[i].versions->last_updated_tmin, 0,
          (((char*)ver_chunk) + sizeof(MV_attributeList::version_t)));

      mv_list_ptr->attr_lists[i].versions->insert(tmp);
      version_pointers.emplace_back(tmp);

      // in the end, update the list-last-upd-tmin to current xid.
      mv_list_ptr->attr_lists[i].versions->last_updated_tmin = xid;
      i++;
    }

    assert(version_pointers.size() == attribute_widths.size());
  }

  return version_pointers;
}

std::bitset<64> MV_attributeList::get_readable_version(
    global_conf::IndexVal* idx_ptr,
    MV_attributeList::attributeVerList_t* list_ptr, uint64_t xid,
    char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    storage::DeltaStore** deltaStore, const ushort* col_idx, ushort num_cols) {
  std::bitset<64> done_mask;
  done_mask.set();

  if (__unlikely(num_cols == 0 || col_idx == nullptr)) {
    uint i = 0;
    for (auto& col_so_pair : column_size_offset_pairs) {
      auto delta_idx = storage::DeltaStore::extract_delta_idx(
          list_ptr->attr_lists[i].delta_tag);
      bool is_valid_list =
          deltaStore[delta_idx]->verifyTag(list_ptr->attr_lists[i].delta_tag);

      // if the list is not valid, then need to be read from main!
      // list is valid, then check if the last_updated_in_list. if it is >=,
      // meaning this attribute is readable from main

      if (!is_valid_list ||
          xid >= list_ptr->attr_lists[i].versions->last_updated_tmin) {
        i++;
        done_mask.reset(i);
        continue;
      }

      auto version =
          list_ptr->attr_lists[i].versions->get_readable_version(xid);
      assert(version != nullptr);
      memcpy(write_loc + col_so_pair.second, version->data, col_so_pair.first);
      // done_mask.set(i);
      i++;
    }
  } else {
    for (auto j = 0; j < num_cols; j++) {
      auto c_idx = col_idx[j];

      auto delta_idx = storage::DeltaStore::extract_delta_idx(
          list_ptr->attr_lists[c_idx].delta_tag);
      bool is_valid_list = deltaStore[delta_idx]->verifyTag(
          list_ptr->attr_lists[c_idx].delta_tag);

      if (!is_valid_list ||
          xid >= list_ptr->attr_lists[c_idx].versions->last_updated_tmin) {
        done_mask.reset(c_idx);
        continue;
      }

      auto col_width = column_size_offset_pairs.at(c_idx).first;
      auto version =
          list_ptr->attr_lists[c_idx].versions->get_readable_version(xid);
      assert(version != nullptr);
      memcpy(write_loc, version->data, col_width);

      // done_mask.set(c_idx);
      write_loc += col_width;
    }
  }

  return done_mask;
}

/* MV_DAG
 *
 * */

std::vector<MV_DAG::version_t*> MV_DAG::create_versions(
    uint64_t xid, global_conf::IndexVal* idx_ptr,
    MV_DAG::attributeVerList_t* mv_list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols) {
  // the main thing here is join the version into single one. and then connect
  // appropriately. tmin thing here would be tricky here.

  return {};
  //
  //  MV_DAG::version_t* version_pointer;
  //
  //
  //  // Create one big version.
  //
  //  // variable to be set by mask_creator
  //  std::vector<size_t> ver_offsets;
  //  std::bitset<64> attr_mask;
  //
  //  auto ver_data_size =
  //  MV_DAG::version_t::get_partial_mask_size(attribute_widths,
  //                                                                ver_offsets,attr_mask,
  //                                                                col_idx,
  //                                                                num_cols  );
  //
  //  void* ver_chunk = deltaStore.create_version(ver_data_size, partition_id);
  //  MV_DAG::version_t *ver_tmp = new (ver_chunk) MV_DAG::version_t(
  //      TMIN, TMAX,
  //      (((char*)ver_chunk) + sizeof(MV_DAG::version_t)), attr_mask,
  //      ver_offsets);
  //
  //  //ver_tmp->create_partial_mask(ver_offsets, attr_mask);
  //
  //  // Now the we have the big memory for version, now create links and alter
  //  the DAG.
  //
  //
  //  if (__likely(num_cols > 0 && col_idx != nullptr)) {
  //    for (auto i = 0; i < num_cols; i++) {
  //      // check if the old-list is valid, if not then create-new.
  //      // 1) if the current list is valid, then used the last-upd-tmin from
  //      the
  //      // list 2) if the list is new, then get the minimum active txn and use
  //      // that.
  //
  //      mv_list_ptr->attr_lists[i].versions =
  //          (MV_DAG::
  //          version_chain_t*)(deltaStore.validate_or_create_list(
  //              mv_list_ptr->attr_lists[i].versions,
  //              mv_list_ptr->attr_lists[i].delta_tag, partition_id));
  //
  ////      void* ver_chunk = deltaStore.create_version(
  ////          attribute_widths.at(col_idx[i]), partition_id);
  ////
  ////      auto* tmp = new (ver_chunk) MV_DAG::version_t(
  ////          mv_list_ptr->attr_lists[i].versions->last_updated_tmin, 0,
  ////          (((char*)ver_chunk) + sizeof(MV_DAG::version_t)));
  //
  //      mv_list_ptr->attr_lists[i].versions->insert(ver_tmp);
  ////      version_pointers.emplace_back(tmp);
  //
  //      // in the end, update the list-last-upd-tmin to current xid.
  //      mv_list_ptr->attr_lists[i].versions->last_updated_tmin = xid;
  //    }
  //  } else {
  //    uint i = 0;
  //    for (auto& col_width : attribute_widths) {
  //      mv_list_ptr->attr_lists[i].versions =
  //          (MV_DAG::
  //          version_chain_t*)(deltaStore.validate_or_create_list(
  //              mv_list_ptr->attr_lists[i].versions,
  //              mv_list_ptr->attr_lists[i].delta_tag, partition_id));
  //
  //      mv_list_ptr->attr_lists[i].versions->insert(ver_tmp);
  //
  //      // in the end, update the list-last-upd-tmin to current xid.
  //      mv_list_ptr->attr_lists[i].versions->last_updated_tmin = xid;
  //      i++;
  //    }
  //  }
  //
  //
  //
  //  return {version_pointer};
}

// void init_reading_mask(std::bitset<64> &done_mask,std::bitset<64>
// &required_mask, std::vector<uint> &return_col_offsets,
//                 const std::vector<std::pair<size_t, size_t>>&
//                 column_size_offset_pairs,
//                       const ushort *col_idx, ushort num_cols){
//  // CREATE containment mask.
//
//  if (num_cols > 0) {
//    assert(num_cols <= 64 && "MAX columns supported: 64");
//
//    done_mask.set();
//    for (auto i = 0, offset = 0; i < num_cols; i++) {
//      done_mask.reset(col_idx[i]);
//      required_mask.set(col_idx[i]);
//
//      offset += column_size_offset_pairs[col_idx[i]].first;
//      return_col_offsets.push_back(offset);
//    }
//  } else {
//    for (auto i = column_size_offset_pairs.size(); i < done_mask.size(); i++)
//      done_mask.reset(i);
//  }
//  required_mask = ~done_mask;
//
//  assert(!(done_mask.all()) && "havent even started and its done?");
//}

std::bitset<64> MV_DAG::get_readable_version(
    global_conf::IndexVal* idx_ptr,
    MV_attributeList::attributeVerList_t* list_ptr, uint64_t xid,
    char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    storage::DeltaStore** deltaStore, const ushort* col_idx, ushort num_cols) {
  std::bitset<64> tmp;
  return tmp;

  // this is tricky, once traversed, then how to get the other attribute.

  // -------
  // create-mask for required and done attributes.

  // first filter-out all those attributes which are not-readable from MV
  // (readable from main)

  // from remaining, choose one list, and get readable.

  // think: keep track of other attributes you encounter.
  // or get the version, if check if other attribute is there, and then see...
}

}  // namespace storage::mv
