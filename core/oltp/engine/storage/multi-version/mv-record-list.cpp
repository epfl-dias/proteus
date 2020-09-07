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

#include "storage/multi-version/mv-record-list.hpp"

#include <iostream>

#include "glo.hpp"

namespace storage::mv {

void* MV_RecordList_Full::get_readable_version(version_t* head,
                                               uint64_t tid_self) {
  version_t* tmp = nullptr;
  {
    tmp = head;
    // C++ standard says that (x == NULL) <=> (x==nullptr)
    while (tmp != nullptr) {
      // if (CC_MV2PL::is_readable(tmp->t_min, tmp->t_max, tid_self)) {
      if (global_conf::ConcurrencyControl::is_readable(tmp->t_min, tid_self)) {
        return tmp->data;
      } else {
        tmp = tmp->next;
      }
    }
  }
  return nullptr;
}
void MV_RecordList_Full::get_readable_version(version_t* head,
                                              uint64_t tid_self,
                                              char* write_loc, uint rec_size) {
  // char* version = (char*)this->get_readable_version(tid_self);
  char* version =
      (char*)MV_RecordList_Full::get_readable_version(head, tid_self);
  assert(version != nullptr);
  assert(write_loc != nullptr);
  memcpy(write_loc, version, rec_size);
}

std::bitset<1> MV_RecordList_Full::get_readable_version(
    void* list_ptr, const uint64_t tid_self, char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    const ushort* col_idx, const ushort num_cols) {
  static thread_local std::bitset<1> ret_bitmask("1");

  version_t* head = static_cast<version_t*>(list_ptr);
  // char* version = (char*)this->get_readable_version(tid_self);
  char* version =
      (char*)MV_RecordList_Full::get_readable_version(head, tid_self);
  assert(version != nullptr);

  if (__unlikely(col_idx == nullptr || num_cols == 0)) {
    for (auto& col_s_pair : column_size_offset_pairs) {
      memcpy(write_loc, version + col_s_pair.second, col_s_pair.first);
      write_loc += col_s_pair.first;
    }

  } else {
    // copy the required attr
    for (ushort i = 0; i < num_cols; i++) {
      auto& col_s_pair = column_size_offset_pairs[col_idx[i]];
      // assumption: full row is in the version.
      memcpy(write_loc, version + col_s_pair.second, col_s_pair.first);
      write_loc += col_s_pair.first;
    }
  }
  return ret_bitmask;
}

//  void *get_readable_ver(uint64_t tid_self, uint col_id) {
//    VERSION *tmp = nullptr;
//    {
//      tmp = head;
//      // C++ standard says that (x == NULL) <=> (x==nullptr)
//      while (tmp != nullptr) {
//        // if (CC_MV2PL::is_readable(tmp->t_min, tmp->t_max, tid_self)) {
//        if ((tmp->col_ids.empty() || tmp->col_ids.find(col_id) !=
//        tmp->col_ids.end()) &&
//            CC_MV2PL::is_readable(tmp->t_min, tid_self)) {
//          return tmp->data;
//        } else {
//          tmp = tmp->next;
//        }
//      }
//    }
//    return nullptr;
//  }

// void print_list(uint64_t print) {
//   VERSION *tmp = head;
//   while (tmp != nullptr) {
//     std::cout << "[" << print << "] xmin:" << tmp->t_min << std::endl;
//     tmp = tmp->next;
//   }
// }

std::bitset<64> MV_RecordList_Partial::get_readable_version(
    void* list_ptr, const uint64_t tid_self, char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    const ushort* col_idx, const ushort num_cols) {
  version_t* head = static_cast<version_t*>(list_ptr);

  // CREATE containment mask.
  std::bitset<64> done_mask;
  std::bitset<64> required_mask;
  std::vector<uint> return_col_offsets;

  if (num_cols > 0) {
    assert(num_cols <= 64 && "MAX columns supported: 64");
    done_mask.set();
    for (auto i = 0, offset = 0; i < num_cols; i++) {
      done_mask.reset(col_idx[i]);
      required_mask.set(col_idx[i]);

      offset += column_size_offset_pairs[col_idx[i]].first;
      return_col_offsets.push_back(offset);
    }
  } else {
    for (auto i = column_size_offset_pairs.size(); i < done_mask.size(); i++)
      done_mask.set(i);
  }
  required_mask = ~done_mask;

  assert(!(done_mask.all()) && "havent even started and its done?");

  // Traverse the list
  version_t* curr = head;
  size_t sanity_checker = 0;
  while (curr != nullptr && !(done_mask.all())) {
    sanity_checker++;

    if (sanity_checker > 250) {
      LOG(INFO) << "WOAHHH!: " << sanity_checker << " | " << done_mask;
    }

    if (global_conf::ConcurrencyControl::is_readable(curr->t_min, tid_self)) {
      // So this version is readable, now check if it contains what we need or
      // not.
      auto tmp = curr->attribute_mask & (~done_mask);

      if (tmp.any()) {
        // set the bits that what we have it.
        done_mask |= tmp;

        // so this version contains some of the required stuff.
        // Possible optimization: use intrinsic to find first set bit and
        // start i from there
        for (auto i = 0; i < tmp.size(); i++) {
          if (tmp[i] == false) {
            continue;
          }
          auto& col_s_pair = column_size_offset_pairs[i];

          // Offset of some column in the version itself
          auto version_col_offset = curr->get_offset(i);

          // Offset of column in the requested set of columns.
          // if requested all columns, the its columns cummulative offset.
          auto offset_idx_output =
              (num_cols == 0)
                  ? col_s_pair.second
                  : return_col_offsets
                        [(required_mask >> (required_mask.size() - i)).count()];

          memcpy((write_loc + offset_idx_output),
                 static_cast<char*>(curr->data) + version_col_offset,
                 col_s_pair.first);
        }
      }
    } else {
      curr = curr->next;
    }
  }

  return done_mask;
}




std::vector<MV_RecordList_Full::version_t*> MV_RecordList_Full::create_versions(
    global_conf::IndexVal* idx_ptr, void* list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols){

  size_t ver_size = 0;
  for (auto &attr_w : attribute_widths){
    ver_size += attr_w;
  }

  auto* ver =
      (MV_RecordList_Full::version_t*) deltaStore.insert_version(idx_ptr, ver_size, partition_id);
  assert(ver != nullptr && ver->data != nullptr);

  return {ver};

}


std::vector<MV_RecordList_Partial::version_t*> MV_RecordList_Partial::create_versions(
    global_conf::IndexVal* idx_ptr, void* list_ptr,
    std::vector<size_t>& attribute_widths, storage::DeltaStore& deltaStore,
    ushort partition_id, const ushort* col_idx, short num_cols){

  size_t ver_size = 0;

  if(__likely(num_cols > 0)){
    for(auto i = 0; i < num_cols; i++){
      ver_size += attribute_widths.at(col_idx[i]);
    }
  } else {
    for (auto &attr_w : attribute_widths){
      ver_size += attr_w;
    }
  }

  auto* ver =
      (MV_RecordList_Partial::version_t*) deltaStore.insert_version(idx_ptr, ver_size, partition_id);
  assert(ver != nullptr && ver->data != nullptr);

  size_t ver_rec_size = ver->create_partial_mask(attribute_widths,col_idx, num_cols);
  assert(ver_size == ver_rec_size && "Sanity check failed");

  return {ver};

}

}  // namespace storage::mv
