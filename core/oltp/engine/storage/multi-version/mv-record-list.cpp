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

void* recordList::VERSION_CHAIN::get_readable_version(uint64_t tid_self) const {
  VERSION* tmp = nullptr;
  {
    tmp = this->head;
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
void recordList::VERSION_CHAIN::get_readable_version(uint64_t tid_self,
                                                     char* write_loc,
                                                     uint rec_size) {
  char* version = (char*)this->get_readable_version(tid_self);
  assert(version != nullptr);
  assert(write_loc != nullptr);
  memcpy(write_loc, version, rec_size);
}

std::bitset<1> recordList::VERSION_CHAIN::get_readable_version(
    const uint64_t tid_self, char* write_loc,
    const std::vector<std::pair<size_t, size_t>>& column_size_offset_pairs,
    const ushort* col_idx, const ushort num_cols) {
  static thread_local std::bitset<1> ret_bitmask("1");

  char* version = (char*)this->get_readable_version(tid_self);
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
}  // namespace storage::mv
