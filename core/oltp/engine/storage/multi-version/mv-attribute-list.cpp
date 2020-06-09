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

namespace storage::mv {

std::bitset<64> attributeList_single::VERSION_CHAIN::get_readable_version(
    const uint64_t tid_self, char *write_loc,
    const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
    const ushort *col_idx, const ushort num_cols) {
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
  VERSION *curr = this->head;
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
          auto &col_s_pair = column_size_offset_pairs[i];

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
                 static_cast<char *>(curr->data) + version_col_offset,
                 col_s_pair.first);
        }
      }
    } else {
      curr = curr->next;
    }
  }

  return done_mask;
}

}  // namespace storage::mv
