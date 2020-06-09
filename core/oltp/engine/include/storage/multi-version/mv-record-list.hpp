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

#include <cassert>
#include <iostream>
#include <storage/column_store.hpp>

namespace storage::mv {

/* Class: recordList
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

class recordList {
 public:
  static constexpr bool isAttributeLevelMV = false;

  class VERSION {
   public:
    const uint64_t t_min;
    const uint64_t t_max;
    void *data;
    VERSION *next;
    VERSION(uint64_t t_min, uint64_t t_max, void *data)
        : t_min(t_min), t_max(t_max), data(data), next(nullptr) {}

    inline void set_attr_mask(std::bitset<64> mask) {}

    [[noreturn]] inline int64_t get_offset(size_t col_idx) {
      throw std::runtime_error("record-list doesnt need offsets");
    }

    [[noreturn]] inline void set_offsets(std::vector<size_t> col_offsets) {
      throw std::runtime_error("record-list doesnt need offsets");
    }

    friend class VERSION_CHAIN;
  };

  class VERSION_CHAIN {
   public:
    VERSION_CHAIN() { head = nullptr; }

    inline void insert(VERSION *val) {
      val->next = head;
      head = val;
    }
    void get_readable_version(uint64_t tid_self, char *write_loc,
                              uint rec_size);

    std::bitset<1> get_readable_version(
        uint64_t tid_self, char *write_loc,
        const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
        const ushort *col_idx = nullptr, ushort num_cols = 0);

    // void print_list(uint64_t print) {
    //   VERSION *tmp = head;
    //   while (tmp != nullptr) {
    //     std::cout << "[" << print << "] xmin:" << tmp->t_min << std::endl;
    //     tmp = tmp->next;
    //   }
    // }

   private:
    VERSION *head;

    [[nodiscard]] void *get_readable_version(uint64_t tid_self) const;

    friend class storage::DeltaStore;
  };
};
}  // namespace storage::mv

#endif  // PROTEUS_MV_RECORD_LIST_HPP
