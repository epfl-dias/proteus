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

#include <bitset>
#include <storage/column_store.hpp>
#include <utility>

#include "glo.hpp"

namespace storage::mv {

class attributeList {
 public:
  class VERSION;
  class VERSION_CHAIN;
  static constexpr bool isAttributeLevelMV = true;
};

class attributeList_multi : public attributeList {
 public:
  class VERSION {
   public:
    uint64_t t_min;
    uint64_t t_max;
    void *data;
    VERSION *next;
    VERSION(uint64_t t_min, uint64_t t_max, void *data)
        : t_min(t_min), t_max(t_max), data(data), next(nullptr) {}
  };

  class VERSION_CHAIN {
   public:
    VERSION_CHAIN() { head = nullptr; }
    explicit VERSION_CHAIN(VERSION *head) : head(head) {}

    inline void insert(VERSION *val) {
      val->next = head;
      head = val;
    }

    bool get_readable_version(uint64_t tid_self, char *write_loc,
                              const size_t elem_size);

   private:
    VERSION *head;
    friend class storage::DeltaStore;
  };
};

/* Class: attributeList_single
 * Description: single: one list per-relation.
 * */

class attributeList_single : public attributeList {
 public:
  class VERSION {
   public:
    uint64_t t_min;
    uint64_t t_max;
    std::bitset<64> attribute_mask;
    std::vector<size_t> offsets;
    void *data;
    VERSION *next;
    VERSION(uint64_t t_min, uint64_t t_max, void *data)
        : t_min(t_min), t_max(t_max), data(data), next(nullptr) {
      attribute_mask.set();
    }
    VERSION(uint64_t t_min, uint64_t t_max, void *data,
            std::bitset<64> attribute_mask)
        : t_min(t_min),
          t_max(t_max),
          data(data),
          next(nullptr),
          attribute_mask(attribute_mask) {}

    inline void set_attr_mask(std::bitset<64> mask) { attribute_mask = mask; }
    inline void set_offsets(std::vector<size_t> col_offsets) {
      this->offsets = std::move(col_offsets);
    }

    inline size_t get_offset(size_t col_idx) {
      auto idx_in_ver =
          (attribute_mask >> (attribute_mask.size() - col_idx)).count();
      return offsets[idx_in_ver];
    }
  };

  class VERSION_CHAIN {
   public:
    VERSION_CHAIN() { head = nullptr; }
    explicit VERSION_CHAIN(VERSION *head) : head(head) {}

    inline void insert(VERSION *val) {
      val->next = head;
      head = val;
    }

    void get_readable_version(uint64_t tid_self, char *write_loc,
                              uint rec_size);

    std::bitset<64> get_readable_version(
        uint64_t tid_self, char *write_loc,
        const std::vector<std::pair<size_t, size_t>> &column_size_offset_pairs,
        const ushort *col_idx = nullptr, ushort num_cols = 0);

   private:
    VERSION *head;
    void *get_readable_version(uint64_t tid_self) const;

    friend class storage::DeltaStore;

  };  // __attribute__((aligned(64)));
};
}  // namespace storage::mv

#endif  // PROTEUS_MV_ATTRIBUTE_LIST_HPP
