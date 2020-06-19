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

#ifndef PROTEUS_MV_HPP
#define PROTEUS_MV_HPP

// MV Types
#include "storage/multi-version/mv-attribute-dag.hpp"
#include "storage/multi-version/mv-attribute-list.hpp"
#include "storage/multi-version/mv-record-list.hpp"

namespace storage::mv {

using mv_type = recordList;
// using mv_type = attributeList_single;

using mv_version_chain = mv_type::VERSION_CHAIN;
using mv_version = mv_type::VERSION;

// use a factory to return specific type of class.

// maybe: __attribute__((packed))
// ALERT: following may cause memory overwrite, be careful!

class perAttributeMVcol {
 private:
  size_t delta_tag;
  mv_version_chain** version_lists;

 public:
  static size_t getSize(size_t num_attributes) {
    return sizeof(perAttributeMVcol) +
           (sizeof(mv_version_chain) * num_attributes);
  }

  static void create(size_t delta_tag, void* ptr, size_t num_attr) {
    auto* tmp = new (ptr) perAttributeMVcol(delta_tag);
    tmp->version_lists =
        (mv_version_chain**)(((char*)ptr) + sizeof(perAttributeMVcol));
    for (auto i = 0; i < num_attr; i++) {
      tmp->version_lists[i] = nullptr;
    }
  }

  [[nodiscard]] size_t getTag() const { return delta_tag; }

 private:
  explicit perAttributeMVcol(size_t delta_tag)
      : delta_tag(delta_tag), version_lists(nullptr) {}
};

}  // namespace storage::mv

#endif  // PROTEUS_MV_HPP
