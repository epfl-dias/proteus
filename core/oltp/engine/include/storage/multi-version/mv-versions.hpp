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

#ifndef PROTEUS_MV_VERSIONS_HPP
#define PROTEUS_MV_VERSIONS_HPP

#include <bitset>
#include <iostream>
#include <vector>

#include "delta_storage.hpp"

namespace storage::mv {

class Version {
 public:
  const uint64_t t_min;
  const uint64_t t_max;
  void *data;

  Version(uint64_t t_min, uint64_t t_max, void *data)
      : t_min(t_min), t_max(t_max), data(data) {}

  virtual void set_attr_mask(std::bitset<64> mask) = 0;
  virtual size_t get_offset(size_t col_idx) = 0;
  virtual void set_offsets(std::vector<size_t> col_offsets) = 0;
  virtual size_t create_partial_mask(std::vector<size_t> &attribute_widths,
                                     const ushort *col_idx, short num_cols) = 0;

  virtual ~Version() = default;
};


class VersionSingle : public Version {
 public:
  VersionSingle *next;

  VersionSingle(uint64_t t_min, uint64_t t_max, void *data)
      : Version(t_min, t_max, data), next(nullptr) {}

  inline void set_attr_mask(std::bitset<64> mask) override {}

  [[noreturn]] inline size_t get_offset(size_t col_idx) override {
    throw std::runtime_error("record-list doesnt need offsets");
  }

  [[noreturn]] inline void set_offsets(
      std::vector<size_t> col_offsets) override {
    throw std::runtime_error("record-list doesnt need offsets");
  }

  size_t create_partial_mask(std::vector<size_t> &attribute_widths,
                             const ushort *col_idx, short num_cols) override {
    return 0;
  }
};

class VersionMultiAttr : public Version {
 public:
  std::bitset<64> attribute_mask;
  std::vector<size_t> offsets;  // FIXME: THIS IS A MEMORY LEAKK!!
  VersionMultiAttr *next;

  VersionMultiAttr(uint64_t t_min, uint64_t t_max, void *data)
      : Version(t_min, t_max, data), next(nullptr) {
    attribute_mask.set();
  }
  VersionMultiAttr(uint64_t t_min, uint64_t t_max, void *data,
                   std::bitset<64> attribute_mask)
      : Version(t_min, t_max, data),
        next(nullptr),
        attribute_mask(attribute_mask) {}

  VersionMultiAttr(uint64_t t_min, uint64_t t_max, void *data,
                   std::bitset<64> attribute_mask,
                   std::vector<size_t> &col_offsets)
      : Version(t_min, t_max, data),
        next(nullptr),
        attribute_mask(attribute_mask),
        offsets(std::move(col_offsets)) {}

  inline void set_attr_mask(std::bitset<64> mask) override {
    attribute_mask = mask;
  }
  inline void set_offsets(std::vector<size_t> col_offsets) override {
    this->offsets = std::move(col_offsets);
  }

  inline size_t get_offset(size_t col_idx) override {
    auto idx_in_ver =
        (attribute_mask >> (attribute_mask.size() - col_idx)).count();
    return offsets[idx_in_ver];
  }

  void create_partial_mask(std::vector<size_t> &ver_offsets,
                           std::bitset<64> attr_mask){
    this->offsets = std::move(ver_offsets);
    this->attribute_mask = attr_mask;
  }

  size_t create_partial_mask(std::vector<size_t> &attribute_widths,
                             const ushort *col_idx, short num_cols) override {
    size_t ver_rec_size = 0;
    offsets.reserve(num_cols > 0 ? num_cols : attribute_widths.size());
    offsets.clear();

    if (__likely(attribute_widths.size() <= 64)) {
      if (__unlikely(num_cols == 0 || col_idx == nullptr)) {
        this->attribute_mask.set();
        for (auto &attr : attribute_widths) {
          this->offsets.push_back(ver_rec_size);
          ver_rec_size += attr;
        }
      } else {
        this->attribute_mask.reset();
        for (ushort i = 0; i < num_cols; i++) {
          attribute_mask.set(col_idx[i]);
          this->offsets.push_back(ver_rec_size);
          ver_rec_size += attribute_widths.at(col_idx[i]);
        }
      }

    } else {
      assert(false && "for now only max 64 columns supported.");
    }
    // LOG(INFO) << "CREATED MASK: " << this->attribute_mask;
    return ver_rec_size;
  }

  static size_t get_partial_mask_size(std::vector<size_t> &attribute_widths,
                                      std::vector<size_t> &ver_offsets,
                                      std::bitset<64> &attr_mask,
                             const ushort *col_idx, short num_cols) {
    size_t ver_rec_size = 0;
    ver_offsets.reserve(num_cols > 0 ? num_cols : attribute_widths.size());

    if (__likely(attribute_widths.size() <= 64)) {
      if (__unlikely(col_idx == nullptr || num_cols == 0)) {
        attr_mask.set();
        for (auto &attr : attribute_widths) {
          ver_offsets.push_back(ver_rec_size);
          ver_rec_size += attr;
        }
      } else {
        attr_mask.reset();
        for (ushort i = 0; i < num_cols; i++) {
          attr_mask.set(col_idx[i]);
          ver_offsets.push_back(ver_rec_size);
          ver_rec_size += attribute_widths.at(col_idx[i]);
        }
      }

    } else {
      assert(false && "for now only max 64 columns supported.");
    }
    return ver_rec_size;
  }
};

template <typename T>
class VersionChain {
 public:
  //  auto get_readable_version(
  //      uint64_t xid, char *write_loc,
  //      const std::vector<std::pair<size_t, size_t>>
  //      &column_size_offset_pairs, const ushort *col_idx = nullptr, ushort
  //      num_cols = 0) {
  //    return T::get_readable_version(head, xid, write_loc,
  //                                   column_size_offset_pairs, col_idx,
  //                                   num_cols);
  //  }

  VersionChain() { head = nullptr; }

  inline void insert(typename T::version_t *val) {
    val->next = head;
    head = val;
  }

  inline void reset_head(typename T::version_t *val) { head = val; }

  auto get_readable_version(uint64_t xid) {
    typename T::version_t *tmp = nullptr;
    {
      tmp = head;
      while (tmp != nullptr) {
        // if (CC_MV2PL::is_readable(tmp->t_min, tmp->t_max, tid_self)) {
        if (global_conf::ConcurrencyControl::is_readable(tmp->t_min, xid)) {
          return tmp;
        } else {
          tmp = tmp->next;
        }
      }
    }
    return tmp;
  }

  typename T::version_t *head;
  uint64_t last_updated_tmin;

  friend class storage::DeltaStore;
  friend T;
};

template <class versionChain>
class MVattributeListCol {
 private:
  struct col_mv_list {
    size_t delta_tag;
    versionChain *versions;
  };

  struct col_mv_list *attr_lists;

 public:
  static size_t getSize(size_t num_attributes) {
    return sizeof(MVattributeListCol) + (sizeof(col_mv_list) * num_attributes);
  }

  static void create(size_t delta_tag, void *ptr, size_t num_attr) {
    auto *tmp = new (ptr) MVattributeListCol();
    tmp->attr_lists =
        (col_mv_list *)(((char *)ptr) + sizeof(MVattributeListCol));
    for (auto i = 0; i < num_attr; i++) {
      tmp->attr_lists[i].versions = nullptr;
      tmp->attr_lists[i].delta_tag = delta_tag;
    }
  }

 private:
  explicit MVattributeListCol() : attr_lists(nullptr) {}

  friend class MV_DAG;
  friend class MV_attributeList;
};



//class VersionDAG {
//  const uint64_t *t_min;
//  const uint64_t *t_max;
//  std::bitset<64> attribute_mask;
//  const size_t *ver_offsets;
//  const void* data;
//  VersionDAG** next;
//};
//
//template <typename T = VersionDAG>
//class VersionChainDAG {
//  VersionChainDAG() { head = nullptr; }
//
//  inline void insert(typename T::version_t *val) {
//    val->next = head;
//    head = val;
//  }
//
//  inline void reset_head(typename T::version_t *val) { head = val; }
//
//
//
//
//  typename T::version_t *head;
//
//  // last_updated_tmin is used as transient timestamp instead of keeping
//  // a timestamp with each attribute w/ main record.
//  uint64_t last_updated_tmin;
//
//  // maybe keep the ts of latest version? duplicate info but can reduce extra
//  // access for weighted edge?
//  //uint64_t latest_version_tmin;
//
//  // chain-length? for traversal starting decision?
//  uint32_t chain_length;
//
//  friend class storage::DeltaStore;
//  friend T;
//};

}  // namespace storage::mv

#endif /* PROTEUS_MV_VERSIONS_HPP */
