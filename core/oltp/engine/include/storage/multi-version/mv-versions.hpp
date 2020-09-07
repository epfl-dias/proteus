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

class Version{

 public:
  const uint64_t t_min;
  const uint64_t t_max;
  void *data;

  Version(uint64_t t_min, uint64_t t_max, void *data): t_min(t_min), t_max(t_max), data(data){}

  virtual void set_attr_mask(std::bitset<64> mask) = 0;
  virtual size_t get_offset(size_t col_idx) = 0;
  virtual void set_offsets(std::vector<size_t> col_offsets) = 0;
  virtual size_t create_partial_mask(std::vector<size_t>& attribute_widths,
                             const ushort* col_idx, short num_cols) = 0;

  virtual ~Version() = default;
};

class VersionSingle : public Version{
 public:
  VersionSingle *next;

  VersionSingle(uint64_t t_min, uint64_t t_max, void *data)
      : Version(t_min, t_max, data), next(nullptr) {}

  inline void set_attr_mask(std::bitset<64> mask) override {}

  [[noreturn]] inline size_t get_offset(size_t col_idx) override {
    throw std::runtime_error("record-list doesnt need offsets");
  }

  [[noreturn]] inline void set_offsets(std::vector<size_t> col_offsets) override {
    throw std::runtime_error("record-list doesnt need offsets");
  }

  size_t create_partial_mask(std::vector<size_t>& attribute_widths,
                             const ushort* col_idx, short num_cols) override{
   return 0;
  }
};

class VersionMultiAttr : public Version{
 public:
  std::bitset<64> attribute_mask;
  std::vector<size_t> offsets;
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

  inline void set_attr_mask(std::bitset<64> mask) override { attribute_mask = mask; }
  inline void set_offsets(std::vector<size_t> col_offsets) override {
    this->offsets = std::move(col_offsets);
  }

  inline size_t get_offset(size_t col_idx) override {
    auto idx_in_ver =
        (attribute_mask >> (attribute_mask.size() - col_idx)).count();
    return offsets[idx_in_ver];
  }


  size_t create_partial_mask(std::vector<size_t>& attribute_widths,
                                             const ushort* col_idx, short num_cols) override {
    size_t ver_rec_size = 0;
    offsets.resize(num_cols > 0 ? num_cols : attribute_widths.size());
    offsets.clear();

    if (__likely(attribute_widths.size() <= 64)) {
      if (__unlikely(num_cols == 0 || col_idx == nullptr)) {
        this->attribute_mask.set();
        for(auto& attr: attribute_widths){
          this->offsets.push_back(ver_rec_size);
          ver_rec_size += attr;
        }
      } else {
        this->attribute_mask.reset();
        for (ushort i = 0; i < num_cols; i++) {
          attribute_mask.set(i);
          this->offsets.push_back(ver_rec_size);
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
 private:
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

  friend class storage::DeltaStore;
  friend T;
};

template <class versionChain>
class MVattributeListCol {
 private:
  size_t *delta_tags;
  versionChain **version_lists;

 public:
  static size_t getSize(size_t num_attributes) {
    return sizeof(MVattributeListCol) +
           (sizeof(versionChain) * num_attributes) +
           (sizeof(size_t) * num_attributes);
  }

  static void create(size_t delta_tag, void *ptr, size_t num_attr) {
    auto *tmp = new (ptr) MVattributeListCol();
    tmp->version_lists =
        (versionChain **)(((char *)ptr) + sizeof(MVattributeListCol));
    tmp->delta_tags = (size_t *)(((char *)ptr) + sizeof(MVattributeListCol) +
                                 (sizeof(versionChain) * num_attr));

    for (auto i = 0; i < num_attr; i++) {
      tmp->version_lists[i] = nullptr;
      tmp->delta_tags[i] = delta_tag;
    }
  }

  [[nodiscard]] inline size_t getTag(uint idx) const { return delta_tags[idx]; }

 private:
  explicit MVattributeListCol() : delta_tags(nullptr), version_lists(nullptr) {}

  friend class MV_DAG;
  friend class MV_attributeList;
};

}  // namespace storage::mv

#endif /* PROTEUS_MV_VERSIONS_HPP */
