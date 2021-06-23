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

#include "oltp/common/constants.hpp"
#include "oltp/storage/multi-version/delta-memory-ptr.hpp"
#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage::mv {

class Version {
 public:
  const xid_t t_min;
  [[maybe_unused]] const xid_t t_max;
  void *data;

  Version(xid_t t_min, xid_t t_max, void *data)
      : t_min(t_min), t_max(t_max), data(data) {}

  [[maybe_unused]] virtual void set_attr_mask(std::bitset<64> mask) = 0;
  virtual uint16_t get_offset(uint16_t col_idx) = 0;
  //  virtual size_t create_partial_mask(std::vector<uint16_t>
  //  &attribute_widths,
  //                                     const ushort *col_idx, short num_cols)
  //                                     = 0;

  virtual ~Version() = default;
};

class VersionSingle : public Version {
 public:
  // VersionSingle *next;
  TaggedDeltaDataPtr<VersionSingle> next;

  [[maybe_unused]] VersionSingle(xid_t t_min, xid_t t_max, void *data)
      : Version(t_min, t_max, data), next(0) {}

  inline void set_attr_mask(std::bitset<64> mask) override {}

  [[noreturn]] inline uint16_t get_offset(uint16_t col_idx) override {
    throw std::runtime_error("record-list doesnt need offsets");
  }
};

class VersionMultiAttr : public Version {
 public:
  std::bitset<64> attribute_mask;
  uint16_t *attribute_offsets;
  TaggedDeltaDataPtr<VersionMultiAttr> next;

  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data)
      : Version(t_min, t_max, data), next(0) {
    attribute_mask.set();
  }
  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data,
                   std::bitset<64> attribute_mask)
      : Version(t_min, t_max, data), next(0), attribute_mask(attribute_mask) {}

  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data,
                   std::bitset<64> attribute_mask, uint16_t *attr_offsets)
      : Version(t_min, t_max, data),
        next(0),
        attribute_mask(attribute_mask),
        attribute_offsets(attr_offsets) {}

  inline void set_attr_mask(std::bitset<64> mask) override {
    attribute_mask = mask;
  }

  inline void set_attr_offsets(uint16_t *attr_offsets) {
    attribute_offsets = attr_offsets;
  }

  inline uint16_t get_offset(uint16_t col_idx) override {
    auto idx_in_ver =
        (attribute_mask >> (attribute_mask.size() - col_idx)).count();
    return attribute_offsets[idx_in_ver];
  }

  void create_partial_mask(uint16_t *attr_offsets, std::bitset<64> attr_mask) {
    this->attribute_offsets = attr_offsets;
    this->attribute_mask = attr_mask;
  }

  static size_t get_partial_mask_size(std::vector<uint16_t> &attribute_widths,
                                      std::vector<uint16_t> &ver_offsets,
                                      std::bitset<64> &attr_mask,
                                      const column_id_t *col_idx,
                                      short num_cols) {
    size_t ver_rec_size = 0;

    assert(attribute_widths.size() <= 64 && "max 64-columns supported");
    if (__unlikely(col_idx == nullptr || num_cols == 0)) {
      attr_mask.set();
      for (auto &attr : attribute_widths) {
        ver_offsets.push_back(ver_rec_size);
        ver_rec_size += attr;
      }
    } else {
      attr_mask.reset();
      for (auto i = 0; i < num_cols; i++) {
        attr_mask.set(col_idx[i]);
        ver_offsets.push_back(ver_rec_size);
        ver_rec_size += attribute_widths.at(col_idx[i]);
      }
    }

    return ver_rec_size;
  }
};

// Should be used when the head of the list contains information
// Example: shouldn't be used for single-list as each version
// can point to next directly.
template <typename T>
class VersionChain {
 public:
  VersionChain() : last_updated_tmin(0), head(0) {}

  inline void insert(TaggedDeltaDataPtr<typename T::version_t> val) {
    assert(val.ptr() != nullptr);
    val.ptr()->next = head;
    head = val;
  }

  inline void reset_head(TaggedDeltaDataPtr<typename T::version_t> val) {
    head = val;
  }

  static typename T::version_t *get_readable_ver(
      TaggedDeltaDataPtr<typename T::version_t> &head_ref,
      const txn::TxnTs &txTs) {
    TaggedDeltaDataPtr<typename T::version_t> tmp = head_ref;

    while (tmp.is_valid()) {
      auto ptr = tmp.ptr();
      if (global_conf::ConcurrencyControl::is_readable(ptr->t_min, txTs)) {
        return ptr;
      } else {
        tmp = ptr->next;
      }
    }
    assert(false && "if asked then why there is no version?");
    return nullptr;
  }

  typename T::version_t *get_readable_version(const txn::TxnTs &txTs) {
    return get_readable_ver(this->head, txTs);
  }

  TaggedDeltaDataPtr<typename T::version_t> head{};
  xid_t last_updated_tmin{};
  xid_t last_updated_tmax{};

  friend class storage::DeltaStore;
  friend T;
};

class MVattributeListCol {
 private:
  DeltaList *version_list;

 public:
  static size_t getSize(size_t num_attributes) {
    return sizeof(MVattributeListCol) + (sizeof(DeltaList) * num_attributes);
  }

  static void create(void *ptr, size_t num_attr) {
    auto *tmp = new (ptr) MVattributeListCol();
    tmp->version_list =
        (DeltaList *)(((char *)ptr) + sizeof(MVattributeListCol));

    // FIXME: is the following necessary?
    for (auto i = 0; i < num_attr; i++) {
      // tmp->version_list[i] = (DeltaList*) new (tmp->version_list + i)
      // DeltaList();
      tmp->version_list[i].updateVal(0);
    }
  }

 private:
  explicit MVattributeListCol() : version_list(nullptr) {}

  friend class MV_DAG;
  friend class MV_attributeList;
};

// class VersionDAG {
//  const uint64_t *t_min;
//  const uint64_t *t_max;
//  std::bitset<64> attribute_mask;
//  const size_t *ver_offsets;
//  const void* data;
//  VersionDAG** next;
//};
//
// template <typename T = VersionDAG>
// class VersionChainDAG {
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
