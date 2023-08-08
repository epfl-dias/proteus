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
#include "oltp/storage/storage-utils.hpp"

namespace storage::mv {

class __attribute__((packed)) Version {
 public:
  Version(Version &&) = delete;
  Version &operator=(Version &&) = delete;
  Version(const Version &) = delete;
  Version &operator=(const Version &) = delete;

 public:
  const xid_t t_min{};
  [[maybe_unused]] const xid_t t_max{};
  void *data{};
  const size_t size;

  Version(xid_t t_min, xid_t t_max, void *data, size_t sz)
      : t_min(t_min), t_max(t_max), data(data), size(sz) {}

  [[maybe_unused]] virtual void set_attr_mask(std::bitset<64> mask) = 0;
  virtual uint16_t get_offset(uint16_t col_idx) = 0;
  //  virtual size_t create_partial_mask(std::vector<uint16_t>
  //  &attribute_widths,
  //                                     const ushort *col_idx, short num_cols)
  //                                     = 0;

  virtual ~Version() = default;
};

class __attribute__((packed)) VersionSingle : public Version {
 public:
  VersionSingle(VersionSingle &&) = delete;
  VersionSingle &operator=(VersionSingle &&) = delete;
  VersionSingle(const VersionSingle &) = delete;
  VersionSingle &operator=(const VersionSingle &) = delete;

 public:
  // VersionSingle *next;
  TaggedDeltaDataPtr<VersionSingle> next{};

  [[maybe_unused]] VersionSingle(xid_t t_min, xid_t t_max, void *data,
                                 size_t sz)
      : Version(t_min, t_max, data, sz), next(0) {}

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

  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data, size_t sz)
      : Version(t_min, t_max, data, sz), next(0) {
    attribute_mask.set();
  }
  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data,
                   std::bitset<64> attribute_mask, size_t sz)
      : Version(t_min, t_max, data, sz),
        next(0),
        attribute_mask(attribute_mask) {}

  VersionMultiAttr(xid_t t_min, xid_t t_max, void *data,
                   std::bitset<64> attribute_mask, uint16_t *attr_offsets,
                   size_t sz)
      : Version(t_min, t_max, data, sz),
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

  static size_t get_partial_mask_size(
      const std::vector<uint16_t> &attribute_widths,
      std::vector<uint16_t> &ver_offsets, std::bitset<64> &attr_mask,
      const column_id_t *col_idx, short num_cols) {
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

  // static typename T::version_t *get_readable_ver(
  static inline const TaggedDeltaDataPtr<typename T::version_t> *
  get_readable_ver(const txn::TxnTs &txTs,
                   const TaggedDeltaDataPtr<typename T::version_t> &head_ref) {
    auto tmp = &head_ref;
    while (tmp->isValid()) {
      auto ptr = tmp->typePtr();

      if (ptr != nullptr &&
          global_conf::ConcurrencyControl::is_readable(ptr->t_min, txTs)) {
        return tmp;
      } else {
        tmp = &(ptr->next);
      }
    }
    return nullptr;
  }

  static inline bool search_in_chain(
      const txn::TxnTs &txTs,
      const TaggedDeltaDataPtr<typename T::version_t> &begin,
      TaggedDeltaDataPtr<typename T::version_t> *&result) {
    auto tmp = &begin;

    // TODO: do not continue search indefinitely,
    //  we know the list itself is N2O order, so if you encounter older version
    //  than you?

    while (tmp->isValid()) {
      auto ptr = tmp->typePtr();
      if (ptr != nullptr &&
          global_conf::ConcurrencyControl::is_readable(ptr->t_min, txTs)) {
        result = const_cast<TaggedDeltaDataPtr<typename T::version_t> *>(tmp);
        return true;
      } else {
        auto nextDeltaIdx = ptr->next.get_delta_idx();
        if (tmp->get_delta_idx() == nextDeltaIdx) {
          tmp = &(ptr->next);
        } else {
          // break and return if the chain crosses the instance.
          // can try to continue until valid but risky.
          return false;
        }
      }
    }
    return false;
  }

  static inline TaggedDeltaDataPtr<typename T::version_t>
  get_version_with_consolidation(
      const txn::TxnTs &txTs,
      const TaggedDeltaDataPtr<typename T::version_t> &head,
      row_uuid_t row_uuid) {
    TaggedDeltaDataPtr<typename T::version_t> *tmpPtr;
    TaggedDeltaDataPtr<typename T::version_t> ret;

    bool found = search_in_chain(txTs, head, tmpPtr);

    if (found) {
      return std::move(
          const_cast<TaggedDeltaDataPtr<typename T::version_t> *>(tmpPtr)
              ->copy());
    }

    // cross-instance-search
    int delta_idx = head.get_delta_idx();
    int sanity_ctr = 0;
    auto pid = storage::StorageUtils::get_pid_from_rowUuid(row_uuid);

    while (!found) {
      if ((--delta_idx) < 0) {
        delta_idx = global_conf::num_delta_storages - 1;
      }
      sanity_ctr++;
      // LOG(INFO) << "Looking into: " << delta_idx;
      LOG_IF(FATAL, sanity_ctr > global_conf::num_delta_storages + 2)
          << "Infinite loop in search versions: "
          << storage::StorageUtils::get_offset(
                 storage::StorageUtils::get_rowId_from_rowUuid(row_uuid))
          << " | tx: " << txTs.txn_start_time;

      try {
        auto head_ptr_list =
            DeltaDataPtr::getDeltaByIdx(delta_idx)->getConsolidateHT(pid).find(
                row_uuid);
        // NOTE: head_ptr_list is in O2N order, therefore reverse iterator
        // Ranges not in clang12
        //        for(auto it = head_ptr_list.rbegin() ; it !=
        //        head_ptr_list.rend(); it++){
        for (auto &p : head_ptr_list) {
          auto &tmpTaggedPtr =
              reinterpret_cast<TaggedDeltaDataPtr<typename T::version_t> &>(p);
          found = search_in_chain(txTs, tmpTaggedPtr, tmpPtr);
          if (found) {
            //            LOG(INFO) << "Found in: " << delta_idx << " | " <<
            //            ret->isValid() << " && " <<
            //            reinterpret_cast<uintptr_t>(ret->typePtr())
            //                      << " && " <<
            //                      reinterpret_cast<uintptr_t>(ret)  ;
            return std::move(tmpPtr->copy());
          }
        }

      } catch (std::out_of_range &) {
        // if key not found in the cuckooMap
      }
    }

    assert(false);
  }

  typename T::version_t *get_readable_version(const txn::TxnTs &txTs) {
    return get_readable_ver(this->head, txTs);
  }

  TaggedDeltaDataPtr<typename T::version_t> head{};
  xid_t last_updated_tmin{};
  xid_t last_updated_tmax{};

  friend storage::DeltaStore;
  friend T;
};

class MVattributeListCol {
 private:
  DeltaPtr *version_list;

 public:
  static size_t getSize(size_t num_attributes) {
    return sizeof(MVattributeListCol) + (sizeof(DeltaPtr) * num_attributes) +
           alignof(DeltaPtr);
  }

  static void create(void *ptr, size_t num_attr) {
    auto *tmp = new (ptr) MVattributeListCol();
    tmp->version_list =
        (DeltaPtr *)(((char *)ptr) + sizeof(MVattributeListCol));

    tmp->version_list =
        new ((((char *)ptr) + sizeof(MVattributeListCol))) DeltaPtr[num_attr];
    // FIXME: is the following necessary?
    //    for (auto i = 0; i < num_attr; i++) {
    //      (DeltaPtr*) new (tmp->version_list + i) DeltaPtr(0);
    //     // tmp->version_list[i].reset();
    //    }
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
