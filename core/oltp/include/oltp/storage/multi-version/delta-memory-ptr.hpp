/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_DELTA_MEMORY_PTR_HPP
#define PROTEUS_DELTA_MEMORY_PTR_HPP

#include <cassert>

#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage {

class DeltaPtr;

template <class T>
class DeltaMemoryPtr {
 public:
  [[nodiscard]] inline char *ptr() { return static_cast<T *>(this)->get_ptr(); }

  [[nodiscard]] inline char *ptr() const {
    return static_cast<T *>(const_cast<DeltaMemoryPtr *>(this))->get_ptr();
  }
  inline void release() { static_cast<T *>(this)->release_impl(); }

  //  [[nodiscard]] constexpr inline uint8_t get_delta_idx() const {
  //    return static_cast<T *>(this)->get_delta_idx();
  //  }
  //  void saveInstanceCrossingPtr(vid_t vid) {
  //    static_cast<T *>(this)->saveInstanceCrossingPtr(vid);
  //  }

  bool operator==(const DeltaMemoryPtr &other) const {
    return (other._val == this->_val);
  }
  bool operator!=(const DeltaMemoryPtr &other) const {
    return (other._val != this->_val);
  }

 protected:
  explicit DeltaMemoryPtr(uintptr_t val) : _val(val) {}

  uintptr_t _val = static_cast<uintptr_t>(0);

  friend class DeltaPtr;

  template <typename>
  friend class TaggedDeltaDataPtr;
};

class ClassicPtrWrapper : public DeltaMemoryPtr<ClassicPtrWrapper> {
 public:
  [[nodiscard]] inline bool isValid() const {
    return (reinterpret_cast<void *>(get_ptrOffset()) != nullptr);
  }
  ClassicPtrWrapper(const ClassicPtrWrapper &) = delete;
  ClassicPtrWrapper &operator=(const ClassicPtrWrapper &) = delete;

 private:
  [[nodiscard]] inline uintptr_t get_ptrOffset() const {
    return ((this->_val) & 0x00FFFFFFFFFFFFFF);
  }

  inline partition_id_t getPid() { return ((this->_val) >> 60u); }

 protected:
  explicit ClassicPtrWrapper(uintptr_t val) : DeltaMemoryPtr(val) {}
  explicit ClassicPtrWrapper(uintptr_t val, partition_id_t pid)
      : DeltaMemoryPtr(val) {
    val = val & 0x00FFFFFFFFFFFFFF;
    val |= ((size_t)pid) << 60u;
    this->_val = val;
  }

  ClassicPtrWrapper(ClassicPtrWrapper &&) noexcept = default;
  ClassicPtrWrapper &operator=(ClassicPtrWrapper &&) noexcept = default;

  [[nodiscard]] inline char *get_ptr() const {
    if (_val == 0) {
      return nullptr;
    }
    return static_cast<char *>(reinterpret_cast<void *>(get_ptrOffset()));
  }

  inline void release_impl() {
    assert(this->_val != 0);
    deltaStore->release(*this);
    this->_val = 0;
  }

 public:
  [[nodiscard]] constexpr inline uint8_t get_delta_idx() const { return 0; }

  [[noreturn]] void saveInstanceCrossingPtr(vid_t vid) { assert(false); }

 private:
  static DeltaStoreMalloc *deltaStore;

  friend class DeltaStoreMalloc;
  friend class DeltaMemoryPtr<ClassicPtrWrapper>;
};

class DeltaDataPtr : public DeltaMemoryPtr<DeltaDataPtr> {
  static_assert(sizeof(uintptr_t) == 8,
                "Invalid size of uintptr_8, expected 8B");

 public:
  //  4 bit     |  4 bit       | 20 bit  | 36-bit ( 64GB addressable)
  //  delta-idx | partition-id | tag     | offset-in-partition

  [[maybe_unused]] static constexpr size_t offset_bits = 36u;
  [[maybe_unused]] static constexpr size_t pid_bits = 4u;
  [[maybe_unused]] static constexpr size_t delta_id_bits = 4u;
  [[maybe_unused]] static constexpr size_t tag_bits = 20u;

 public:
  explicit DeltaDataPtr(uintptr_t val) : DeltaMemoryPtr(val) {}

  // DeltaDataPtr(DeltaDataPtr &other) : DeltaDataPtr(other._val) {}

  explicit DeltaDataPtr(const char *data_ptr, uint32_t tag,
                        delta_id_t delta_idx, partition_id_t pid)
      : DeltaMemoryPtr(0) {
    this->update(data_ptr, tag, delta_idx, pid);
  }

  DeltaDataPtr(const DeltaDataPtr &) = default;
  DeltaDataPtr &operator=(const DeltaDataPtr &) = default;
  DeltaDataPtr(DeltaDataPtr &&) noexcept = default;
  DeltaDataPtr &operator=(DeltaDataPtr &&) noexcept = default;

 public:
  [[nodiscard]] constexpr inline uint8_t get_delta_idx() const {
    return static_cast<uint8_t>(this->_val >> (60u));
  }

  [[maybe_unused]] [[nodiscard]] constexpr inline uint8_t get_partition_idx()
      const {
    return static_cast<uint8_t>(this->_val >> (56u)) & 0x0fu;
  }

  [[nodiscard]] constexpr inline uint32_t get_tag() const {
    return static_cast<uint32_t>(this->_val >> (36u)) & 0x000fffffu;
  }

  [[nodiscard]] constexpr inline uint64_t get_offset() const {
    return (this->_val) & 0x0000000fffffffffu;
  }

  [[nodiscard]] constexpr inline uint8_t get_delta_idx_pid_pair() const {
    return static_cast<uint8_t>(this->_val >> (56u));
  }

  [[nodiscard]] static constexpr inline uint8_t create_delta_idx_pid_pair(
      delta_id_t delta_id, partition_id_t pid) {
    return ((((delta_id & 0x0fu)) << 4u) | (pid & 0x0fu));
  }

  inline void update(uintptr_t val) { this->_val = val; }
  inline void update(DeltaDataPtr &other) { this->_val = other._val; }
  inline void update(const char *data_ptr, uint32_t tag, delta_id_t delta_idx,
                     partition_id_t pid) {
    auto offset = reinterpret_cast<uintptr_t>(data_ptr) -
                  data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)];

    _val = 0;
    _val |= (offset & 0x0000000fffffffffu);
    _val |= (static_cast<uint64_t>(tag & 0x000fffffu)) << 36u;
    _val |= (static_cast<uint64_t>(pid & 0x0fu)) << 56u;
    _val |= (static_cast<uint64_t>(delta_idx & 0x0fu)) << 60u;
  }

  [[nodiscard]] inline bool verifyTag() const {
    assert(typeid(DeltaStore) == typeid(CircularDeltaStore));
    auto dTag = deltaStore_map[this->get_delta_idx()]->tag.load();
    dTag &= 0x000FFFFFu;

    if (dTag == this->get_tag())
      return true;
    else
      return false;
  }

  [[nodiscard]] inline bool isValid() const {
    if (__likely(this->verifyTag())) {
      return true;
    } else {
      return false;
    }
  }

  inline void debug() const {
    auto dTag = deltaStore_map[this->get_delta_idx()]->tag.load();
    dTag &= 0x000FFFFFu;

    if (dTag != this->get_tag())
      LOG(INFO) << "DTag: " << dTag << " | getTag: " << this->get_tag()
                << " DiD: " << (uint)get_delta_idx();
  }

  [[nodiscard]] inline char *get_ptr() const {
    // first-verify tag, else return nullptr and log error.
    if (this->verifyTag()) {
      // deference ptr.
      return reinterpret_cast<char *>(
          data_memory_base[this->get_delta_idx_pid_pair()] +
          this->get_offset());
    } else {
      // invalid list
      return nullptr;
    }
  }

  inline void release_impl() {}

  void saveInstanceCrossingPtr(vid_t vid) {
    // TODO CONSOLIDATION:
    //  this will create a copy.
    deltaStore_map[this->get_delta_idx()]->saveInstanceCrossingPtr(vid, *this);
  }

 public:
  [[nodiscard]] static auto getDeltaByIdx(delta_id_t idx) {
    // wrap-around?
    return deltaStore_map[idx];
  }

 protected:
  alignas(64) static std::unordered_map<delta_id_t, uintptr_t> data_memory_base;

  // to verify tag directly with required delta-store.
  alignas(64) static std::unordered_map<delta_id_t,
                                        CircularDeltaStore *> deltaStore_map;

  friend class CircularDeltaStore;
};

class DeltaPtr : public decltype(DeltaStore::ptrType) {
 public:
  DeltaPtr() : decltype(DeltaStore::ptrType)(0) {}
  explicit DeltaPtr(uintptr_t val) : decltype(DeltaStore::ptrType)(val) {}

  DeltaPtr(const DeltaPtr &other) = delete;
  DeltaPtr &operator=(const DeltaPtr &) = delete;

  DeltaPtr(DeltaPtr &&) noexcept = default;
  DeltaPtr &operator=(DeltaPtr &&) noexcept = default;

  DeltaPtr(decltype(DeltaStore::ptrType) &&o) noexcept
      : decltype(DeltaStore::ptrType)(std::move(o)) {}
  DeltaPtr &operator=(decltype(DeltaStore::ptrType) &&o) noexcept {
    this->_val = o._val;
    o._val = 0;
    return *this;
  }
};

template <class ptrType>
class TaggedDeltaDataPtr : public DeltaPtr {
 public:
  TaggedDeltaDataPtr() : DeltaPtr(0) {}
  explicit TaggedDeltaDataPtr(uintptr_t val) : DeltaPtr(val) {}
  TaggedDeltaDataPtr(const TaggedDeltaDataPtr &other) = delete;
  TaggedDeltaDataPtr &operator=(const TaggedDeltaDataPtr &) = delete;
  TaggedDeltaDataPtr(TaggedDeltaDataPtr &&) noexcept = default;
  TaggedDeltaDataPtr &operator=(TaggedDeltaDataPtr &&) noexcept = default;

  explicit TaggedDeltaDataPtr(DeltaPtr &&o) noexcept : DeltaPtr(std::move(o)) {}
  TaggedDeltaDataPtr &operator=(DeltaPtr &&o) noexcept {
    this->_val = o._val;
    o._val = 0;
    return *this;
  }

  TaggedDeltaDataPtr copy() { return TaggedDeltaDataPtr(this->_val); }

  //  inline void update(TaggedDeltaDataPtr &other) { this->_val = other._val; }

  [[nodiscard]] inline ptrType *typePtr() const {
    return reinterpret_cast<ptrType *>(this->DeltaPtr::ptr());
  }

  [[nodiscard]] inline ptrType *typePtr() {
    return reinterpret_cast<ptrType *>(this->DeltaPtr::ptr());
  }

  //  X* operator->(){
  //    return this->ptr();
  //  }

  [[nodiscard]] TaggedDeltaDataPtr reset() {
    TaggedDeltaDataPtr tmp{this->_val};
    this->_val = 0;
    return tmp;
  }
};

static_assert(sizeof(TaggedDeltaDataPtr<decltype(DeltaStore::ptrType)>) ==
                  sizeof(DeltaPtr),
              "Invalid sizes, may cause casting issues");
static_assert(sizeof(DeltaPtr) == sizeof(DeltaDataPtr),
              "Invalid sizes, may cause casting issues");
static_assert(sizeof(DeltaPtr) == sizeof(ClassicPtrWrapper),
              "Invalid sizes, may cause casting issues");
// following two are unnecessary.
static_assert(sizeof(DeltaPtr) == sizeof(DeltaMemoryPtr<DeltaDataPtr>),
              "Invalid sizes, may cause casting issues");
static_assert(sizeof(DeltaPtr) == sizeof(DeltaMemoryPtr<ClassicPtrWrapper>),
              "Invalid sizes, may cause casting issues");
}  // namespace storage

#endif  // PROTEUS_DELTA_MEMORY_PTR_HPP
