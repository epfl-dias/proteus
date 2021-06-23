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

#include "oltp/storage/multi-version/delta_storage.hpp"

namespace storage {

class DeltaMemoryPtr {
  //  4 bit     |  4 bit       | 20 bit  | 36-bit ( 64GB addressable)
  //  delta-idx | partition-id | tag     | offset-in-partition

 public:
  [[maybe_unused]] static constexpr size_t offset_bits = 36u;
  [[maybe_unused]] static constexpr size_t pid_bits = 4u;
  [[maybe_unused]] static constexpr size_t delta_id_bits = 4u;
  [[maybe_unused]] static constexpr size_t tag_bits = 20u;

 public:
  explicit DeltaMemoryPtr(size_t val) : _val(val) {}

 protected:
  inline void updateInternal(uintptr_t offset, uint32_t tag,
                             delta_id_t delta_idx, partition_id_t pid) {
    _val = 0;
    _val |= (offset & 0x0000000fffffffffu);
    _val |= (static_cast<uint64_t>(tag & 0x000fffffu)) << 36u;
    _val |= (static_cast<uint64_t>(pid & 0x0fu)) << 56u;
    _val |= (static_cast<uint64_t>(delta_idx & 0x0fu)) << 60u;
  }

 public:
  inline void updateVal(uint64_t val) { this->_val = val; }

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

  [[nodiscard]] inline bool verifyTag() const {
    auto dTag = deltaStore_map[this->get_delta_idx()]->tag.load();
    dTag &= 0x000FFFFFu;

    if (dTag == this->get_tag())
      return true;
    else
      return false;
  }

  void debug() {
    auto dTag = deltaStore_map[this->get_delta_idx()]->tag.load();
    dTag &= 0x000FFFFFu;

    if (dTag != this->get_tag())
      LOG(INFO) << "DTag: " << dTag << " | getTag: " << this->get_tag()
                << " DiD: " << (uint)get_delta_idx();
  }

 public:
  uint64_t _val{};

  // to verify tag directly with required delta-store.
  static std::map<delta_id_t, DeltaStore *> deltaStore_map;

  friend class DeltaStore;
};

class DeltaList : public DeltaMemoryPtr {
 public:
  // DeltaList() = default;
  explicit DeltaList(size_t val) : DeltaMemoryPtr(val) {}

  explicit DeltaList(DeltaMemoryPtr other) : DeltaMemoryPtr(other._val) {}

  inline void update(const char *list_ptr, uint32_t tag, delta_id_t delta_idx,
                     partition_id_t pid) {
    assert(reinterpret_cast<uintptr_t>(list_ptr) >=
           list_memory_base[create_delta_idx_pid_pair(delta_idx, pid)]);

    auto offset = reinterpret_cast<uintptr_t>(list_ptr) -
                  list_memory_base[create_delta_idx_pid_pair(delta_idx, pid)];
    this->DeltaMemoryPtr::updateInternal(offset, tag, delta_idx, pid);
  }

  [[nodiscard]] inline char *ptr() const {
    // first-verify tag, else return nullptr and log error.
    if (this->verifyTag()) {
      // deference ptr.
      return reinterpret_cast<char *>(
          list_memory_base[this->get_delta_idx_pid_pair()] +
          this->get_offset());
    } else {
      // invalid list
      return nullptr;
    }
  }

 private:
  static std::map<delta_id_t, uintptr_t> list_memory_base;

  friend class DeltaStore;
};

class DeltaDataPtr : public DeltaMemoryPtr {
 public:
  explicit DeltaDataPtr(size_t val) : DeltaMemoryPtr(val) {}

  explicit DeltaDataPtr(DeltaMemoryPtr other) : DeltaMemoryPtr(other._val) {}

  explicit DeltaDataPtr(const char *data_ptr, uint32_t tag,
                        delta_id_t delta_idx, partition_id_t pid)
      : DeltaMemoryPtr(0) {
    this->update(data_ptr, tag, delta_idx, pid);
  }

  inline void update(const char *data_ptr, uint32_t tag, delta_id_t delta_idx,
                     partition_id_t pid) {
    auto offset = reinterpret_cast<uintptr_t>(data_ptr) -
                  data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)];

    this->DeltaMemoryPtr::updateInternal(offset, tag, delta_idx, pid);
  }

  inline bool is_valid() {
    if (__likely(this->verifyTag())) {
      return true;
    } else {
      return false;
    }
  }
  inline void debug() { this->DeltaMemoryPtr::debug(); }

  inline char *get_ptr() const {
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

 protected:
  static std::map<delta_id_t, uintptr_t> data_memory_base;

  friend class DeltaStore;
};

template <class X>
class TaggedDeltaDataPtr final : public DeltaDataPtr {
 public:
  // TaggedDeltaDataPtr() = default;
  explicit TaggedDeltaDataPtr(size_t val) : DeltaDataPtr(val) {}

  explicit TaggedDeltaDataPtr(DeltaMemoryPtr other)
      : DeltaDataPtr(other._val) {}

  explicit TaggedDeltaDataPtr(const char *data_ptr, uint32_t tag,
                              delta_id_t delta_idx, partition_id_t pid)
      : DeltaDataPtr(0) {
    this->update(data_ptr, tag, delta_idx, pid);
  }

  [[nodiscard]] inline X *ptr() const {
    return reinterpret_cast<X *>(this->DeltaDataPtr::get_ptr());
  }

  void debug() { this->DeltaDataPtr::debug(); }

  //  X* operator->(){
  //    return this->ptr();
  //  }

  friend class DeltaStore;
  friend class DeltaDataPtr;
};
}  // namespace storage

#endif  // PROTEUS_DELTA_MEMORY_PTR_HPP
