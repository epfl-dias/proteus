/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef STORAGE_DELTA_STORAGE_HPP_
#define STORAGE_DELTA_STORAGE_HPP_

#include <sys/mman.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <platform/memory/memory-manager.hpp>

#include "oltp/common/common.hpp"
#include "oltp/common/memory-chunk.hpp"

#define DELTA_DEBUG 1
#define GC_CAPACITY_MIN_PERCENT 75

namespace storage {

/* Currently DeltaStore is not resizeable*/

class DeltaList;

class alignas(4096) DeltaStore {
 public:
  DeltaStore(delta_id_t delta_id, uint64_t ver_list_capacity = 4,
             uint64_t ver_data_capacity = 4, partition_id_t num_partitions = 1);
  ~DeltaStore();

  void print_info();
  void *insert_version(DeltaList &delta_list, xid_t t_min, xid_t t_max,
                       size_t rec_size, partition_id_t partition_id);
  //  void *validate_or_create_list(void *list_ptr, size_t &delta_ver_tag,
  //                                ushort partition_id);
  void *validate_or_create_list(DeltaList &delta_list,
                                partition_id_t partition_id);
  void *getTransientChunk(DeltaList &delta_list, size_t size,
                          partition_id_t partition_id);
  void *create_version(size_t size, partition_id_t partition_id);
  void gc();
  // void gc_with_counter_arr(int wrk_id);

  inline bool should_gc() {
#if (GC_CAPACITY_MIN_PERCENT > 0)
    for (auto &p : partitions) {
      // std::cout << "usage: " << p->usage() << std::endl;
      if (p->usage() > ((double)GC_CAPACITY_MIN_PERCENT) / 100) {
        // std::cout << "usage: " << p->usage() << std::endl;
        return true;
      }
    }
    return false;
#else
    return true;
#endif
  }

  inline void __attribute__((always_inline))
  increment_reader(uint64_t epoch, worker_id_t worker_id) {
    while (gc_lock < 0 && !should_gc())
      ;

    if (max_active_epoch < epoch) {
      max_active_epoch = epoch;
    }
    this->readers++;
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, worker_id_t worker_id) {
    if (readers.fetch_sub(1) <= 1 && touched) {
      gc();
    }
  }

 private:
  class alignas(2 * 1024 * 1024) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    partition_id_t pid;
    const oltp::common::mem_chunk ver_list_mem;
    const oltp::common::mem_chunk ver_data_mem;
    bool touched;
    std::mutex print_lock;
    bool printed;
    const char *list_cursor_max;
    const char *data_cursor_max;

    std::vector<bool> reset_listeners;

   public:
    DeltaPartition(char *ver_list_cursor, oltp::common::mem_chunk ver_list_mem,
                   char *ver_data_cursor, oltp::common::mem_chunk ver_data_mem,
                   partition_id_t pid);

    ~DeltaPartition() {
      if (DELTA_DEBUG) {
        LOG(INFO) << "Clearing DeltaPartition-" << (int)pid;
      }
      MemoryManager::freePinned(ver_list_mem.data);
      MemoryManager::freePinned(ver_data_mem.data);
    }

    void reset() {
      if (__builtin_expect(touched, 1)) {
        ver_list_cursor = (char *)ver_list_mem.data;
        ver_data_cursor = (char *)ver_data_mem.data;

        for (auto &&reset_listener : reset_listeners) {
          reset_listener = true;
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      touched = false;
    }
    void *getListChunk();

    void *getChunk(size_t size);

    void *getVersionDataChunk(size_t rec_size);

    inline double usage() {
      return ((double)(((char *)ver_data_cursor.load() -
                        (char *)this->ver_data_mem.data))) /
             this->ver_data_mem.size;
    }

    void report() {
      /* data memory only */
      char *curr_ptr = (char *)ver_data_cursor.load();
      char *start_ptr = (char *)ver_data_mem.data;

      auto diff = curr_ptr - start_ptr;
      double percent = ((double)diff / ver_data_mem.size) * 100;
      std::cout.precision(2);
      std::cout << "\tMemory Consumption: "
                << ((double)diff) / (1024 * 1024 * 1024) << "GB/"
                << (double)ver_data_mem.size / (1024 * 1024 * 1024)
                << "GB  --  " << percent << "%%" << std::endl;
    }

    friend class DeltaStore;
  };

  std::atomic<uint32_t> tag{};
  xid_t max_active_epoch;

  std::vector<DeltaPartition *> partitions{};
  std::atomic<uint> readers{};
  std::atomic<short> gc_lock{};
  bool touched;
  std::atomic<uint> gc_reset_success{};
  std::atomic<uint> gc_requests{};

 public:
  const delta_id_t delta_id;
  uint64_t total_mem_reserved;

  friend class DeltaMemoryPtr;
};

class DeltaMemoryPtr {
  //  4 bit     |  4 bit       | 20 bit  | 36-bit ( 64GB addressable)
  //  delta-idx | partition-id | tag     | offset-in-partition

 public:
  [[maybe_unused]] static constexpr size_t offset_bits = 36u;
  [[maybe_unused]] static constexpr size_t pid_bits = 4u;
  [[maybe_unused]] static constexpr size_t delta_id_bits = 4u;
  [[maybe_unused]] static constexpr size_t tag_bits = 20u;

  enum ptrType { LIST_PTR, DATA_PTR };

 protected:
  // DeltaMemoryPtr() = default;
  explicit DeltaMemoryPtr(size_t val) : _val(val) {}
  //  explicit DeltaMemoryPtr(uint64_t offset, uint32_t tag, delta_id_t
  //  delta_idx, partition_id_t pid, ptrType type): type(type) {
  //    this->update(offset, tag, delta_idx, pid);
  //  }

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
    auto dTag = deltaStore_map[this->get_delta_idx()]->tag.load(
        std::memory_order_acquire);
    dTag &= 0x000FFFFFu;

    if (dTag == this->get_tag())
      return true;
    else
      return false;
  }

 public:
  uint64_t _val{};

  // to verify tag directly with required delta-store.
  static std::map<delta_id_t, DeltaStore *> deltaStore_map;

  friend class DeltaStore;
};

class DeltaList : public DeltaMemoryPtr {
 public:
  // static constexpr ptrType type = DeltaMemoryPtr::ptrType::LIST_PTR;

 public:
  // DeltaList() = default;
  explicit DeltaList(size_t val) : DeltaMemoryPtr(val) {}

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
  // static constexpr ptrType type = DeltaMemoryPtr::ptrType::DATA_PTR;

 public:
  // DeltaDataPtr() = default;
  virtual ~DeltaDataPtr() = default;
  explicit DeltaDataPtr(size_t val) : DeltaMemoryPtr(val) {}

  explicit DeltaDataPtr(const char *data_ptr, uint32_t tag,
                        delta_id_t delta_idx, partition_id_t pid)
      : DeltaMemoryPtr(0) {
    this->update(data_ptr, tag, delta_idx, pid);
  }

  inline void update(const char *data_ptr, uint32_t tag, delta_id_t delta_idx,
                     partition_id_t pid) {
    assert(reinterpret_cast<uintptr_t>(data_ptr) >=
           data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)]);

    auto offset = reinterpret_cast<uintptr_t>(data_ptr) -
                  data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)];
    // LOG(INFO) << "offset: " << offset;

    this->DeltaMemoryPtr::updateInternal(offset, tag, delta_idx, pid);

    if (reinterpret_cast<uintptr_t>(data_ptr) <
        data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)]) {
      LOG(INFO) << "Delta-ID: " << (uint32_t)this->get_delta_idx();
      LOG(INFO) << "Offset: " << (uint32_t)this->get_offset();
      LOG(INFO) << "Tag: " << (uint32_t)this->get_tag();
      LOG(INFO) << "Data-ptr: " << reinterpret_cast<uintptr_t>(data_ptr);
      LOG(INFO) << "d_memory_base: "
                << data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)];
    }

    assert(reinterpret_cast<uintptr_t>(data_ptr) >=
           data_memory_base[create_delta_idx_pid_pair(delta_idx, pid)]);
  }

  inline bool is_valid() {
    if (__likely(this->verifyTag())) {
      return true;
    } else {
      return false;
    }
  }

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

  explicit TaggedDeltaDataPtr(const char *data_ptr, uint32_t tag,
                              delta_id_t delta_idx, partition_id_t pid)
      : DeltaDataPtr(0) {
    this->update(data_ptr, tag, delta_idx, pid);
  }

  [[nodiscard]] inline X *ptr() const {
    return reinterpret_cast<X *>(this->DeltaDataPtr::get_ptr());
  }

  //  X* operator->(){
  //    return this->ptr();
  //  }

  //  ~TaggedDeltaDataPtr(){
  //    LOG(INFO) << "Destructor data ptr called!";
  //  }

  friend class DeltaStore;
  friend class DeltaDataPtr;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
