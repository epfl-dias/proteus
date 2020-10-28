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

#include "memory/memory-manager.hpp"
#include "oltp/common/memory-chunk.hpp"

#define DELTA_DEBUG 1
#define GC_CAPACITY_MIN_PERCENT 75

namespace storage {

/* Currently DeltaStore is not resizeable*/

class DeltaList;

class alignas(4096) DeltaStore {
 public:
  DeltaStore(uint8_t delta_id, uint64_t ver_list_capacity = 4,
             uint64_t ver_data_capacity = 4, int num_partitions = 1);
  ~DeltaStore();

  void print_info();
  void *insert_version(DeltaList &delta_chunk, uint64_t t_min, uint64_t t_max,
                       uint rec_size, ushort partition_id);
  //  void *validate_or_create_list(void *list_ptr, size_t &delta_ver_tag,
  //                                ushort partition_id);
  void *validate_or_create_list(DeltaList &delta_chunk, ushort partition_id);
  void *getTransientChunk(DeltaList &delta_chunk, uint size,
                          ushort partition_id);
  void *create_version(size_t size, ushort partition_id);
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
  increment_reader(uint64_t epoch, uint8_t worker_id) {
    while (gc_lock < 0 && !should_gc())
      ;

    if (max_active_epoch < epoch) {
      max_active_epoch = epoch;
    }
    this->readers++;
  }

  inline void __attribute__((always_inline))
  decrement_reader(uint64_t epoch, uint8_t worker_id) {
    if (readers.fetch_sub(1) <= 1 && touched) {
      gc();
    }
  }

 private:
  class alignas(4096) DeltaPartition {
    std::atomic<char *> ver_list_cursor;
    std::atomic<char *> ver_data_cursor;
    int pid;
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
                   int pid);

    ~DeltaPartition() {
      if (DELTA_DEBUG) {
        LOG(INFO) << "Clearing DeltaPartition-" << pid;
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
  uint64_t max_active_epoch;
  uint32_t delta_id;
  std::vector<DeltaPartition *> partitions;
  std::atomic<uint> readers{};
  std::atomic<short> gc_lock{};
  bool touched;
  std::atomic<uint> gc_reset_success{};
  std::atomic<uint> gc_requests{};

 public:
  uint64_t total_mem_reserved;

  friend class DeltaList;
};

class DeltaList {
  //  4 bit     |  4 bit       | 20 bit  | 36-bit ( 64GB addressable)
  //  delta-idx | partition-id | tag     | offset-in-partition

 public:
  DeltaList() = default;
  explicit DeltaList(size_t val) : _val(val) {}
  explicit DeltaList(uint64_t offset, uint32_t tag, uint8_t delta_idx,
                     uint8_t pid) {
    this->update(offset, tag, delta_idx, pid);
  }
  DeltaList(DeltaList &) = default;
  DeltaList(DeltaList &&) = default;

  inline void update(uint64_t val) { this->_val = val; }

  inline void update(uint64_t offset, uint32_t tag, uint8_t delta_idx,
                     uint8_t pid) {
    _val = 0;
    _val |= (offset & 0x0000000fffffffffu);
    _val |= (static_cast<uint64_t>(tag & 0x000fffffu)) << 36u;
    _val |= (static_cast<uint64_t>(pid & 0x0fu)) << 56u;
    _val |= (static_cast<uint64_t>(delta_idx & 0x0fu)) << 60u;
  }

  inline void update(const char *list_ptr, uint32_t tag, uint8_t delta_idx,
                     uint8_t pid) {
    auto offset = (list_ptr -
                   list_memory_base[create_delta_idx_pid_pair(delta_idx, pid)]);
    this->update(offset, tag, delta_idx, pid);
  }

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
      uint8_t delta_id, uint8_t pid) {
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

  [[nodiscard]] inline char *ptr() const {
    // first-verify tag, else return nullptr and log error.
    if (this->verifyTag()) {
      // deference ptr.
      return list_memory_base[this->get_delta_idx_pid_pair()] +
             this->get_offset();
    } else {
      // invalid list
      return nullptr;
    }
  }

 private:
  uint64_t _val{};

  // to verify tag directly with required delta-store.
  static std::map<uint8_t, DeltaStore *> deltaStore_map;
  static std::map<uint8_t, char *> list_memory_base;

 public:
  [[maybe_unused]] static constexpr size_t offset_bits = 36u;
  [[maybe_unused]] static constexpr size_t pid_bits = 4u;
  [[maybe_unused]] static constexpr size_t delta_id_bits = 4u;
  [[maybe_unused]] static constexpr size_t tag_bits = 20u;

  friend class DeltaStore;
};

}  // namespace storage

#endif /* STORAGE_DELTA_STORAGE_HPP_ */
