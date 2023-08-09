/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef MEMORY_MANAGER_HPP_
#define MEMORY_MANAGER_HPP_

#include <atomic>
#include <map>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <unordered_set>

void set_trace_allocations(bool val = true, bool silent_set_fail = false);

class GpuMemAllocator {
 public:
  static std::atomic<size_t> usage;
  static void *malloc(size_t bytes);
  static void free(void *ptr);

  static void freed();
  static size_t rem() { return usage; }
};

class NUMAMemAllocator {
  static std::mutex m_sizes;
  static std::unordered_map<void *, size_t> sizes;

 public:
  static std::atomic<size_t> usage;

  static void *malloc(size_t bytes);
  static void free(void *ptr);

  static void freed();
};

class NUMAPinnedMemAllocator {
  //  static std::unordered_map<void *, bool> sizes;

  //  static void *reg(void *mem, size_t bytes, bool allocated);
 public:
  static void *reg(void *mem, size_t bytes);
  static void *malloc(size_t bytes);
  static void free(void *ptr);
  static void unreg(void *mem);

  static size_t rem() { return NUMAMemAllocator::usage; }

  static void freed();
};

constexpr size_t unit_capacity_gpu = 128 * 1024 * 1024;
constexpr size_t unit_capacity_cpu = 1024 * 1024 * 1024;

template <typename allocator, size_t unit_cap = unit_capacity_gpu>
class SingleDeviceMemoryManager {
  struct alloc_unit_info {
    void *base;

    size_t fill;
    size_t sub_units;

    alloc_unit_info(void *base) : base(base), fill(0), sub_units(0) {}
  };

  struct allocation_t {
    static constexpr size_t backtrace_limit =
#ifndef NDEBUG
        32;
#else
        0;
#endif

    void *ptr;

    void *backtrace[backtrace_limit];
    int backtrace_size;

    inline allocation_t(void *ptr) : ptr(ptr) {}

    operator void *() const { return ptr; }
  };

  std::mutex m;
  std::mutex m_big_units;

  //  std::unordered_set<void *> big_units;
  std::unordered_map<void *, size_t> big_units;

  std::map<void *, alloc_unit_info> units;
  //  std::unordered_map<void *, void *> mappings;
  //  std::unordered_map<void *, std::pair<void **, size_t>> dmappings;
  std::stack<allocation_t> allocations;

  std::stack<void *> free_cache;

  std::atomic<size_t> active_bytes;

 protected:
  SingleDeviceMemoryManager();
  ~SingleDeviceMemoryManager();

  alloc_unit_info &create_allocation();
  void destroy_allocation();

  void *malloc(size_t bytes);
  void free(void *ptr);

  friend class MemoryManager;

 public:
  size_t usage_unsafe() const { return active_bytes; }
};

typedef SingleDeviceMemoryManager<GpuMemAllocator, unit_capacity_gpu>
    SingleGpuMemoryManager;
typedef SingleDeviceMemoryManager<NUMAPinnedMemAllocator, unit_capacity_cpu>
    SingleCpuMemoryManager;

namespace proteus {
class platform;
}

class MemoryManager {
 public:
  static SingleGpuMemoryManager **gpu_managers;
  static SingleCpuMemoryManager **cpu_managers;

 private:
  static void init(float gpu_mem_pool_percentage = 0.1,
                   float cpu_mem_pool_percentage = 0.1,
                   size_t log_buffers = 250);
  static void destroy();

  friend class proteus::platform;

 public:
  static void *mallocGpu(size_t bytes);
  static void freeGpu(void *ptr);

  static void *mallocPinned(size_t bytes);
  static void *mallocPinnedOnNode(size_t bytes, uint32_t node);
  static void freePinned(void *ptr);

  static void *mallocPinnedAligned(size_t bytes, size_t align);
  static void freePinnedAligned(void *ptr);

  template <typename P>
  static inline bool is_aligned(P *ptr, size_t align) {
    return (reinterpret_cast<uintptr_t>(ptr) % align == 0);
  }

 private:
};

#endif /* MEMORY_MANAGER_HPP_ */
