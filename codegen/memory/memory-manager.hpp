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

#include <mutex>
#include <stack>
#include <unordered_map>
#include <unordered_set>

class GpuMemAllocator {
 public:
  static void *malloc(size_t bytes);
  static void free(void *ptr);
};

class NUMAMemAllocator {
  static std::unordered_map<void *, size_t> sizes;

 public:
  static void *malloc(size_t bytes);
  static void free(void *ptr);
};

class NUMAPinnedMemAllocator {
  //  static std::unordered_map<void *, bool> sizes;

  //  static void *reg(void *mem, size_t bytes, bool allocated);

 public:
  static void *reg(void *mem, size_t bytes);
  static void *malloc(size_t bytes);
  static void free(void *ptr);
  static void unreg(void *mem);
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

  std::mutex m;
  std::mutex m_big_units;

  std::unordered_set<void *> big_units;

  std::unordered_map<void *, alloc_unit_info> units;
  std::unordered_map<void *, void *> mappings;
  std::stack<void *> allocations;

  std::stack<void *> free_cache;

 protected:
  SingleDeviceMemoryManager();
  ~SingleDeviceMemoryManager();

  alloc_unit_info &create_allocation();

  void *malloc(size_t bytes);
  void free(void *ptr);

  friend class MemoryManager;
};

typedef SingleDeviceMemoryManager<GpuMemAllocator, unit_capacity_gpu>
    SingleGpuMemoryManager;
typedef SingleDeviceMemoryManager<NUMAPinnedMemAllocator, unit_capacity_cpu>
    SingleCpuMemoryManager;

class MemoryManager {
 public:
  static SingleGpuMemoryManager **gpu_managers;
  static SingleCpuMemoryManager **cpu_managers;
  static void init(size_t gpu_buffers = 512, size_t cpu_buffers = 1024);
  static void destroy();

  static void *mallocGpu(size_t bytes);
  static void freeGpu(void *ptr);

  static void *mallocPinned(size_t bytes);
  static void freePinned(void *ptr);

 private:
};

#endif /* MEMORY_MANAGER_HPP_ */
