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

#include "memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

constexpr size_t freed_cache_cap = 16;

void buffer_manager_init(size_t gpu_buffs, size_t cpu_buffs);
void buffer_manager_destroy();

void MemoryManager::init() {
  const topology &topo = topology::getInstance();
  std::cout << topo << std::endl;

  gpu_managers = new SingleGpuMemoryManager *[topo.getGpuCount()];
  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    gpu_managers[gpu.index_in_topo] = new SingleGpuMemoryManager();
    // warm-up
    // NOTE: how many buffers should we warm up ? how do we calibrate that
    // without cheating ?
    void *ptrs[8];
    for (size_t i = 0; i < 8; ++i) {
      ptrs[i] =
          gpu_managers[gpu.index_in_topo]->malloc((unit_capacity_gpu / 4) * 3);
    }
    for (size_t i = 0; i < 8; ++i) {
      gpu_managers[gpu.index_in_topo]->free(ptrs[i]);
    }
  }
  cpu_managers = new SingleCpuMemoryManager *[topo.getCpuNumaNodeCount()];
  for (const auto &cpu : topo.getCpuNumaNodes()) {
    set_exec_location_on_scope d(cpu);
    cpu_managers[cpu.index_in_topo] = new SingleCpuMemoryManager();
    // warm-up
    // NOTE: how many buffers should we warm up ? how do we calibrate that
    // without cheating ?
    void *ptrs[8];
    for (size_t i = 0; i < 8; ++i) {
      ptrs[i] =
          cpu_managers[cpu.index_in_topo]->malloc((unit_capacity_cpu / 4) * 3);
    }
    for (size_t i = 0; i < 8; ++i) {
      cpu_managers[cpu.index_in_topo]->free(ptrs[i]);
    }
  }

  buffer_manager_init(4*256, 1024); // (*4*4, *4*4)
  // might need it for out of gpu
  // buffer_manager_init(4 * 256, 1024 * 16);  // (*4*4, *4*4)
}

void MemoryManager::destroy() {
  buffer_manager_destroy();
  const auto &topo = topology::getInstance();
  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    delete gpu_managers[gpu.index_in_topo];
  }
  delete[] gpu_managers;
  for (const auto &cpu : topo.getCpuNumaNodes()) {
    set_exec_location_on_scope d(cpu);
    delete cpu_managers[cpu.index_in_topo];
  }
  delete[] cpu_managers;
}

constexpr inline size_t fixSize(size_t bytes) { return (bytes + 0xF) & ~0xF; }

void *MemoryManager::mallocGpu(size_t bytes) {
  eventlogger.log(NULL, log_op::MEMORY_MANAGER_ALLOC_GPU_START);
  nvtxRangePushA("mallocGpu");
  bytes = fixSize(bytes);
  const auto &dev = topology::getInstance().getActiveGpu();
  void *ptr = gpu_managers[dev.id]->malloc(bytes);
  assert(ptr);
  nvtxRangePop();
  eventlogger.log(NULL, log_op::MEMORY_MANAGER_ALLOC_GPU_END);
  return ptr;
}

void MemoryManager::freeGpu(void *ptr) {
  nvtxRangePushA("freeGpu");
  const auto *dev = topology::getInstance().getGpuAddressed(ptr);
  assert(dev);
  set_device_on_scope d(*dev);
  gpu_managers[dev->id]->free(ptr);
  nvtxRangePop();
}

void *MemoryManager::mallocPinned(size_t bytes) {
  const auto &topo = topology::getInstance();
  eventlogger.log(NULL, log_op::MEMORY_MANAGER_ALLOC_PINNED_START);
  nvtxRangePushA("mallocPinned");
  bytes = fixSize(bytes);
  const auto &cpu = affinity::get();
  uint32_t node = cpu.index_in_topo;
  void *ptr = cpu_managers[node]->malloc(bytes);
  assert(ptr);
  nvtxRangePop();
  eventlogger.log(NULL, log_op::MEMORY_MANAGER_ALLOC_PINNED_END);
  // std::cout << "Alloc: " << node << " " << ptr << " " << bytes << " " <<
  // topo.getCpuNumaNodeAddressed(ptr)->id; //std::endl;
  return ptr;
}

void MemoryManager::freePinned(void *ptr) {
  nvtxRangePushA("freePinned");
  const auto *dev = topology::getInstance().getCpuNumaNodeAddressed(ptr);
  assert(dev);
  uint32_t node = dev->index_in_topo;
  // std::cout << "Free: " << dev->id << " (" << node << ") " << ptr <<
  // std::endl;
  cpu_managers[node]->free(ptr);
  nvtxRangePop();
}

void *GpuMemAllocator::malloc(size_t bytes) {
#ifndef NCUDA
  void *ptr;
  gpu_run(cudaMalloc(&ptr, bytes));
  return ptr;
#else
  assert(false);
  return NULL;
#endif
}

void GpuMemAllocator::free(void *ptr) { gpu_run(cudaFree(ptr)); }

void *NUMAPinnedMemAllocator::malloc(size_t bytes) {
  void *ptr = affinity::get().alloc(bytes);
  assert(ptr && "Memory allocation failed!");
  assert(bytes > 4);
  ((int *)ptr)[0] = 0;  // force allocation of first 4bytes
  // NOTE: do we want to force allocation of all pages? If yes, use:
  // memset(mem, 0, bytes);
  gpu_run(cudaHostRegister(ptr, bytes, 0));
  sizes.emplace(ptr, bytes);
  return ptr;
}

void NUMAPinnedMemAllocator::free(void *ptr) {
  gpu_run(cudaHostUnregister(ptr));
  auto it = sizes.find(ptr);
  assert(
      it != sizes.end() &&
      "Memory did not originate from this allocator (or is already released)!");
  topology::cpunumanode::free(ptr, it->second);
  sizes.erase(it);
}

template <typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::SingleDeviceMemoryManager() {}

template <typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::~SingleDeviceMemoryManager() {
  while (!free_cache.empty()) {
    allocator::free(free_cache.top());
    free_cache.pop();
  }
  // assert(allocations.empty());
  // assert(mappings   .empty());
  // assert(units      .empty());
  // assert(big_units  .empty());
  // assert(free_cache .empty());
  exit(0);
}

template <typename allocator, size_t unit_cap>
typename SingleDeviceMemoryManager<allocator, unit_cap>::alloc_unit_info &
SingleDeviceMemoryManager<allocator, unit_cap>::create_allocation() {
  void *ptr;
  if (free_cache.empty()) {
    ptr = allocator::malloc(unit_cap);
  } else {
    ptr = free_cache.top();
    free_cache.pop();
  }
  allocations.emplace(ptr);
  return units.emplace(ptr, ptr).first->second;
}

template <typename allocator, size_t unit_cap>
void *SingleDeviceMemoryManager<allocator, unit_cap>::malloc(size_t bytes) {
  bytes = fixSize(bytes);

  if (bytes >= unit_cap) {
    void *ptr = allocator::malloc(bytes);

    {
      std::lock_guard<std::mutex> lock(m_big_units);
      big_units.emplace(ptr);
    }

    return ptr;
  }

  {
    std::lock_guard<std::mutex> lock(m);

    alloc_unit_info *info;
    if (allocations.empty()) {
      info = &(create_allocation());
    } else {
      void *latest = allocations.top();

      auto match = units.find(latest);
      assert(match != units.end() && "Unit not found!");
      info = &(match->second);

      if (info->fill + bytes > unit_cap) {
        info = &(create_allocation());
      }
    }

    void *ptr = ((void *)(((char *)info->base) + info->fill));
    info->fill += bytes;
    info->sub_units += 1;
    mappings.emplace(ptr, info->base);

    return ptr;
  }
}

template <typename allocator, size_t unit_cap>
void SingleDeviceMemoryManager<allocator, unit_cap>::free(void *ptr) {
  {
    std::lock_guard<std::mutex> lock(m_big_units);
    auto f = big_units.find(ptr);
    if (f != big_units.end()) {
      big_units.erase(f);
      allocator::free(ptr);
      return;
    }
  }

  {
    std::lock_guard<std::mutex> lock(m);
    auto f = mappings.find(ptr);
    if (f == mappings.end()) {
      for (auto &t : mappings)
        std::cout << t.first << " " << t.second << std::endl;
    }
    assert(f != mappings.end() && "Mapping does not exist!");

    void *base = f->second;
    mappings.erase(f);

    auto fu = units.find(base);
    assert(fu != units.end() && "Unit not found!");
    alloc_unit_info &info = fu->second;
    assert(info.sub_units > 0);
    info.sub_units = info.sub_units - 1;
    if (info.sub_units == 0) {
      if (!allocations.empty() && allocations.top() == base) {
        allocations.pop();
        while (!allocations.empty()) {
          void *tmp_base = allocations.top();
          bool still_valid = (units.find(tmp_base) != units.end());
          if (!still_valid)
            allocations.pop();
          else
            break;
        }
      }
      if (free_cache.size() < freed_cache_cap) {
        free_cache.push(base);
      } else {
        allocator::free(ptr);
      }
      units.erase(fu);
    }
  }
}

SingleGpuMemoryManager **MemoryManager::gpu_managers;
SingleCpuMemoryManager **MemoryManager::cpu_managers;

std::unordered_map<void *, size_t> NUMAPinnedMemAllocator::sizes;
