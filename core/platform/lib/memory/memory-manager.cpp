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

#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/logging.hpp>

#ifndef NDEBUG
#include <execinfo.h>

#include <platform/util/timing.hpp>
#endif

#ifndef NDEBUG
static bool trace_allocations;
#else
constexpr bool trace_allocations = false;
#endif

void set_trace_allocations(bool val, bool silent_set_fail) {
#ifndef NDEBUG
  trace_allocations = val;
#else
  if ((!silent_set_fail) && val) {
    auto msg = "Can not enable memory & leak tracing in a NDEBUG build";
    LOG(FATAL) << msg;
    throw runtime_error(msg);
  }
#endif
}

constexpr size_t freed_cache_cap = 16;

void buffer_manager_init(float gpu_mem_pool_percentage,
                         float cpu_mem_pool_percentage, size_t log_buffers);
void buffer_manager_destroy();

void MemoryManager::init(float gpu_mem_pool_percentage,
                         float cpu_mem_pool_percentage, size_t log_buffers) {
  const topology &topo = topology::getInstance();

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

  buffer_manager_init(gpu_mem_pool_percentage, cpu_mem_pool_percentage,
                      log_buffers);
}

void MemoryManager::destroy() {
  buffer_manager_destroy();
  const auto &topo = topology::getInstance();
  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    delete gpu_managers[gpu.index_in_topo];
  }
  GpuMemAllocator::freed();
  delete[] gpu_managers;
  for (const auto &cpu : topo.getCpuNumaNodes()) {
    set_exec_location_on_scope d(cpu);
    delete cpu_managers[cpu.index_in_topo];
  }
  NUMAPinnedMemAllocator::freed();
  delete[] cpu_managers;
}

constexpr inline size_t fixSize(size_t bytes) {
  return (std::max(bytes, size_t{1}) + 0xF) & ~0xF;
}

void *MemoryManager::mallocGpu(size_t bytes) {
  eventlogger.log(nullptr, log_op::MEMORY_MANAGER_ALLOC_GPU_START);
  nvtxRangePushA("mallocGpu");
  bytes = fixSize(bytes);
  const auto &dev = topology::getInstance().getActiveGpu();
  void *ptr = gpu_managers[dev.id]->malloc(bytes);
  assert(ptr);
  nvtxRangePop();
  eventlogger.log(nullptr, log_op::MEMORY_MANAGER_ALLOC_GPU_END);
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

void *MemoryManager::mallocPinnedOnNode(size_t bytes, uint32_t node) {
  static const auto &numaNodes = topology::getInstance().getCpuNumaNodes();
  assert(node < numaNodes.size());
  set_exec_location_on_scope d{numaNodes[node]};
  return mallocPinned(bytes);
}

void *MemoryManager::mallocPinned(size_t bytes) {
  // const auto &topo = topology::getInstance();
  //  eventlogger.log(nullptr, log_op::MEMORY_MANAGER_ALLOC_PINNED_START);
  nvtxRangePushA("mallocPinned");
  bytes = fixSize(bytes);
  const auto &cpu = affinity::get();
  uint32_t node = cpu.index_in_topo;
  void *ptr = cpu_managers[node]->malloc(bytes);
  assert(ptr);
  nvtxRangePop();
  //  eventlogger.log(nullptr, log_op::MEMORY_MANAGER_ALLOC_PINNED_END);
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
  void *ptr = nullptr;
  gpu_run(cudaMalloc(&ptr, bytes));
  return ptr;
#else
  assert(false);
  return nullptr;
#endif
}

void GpuMemAllocator::free(void *ptr) { gpu_run(cudaFree(ptr)); }

void *NUMAMemAllocator::malloc(size_t bytes) {
  void *ptr = affinity::get().alloc(bytes);
  assert(ptr && "Memory allocation failed!");
  {
    std::lock_guard<std::mutex> lock{m_sizes};
    sizes.emplace(ptr, bytes);
  }
  usage.fetch_add(bytes);
  return ptr;
}

void NUMAMemAllocator::free(void *ptr) {
  size_t bytes;
  {
    std::lock_guard<std::mutex> lock{m_sizes};
    auto it = sizes.find(ptr);
    bytes = it->second;
    assert(it != sizes.end() &&
           "Memory did not originate from this allocator (or is already "
           "released)!");
    sizes.erase(it);
  }
  topology::cpunumanode::free(ptr, bytes);
  usage.fetch_sub(bytes);
}

void *NUMAPinnedMemAllocator::reg(void *ptr, size_t bytes) {
  assert(ptr && "Registering NULL ptr!");
  // Do not remove below line, it forces allocation of first 4bytes (and its
  // page)
  //  std::cout << "Registering: " << ptr << " " << ((char *)ptr)[0] << '\n';
  //  If you hit an "invalid argument" error here, check the number of
  //  configured huge pages. You may have run out of memory, the error can be
  //  misleading
  gpu_run(cudaHostRegister(ptr, bytes, 0));
  return ptr;
}

void NUMAPinnedMemAllocator::unreg(void *ptr) {
  gpu_run(cudaHostUnregister(ptr));
}

void *NUMAPinnedMemAllocator::malloc(size_t bytes) {
  void *ptr = NUMAMemAllocator::malloc(bytes);
  assert(ptr && "Memory allocation failed!");
  assert(bytes > 4);
  ((int *)ptr)[0] = 0;  // force allocation of first 4bytes
  // NOTE: do we want to force allocation of all pages? If yes, use:
  // memset(mem, 0, bytes);
  return reg(ptr, bytes);
}

void NUMAPinnedMemAllocator::free(void *ptr) {
  unreg(ptr);
  NUMAMemAllocator::free(ptr);
}

template <typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::SingleDeviceMemoryManager()
    : active_bytes(0) {}

template <typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::~SingleDeviceMemoryManager() {
  while (!free_cache.empty()) {
    allocator::free(free_cache.top());
    free_cache.pop();
  }

  //  for (const auto &p: dmappings){
  //    auto bt = p.second.first;
  //    auto n = p.second.second;
  //
  //    char **trace = backtrace_symbols(bt, n);
  //    std::cout << "Detected memory leak created from: " << std::endl;
  //    for (size_t i = 0; i < n; ++i) {
  //      std::cout << trace[i] << std::endl;
  //    }
  //  }

#ifndef NDEBUG
  // If not trace_allocations, then we do not have the necessary info to proceed
  if (trace_allocations && !allocations.empty()) {
    while (!allocations.empty()) {
      auto alloc = allocations.top();
      char **trace = backtrace_symbols(alloc.backtrace, alloc.backtrace_size);
      std::cout << "Detected memory leak created from: " << std::endl;
      for (size_t i = 0; i < alloc.backtrace_size; ++i) {
        std::cout << trace[i] << std::endl;
      }
      destroy_allocation();
    }
    assert(false);
  }
#endif

  assert(allocations.empty());
  //#ifndef NDEBUG
  //  assert(mappings.empty());
  //#endif
  assert(units.empty());
  assert(big_units.empty());
  assert(free_cache.empty());
  assert(active_bytes == 0);
}

template <typename allocator, size_t unit_cap>
typename SingleDeviceMemoryManager<allocator, unit_cap>::alloc_unit_info &
SingleDeviceMemoryManager<allocator, unit_cap>::create_allocation() {
  active_bytes += unit_cap;

  void *ptr;
  if (free_cache.empty()) {
    ptr = allocator::malloc(unit_cap);
  } else {
    ptr = free_cache.top();
    free_cache.pop();
  }

  {
    time_block t([](auto ms) {
      LOG_IF(INFO, ms.count() > 0) << "Tlong_emplace: " << ms.count();
    });
    auto &al = allocations.emplace(ptr);

    if (trace_allocations) {
      time_block t("trace_allocations: ");
      al.backtrace_size =
          backtrace(al.backtrace, allocation_t::backtrace_limit);
    }
  }

  return units.emplace(ptr, ptr).first->second;
}

template <typename allocator, size_t unit_cap>
void SingleDeviceMemoryManager<allocator, unit_cap>::destroy_allocation() {
  void *base = allocations.top();
  allocations.pop();

  active_bytes -= unit_cap;
  if (free_cache.size() < freed_cache_cap) {
    free_cache.push(base);
  } else {
    allocator::free(base);
  }
  units.erase(base);
}

template <typename allocator, size_t unit_cap>
void *SingleDeviceMemoryManager<allocator, unit_cap>::malloc(size_t bytes) {
  bytes = fixSize(bytes);

  if (bytes >= unit_cap) {
    void *ptr = allocator::malloc(bytes);

    {
      std::lock_guard<std::mutex> lock(m_big_units);
      big_units.emplace(ptr, bytes);
    }

    active_bytes += bytes;

    return ptr;
  }

  {
    std::lock_guard<std::mutex> lock(m);

    alloc_unit_info *info = nullptr;
    do {
      if (allocations.empty()) {
        info = &(create_allocation());
      } else {
        void *latest = allocations.top();

        auto match = units.find(latest);
        assert(match != units.end());
        info = &(match->second);

        if (info->fill + bytes > unit_cap) {
          info = &(create_allocation());
        }
      }
    } while (!info);

    void *ptr = ((void *)(((char *)info->base) + info->fill));
    info->fill += bytes;
    info->sub_units += 1;

    //#ifndef NDEBUG
    //    mappings.emplace(ptr, info->base);
    //#endif

    //    {
    //      void ** bt = new void*[32];
    //      size_t n = backtrace(bt, 31);
    //      dmappings.emplace(ptr, std::make_pair(bt, n));
    //    }

    return ptr;
  }
}

template <typename allocator, size_t unit_cap>
void SingleDeviceMemoryManager<allocator, unit_cap>::free(void *ptr) {
  if (!ptr) return;  // Ignore nullptr

  {
    std::lock_guard<std::mutex> lock(m_big_units);
    auto f = big_units.find(ptr);
    if (f != big_units.end()) {
      active_bytes -= f->second;
      big_units.erase(f);
      allocator::free(ptr);
      return;
    }
  }

  {
    std::lock_guard<std::mutex> lock(m);
    auto itabove = units.upper_bound(ptr);
    assert(itabove != units.begin() && "Unit not found!");
    auto fu = --itabove;
    assert((ptr <= ((char *)itabove->second.base) + unit_cap) &&
           "Mapping does not exist!");
    alloc_unit_info &info = fu->second;
    auto base = info.base;

    //#ifndef NDEBUG
    //    {
    //      auto f = mappings.find(ptr);
    //      if (f == mappings.end()) {
    //        for (auto &t : mappings) {
    //          std::cout << t.first << " " << t.second << std::endl;
    //        }
    //      }
    //      assert(f != mappings.end() && "Mapping does not exist!");
    //
    //      assert(base == f->second);
    //      mappings.erase(f);
    //    }
    //#endif

    //    dmappings.erase(dmappings.find(ptr));

    assert(info.sub_units > 0);
    info.sub_units = info.sub_units - 1;
    if (info.sub_units == 0) {
      while (!allocations.empty()) {
        void *tmp_base = allocations.top();
        auto it = units.find(tmp_base);
        assert(it != units.end());
        if (it->second.sub_units == 0) {
          destroy_allocation();
        } else {
          break;
        }
      }
    }
  }
}

SingleGpuMemoryManager **MemoryManager::gpu_managers;
SingleCpuMemoryManager **MemoryManager::cpu_managers;

std::mutex NUMAMemAllocator::m_sizes;
std::unordered_map<void *, size_t> NUMAMemAllocator::sizes;

std::atomic<size_t> NUMAMemAllocator::usage = 0;

void NUMAMemAllocator::freed() { assert(usage == 0); }

std::atomic<size_t> GpuMemAllocator::usage = 0;

void GpuMemAllocator::freed() { assert(usage == 0); }

void NUMAPinnedMemAllocator::freed() { NUMAMemAllocator::freed(); }
