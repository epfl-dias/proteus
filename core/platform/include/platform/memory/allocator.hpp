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

#ifndef PROTEUS_ALLOCATOR_HPP
#define PROTEUS_ALLOCATOR_HPP

#include <exception>
#include <limits>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

namespace proteus::memory {

template <typename T>
class PinnedMemoryAllocator {
 public:
  typedef T value_type;

  PinnedMemoryAllocator() = default;
  template <class U>
  constexpr explicit PinnedMemoryAllocator(
      const PinnedMemoryAllocator<U> &o) noexcept
      : _allocMap(o._allocMap), _allocMapMutex(o._allocMapMutex) {}

  [[nodiscard]] T *allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    size_t sz = alignof(T) + (n * sizeof(T));
    auto x = MemoryManager::mallocPinned(sz);
    auto xCpy = x;
    auto aligned_ptr = std::align(alignof(T), n * sizeof(T), x, sz);
    assert(aligned_ptr != nullptr);
    {
      std::lock_guard<std::mutex> lk(*_allocMapMutex);
      _allocMap->operator[](aligned_ptr) = xCpy;
    }
    return static_cast<T *>(aligned_ptr);
  }

  void deallocate(T *mem, size_t) noexcept {
    std::lock_guard<std::mutex> lk(*_allocMapMutex);
    if (_allocMap->contains(mem)) {
      MemoryManager::freePinned(_allocMap->operator[](mem));
      _allocMap->erase(mem);
    } else {
      std::terminate();
    }
  }

 private:
  std::shared_ptr<std::map<void *, void *>> _allocMap =
      std::make_shared<std::map<void *, void *>>();
  std::shared_ptr<std::mutex> _allocMapMutex = std::make_shared<std::mutex>();

  template <typename>
  friend class PinnedMemoryAllocator;
};

template <typename T>
class ExplicitSocketPinnedMemoryAllocator {
 private:
  const int numa_memset_id;

 public:
  typedef T value_type;

  template <class U>
  constexpr explicit ExplicitSocketPinnedMemoryAllocator(
      const ExplicitSocketPinnedMemoryAllocator<U> &o) noexcept
      : numa_memset_id(o.numa_memset_id) {}

  inline explicit ExplicitSocketPinnedMemoryAllocator(int numa)
      : numa_memset_id(numa) {}

  [[nodiscard]] T *allocate(size_t n) {
    if (numa_memset_id >= 0) {
      static const auto &nodes = topology::getInstance().getCpuNumaNodes();
      set_exec_location_on_scope d{nodes[numa_memset_id]};
      return _allocator.allocate(n);
    } else {
      return _allocator.allocate(n);
    }
  }

  void deallocate(T *mem, size_t sz) { _allocator.deallocate(mem, sz); }

 private:
  PinnedMemoryAllocator<T> _allocator;
};

}  // namespace proteus::memory

#endif  // PROTEUS_ALLOCATOR_HPP
