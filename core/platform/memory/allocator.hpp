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

#include <limits>

#include "memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

namespace proteus::memory {

template <typename T>
class ExplicitSocketPinnedMemoryAllocator {
 private:
  const int numa_memset_id;

 public:
  typedef T value_type;

  inline explicit ExplicitSocketPinnedMemoryAllocator(int numa)
      : numa_memset_id(numa) {}

  [[nodiscard]] T *allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    if (numa_memset_id >= 0) {
      static const auto &nodes = topology::getInstance().getCpuNumaNodes();
      set_exec_location_on_scope d{nodes[numa_memset_id]};

      return static_cast<T *>(MemoryManager::mallocPinned(n * sizeof(T)));

    } else {
      return static_cast<T *>(MemoryManager::mallocPinned(n * sizeof(T)));
    }
  }

  void deallocate(T *mem, size_t) noexcept { MemoryManager::freePinned(mem); }
};

template <typename T>
class PinnedMemoryAllocator {
 public:
  typedef T value_type;

  inline explicit PinnedMemoryAllocator() = default;

  [[nodiscard]] T *allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    return static_cast<T *>(MemoryManager::mallocPinned(n * sizeof(T)));
  }

  void deallocate(T *mem, size_t) noexcept { MemoryManager::freePinned(mem); }
};

}  // namespace proteus::memory

#endif  // PROTEUS_ALLOCATOR_HPP
