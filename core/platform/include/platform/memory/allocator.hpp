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
  using type = PinnedMemoryAllocator<T>;
  using other = PinnedMemoryAllocator<T>;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  PinnedMemoryAllocator() = default;

  // NOTE: do not put explicit in the following constructor, otherwise rebind
  // fails for some data structures, including libcuckoo::cuckoohash_map
  template <class U>
  constexpr PinnedMemoryAllocator(const PinnedMemoryAllocator<U> &o) noexcept {}

 private:
  using metadata_t = T *;
  static constexpr size_t requiredAlignment =
      std::max(alignof(T), alignof(metadata_t));

  template <typename K>
  bool aligned_as(const void *mem) {
    return reinterpret_cast<uintptr_t>(mem) % alignof(K) == 0;
  }

 public:
  template <class U>
  using rebind = PinnedMemoryAllocator<U>;

  /**
   * @param n requested allocation size
   * @return An aligned pointer to the allocated memory
   */
  [[nodiscard]] T *allocate(size_t n) {
    if (n >
        (std::numeric_limits<size_t>::max() - sizeof(metadata_t) - sizeof(T)) /
            sizeof(T)) {
      throw std::bad_alloc();
    }
    auto sz = (requiredAlignment - 1) + (n * sizeof(T)) + sizeof(metadata_t);
    auto allocBase = MemoryManager::mallocPinned(sz);
    auto spaceBase = static_cast<void *>(
        /* Note that doing a reinterpret to metadata_t * and +1 wouldn't
         * work here, as allocBase may have any alignment making the cast
         * invalid
         */
        reinterpret_cast<char *>(allocBase) + sizeof(metadata_t));
    auto spaceSize = sz - sizeof(metadata_t);
    auto aligned_ptr =
        std::align(requiredAlignment, n * sizeof(T), spaceBase, spaceSize);
    assert(aligned_ptr && "Insufficient space calculation");
    assert(aligned_as<T>(aligned_ptr));
    assert(aligned_as<metadata_t>(aligned_ptr));

    *(static_cast<metadata_t *>(aligned_ptr) - 1) =
        static_cast<metadata_t>(allocBase);

    return static_cast<T *>(aligned_ptr);
  }

  void deallocate(T *mem, size_t) noexcept {
    assert(reinterpret_cast<uintptr_t>(mem) % requiredAlignment == 0);
    auto allocBase = *(reinterpret_cast<metadata_t *>(mem) - 1);
    assert(allocBase < mem);  // not equal, as we have the metadata
    assert(reinterpret_cast<const char *>(mem) -
               reinterpret_cast<const char *>(allocBase) <
           sizeof(metadata_t) + requiredAlignment);
    MemoryManager::freePinned(allocBase);
  }

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

  /**
   *
   * @param numa index in topology of the cpunumanode to use for this allocator
   */
  inline explicit ExplicitSocketPinnedMemoryAllocator(int numa)
      : numa_memset_id(numa) {}

  /**
   * @param n requested allocation size
   * @return An aligned pointer to the allocated memory
   */
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
