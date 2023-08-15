/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_ART_ALLOCATOR_HPP
#define PROTEUS_ART_ALLOCATOR_HPP

#include <iostream>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/util/erase-constructor-idioms.hpp>
#include <vector>

namespace indexes::art {

// FIXME: should me migrated to memory_resource/synchronized_pool_resource

template <size_t N>
class ARTAllocator : proteus::utils::remove_copy_move {
  static constexpr auto INITIAL_QTY = 32_K;

 public:
  static inline ARTAllocator<N> &getInstance() {
    static ARTAllocator<N> instance;
    return instance;
  }

  void warmup() {
    auto x = this->allocate();
    this->free(x);
  }

  inline void *allocate() {
    std::unique_lock<std::mutex> lk(_lock);
    if (unlikely(_pool.empty())) {
      expand();
    }
    auto *ret = _pool.top();
    _pool.pop();
    return ret;
  }

  inline void free(void *ptr) {
    std::unique_lock<std::mutex> lk(_lock);
    _pool.push(ptr);
  }

 private:
  std::stack<void *, std::vector<void *>> _pool{};
  std::mutex _lock{};
  std::vector<void *> basePtr{};

  inline void expand() {
    auto *mem =
        static_cast<uint8_t *>(MemoryManager::mallocPinned(N * INITIAL_QTY));
    basePtr.emplace_back((void *)mem);

    for (auto i = 0; i < INITIAL_QTY; i++) {
      _pool.push((void *)(mem + (i * N)));
    }
  }

  ARTAllocator() {
    std::unique_lock<std::mutex> lk(_lock);
    expand();
  }

  ~ARTAllocator() {
    std::unique_lock<std::mutex> lk(_lock);
    auto pool_sz = _pool.size();
    while (!(_pool.empty())) {
      _pool.pop();
    }
    for (auto &x : basePtr) {
      MemoryManager::freePinned(x);
    }
  }
};

extern template class ARTAllocator<4>;
extern template class ARTAllocator<16>;
extern template class ARTAllocator<48>;
extern template class ARTAllocator<256>;
}  // namespace indexes::art

#endif  // PROTEUS_ART_ALLOCATOR_HPP
