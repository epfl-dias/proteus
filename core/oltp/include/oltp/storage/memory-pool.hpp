/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_MEMORY_POOL_HPP
#define PROTEUS_MEMORY_POOL_HPP

#include <cassert>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <stack>
#include <thread>
#include <vector>

namespace oltp::mv::memorypool {

constexpr size_t pool_size_mb = 1536;

constexpr size_t pool_size = pool_size_mb * 1024 * 1024;

template <size_t N>
class BucketMemoryPool;

template <size_t N>
class BucketMemoryPool_threadLocal {
 public:
  BucketMemoryPool_threadLocal() : destructed(false) {
    auto x = MemoryManager::mallocPinned(pool_size);
    assert(x);
    basePtr.emplace_back(x);

    size_t totalChunks = pool_size / N;
    char* ptr = reinterpret_cast<char*>(x);
    char* baseRef = ptr + pool_size;

    for (size_t i = 0; i < totalChunks; i++) {
      assert(ptr <= baseRef);
      *(reinterpret_cast<size_t*>(ptr)) = 0;
      _pool.push(reinterpret_cast<void*>(ptr));
      ptr += N;
    }
    LOG_FIRST_N(INFO, 1) << "POOL SIZE: " << _pool.size();
    // LOG(INFO) << std::this_thread::get_id() <<" - POOL SIZE: " <<
    // _pool.size();
    assert(!_pool.empty());
    BucketMemoryPool<N>::getInstance().registerThreadLocalPool(this);
  }
  ~BucketMemoryPool_threadLocal() {
    if (!destructed) {
      destruct();
      BucketMemoryPool<N>::getInstance().deregisterThreadLocalPool(this);
    }
  }

  BucketMemoryPool_threadLocal(BucketMemoryPool_threadLocal const&) = delete;
  void operator=(BucketMemoryPool_threadLocal const&) = delete;

  BucketMemoryPool_threadLocal(BucketMemoryPool_threadLocal&&) = delete;
  void operator=(BucketMemoryPool_threadLocal&&) = delete;

  inline void* allocate() {
    std::unique_lock<std::mutex> lk(_lock);
    assert(!destructed);
    if (!(_pool.empty())) {
      return this->get();
    } else {
      expand();
      LOG(INFO) << "Expanding pool: " << std::this_thread::get_id();
      return this->get();
    }
  }

  inline void free(void* pt) {
    std::unique_lock<std::mutex> lk(_lock);
    assert(!destructed);
    _pool.push(pt);
  }

 private:
  inline void* get() {
    auto ret = _pool.top();
    _pool.pop();
    return ret;
  }

  inline void expand() {
    assert(false);
    auto x = MemoryManager::mallocPinned(pool_size);
    assert(x);
    basePtr.emplace_back(x);

    size_t totalChunks = pool_size / N;
    char* ptr = reinterpret_cast<char*>(x);
    char* baseRef = ptr;

    for (size_t i = 0; i < totalChunks; i++) {
      assert(ptr <= baseRef);
      *(reinterpret_cast<size_t*>(ptr)) = 0;
      _pool.push(ptr);
      ptr += N;
    }
  }

  void destruct() {
    std::unique_lock<std::mutex> lk(_lock);
    destructed = true;

    for (auto& x : basePtr) {
      MemoryManager::freePinned(x);
    }
  }

 private:
  std::stack<void*> _pool{};
  std::mutex _lock{};

  std::vector<void*> basePtr{};

  bool destructed;

  template <size_t>
  friend class BucketMemoryPool;
};

template <size_t N>
class BucketMemoryPool {
 public:
  static inline BucketMemoryPool& getInstance() {
    static BucketMemoryPool instance;
    return instance;
  }
  ~BucketMemoryPool() {
    LOG(INFO) << "Destructing BucketMemoryPool<" << N << ">";
  }
  BucketMemoryPool(BucketMemoryPool const&) = delete;
  void operator=(BucketMemoryPool const&) = delete;

  BucketMemoryPool(BucketMemoryPool&&) = delete;
  void operator=(BucketMemoryPool&&) = delete;

  void destruct() {
    std::unique_lock<std::mutex> lk(registryLock);
    for (const auto& p : registry) {
      p->destruct();
    }
    registry.clear();
    destructed = true;
  }

 private:
  BucketMemoryPool() : destructed(false) {}

 private:
  std::vector<BucketMemoryPool_threadLocal<N>*> registry;
  std::mutex registryLock;
  void registerThreadLocalPool(BucketMemoryPool_threadLocal<N>* tpool) {
    std::unique_lock<std::mutex> lk(registryLock);
    registry.emplace_back(tpool);
  }
  void deregisterThreadLocalPool(BucketMemoryPool_threadLocal<N>* tpool) {
    std::unique_lock<std::mutex> lk(registryLock);
    registry.erase(std::remove(registry.begin(), registry.end(), tpool),
                   registry.end());
  }

 private:
  bool destructed;

  template <size_t>
  friend class BucketMemoryPool_threadLocal;
};
}  // namespace oltp::mv::memorypool

#endif  // PROTEUS_MEMORY_POOL_HPP
