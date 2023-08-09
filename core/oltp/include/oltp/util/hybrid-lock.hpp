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

#ifndef PROTEUS_HYBRID_LOCK_HPP
#define PROTEUS_HYBRID_LOCK_HPP

#include <platform/util/erase-constructor-idioms.hpp>
#include <shared_mutex>

namespace lock {

class naiveRWLock {
 private:
  std::shared_mutex lk{};
  std::atomic<bool> isExclusiveLocked{};

 public:
  inline void lock_shared() { lk.lock_shared(); }
  inline void unlock_shared() { lk.unlock_shared(); }

  inline void lock() {
    lk.lock();
    isExclusiveLocked = true;
  }

  inline void unlock() {
    lk.unlock();
    isExclusiveLocked = false;
  }

  inline bool isLockedExclusive() { return isExclusiveLocked; }
};

// based on: Scalable and Robust Latches for Database Systems
template <class RWLock = naiveRWLock>
class HybridLock : proteus::utils::remove_copy_move {
 public:
  HybridLock() = default;

 private:
  RWLock rwLock;
  std::atomic<uint64_t> version;

  static constexpr bool optimisticRead = true;

 public:
  inline void acquire() { lockExclusive(); }
  inline void release() { unlockExclusive(); }

  void lockShared() { rwLock.lock_shared(); }
  void unlockShared() { rwLock.unlock_shared(); }
  void lockExclusive() { rwLock.lock(); }

  void unlockExclusive() {
    ++version;
    rwLock.unlock();
  }

  template <class lambda, class Arg>
  inline bool tryReadOptimistically(lambda &&readCallback, Arg arg) {
    if (rwLock.isLockedExclusive()) {
      return false;
    }

    auto preVersion = version.load();

    // execute read callback
    readCallback(arg);

    // was locked meanwhile
    if (rwLock.isLockedExclusive()) {
      return false;
    }

    // version still the same
    return preVersion == version.load();
  }

  template <class Lambda, class Arg>
  inline void readWithLatch(Lambda &&readCallback, Arg arg) {
    if constexpr (optimisticRead) {
      if (!tryReadOptimistically(readCallback, arg)) {
        // Fall back to pessimistic locking
        rwLock.lock_shared();
        readCallback(arg);
        rwLock.unlock_shared();
      }
    } else {
      rwLock.lock_shared();
      readCallback(arg);
      rwLock.unlock_shared();
    }
  }

  template <class Lambda, class Arg>
  inline void writeWithLatch(Lambda &&readCallback, Arg arg) {
    rwLock.lock();
    readCallback(arg);
    rwLock.unlock();
  }
};

using HybridLatch = HybridLock<naiveRWLock>;

class SpinLatch : proteus::utils::remove_copy_move {
 public:
  SpinLatch() : lk(false) {}

 private:
  std::atomic<bool> lk{};

 public:
  inline bool try_acquire() {
    bool e_false = false;
    if (lk.compare_exchange_strong(e_false, true))
      return true;
    else
      return false;
  }

  inline bool try_acquire100() {
    for (auto tries = 0; tries < 100; tries++) {
      bool e_false = false;
      if (lk.compare_exchange_strong(e_false, true)) {
        return true;
      }
    }
    return false;
  }

  inline void acquire() {
    for (int tries = 0; true; ++tries) {
      bool e_false = false;
      if (lk.compare_exchange_strong(e_false, true)) return;
      if (tries == 100) {
        tries = 0;
        sched_yield();
      }
    }
  }
  inline void release() { lk.store(false); }

  template <class Lambda, class Arg>
  inline void readWithLatch(Lambda &&readCallback, Arg arg) {
    this->acquire();
    readCallback(arg);
    this->release();
  }

  template <class Lambda, class Arg>
  inline void writeWithLatch(Lambda &&writeCallback, Arg arg) {
    this->acquire();
    writeCallback(arg);
    this->release();
  }
};

//
// template <class BaseLatch>
// class Latch{
// private:
//  BaseLatch lt;
// public:
//  template <class Lambda, class Arg>
//  inline void readWithLatch(Lambda&& readCallback, Arg arg){
//    lt.read(readCallback, arg);
//  }
//  template <class Lambda, class Arg>
//  inline void writeWithLatch(Lambda&& readCallback, Arg arg){
//    lt.write(readCallback, arg);
//  }
//
//  inline void acquire(){
//    lt.acquire();
//  }
//  inline void release(){
//    lt.release();
//  }
//
//};

}  // namespace lock

#endif  // PROTEUS_HYBRID_LOCK_HPP
