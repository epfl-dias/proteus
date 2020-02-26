/*
     AEOLUS - In-Memory HTAP-Ready OLTP Engine

                            Copyright (c) 2019
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

#ifndef LOCK_HPP_
#define LOCK_HPP_

#include <atomic>
#include <cassert>

namespace lock {

struct AtomicTryLock {
  AtomicTryLock() {
    lck.store(false);
    assert(lck.is_lock_free());
  }

  inline bool try_lock() {
    bool e_false = false;
    return lck.compare_exchange_strong(e_false, true,
                                       std::memory_order_acquire);
  }

  inline void unlock() { lck.store(false, std::memory_order_release); }

  std::atomic<bool> lck;
};

}  // namespace lock

#endif /* LOCK_HPP_ */
