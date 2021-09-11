/*
     Proteus -- High-performance query processing on heterogeneous hardware.

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

#ifndef CONCURRENCY_CONTROL_HPP_
#define CONCURRENCY_CONTROL_HPP_

constexpr bool optimistic_read = true;

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "oltp/common/lock.hpp"
#include "oltp/common/spinlock.h"
#include "oltp/storage/multi-version/delta-memory-ptr.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/txn_utils.hpp"
#include "oltp/util/hybrid-lock.hpp"

namespace txn {

class CC_MV2PL {
 public:
  class PRIMARY_INDEX_VAL {
   public:
    xid_t t_min;  //  | 1-byte w_id | 6 bytes xid |
    rowid_t VID;  // internal encoding defined in storage-utils.hpp
    // lock::Spinlock_Weak latch;
    // lock::HybridLatch latch;
    lock::SpinLatch latch;
    lock::AtomicTryLock write_lck;

    storage::DeltaPtr delta_list{0};
    PRIMARY_INDEX_VAL(xid_t tid, rowid_t vid) : t_min(tid), VID(vid) {}

    PRIMARY_INDEX_VAL(PRIMARY_INDEX_VAL &&) = delete;
    PRIMARY_INDEX_VAL &operator=(PRIMARY_INDEX_VAL &&) = delete;

    PRIMARY_INDEX_VAL(const PRIMARY_INDEX_VAL &) = delete;
    PRIMARY_INDEX_VAL &operator=(const PRIMARY_INDEX_VAL &) = delete;

    template <class lambda>
    inline void writeWithLatch(lambda &&func, Txn &txn, table_id_t table_id) {
      latch.template writeWithLatch(func, this);
      txn.undoLogVector.emplace_back(
          storage::StorageUtils::get_row_uuid(table_id, VID));
    }

    template <class lambda>
    inline void readWithLatch(lambda &&func) {
      latch.template readWithLatch(func, this);
    }

    template <class lambda>
    inline void withLatch(lambda &&func) {
      this->latch.acquire();
      func(this);
      this->latch.release();
    }
  };

  static inline bool __attribute__((always_inline))
  is_readable(const PRIMARY_INDEX_VAL &indexVal, const TxnTs &xact) {
    if (indexVal.t_min <= xact.txn_start_time ||
        indexVal.t_min == xact.txn_id) {
      return true;
    } else {
      return false;
    }
  }

  static inline bool __attribute__((always_inline))
  is_readable(const xid_t tmin, const TxnTs &xact) {
    if (tmin <= xact.txn_start_time || tmin == xact.txn_id) {
      return true;
    } else {
      return false;
    }
  }

  static inline void __attribute__((always_inline)) release_locks(
      std::vector<CC_MV2PL::PRIMARY_INDEX_VAL *> &hash_ptrs_lock_acquired) {
    for (auto c : hash_ptrs_lock_acquired) c->write_lck.unlock();
  }

  static inline void __attribute__((always_inline))
  release_locks(CC_MV2PL::PRIMARY_INDEX_VAL **hash_ptrs, uint count) {
    for (int i = 0; i < count; i++) {
      hash_ptrs[i]->write_lck.unlock();
    }
  }
};

}  // namespace txn

#endif /* CONCURRENCY_CONTROL_HPP_ */
