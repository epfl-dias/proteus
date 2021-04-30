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

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>
//#include <shared_mutex>

#include "oltp/common/lock.hpp"
#include "oltp/common/spinlock.h"
#include "oltp/storage/multi-version/delta-memory-ptr.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/txn_utils.hpp"

namespace txn {

class CC_MV2PL {
 public:
  struct PRIMARY_INDEX_VAL {
    xid_t t_min;  //  | 1-byte w_id | 6 bytes xid |
    rowid_t VID;  // internal encoding defined in storage-utils.hpp
    lock::Spinlock_Weak latch;

    lock::AtomicTryLock write_lck;

    storage::DeltaList delta_list{0};
    PRIMARY_INDEX_VAL(xid_t tid, rowid_t vid) : t_min(tid), VID(vid) {}
  };

  // FIXME: neumann paper states that i can read my own version in delta, but if
  // the write-write are detected early, then how come one txn will update
  // and then end up being reading the delta?

  static inline bool __attribute__((always_inline))
  is_readable(const PRIMARY_INDEX_VAL &indexVal, const TxnTs &xact) {
    if (indexVal.t_min < xact.txn_start_time || indexVal.t_min == xact.txn_id) {
      return true;
    } else {
      return false;
    }
  }

  static inline bool __attribute__((always_inline))
  is_readable(const xid_t tmin, const TxnTs &xact, bool read_committed) {
    if (__unlikely(read_committed)) {
      if (tmin < TXN_ID_BASE && tmin < xact.txn_start_time) {
        return true;
      } else {
        return false;
      }
    } else {
      if (tmin < xact.txn_start_time || tmin == xact.txn_id) {
        return true;
      } else {
        return false;
      }
    }
  }

  static inline bool __attribute__((always_inline))
  is_readable(const xid_t tmin, const TxnTs &xact) {
    if (tmin < xact.txn_start_time || tmin == xact.txn_id) {
      return true;
    } else {
      return false;
    }
  }

  static inline bool __attribute__((always_inline))
  is_readable_committed_only(const xid_t tmin, const TxnTs &xact) {
    if (tmin < xact.txn_start_time) {
      return true;
    } else {
      return false;
    }
  }

  //  // TODO: this needs to be modified as we changed the format of TIDs
  //  static inline bool __attribute__((always_inline))
  //  is_readable(xid_t tmin, xid_t tid) {
  //    // FIXME: the following is wrong as we have encoded the worker_id in the
  //    //  txn_id. the comparison should be of the xid only and if same then
  //    idk
  //    //  because two threads can read_tsc at the same time. it doesnt mean
  //    thread
  //    //  with lesser ID comes first.
  //
  //    xid_t w_tid = tid & 0x00FFFFFFFFFFFFFFu;
  //    xid_t w_tmin = tmin & 0x00FFFFFFFFFFFFFFu;
  //
  //    assert(w_tmin != w_tid);
  //
  //    if (w_tid >= w_tmin) {
  //      return true;
  //    } else {
  //      return false;
  //    }
  //  }

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
