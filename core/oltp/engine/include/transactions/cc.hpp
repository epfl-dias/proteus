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

#ifndef CC_HPP_
#define CC_HPP_

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "storage/multi-version/delta_storage.hpp"
#include "transactions/txn_utils.hpp"
#include "utils/lock.hpp"
#include "utils/spinlock.h"

namespace txn {

class CC_MV2PL;

#define CC_extract_offset(v) (v & 0x000000FFFFFFFFFFu)
#define CC_extract_pid(v) ((v & 0x0000FF0000000000u) >> 40u)
#define CC_extract_m_ver(v) ((v & 0x00FF000000000000u) >> 48u)
//#define CC_extract_delta_id(v) ((v & 0xFF00000000000000u) >> 56u)
// #define CC_gen_vid(v, p, m, d)                                 \
//   ((v & 0x000000FFFFFFFFFF) | ((p & 0x00FF) << 40) | \
//    ((m & 0x00FF) << 48)((d & 0x00FF) << 56))

class CC_MV2PL {
 public:
  struct __attribute__((packed)) PRIMARY_INDEX_VAL {
    uint64_t t_min;  //  | 1-byte w_id | 6 bytes xid |
    uint64_t VID;    // | 1-byte delta-id | 1-byte master_ver | 1-byte
                     // partition_id | 5-byte VID |
    lock::Spinlock_Weak latch;
    lock::AtomicTryLock write_lck;

    storage::DeltaList delta_list;

    //    void *delta_ver;       // delta-list
    //    size_t delta_ver_tag;  // 4 byte delta_idx| 4-byte delta-tag

    PRIMARY_INDEX_VAL(uint64_t tid, uint64_t vid) : t_min(tid), VID(vid) {}
  };

  // TODO: this needs to be modified as we changed the format of TIDs
  static inline bool __attribute__((always_inline))
  is_readable(uint64_t tmin, uint64_t tid) {
    // FIXME: the following is wrong as we have encoded the worker_id in the
    // txn_id. the comparision should be of the xid only and if same then idk
    // because two threads can read_tsc at the same time. it doesnt mean thread
    // with lesser ID comes first.

    uint64_t w_tid = tid & 0x00FFFFFFFFFFFFFF;
    uint64_t w_tmin = tmin & 0x00FFFFFFFFFFFFFF;

    assert(w_tmin != w_tid);

    if (w_tid >= w_tmin) {
      return true;
    } else {
      return false;
    }
  }

  static inline bool is_mv() { return true; }
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

#endif /* CC_HPP_ */
