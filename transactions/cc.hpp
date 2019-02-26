/*
                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef CC_HPP_
#define CC_HPP_

#include <iostream>
#include <mutex>
#include <vector>
#include "transactions/txn_utils.hpp"

namespace txn {

class CC_MV2PL;
class CC_GlobalLock;

class CC_GlobalLock {
 public:
  struct PRIMARY_INDEX_VAL {
    uint64_t VID;
    PRIMARY_INDEX_VAL(uint64_t vid) : VID(vid) {}
  };

  bool execute_txn(void *stmts, uint64_t xid);

  CC_GlobalLock() { std::cout << "CC Protocol: GlobalLock" << std::endl; }

 private:
  std::mutex global_lock;
};

class CC_MV2PL {
 public:
  struct PRIMARY_INDEX_VAL {
    uint64_t t_min;  // transaction id that inserted the record
    uint64_t t_max;  // transaction id that deleted the row
    uint64_t VID;    // VID of the record in memory
    std::atomic<bool> write_lck;
    std::atomic<int> read_cnt;
    // some ptr to list of versions

    PRIMARY_INDEX_VAL();
    PRIMARY_INDEX_VAL(uint64_t tid, uint64_t vid)
        : t_min(tid), t_max(0), VID(vid) {
      write_lck = 0;
      read_cnt = 0;
    }
  };

  CC_MV2PL() { std::cout << "CC Protocol: MV2PL" << std::endl; }
  bool execute_txn(void *stmts, uint64_t xid);
  inline bool is_record_visible(struct PRIMARY_INDEX_VAL *rec,
                                uint64_t tid_self);

  void gc() { modified_vids.clear(); }

 private:
  std::vector<uint64_t> modified_vids;
};

}  // namespace txn

#endif /* CC_HPP_ */
