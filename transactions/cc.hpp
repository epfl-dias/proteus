/*
                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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
#include "transactions/txn_utils.hpp"

namespace txn {

class CC_MV2PL;
class CC_GlobalLock;

class CC_GlobalLock {
 public:
  struct PRIMARY_INDEX_VAL {
    uint64_t VID;
  };

  bool execute_txn(void *stmts, uint64_t xid);

  CC_GlobalLock() { std::cout << "CC Protocol: GlobalLock" << std::endl; }

 private:
  std::mutex global_lock;
};

class CC_TupleLock {};

template <class MV_PROTOCOL = CC_MV2PL>
class CC_MVCC {};

class CC_MV2PL {};

}  // namespace txn

#endif /* CC_HPP_ */
