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

#ifndef TRANSACTION_MANAGER_HPP_
#define TRANSACTION_MANAGER_HPP_

#include <atomic>
#include <iostream>

#include "transactions/cc.hpp"
#include "transactions/txn_utils.hpp"

namespace txn {

// template <class CC = CC_GlobalLock>
class TransactionManager {
 protected:
 public:
  // Singleton
  static inline TransactionManager &getInstance() {
    static TransactionManager instance;
    return instance;
  }
  TransactionManager(TransactionManager const &) = delete;  // Don't Implement
  void operator=(TransactionManager const &) = delete;      // Don't implement

  // array of active txns -- do we really need it?

  inline int64_t get_next_xid() { return g_xid++; }

  static void init() { std::cout << "TXN Manager Init" << std::endl; }

  bool execute_txn(void *stmts) {
    if (executor.execute_txn(stmts, get_next_xid())) {
      n_txns++;
      n_commits++;
      return true;
    } else {
      n_txns++;
      n_aborts++;
      return false;
    }
  }

  CC_GlobalLock executor;

 private:
  std::atomic<uint64_t> g_xid;
  // std::vector<uint64_t> active_txns;

  // stats
  std::atomic<uint64_t> n_txns;
  std::atomic<uint64_t> n_aborts;
  std::atomic<uint64_t> n_commits;

  TransactionManager() {
    std::cout << "TransactionManager constructor\n" << std::endl;
  }
};

/*
class Transaction {
  int64_t xid;
  struct TXN *stmts;
  // array of rollback actions

  Transaction(struct TXN *stmts) {
    this->xid = TransactionManager::get_next_xid();
    this->stmts = stmts;
  }

  void execute() {

  }

  bool record_is_visible(int xmin) { return false; }
}; */

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
