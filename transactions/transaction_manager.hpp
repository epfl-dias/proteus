/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

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

#ifndef TRANSACTION_MANAGER_HPP_
#define TRANSACTION_MANAGER_HPP_

#include <atomic>
#include <iostream>

#include "glo.hpp"
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

  inline int64_t get_next_xid() { return ++g_xid; }

  static void init() {
    std::cout << "Initializing Txn Manager..." << std::endl;
  }

  bool execute_txn(void *stmts) {
    return executor.execute_txn(stmts, get_next_xid());
  }

  global_conf::ConcurrencyControl executor;

 private:
  std::atomic<uint64_t> g_xid;
  // std::vector<uint64_t> active_txns;

  TransactionManager() {}
};

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
