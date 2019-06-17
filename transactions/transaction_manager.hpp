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
#include <chrono>
#include <iostream>

#include "glo.hpp"
#include "transactions/cc.hpp"
#include "transactions/txn_utils.hpp"
//#include "utils/utils.hpp"

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

  static inline uint64_t __attribute__((always_inline)) read_tsc(uint8_t wid) {
    uint32_t a, d;
    __asm __volatile("rdtsc" : "=a"(a), "=d"(d));

    return ((((uint64_t)a) | (((uint64_t)d) << 32)) & 0x00FFFFFFFFFFFFFF) |
           (((uint64_t)wid) << 56);

    // return (((uint64_t)((d & 0x00FFFFFF) | (((uint32_t)wid) << 24))) << 32) |
    //       ((uint64_t)a);
  }
  inline uint64_t __attribute__((always_inline)) get_next_xid(uint8_t wid) {
    // Global Atomic
    // return ++g_xid;

    // WorkerID + timestamp
    // thread_local std::chrono::time_point<std::chrono::system_clock,
    //                                     std::chrono::nanoseconds>
    //    curr;

    // curr = std::chrono::system_clock::now().time_since_epoch().count();

    // uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
    //                   std::chrono::system_clock::now().time_since_epoch())
    //                   .count();
    // uint64_t cc = ((now << 8) >> 8) + (((uint64_t)wid) << 56);
    // uint64_t cc = (now & 0x00FFFFFFFFFFFFFF) + (((uint64_t)wid) << 56);

    uint64_t now = read_tsc(wid);
    // uint64_t cc = now + (((uint64_t)wid) << 56);

    // std::cout << "NOW:" << now << "|cc:" << cc << std::endl;
    return now;

    // return 0;
  }

  // void init();

  // bool execute_txn(void *stmts, uint64_t xid) {
  //   return executor.execute_txn(stmts, xid);
  //   // ,std::chrono::duration_cast<std::chrono::nanoseconds>(
  //   //     txn_start_time.time_since_epoch())
  //   //     .count());
  // }

  // void switch_master();
  // void gc();

  global_conf::ConcurrencyControl executor;
  // std::atomic<ushort> curr_master;
  // const std::chrono::time_point<std::chrono::system_clock,
  //                             std::chrono::nanoseconds>
  uint64_t txn_start_time;
  std::atomic<int> current_master;

 private:
  TransactionManager()
      : txn_start_time(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count()) {
    // curr_master = 0;
    current_master = 0;
    txn_start_time = read_tsc(0);
  }
};

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
