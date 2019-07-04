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
#include "scheduler/worker.hpp"
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



  uint64_t switch_master(uint8_t &curr_master){

    assert(global_conf::num_master_versions > 1);

    std::cout << "Master switch request" << std::endl;
    curr_master = this->current_master;
    /* 
          - switch master_id
          - clear the update bits of the new master. ( keep a seperate column or bit per column?)
    */

    // Before switching, clear up the new master. OR proteus do it.

    uint64_t epoch_num = scheduler::WorkerPool::getInstance().get_max_active_txn();

    ushort tmp = (current_master.load() + 1) % global_conf::num_master_versions;
    current_master.store(tmp);

    while(scheduler::WorkerPool::getInstance().is_all_worker_on_master_id(tmp) == false);

    std::cout << "Master switch completed" << std::endl;
    return epoch_num;

  }

  inline uint64_t __attribute__((always_inline)) get_next_xid(uint8_t wid) {
    
    uint32_t a, d;
    __asm __volatile("rdtsc" : "=a"(a), "=d"(d));

    return ((((uint64_t)a) | (((uint64_t)d) << 32)) & 0x00FFFFFFFFFFFFFF) |
           (((uint64_t)wid) << 56);
  }
  

  global_conf::ConcurrencyControl executor;
  uint64_t txn_start_time;
  std::atomic<ushort> current_master;

 private:
  TransactionManager()
      : txn_start_time(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count()) {
    // curr_master = 0;
    current_master = 0;
    txn_start_time = get_next_xid(0);
  }
};

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
