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
#include "scheduler/worker.hpp"
#include "storage/table.hpp"
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

  bool snapshot() {
    // FIXME: why get max active txn for double master while a new txn id for
    // cow-snapshot. i guess this is because for switching master we dont need
    // to pause the workers. it can happend on the fly.

#if HTAP_DOUBLE_MASTER
    uint64_t epoch_num =
        scheduler::WorkerPool::getInstance().get_max_active_txn();

    uint8_t snapshot_master_ver =
        txn::TransactionManager::getInstance().switch_master();

    storage::Schema::getInstance().snapshot(epoch_num, snapshot_master_ver);

#elif HTAP_COW
    storage::Schema::getInstance().snapshot(this->get_next_xid(0), 0);

#else
    assert(false && "Undefined snapshotting mechanism.");
#endif
    return true;
  }

  uint8_t switch_master() {
    assert(global_conf::num_master_versions > 1 &&
           "cannot switch master with master_version <= 1");

    uint8_t curr_master;
    std::cout << "Master switch request" << std::endl;
    curr_master = this->current_master;
    /*
          - switch master_id
          - clear the update bits of the new master. ( keep a seperate column or
       bit per column?)
    */

    // Before switching, clear up the new master. OR proteus do it.

    ushort tmp = (current_master.load() + 1) % global_conf::num_master_versions;
    current_master.store(tmp);

    while (scheduler::WorkerPool::getInstance().is_all_worker_on_master_id(
               tmp) == false)
      ;

    std::cout << "Master switch completed" << std::endl;
    return curr_master;
  }

#if defined(__i386__)

  static __inline__ uint64_t rdtsc(void) {
    uint64_t x;
    __asm__ volatile(".byte 0x0f, 0x31" : "=A"(x));
    return x;
  }
#elif defined(__x86_64__)

  static __inline__ uint64_t rdtsc(void) {
    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
  }

// #elif defined(__powerpc__)

//   static __inline__ uint64_t rdtsc(void) {
//     unsigned long long int result = 0;
//     unsigned long int upper, lower, tmp;
//     __asm__ volatile(
//         "0:                  \n"
//         "\tmftbu   %0           \n"
//         "\tmftb    %1           \n"
//         "\tmftbu   %2           \n"
//         "\tcmpw    %2,%0        \n"
//         "\tbne     0b         \n"
//         : "=r"(upper), "=r"(lower), "=r"(tmp));
//     result = upper;
//     result = result << 32;
//     result = result | lower;

//     return (result);
//   }
#elif defined(__powerpc64__) || defined(__ppc64__)
  static __inline__ uint64_t rdtsc() {
    uint64_t c;
    asm volatile("mfspr %0, 268" : "=r"(c));
    return c;
  }

#endif

  inline uint64_t __attribute__((always_inline)) get_next_xid(uint8_t wid) {
    // uint32_t a, d;
    // __asm __volatile("rdtsc" : "=a"(a), "=d"(d));

    // return ((((uint64_t)a) | (((uint64_t)d) << 32)) & 0x00FFFFFFFFFFFFFF) |
    //        (((uint64_t)wid) << 56);

    return (rdtsc() & 0x00FFFFFFFFFFFFFF) | (((uint64_t)wid) << 56);
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
