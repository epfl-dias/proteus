/*
     AEOLUS - In-Memory HTAP-Ready OLTP Engine

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

#ifndef TRANSACTION_MANAGER_HPP_
#define TRANSACTION_MANAGER_HPP_

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>

#include "glo.hpp"
#include "scheduler/worker.hpp"
#include "storage/table.hpp"
#include "transactions/cc.hpp"
#include "transactions/txn_utils.hpp"
//#include "utils/utils.hpp"

#include "util/timing.hpp"

namespace txn {

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

  // TODO: move the following to snapshot manager
  bool snapshot() {
    time_block t("[TransactionManger] snapshot_: ");
    // FIXME: why get max active txn for double master while a new txn id for
    // cow-snapshot. i guess this is because for switching master we dont need
    // to pause the workers. it can happend on the fly.

    // Full barrier ( we need barier to get num_records in that snapshot)

    // std::cout << "pausing" << std::endl;
    scheduler::WorkerPool::getInstance().pause();
    //::cout << "paused" << std::endl;

    ushort snapshot_master_ver = this->switch_master();

    storage::Schema::getInstance().snapshot(this->get_next_xid(0),
                                            snapshot_master_ver);

    // storage::Schema::getInstance().snapshot(this->get_next_xid(0), 0);

    // std::cout << "resuming" << std::endl;
    scheduler::WorkerPool::getInstance().resume();
    // std::cout << "resumed" << std::endl;
    return true;
  }

  ushort switch_master() {
    ushort curr_master = this->current_master.load();

    /*
          - switch master_id
          - clear the update bits of the new master. ( keep a seperate column or
       bit per column?)
    */

    // Before switching, clear up the new master. OR proteus do it.

    ushort tmp = (curr_master + 1) % global_conf::num_master_versions;

    // std::cout << "Master switch request, from: " << (uint)curr_master
    //           << " to: " << (uint)tmp << std::endl;

    current_master.store(tmp);

    // while (scheduler::WorkerPool::getInstance().is_all_worker_on_master_id(
    //            tmp) == false)
    //   ;

    // std::cout << "Master switch completed" << std::endl;
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

#elif defined(__powerpc64__) || defined(__ppc64__)
  static __inline__ uint64_t rdtsc() {
    uint64_t c;
    asm volatile("mfspr %0, 268" : "=r"(c));
    return c;
  }

#endif

  inline uint64_t __attribute__((always_inline)) get_next_xid(uint8_t wid) {
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
