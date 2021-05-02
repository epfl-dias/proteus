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

#ifndef TRANSACTION_MANAGER_HPP_
#define TRANSACTION_MANAGER_HPP_

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <platform/util/timing.hpp>

#include "oltp/common/constants.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/concurrency-control/concurrency-control.hpp"
#include "oltp/transaction/txn_utils.hpp"

namespace txn {

class TransactionManager {
 public:
  static inline TransactionManager &getInstance() {
    static TransactionManager instance;
    return instance;
  }
  TransactionManager(TransactionManager const &) = delete;  // Don't Implement
  void operator=(TransactionManager const &) = delete;      // Don't implement

  // TODO: move the following to snapshot manager
  bool snapshot_twinColumn() {
    time_block t("[TransactionManger] snapshot_: ");
    /* FIXME: full-barrier is needed only to get num_records of each relation
     *        which is implementation specific. it can be saved by the
     *        storage-layer whenever it sees the master-version different from
     *        the last or previous write.
     * */

    scheduler::WorkerPool::getInstance().pause();
    ushort snapshot_master_ver = this->switch_master();
    storage::Schema::getInstance().twinColumn_snapshot(this->get_next_xid(0),
                                                       snapshot_master_ver);
    // storage::Schema::getInstance().snapshot(this->get_next_xid(0), 0);
    scheduler::WorkerPool::getInstance().resume();
    return true;
  }

  inline auto get_snapshot_masterVersion(xid_t epoch) {
    assert(epoch_to_master_ver_map.contains(epoch));
    return epoch_to_master_ver_map[epoch];
  }

  bool snapshot() {
    time_block t("[TransactionManger] snapshot_: ");

    scheduler::WorkerPool::getInstance().pause();
    auto snapshot_master_ver = this->switch_master();
    auto snapshot_epoch = this->get_next_xid(0);
    epoch_to_master_ver_map.emplace(snapshot_epoch, snapshot_master_ver);
    //    storage::Schema::getInstance().twinColumn_snapshot(this->get_next_xid(0),
    //                                                       snapshot_master_ver);

    storage::Schema::getInstance().snapshot(snapshot_epoch, nullptr);
    scheduler::WorkerPool::getInstance().resume();
    return true;
  }

  master_version_t switch_master() {
    master_version_t curr_master = this->current_master.load();

    /*
          - switch master_id
          - clear the update bits of the new master. ( keep a seperate column or
       bit per column?)
    */

    // Before switching, clear up the new master. OR proteus do it.

    master_version_t tmp = (curr_master + 1) % global_conf::num_master_versions;
    current_master.store(tmp);

    // while (scheduler::WorkerPool::getInstance().is_all_worker_on_master_id(
    //            tmp) == false)
    //   ;

    // std::cout << "Master switch completed" << std::endl;
    return curr_master;
  }

  inline xid_t __attribute__((always_inline)) get_next_xid(worker_id_t wid) {
    // return (rdtsc() & 0x00FFFFFFFFFFFFFF) | (((uint64_t)wid) << 56u);
    // return (commit_ts_gen++ & 0x00FFFFFFFFFFFFFF) | (((uint64_t)wid) << 56u);
    return commit_ts_gen++;
  }

  [[maybe_unused]] inline std::pair<xid_t, xid_t> get_txnID_startTime_Pair() {
    if (txn_id_gen <= UINT64_C(134217727)) {
      LOG(INFO) << "FUCK" << txn_id_gen;
    }
    assert(txn_id_gen > UINT64_C(134217727));
    return std::make_pair(txn_id_gen++, commit_ts_gen++);
  }

  //  [[maybe_unused]] inline xid_t get_start_ts() { return txn_id_gen++; }

  [[maybe_unused]] inline xid_t get_commit_ts() { return commit_ts_gen++; }

  inline master_version_t get_current_master_version() {
    return current_master.load();
  }

 private:
  TransactionManager() : txn_start_time(rdtsc() & 0x00FFFFFFFFFFFFFF) {
    /*: txn_start_time(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()) {*/

    current_master = 0;
    txn_id_gen = std::pow(2, 27);
    LOG(INFO) << "Starting txn_id_gen: " << txn_id_gen;
    commit_ts_gen = 10;
  }

  std::map<xid_t, master_version_t> epoch_to_master_ver_map{};
  std::atomic<size_t> txn_id_gen{};
  std::atomic<size_t> commit_ts_gen{};

  std::atomic<master_version_t> current_master;

 public:
  const size_t txn_start_time;
};

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
