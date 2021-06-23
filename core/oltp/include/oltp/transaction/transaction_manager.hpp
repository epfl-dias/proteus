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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <platform/memory/allocator.hpp>
#include <platform/util/timing.hpp>

#include "oltp/common/constants.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/txn-table.hpp"
#include "oltp/transaction/txn_utils.hpp"

namespace txn {

class TransactionManager {
 public:
  static inline TransactionManager& getInstance() {
    static TransactionManager instance;
    return instance;
  }
  TransactionManager(TransactionManager const&) = delete;  // Don't Implement
  void operator=(TransactionManager const&) = delete;      // Don't implement

  //  static inline auto extractEpoch(xid_t xid) {
  //    // epochs are time-based or txnId rolling basis?
  //    // if time-based, then we need to account for txnTime, not txnId.
  //    // if rolling basis, then it is not really an epoch, is it?
  //
  //    // TxnID basis:
  //    // 10 bit means change delta every 1024 transactions.
  //    return xid >> 10;
  //  }

  static constexpr auto delta_switch_bit = 10;
  static constexpr auto extract_epoch(xid_t xid) {
    return (xid >> (delta_switch_bit)) << delta_switch_bit;
    // return xid;
  }
  static constexpr delta_id_t get_delta_ver(xid_t xid) {
    // return extract_epoch(xid) % global_conf::num_delta_storages;
    return (xid >> (delta_switch_bit)) % global_conf::num_delta_storages;
  }

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

  inline master_version_t get_current_master_version() {
    return current_master;
  }

  static bool execute(StoredProcedure& storedProcedure, worker_id_t workerId,
                      partition_id_t partitionId);
  static bool executeFullQueue(StoredProcedure& storedProcedure,
                               worker_id_t workerId,
                               partition_id_t partitionId);

  const auto& getTxnTables() { return registry; }

  void registerTxnTable(ThreadLocal_TransactionTable* tbl) {
    registry.emplace_back(tbl);
  }

  [[nodiscard]] xid_t get_min_activeTxn() const {
    xid_t min = std::numeric_limits<xid_t>::max();
    for (const auto& t : registry) {
      auto x = t->get_min_active_tx();
      if (x < min) {
        min = x;
      }
    }
    return min;
  }

 private:
  TransactionManager() : current_master(0) {
    //, commit_ts_gen(1), txn_id_gen(std::pow(2, 27))
    //: txn_start_time(rdtsc() & 0x00FFFFFFFFFFFFFF)
    /*: txn_start_time(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()) {*/
    registry.reserve(topology::getInstance().getCoreCount());
  }

  std::atomic<master_version_t> current_master;
  std::map<xid_t, master_version_t> epoch_to_master_ver_map{};
  std::vector<ThreadLocal_TransactionTable*> registry;

  class txnPairGen {
    //    std::atomic<size_t> txn_id_gen{};
    //    std::atomic<size_t> commit_ts_gen{};
    std::atomic<size_t> gen{};
    static constexpr auto baseShift = 27u;

   public:
    txnPairGen() : gen(1) {}
    //, commit_ts_gen(1), txn_id_gen(std::pow(2, 27)){}

    inline TxnTs __attribute__((always_inline)) get_txnID_startTime_Pair() {
      auto x = gen.fetch_add(1, std::memory_order_relaxed);

      return {x << baseShift, x};
      // return {txn_id_gen++, commit_ts_gen++};
    }
    inline xid_t __attribute__((always_inline)) get_commit_ts() {
      //  return commit_ts_gen++;
      return gen.fetch_add(1, std::memory_order_relaxed);
    }
  };

  alignas(hardware_destructive_interference_size) txnPairGen txn_gen;

 private:
  inline TxnTs __attribute__((always_inline)) get_txnID_startTime_Pair() {
    // return std::make_pair(txn_id_gen++, commit_ts_gen++);
    return txn_gen.get_txnID_startTime_Pair();
  }

  inline xid_t __attribute__((always_inline)) get_commit_ts() {
    // return commit_ts_gen++;
    return txn_gen.get_commit_ts();
  }

  inline xid_t __attribute__((always_inline)) get_next_xid(worker_id_t wid) {
    // return (rdtsc() & 0x00FFFFFFFFFFFFFF) | (((uint64_t)wid) << 56u);
    // return (commit_ts_gen++ & 0x00FFFFFFFFFFFFFF) | (((uint64_t)wid) << 56u);
    // return commit_ts_gen++;
    return txn_gen.get_commit_ts();
  }

 private:
  friend class Txn;
  friend class TransactionExecutor;
  friend class ThreadLocal_TransactionTable;
};

};  // namespace txn

#endif /* TRANSACTION_MANAGER_HPP_ */
