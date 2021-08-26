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

#include "oltp/transaction/transaction_manager.hpp"

#include "oltp/transaction/txn-executor.hpp"

namespace txn {

bool TransactionManager::executeFullQueue(StoredProcedure& storedProcedure,
                                          worker_id_t workerId,
                                          partition_id_t partitionId) {
  static thread_local TransactionExecutor executor;
  static thread_local ThreadLocal_TransactionTable txnTable(workerId);
  static thread_local storage::Schema& schema = storage::Schema::getInstance();

  // if current_delta_id == -1, means not registered anywhere
  // if exist, that is, this thread is registered somewhere.

  // begin
  auto txn = txnTable.beginTxn(workerId, partitionId, storedProcedure.readOnly);

  // LOG(INFO) << "W " << (uint)workerId << " - " << txn.txnTs.txn_start_time;

  //  LOG(INFO) << "Starting Txn: " << txn.txnTs.txn_start_time;
  if constexpr (GcMechanism == GcTypes::OneShot) {
    static thread_local int32_t current_delta_id = -1;
    auto txn_st_time = txn.txnTs.txn_start_time;

    // FIXME: what about readOnly? lets ignore them for now.

    if (__unlikely(current_delta_id == -1)) {
      // first-time
      current_delta_id = txn.delta_version;
      schema.add_active_txn(current_delta_id, txn_st_time, workerId);
    } else {
      if (current_delta_id != txn.delta_version) {
        // we need to switch delta
        // switching means unregister from old, and add to new.
        schema.switch_delta(current_delta_id, txn.delta_version, txn_st_time,
                            workerId);
        current_delta_id = txn.delta_version;
      } else {
        // just update the max_active_epoch.
        schema.update_delta_epoch(current_delta_id, txn_st_time, workerId);
      }
    }
  }

  // execute
  bool success = storedProcedure.tx(executor, txn, storedProcedure.params);

  // end
  // auto& finishedTxn =
  txnTable.endTxn(txn);

  // for STEAM-BASIC, perform thread-local GC.
  // for Committed txns, if they have fall behind the global min, clean them.

  if constexpr (GcMechanism == GcTypes::SteamGC) {
    // what if there is a single txn in the system only. then min becomes
    // numeric_limitsx<max>
    // auto min = this->get_min_activeTxn();
    if (!txn.undoLogMap.empty()) {
      auto min = this->get_last_alive();
      txnTable.steamGC({min << txnPairGen::baseShift, min});
    }
  }

  return success;
}

bool TransactionManager::execute(StoredProcedure& storedProcedure,
                                 worker_id_t workerId,
                                 partition_id_t partitionId) {
  static thread_local TransactionExecutor executor;
  static thread_local ThreadLocal_TransactionTable txnTable(workerId);
  static thread_local storage::Schema& schema = storage::Schema::getInstance();

  //  // begin
  //  auto& txn =
  //      txnTable.beginTxn(workerId, partitionId, storedProcedure.readOnly);
  //
  //  auto dver = txn.delta_version;
  //  auto txn_st_time = txn.txnTs.txn_start_time;
  //
  //  if (__likely(!storedProcedure.readOnly))
  //    schema.add_active_txn(dver, txn_st_time, workerId);
  //
  //  // execute
  //  bool success = storedProcedure.tx(executor, txn, storedProcedure.params);
  //
  //  // end
  //
  //  auto& finishedTxn = txnTable.endTxn(txn);
  //
  //  if (__likely(!storedProcedure.readOnly))
  //    schema.remove_active_txn(dver, txn_st_time, workerId);
  //
  //  return success;

  return false;
}

/*
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

uint64_t rdtscl() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}*/

}  // namespace txn
