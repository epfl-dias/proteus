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

#include "oltp/transaction/txn-table.hpp"

#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace txn {

ThreadLocal_TransactionTable::ThreadLocal_TransactionTable(worker_id_t workerId)
    : min_active(std::numeric_limits<xid_t>::max()) {
  activeTxnPtr = static_cast<Txn*>(
      aligned_alloc(hardware_destructive_interference_size, sizeof(Txn)));
  TransactionManager::getInstance().registerTxnTable(this);
}

ThreadLocal_TransactionTable::~ThreadLocal_TransactionTable() {
  free(activeTxnPtr);
}

Txn& ThreadLocal_TransactionTable::beginTxn(worker_id_t workerId,
                                            partition_id_t partitionId,
                                            bool read_only) {
  min_active = Txn::getTxn(activeTxnPtr, workerId, partitionId, read_only);

  // static thread_local auto &txnManager = TransactionManager::getInstance();
  // TxnTs ts(txnManager.get_txnID_startTime_Pair());
  // min_active = ts.txn_start_time;

  // this->activeTxnPtr = new (activeTxnPtr) Txn(ts, workerId, partitionId,
  // read_only);

  return (*activeTxnPtr);

  //
  //  //std::unique_lock<std::mutex> lk(changeLk);
  //
  //  TxnTs ts(TransactionManager::getInstance().get_txnID_startTime_Pair());
  //
  //  auto mapIt = active_txn.emplace_hint(
  //      active_txn.end(), ts, Txn{ts, workerId, partitionId, read_only});
  //
  //  //auto minActiveTxn = min_active.load();
  //
  //  // Following cannot be true as every new transaction wont be minimum.
  //  // mapIt->first.txn_start_time < minActiveTxn
  //
  //  // Following will be true when first transaction enters the system.
  ////  if (minActiveTxn == 0) {
  ////    min_active.compare_exchange_strong(minActiveTxn, ts.txn_start_time);
  ////  }
  //  min_active = active_txn.cbegin()->first.txn_start_time;
  //
  //
  //  return mapIt->second;
}

Txn& ThreadLocal_TransactionTable::endTxn(Txn& txn) {
  // std::unique_lock<std::mutex> lk(changeLk);

  this->min_active = std::numeric_limits<xid_t>::max();
  return (*activeTxnPtr);
  //
  //  auto ex = active_txn.extract(txn.txnTs);
  //  auto it = finished_txn.insert(std::move(ex));
  //  assert(it.inserted);
  //  // update the min active
  //
  ////  if (__likely(!active_txn.empty())) {
  ////    auto minActiveTxn = min_active.load();
  ////    if (minActiveTxn < active_txn.cbegin()->first.txn_start_time) {
  ////      min_active = active_txn.cbegin()->first.txn_start_time;
  ////    }
  ////  } else {
  ////    min_active = 0;
  ////  }
  //  if(!active_txn.empty()){
  //    min_active = active_txn.cbegin()->first.txn_start_time;
  //  } else {
  //    min_active = std::numeric_limits<xid_t>::max();
  //  }
  //
  //  return it.position->second;
}

}  // namespace txn
