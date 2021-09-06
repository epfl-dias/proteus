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
    : workerID(workerId),
      thread_id(std::this_thread::get_id()),
      min_active(std::numeric_limits<xid_t>::max()),
      max_last_alive(0) {
  activeTxnPtr = static_cast<Txn *>(
      aligned_alloc(hardware_destructive_interference_size, sizeof(Txn)));

  TransactionManager::getInstance().registerTxnTable(this);
}

ThreadLocal_TransactionTable::~ThreadLocal_TransactionTable() {
  LOG(INFO) << "~ThreadLocal_TransactionTable() BEGIN TxnTable for worker_id: "
            << (uint)(this->workerID);
  assert(thread_id == std::this_thread::get_id());

  TransactionManager::getInstance().deregisterTxnTable(this);
  //  if(GcMechanism == GcTypes::SteamGC){
  //    while (!_committedTxn.empty()) {
  //      auto min_last_alive =
  //      TransactionManager::getInstance().get_last_alive();
  //      this->steamGC({min_last_alive <<
  //      TransactionManager::txnPairGen::baseShift,
  //                     min_last_alive});
  //    }
  //  }
  free(activeTxnPtr);
  LOG(INFO) << "[~ThreadLocal_TransactionTable()] END - worker_id: "
            << (uint)(this->workerID);
}

void ThreadLocal_TransactionTable::steamGC(txn::TxnTs global_min) {
  static thread_local auto &schema = storage::Schema::getInstance();
  assert(thread_id == std::this_thread::get_id());

  // inverted map for getting all cleanable for same table together.
  std::map<table_id_t, std::vector<vid_t>> cleanables;

  // committedTxn will be sorted, so
  auto it = _committedTxn.begin();

  for (; it != _committedTxn.end(); it++) {
    if (it->commit_ts <= global_min.txn_start_time) {
      for (const auto &[table_id, row_vector] : it->undoLogMap) {
        cleanables[table_id].insert(cleanables[table_id].end(),
                                    row_vector.begin(), row_vector.end());
      }
    } else {
      break;
    }
  }
  if (!cleanables.empty()) {
    schema.steamGC(cleanables, global_min);
    // erase upto the mark where it has been cleaned.
    _committedTxn.erase(_committedTxn.begin(), it);
  }
}

Txn ThreadLocal_TransactionTable::beginTxn(worker_id_t workerId,
                                           partition_id_t partitionId,
                                           bool read_only) {
  assert(thread_id == std::this_thread::get_id());
  auto tx = Txn::getTxn(workerId, partitionId, read_only);
  min_active = tx.txnTs.txn_start_time;
  return tx;
}

void ThreadLocal_TransactionTable::endTxn(Txn &txn) {
  assert(thread_id == std::this_thread::get_id());

  this->min_active = std::numeric_limits<xid_t>::max();
  auto lastAlive = this->max_last_alive.load();
  while (lastAlive < txn.txnTs.txn_start_time) {
    this->max_last_alive.compare_exchange_strong(lastAlive,
                                                 txn.txnTs.txn_start_time);
  }

  //  if constexpr (GcMechanism == GcTypes::SteamGC) {
  //    _committedTxn.emplace_back(std::move(txn));
  //  }
  // this drops a little bit (1.8 -> 1.5)
  _committedTxn.emplace_back(std::move(txn));
}

}  // namespace txn
