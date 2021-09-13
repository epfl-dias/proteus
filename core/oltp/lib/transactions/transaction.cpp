/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include "oltp/transaction/transaction.hpp"

#include <memory>

#include "oltp/common/common.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace txn {

Txn::Txn(TxnTs txnTs, worker_id_t workerId, partition_id_t partitionId,
         master_version_t master_version, bool read_only)
    : txnTs(txnTs),
      worker_id(workerId),
      partition_id(partitionId),
      master_version(master_version),
      delta_version(TransactionManager::get_delta_ver(txnTs.txn_start_time)),
      read_only(read_only) {}

Txn::Txn(TxnTs txnTs, worker_id_t workerId, partition_id_t partitionId,
         bool read_only)
    : txnTs(txnTs),
      worker_id(workerId),
      partition_id(partitionId),
      master_version(
          TransactionManager::getInstance().get_current_master_version()),
      delta_version(TransactionManager::get_delta_ver(txnTs.txn_start_time)),
      read_only(read_only) {
  assert(delta_version < global_conf::num_delta_storages);
}

Txn::Txn(worker_id_t workerId, partition_id_t partitionId, bool read_only)
    : Txn({TransactionManager::getInstance().get_txnID_startTime_Pair()},
          workerId, partitionId, read_only) {}

Txn Txn::getTxn(worker_id_t workerId, partition_id_t partitionId,
                bool readOnly) {
  static thread_local auto &txnManager = TransactionManager::getInstance();
  return Txn(txnManager.get_txnID_startTime_Pair(), workerId, partitionId,
             txnManager.get_current_master_version(), readOnly);
}

std::unique_ptr<Txn> Txn::make_unique(worker_id_t workerId,
                                      partition_id_t partitionId,
                                      bool read_only) {
  static thread_local auto &txnManager = TransactionManager::getInstance();
  return std::make_unique<Txn>(
      txnManager.get_txnID_startTime_Pair(), workerId, partitionId,
      txnManager.get_current_master_version(), read_only);
}

xid_t Txn::getTxn(Txn *txnPtr, worker_id_t workerId, partition_id_t partitionId,
                  bool readOnly) {
  static thread_local auto &txnManager = TransactionManager::getInstance();
  txnPtr = new (txnPtr)
      Txn(txnManager.get_txnID_startTime_Pair(), workerId, partitionId,
          txnManager.get_current_master_version(), readOnly);
  return txnPtr->txnTs.txn_start_time;
}

}  // namespace txn
