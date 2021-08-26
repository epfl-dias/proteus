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

#ifndef PROTEUS_TXN_QUEUE_HPP
#define PROTEUS_TXN_QUEUE_HPP

#include <cassert>
#include <functional>
#include <mutex>
#include <queue>

#include "oltp/common/common.hpp"
#include "oltp/transaction/stored-procedure.hpp"
#include "oltp/transaction/transaction.hpp"

namespace txn {

enum QueueType { FIFO_QUEUE, BENCH_QUEUE };

class TxnQueue {
 public:
  TxnQueue() : type(QueueType::FIFO_QUEUE) {}

  virtual StoredProcedure pop(worker_id_t workerId,
                              partition_id_t partitionId) {
    assert(false && "unimplemented");
  }

  virtual void enqueue(StoredProcedure xact) {
    assert(false && "unimplemented");
  }

  virtual StoredProcedure popEmpty(worker_id_t workerId,
                                   partition_id_t partitionId) {
    return txn::StoredProcedure([](txn::TransactionExecutor &executor,
                                   txn::Txn &txn,
                                   void *params) { return true; });
  }

  virtual ~TxnQueue() = default;  //{ /*LOG(INFO) << "~TxnQueue()";*/ }

 public:
  QueueType type;

 private:
  //  TxnQueue():terminate(false){}
  //
  //  std::queue<txnSignature> tasks;
  //  std::atomic<bool> terminate;
  //
  //  std::mutex m;
  //  std::condition_variable cv;
};

}  // namespace txn

#endif  // PROTEUS_TXN_QUEUE_HPP
