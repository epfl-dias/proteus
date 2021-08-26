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

#ifndef PROTEUS_TXN_TABLE_HPP
#define PROTEUS_TXN_TABLE_HPP

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <platform/memory/allocator.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/topology.hpp>
#include <shared_mutex>

#include "oltp/common/common.hpp"
#include "oltp/transaction/transaction.hpp"

namespace txn {

class alignas(hardware_destructive_interference_size)
    ThreadLocal_TransactionTable {
 public:
  ThreadLocal_TransactionTable(ThreadLocal_TransactionTable &&) = delete;
  ThreadLocal_TransactionTable &operator=(ThreadLocal_TransactionTable &&) =
      delete;
  ThreadLocal_TransactionTable(const ThreadLocal_TransactionTable &) = delete;
  ThreadLocal_TransactionTable &operator=(
      const ThreadLocal_TransactionTable &) = delete;

 public:
  void endTxn(Txn &txn);
  Txn beginTxn(worker_id_t workerId, partition_id_t partitionId,
               bool read_only = false);

  inline xid_t get_last_alive_tx() { return max_last_alive.load(); }

  inline xid_t get_min_active_tx() { return min_active.load(); }

  void steamGC(txn::TxnTs global_min);

  explicit ThreadLocal_TransactionTable(worker_id_t workerId);
  ~ThreadLocal_TransactionTable();

 private:
  const worker_id_t workerID;
  const std::thread::id thread_id;

  std::atomic<xid_t> min_active;
  std::atomic<xid_t> max_last_alive;
  Txn *activeTxnPtr;
  std::deque<Txn> _committedTxn;

  // we dont need set for activeTxn as we only have one active txn per thread in
  // the system.
  // std::set<Txn, Txn::TxnCmp, proteus::memory::PinnedMemoryAllocator<Txn>>
  // _activeTxn;
};

}  // namespace txn

#endif  // PROTEUS_TXN_TABLE_HPP
