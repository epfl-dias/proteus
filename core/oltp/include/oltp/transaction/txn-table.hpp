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
#include <platform/util/erase-constructor-idioms.hpp>
#include <shared_mutex>
#include <type_traits>

#include "oltp/common/common.hpp"
#include "oltp/transaction/transaction.hpp"

namespace txn {

class ThreadLocal_TransactionTable : proteus::utils::remove_copy_move {
 public:
  explicit ThreadLocal_TransactionTable(worker_id_t workerId);
  ~ThreadLocal_TransactionTable();

  template <class Allocator = proteus::memory::PinnedMemoryAllocator<
                ThreadLocal_TransactionTable>>
  static auto allocate_shared(worker_id_t workerId) {
    return std::move(std::allocate_shared<ThreadLocal_TransactionTable>(
        Allocator(), workerId));
  }

 public:
  inline Txn beginTxn(worker_id_t workerId, partition_id_t partitionId,
                      bool read_only = false) {
    assert(thread_id == std::this_thread::get_id());
    // FIXME: instead of stack obj, create unique_ptr for easy move later.
    auto tx = Txn::getTxn(workerId, partitionId, read_only);
    min_active = tx.txnTs.txn_start_time;
    return tx;
  }

  const Txn &endTxn(Txn &txn);
  void steamGC(txn::TxnTs global_min);

  inline xid_t get_last_alive_tx() const { return max_last_alive.load(); }
  inline xid_t get_min_active_tx() { return min_active.load(); }

 private:
  const worker_id_t workerID;
  const std::thread::id thread_id;

  std::atomic<xid_t> min_active;
  std::atomic<xid_t> max_last_alive;
  std::mutex gcLock;
  std::deque<Txn> _committedTxn;

  // we dont need set for activeTxn as we only have one active txn per thread in
  // the system.
  // std::set<Txn, Txn::TxnCmp, proteus::memory::PinnedMemoryAllocator<Txn>>
  // _activeTxn;
};

}  // namespace txn

#endif  // PROTEUS_TXN_TABLE_HPP
