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

#include "oltp/common/common.hpp"
#include "oltp/transaction/transaction.hpp"

namespace txn {

// BlockManager::block_size
class alignas(hardware_destructive_interference_size)
    ThreadLocal_TransactionTable {
 public:
  Txn& endTxn(Txn& txn);
  Txn& beginTxn(worker_id_t workerId, partition_id_t partitionId,
                bool read_only = false);

  //  auto size()const{
  //    return active_txn.size();
  //  }

  inline xid_t get_min_active_tx() const {
    return min_active.load();
    //    std::unique_lock<std::mutex> lk(changeLk);
    //    if(!active_txn.empty()){
    //      return active_txn.cbegin()->first.txn_start_time;
    //    } else {
    //      return 0;
    //    }
  }

  explicit ThreadLocal_TransactionTable(worker_id_t workerId);
  ~ThreadLocal_TransactionTable();

 private:
  std::atomic<xid_t> min_active;
  std::mutex changeLk{};
  //  std::set<Txn, Txn::TxnCmp, proteus::memory::PinnedMemoryAllocator<Txn>>
  //  active_txn_map{}; std::set<Txn, Txn::TxnCmp,
  //  proteus::memory::PinnedMemoryAllocator<Txn>> committed_txn_map{};
  Txn* activeTxnPtr;
  std::map<TxnTs, Txn, TxnTs::TxnTsCmp> active_txn;
  std::map<TxnTs, Txn, TxnTs::TxnTsCmp> finished_txn;
};

}  // namespace txn

#endif  // PROTEUS_TXN_TABLE_HPP
