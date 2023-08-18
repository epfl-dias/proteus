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

#include "oltp/execution/worker.hpp"

#include <cassert>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>

#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace scheduler {

void Worker::run() {
  set_exec_location_on_scope d{
      topology::getInstance().getCores()[exec_core->index_in_topo]};

  auto* pool = &WorkerPool::getInstance();
  auto* txnManager = &txn::TransactionManager::getInstance();
  auto txnTable = txn::ThreadLocal_TransactionTable::allocate_shared(this->id);
  storage::Schema* schema = &storage::Schema::getInstance();
  schema->initMemoryPools(this->partition_id);

  if (pool->_txn_bench != nullptr) {
    this->txnQueue = dynamic_cast<txn::TxnQueue*>(
        pool->_txn_bench->getBenchQueue(this->id, this->partition_id));
  } else {
    this->txnQueue = pool->txnQueue;
  }

  // pre-run / data-loaders
  if (txnQueue->type == txn::BENCH_QUEUE && !is_hot_plugged) {
    auto* benchQueue = dynamic_cast<bench::BenchQueue*>(txnQueue);
    this->state = PRE_RUN;
    benchQueue->pre_run();
    this->state = READY;
    {
      pool->pre_barrier++;
      // warm-up txnTable with any thread_local init.
      auto tx = this->txnQueue->popEmpty(this->id, this->partition_id);
      txnManager->executeFullQueue(*txnTable, tx, this->id, this->partition_id);

      std::unique_lock<std::mutex> lk(pool->pre_m);
      pool->pre_cv.wait(lk, [pool, this] {
        return pool->pre_barrier == pool->workers.size() + 1 || terminate;
      });
    }
  }

  this->state = RUNNING;
  this->txn_start_time = std::chrono::system_clock::now();

  while (!terminate) {
    if (change_affinity) {
      assert(false && "change affinity not implemented");
    }

    if (pause) {
      assert(false && "pause not implemented");
    }

    auto tx = this->txnQueue->pop(this->id, this->partition_id);
    bool res = txnManager->executeFullQueue(*txnTable, tx, this->id,
                                            this->partition_id);

    if (__likely(res)) {
      num_commits++;
    } else {
      num_aborts++;
    }
    num_txns++;

    if (num_txns == num_iters) break;
  }

  this->txn_end_time = std::chrono::system_clock::now();

  // post-run / data-loaders
  if (txnQueue->type == txn::BENCH_QUEUE && !is_hot_plugged) {
    auto* benchQueue = dynamic_cast<bench::BenchQueue*>(txnQueue);

    this->state = POST_RUN;
    pool->post_barrier++;

    if (this->id == 0) {
      // thread-0 waits until all reaches above, then take snapshot.
      while (pool->post_barrier != pool->workers.size())
        std::this_thread::yield();

      txnManager->snapshot();
      while (schema->is_sync_in_progress()) std::this_thread::yield();

      // benchQueue->dump("post");
      pool->post_barrier++;
    }

    // all-threads wait for +1
    while (pool->post_barrier != pool->workers.size() + 1)
      std::this_thread::yield();

    benchQueue->post_run();
  }

  this->state = TERMINATED;

  if (txnQueue->type == txn::BENCH_QUEUE) {
    // delete dynamic_cast<bench::BenchQueue*>(this->txnQueue);
    pool->_txn_bench->clearBenchQueue(
        dynamic_cast<bench::BenchQueue*>(this->txnQueue));
  }
}

}  // namespace scheduler
