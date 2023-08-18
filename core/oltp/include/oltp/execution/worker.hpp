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

#ifndef SCHEDULER_WORKER_POOL_HPP_
#define SCHEDULER_WORKER_POOL_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <platform/topology/topology.hpp>
#include <platform/util/erase-constructor-idioms.hpp>
#include <platform/util/percentile.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

#include "oltp/common/common.hpp"
#include "oltp/common/constants.hpp"
#include "oltp/interface/bench.hpp"

namespace scheduler {

enum WORKER_STATE { READY, RUNNING, PAUSED, TERMINATED, PRE_RUN, POST_RUN };

class Worker {
  worker_id_t id;
  volatile bool terminate;
  volatile bool pause;
  volatile bool change_affinity;
  volatile WORKER_STATE state;
  volatile bool revert_affinity;

  partition_id_t partition_id;

  const topology::core *exec_core;
  topology::core *affinity_core{};

  bool is_hot_plugged;
  volatile int64_t num_iters;

  txn::TxnQueue *txnQueue{};

  // STATS
  size_t num_txns;
  size_t num_commits;
  size_t num_aborts;
  // size_t txn_start_tsc;

  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
      txn_start_time;

  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
      txn_end_time;

 public:
  Worker(worker_id_t id, const topology::core *exec_core,
         int64_t num_iters = -1, partition_id_t partition_id = 0)
      : id(id),
        num_iters(num_iters),
        terminate(false),
        exec_core(exec_core),
        pause(false),
        change_affinity(false),
        revert_affinity(false),
        state(READY),
        partition_id(partition_id),
        is_hot_plugged(false),
        num_txns(0),
        num_commits(0),
        num_aborts(0) {
    pause = false;
  }

 private:
  void run();
  friend class WorkerPool;
};

}  // namespace scheduler

#endif /* SCHEDULER_WORKER_POOL_HPP_ */
