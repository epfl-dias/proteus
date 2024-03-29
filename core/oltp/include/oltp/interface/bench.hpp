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

#ifndef BENCH_HPP_
#define BENCH_HPP_

#include <iostream>
#include <string>

#include "oltp/common/common.hpp"
#include "oltp/transaction/transaction.hpp"
#include "oltp/transaction/txn-queue.hpp"

namespace bench {

class BenchQueue : public txn::TxnQueue {
 public:
  BenchQueue() : txn::TxnQueue(txn::BENCH_QUEUE) {}

  virtual txn::StoredProcedure pop(worker_id_t workerId,
                                   partition_id_t partitionId) override = 0;

  // NOTE: Following will run before/after the workers starts the execution. it
  // will be synchronized, i.e., worker will not start transaction until all
  // workers finish the pre-run and a worker will not start post-run unless all
  // workers are ready to start the post run. Moreover, this will not apply to
  // the hot-plugged workers.
  virtual void post_run() = 0;
  virtual void pre_run() = 0;

  virtual void dump(std::string name) = 0;

  void enqueue(txn::StoredProcedure xact) override {
    assert(false && "enqueue not possible in bench");
  }
};

class Benchmark {
 public:
  virtual void init() {}
  virtual void deinit() {}

  virtual BenchQueue* getBenchQueue(worker_id_t workerId,
                                    partition_id_t partitionId) = 0;

  virtual void clearBenchQueue(BenchQueue* pt) { delete pt; }

 protected:
  Benchmark(std::string name = "BENCH-DUMMY",
            worker_id_t num_active_workers = 1, worker_id_t num_max_workers = 1,
            partition_id_t num_partitions = 1)
      : name(name),
        num_active_workers(num_active_workers),
        num_max_workers(num_max_workers),
        num_partitions(num_partitions),
        num_readonly_worker(0) {}

 public:
  void setReadOnlyThreadCount(size_t num_ro_workers) {
    num_readonly_worker = num_ro_workers;
  }

  void incrementActiveWorker() { num_active_workers += 1; }

  auto get_n_readonly_workers() { return num_readonly_worker; }

 public:
  const std::string name;
  const worker_id_t num_max_workers;
  const partition_id_t num_partitions;

 protected:
  size_t num_readonly_worker;
  worker_id_t num_active_workers;

 public:
  virtual ~Benchmark() { LOG(INFO) << "~Benchmark"; }
};

}  // namespace bench

#endif /* BENCH_HPP_ */
