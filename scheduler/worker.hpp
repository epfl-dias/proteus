/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef WORKER_POOL_HPP_
#define WORKER_POOL_HPP_

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
//#include "scheduler/affinity_manager.hpp"

#include "scheduler/topology.hpp"

#include "benchmarks/bench.hpp"
#include "glo.hpp"

namespace scheduler {

/*
  TODO: we need some sort of stats collector/logger/display mechanism per worker
  in order to report txn throughput after every x interval or at the very end
  when workers go down.

  if worker is not running a benchmark then throughput would be meaningless.
*/

/*enum WORKER_TYPE {
  TXN,    // dequeue task and perform those tasks
  CUSTOM  // custom function on threads (data_load, etc.)
};*/

class Worker {
  uint8_t id;
  volatile bool terminate;

  core *exec_core;

  uint64_t curr_delta;
  uint64_t prev_delta;
  uint64_t curr_txn;

  // STATS
  uint64_t num_txns;
  uint64_t num_commits;
  uint64_t num_aborts;
  uint64_t txn_start_tsc;
  // std::chrono::time_point<std::chrono::system_clock> txn_start_time;
  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
      txn_start_time;

  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
      txn_end_time;

 public:
  Worker(uint8_t id, core *exec_core)
      : id(id),
        terminate(false),
        exec_core(exec_core),
        num_txns(0),
        num_commits(0),
        num_aborts(0) {}

  void run();
  friend class WorkerPool;
};

class WorkerPool {
 protected:
 public:
  // Singleton
  static WorkerPool &getInstance() {
    static WorkerPool instance;
    return instance;
  }

  // Prevent copies
  WorkerPool(const WorkerPool &) = delete;
  void operator=(const WorkerPool &) = delete;

  WorkerPool(WorkerPool &&) = delete;
  WorkerPool &operator=(WorkerPool &&) = delete;

  void init(bench::Benchmark *txn_bench = nullptr);
  void shutdown(bool print_stats = false) { this->~WorkerPool(); }

  void start_workers(int num_workers = 1);
  void add_worker(core *exec_location);
  void remove_worker(core *exec_location);
  void print_worker_stats(bool global_only = true);

  std::vector<uint64_t> get_active_txns();
  uint64_t get_min_active_txn();

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueueTask(
      F &&f, Args &&... args);

  uint8_t size() { return workers.size(); }

 private:
  WorkerPool() { worker_counter = 0; }

  int worker_counter;
  std::atomic<bool> terminate;
  std::unordered_map<uint8_t, std::pair<std::thread *, Worker *> > workers;

  // TXN benchmark
  bench::Benchmark *txn_bench;

  // External TXN Queue
  std::queue<std::function<bool(uint64_t)> > tasks;
  std::mutex m;
  std::condition_variable cv;

  // Global Snapshotting
  // std::vector<std::vector<int> > active_worker_per_master;

  ~WorkerPool() {
    if (terminate) return;
    std::cout << "[destructor] shutting down workers" << std::endl;
    terminate = true;
    // cv.notify_all();
    for (auto &worker : workers) {
      worker.second.second->terminate = true;
      worker.second.first->join();
    }
    print_worker_stats();
    workers.clear();
  }
  friend class Worker;
};

}  // namespace scheduler

#endif /* WORKER_POOL_HPP_ */
