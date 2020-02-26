/*
     AEOLUS - In-Memory HTAP-Ready OLTP Engine

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
#include <queue>
#include <thread>
#include <unordered_map>
//#include "scheduler/affinity_manager.hpp"

#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/topology.hpp"

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

// struct STATS {
//   double id;
//   double tps;
//   double num_commits;
//   double num_aborts;
//   double num_txns;
//   double runtime_duration;
// };

// struct GLOBAL_STATS {
//   std::vector<struct STATS> global_stats;
//   std::vector<struct STATS> worker_stats;
//   std::vector<struct STATS> socket_stats;
// };

enum WORKER_STATE { READY, RUNNING, PAUSED, TERMINATED, PRERUN, POSTRUN };

class Worker {
  uint8_t id;
  volatile bool terminate;
  volatile bool pause;
  volatile bool change_affinity;
  volatile WORKER_STATE state;
  volatile bool revert_affinity;

  uint partition_id;

  const core *exec_core;
  core *affinity_core;

  uint64_t curr_txn;
  uint64_t prev_delta;
  uint64_t curr_delta;
  volatile ushort curr_master;
  bool is_hotplugged;
  volatile int64_t num_iters;

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
  Worker(uint8_t id, const core *exec_core, int64_t num_iters = -1,
         uint partition_id = 0)
      : id(id),
        num_iters(num_iters),
        terminate(false),
        exec_core(exec_core),
        pause(false),
        change_affinity(false),
        revert_affinity(false),
        state(READY),
        partition_id(partition_id),
        is_hotplugged(false),
        num_txns(0),
        num_commits(0),
        num_aborts(0) {
    pause = false;
  }

 private:
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

  void init(bench::Benchmark *txn_bench = nullptr, uint num_workers = 1,
            uint num_partitions = 1, uint worker_sched_mode = 0,
            int num_iter_per_worker = -1, bool elastic_workload = false);
  void shutdown(bool print_stats = false);
  void shutdown_manual();

  void start_workers();
  void add_worker(const core *exec_location, short partition_id = -1);
  void remove_worker(const core *exec_location);
  void migrate_worker(bool return_back = false);

  const std::vector<uint> &scale_down(uint num_cores = 1);
  void scale_back();

  void print_worker_stats(bool global_only = true);
  void print_worker_stats_diff();
  std::pair<double, double> get_worker_stats_diff(bool print = false);

  std::vector<uint64_t> get_active_txns();
  uint64_t get_min_active_txn();
  uint64_t get_max_active_txn();
  bool is_all_worker_on_master_id(ushort master_id);

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueueTask(
      F &&f, Args &&... args);

  uint8_t size() { return workers.size(); }
  std::string get_benchmark_name() { return this->txn_bench->name; }
  void pause();
  void resume();

 private:
  WorkerPool() {
    worker_counter = 0;
    terminate = false;
    proc_completed = false;
    pre_barrier = false;
  }

  int worker_counter;
  std::atomic<bool> terminate;
  std::atomic<bool> proc_completed;

  std::atomic<uint> pre_barrier;
  std::condition_variable pre_cv;
  std::mutex pre_m;

  std::unordered_map<uint, std::pair<std::thread *, Worker *>> workers;
  std::vector<uint> elastic_set;

  uint num_iter_per_worker;
  uint worker_sched_mode;
  uint num_partitions;
  bool elastic_workload;

  // Stats

  std::vector<std::chrono::time_point<std::chrono::system_clock,
                                      std::chrono::nanoseconds>>
      prev_time_tps;

  std::vector<double> prev_sum_tps;

  // TXN benchmark
  bench::Benchmark *txn_bench;

  // External TXN Queue
  std::queue<std::function<bool(uint64_t)>> tasks;
  std::mutex m;
  std::condition_variable cv;

  ~WorkerPool() {
    // if (terminate == true) {
    //   if (!proc_completed) {
    //     print_worker_stats();
    //   }
    // } else {
    //   std::cout << "[destructor] shutting down workers" << std::endl;
    //   terminate = true;
    //   // cv.notify_all();
    //   for (auto &worker : workers) {
    //     if (!worker.second.second->terminate) {
    //       worker.second.second->terminate = true;
    //       worker.second.first->join();
    //     }
    //   }
    //   print_worker_stats();
    //   workers.clear();
    // }
  }
  friend class Worker;
};

}  // namespace scheduler

#endif /* SCHEDULER_WORKER_POOL_HPP_ */
