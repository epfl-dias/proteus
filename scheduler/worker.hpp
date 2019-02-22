/*
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
  std::atomic<bool> terminate;
  core *exec_core;
  // WORKER_TYPE type;
  // core *exec_core;
  // void *payload;

  // std::function<void*(void)> gen_txn;
  // std::function<void(void*)> exec_txn;

  // for stats MAYBE
  uint64_t num_txns;
  uint64_t num_commits;
  uint64_t num_aborts;

  std::chrono::time_point<std::chrono::system_clock> txn_start_time;

 public:
  Worker(uint8_t id, core *exec_core)
      : id(id),
        terminate(false),
        exec_core(exec_core),
        num_txns(0),
        num_commits(0),
        num_aborts(0) {
    txn_start_time = std::chrono::system_clock::now();
  }
  // Worker(uint8_t id, core *exec_core, std::function<void*(void)> gen_txn,
  // std::function<void(void*)> exec_txn)
  //    : id(id), terminate(false), exec_core(exec_core), gen_txn(gen_txn),
  //    exec_txn(exec_txn) {}

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

  void init(bench::Benchmark *txn_bench = nullptr) {
    std::cout << "[WorkerPool] Init" << std::endl;

    if (txn_bench == nullptr) {
      this->txn_bench = new bench::Benchmark();
    } else {
      this->txn_bench = txn_bench;
    }
    std::cout << "[WorkerPool] TXN Bench: " << this->txn_bench->name
              << std::endl;
    txn_bench->exec_txn(txn_bench->gen_txn(0));
    std::cout << "[WorkerPool] TEST TXN" << std::endl;
  }

  void print_worker_stats() {
    std::cout << "------------ WORKER STATS ------------" << std::endl;
    for (auto it = workers.begin(); it != workers.end(); ++it) {
      // std::cout << " " << it->first << ":" << it->second;
      std::cout << "Worker-" << it->first << std::endl;
      Worker *tmp = it->second.second;
      std::cout << "\t# of txns\t" << tmp->num_txns << std::endl;
    }

    std::cout << "------------ END WORKER STATS ------------" << std::endl;
  }

  void shutdown(bool print_stats = false) {
    if (print_stats) print_worker_stats();
    this->~WorkerPool();
  }

  // void start_workers(int num_workers = -1);  // default case should be
  // (topology.get_total_cores - 2)

  void start_workers(int num_workers = 1);
  void add_worker(core *exec_location);
  void remove_worker(core *exec_location);

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueueTask(
      F &&f, Args &&... args);

  uint8_t size() { return workers.size(); }

 private:
  WorkerPool() { worker_id = 0; }

  volatile uint8_t worker_id;
  std::atomic<bool> terminate;
  std::unordered_map<uint8_t, std::pair<std::thread *, Worker *> > workers;
  bench::Benchmark *txn_bench;

  /*
  OPTMIZATION:
  Currently, its a single large global queue. we can
  have one queue per socket and then some how map the tasks to the queues
  dependent on data locality. this way we can ensure local NUMA access.
  but the main question here is how we would know that the data of the given
  task is local to which socket without actually touching the data. maybe key
  partioning, we can but that wont be shared-everything concept anymore?

*/
  std::queue<std::function<void()> > tasks;
  std::mutex m;
  std::condition_variable cv;
  ~WorkerPool() {
    std::cout << "[destructor] shutting down workers" << std::endl;
    terminate = true;
    cv.notify_all();
    for (auto &worker : workers) {
      worker.second.second->terminate = true;
      worker.second.first->join();
    }

    workers.clear();
  }
  friend class Worker;
};

}  // namespace scheduler

#endif /* WORKER_POOL_HPP_ */
