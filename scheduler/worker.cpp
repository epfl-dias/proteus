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

#include "scheduler/worker.hpp"
#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include "scheduler/affinity_manager.hpp"
#include "transactions/transaction_manager.hpp"

namespace scheduler {

// NOTE: There might be a problem because I am passing a local variable
// reference to a thread

void Worker::run() {
  std::cout << "[WORKER] Worker (TID:" << (int)(this->id)
            << "): Assigining Core ID:" << this->exec_core->id << std::endl;

  AffinityManager::getInstance().set(this->exec_core);

  WorkerPool* pool = &WorkerPool::getInstance();
  txn::TransactionManager* txnManager = &txn::TransactionManager::getInstance();

  while (true) {
    if (this->terminate) break;
    // std::this_thread::sleep_for (std::chrono::seconds(1));
    // std::cout << "[WORKER] Worker --" << (int)(this->id) << std::endl;

    std::function<void()> task;
    bool has_task = false;
    {
      std::unique_lock<std::mutex> lock(pool->m);
      if (!pool->tasks.empty()) {
        //    NO-WAIT -> If task in queue, exec ELSE gen/exec txn
        //    pool->cv.wait(
        //    lock, [this, pool] { return this->terminate ||
        //    !pool->tasks.empty(); });
        task = std::move(pool->tasks.front());
        pool->tasks.pop();
      }
    }
    if (has_task) {
      std::cout << "[WORKER] Worker (TID:" << (int)(this->id) << ") Got a Task!"
                << std::endl;
      /* FIXME: how to keep track of abort/commit and results of the tasks
       * (transactions submitted through frontend interface)
       */
      task();
    } else {
      /* Do we really need per worker stats? becuase this if/else is gonna slow
       * things down when each worker needs to scale to millions of txns/sec
       */

      // pool->txn_bench->exec_txn(pool->txn_bench->gen_txn(this->id));
      void* c = pool->txn_bench->gen_txn(this->id);
      if (txnManager->execute_txn(c))
        num_commits++;
      else
        num_aborts++;

      delete (struct txn::TXN*)c;
    }
    num_txns++;
  }
}

// void WorkerPool::init() { std::cout << "[WorkerPool] Init" << std::endl; }

// Hot Plug
void WorkerPool::add_worker(core* exec_location) {
  assert(workers.find(exec_location->id) == workers.end());
  Worker* wrkr = new Worker(worker_id++, exec_location);
  std::thread* thd = new std::thread(&Worker::run, wrkr);

  workers.emplace(std::make_pair(exec_location->id, std::make_pair(thd, wrkr)));
}

// Hot Plug
void WorkerPool::remove_worker(core* exec_location) {
  auto get = workers.find(exec_location->id);
  assert(get != workers.end());
  get->second.second->terminate = true;
}

void WorkerPool::start_workers(int num_workers) {
  std::cout << "[WorkerPool] start_workers -- requested_num_workers: "
            << num_workers << std::endl;
  std::vector<core>* worker_cores =
      Topology::getInstance().get_worker_cores(num_workers);

  std::cout << "[WorkerPool] Number of Workers (AUTO) " << num_workers
            << std::endl;

  /* FIX ME:HACKED because we dont have topology returning specific number of
   * cores, this will be fixed when the elasticity and container stuff. until
   * then, just manually limit the number of wokrers
   */

  int i = 0;
  for (auto& exec_core : *worker_cores) {
    Worker* wrkr = new Worker(++worker_id, &exec_core);
    std::thread* thd = new std::thread(&Worker::run, wrkr);

    workers.emplace(std::make_pair(exec_core.id, std::make_pair(thd, wrkr)));
    if (++i == num_workers) {
      break;
    }
  }
}

template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> WorkerPool::enqueueTask(
    F&& f, Args&&... args) {
  using packaged_task_t =
      std::packaged_task<typename std::result_of<F(Args...)>::type()>;

  std::shared_ptr<packaged_task_t> task(new packaged_task_t(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)));

  auto res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(m);
    tasks.emplace([task]() { (*task)(); });
  }

  cv.notify_one();
  return res;
}

}  // namespace scheduler
