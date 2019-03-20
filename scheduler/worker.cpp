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

// TODO: check the reader lock on master and fall back thingy
inline ushort __attribute__((always_inline))
calculate_master_ver(uint64_t txn_id, uint64_t start_time) {
  // global_conf::time_master_switch_ms;
  // assuming start time is nanosec  so convert to millisec
  // start_time = start_time / 1000000;
  // uint di = start_time / global_conf::time_master_switch_ms;
  // uint mod = start_time % global_conf::time_master_switch_ms;

  // txn_id = ((txn_id << 8) >> 8);  // remove the worker_id

  uint64_t duration = ((txn_id & 0x00FFFFFF) - (start_time & 0x00FFFFFF)) /
                      1000000000;  // nanosec

  // ushort curr_master = (duration / global_conf::num_master_versions) %
  //                     global_conf::num_master_versions;

  // check which master should it be in. then check if there is a lock
  // on that master version

  // return curr_master;
  return (duration / global_conf::num_master_versions);
}

void Worker::run() {
  // std::cout << "[WORKER] Worker (TID:" << (int)(this->id)
  //          << "): Assigining Core ID:" << this->exec_core->id << std::endl;

  AffinityManager::getInstance().set(this->exec_core);

  WorkerPool* pool = &WorkerPool::getInstance();
  txn::TransactionManager* txnManager = &txn::TransactionManager::getInstance();

  txn_start_time =
      std::chrono::system_clock::now();  // txnManager->txn_start_time;

  while (true) {
    // std::cout << "T:" << calculate_master_ver(xid, start_time) << std::endl;
    if (terminate) {
      // std::cout << "break" << std::endl;
      break;
    }
    // std::this_thread::sleep_for (std::chrono::seconds(1));
    // std::cout << "[WORKER] Worker --" << (int)(this->id) << std::endl;

    // check which master i should be on.
    this->curr_txn = txnManager->get_next_xid(this->id);
    this->prev_master = this->curr_master;
    this->curr_master =
        calculate_master_ver(this->curr_txn, txnManager->txn_start_time) -
        txnManager->master_switch_delta;

    /* CASES
       1- Switched and I am the last one to switch with no active one on prev.
       (switch_master) 2- Switched and rotated back to the old one with someone
       active from    (GC) newer epoch. 3- Switched and rotated back to the old
       one with everyone on new epoch. 3- Switched and rotated back to old one
       and am the first one here. ( as per (1) it should be clean already) 4-
       Epoch changed but master remain the same due to delta -- read_lock

    */

    if (prev_master != curr_master) {
      // switched

      if (check_am_i_last) {
      } else {
        if ((prev_master % global_conf::num_master_versions) ==
            (curr_master % global_conf::num_master_versions)) {
          // rotated back
        } else {
          // absolutely new master
        }
      }

    } else {
      // didnt switch
    }

    if (prev_master == curr_master) {
      // do nothing, regular txn within master boundaries.
      ;
    } else if ((prev_master % global_conf::num_master_versions) ==
               (curr_master % global_conf::num_master_versions)) {
      // switched but rotated back to the old one
    } else if (prev_master != curr_master) {
      // switched

      // I guess store in some data structure that I am operational on this one.
      // also remove yourself from the previous one.
    }

    if (this->id == 0) {
      std::function<bool(uint64_t)> task;
      std::unique_lock<std::mutex> lock(pool->m);
      if (!pool->tasks.empty()) {
        //    NO-WAIT -> If task in queue, exec ELSE gen/exec txn
        //    pool->cv.wait(
        //    lock, [this, pool] { return this->terminate ||
        //    !pool->tasks.empty(); });
        task = std::move(pool->tasks.front());
        pool->tasks.pop();
        std::cout << "[WORKER] Worker (TID:" << (int)(this->id)
                  << ") Got a Task!" << std::endl;
        if (task(this->curr_txn))
          num_commits++;
        else
          num_aborts++;
      }
    }

    void* c = pool->txn_bench->gen_txn((int)this->id);

    if (txnManager->executor.execute_txn(
            c, this->curr_txn,
            this->curr_master % global_conf::num_master_versions))
      num_commits++;
    else
      num_aborts++;

    delete (struct txn::TXN*)c;

    num_txns++;
  }
}

std::vector<uint64_t> WorkerPool::get_active_txns() {
  std::vector<uint64_t> ret = std::vector<uint64_t>(this->size());

  for (auto& wr : workers) {
    ret.push_back(wr.second.second->curr_txn);
  }

  return ret;
}

void WorkerPool::print_worker_stats(bool global_only) {
  std::cout << "------------ WORKER STATS ------------" << std::endl;
  double tps = 0;
  double num_commits = 0;
  double num_aborts = 0;
  double num_txns = 0;
  for (auto it = workers.begin(); it != workers.end(); ++it) {
    // std::cout << " " << it->first << ":" << it->second;
    Worker* tmp = it->second.second;
    std::chrono::duration<double> diff =
        std::chrono::system_clock::now() - tmp->txn_start_time;

    if (!global_only) {
      std::cout << "Worker-" << (int)(tmp->id)
                << "(core_id: " << tmp->exec_core->id << ")" << std::endl;
      std::cout << "\t# of txns\t" << (tmp->num_txns / 1000000.0) << " M"
                << std::endl;
      std::cout << "\t# of commits\t" << (tmp->num_commits / 1000000.0) << " M"
                << std::endl;
      std::cout << "\t# of aborts\t" << (tmp->num_aborts / 1000000.0) << " M"
                << std::endl;
      std::cout << "\tTPS\t\t" << (tmp->num_txns / 1000000.0) / diff.count()
                << " mTPS" << std::endl;
    }

    tps += (tmp->num_txns / 1000000.0) / diff.count();
    num_commits += (tmp->num_commits / 1000000.0);
    num_aborts += (tmp->num_aborts / 1000000.0);
    num_txns += (tmp->num_txns / 1000000.0);
  }

  std::cout << "---- GLOBAL ----" << std::endl;
  std::cout << "\t# of txns\t" << num_txns << " M" << std::endl;
  std::cout << "\t# of commits\t" << num_commits << " M" << std::endl;
  std::cout << "\t# of aborts\t" << num_aborts << " M" << std::endl;
  std::cout << "\tTPS\t\t" << tps << " mTPS" << std::endl;

  std::cout << "------------ END WORKER STATS ------------" << std::endl;
}

void WorkerPool::init(bench::Benchmark* txn_bench) {
  std::cout << "[WorkerPool] Init" << std::endl;

  if (txn_bench == nullptr) {
    this->txn_bench = new bench::Benchmark();
  } else {
    this->txn_bench = txn_bench;
  }
  std::cout << "[WorkerPool] TXN Bench: " << this->txn_bench->name << std::endl;
  // txn_bench->exec_txn(txn_bench->gen_txn(0));
  // std::cout << "[WorkerPool] TEST TXN" << std::endl;
}

// Hot Plug
void WorkerPool::add_worker(core* exec_location) {
  assert(workers.find(exec_location->id) == workers.end());
  Worker* wrkr = new Worker(worker_counter++, exec_location);
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
    Worker* wrkr = new Worker(worker_counter++, &exec_core);
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

/*
<< operator for Worker pool to print stats of worker pool.
format:
*/
std::ostream& operator<<(std::ostream& out, const WorkerPool& topo) {
  out << "NOT IMPLEMENTED\n";
  return out;
}

}  // namespace scheduler
