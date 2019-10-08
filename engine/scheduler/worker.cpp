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
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "scheduler/affinity_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"

#include "codegen/util/timing.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

#define HT false

#define RUN_WORKER 0

namespace scheduler {

void WorkerPool::shutdown_manual() {
  bool e_false_s = false;
  if (this->terminate.compare_exchange_strong(e_false_s, true)) {
    // cv.notify_all();
    std::cout << "Manual worker shutdown." << std::endl;
    for (auto& worker : workers) {
      worker.second.second->terminate = true;
    }
  }
}

// TODO: check the reader lock on master and fall back thingy
inline uint64_t __attribute__((always_inline))
calculate_delta_ver(uint64_t txn_id, uint64_t start_time) {
  // global_conf::time_master_switch_ms;
  // assuming start time is nanosec  so convert to millisec
  // start_time = start_time / 1000000;
  // uint di = start_time / global_conf::time_master_switch_ms;
  // uint mod = start_time % global_conf::time_master_switch_ms;

  // txn_id = ((txn_id << 8) >> 8);  // remove the worker_id

  uint64_t duration =
      ((txn_id & 0x00FFFFFFFFFFFFFF) -
       (start_time & 0x00FFFFFFFFFFFFFF));  ///
                                            // 1000000000;  // nanosec

  // ushort curr_master = (duration / global_conf::num_master_versions) %
  //                     global_conf::num_master_versions;

  // check which master should it be in. then check if there is a lock
  // on that master version

  // return curr_master;
  // std::cout << duration << std::endl;
  // return duration >> 6;  //(duration / global_conf::num_delta_storages);

  return duration >> 20;  // Magic Number
}

void Worker::run() {
  // std::cout << "[WORKER] Worker (TID:" << (int)(this->id)
  //          << "): Assigining Core ID:" << this->exec_core->id << std::endl;

  AffinityManager::getInstance().set(this->exec_core);

  WorkerPool* pool = &WorkerPool::getInstance();
  txn::TransactionManager* txnManager = &txn::TransactionManager::getInstance();
  storage::Schema* schema = &storage::Schema::getInstance();
  void* txn_mem = pool->txn_bench->get_query_struct_ptr();

  curr_delta = 0;
  prev_delta = 0;

  this->txn_start_tsc = txnManager->get_next_xid(this->id);
  uint64_t tx_st = txnManager->txn_start_time;
  this->curr_master = txnManager->current_master;
  this->curr_txn = txnManager->get_next_xid(this->id);
  this->prev_delta = this->curr_delta;
  this->curr_delta = calculate_delta_ver(this->curr_txn, tx_st);
  schema->add_active_txn(curr_delta % global_conf::num_delta_storages,
                         this->curr_delta, this->id);

  if (!is_hotplugged) {
    this->state = PRERUN;
    pool->txn_bench->pre_run(this->id, curr_txn, this->partition_id,
                             this->curr_master);

    this->state = READY;

    {
      std::unique_lock<std::mutex> lk(pool->pre_m);
      pool->pre_barrier++;
      pool->pre_cv.wait(lk, [pool, this] {
        return pool->pre_barrier == pool->workers.size() + 1 || terminate;
      });
    }
  }

  this->txn_start_time = std::chrono::system_clock::now();
  this->state = RUNNING;

  while (!terminate) {
    if (change_affinity) {
      AffinityManager::getInstance().set(this->affinity_core);
      change_affinity = false;
    }
    if (pause) {
      state = PAUSED;

      // FIXME: TODO:
      // while paused, remove itself from the delta versioning, and when
      // continuing back, add itself. otherwise, until paused, it will block the
      // garbage collection for no reason.
      // schema->remove_active_txn( this->curr_delta %
      // global_conf::num_delta_storages, this->curr_delta, this->id );

      while (pause && !terminate) {
        // std::this_thread::sleep_for(std::chrono::microseconds(50));
        std::this_thread::yield();
        this->curr_master = txnManager->current_master;
      }
      state = RUNNING;
      continue;
    }

    // Master-version upd
    this->curr_master = txnManager->current_master;

    // Delta versioning
    this->curr_txn = txnManager->get_next_xid(this->id);
    this->prev_delta = this->curr_delta;
    this->curr_delta = calculate_delta_ver(this->curr_txn, tx_st);

    ushort curr_delta_id = curr_delta % global_conf::num_delta_storages;
    ushort prev_delta_id = prev_delta % global_conf::num_delta_storages;

    if (prev_delta != curr_delta) {  // && curr_delta_id != prev_delta_id
      schema->switch_delta(prev_delta_id, curr_delta_id, curr_delta, this->id);

    }  // else didnt switch.

    // custom query coming in.
    // if (this->id == 0) {
    //   std::function<bool(uint64_t)> task;
    //   std::unique_lock<std::mutex> lock(pool->m);
    //   if (!pool->tasks.empty()) {
    //     //    NO-WAIT -> If task in queue, exec ELSE gen/exec txn
    //     //    pool->cv.wait(
    //     //    lock, [this, pool] { return this->terminate ||
    //     //    !pool->tasks.empty(); });
    //     task = std::move(pool->tasks.front());
    //     pool->tasks.pop();
    //     std::cout << "[WORKER] Worker (TID:" << (int)(this->id)
    //               << ") Got a Task!" << std::endl;
    //     if (task(this->curr_txn))
    //       num_commits++;
    //     else
    //       num_aborts++;
    //   }
    // }

    pool->txn_bench->gen_txn((uint)this->id, txn_mem, this->partition_id);

    // if (txnManager->executor.execute_txn(
    //         c, curr_txn, txnManager->current_master, curr_delta_id))
    if (pool->txn_bench->exec_txn(txn_mem, curr_txn, this->curr_master,
                                  curr_delta_id, this->partition_id))
      num_commits++;
    else
      num_aborts++;

    num_txns++;

    if (num_txns == num_iters) break;
  }

  txn_end_time = std::chrono::system_clock::now();

  schema->remove_active_txn(this->curr_delta % global_conf::num_delta_storages,
                            this->curr_delta, this->id);

  // POST RUN
  if (!is_hotplugged) {
    this->state = POSTRUN;
    this->curr_txn = txnManager->get_next_xid(this->id);
    this->curr_master = txnManager->current_master;

    pool->txn_bench->post_run(this->id, curr_txn, this->partition_id,
                              this->curr_master);
  }
  state = TERMINATED;
  pool->txn_bench->free_query_struct_ptr(txn_mem);
}

std::vector<uint64_t> WorkerPool::get_active_txns() {
  std::vector<uint64_t> ret = std::vector<uint64_t>(this->size());

  for (auto& wr : workers) {
    if (wr.second.second->state == RUNNING)
      ret.push_back(wr.second.second->curr_txn);
  }

  return ret;
}

uint64_t WorkerPool::get_min_active_txn() {
  uint64_t min_epoch = std::numeric_limits<uint64_t>::max();

  for (auto& wr : workers) {
    if (wr.second.second->state == RUNNING &&
        wr.second.second->curr_delta < min_epoch) {
      min_epoch = wr.second.second->curr_delta;
    }
  }

  return min_epoch;
}

uint64_t WorkerPool::get_max_active_txn() {
  uint64_t max_epoch = std::numeric_limits<uint64_t>::min();

  for (auto& wr : workers) {
    if (wr.second.second->state == RUNNING &&
        wr.second.second->curr_delta > max_epoch) {
      max_epoch = wr.second.second->curr_delta;
    }
  }

  return max_epoch;
}

bool WorkerPool::is_all_worker_on_master_id(ushort master_id) {
  for (auto& wr : workers) {
    if (wr.second.second->state == RUNNING &&
        wr.second.second->curr_master != master_id) {
      return false;
    }
  }

  return true;
}

void WorkerPool::pause() {
  time_block t("[WorkerPool] pause_: ");
  for (auto& wr : workers) {
    wr.second.second->pause = true;
  }

  for (auto& wr : workers) {
    if (wr.second.second->state != RUNNING && wr.second.second->state != PRERUN)
      continue;
    while (wr.second.second->state != PAUSED) {
      // std::this_thread::sleep_for(std::chrono::microseconds(10));
      std::this_thread::yield();
    }
  }
}
void WorkerPool::resume() {
  time_block t("[WorkerPool] resume_: ");
  for (auto& wr : workers) {
    wr.second.second->pause = false;
  }

  for (auto& wr : workers) {
    while (wr.second.second->state == PAUSED) {
      // std::this_thread::sleep_for(std::chrono::microseconds(10));
      std::this_thread::yield();
    }
  }
}

// double get_tps_diff_currently_active(struct hash_table* hash_table) {
//   //  printf("Have %d active servers\n", g_active_servers);
//   //  double avg_latch_wait = 0;
//   double tps = 0;
//   for (int i = 0; i < g_active_servers; i++) {
//     struct partition* p = &hash_table->partitions[i];
//     double cur_time = now();
//     if (prev_time_tps[i] == 0) {
//       prev_time_tps[i] = p->txn_start_time;
//     }
//     tps +=
//         ((double)(p->q_idx - prev_sum_tps[i])) / (cur_time -
//         prev_time_tps[i]);

//     prev_sum_tps[i] = p->q_idx;
//     prev_time_tps[i] = cur_time;
//   }

//   return tps / 1000000;
// }
void WorkerPool::print_worker_stats_diff() {
  static const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
  static uint num_sockets = vec.size();

  constexpr auto min_point =
      std::chrono::time_point<std::chrono::system_clock,
                              std::chrono::nanoseconds>::min();

  double tps = 0;
  double duration = 0;
  uint duration_ctr = 0;

  uint wl = 0;
  for (auto it = workers.begin(); it != workers.end(); ++it, wl++) {
    Worker* tmp = it->second.second;
    if (tmp->terminate) continue;

    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>
        curr_time = std::chrono::system_clock::now();

    std::chrono::duration<double> diff =
        curr_time - (prev_time_tps[wl] == min_point ? tmp->txn_start_time
                                                    : prev_time_tps[wl]);

    duration += diff.count();
    duration_ctr += 1;
    tps += ((tmp->num_txns - prev_sum_tps[wl]) / 1000000.0) / diff.count();
    prev_sum_tps[wl] = tmp->num_txns;
    prev_time_tps[wl] = curr_time;

    // tps += ((double) (p->q_idx - prev_sum_tps[i])) / (cur_time -
    // prev_time_tps[i]);
  }

  std::cout << "---- DIFF WORKER STATS ----" << std::endl;
  std::cout << "\tDuration\t" << (duration / duration_ctr) << " sec"
            << std::endl;
  std::cout << "\tTPS\t\t" << tps << " mTPS" << std::endl;

  std::cout << "------------ END ------------" << std::endl;
}

void WorkerPool::print_worker_stats(bool global_only) {
  std::cout << "------------ WORKER STATS ------------" << std::endl;
  double tps = 0;
  double num_commits = 0;
  double num_aborts = 0;
  double num_txns = 0;

  static const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
  static uint num_sockets = vec.size();

  std::vector<double> socket_tps(num_sockets, 0.0);

  double duration = 0;
  uint duration_ctr = 0;

  for (auto it = workers.begin(); it != workers.end(); ++it) {
    // std::cout << " " << it->first << ":" << it->second;
    Worker* tmp = it->second.second;
    if (tmp->is_hotplugged && tmp->terminate) continue;
    std::chrono::duration<double> diff =
        (this->terminate ? tmp->txn_end_time
                         : std::chrono::system_clock::now()) -
        tmp->txn_start_time;

    if (!global_only) {
      std::cout << "Worker-" << (int)(tmp->id)
                << "(core_id: " << tmp->exec_core->id << ")" << std::endl;

      std::cout << "\tDuration\t" << diff.count() << " sec" << std::endl;

      std::cout << "\t# of txns\t" << (tmp->num_txns / 1000000.0) << " M"
                << std::endl;
      std::cout << "\t# of commits\t" << (tmp->num_commits / 1000000.0) << " M"
                << std::endl;
      std::cout << "\t# of aborts\t" << (tmp->num_aborts / 1000000.0) << " M"
                << std::endl;
      std::cout << "\tTPS\t\t" << (tmp->num_txns / 1000000.0) / diff.count()
                << " mTPS" << std::endl;
    }
    socket_tps[tmp->exec_core->local_cpu_index] +=
        (tmp->num_txns / 1000000.0) / diff.count();
    duration += diff.count();
    duration_ctr += 1;

    tps += (tmp->num_txns / 1000000.0) / diff.count();
    num_commits += (tmp->num_commits / 1000000.0);
    num_aborts += (tmp->num_aborts / 1000000.0);
    num_txns += (tmp->num_txns / 1000000.0);
  }

  std::cout << "---- SOCKET ----" << std::endl;
  int i = 0;
  for (const auto& stps : socket_tps) {
    std::cout << "\tSocket-" << i++ << ": TPS\t\t" << stps << " mTPS"
              << std::endl;
  }

  duration = duration / duration_ctr;

  std::cout << "---- GLOBAL ----" << std::endl;
  std::cout << "\tDuration\t" << duration << " sec" << std::endl;
  std::cout << "\t# of txns\t" << num_txns << " M" << std::endl;
  std::cout << "\t# of commits\t" << num_commits << " M" << std::endl;
  std::cout << "\t# of aborts\t" << num_aborts << " M" << std::endl;
  std::cout << "\tTPS\t\t" << tps << " mTPS" << std::endl;

  std::cout << "------------ END WORKER STATS ------------" << std::endl;
  // if (this->terminate) proc_completed = true;
}

void WorkerPool::init(bench::Benchmark* txn_bench, uint num_workers,
                      uint num_partitions, uint worker_sched_mode,
                      int num_iter_per_worker, bool elastic_workload) {
  std::cout << "[WorkerPool] Init" << std::endl;

  if (txn_bench == nullptr) {
    this->txn_bench = new bench::Benchmark();
  } else {
    this->txn_bench = txn_bench;
  }
  std::cout << "[WorkerPool] TXN Bench: " << this->txn_bench->name << std::endl;

  this->num_iter_per_worker = num_iter_per_worker;
  this->worker_sched_mode = worker_sched_mode;
  this->num_partitions = num_partitions;
  this->elastic_workload = elastic_workload;

  prev_time_tps.reserve(Topology::getInstance().getCoreCount());
  prev_sum_tps.reserve(Topology::getInstance().getCoreCount());

  std::cout << "[WorkerPool] start_workers -- requested_num_workers: "
            << num_workers << std::endl;

  std::cout << "[WorkerPool] Number of Workers " << num_workers << std::endl;

  /* FIX ME:HACKED because we dont have topology returning specific number of
   * cores, this will be fixed when the elasticity and container stuff. until
   * then, just manually limit the number of wokrers
   */

  // auto worker_cores = Topology::getInstance().getCoresPtr();

  pre_barrier.store(0);
  int i = 0;

  if (worker_sched_mode <= 2) {  // default / inteleave

    for (const auto& exec_core : Topology::getInstance().getCores()) {
      if (worker_sched_mode == 1) {  // interleave - even
        if (worker_counter % 2 != 0) continue;
      } else if (worker_sched_mode == 2) {  // interleave - odd
        if (worker_counter % 2 == 0) continue;
      }

      void* obj_ptr = storage::MemoryManager::alloc(
          sizeof(Worker), exec_core.local_cpu_index, MADV_DONTFORK);
      void* thd_ptr = storage::MemoryManager::alloc(
          sizeof(std::thread), exec_core.local_cpu_index, MADV_DONTFORK);

      Worker* wrkr = new (obj_ptr) Worker(worker_counter++, &exec_core);
      // Worker* wrkr = new Worker(exec_core.id, &exec_core);
      // worker_counter++;

      wrkr->partition_id = (exec_core.local_cpu_index % num_partitions);
      wrkr->num_iters = num_iter_per_worker;

      std::cout << "Worker-" << (uint)wrkr->id << "(" << exec_core.id
                << "): Allocated partition # " << wrkr->partition_id
                << std::endl;
      std::thread* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

      workers.emplace(std::make_pair(exec_core.id, std::make_pair(thd, wrkr)));
      prev_time_tps.emplace_back(
          std::chrono::time_point<std::chrono::system_clock,
                                  std::chrono::nanoseconds>::min());
      prev_sum_tps.emplace_back(0);

      if (++i == num_workers) {
        break;
      }
    }

  } else if (worker_sched_mode == 3) {  // reversed
    assert(false && "Turned off due to buggy code");
    // for (std::vector<core>::reverse_iterator c = worker_cores.rbegin();
    //      c != worker_cores.rend(); ++c) {
    //   void* obj_ptr =
    //       storage::MemoryManager::alloc(sizeof(Worker), c->local_cpu_index);
    //   void* thd_ptr = storage::MemoryManager::alloc(sizeof(std::thread),
    //                                                 c->local_cpu_index);

    //   Worker* wrkr = new (obj_ptr) Worker(worker_counter++, &(*c));
    //   wrkr->partition_id = (c->local_cpu_index % num_partitions);
    //   wrkr->num_iters = num_iter_per_worker;

    //   std::cout << "Worker-" << (uint)wrkr->id << ": Allocated partition # "
    //             << wrkr->partition_id << std::endl;
    //   std::thread* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

    //   workers.emplace(std::make_pair(c->id, std::make_pair(thd, wrkr)));
    //   if (++i == num_workers) {
    //     break;
    //   }
    // }

  } else {
    assert(false && "Unknown scheduling mode.");
  }

  if (elastic_workload) {
    // hack: initiate pre-run for hotplugged workers
    std::vector<std::thread> loaders;
    auto curr_master =
        txn::TransactionManager::getInstance().current_master.load();

    const auto& wrks_crs = Topology::getInstance().getCores();

    for (int i = txn_bench->num_active_workers; i < txn_bench->num_max_workers;
         i++) {
      loaders.emplace_back([this, i, wrks_crs, curr_master, num_partitions]() {
        auto tid = txn::TransactionManager::getInstance().get_next_xid(i);
        this->txn_bench->pre_run(
            i, tid, wrks_crs.at(i).local_cpu_index % num_partitions,
            curr_master);
      });
    }

    for (auto& th : loaders) {
      th.join();
    }

    std::cout << "Elastic Workload: Worker-group-size: " << workers.size()
              << std::endl;
  }

  while (pre_barrier != workers.size()) {
    std::this_thread::yield();
  }
}

void WorkerPool::migrate_worker() {
  static std::vector<scheduler::core> worker_cores =
      Topology::getInstance().getCoresCopy();
  static const uint pool_size = workers.size();
  assert(worker_cores.size() == (pool_size * 2));

  static uint worker_num = 0;

  auto get = workers.find(worker_cores[worker_num].id);
  assert(get != workers.end());

  get->second.second->affinity_core = &(worker_cores[pool_size + worker_num]);
  get->second.second->change_affinity = true;

  worker_num++;
}

// Hot Plug
void WorkerPool::add_worker(const core* exec_location, short partition_id) {
  // assert(workers.find(exec_location->id) == workers.end());
  void* obj_ptr = storage::MemoryManager::alloc(
      sizeof(Worker), exec_location->local_cpu_index, MADV_DONTFORK);
  void* thd_ptr = storage::MemoryManager::alloc(
      sizeof(std::thread), exec_location->local_cpu_index, MADV_DONTFORK);

  Worker* wrkr = new (obj_ptr) Worker(worker_counter++, exec_location);
  wrkr->partition_id =
      (partition_id == -1
           ? (exec_location->local_cpu_index % this->num_partitions)
           : partition_id);
  wrkr->num_iters = this->num_iter_per_worker;
  wrkr->is_hotplugged = true;
  txn_bench->num_active_workers++;
  std::thread* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

  workers.emplace(std::make_pair(exec_location->id, std::make_pair(thd, wrkr)));
  prev_time_tps.emplace_back(
      std::chrono::time_point<std::chrono::system_clock,
                              std::chrono::nanoseconds>::min());
  prev_sum_tps.emplace_back(0);
}

// Hot Plug
void WorkerPool::remove_worker(const core* exec_location) {
  auto get = workers.find(exec_location->id);
  txn_bench->num_active_workers--;
  assert(get != workers.end());
  get->second.second->terminate = true;
  // TODO: remove from the vector too?
}

void WorkerPool::start_workers() {
  while (pre_barrier != workers.size()) {
    std::this_thread::yield();
  }

  {
    std::lock_guard<std::mutex> lk(pre_m);
    std::cout << "[WorkerPool] " << workers.size() << " Workers starting txn.."
              << std::endl;

    // storage::Schema::getInstance().report();
    pre_barrier++;
  }

  __itt_resume();
  pre_cv.notify_all();
}

void WorkerPool::shutdown(bool print_stats) {
  __itt_pause();
  // this->~WorkerPool();

  this->terminate.store(true);
  // cv.notify_all();
  for (auto& worker : workers) {
    if (!worker.second.second->terminate) {
      worker.second.second->terminate = true;
      worker.second.first->join();
    }
  }
  print_worker_stats();
  workers.clear();
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
// std::ostream& operator<<(std::ostream& out, const WorkerPool& topo) {
//   out << "NOT IMPLEMENTED\n";
//   return out;
// }

}  // namespace scheduler
