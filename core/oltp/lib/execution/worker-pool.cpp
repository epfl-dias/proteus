/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "oltp/execution/worker-pool.hpp"

namespace scheduler {

void WorkerPool::init(bench::Benchmark* txn_bench, worker_id_t num_workers,
                      partition_id_t n_partitions,
                      WorkerSchedulePolicy schedule_policy,
                      int num_iterations_per_worker, bool is_elastic_workload) {
  LOG(INFO) << "[WorkerPool] Init";

  if (txn_bench != nullptr) {
    _txn_bench = txn_bench;
  }

  if (txn_bench != nullptr)
    LOG(INFO) << "[WorkerPool] TXN Bench: " << txn_bench->name;
  else
    LOG(INFO) << "[WorkerPool] Interactive Queue mode";

  this->num_iter_per_worker = num_iterations_per_worker;
  this->worker_schedule_policy = schedule_policy;
  this->num_partitions = n_partitions;
  this->elastic_workload = is_elastic_workload;

  prev_time_tps.reserve(topology::getInstance().getCoreCount());
  prev_sum_tps.reserve(topology::getInstance().getCoreCount());

  LOG(INFO) << "[WorkerPool] Number of Workers " << (int)num_workers;
  LOG(INFO) << "[WorkerPool] Scheduling Mode: " << schedule_policy;

  pre_barrier.store(0);
  post_barrier.store(0);

  ScheduleWorkers::getInstance().schedule(
      schedule_policy,
      [&](const topology::core* workerThread, bool isPhysical) {
        create_worker(*workerThread, isPhysical);
      },
      num_workers);

  if (txn_bench != nullptr) {
    if (elastic_workload) {
      assert(false &&
             "deprecated old code, needs to be updated to conform to new "
             "worker design.");
      //
      //      // hack: initiate pre-run for hotplugged workers
      //      std::vector<std::thread> loaders;
      //      auto curr_master =
      //
      //          txn::TransactionManager::getInstance().get_current_master_version();
      //
      //      const auto& wrks_crs = topology::getInstance().getCores();
      //
      //      for (int ew = _txn_bench->num_active_workers;
      //           ew < _txn_bench->num_max_workers; i++) {
      //        loaders.emplace_back(
      //            [this, ew, curr_master](auto partition_id) {
      //              auto tid =
      //                  txn::TransactionManager::getInstance().get_next_xid(ew);
      //              this->_txn_bench->pre_run(ew, tid, partition_id,
      //              curr_master);
      //            },
      //            wrks_crs.at(ew).local_cpu_index % num_partitions);
      //      }
      //
      //      for (auto& th : loaders) {
      //        th.join();
      //      }
      //
      //      std::cout << "Elastic Workload: Worker-group-size: " <<
      //      workers.size()
      //                << std::endl;
    }

    while (pre_barrier != workers.size()) {
      std::this_thread::yield();
    }
  }
}

void WorkerPool::create_worker(const topology::core& exec_core,
                               bool physical_thread) {
  std::unique_lock<std::mutex> lk(worker_pool_lk);
  void* obj_ptr = MemoryManager::mallocPinnedOnNode(sizeof(Worker),
                                                    exec_core.local_cpu_index);
  void* thd_ptr = MemoryManager::mallocPinnedOnNode(sizeof(std::thread),
                                                    exec_core.local_cpu_index);

  auto* txn_worker = new (obj_ptr) Worker(worker_counter++, &exec_core);
  txn_worker->partition_id = (exec_core.local_cpu_index % num_partitions);
  txn_worker->num_iters = num_iter_per_worker;

  LOG(INFO) << "Worker-" << (int)txn_worker->id << " ( core-" << exec_core.id
            << " ): Allocated partition # " << (int)txn_worker->partition_id
            << "\t\t(" << (physical_thread ? "Physical" : "HyperThread") << ")";
  auto* thd = new (thd_ptr) std::thread(&Worker::run, txn_worker);

  workers.emplace(exec_core.id, std::make_pair(thd, txn_worker));
  prev_time_tps.emplace_back(
      std::chrono::time_point<std::chrono::system_clock,
                              std::chrono::nanoseconds>::min());
  prev_sum_tps.emplace_back(0);
}

void WorkerPool::start_workers() {
  // this->txnQueue->type == txn::BENCH_QUEUE
  if (_txn_bench != nullptr) {
    while (pre_barrier != workers.size()) {
      std::this_thread::yield();
    }

    {
      std::lock_guard<std::mutex> lk(pre_m);
      LOG(INFO) << workers.size() << " workers starting txn.." << std::endl;

      // storage::Schema::getInstance().report();
      pre_barrier++;
    }

    pre_cv.notify_all();
  }
}

void WorkerPool::shutdown(bool print_stats) {
  this->terminate.store(true);

  // cv.notify_all();
  for (auto& worker : workers) {
    worker.second.second->terminate = true;
  }

  for (auto& worker : workers) {
    if (worker.second.first->joinable()) {
      worker.second.first->join();
    }
  }

  print_worker_stats(true);

  for (auto& worker : workers) {
    // FIXME: why not using an allocator with the map?
    worker.second.first->~thread();
    MemoryManager::freePinned(worker.second.first);

    worker.second.second->~Worker();
    MemoryManager::freePinned(worker.second.second);
  }
  workers.clear();
}

std::pair<double, double> WorkerPool::get_worker_stats_diff(bool print) {
  static const auto& vec = topology::getInstance().getCpuNumaNodes();
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
    tps += ((tmp->num_commits - prev_sum_tps[wl]) / 1000000.0) / diff.count();
    prev_sum_tps[wl] = tmp->num_commits;
    prev_time_tps[wl] = curr_time;

    // tps += ((double) (p->q_idx - prev_sum_tps[i])) / (cur_time -
    // prev_time_tps[i]);
  }

  if (print) {
    std::cout << "---- DIFF WORKER STATS ----" << std::endl;
    std::cout << "\tDuration\t" << (duration / duration_ctr) << " sec"
              << std::endl;
    std::cout << "\tTPS\t\t" << tps << " MTPS" << std::endl;

    std::cout << "------------ END ------------" << std::endl;
  }

  return std::make_pair((duration / duration_ctr), tps);
}

void WorkerPool::print_worker_stats_diff() {
  static const auto& vec = topology::getInstance().getCpuNumaNodes();
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
    tps += ((tmp->num_commits - prev_sum_tps[wl]) / 1000000.0) / diff.count();
    prev_sum_tps[wl] = tmp->num_commits;
    prev_time_tps[wl] = curr_time;

    // tps += ((double) (p->q_idx - prev_sum_tps[i])) / (cur_time -
    // prev_time_tps[i]);
  }

  std::cout << "---- DIFF WORKER STATS ----" << std::endl;
  std::cout << "\tDuration\t" << (duration / duration_ctr) << " sec"
            << std::endl;
  std::cout << "\tTPS\t\t" << tps << " MTPS" << std::endl;

  std::cout << "------------ END ------------" << std::endl;
}

void WorkerPool::print_worker_stats(bool global_only) {
  auto n_ro_workers = _txn_bench->get_n_readonly_workers();
  std::cout << "------------ WORKER STATS ------------" << std::endl;
  std::cout << "----- num_readonly_workers: " << n_ro_workers << std::endl;
  double tps = 0;
  double qps = 0;
  double num_commits = 0;
  double num_aborts = 0;
  double num_txns = 0;

  static const auto& vec = topology::getInstance().getCpuNumaNodes();
  static uint num_sockets = vec.size();

  std::vector<double> socket_tps(num_sockets, 0.0);

  double duration = 0;
  uint duration_ctr = 0;

  for (auto& worker : workers) {
    Worker* tmp = worker.second.second;
    if (tmp->is_hot_plugged && tmp->terminate) continue;
    std::chrono::duration<double> diff =
        (this->terminate ? tmp->txn_end_time
                         : std::chrono::system_clock::now()) -
        tmp->txn_start_time;

    if (!global_only) {
      std::cout << "Worker-" << (int)(tmp->id)
                << "(core_id: " << tmp->exec_core->id << ")"
                << (n_ro_workers > tmp->id ? " [READER] " : "") << std::endl;

      std::cout << "\tDuration\t" << diff.count() << " sec" << std::endl;

      std::cout << "\t# of txns\t" << (tmp->num_txns) << std::endl;
      std::cout << "\t# of commits\t" << (tmp->num_commits / 1000000.0) << " M"
                << std::endl;
      std::cout << "\t# of aborts\t" << (tmp->num_aborts / 1000000.0) << " M"
                << std::endl;
      std::cout << "\tTPS\t\t" << (tmp->num_commits) / diff.count() << " TPS"
                << std::endl;
    }

    socket_tps[tmp->exec_core->local_cpu_index] +=
        (tmp->num_commits / 1000000.0) / diff.count();
    duration += diff.count();
    duration_ctr += 1;

    if (n_ro_workers > tmp->id) {
      qps += ((double)(tmp->num_commits)) /
             diff.count();  // WHY I HAD A divideBy60 here?
    } else {
      tps += (tmp->num_commits / 1000000.0) / diff.count();
      num_commits += (tmp->num_commits / 1000000.0);
      num_aborts += (tmp->num_aborts / 1000000.0);
      num_txns += (tmp->num_txns / 1000000.0);
    }
  }

  std::cout << "---- SOCKET ----" << std::endl;
  int i = 0;
  for (const auto& stps : socket_tps) {
    std::cout << "\tSocket-" << i++ << ": TPS\t\t" << stps << " MTPS"
              << std::endl;
  }

  duration = duration / duration_ctr;

  std::cout << "---- GLOBAL ----" << std::endl;
  std::cout << "\tDuration\t" << duration << " sec" << std::endl;
  std::cout << "\t# of txns\t" << num_txns << " M" << std::endl;
  std::cout << "\t# of commits\t" << num_commits << " M" << std::endl;
  std::cout << "\t# of aborts\t" << num_aborts << " M" << std::endl;
  std::cout << "\tTPS\t\t" << tps << " MTPS" << std::endl;

  if (n_ro_workers > 0) {
    std::cout << "\tQPS\t\t" << qps << " QPS" << std::endl;
  }

  std::cout << "------------ END WORKER STATS ------------" << std::endl;
  // if (this->terminate) proc_completed = true;
}

// NOTE: Following functions are not tested and may not work.
void WorkerPool::pause() {
  time_block t("[WorkerPool] pause_: ");
  for (auto& wr : workers) {
    wr.second.second->pause = true;
  }

  for (auto& wr : workers) {
    if (wr.second.second->state != RUNNING &&
        wr.second.second->state != PRE_RUN)
      continue;
    if (wr.second.second->terminate) continue;
    while (wr.second.second->state != PAUSED) {
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
    if (wr.second.second->terminate) continue;
    while (wr.second.second->state == PAUSED) {
      std::this_thread::yield();
    }
  }
}

void WorkerPool::migrate_worker(bool return_back) {
  static const auto& worker_cores = topology::getInstance().getCores();
  static const uint pool_size = workers.size();
  assert(worker_cores.size() == (pool_size * 2));

  static uint worker_num = 0;
  assert(worker_num <= (workers.size() / 2) &&
         "Migration with HT happens in pair, that is, physical core");

  if (global_conf::reverse_partition_numa_mapping) {
    const auto ht_pair =
        topology::getInstance().getCpuNumaNodes()[0].local_cores.size() / 2;

    if (!return_back) {
      auto get = workers.find(worker_cores[pool_size + worker_num].id);

      LOG(INFO) << "Worker-Migrate: From-"
                << worker_cores[pool_size + worker_num].id << "  To-"
                << worker_cores[worker_num].id;
      assert(get != workers.end());
      get->second.second->affinity_core =
          const_cast<topology::core*>(&(worker_cores[worker_num]));
      get->second.second->change_affinity = true;

      // HT-pair now
      auto ht_worker = worker_num + ht_pair;
      auto getHT = workers.find(worker_cores[pool_size + ht_worker].id);

      LOG(INFO) << "Worker-Migrate: From-"
                << worker_cores[pool_size + ht_worker].id << "  To-"
                << worker_cores[ht_worker].id;
      assert(getHT != workers.end());
      getHT->second.second->affinity_core =
          const_cast<topology::core*>(&(worker_cores[ht_worker]));
      getHT->second.second->change_affinity = true;

    } else {
      auto get = workers.find(worker_cores[pool_size + worker_num - 1].id);

      LOG(INFO) << "Worker-Migrate: From " << worker_cores[worker_num - 1].id
                << "  returning to "
                << worker_cores[pool_size + worker_num - 1].id;
      assert(get != workers.end());
      get->second.second->revert_affinity = true;
      get->second.second->change_affinity = true;

      // HT-pair now
      auto ht_worker = worker_num - 1 + ht_pair;
      auto getHT = workers.find(worker_cores[pool_size + ht_worker].id);

      LOG(INFO) << "Worker-Migrate: From " << worker_cores[ht_worker].id
                << "  returning to " << worker_cores[pool_size + ht_worker].id;
      assert(getHT != workers.end());
      getHT->second.second->revert_affinity = true;
      getHT->second.second->change_affinity = true;
    }

  } else {
    if (!return_back) {
      auto get = workers.find(worker_cores[worker_num].id);
      assert(get != workers.end());
      get->second.second->affinity_core =
          const_cast<topology::core*>(&(worker_cores[pool_size + worker_num]));
      get->second.second->change_affinity = true;
    } else {
      auto get = workers.find(worker_cores[worker_num - 1].id);
      assert(get != workers.end());

      get->second.second->revert_affinity = true;
      get->second.second->change_affinity = true;
    }
  }
  if (!return_back)
    worker_num += 1;
  else
    worker_num -= 1;
}

const std::vector<worker_id_t>& WorkerPool::scale_down(uint num_cores) {
  // std::vector<uint> core_ids{num_cores};

  uint ctr = 0;

  const auto& cres = topology::getInstance().getCores();

  const auto ht_pair =
      topology::getInstance().getCpuNumaNodes()[0].local_cores.size() / 2;
  const auto st = cres.size() / 2;
  const auto end = (cres.size() / 2) + num_cores;
  for (int ci = st; ci <= end; ci++) {
    Worker* tmp = workers[cres[ci].id].second;
    Worker* tmp2 = workers[cres[ci + ht_pair].id].second;

    if (ctr == num_cores) {
      break;
    }

    std::cout << "Core: " << cres[ci].id
              << " w/ HT-pair: " << cres[ci + ht_pair].id << std::endl;

    std::cout << "Turning off core with id: " << cres[ci].id << " | "
              << tmp->exec_core->id << std::endl;

    std::cout << "Turning off core with id: " << cres[ci + ht_pair].id << " | "
              << tmp2->exec_core->id << std::endl;

    // TODO : FIX!
    tmp->terminate = true;
    tmp2->terminate = true;

    // if (tmp->state != PAUSED) {
    //   tmp->pause = true;
    // }

    // if (tmp2->state != PAUSED) {
    //   tmp2->pause = true;
    // }

    elastic_set.push_back(tmp->exec_core->id);
    elastic_set.push_back(tmp2->exec_core->id);
    ctr++;
    ctr++;
  }

  // for (auto it = workers.begin(); it != workers.end(); ++it) {

  //   if (ctr == 0) {
  //     size_t mv = workers.size() - num_cores;

  //     for (uint c = 0; c < mv; c++) {
  //       it++;
  //     }
  //   }

  //   Worker* tmp = it->second.second;

  //   if (tmp->state != PAUSED) {
  //     tmp->state = PAUSED;
  //     // core_ids[ctr] = tmp->exec_core->id;
  //     elastic_set.push_back(tmp->exec_core->id);
  //     ctr++;
  //   }
  // }
  return elastic_set;
}

void WorkerPool::scale_back() {
  for (const auto& id : elastic_set) {
    if (workers[id].second->state == PAUSED) {
      workers[id].second->pause = false;
    }
  }
  elastic_set.clear();
}

// Hot Plug
void WorkerPool::add_worker(const topology::core* exec_location,
                            int partition_id) {
  // assert(workers.find(exec_location->id) == workers.end());
  void* obj_ptr = MemoryManager::mallocPinnedOnNode(
      sizeof(Worker), exec_location->local_cpu_index);
  void* thd_ptr = MemoryManager::mallocPinnedOnNode(
      sizeof(std::thread), exec_location->local_cpu_index);

  auto* wrkr = new (obj_ptr) Worker(worker_counter++, exec_location);
  wrkr->partition_id =
      (partition_id == -1
           ? (exec_location->local_cpu_index % this->num_partitions)
           : partition_id);
  wrkr->num_iters = this->num_iter_per_worker;
  wrkr->is_hot_plugged = true;
  _txn_bench->incrementActiveWorker();
  //_txn_bench->num_active_workers++;

  auto* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

  workers.emplace(std::make_pair(exec_location->id, std::make_pair(thd, wrkr)));
  prev_time_tps.emplace_back(
      std::chrono::time_point<std::chrono::system_clock,
                              std::chrono::nanoseconds>::min());
  prev_sum_tps.emplace_back(0);
}

// Hot Plug
void WorkerPool::remove_worker(const topology::core* exec_location) {
  auto get = workers.find(exec_location->id);
  //_txn_bench->num_active_workers--;
  assert(get != workers.end());
  get->second.second->terminate = true;
  // TODO: remove from the vector too?
}

}  // namespace scheduler
