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

  WorkerPool* pool = &WorkerPool::getInstance();
  txn::TransactionManager* txnManager = &txn::TransactionManager::getInstance();
  storage::Schema* schema = &storage::Schema::getInstance();

  if (pool->_txn_bench != nullptr) {
    this->txnQueue = dynamic_cast<txn::TxnQueue*>(
        pool->_txn_bench->getBenchQueue(this->id, this->partition_id));
  } else {
    this->txnQueue = pool->txnQueue;
  }

  // pre-run / data-loaders
  if (txnQueue->type == txn::BENCH_QUEUE && !is_hotplugged) {
    auto* benchQueue = dynamic_cast<bench::BenchQueue*>(txnQueue);
    this->state = PRERUN;
    benchQueue->pre_run();
    this->state = READY;
    {
      pool->pre_barrier++;
      // following is that threads initialize there threadLocal txnTables
      // and be ready for actual txns.

      auto tx = this->txnQueue->popEmpty(this->id, this->partition_id);
      bool res = txnManager->executeFullQueue(tx, this->id, this->partition_id);

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
    bool res = txnManager->executeFullQueue(tx, this->id, this->partition_id);

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
  if (txnQueue->type == txn::BENCH_QUEUE && !is_hotplugged) {
    auto* benchQueue = dynamic_cast<bench::BenchQueue*>(txnQueue);

    this->state = POSTRUN;
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
  delete dynamic_cast<bench::BenchQueue*>(this->txnQueue);
}

void WorkerPool::init(bench::Benchmark* txn_bench, worker_id_t num_workers,
                      partition_id_t n_partitions, uint worker_scheduling_mode,
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
  this->worker_sched_mode = worker_scheduling_mode;
  this->num_partitions = n_partitions;
  this->elastic_workload = is_elastic_workload;

  prev_time_tps.reserve(topology::getInstance().getCoreCount());
  prev_sum_tps.reserve(topology::getInstance().getCoreCount());

  LOG(INFO) << "[WorkerPool] Number of Workers " << (int)num_workers;
  LOG(INFO) << "[WorkerPool] Scheduling Mode: " << worker_sched_mode;

  pre_barrier.store(0);
  post_barrier.store(0);
  worker_id_t i = 0;

  if (worker_sched_mode <= 2) {  // default / inteleave

    for (const auto& exec_core : topology::getInstance().getCores()) {
      // BROKEN
      // if (worker_sched_mode == 1) {  // interleave - even
      //   if (worker_counter % 2 != 0) {continue;}
      // } else if (worker_sched_mode == 2) {  // interleave - odd
      //   if (worker_counter % 2 == 0) continue;
      // }

      void* obj_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(Worker), exec_core.local_cpu_index);
      void* thd_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(std::thread), exec_core.local_cpu_index);

      auto* wrkr = new (obj_ptr) Worker(worker_counter++, &exec_core);
      // Worker* wrkr = new Worker(exec_core.id, &exec_core);
      // worker_counter++;

      wrkr->partition_id = (exec_core.local_cpu_index % num_partitions);
      wrkr->num_iters = num_iter_per_worker;

      LOG(INFO) << "Worker-" << (int)wrkr->id << "(" << exec_core.id
                << "): Allocated partition # " << (int)wrkr->partition_id;
      auto* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

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

    const auto& sys_cores = topology::getInstance().getCores();
    uint ctr = 0;
    for (int ci = sys_cores.size() - 1; ci >= 0; ci--) {
      void* obj_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(Worker), sys_cores[ci].local_cpu_index);
      void* thd_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(std::thread), sys_cores[ci].local_cpu_index);

      Worker* wrkr = new (obj_ptr) Worker(worker_counter++, &(sys_cores[ci]));

      wrkr->partition_id =
          0;  //(sys_cores[i].local_cpu_index % num_partitions);

      // if (global_conf::reverse_partition_numa_mapping) {
      //   wrkr->partition_id = (sys_cores[i].local_cpu_index % num_partitions);
      // } else {
      //   wrkr->partition_id = num_partitions - 1 -
      //                        (sys_cores[i].local_cpu_index % num_partitions);
      // }

      wrkr->num_iters = num_iter_per_worker;

      std::cout << "Worker-" << (uint)wrkr->id << "(" << sys_cores[ci].id
                << "): Allocated partition # " << (int)wrkr->partition_id
                << std::endl;
      auto* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

      workers.emplace(
          std::make_pair(sys_cores[ci].id, std::make_pair(thd, wrkr)));
      prev_time_tps.emplace_back(
          std::chrono::time_point<std::chrono::system_clock,
                                  std::chrono::nanoseconds>::min());
      prev_sum_tps.emplace_back(0);

      if (++ctr == num_workers) {
        break;
      }
    }

  } else if (worker_sched_mode ==
             4) {  // colocated - second-half of both socket
    assert(topology::getInstance().getCpuNumaNodeCount() == 2);
    // assert(false && "Turned off due to buggy code");
    const auto remote_cores = topology::getInstance().getCores().size() / 4;

    uint skipper1 = 0, skipper2 = 0;

    for (const auto& exec_core : topology::getInstance().getCores()) {
      if (skipper1 < remote_cores) {
        // skip half of first-socket
        skipper1++;
        continue;
      }

      if (worker_counter == remote_cores && skipper2 < remote_cores) {
        skipper2++;
        continue;
      }

      void* obj_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(Worker), exec_core.local_cpu_index);
      void* thd_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(std::thread), exec_core.local_cpu_index);

      auto* wrkr = new (obj_ptr) Worker(worker_counter++, &exec_core);
      // Worker* wrkr = new Worker(exec_core.id, &exec_core);
      // worker_counter++;
      if (global_conf::reverse_partition_numa_mapping) {
        wrkr->partition_id =
            num_partitions - 1 - (exec_core.local_cpu_index % num_partitions);
      } else {
        wrkr->partition_id = (exec_core.local_cpu_index % num_partitions);
      }

      wrkr->num_iters = num_iter_per_worker;

      std::cout << "Worker-" << (uint)wrkr->id << "(" << exec_core.id
                << "): Allocated partition # " << (int)wrkr->partition_id
                << std::endl;

      auto* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

      workers.emplace(std::make_pair(exec_core.id, std::make_pair(thd, wrkr)));
      prev_time_tps.emplace_back(
          std::chrono::time_point<std::chrono::system_clock,
                                  std::chrono::nanoseconds>::min());
      prev_sum_tps.emplace_back(0);
      if (++i == num_workers) {
        break;
      }
    }

  } else if (worker_sched_mode == 5) {
    // colocated - interleaved, leaving first one.
    assert(topology::getInstance().getCpuNumaNodeCount() == 2);
    // assert(false && "Turned off due to buggy code");
    const auto remote_cores = topology::getInstance().getCores().size() / 4;

    int skipper = -1;

    for (const auto& exec_core : topology::getInstance().getCores()) {
      skipper++;
      if (skipper % 2 == 0) {
        continue;
      }

      void* obj_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(Worker), exec_core.local_cpu_index);
      void* thd_ptr = MemoryManager::mallocPinnedOnNode(
          sizeof(std::thread), exec_core.local_cpu_index);

      auto* wrkr = new (obj_ptr) Worker(worker_counter++, &exec_core);
      // Worker* wrkr = new Worker(exec_core.id, &exec_core);
      // worker_counter++;
      if (global_conf::reverse_partition_numa_mapping) {
        wrkr->partition_id =
            num_partitions - 1 - (exec_core.local_cpu_index % num_partitions);
      } else {
        wrkr->partition_id = (exec_core.local_cpu_index % num_partitions);
      }

      wrkr->num_iters = num_iter_per_worker;

      std::cout << "Worker-" << (uint)wrkr->id << "(" << exec_core.id
                << "): Allocated partition # " << (int)wrkr->partition_id
                << std::endl;
      auto* thd = new (thd_ptr) std::thread(&Worker::run, wrkr);

      workers.emplace(std::make_pair(exec_core.id, std::make_pair(thd, wrkr)));
      prev_time_tps.emplace_back(
          std::chrono::time_point<std::chrono::system_clock,
                                  std::chrono::nanoseconds>::min());
      prev_sum_tps.emplace_back(0);
      if (++i == num_workers) {
        break;
      }
    }

  } else {
    assert(false && "Unknown scheduling mode.");
  }
  if (txn_bench != nullptr) {
    if (elastic_workload) {
      assert(false);

      //      // hack: initiate pre-run for hotplugged workers
      //      std::vector<std::thread> loaders;
      //      auto curr_master =
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

void WorkerPool::start_workers() {
  // this->txnQueue->type == txn::BENCH_QUEUE
  if (_txn_bench != nullptr) {
    while (pre_barrier != workers.size()) {
      std::this_thread::yield();
    }

    {
      std::lock_guard<std::mutex> lk(pre_m);
      LOG(INFO) << workers.size() << " Workers starting txn.." << std::endl;

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

  print_worker_stats();

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
  std::cout << "------------ WORKER STATS ------------" << std::endl;
  double tps = 0;
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
      std::cout << "\tTPS\t\t" << (tmp->num_commits / 1000000.0) / diff.count()
                << " MTPS" << std::endl;
    }
    socket_tps[tmp->exec_core->local_cpu_index] +=
        (tmp->num_commits / 1000000.0) / diff.count();
    duration += diff.count();
    duration_ctr += 1;

    tps += (tmp->num_commits / 1000000.0) / diff.count();
    num_commits += (tmp->num_commits / 1000000.0);
    num_aborts += (tmp->num_aborts / 1000000.0);
    num_txns += (tmp->num_txns / 1000000.0);
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
    if (wr.second.second->state != RUNNING && wr.second.second->state != PRERUN)
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

  if (worker_sched_mode == 3) {
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
  wrkr->is_hotplugged = true;
  _txn_bench->num_active_workers++;

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
