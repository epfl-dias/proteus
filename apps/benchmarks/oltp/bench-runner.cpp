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

#include <gflags/gflags.h>
#include <unistd.h>

#include <bitset>
#include <cli-flags.hpp>
#include <iostream>
#include <thread>

// OLTP Engine
#include "oltp/common/constants.hpp"
#include "oltp/common/oltp-cli-flags.hpp"
#include "oltp/common/utils.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/interface/bench.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction_manager.hpp"

// Bench Includes
#include "bench-cli-flags.hpp"
#include "tpcc/tpcc_64.hpp"
#include "ycsb.hpp"

// Platform Includes
#include <platform/common/common.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/interval-runner.hpp>
#include <platform/util/profiling.hpp>

int main(int argc, char** argv) {
  auto olap = proteus::from_cli::olap(
      "Simple command line interface for proteus-oltp", &argc, &argv);

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();
  if constexpr (global_conf::reverse_partition_numa_mapping) {
    set_exec_location_on_scope d{nodes[nodes.size() - 1]};
  } else {
    set_exec_location_on_scope d{nodes[0]};
  }

  if (FLAGS_num_workers == 0) FLAGS_num_workers = topo.getCoreCount();

  g_use_hyperThreads = FLAGS_use_hyperthreads;
  g_delta_size = FLAGS_delta_size;

  auto ht_size = topo.getCores()[0].ht_pairs_id.size();
  auto n_physical_cores = topo.getCoreCount() / ht_size;
  auto n_physical_cores_per_numa =
      n_physical_cores / topo.getCpuNumaNodeCount();

  if (FLAGS_num_workers > n_physical_cores) {
    LOG(INFO) << "FLAGS_num_workers > n_physical_cores :: Using hyperThreads ( "
                 "g_use_hyperThreads: "
              << g_use_hyperThreads << " )";
    g_use_hyperThreads = true;
    FLAGS_use_hyperthreads = true;
  }

  if (FLAGS_num_partitions == 0) {
    if (FLAGS_elastic_workload > 0) {
      g_num_partitions = topo.getCpuNumaNodeCount();
    } else {
      LOG(INFO) << "Use hyper-threads: " << g_use_hyperThreads;
      if (g_use_hyperThreads) {
        g_num_partitions = ((FLAGS_num_workers - 1) /
                            topo.getCpuNumaNodes()[0].local_cores.size()) +
                           1;
      } else {
        assert(FLAGS_num_workers <= n_physical_cores);

        g_num_partitions =
            ((FLAGS_num_workers - 1) / n_physical_cores_per_numa) + 1;
      }
    }

  } else {
    g_num_partitions = FLAGS_num_partitions;
  }
  LOG(INFO) << "PARTITIONS: " << g_num_partitions << std::endl;

  storage::Schema* schema = &storage::Schema::getInstance();

  // init benchmark
  bench::Benchmark* bench = nullptr;
  if (FLAGS_benchmark == 1) {
    if (FLAGS_tpcc_num_wh == 0) FLAGS_tpcc_num_wh = FLAGS_num_workers;

    bench =
        new bench::TPCC("TPCC", FLAGS_tpcc_num_wh,
                        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
                        FLAGS_layout_column_store, {}, FLAGS_ch_scale_factor);

  } else if (FLAGS_benchmark == 2) {
    if (FLAGS_tpcc_num_wh == 0) FLAGS_tpcc_num_wh = FLAGS_num_workers;
    bench =
        new bench::TPCC("TPCC", FLAGS_tpcc_num_wh,
                        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
                        FLAGS_layout_column_store, {}, FLAGS_ch_scale_factor,
                        FLAGS_tpcc_dist_threshold, FLAGS_tpcc_csv_dir);

  } else {  // Default YCSB

    LOG(INFO) << "Write Threshold: " << FLAGS_ycsb_write_ratio;
    LOG(INFO) << "Theta: " << FLAGS_ycsb_zipf_theta;
    if (FLAGS_ycsb_num_records == 0) {
      FLAGS_ycsb_num_records = FLAGS_num_workers * 1000000;
    }
    if (FLAGS_ycsb_num_col_upd != 0) {
      assert(FLAGS_ycsb_num_col_upd <= FLAGS_ycsb_num_cols);
    } else {
      FLAGS_ycsb_num_col_upd = FLAGS_ycsb_num_cols;
    }

    if (FLAGS_ycsb_num_col_read != 0) {
      assert(FLAGS_ycsb_num_col_read <= FLAGS_ycsb_num_cols);
    } else {
      FLAGS_ycsb_num_col_read = FLAGS_ycsb_num_cols;
    }

    bench = new bench::YCSB(
        "YCSB", FLAGS_ycsb_num_cols, FLAGS_ycsb_num_records,
        FLAGS_ycsb_zipf_theta, FLAGS_num_iter_per_worker,
        FLAGS_ycsb_num_ops_per_txn, FLAGS_ycsb_write_ratio,
        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
        (FLAGS_elastic_workload > 0 ? topo.getCoreCount() : FLAGS_num_workers),
        g_num_partitions, FLAGS_layout_column_store, FLAGS_ycsb_num_col_upd,
        FLAGS_ycsb_num_col_read, FLAGS_ycsb_num_col_read_offset);
  }

  scheduler::WorkerPool::getInstance().init(
      bench, (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
      g_num_partitions, FLAGS_worker_sched_mode, FLAGS_num_iter_per_worker,
      (FLAGS_elastic_workload > 0 ? true : false));

  schema->report();
  profiling::resume();
  scheduler::WorkerPool::getInstance().start_workers();

  if (FLAGS_migrate_worker > 0) {
    proteus::utils::timed_func::interval_runner(
        [] {
          scheduler::WorkerPool::getInstance().print_worker_stats_diff();
          scheduler::WorkerPool::getInstance().migrate_worker();
        },
        (FLAGS_migrate_worker * 1000));
  }

  if (FLAGS_elastic_workload > 0) {
    uint curr_active_worker = 1;
    bool removal = false;
    const auto& worker_cores = topo.getCores();
    proteus::utils::timed_func::interval_runner(
        [curr_active_worker, removal, &worker_cores]() mutable {
          if (curr_active_worker < FLAGS_num_workers && !removal) {
            LOG(INFO) << "Adding Worker: " << (curr_active_worker + 1);
            scheduler::WorkerPool::getInstance().add_worker(
                &worker_cores.at(curr_active_worker));
            curr_active_worker++;
          } else if (curr_active_worker > 2 && removal) {
            LOG(INFO) << "Removing Worker: " << (curr_active_worker - 1);
            scheduler::WorkerPool::getInstance().remove_worker(
                &worker_cores.at(curr_active_worker - 1));
            curr_active_worker--;
          } else {
            removal = true;
          }
        },
        (FLAGS_elastic_workload * 1000));
  }

  /* Report stats every 1 sec */
  if (FLAGS_report_stat_sec > 0) {
    LOG(INFO) << "FLAGS_report_stat_sec: " << FLAGS_report_stat_sec;
    proteus::utils::timed_func::interval_runner(
        [] { scheduler::WorkerPool::getInstance().print_worker_stats_diff(); },
        (FLAGS_report_stat_sec * 1000));
  }

  if (FLAGS_switch_master_sec > 0) {
    proteus::utils::timed_func::interval_runner(
        [] { txn::TransactionManager::getInstance().snapshot(); },
        (FLAGS_switch_master_sec * 1000));
  }

  /* This shouldn't be a sleep, but this thread should sleep until all workers
   * finished required number of txns. but dilemma here is that either the
   * worker executes transaction forever (using a benchmark) or finished after
   * executing certain number of txns/iterations. */
  usleep((FLAGS_runtime - 1) * 1000000);

  proteus::utils::timed_func::terminate_all_timed();
  usleep(1000000);  // sanity sleep so that async threads can exit gracefully.

  profiling::pause();
  LOG(INFO) << "Tear Down Initiated";
  scheduler::WorkerPool::getInstance().shutdown(true);

  LOG(INFO) << "DBMS Storage: ";
  storage::Schema::getInstance().report();

  LOG(INFO) << "Storage teardown: ";
  storage::Schema::getInstance().teardown();

  LOG(INFO) << "Bench deinit: ";
  bench->deinit();
  delete bench;

  StorageManager::getInstance().unloadAll();
  return 0;
}
