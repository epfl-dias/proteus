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
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>

// OLTP Engine
#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "oltp-cli-flags.hpp"
#include "scheduler/worker.hpp"
#include "storage/table.hpp"
#include "topology/topology.hpp"
#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

// Bench Includes

//#include "benchmarks/micro_ssb.hpp"
#include "bench-cli-flags.hpp"
#include "bench/ycsb.hpp"
#include "tpcc/tpcc_64.hpp"

// Platform Includes
#include "common/common.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/profiling.hpp"

int main(int argc, char** argv) {
  auto olap = proteus::from_cli::olap(
      "Simple command line interface for aeolus", &argc, &argv);

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();
  set_exec_location_on_scope d{nodes[0]};

  if (FLAGS_num_workers == 0)
    FLAGS_num_workers = topology::getInstance().getCoreCount();

  if (FLAGS_num_partitions == 0) {
    if (FLAGS_elastic_workload > 0) {
      g_num_partitions = topology::getInstance().getCpuNumaNodeCount();
    } else {
      g_num_partitions =
          ((FLAGS_num_workers - 1) /
           topology::getInstance().getCpuNumaNodes()[0].local_cores.size()) +
          1;
    }

  } else {
    g_num_partitions = FLAGS_num_partitions;
  }

  LOG(INFO) << "PARTITIONS: " << g_num_partitions << std::endl;

  g_delta_size = FLAGS_delta_size;
  LOG(INFO) << "DeltaSize: " << FLAGS_delta_size;
  g_delta_size = 6;

  storage::Schema* schema = &storage::Schema::getInstance();

  // init benchmark
  bench::Benchmark* bench = nullptr;
  if (FLAGS_benchmark == 1) {
    if (FLAGS_tpcc_num_wh == 0) FLAGS_tpcc_num_wh = FLAGS_num_workers;

    bench =
        new bench::TPCC("TPCC", FLAGS_tpcc_num_wh,
                        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
                        FLAGS_layout_column_store, FLAGS_ch_scale_factor);

  } else if (FLAGS_benchmark == 2) {
    if (FLAGS_tpcc_num_wh == 0) FLAGS_tpcc_num_wh = FLAGS_num_workers;
    bench =
        new bench::TPCC("TPCC", FLAGS_tpcc_num_wh,
                        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
                        FLAGS_layout_column_store, FLAGS_ch_scale_factor,
                        FLAGS_tpcc_dist_threshold, FLAGS_tpcc_csv_dir);

  } else if (FLAGS_benchmark == 3) {
    // bench = new bench::MicroSSB();

  } else {  // Defult YCSB

    std::cout << "Write Threshold: " << FLAGS_ycsb_write_ratio << std::endl;
    std::cout << "Theta: " << FLAGS_ycsb_zipf_theta << std::endl;
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
        (FLAGS_elastic_workload > 0 ? topology::getInstance().getCoreCount()
                                    : FLAGS_num_workers),
        g_num_partitions, FLAGS_layout_column_store, FLAGS_ycsb_num_col_upd,
        FLAGS_ycsb_num_col_read, FLAGS_ycsb_num_col_read_offset,
        FLAGS_cdf_path);
  }

  scheduler::WorkerPool::getInstance().init(
      bench, (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
      g_num_partitions, FLAGS_worker_sched_mode, FLAGS_num_iter_per_worker,
      (FLAGS_elastic_workload > 0 ? true : false));

  schema->report();
  profiling::resume();
  scheduler::WorkerPool::getInstance().start_workers();

  if (FLAGS_migrate_worker > 0) {
    timed_func::interval_runner(
        [] {
          scheduler::WorkerPool::getInstance().print_worker_stats_diff();
          scheduler::WorkerPool::getInstance().migrate_worker();
        },
        (FLAGS_migrate_worker * 1000));
  }

  if (FLAGS_elastic_workload > 0) {
    uint curr_active_worker = 1;
    bool removal = false;
    const auto& worker_cores = topology::getInstance().getCores();
    timed_func::interval_runner(
        [curr_active_worker, removal, worker_cores]() mutable {
          if (curr_active_worker < FLAGS_num_workers && !removal) {
            std::cout << "Adding Worker: " << (curr_active_worker + 1)
                      << std::endl;
            scheduler::WorkerPool::getInstance().add_worker(
                &worker_cores.at(curr_active_worker));
            curr_active_worker++;
          } else if (curr_active_worker > 2 && removal) {
            std::cout << "Removing Worker: " << (curr_active_worker - 1)
                      << std::endl;
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
    timed_func::interval_runner(
        [] { scheduler::WorkerPool::getInstance().print_worker_stats_diff(); },
        (FLAGS_report_stat_sec * 1000));
  }

  if (FLAGS_switch_master_sec > 0) {
    timed_func::interval_runner(
        [] { txn::TransactionManager::getInstance().snapshot(); },
        (FLAGS_switch_master_sec * 1000));
  }

  // timed_func::interval_runner(
  //     [] { txn::TransactionManager::getInstance().snapshot(); }, (5 * 1000));

  /* This shouldnt be a sleep, but this thread should sleep until all workers
   * finished required number of txns. but dilemma here is that either the
   * worker executes transaction forever (using a benchmark) or finished after
   * executing certain number of txns/iterations. */

  usleep((FLAGS_runtime - 1) * 1000000);

  timed_func::terminate_all_timed();
  usleep(1000000);  // sanity sleep so that async threads can exit gracefully.

  profiling::pause();
  LOG(INFO) << "Tear Down Initiated";
  scheduler::WorkerPool::getInstance().shutdown(true);

  LOG(INFO) << "DBMS Storage: ";
  storage::Schema::getInstance().report();

  // std::cout << "AFTER" << std::endl;
  // for (auto& tb : storage::Schema::getInstance().getAllTables()) {
  //   ((storage::ColumnStore*)tb)->num_upd_tuples();
  // }

  storage::Schema::getInstance().teardown();

  bench->deinit();
  delete bench;
  return 0;
}
