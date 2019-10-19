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

#include <gflags/gflags.h>
#include <unistd.h>

#include <bitset>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>

#include "glo.hpp"

// #include "benchmarks/micro_ssb.hpp"
#include "benchmarks/tpcc_64.hpp"
#include "benchmarks/ycsb.hpp"
#include "indexes/hash_index.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

// proteus
//#if PROTEUS_MEM_MANAGER
#include "common/common.hpp"
//#endif

/*std::ostream& operator<<(std::ostream& os, int64_t i) {
  char buf[20];
  sprintf(buf, "%li", i);
  os << buf;
  return os;
}
std::ostream& operator<<(std::ostream& os, uint64_t i) {
  char buf[20];
  sprintf(buf, "%lu", i);
  os << buf;
  return os;
}*/

DEFINE_bool(debug, false, "Debug mode");
DEFINE_uint64(num_workers, 0, "Number of txn-workers");
DEFINE_uint64(benchmark, 0,
              "Benchmark: 0:YCSB, 1:TPC-C (gen),  2:TPC-C (csv), 3:Micro-SSB");
DEFINE_uint64(num_partitions, 0,
              "Number of storage partitions ( round robin NUMA nodes)");
DEFINE_int64(num_iter_per_worker, -1, "# of iterations per worker");
DEFINE_uint64(runtime, 60, "Duration of experiments in seconds");
DEFINE_uint64(delta_size, 8, "Size of delta storage in GBs.");
DEFINE_bool(layout_column_store, true, "True: ColumnStore / False: RowStore");
DEFINE_uint64(worker_sched_mode, 0,
              "Scheduling of worker: 0-default, 1-interleaved-even, "
              "2-interleaved-odd, 3-reversed.");
DEFINE_uint64(report_stat_sec, 0, "Report stats every x secs");
DEFINE_uint64(elastic_workload, 0, "if > 0, add a worker every x seconds");
DEFINE_uint64(migrate_worker, 0, "if > 0, migrate worker to other side");
DEFINE_uint64(switch_master_sec, 0, "if > 0, add a worker every x seconds");

// YCSB
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");
DEFINE_double(ycsb_zipf_theta, 0.5, "YCSB - Zipfian theta");
DEFINE_uint64(ycsb_num_cols, 1, "YCSB - # Columns");
DEFINE_uint64(ycsb_num_ops_per_txn, 10, "YCSB - # ops / txn");
DEFINE_uint64(ycsb_num_records, 0, "YCSB - # records");

// TPC-C
DEFINE_uint64(tpcc_num_wh, 0, "TPC-C - # of Warehouses ( 0 = one per worker");
DEFINE_uint64(ch_scale_factor, 0, "CH-Bench scale factor");
DEFINE_uint64(tpcc_dist_threshold, 0, "TPC-C - Distributed txn threshold");
DEFINE_string(tpcc_csv_dir, "/scratch/data/ch100w/raw",
              "CSV Dir for loading tpc-c data (bench-2)");

// void check_num_upd_by_bits();

int main(int argc, char** argv) {
  // uint64_t c = 1000000;

  // std::bitset<64000000> b1;
  // b1.flip();
  // std::cout << "b: " << b1.count() << std::endl;

  // return 0;
  // std::fstream myfile;
  // myfile = std::fstream("part.csv.p_stocklevel_uniform",
  //                       std::ios::out | std::ios::binary);
  // // Here would be some error handling
  // unsigned int seed = rand();

  // for (uint i = 0; i < 1400000; ++i) {
  //   // Some calculations to fill a[]
  //   uint32_t a = URand(&seed, 1000, 10000);
  //   myfile.write((char*)&a, sizeof(uint32_t));
  // }
  // myfile.close();
  // return 0;
  proteus::init();
  gflags::SetUsageMessage("Simple command line interface for aeolus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // google::InitGoogleLogging(argv[0]);
  // FLAGS_logtostderr = 1; // FIXME: the command line flags/defs seem to
  // fail...

  if (FLAGS_num_workers == 0)
    FLAGS_num_workers = scheduler::Topology::getInstance().getCoreCount();

  if (FLAGS_num_partitions == 0) {
    g_num_partitions = scheduler::Topology::getInstance().getCpuNumaNodeCount();
  } else {
    g_num_partitions = FLAGS_num_partitions;
  }

  g_delta_size = FLAGS_delta_size;

  std::cout << "------- AELOUS ------" << std::endl;
  std::cout << "# Workers: " << FLAGS_num_workers << std::endl;
  std::cout << "---------------------" << std::endl;

  // INIT
  std::cout << "Initializing..." << std::endl;
  std::cout << "\tInitializing memory manager..." << std::endl;
  storage::MemoryManager::init();

#if PROTEUS_MEM_MANAGER
  proteus::init();
#elif HTAP_RM_SERVER
  std::cout << "\tInitializing communication manager..." << std::endl;
  scheduler::CommManager::getInstance().init();
#else
  std::cout << scheduler::Topology::getInstance() << std::endl;
  std::cout << "------------------------------------" << std::endl;
#endif

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
    ;
  } else {  // Defult YCSB

    std::cout << "Write Threshold: " << FLAGS_ycsb_write_ratio << std::endl;
    std::cout << "Theta: " << FLAGS_ycsb_zipf_theta << std::endl;
    if (FLAGS_ycsb_num_records == 0) {
      FLAGS_ycsb_num_records = FLAGS_num_workers * 1000000;
    }

    bench =
        new bench::YCSB("YCSB", FLAGS_ycsb_num_cols, FLAGS_ycsb_num_records,
                        FLAGS_ycsb_zipf_theta, FLAGS_num_iter_per_worker,
                        FLAGS_ycsb_num_ops_per_txn, FLAGS_ycsb_write_ratio,
                        (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
                        scheduler::Topology::getInstance().getCoreCount(),
                        FLAGS_num_partitions, FLAGS_layout_column_store);
  }

  scheduler::WorkerPool::getInstance().init(
      bench, (FLAGS_elastic_workload > 0 ? 1 : FLAGS_num_workers),
      FLAGS_num_partitions, FLAGS_worker_sched_mode, FLAGS_num_iter_per_worker,
      (FLAGS_elastic_workload > 0 ? true : false));

  schema->report();
  __itt_resume();
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
    const auto& worker_cores = scheduler::Topology::getInstance().getCores();
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

  std::cout << "Tear Down Inititated" << std::endl;
  __itt_pause();
  scheduler::WorkerPool::getInstance().shutdown(true);

  if (HTAP_RM_SERVER) {
    std::cout << "\tShutting down communication manager..." << std::endl;
    scheduler::CommManager::getInstance().shutdown();
  }

  // std::cout << "AFTER" << std::endl;
  // for (auto& tb : storage::Schema::getInstance().getAllTables()) {
  //   ((storage::ColumnStore*)tb)->num_upd_tuples();
  // }

  // TODO: cleanup
  // storage::Schema::getInstance().teardown();

  return 0;
}
