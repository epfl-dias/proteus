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

#define RUNTIME 60  // seconds

#include <gflags/gflags.h>
#include <unistd.h>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>

#include "benchmarks/micro_ssb.hpp"
#include "benchmarks/tpcc.hpp"
#include "benchmarks/ycsb.hpp"
#include "glo.hpp"
#include "indexes/hash_index.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

#include "lib/cxxopts.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

// proteus
#include "common/common.hpp"

// TODO: a race condition exists in acquiring write lock and updating the
// version, a read might read at the same time as readers are not blocked in any
// manner. for this, read andy pavlos paper on MVCC and see how to do it
// otherwise use a spinlock when actually modifying the index val.

// From PROTEUS
//#include <codegen/memory/memory-manager.hpp>
//#include <codegen/topology/affinity_manager.hpp>
//#include <codegen/topology/topology.hpp>

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

// TODO: Add warm-code!!

DEFINE_uint64(num_workers, 0, "Number of txn-workers");
DEFINE_uint64(benchmark, 0,
              "Benchmark: 0:YCSB, 1:TPC-C (gen),  2:TPC-C (csv), 3:Micro-SSB");
DEFINE_uint64(num_partitions, 1,
              "Number of storage partitions ( round robin NUMA nodes)");
DEFINE_int64(num_iter_per_worker, -1, "# of iterations per worker");
DEFINE_uint64(runtime, 60, "Duration of experiments in seconds");

DEFINE_uint64(worker_sched_mode, 0,
              "Scheduling of worker: 0-default, 1-interleaved-even, "
              "2-interleaved-odd, 3-reversed.");

// YCSB
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");
DEFINE_double(ycsb_zipf_theta, 0.5, "YCSB - Zipfian theta");
DEFINE_uint64(ycsb_num_cols, 1, "YCSB - # Columns");
DEFINE_uint64(ycsb_num_ops_per_txn, 10, "YCSB - # ops / txn");
DEFINE_uint64(ycsb_num_records, 72000000, "YCSB - # records");

// TPC-C
DEFINE_uint64(tpcc_num_wh, 0, "TPC-C - # of Warehouses ( 0 = one per worker");
DEFINE_uint64(tpcc_dist_threshold, 0, "TPC-C - Distributed txn threshold");
DEFINE_string(tpcc_csv_dir, "/scratch/data/ch100w/raw",
              "CSV Dir for loading tpc-c data (bench-2)");

void check_num_upd_by_bits();

int main(int argc, char** argv) {
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

  gflags::SetUsageMessage("Simple command line interface for aeolus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // google::InitGoogleLogging(argv[0]);
  // FLAGS_logtostderr = 1; // FIXME: the command line flags/defs seem to
  // fail...

  if (FLAGS_num_workers == 0)
    FLAGS_num_workers = scheduler::Topology::getInstance().getCoreCount();

  if (FLAGS_tpcc_num_wh == 0) FLAGS_tpcc_num_wh = FLAGS_num_workers;

  std::cout << "------- AELOUS ------" << std::endl;
  std::cout << "# Workers: " << FLAGS_num_workers << std::endl;
  std::cout << "---------------------" << std::endl;

  // INIT
  std::cout << "Initializing..." << std::endl;
  std::cout << "\tInitializing memory manager..." << std::endl;
  storage::MemoryManager::init();

  if (HTAP_DOUBLE_MASTER) {
    proteus::init(false);
  } else if (HTAP_RM_SERVER) {
    std::cout << "\tInitializing communication manager..." << std::endl;
    scheduler::CommManager::getInstance().init();
  } else {
    std::cout << scheduler::Topology::getInstance() << std::endl;
    std::cout << "------------------------------------" << std::endl;
  }

  storage::Schema* schema = &storage::Schema::getInstance();

  // txn::TransactionManager::getInstance().init();

  /* ------------------------------------ */

  // TODO: set affinity for the master server thread.

  // std::cout << "hardcoding execution location to NUMA node ID: 0" <<
  // std::endl; topology::getInstance().getCpuNumaNodes()[0] const auto
  // &exec_node =
  //   scheduler::Topology::getInstance().getCpuNumaNodeById(0);

  // set_exec_location_on_scope d(exec_node);

  //---------------
  // bench::Benchmark* benchm = new bench::MicroSSB();
  // std::cout << "creation done" << std::endl;
  // benchm->load_data(num_workers);

  // return 0;
  //---------------

  // init benchmark
  bench::Benchmark* bench = nullptr;
  if (FLAGS_benchmark == 1) {
    bench = new bench::TPCC("TPCC", FLAGS_tpcc_num_wh);
    bench->load_data(FLAGS_num_workers);

  } else if (FLAGS_benchmark == 2) {
    bench = new bench::TPCC("TPCC", FLAGS_tpcc_num_wh,
                            FLAGS_tpcc_dist_threshold, FLAGS_tpcc_csv_dir);
  } else if (FLAGS_benchmark == 3) {
    bench = new bench::MicroSSB();

  } else {  // Defult YCSB

    std::cout << "Write Threshold: " << FLAGS_ycsb_write_ratio << std::endl;
    std::cout << "Theta: " << FLAGS_ycsb_zipf_theta << std::endl;
    bench = new bench::YCSB("YCSB", FLAGS_ycsb_num_cols, FLAGS_ycsb_num_records,
                            FLAGS_ycsb_zipf_theta, FLAGS_num_iter_per_worker,
                            FLAGS_ycsb_num_ops_per_txn, FLAGS_ycsb_write_ratio,
                            FLAGS_num_workers,
                            scheduler::Topology::getInstance().getCoreCount());
  }

  /* As soon as worker starts, they start transactions. so make sure you setup
   * everything needed for benchmark transactions before hand.
   */

  /* Currently, worker looks for the tasks(TXN func pointers) in the queue, if
   * the queue has nothing, it will get and execute a txn from the benchmark
   */

  scheduler::AffinityManager::getInstance().set(
      &scheduler::Topology::getInstance().getCores().front());

  scheduler::WorkerPool::getInstance().init(bench);

  __itt_resume();

  scheduler::WorkerPool::getInstance().start_workers(
      FLAGS_num_workers, FLAGS_num_partitions, FLAGS_worker_sched_mode,
      FLAGS_num_iter_per_worker);

  /* Report stats every 1 sec */
  // timed_func::interval_runner(
  //    [] { scheduler::WorkerPool::getInstance().print_worker_stats(); },
  //    1000);

  /* This shouldnt be a sleep, but this thread should sleep until all workers
   * finished required number of txns. but dilemma here is that either the
   * worker executes transaction forever (using a benchmark) or finished after
   * executing certain number of txns/iterations. */

  usleep(FLAGS_runtime * 1000000);

  // usleep((RUNTIME/2) * 1000000);

  // uint64_t last_epoch =
  // txn::TransactionManager::getInstance().switch_master(curr_master);

  // usleep((RUNTIME/2) * 1000000);

  /* TODO: gather stats about every thread or something*/
  // scheduler::WorkerPool::getInstance().print_worker_stats();

  std::cout << "Tear Down Inititated" << std::endl;
  scheduler::WorkerPool::getInstance().shutdown(true);
  __itt_pause();

  std::cout << "\tShutting down memory manager" << std::endl;

  if (HTAP_RM_SERVER) {
    std::cout << "\tShutting down communication manager..." << std::endl;
    scheduler::CommManager::getInstance().shutdown();
  }

  // std::cout << "----" << std::endl;
  // check_num_upd_by_bits();

  std::cout << "----" << std::endl;
  storage::Schema::getInstance().teardown();
  std::cout << "----" << std::endl;

  // storage::MemoryManager::destroy();

  return 0;
}

void check_num_upd_by_bits() {
  std::vector<storage::Table*> tables =
      storage::Schema::getInstance().getAllTables();

  for (auto& tbl : tables) {
    storage::ColumnStore* tmp = (storage::ColumnStore*)tbl;

    tmp->num_upd_tuples();
  }
}
