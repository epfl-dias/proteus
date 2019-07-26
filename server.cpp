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

#include <unistd.h>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>
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
//#include "utils/utils.hpp"

#include "lib/cxxopts.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

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

void check_num_upd_by_bits();

int main(int argc, char** argv) {
  cxxopts::Options options("AEOLUS", "Column-major, Elastic OLTP Engine");

  options.add_options()("d,debug", "Enable debugging")(
      "w,workers", "Number of Workers", cxxopts::value<uint>())(
      "r,write_ratio", "Reader to writer ratio", cxxopts::value<double>())(
      "t,theta", "Zipf theta", cxxopts::value<double>())(
      //"g,gc_mode", "GC Mode: 1-Snapshot, 2-TupleGC", cxxopts::value<uint>())(
      "b,benchmark", "Benchmark: 0:YCSB, 1:TPC-C (gen),  2:TPC-C (csv)",
      cxxopts::value<uint>())("c,ycsb_num_cols", "Number of YCSB Columns",
                              cxxopts::value<uint>());

  auto result = options.parse(argc, argv);

  // result.count("option")
  // result["opt"].as<type>()
  // cxxopts::value<std::string>()->default_value("value")
  // cxxopts::value<std::string>()->implicit_value("implicit")

  // conf
  int num_workers = MAX_WORKERS;  // std::thread::hardware_concurrency();
  uint gc_mode = 1;
  uint bechnmark = 0;  // default: YCSB

  // TPC-C vars

  // ycsb vars
  // (10-G * (1024^3))/(8*10-num_field)
  int num_fields = 1;  // 2;
  // int num_records = 134217728;  // 10GB
  // int num_records = 268435456;  // 20GB
  int num_records = 72000000;
  // int num_records = 1000000;
  double theta = 0.5;
  int num_iterations_per_worker = 1000000;
  int num_ops_per_txn = 10;
  double write_threshold = 0.5;

  if (result.count("w") > 0) {
    num_workers = result["w"].as<uint>();
  }
  if (result.count("b") > 0) {
    bechnmark = result["b"].as<uint>();
  }

  if (result.count("r") > 0) {
    write_threshold = result["r"].as<double>();
  }
  if (result.count("t") > 0) {
    theta = result["t"].as<double>();
  }
  if (result.count("g") > 0) {
    gc_mode = result["g"].as<uint>();
  }
  if (result.count("c") > 0) {
    num_fields = result["c"].as<uint>();
  }

  std::cout << "------- AELOUS ------" << std::endl;
  std::cout << "# Workers: " << num_workers << std::endl;
  std::cout << "---------------------" << std::endl;

  // INIT
  std::cout << "Initializing..." << std::endl;
  std::cout << "\tInitializing memory manager..." << std::endl;
  storage::MemoryManager::init();

  if (!HTAP) {
    std::cout << scheduler::Topology::getInstance() << std::endl;
  }

  std::cout << "------------------------------------" << std::endl;

  if (HTAP) {
    std::cout << "\tInitializing communication manager..." << std::endl;
    scheduler::CommManager::getInstance().init();
  }

  // txn::TransactionManager::getInstance().init();

  /* ------------------------------------ */

  // TODO: set affinity for the master server thread.

  // std::cout << "hardcoding execution location to NUMA node ID: 0" <<
  // std::endl; topology::getInstance().getCpuNumaNodes()[0] const auto
  // &exec_node =
  //   scheduler::Topology::getInstance().getCpuNumaNodeById(0);

  // set_exec_location_on_scope d(exec_node);

  // init benchmark
  bench::Benchmark* bench = nullptr;
  if (bechnmark == 1) {
    bench = new bench::TPCC("TPCC", num_workers);

  } else if (bechnmark == 2) {
    bench = new bench::TPCC("TPCC", 100, 0,
                            "/home/raza/local/chBenchmark_1_0/w100/raw");
  } else {  // Defult YCSB

    std::cout << "Write Threshold: " << write_threshold << std::endl;
    std::cout << "Theta: " << theta << std::endl;
    bench = new bench::YCSB("YCSB", num_fields, num_records, theta,
                            num_iterations_per_worker, num_ops_per_txn,
                            write_threshold, num_workers, num_workers);
  }

  bench->load_data(num_workers);

  /* As soon as worker starts, they start transactions. so make sure you setup
   * everything needed for benchmark transactions before hand.
   */

  /* Currently, worker looks for the tasks(TXN func pointers) in the queue, if
   * the queue has nothing, it will get and execute a txn from the benchmark
   */

  scheduler::AffinityManager::getInstance().set(
      &scheduler::Topology::getInstance().get_worker_cores()->front());
  scheduler::WorkerPool::getInstance().init(bench);
  __itt_resume();
  scheduler::WorkerPool::getInstance().start_workers(num_workers);

  /* Report stats every 1 sec */
  // timed_func::interval_runner(
  //    [] { scheduler::WorkerPool::getInstance().print_worker_stats(); },
  //    1000);

  /* This shouldnt be a sleep, but this thread should sleep until all workers
   * finished required number of txns. but dilemma here is that either the
   * worker executes transaction forever (using a benchmark) or finished after
   * executing certain number of txns/iterations. */

  usleep(RUNTIME * 1000000);

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
  if (HTAP) {
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
      storage::Schema::getInstance().getAllTable();

  for (auto& tbl : tables) {
    storage::ColumnStore* tmp = (storage::ColumnStore*)tbl;

    tmp->num_upd_tuples();
  }
}
