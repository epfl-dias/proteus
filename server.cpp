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

#include <unistd.h>
#include <functional>
#include <iostream>
#include <tuple>
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"

#include "benchmarks/ycsb.hpp"
#include "transactions/transaction_manager.hpp"

#include "indexes/hash_index.hpp"

#include "utils/utils.hpp"

#define RUNTIME 5000000

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

int main() {
  // INIT
  std::cout << "Initializing..." << std::endl;
  std::cout << "\tInitializing memory manager..." << std::endl;
  storage::MemoryManager::init();
  std::cout << scheduler::Topology::getInstance() << std::endl;

  std::cout << "------------------------------------" << std::endl;

  // txn::TransactionManager::getInstance().init();

  return 0;

  /* ------------------------------------ */

  // TODO: set affinity for the master server thread.

  // std::cout << "hardcoding execution location to NUMA node ID: 0" <<
  // std::endl; topology::getInstance().getCpuNumaNodes()[0] const auto
  // &exec_node =
  //   scheduler::Topology::getInstance().getCpuNumaNodeById(0);

  // set_exec_location_on_scope d(exec_node);

  // init benchmark
  bench::YCSB* ycsb_bench = new bench::YCSB();
  /* As soon as worker starts, they start transactions. so make sure you setup
   * everything needed for benchmark transactions before hand.
   */

  /* Currently, worker looks for the tasks(TXN func pointers) in the queue, if
   * the queue has nothing, it will get and execute a txn from the benchmark
   */

  scheduler::WorkerPool::getInstance().init(ycsb_bench);
  scheduler::WorkerPool::getInstance().start_workers(1);

  /* This shouldnt be a sleep, but this thread should sleep until all workers
   * finished required number of txns. but dilemma here is that either the
   * worker executes transaction forever (using a benchmark) or finished after
   * executing certain number of txns/iterations. */

  usleep(RUNTIME);
  scheduler::WorkerPool::getInstance().shutdown(true);

  /* TODO: gather stats about every thread or something*/

  std::cout << "Tead Down Inititated" << std::endl;

  std::cout << "\tShutting  down memory manager" << std::endl;
  storage::MemoryManager::destroy();

  return 0;
}
