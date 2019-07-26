/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#define NUM_TPCH_QUERIES 3
#define NUM_OLAP_REPEAT 10

#include <filesystem>
#include <unistd.h>

#include <gflags/gflags.h>

#include "benchmarks/tpcc.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

#include "codegen/communication/comm-manager.hpp"
#include "codegen/memory/block-manager.hpp"
#include "codegen/memory/memory-manager.hpp"
#include "codegen/plan/prepared-statement.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/util/jit/pipeline.hpp"
#include "codegen/util/parallel-context.hpp"
#include "storage/storage-manager.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

DEFINE_bool(query_topology, false, "Print the system topology and exit");
DEFINE_bool(trace_allocations, false,
            "Trace memory allocation and leaks (requires a build with "
            "undefined NDEBUG)");
DEFINE_bool(inc_buffers, false, "Use bigger block pools");
DEFINE_uint64(num_olap_clients, 1, "Number of OLAP clients");
DEFINE_uint64(num_oltp_clients, 1, "Number of OLTP clients");
DEFINE_string(plan_json, "", "Plan to execute, takes priority over plan_dir");
DEFINE_string(plan_dir, "inputs/plans/cpu-ssb",
              "Directory with plans to be executed");
DEFINE_string(inputs_dir, "inputs/", "Data and catalog directory");
DEFINE_bool(run_oltp, true, "Run OLTP");

std::vector<pid_t> children;

struct OLAP_STATS {
  bool shutdown;
  uint64_t runtime_stats[NUM_OLAP_REPEAT * NUM_TPCH_QUERIES];
};

void init_olap_warmup() { proteus::init(FLAGS_inc_buffers); }

std::vector<PreparedStatement>
init_olap_sequence(int &client_id, const topology::cpunumanode &numa_node) {
  // chdir("/home/raza/local/htap/opt/pelago");

  time_block t("TcodegenTotal_: ");
  // TODO: codegen all the TPC-H queries beforehand. keep them ready anyway.
  // BUG: if we generate code before fork, will the codegen will be valid after
  // a fork? as in memory pointers which are pushed in codegen, are they valid
  // on runtime?

  std::vector<PreparedStatement> stmts;

  // std::string label_prefix("htap_server_" + std::to_string(getpid()) + "_");
  std::string label_prefix("htap_" + std::to_string(getpid()) + "_c" +
                           std::to_string(client_id) + "_q");

  if (FLAGS_plan_json.length()) {
    LOG(INFO) << "Compiling Plan:" << FLAGS_plan_json << std::endl;

    stmts.emplace_back(PreparedStatement::from(
        FLAGS_plan_json, label_prefix + std::to_string(0), FLAGS_inputs_dir));
  } else {
    uint i = 0;
    for (const auto &entry :
         std::filesystem::directory_iterator(FLAGS_plan_dir)) {

      if (entry.path().filename().string()[0] == '.')
        continue;

      if (entry.path().extension() == ".json") {

        std::string plan_path = entry.path().string();
        std::string label = label_prefix + std::to_string(i++);

        LOG(INFO) << "Compiling Query:" << plan_path << std::endl;

        stmts.emplace_back(
            PreparedStatement::from(plan_path, label, FLAGS_inputs_dir));
      }
    }
  }

  return stmts;
}
void run_olap_sequence(int &client_id,
                       std::vector<PreparedStatement> &olap_queries,
                       struct OLAP_STATS *olap_stats,
                       const topology::cpunumanode &numa_node) {
  // TODO: execute the generate pipelines. for TPC-H, 22 query sequence, and
  // return.
  // TODO: add the stats of query runtime to a global-shared stats file, maybe
  // mmap(MAP_SHARED) for keeping global statistics. client_id here is to keep
  // the record for each OLAP session for stats file

  // There should be some laundry work here, because one the codegen is done,
  // the memory pointers and everything is already set. we need to update the
  // scan pointers ( and limit) because this is happening after a fresh fork
  // meaning underlying things have change. if not pointers, the size (#
  // records) has definitely changed.

  // Make affinity deterministic
  exec_location{numa_node}.activate();

  for (size_t i = 0, j = 0; i < NUM_OLAP_REPEAT; i++) {

    for (auto &q : olap_queries) {

      std::chrono::time_point<std::chrono::system_clock> start =
          std::chrono::system_clock::now();

      q.execute();

      olap_stats->runtime_stats[j] =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now() - start)
              .count();
      j++;
    }
  }
}

void shutdown_olap() {
  LOG(INFO) << "Unloading files...";
  StorageManager::unloadAll();

  LOG(INFO) << "Shuting down memory manager...";
  MemoryManager::destroy();

  LOG(INFO) << "Shut down finished";
}

bench::Benchmark *init_oltp(int num_warehouses, std::string csv_path) {
  // initialize OLAP..

  bench::Benchmark *bench = nullptr;
  if (csv_path.length() < 2) {
    bench = new bench::TPCC("TPCC", num_warehouses);
  } else {
    bench = new bench::TPCC("TPCC", 100, 0, csv_path);
  }

  bench->load_data(num_warehouses);

  return bench;
}

void run_oltp(const scheduler::cpunumanode &numa_node, uint num_workers,
              bench::Benchmark *bench) {

  scheduler::AffinityManager::getInstance().set(
      &scheduler::Topology::getInstance().get_worker_cores()->front());

  // exec_location{(topology::cpunumanode)numa_node}.activate();
  auto &wp = scheduler::WorkerPool::getInstance();
  wp.init(bench);
  wp.start_workers(num_workers);
}

void shutdown_oltp(bool print_stat = true) {
  scheduler::WorkerPool::getInstance().shutdown(print_stat);
}

void *get_shm(std::string name, size_t size) {
  int fd = shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  if (fd == -1) {
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
               << std::endl;
    assert(false);
  }

  if (ftruncate(fd, size) < 0) { //== -1){
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
               << std::endl;
    assert(false);
  }

  void *mem = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  if (!mem) {
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
               << std::endl;
    assert(false);
  }

  close(fd);
  return mem;
}

[[noreturn]] void kill_orphans_and_widows(int s) {
  for (auto &pid : children) {
    kill(pid, SIGTERM);
  }
  exit(1);
}

void register_handler() { signal(SIGINT, kill_orphans_and_widows); }

int main(int argc, char *argv[]) {
  register_handler();
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1; // FIXME: the command line flags/defs seem to fail...

  // google::InstallFailureSignalHandler();

  set_trace_allocations(FLAGS_trace_allocations);

  struct OLAP_STATS *olap_stats = (struct OLAP_STATS *)get_shm(
      "olap_stats_" + std::to_string(getpid()),
      sizeof(struct OLAP_STATS) * FLAGS_num_olap_clients);

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {

    olap_stats[i].shutdown = true;
    for (int j = 0; j < (NUM_TPCH_QUERIES * NUM_OLAP_REPEAT); j++) {
      olap_stats[i].runtime_stats[j] = 0;
    }
  }

  auto &txn_topo = scheduler::Topology::getInstance();
  auto &txn_nodes = txn_topo.getCpuNumaNodes();
  bench::Benchmark *oltp_bench = init_oltp(txn_nodes[0].local_cores.size(), "");
  // bench::Benchmark *oltp_bench = init_oltp(4, "");

  __itt_resume();

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    olap_stats[i].shutdown = false;
    pid_t tmp = fork();

    if (tmp == 0) {
      // run olap stuff
      // warmup should be inside or outside? cpu can be outside but dont know
      // about gpu init stuff can be forked or not.
      init_olap_warmup();

      auto &topo = topology::getInstance();
      auto &nodes = topo.getCpuNumaNodes();

      assert(nodes.size() >= 2);
      assert(FLAGS_num_oltp_clients <= nodes[0].local_cores.size());

      exec_location{nodes[1]}.activate();

      LOG(INFO) << "[SERVER-COW] OLAP Client #" << i
                << ": Compiling OLAP Sequence";
      std::vector<PreparedStatement> olap_queries =
          init_olap_sequence(i, nodes[1]);

      LOG(INFO) << "[SERVER-COW] OLAP Client #" << i
                << ": Running OLAP Sequence";
      run_olap_sequence(i, olap_queries, olap_stats + i, nodes[1]);

      LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Shutdown";
      shutdown_olap();
      olap_stats[i].shutdown = true;
      break;
    } else {
      children.emplace_back(tmp);
    }
  }

  // some child process
  if (children.size() != FLAGS_num_olap_clients)
    return 0;

  if (FLAGS_run_oltp)
    run_oltp(txn_nodes[0], txn_nodes[0].local_cores.size(), oltp_bench);

  // Termination Condition.
  uint client_ctr = 0;
  while (true) {
    if (client_ctr == FLAGS_num_olap_clients)
      break;
    else if (olap_stats[client_ctr].shutdown)
      client_ctr++;
    else
      usleep(1000000);
  }

  shutdown_oltp(true);

  // collect and print stats here.

  LOG(INFO) << "Process Completed.";
  kill_orphans_and_widows(0);
  return 0;
}
