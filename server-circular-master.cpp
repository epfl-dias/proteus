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

#define NUM_TPCH_QUERIES 1
#define NUM_OLAP_REPEAT 16
#define HTAP true

#include <gflags/gflags.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <thread>

#include "benchmarks/tpcc_64.hpp"
#include "benchmarks/ycsb.hpp"
#include "cli-flags.hpp"
#include "codegen/communication/comm-manager.hpp"
#include "codegen/memory/block-manager.hpp"
#include "codegen/memory/memory-manager.hpp"
#include "codegen/plan/prepared-statement.hpp"
#include "codegen/storage/storage-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/util/jit/pipeline.hpp"
#include "codegen/util/parallel-context.hpp"
#include "codegen/util/profiling.hpp"
#include "codegen/util/timing.hpp"
#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "queries/queries.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

DEFINE_uint64(num_olap_clients, 1, "Number of OLAP clients");
DEFINE_uint64(num_olap_repeat, 1, "Number of OLAP clients");
DEFINE_uint64(num_oltp_clients, 0, "Number of OLTP clients");
DEFINE_string(plan_json, "", "Plan to execute, takes priority over plan_dir");
DEFINE_string(plan_dir, "inputs/plans/cpu-ssb",
              "Directory with plans to be executed");
DEFINE_string(inputs_dir, "inputs/", "Data and catalog directory");
DEFINE_bool(run_oltp, true, "Run OLTP");
DEFINE_bool(run_olap, true, "Run OLAP");
DEFINE_bool(elastic, false, "elastic_oltp");

DEFINE_bool(etl, false, "ETL on snapshot");

DEFINE_bool(bench_ycsb, false, "OLTP Bench: true-ycsb, false-tpcc (default)");
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");

std::vector<std::thread> children;

struct OLAP_STATS {
  bool shutdown;
  // uint64_t runtime_stats[NUM_TPCH_QUERIES][NUM_OLAP_REPEAT];
};

void init_olap_warmup() {
  proteus::init(FLAGS_gpu_buffers, FLAGS_cpu_buffers, FLAGS_log_buffer_usage);
}

std::vector<PreparedStatement> init_olap_sequence(
    int &client_id, const topology::cpunumanode &numa_node) {
  // chdir("/home/raza/local/htap/opt/pelago");

  time_block t("TcodegenTotal_: ");
  // TODO: codegen all the TPC-H queries beforehand. keep them ready anyway.
  // BUG: if we generate code before fork, will the codegen will be valid after
  // a fork? as in memory pointers which are pushed in codegen, are they valid
  // on runtime?

  std::vector<PreparedStatement> stmts;

  // std::string label_prefix("htap_" + std::to_string(getpid()) + "_c" +
  //                          std::to_string(client_id) + "_q");

  // if (FLAGS_plan_json.length()) {
  //   LOG(INFO) << "Compiling Plan:" << FLAGS_plan_json << std::endl;

  //   stmts.emplace_back(PreparedStatement::from(
  //       FLAGS_plan_json, label_prefix + std::to_string(0),
  //       FLAGS_inputs_dir));
  // } else {
  //   uint i = 0;
  //   for (const auto &entry :
  //        std::filesystem::directory_iterator(FLAGS_plan_dir)) {
  //     if (entry.path().filename().string()[0] == '.') continue;

  //     if (entry.path().extension() == ".json") {
  //       std::string plan_path = entry.path().string();
  //       std::string label = label_prefix + std::to_string(i++);

  //       LOG(INFO) << "Compiling Query:" << plan_path << std::endl;

  //       stmts.emplace_back(
  //           PreparedStatement::from(plan_path, label, FLAGS_inputs_dir));
  //     }
  //   }
  // }

  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  for (auto id : numa_node.local_cores) {
    coreids.emplace_back(id);
  }

  // {
  //   for (const auto &n : topology::getInstance().getCpuNumaNodes()) {
  //     if (n != numa_node) {
  //       for (size_t i = 0; i < std::min(4, n.local_cores.size()); ++i) {
  //         coreids.emplace_back(n.local_cores[i]);
  //       }
  //     }
  //   }
  // }

  // return stmts;
  DegreeOfParallelism dop{coreids.size()};
  for (const auto &q : {q_ch1, q_ch6}) {
    // std::unique_ptr<Affinitizer> aff_parallel =
    //     std::make_unique<CpuCoreAffinitizer>();

    std::unique_ptr<Affinitizer> aff_parallel =
        std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
    std::unique_ptr<Affinitizer> aff_reduce =
        std::make_unique<CpuCoreAffinitizer>();

    stmts.emplace_back(q(dop, std::move(aff_parallel), std::move(aff_reduce)));
  }
  return stmts;  // q_sum_c1t(), q_ch_c1t(), q_ch2_c1t()};
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

  olap_queries[0].execute();

  {
    time_block t("T_OLAP: ");

    for (uint i = 0; i < FLAGS_num_olap_repeat; i++) {
      uint j = 0;
      for (auto &q : olap_queries) {
        // std::chrono::time_point<std::chrono::system_clock> start =
        //     std::chrono::system_clock::now();

        LOG(INFO) << q.execute();

        // olap_stats->runtime_stats[j][i] =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(
        //         std::chrono::system_clock::now() - start)
        //         .count();
        j++;
      }
    }
  }
}

void shutdown_olap() {
  StorageManager::unloadAll();
  MemoryManager::destroy();

  LOG(INFO) << "OLAP Shutdown complete";
}

void init_oltp(uint num_workers, std::string csv_path) {
  // TODO: # warehouse for csv should will be variable.

  bench::Benchmark *bench = nullptr;
  if (FLAGS_bench_ycsb) {
    bench = new bench::YCSB("YCSB", 1, 72 * 1000000, 0.5, -1, 10,
                            FLAGS_ycsb_write_ratio, num_workers,
                            scheduler::Topology::getInstance().getCoreCount(),
                            4, true);
  } else {
    if (csv_path.length() < 2) {
      bench = new bench::TPCC("TPCC", num_workers, num_workers,
                              storage::COLUMN_STORE);

    } else {
      bench = new bench::TPCC("TPCC", num_workers, num_workers,
                              storage::COLUMN_STORE, 0, csv_path);
    }
  }
  scheduler::WorkerPool::getInstance().init(bench, num_workers, 1, 3);
}

void run_oltp(const scheduler::cpunumanode &numa_node) {
  scheduler::WorkerPool::getInstance().start_workers();
}

void shutdown_oltp(bool print_stat = true) {
  scheduler::WorkerPool::getInstance().shutdown(print_stat);
  storage::Schema::getInstance().teardown();
  LOG(INFO) << "OLTP Shutdown complete";
}

void *get_shm(std::string name, size_t size) {
  int fd = shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  if (fd == -1) {
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
               << std::endl;
    assert(false);
  }

  if (ftruncate(fd, size) < 0) {  //== -1){
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

// [[noreturn]] void kill_orphans_and_widows(int s) {
//   for (auto &pid : children) {
//     kill(pid, SIGTERM);
//   }
//   exit(1);
// }

// void handle_child_termination(int sig) {
//   pid_t p;
//   int status;

//   while ((p = waitpid(-1, &status, WNOHANG)) > 0) {
//     LOG(INFO) << "Process " << p << " stopped or died.";
//     exit(-1);
//   }
// }

// void register_handler() {
//   signal(SIGINT, kill_orphans_and_widows);

//   {

//     struct sigaction sa {};
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
//     sa.sa_handler = handle_child_termination;
// #pragma clang diagnostic pop
//     sigaction(SIGCHLD, &sa, NULL);
//   }
// }

// void register_handler() { signal(SIGINT, kill_orphans_and_widows); }

void snapshot_oltp() { txn::TransactionManager::getInstance().snapshot(); }

void fly_olap(struct OLAP_STATS *olap_stats, int i,
              std::vector<PreparedStatement> &olap_queries,
              const topology::cpunumanode &node) {
  LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Running OLAP Sequence";
  run_olap_sequence(i, olap_queries, olap_stats + i, node);
  // olap_stats[i].shutdown = true;
}

int main(int argc, char *argv[]) {
  // assert(HTAP_DOUBLE_MASTER && !HTAP_COW && "wrong snapshot mode in oltp");
  if (FLAGS_etl) {
    assert(HTAP_ETL && "ETL MODE NOT ON");
  }
  // register_handler();
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;  // FIXME: the command line flags/defs seem to fail...
  g_num_partitions = 1;
  google::InstallFailureSignalHandler();

  FLAGS_num_olap_repeat =
      (NUM_OLAP_REPEAT / FLAGS_num_olap_clients);  // warmmup

  init_olap_warmup();

  if (FLAGS_num_oltp_clients == 0) {
    FLAGS_num_oltp_clients =
        topology::getInstance().getCpuNumaNodes()[0].local_cores.size();
  }

  std::cout << "QUERIES_PER_SESSION: " << (FLAGS_num_olap_repeat) << std::endl;
  std::cout << "OLAP Clients: " << FLAGS_num_olap_clients << std::endl;

  // google::InstallFailureSignalHandler();

  set_trace_allocations(FLAGS_trace_allocations);

  struct OLAP_STATS *olap_stats = (struct OLAP_STATS *)get_shm(
      "olap_stats_" + std::to_string(getpid()),
      sizeof(struct OLAP_STATS) * FLAGS_num_olap_clients);

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    olap_stats[i].shutdown = true;
    // for (int j = 0; j < NUM_TPCH_QUERIES; j++) {
    //   for (int k = 0; k < NUM_OLAP_REPEAT; k++)
    //     olap_stats[i].runtime_stats[j][k] = 0;
    // }
  }

  const auto &txn_topo = scheduler::Topology::getInstance();
  const auto &txn_nodes = txn_topo.getCpuNumaNodes();
  // init_oltp(txn_nodes[0].local_cores.size(), "");
  init_oltp(FLAGS_num_oltp_clients, "");
  storage::Schema::getInstance().report();

  // bench::Benchmark *oltp_bench = init_oltp(4, "");

  // OLAP INIT

  const auto &topo = topology::getInstance();
  const auto &nodes = topo.getCpuNumaNodes();

  // assert(nodes.size() >= 2);
  // assert(FLAGS_num_oltp_clients <= nodes[0].local_cores.size());
  auto OLAP_SOCKET = 0;
  auto OLTP_SOCKET = 1;

  exec_location{nodes[OLAP_SOCKET]}.activate();

  // std::vector<std::vector<PreparedStatement>> olap_queries;

  // for (int i = 0; i < FLAGS_num_olap_clients; i++) {
  //   // LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Compiling OLAP
  //   // Sequence";
  //   olap_queries.push_back(init_olap_sequence(i, OLAP_SOCKET));
  // }

  {
    time_block t("T_FIRST_SNAPSHOT_ETL_: ");
    snapshot_oltp();
    storage::Schema::getInstance().ETL(OLAP_SOCKET);
  }
  int client_id = 1;
  auto olap_queries = init_olap_sequence(client_id, nodes[OLAP_SOCKET]);

  profiling::resume();
  if (FLAGS_run_oltp && FLAGS_num_oltp_clients > 0)
    run_oltp(txn_nodes[OLTP_SOCKET]);

  usleep(2000000);

  scheduler::WorkerPool::getInstance().print_worker_stats_diff();
  if (FLAGS_elastic) {
    std::cout << "Scale-down OLTP" << std::endl;
    scheduler::WorkerPool::getInstance().scale_down(4);  // 4 core scale down
  }

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    scheduler::WorkerPool::getInstance().print_worker_stats_diff();

    std::cout << "Snapshot Request" << std::endl;
    snapshot_oltp();
    std::cout << "Snapshot Done" << std::endl;
    if (FLAGS_etl) {
      time_block t("T_ETL_: ");
      storage::Schema::getInstance().ETL(OLAP_SOCKET);
    }
    scheduler::WorkerPool::getInstance().print_worker_stats_diff();

    {
      time_block t("T_fly_olap_: ");
      fly_olap(olap_stats, i, olap_queries, nodes[OLAP_SOCKET]);
    }
    // usleep(500000);
    std::cout << "exited: " << i << std::endl;
  }

  //  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
  //   olap_stats[i].shutdown = false;

  //   children.emplace_back([olap_stats, i]() { activate_olap(olap_stats, i);
  //   });
  // }

  // if (FLAGS_run_olap) {
  //   // Termination Condition.
  //   for (auto &th : children) {
  //     th.join();
  //   }
  // } else {
  //   usleep(30000000);
  // }

  // shutdown_oltp(true);
  scheduler::WorkerPool::getInstance().print_worker_stats_diff();
  scheduler::WorkerPool::getInstance().print_worker_stats();
  std::cout << "[Master] Shutting down everything" << std::endl;

  if (!FLAGS_run_oltp && FLAGS_num_oltp_clients > 0) {
    // FIXME: hack because it needs to run before it can be stopped
    run_oltp(txn_nodes[OLTP_SOCKET]);
  }

  exit(0);

  shutdown_oltp(true);

  // LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Shutdown";
  shutdown_olap();

  // std::cout << "[Master] Killing orphans and widows" << std::endl;
  // kill_orphans_and_widows(0);

  return 0;
}
