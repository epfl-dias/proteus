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
DEFINE_uint64(elastic, 0, "elastic_oltp cores");

DEFINE_uint64(ch_scale_factor, 0, "CH-Bench scale factor");
DEFINE_bool(etl, false, "ETL on snapshot");
DEFINE_bool(trade_core, false, "trade case for elasticiy");

DEFINE_bool(bench_ycsb, false, "OLTP Bench: true-ycsb, false-tpcc (default)");
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");

void init_olap_warmup() {
  proteus::init(FLAGS_gpu_buffers, FLAGS_cpu_buffers, FLAGS_log_buffer_usage);
}

std::vector<PreparedStatement> init_olap_sequence(
    int &client_id, const topology::cpunumanode &numa_node,
    const topology::cpunumanode &oltp_node) {
  // chdir("/home/raza/local/htap/opt/pelago");

  time_block t("TcodegenTotal_: ");

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

  uint j = 0;
  uint cc = 0;
  for (auto id : numa_node.local_cores) {
    if (FLAGS_trade_core && FLAGS_elastic > 0 && j < FLAGS_elastic) {
      j++;
      continue;
    }
    coreids.emplace_back(id);

    cc++;
    if (cc == numa_node.local_cores.size() / 2) break;
  }

  for (const auto &id : coreids) {
    LOG(INFO) << "OLAP Cores: " << id;
  }

  if (FLAGS_elastic > 0) {
    uint i = 0;
    for (auto id : oltp_node.local_cores) {
      coreids.emplace_back(id);

      if (++i >= FLAGS_elastic) {
        break;
      }
    }

    if (FLAGS_trade_core) {
      for (auto id : numa_node.local_cores) {
        coreids.emplace_back(id);
      }
    }
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

  DegreeOfParallelism dop{coreids.size()};
  // DegreeOfParallelism dop{1};

  auto aff_parallel = [=]() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
  };

  auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<CpuCoreAffinitizer>();
  };

  typedef decltype(aff_parallel) aff_t;
  typedef decltype(aff_reduce) red_t;

  // for (const auto &q : {q_ch1<aff_t, red_t, plugin_t>, q_ch6<aff_t, red_t,
  // plugin_t>,
  //                       q_ch19<aff_t, red_t, plugin_t>}) {

  using plugin_t = AeolusLocalPlugin;
  for (const auto &q :
       {q_ch1<aff_t, red_t, plugin_t>, q_ch4<aff_t, red_t, plugin_t>,
        q_ch6<aff_t, red_t, plugin_t>, q_ch12<aff_t, red_t, plugin_t>,
        q_ch18<aff_t, red_t, plugin_t>, q_ch19<aff_t, red_t, plugin_t>}) {
    // std::unique_ptr<Affinitizer> aff_parallel =
    //     std::make_unique<CpuCoreAffinitizer>();

    stmts.emplace_back(q(dop, aff_parallel, aff_reduce));
  }
  return stmts;
}
void run_olap_sequence(int &client_id,
                       std::vector<PreparedStatement> &olap_queries,
                       const topology::cpunumanode &numa_node) {
  // Make affinity deterministic
  exec_location{numa_node}.activate();

  olap_queries[0].execute();

  {
    time_block t("T_OLAP: ");

    for (uint i = 0; i < FLAGS_num_olap_repeat; i++) {
      uint j = 0;
      for (auto &q : olap_queries) {
        profiling::resume();
        LOG(INFO) << q.execute();
        profiling::pause();
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
}

void run_oltp(const scheduler::cpunumanode &numa_node) {
  scheduler::WorkerPool::getInstance().start_workers();
}

void shutdown_oltp(bool print_stat = true) {
  scheduler::WorkerPool::getInstance().shutdown(print_stat);
  storage::Schema::getInstance().teardown();
  LOG(INFO) << "OLTP Shutdown complete";
}

void snapshot_oltp() { txn::TransactionManager::getInstance().snapshot(); }

void fly_olap(int i, std::vector<PreparedStatement> &olap_queries,
              const topology::cpunumanode &node) {
  LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Running OLAP Sequence";
  run_olap_sequence(i, olap_queries, node);
}

int main(int argc, char *argv[]) {
  // assert(HTAP_DOUBLE_MASTER && !HTAP_COW && "wrong snapshot mode in oltp");
  if (FLAGS_etl) {
    assert(HTAP_ETL && "ETL MODE NOT ON");
  }

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

  const auto &txn_topo = scheduler::Topology::getInstance();
  const auto &txn_nodes = txn_topo.getCpuNumaNodes();
  // init_oltp(txn_nodes[0].local_cores.size(), "");
  // init_oltp(FLAGS_num_oltp_clients, "");
  // OLAP INIT

  const auto &topo = topology::getInstance();
  const auto &nodes = topo.getCpuNumaNodes();

  auto OLAP_SOCKET = 0;
  auto OLTP_SOCKET = 1;

  bench::Benchmark *bench =
      new bench::TPCC("TPCC", FLAGS_num_oltp_clients, FLAGS_num_oltp_clients,
                      storage::COLUMN_STORE, FLAGS_ch_scale_factor);

  snapshot_oltp();
  int client_id = 1;
  auto olap_queries =
      init_olap_sequence(client_id, nodes[OLAP_SOCKET], nodes[OLTP_SOCKET]);

  scheduler::WorkerPool::getInstance().init(bench, FLAGS_num_oltp_clients, 1,
                                            3);

  storage::Schema::getInstance().report();

  exec_location{nodes[OLAP_SOCKET]}.activate();

  {
    time_block t("T_FIRST_SNAPSHOT_ETL_: ");
    snapshot_oltp();
    storage::Schema::getInstance().ETL(OLAP_SOCKET);
  }

  profiling::resume();
  if (FLAGS_run_oltp && FLAGS_num_oltp_clients > 0)
    run_oltp(txn_nodes[OLTP_SOCKET]);

  usleep(2000000);

  scheduler::WorkerPool::getInstance().print_worker_stats_diff();
  if (FLAGS_elastic > 0) {
    if (!FLAGS_trade_core) {
      std::cout << "Scale-down OLTP" << std::endl;
      scheduler::WorkerPool::getInstance().scale_down(FLAGS_elastic);
    } else {
      for (uint i = 0; i < FLAGS_elastic; i++) {
        scheduler::WorkerPool::getInstance().migrate_worker();
      }
    }
  }

  usleep(2000000);

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
      fly_olap(i, olap_queries, nodes[OLAP_SOCKET]);
    }
    std::cout << "exited: " << i << std::endl;
  }

  scheduler::WorkerPool::getInstance().print_worker_stats_diff();
  scheduler::WorkerPool::getInstance().print_worker_stats();
  std::cout << "[Master] Shutting down everything" << std::endl;

  if (!FLAGS_run_oltp && FLAGS_num_oltp_clients > 0) {
    // FIXME: hack because it needs to run before it can be stopped
    run_oltp(txn_nodes[OLTP_SOCKET]);
  }

  exit(0);

  shutdown_oltp(true);
  shutdown_olap();

  return 0;
}
