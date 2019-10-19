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

#include <filesystem>
#include <iostream>
#include <string>

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
#include "codegen/util/timing.hpp"
#include "glo.hpp"
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

DEFINE_uint64(num_olap_clients, 1, "Number of OLAP clients");
DEFINE_uint64(num_olap_repeat, 1, "Number of OLAP clients");
DEFINE_uint64(num_oltp_clients, 1, "Number of OLTP clients");
DEFINE_string(plan_json, "", "Plan to execute, takes priority over plan_dir");
DEFINE_string(plan_dir, "inputs/plans/cpu-ssb",
              "Directory with plans to be executed");
DEFINE_string(inputs_dir, "inputs/", "Data and catalog directory");
DEFINE_bool(run_oltp, true, "Run OLTP");
DEFINE_bool(run_olap, true, "Run OLAP");
DEFINE_uint64(ch_scale_factor, 0, "CH-Bench scale factor");

DEFINE_bool(bench_ycsb, false, "OLTP Bench: true-ycsb, false-tpcc (default)");
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");

std::vector<pid_t> children;

struct OLAP_STATS {
  std::atomic<bool> shutdown;
  bool begin;
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
      if (entry.path().filename().string()[0] == '.') continue;

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

  olap_queries[0].execute();

  {
    time_block t("T_OLAP: ");

    for (uint i = 0; i < FLAGS_num_olap_repeat; i++) {
      uint j = 0;
      for (auto &q : olap_queries) {
        // std::chrono::time_point<std::chrono::system_clock> start =
        //     std::chrono::system_clock::now();

        q.execute();

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
  LOG(INFO) << "Unloading files...";
  StorageManager::unloadAll();

  LOG(INFO) << "Shuting down memory manager...";
  MemoryManager::destroy();

  LOG(INFO) << "Shut down finished";
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
                              FLAGS_ch_scale_factor, storage::COLUMN_STORE);

    } else {
      bench = new bench::TPCC("TPCC", num_workers, num_workers,
                              FLAGS_ch_scale_factor, storage::COLUMN_STORE, 0,
                              csv_path);
    }
  }
  scheduler::WorkerPool::getInstance().init(bench, num_workers, 1);
}

void run_oltp(const scheduler::cpunumanode &numa_node) {
  scheduler::WorkerPool::getInstance().start_workers();
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

[[noreturn]] void kill_orphans_and_widows(int s) {
  for (auto &pid : children) {
    std::cout << "killing: " << pid << std::endl;
    kill(pid, SIGKILL);
  }
  exit(0);
}

void register_handler() { signal(SIGINT, kill_orphans_and_widows); }

void snapshot_oltp() { txn::TransactionManager::getInstance().snapshot(); }

[[noreturn]] void fly_olap(struct OLAP_STATS *olap_stats, int i) {
  init_olap_warmup();

  auto &topo = topology::getInstance();
  auto &nodes = topo.getCpuNumaNodes();
  uint OLAP_SOCKET = nodes.size() / 2;
  exec_location{nodes[OLAP_SOCKET]}.activate();

  // assert(nodes.size() >= 2);
  // assert(FLAGS_num_oltp_clients <= nodes[0].local_cores.size());

  LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Compiling OLAP Sequence";
  std::vector<PreparedStatement> olap_queries;
  {
    time_block t("T_init_olap_sequence_: ");
    olap_queries = init_olap_sequence(i, nodes[OLAP_SOCKET]);
  }
  // while (olap_stats[0].begin == false)
  //   ;

  LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Running OLAP Sequence";
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();
  {
    time_block t("T_run_olap_sequence_: ");
    run_olap_sequence(i, olap_queries, olap_stats + i, nodes[OLAP_SOCKET]);
  }

  // double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(
  //                        std::chrono::system_clock::now() - start)
  //                        .count()) /
  //                   1000.0; // in seconds

  // double thrughput = (NUM_TPCH_QUERIES * NUM_OLAP_REPEAT) / duration;
  // std::cout << "Duration[" << getpid() << "] : " << duration << std::endl;
  // std::cout << "Queries[" << getpid()
  //           << "] : " << (NUM_TPCH_QUERIES * NUM_OLAP_REPEAT) << std::endl;
  // std::cout << "Throughput[" << getpid() << "] : " << thrughput << std::endl;

  LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Shutdown";
  olap_stats[i].shutdown = true;
  exit(0);
  shutdown_olap();
  exit(0);
}

int main(int argc, char *argv[]) {
  assert(HTAP_COW && "OLTP not set to COW mode");
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 0;  // FIXME: the command line flags/defs seem to fail...

  // google::InstallFailureSignalHandler();

  set_trace_allocations(FLAGS_trace_allocations);

  register_handler();

  FLAGS_num_olap_repeat =
      (NUM_OLAP_REPEAT / FLAGS_num_olap_clients);  // warmmup

  std::cout << "QUERIES_PER_SESSION: " << (FLAGS_num_olap_repeat) << std::endl;
  std::cout << "OLAP Clients: " << FLAGS_num_olap_clients << std::endl;

  g_num_partitions = 2;

  std::cout << "Num Partitions: " << g_num_partitions << std::endl;

  struct OLAP_STATS *olap_stats = (struct OLAP_STATS *)new (get_shm(
      "olap_stats_" + std::to_string(getpid()),
      sizeof(struct OLAP_STATS) * FLAGS_num_olap_clients)) struct OLAP_STATS();

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    olap_stats[i].shutdown = false;
    // for (int j = 0; j < NUM_TPCH_QUERIES; j++) {
    //   for (int k = 0; k < NUM_OLAP_REPEAT; k++)
    //     olap_stats[i].runtime_stats[j][k] = 0;
    // }
  }

  auto &txn_topo = scheduler::Topology::getInstance();
  auto &txn_nodes = txn_topo.getCpuNumaNodes();
  // init_oltp(txn_nodes[0].local_cores.size(), "");
  init_oltp(FLAGS_num_oltp_clients, "");
  std::cout << "Init oltp done" << std::endl;

  // bench::Benchmark *oltp_bench = init_oltp(4, "");
  snapshot_oltp();

  if (FLAGS_run_oltp && FLAGS_num_oltp_clients > 0) {
    // timed_func::interval_runner(
    //     [] { scheduler::WorkerPool::getInstance().print_worker_stats(); },
    //     (5 * 1000));

    run_oltp(txn_nodes[0]);
  }
  usleep(2000000);

  __itt_resume();
  std::cout << "Starting now.." << std::endl;

  std::thread t([olap_stats, txn_nodes]() {
    for (int i = 0; i < FLAGS_num_olap_clients; i++) {
      scheduler::WorkerPool::getInstance().print_worker_stats_diff();
      std::cout << "Snapshot Request" << std::endl;
      snapshot_oltp();
      std::cout << "Snapshot Done" << std::endl;
      pid_t tmp = fork();
      if (tmp == 0) {
        fly_olap(olap_stats, i);
        exit(0);
      }
      scheduler::WorkerPool::getInstance().print_worker_stats_diff();

      while (olap_stats[i].shutdown != true)
        ;

      scheduler::WorkerPool::getInstance().print_worker_stats_diff();
      std::cout << "exited: " << i << std::endl;
    }
  });

  t.join();

  // for (int i = 0; i < FLAGS_num_olap_clients; i++) {
  //   olap_stats[i].shutdown = false;
  //   olap_stats[i].begin = false;

  //   pid_t tmp = fork();

  //   if (tmp == 0) {

  //     // run olap stuff
  //     // warmup should be inside or outside? cpu can be outside but dont know
  //     // about gpu init stuff can be forked or not.
  //     std::cout << "Snapshot Request" << std::endl;
  //     snapshot_oltp();
  //     std::cout << "Snapshot Done" << std::endl;

  //     if (FLAGS_run_olap) {
  //       init_olap_warmup();

  //       auto &topo = topology::getInstance();
  //       auto &nodes = topo.getCpuNumaNodes();
  //       exec_location{nodes[3]}.activate();

  //       // assert(nodes.size() >= 2);
  //       // assert(FLAGS_num_oltp_clients <= nodes[0].local_cores.size());

  //       LOG(INFO) << "[SERVER-COW] OLAP Client #" << i
  //                 << ": Compiling OLAP Sequence";
  //       std::vector<PreparedStatement> olap_queries;
  //       {
  //         time_block t("T_init_olap_sequence_: ");
  //         olap_queries = init_olap_sequence(i, nodes[3]);
  //       }
  //       while (olap_stats[0].begin == false)
  //         ;

  //       LOG(INFO) << "[SERVER-COW] OLAP Client #" << i
  //                 << ": Running OLAP Sequence";
  //       std::chrono::time_point<std::chrono::system_clock> start =
  //           std::chrono::system_clock::now();
  //       {
  //         time_block t("T_run_olap_sequence_: ");
  //         run_olap_sequence(i, olap_queries, olap_stats + i, nodes[3]);
  //       }

  //       double duration =
  //           (std::chrono::duration_cast<std::chrono::milliseconds>(
  //                std::chrono::system_clock::now() - start)
  //                .count()) /
  //           1000.0; // in seconds

  //       double thrughput = (NUM_TPCH_QUERIES * NUM_OLAP_REPEAT) / duration;
  //       std::cout << "Duration[" << getpid() << "] : " << duration <<
  //       std::endl; std::cout << "Queries[" << getpid()
  //                 << "] : " << (NUM_TPCH_QUERIES * NUM_OLAP_REPEAT)
  //                 << std::endl;
  //       std::cout << "Throughput[" << getpid() << "] : " << thrughput
  //                 << std::endl;

  //       LOG(INFO) << "[SERVER-COW] OLAP Client #" << i << ": Shutdown";
  //       olap_stats[i].shutdown = true;
  //       shutdown_olap();
  //       break;
  //     } else {

  //       while (true) {
  //         usleep(1000000);
  //       }

  //       // storage::ColumnStore *tbl = nullptr;

  //       // auto &txn_storage = storage::Schema::getInstance();

  //       // std::string aeolus_rel_name = "tpcc_orderline";
  //       // std::string col_name = "ol_o_id";

  //       // for (auto &tb : txn_storage.getTables()) {
  //       //   if (aeolus_rel_name.compare(tb->name) == 0) {
  //       //     assert(tb->storage_layout == storage::COLUMN_STORE);
  //       //     tbl = (storage::ColumnStore *)tb;
  //       //     break;
  //       //   }
  //       // }
  //       // assert(tbl != nullptr);

  //       // uint64_t *data = nullptr;
  //       // uint64_t num_recs = 0;

  //       // std::vector<std::pair<void *, uint64_t>> mfiles;

  //       // for (auto &c : tbl->getColumns()) {
  //       //   if (c->name.compare(col_name) == 0) {
  //       //     auto d = c->snapshot_get_data();

  //       //     for (size_t i = 0; i < d.size(); ++i) {
  //       //       std::cout << "AEO name: " << c->name << std::endl;
  //       //       std::cout << "AEO #-records: "
  //       //                 << (d[i].first.size / sizeof(uint64_t)) <<
  //       std::endl;

  //       //       mfiles.emplace_back(std::make_pair(
  //       //           d[i].first.data, (d[i].first.size / sizeof(uint64_t))));
  //       //     }
  //       //   }
  //       // }

  //       // uint64_t sum = 0;
  //       // while (true) {

  //       //   for (const auto &clr : mfiles) {
  //       //     uint64_t *dt_ptr = (uint64_t *)clr.first;
  //       //     uint64_t num_recs = clr.second;

  //       //     for (uint64_t i = 0; i < num_recs; i++) {
  //       //       sum += dt_ptr[i];
  //       //     }
  //       //   }
  //       // }
  //     }
  //   } else {
  //     children.emplace_back(tmp);
  //   }
  // }

  // // some child process
  // if (children.size() != FLAGS_num_olap_clients)
  //   exit(1);

  // if (FLAGS_run_oltp && FLAGS_num_oltp_clients > 0)
  //   run_oltp(txn_nodes[0]);

  // std::cout << "Total Children: " << children.size() << std::endl;
  // std::cout << "Total OLAP Clients: " << FLAGS_num_olap_clients << std::endl;

  // if (FLAGS_run_olap) {
  //   std::cout << "HERE" << std::endl;
  //   olap_stats[0].begin = true;

  //   usleep(7000000);
  //   // Termination Condition.
  //   // uint client_ctr = 0;
  //   // while (true) {
  //   //   if (client_ctr == FLAGS_num_olap_clients) {
  //   //     std::cout << "All Done: " << client_ctr << std::endl;
  //   //     break;
  //   //   } else if (olap_stats[client_ctr].shutdown) {
  //   //     std::cout << "Children merger: " << client_ctr << std::endl;
  //   //     client_ctr++;
  //   //   } else {
  //   //     // std::this_thread::yield();
  //   //     std::cout << "Children wait: " << client_ctr << std::endl;
  //   //     usleep(1000000);
  //   //   }
  //   // }
  // } else if (FLAGS_run_oltp && FLAGS_num_oltp_clients > 0) {
  //   std::cout << "OLTP STANDALONE RUNTIME" << std::endl;
  //   usleep(30000000);
  // }

  std::cout << "[Master] Shutting down everything" << std::endl;
  shutdown_oltp(true);

  std::cout << "[Master] Killing orphans and widows" << std::endl;
  kill_orphans_and_widows(0);

  // collect and print stats here.

  // LOG(INFO) << "Process Completed.";

  return 0;
}
