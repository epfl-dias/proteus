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

#define RUNTIME_SECONDS 60
#define NUM_OLAP_CLIENTS 1
#define NUM_OLTP_CLIENTS 64
#define NUM_TPCH_QUERIES 22
#define NUM_OLAP_REPEAT 2
#define PLAN_DIR "/scratch/raza/htap/opt/pelago/inputs/plans/cpu-ssb"

#include <filesystem>
#include <unistd.h>

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
#include "plan/plan-parser.hpp"
#include "storage/storage-manager.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

struct OLAP_STATS {
  uint64_t tpch_runtimes[NUM_TPCH_QUERIES * NUM_OLAP_REPEAT];
};

std::vector<pid_t> children;

// https://stackoverflow.com/a/25829178/1237824
std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first)
    return str;
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

// https://stackoverflow.com/a/7756105/1237824
bool starts_with(const std::string &s1, const std::string &s2) {
  return s2.size() <= s1.size() && s1.compare(0, s2.size(), s2) == 0;
}

constexpr size_t clen(const char *str) {
  return (*str == 0) ? 0 : clen(str + 1) + 1;
}

const char *catalogJSON = "inputs";

void thread_warm_up() {}

void init_olap_warmup(const topology::cpunumanode &numa_node) {
  time_block t("Tolap init: ");
  LOG(INFO) << "[OLAP] Initializing ...";

  bool FLAGS_trace_allocations = false;
  bool FLAGS_inc_buffers = false;

  srand(time(NULL));

  FLAGS_logtostderr = 1; // FIXME: the command line flags/defs seem to fail...

  google::InstallFailureSignalHandler();

  set_trace_allocations(FLAGS_trace_allocations);

  size_t cpu_buffers = 1024;
  size_t gpu_buffers = 512;
  if (FLAGS_inc_buffers) {
    cpu_buffers = 16 * 1024;
    gpu_buffers = 1024;
  }

  LOG(INFO) << "[OLAP] Warming up threads...";

  std::vector<std::thread> thrds;
  for (int i = 0; i < 1024; ++i)
    thrds.emplace_back(thread_warm_up);
  for (auto &t : thrds)
    t.join();

  // srand(time(0));

  LOG(INFO) << "[OLAP] Initializing codegen...";

  PipelineGen::init();

  LOG(INFO) << "[OLAP] Initializing memory manager...";
  MemoryManager::init(gpu_buffers, cpu_buffers);

  // Make affinity deterministic
  // exec_location{numa_node}.activate();
  auto &topo = topology::getInstance();
  if (topo.getGpuCount() > 0) {
    exec_location{topo.getGpus()[0]}.activate();
  } else {
    exec_location{topo.getCpuNumaNodes()[0]}.activate();
  }

  LOG(INFO) << "Finished initialization";
}

std::vector<PreparedStatement>
init_olap_sequence(const topology::cpunumanode &numa_node,
                   // PreparedStatement &stmts[],
                   std::string query_plans_dir) {
  // chdir("/home/raza/local/htap/opt/pelago");

  time_block t("TcodegenTotal_: ");
  // TODO: codegen all the TPC-H queries beforehand. keep them ready anyway.
  // BUG: if we generate code before fork, will the codegen will be valid after
  // a fork? as in memory pointers which are pushed in codegen, are they valid
  // on runtime?

  std::vector<PreparedStatement> stmts;
  Catalog *catalog = &Catalog::getInstance();
  CachingService *caches = &CachingService::getInstance();
  catalog->clear();
  caches->clear();

  std::string label_prefix("htap_server_" + std::to_string(getpid()) + "_q");
  uint i = 0;
  for (const auto &entry :
       std::filesystem::directory_iterator(query_plans_dir)) {
    if (entry.path().extension() == ".json") {

      std::string plan_path = entry.path().string();
      std::string label = label_prefix + std::to_string(i++);

      time_block t("Tcodegen_" + label + ": ");

      std::cout << "Compiling Query:" << plan_path << std::endl;

      ParallelContext *ctx = new ParallelContext(label, false);
      CatalogParser catalog = CatalogParser(catalogJSON, ctx);
      PlanExecutor exec =
          PlanExecutor(plan_path.c_str(), catalog, label.c_str(), ctx);

      ctx->compileAndLoad();

      stmts.emplace_back(ctx->getPipelines());
    }
  }

  return stmts;
}
void run_olap_sequence(std::vector<PreparedStatement> &olap_queries,
                       int &client_id, struct OLAP_STATS *analytical_stats,
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

  for (ushort i = 0, j = 0; i < NUM_OLAP_REPEAT; i++) {

    for (auto &q : olap_queries) {

      std::chrono::time_point<std::chrono::system_clock> start =
          std::chrono::system_clock::now();

      q.execute();

      analytical_stats->tpch_runtimes[i * j] =
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

  bench->load_data(topology::getInstance().getCoreCount());

  return bench;
}

void run_oltp(const topology::cpunumanode &numa_node, uint num_workers,
              bench::Benchmark *bench) {

  // scheduler::AffinityManager::getInstance().set(
  //    &scheduler::Topology::getInstance().get_worker_cores()->front());

  exec_location{numa_node}.activate();
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

void kill_orphans_and_widows(int s) {
  for (auto &pid : children) {
    kill(pid, SIGTERM);
  }
  exit(1);
}

void register_handler() { signal(SIGINT, kill_orphans_and_widows); }

int main(int argc, char *argv[]) {

  google::InitGoogleLogging(argv[0]);

  auto &topo = topology::getInstance();
  auto &nodes = topo.getCpuNumaNodes();

  assert(nodes.size() >= 2);
  assert(NUM_OLTP_CLIENTS <= nodes[0].local_cores.size());

  // INIT
  struct OLAP_STATS *analytical_stats = (struct OLAP_STATS *)get_shm(
      "olap_stats", sizeof(struct OLAP_STATS) * NUM_OLAP_CLIENTS);

  // bench::Benchmark *oltp_bench = init_oltp(nodes[0].local_cores.size(), "");

  // RUNOLTP
  // run_oltp(nodes[0], nodes[0].local_cores.size(), oltp_bench);

  for (int i = 0; i < NUM_OLAP_CLIENTS; i++) {

    pid_t tmp = fork();

    if (tmp == 0) {
      // run olap stuff
      // warmup should be inside or outside? cpu can be outside but dont know
      // about gpu init stuff can be forked or not.
      init_olap_warmup(nodes[1]);

      std::vector<PreparedStatement> olap_queries =
          init_olap_sequence(nodes[1], PLAN_DIR);

      run_olap_sequence(olap_queries, i, analytical_stats + i, nodes[1]);

      shutdown_olap();
      break;
    } else {
      children.emplace_back(tmp);
    }
  }

  // some child process
  if (children.size() != NUM_OLAP_CLIENTS)
    return 0;

  register_handler();
  usleep(RUNTIME_SECONDS * 1000000);

  // shutdown_oltp(true);
  // shutdown_olap();

  // collect and print stats here.

  LOG(INFO) << "Shutting down...";

  return 0;
}
