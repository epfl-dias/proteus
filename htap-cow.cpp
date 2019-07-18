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


#include "benchmarks/bench.hpp"
#include "benchmarks/tpcc.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

#include "codegen/communication/comm-manager.hpp"
#include "codegen/memory/block-manager.hpp"
#include "codegen/memory/memory-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/util/jit/pipeline.hpp"
#include "codegen/util/parallel-context.hpp"
#include "plan/plan-parser.hpp"
#include "storage/storage-manager.hpp"
#include "topology/topology.hpp"
#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif


struct OLAP_STATS{
  uint64_t tpch_runtimes[NUM_TPCH_QUERIES];
};

std::vector<pid_t> children;

// https://stackoverflow.com/a/25829178/1237824
std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) return str;
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

void executePlan(const char *label, const char *planPath,
                 const char *catalogJSON) {
  auto &topo = topology::getInstance();
  {
    Catalog *catalog = &Catalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  // gpu_run(cudaSetDevice(0));

  std::vector<Pipeline *> pipelines;
  {
    time_block t("Tcodegen: ");

    ParallelContext *ctx = new ParallelContext(label, false);
    CatalogParser catalog = CatalogParser(catalogJSON, ctx);
    PlanExecutor exec = PlanExecutor(planPath, catalog, label, ctx);

    ctx->compileAndLoad();

    pipelines = ctx->getPipelines();
  }

  // just to be sure...
  for (const auto &gpu : topo.getGpus()) {
    set_exec_location_on_scope d{gpu};
    gpu_run(cudaDeviceSynchronize());
  }

  for (const auto &gpu : topo.getGpus()) {
    set_exec_location_on_scope d{gpu};
    gpu_run(cudaProfilerStart());
  }
  __itt_resume();

  // Make affinity deterministic
  if (topo.getGpuCount() > 0) {
    exec_location{topo.getGpus()[0]}.activate();
  } else {
    exec_location{topo.getCpuNumaNodes()[0]}.activate();
  }

  {
    time_block t("Texecute w sync: ");

    {
      time_block t("Texecute       : ");

      for (Pipeline *p : pipelines) {
        nvtxRangePushA("pip");
        {
          time_block t("T: ");

          p->open();
          p->consume(0);
          p->close();
        }
        nvtxRangePop();
      }

      std::cout << dec;
    }

    // just to be sure...
    for (const auto &gpu : topo.getGpus()) {
      set_exec_location_on_scope d{gpu};
      gpu_run(cudaDeviceSynchronize());
    }
  }

  __itt_pause();
  for (const auto &gpu : topo.getGpus()) {
    set_exec_location_on_scope d{gpu};
    gpu_run(cudaProfilerStop());
  }

  // Make affinity deterministic
  if (topo.getGpuCount() > 0) {
    exec_location{topo.getGpus()[0]}.activate();
  } else {
    exec_location{topo.getCpuNumaNodes()[0]}.activate();
  }
}

void executePlan(const char *label, const char *planPath) {
  executePlan(label, planPath, catalogJSON);
}

class unlink_upon_exit {
  size_t query;
  std::string label_prefix;

  std::string last_label;

 public:
  unlink_upon_exit()
      : query(0),
        label_prefix("raw_server_" + std::to_string(getpid()) + "_q"),
        last_label("") {}

  unlink_upon_exit(size_t unique_id)
      : query(0),
        label_prefix("raw_server_" + std::to_string(unique_id) + "_q"),
        last_label("") {}

  ~unlink_upon_exit() {
    if (last_label != "") shm_unlink(last_label.c_str());
  }

  std::string get_label() const { return last_label; }

  std::string inc_label() {
    if (query > 0) shm_unlink(last_label.c_str());
    last_label = label_prefix + std::to_string(query++);
    return last_label;
  }
};

std::string runPlanFile(std::string plan, unlink_upon_exit &uue,
                        bool echo = true) {
  std::string label = uue.inc_label();
  executePlan(label.c_str(), plan.c_str());

  if (echo) {
    std::cout << "result echo" << std::endl;
    /* current */
    int fd2 = shm_open(label.c_str(), O_RDONLY, S_IRWXU);
    if (fd2 == -1) {
      throw runtime_error(string(__func__) + string(".open (output): ") +
                          label);
    }
    struct stat statbuf;
    if (fstat(fd2, &statbuf)) {
      fprintf(stderr, "FAILURE to stat test results! (%s)\n",
              std::strerror(errno));
      assert(false);
    }
    size_t fsize2 = statbuf.st_size;
    char *currResultBuf =
        (char *)mmap(NULL, fsize2, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd2, 0);
    fwrite(currResultBuf, sizeof(char), fsize2, stdout);
    std::cout << std::endl;

    munmap(currResultBuf, fsize2);
  }

  return label;
}

void thread_warm_up() {}

void olap_init_warmup(const topology::cpunumanode &numa_node) {
  time_block t("Tolap init: ");
  LOG(INFO) << "[OLAP] Initializing ...";

  bool FLAGS_trace_allocations = false;
  bool FLAGS_inc_buffers = false;

  srand(time(NULL));

  FLAGS_logtostderr = 1;  // FIXME: the command line flags/defs seem to fail...

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
  for (int i = 0; i < 1024; ++i) thrds.emplace_back(thread_warm_up);
  for (auto &t : thrds) t.join();

  // srand(time(0));

  LOG(INFO) << "[OLAP] Initializing codegen...";

  PipelineGen::init();

  LOG(INFO) << "[OLAP] Initializing memory manager...";
  MemoryManager::init(gpu_buffers, cpu_buffers);

  // Make affinity deterministic
  exec_location{numa_node}.activate();

  LOG(INFO) << "Finished initialization";
}

void init_olap_sequence(const topology::cpunumanode &numa_node) {
  // TODO: codegen all the TPC-H queries beforehand. keep them ready anyway.
  // BUG: if we generate code before fork, will the codegen will be valid after
  // a fork? as in memory pointers which are pushed in codegen, are they valid
  // on runtime?
}
void run_olap_sequence(int &client_id, struct OLAP_STATS *analytical_stats, const topology::cpunumanode &numa_node) {
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

void run_oltp(const topology::cpunumanode &numa_node, uint num_workers, bench::Benchmark *bench) {
  
  //scheduler::AffinityManager::getInstance().set(
  //    &scheduler::Topology::getInstance().get_worker_cores()->front());

  exec_location{numa_node}.activate();
  auto &wp = scheduler::WorkerPool::getInstance();
  wp.init(bench);
  wp.start_workers(num_workers);
}

void shutdown_oltp(bool print_stat = true) {
  scheduler::WorkerPool::getInstance().shutdown(print_stat);
}


void *get_shm(std::string name, size_t size){
  int fd =
      shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  

  if (fd == -1) {
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno) << std::endl;
    assert(false);
  }

  if (ftruncate(fd, size) < 0) {  //== -1){
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno) << std::endl;
    assert(false);
  }

  void *mem = mmap(NULL, size, PROT_WRITE | PROT_READ,
                          MAP_SHARED, fd, 0);
  if (!mem) {
    LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno) << std::endl;
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

void register_handler() {
  signal(SIGINT, kill_orphans_and_widows);
}



int main(int argc, char *argv[]) {

  google::InitGoogleLogging(argv[0]);




  auto &topo = topology::getInstance();
  auto &nodes =   topo.getCpuNumaNodes();

  assert(nodes.size() >= 2 );
  assert(NUM_OLTP_CLIENTS <= nodes[0].local_cores.size());

  // INIT
  struct OLAP_STATS *analytical_stats = (struct OLAP_STATS*) get_shm("olap_stats", sizeof(struct OLAP_STATS)*NUM_OLAP_CLIENTS);
  

  bench::Benchmark *oltp_bench = init_oltp(nodes[0].local_cores.size(), "");
  


  olap_init_warmup(nodes[1]);
  init_olap_sequence(nodes[1]);

  // RUNOLTP
  run_oltp(nodes[0], nodes[0].local_cores.size(), oltp_bench);


  
  for(int i =0; i < NUM_OLAP_CLIENTS; i++){

    pid_t tmp = fork();
    if(tmp == 0){
        // run olap stuff
      run_olap_sequence(i, analytical_stats + (i*NUM_TPCH_QUERIES) , nodes[1]);
      break;
    } else {
      children.emplace_back(tmp);
    }

  }

  // some child process
  if(children.size() != NUM_OLAP_CLIENTS)
    return 0;

  

  register_handler();
  usleep(RUNTIME_SECONDS * 1000000);

  shutdown_oltp(true);
  shutdown_olap();

  // collect and print stats here.
  

  LOG(INFO) << "Shutting down...";

  return 0;
}
