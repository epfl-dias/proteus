/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "common/common.hpp"

#include <mutex>
#include <storage/storage-manager.hpp>
#include <thread>

#include "memory/memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/logging.hpp"

double diff(struct timespec st, struct timespec end) {
  struct timespec tmp;

  if ((end.tv_nsec - st.tv_nsec) < 0) {
    tmp.tv_sec = end.tv_sec - st.tv_sec - 1;
    tmp.tv_nsec = 1e9 + end.tv_nsec - st.tv_nsec;
  } else {
    tmp.tv_sec = end.tv_sec - st.tv_sec;
    tmp.tv_nsec = end.tv_nsec - st.tv_nsec;
  }

  return tmp.tv_sec + tmp.tv_nsec * 1e-9;
}

void fatal(const char *err) {
  perror(err);
  exit(1);
}

void exception(const char *err) {
  printf("Exception: %s\n", err);
  exit(1);
}

std::ostream &operator<<(std::ostream &out, const bytes &b) {
  const char *units[]{"B", "KB", "MB", "GB", "TB", "ZB"};
  constexpr size_t max_i = sizeof(units) / sizeof(units[0]);
  size_t bs = b.b * 10;

  size_t i = 0;
  while (bs >= 10240 && i < max_i) {
    bs /= 1024;
    ++i;
  }

  out << (bs / 10.0) << units[i];
  return out;
}

namespace std {
std::string to_string(const bytes &b) {
  std::stringstream ss;
  ss << b;
  return ss.str();
}
};  // namespace std

#ifndef NLOG

struct log_info {
  void *dop;
  unsigned long long timestamp;
  int cpu_id;
  std::thread::id tid;
  log_op op;

  void flush() const;
};

// the definition order matters!
static stringstream global_log;
// std::mutex                      global_log_lock    ;

#if defined(__powerpc64__) || defined(__ppc64__)
uint64_t __rdtsc() {
  uint64_t c;
  asm volatile("mfspr %0, 268" : "=r"(c));
  return c;
}
#else
#include <x86intrin.h>
#endif

// logger::~logger(){
//     std::lock_guard<std::mutex> lg(global_log_lock);
//     for (const auto &t: data) t.flush();
// }

void logger::log(void *dop, log_op op) {
  data->push_back(
      log_info{dop, __rdtsc(), sched_getcpu(), std::this_thread::get_id(), op});
}

class flush_log {
  std::deque<std::deque<log_info>> logs;
  std::mutex m;

 public:
  flush_log() : logs(1024) {}

  std::deque<log_info> *create_logger() {
    std::lock_guard<std::mutex> lock(m);
    logs.emplace_back();
    return &(logs.back());
  }

  ~flush_log() {
    for (const auto &data : logs) {
      for (const auto &t : data) t.flush();
    }
    ofstream out("../../src/panorama/public/assets/timeline.csv");
    out << "timestamp,operator,thread_id,coreid,op" << std::endl;
    out << global_log.str();
    std::cout << "log flushed" << std::endl;
  }
};

// the definition order matters!
static flush_log global_exchange_flush_lock;
#endif

thread_local logger eventlogger;

#ifndef NLOG
logger::logger() { data = global_exchange_flush_lock.create_logger(); }

void log_info::flush() const {
  global_log << timestamp << "," << dop << "," << tid << "," << cpu_id << ","
             << op << "\n";
}

#endif

namespace proteus {

void thread_warm_up() {}

class platform::impl {
 public:
  impl(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
       size_t log_buffers) {
    topology::init();

    // Initialize Google's logging library.
    LOG(INFO) << "Starting up server...";

    LOG(INFO) << "Warming up GPUs...";
    for (const auto &gpu : topology::getInstance().getGpus()) {
      set_exec_location_on_scope d{gpu};
      gpu_run(cudaFree(nullptr));
    }

    gpu_run(cudaFree(nullptr));

    // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

    LOG(INFO) << "Warming up threads...";

    std::vector<std::thread> thrds;
    thrds.reserve(1024);
    for (int i = 0; i < 1024; ++i) thrds.emplace_back(thread_warm_up);
    for (auto &t : thrds) t.join();

    // srand(time(0));

    LOG(INFO) << "Initializing memory manager...";
    MemoryManager::init(gpu_mem_pool_percentage, cpu_mem_pool_percentage,
                        log_buffers);

    // Make affinity deterministic
    auto &topo = topology::getInstance();
    if (topo.getGpuCount() > 0) {
      exec_location{topo.getGpus()[0]}.activate();
    } else {
      exec_location{topo.getCpuNumaNodes()[0]}.activate();
    }
  }

  ~impl() {
    LOG(INFO) << "Shutting down...";

    LOG(INFO) << "Unloading files...";
    StorageManager::getInstance().unloadAll();

    LOG(INFO) << "Shuting down memory manager...";
    MemoryManager::destroy();

    LOG(INFO) << "Shut down finished";
  }
};

platform::platform(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
                   size_t log_buffers)
    : p_impl(std::make_unique<impl>(gpu_mem_pool_percentage,
                                    cpu_mem_pool_percentage, log_buffers)) {}

platform::~platform() = default;

}  // namespace proteus
