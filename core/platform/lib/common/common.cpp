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

#include <magic_enum.hpp>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/storage/storage-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/logging.hpp>
#include <platform/util/rdtsc.hpp>
#include <thread>

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

struct ranged_log_info {
  const void *dop;
  unsigned long long timestamp_start;
  unsigned long long timestamp_end;
  int start_cpu_id;
  int end_cpu_id;  // start may differ from end, if a context switch happened
                   // in-between
  std::thread::id tid;
  range_log_op op;
  void *pipeline_id;
  int64_t instance_id;  // Similar to PipelineGroupId

  void flush() const;
};

// the definition order matters!
static stringstream global_ranged_log;
static stringstream global_log;
// std::mutex                      global_log_lock    ;

// logger::~logger(){
//     std::lock_guard<std::mutex> lg(global_log_lock);
//     for (const auto &t: data) t.flush();
// }

void logger::log(void *dop, log_op op) {
  data->push_back(
      log_info{dop, rdtsc(), sched_getcpu(), std::this_thread::get_id(), op});
}

ranged_logger::start_rec ranged_logger::log_start(const void *dop,
                                                  range_log_op op,
                                                  void *pipeline_id,
                                                  int64_t instance_id) {
  return start_rec{dop, rdtsc(), sched_getcpu(), op, pipeline_id, instance_id};
}

void ranged_logger::log(ranged_logger::start_rec r) {
  data->push_back(ranged_log_info{r.dop, r.timestamp_start, rdtsc(), r.cpu_id,
                                  sched_getcpu(), std::this_thread::get_id(),
                                  r.op, r.pipeline_id, r.instance_id});
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
    event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
        reinterpret_cast<void *>(static_cast<uintptr_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count()))};
    for (const auto &data : logs) {
      for (const auto &t : data) t.flush();
    }
    {
      ofstream out("timeline.csv");
      if (out.is_open()) {
        out << "timestamp,operator,thread_id,coreid,op\n";
        out << global_log.str();
        LOG(INFO) << "event log flushed";
      }
    }
    {
      std::ofstream out{"timeline-oplegend.csv"};
      if (out.is_open()) {
        out << "op,value\n";
        for (auto &e : magic_enum::enum_entries<decltype(logs[0][0].op)>()) {
          out << e.second << ',' << e.first << '\n';
        }
      }
    }
  }
};

class flush_range_log {
  std::mutex m;
  std::deque<std::deque<ranged_log_info>> logs;

 public:
  flush_range_log() {}

  std::deque<ranged_log_info> *create_logger() {
    std::lock_guard<std::mutex> lock(m);
    logs.emplace_back();
    return &(logs.back());
  }

  ~flush_range_log() {
    event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
        reinterpret_cast<void *>(static_cast<uintptr_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count()))};
    for (const auto &data : logs) {
      for (const auto &t : data) t.flush();
    }
    {
      ofstream out("timeline-ranges.csv");
      if (out.is_open()) {
        out << "timestamp_start,timestamp_end,operator,thread_id,coreid_start,"
               "coreid_end,op,pipeline_id,instance_id\n";
        out << global_ranged_log.str();
        LOG(INFO) << "range log flushed";
      }
    }
    {
      std::ofstream out{"timeline-ranges-oplegend.csv"};
      if (out.is_open()) {
        out << "op,value\n";
        for (auto &e : magic_enum::enum_entries<decltype(logs[0][0].op)>()) {
          out << e.second << ',' << static_cast<int>(e.first) << '\n';
        }
      }
    }
  }
};

// the definition order matters!
static flush_range_log global_exchange_flush_range_lock;
static flush_log global_exchange_flush_lock;
#endif

thread_local ranged_logger rangelogger;
thread_local logger eventlogger;

#ifndef NLOG
logger::logger() {
  data = global_exchange_flush_lock.create_logger();
  event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
      reinterpret_cast<void *>(static_cast<uintptr_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count()))};
}

void log_info::flush() const {
  global_log << timestamp << "," << dop << "," << tid << "," << cpu_id << ","
             << op << "\n";
}

ranged_logger::ranged_logger() {
  data = global_exchange_flush_range_lock.create_logger();
  event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
      reinterpret_cast<void *>(static_cast<uintptr_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count()))};
}

void ranged_log_info::flush() const {
  global_ranged_log << timestamp_start << ',' << timestamp_end << "," << dop
                    << "," << tid << "," << start_cpu_id << ',' << end_cpu_id
                    << "," << static_cast<int>(op) << ',' << pipeline_id << ','
                    << instance_id << "\n";
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
