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
#include <thread>

#include "communication/comm-manager.hpp"
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

size_t getFileSize(const char *filename) {
  struct stat st;
  stat(filename, &st);
  return st.st_size;
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
stringstream global_log;
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
flush_log global_exchange_flush_lock;
#endif

thread_local logger eventlogger;

#ifndef NLOG
logger::logger() { data = global_exchange_flush_lock.create_logger(); }

void log_info::flush() const {
  global_log << timestamp << "," << dop << "," << tid << "," << cpu_id << ","
             << op << "\n";
}

#endif
