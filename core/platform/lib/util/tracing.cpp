/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include "platform/util/tracing.hpp"

#include <fstream>
#include <iostream>
#include <magic_enum.hpp>
#include <stduuid/uuid.hpp>
#include <thread>

#include "glog/logging.h"
#include "platform/util/rdtsc.hpp"

#ifndef NTRACE
struct log_info {
  const uuids::uuid id;
  const unsigned long long timestamp;  /// rdtsc
  const int cpu_id;
  const std::thread::id tid;
  const log_op op;

  void flush() const;
};

struct counter_log_info {
  const uuids::uuid id;
  const unsigned long long timestamp;  /// rdtsc
  const int index;
  const int value;
  const counter_type op;

  void flush() const;
};


struct ranged_log_info {
  uuids::uuid id;
  unsigned long long timestamp_start;  /// rdtsc
  unsigned long long timestamp_end;    /// rdtsc
  int start_cpu_id;
  int end_cpu_id;  /// start may differ from end, if a context switch happened
                   /// in-between
  std::thread::id tid;
  range_log_op op;
  std::optional<uuids::uuid> pipeline_id;  /// PipelineGen->getUUID(). i.e
                                           /// unique id for a compiled pipeline
  int64_t instance_id;  /// Usually Pipeline->GetGroup(). i.e index of a
                        /// pipeline instance created by the same PipelineGen

  void flush() const;
};

// the definition order matters!
static std::stringstream global_ranged_log;
static std::stringstream global_counter_log;
static std::stringstream global_log;

void logger::log(uuids::uuid id, log_op op) {
  data->push_back(
      log_info{id, rdtsc(), sched_getcpu(), std::this_thread::get_id(), op});
}

void counter_logger::log(uuids::uuid id, counter_type op, int value,
                         int index) {
  data->push_back(counter_log_info{id, rdtsc(), index, value, op});
}

ranged_logger::start_rec ranged_logger::log_start(
    const uuids::uuid id, range_log_op op,
    std::optional<const uuids::uuid> pipeline_id, int64_t instance_id) {
  return start_rec{id, rdtsc(), sched_getcpu(), op, pipeline_id, instance_id};
}

void ranged_logger::log(ranged_logger::start_rec r) {
  data->push_back(ranged_log_info{r.id, r.timestamp_start, rdtsc(), r.cpu_id,
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
        // Avoid using event_range during global teardown: it relies on
        // thread_local loggers, whose destruction order relative to globals is
        // not guaranteed.
        for (const auto &data : logs) {
          for (const auto &t : data) t.flush();
        }
        {
          std::ofstream out("timeline.csv");
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

class flush_counter_log {
  std::deque<std::deque<counter_log_info>> logs;
  std::mutex m;

 public:
  flush_counter_log() : logs(1024) {}

  std::deque<counter_log_info> *create_logger() {
    std::lock_guard<std::mutex> lock(m);
    logs.emplace_back();
    return &(logs.back());
  }

  ~flush_counter_log() {
        // Avoid using event_range during global teardown: it relies on
        // thread_local loggers, whose destruction order relative to globals is
        // not guaranteed.
        for (const auto &data : logs) {
          for (const auto &t : data) t.flush();
        }
        {
          std::ofstream out("timeline-counters.csv");
          if (out.is_open()) {
            out << "timestamp,operator,value,counter_index,counter\n";
            out << global_counter_log.str();
            LOG(INFO) << "counter log flushed";
          }
        }
        {
          std::ofstream out{"timeline-counters-oplegend.csv"};
          if (out.is_open()) {
            out << "counter_name,value\n";
            for (auto &e : magic_enum::enum_entries<decltype(logs[0][0].op)>()) {
              out << e.second << ',' << static_cast<int>(e.first) << '\n';
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
        // Avoid using event_range during global teardown: it relies on
        // thread_local loggers, whose destruction order relative to globals is
        // not guaranteed.
        for (const auto &data : logs) {
          for (const auto &t : data) t.flush();
        }
        {
          std::ofstream out("timeline-ranges.csv");
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
static flush_counter_log global_exchange_flush_counter_lock;
static flush_log global_exchange_flush_lock;
#endif

thread_local ranged_logger rangelogger;
thread_local logger eventlogger;
thread_local counter_logger counterlogger;

#ifndef NTRACE
logger::logger() {
  data = global_exchange_flush_lock.create_logger();
  event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
      {},
      std::nullopt,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count()};
}

counter_logger::counter_logger() {
  data = global_exchange_flush_counter_lock.create_logger();
}

void log_info::flush() const {
  global_log << timestamp << "," << uuids::to_string(id) << "," << tid << ","
             << cpu_id << "," << op << "\n";
}

void counter_log_info::flush() const {
  global_counter_log << timestamp << "," << uuids::to_string(id) << "," << value
                     << "," << index << "," << static_cast<int>(op) << "\n";
}

ranged_logger::ranged_logger() {
  data = global_exchange_flush_range_lock.create_logger();
  // Used in post-processing to convert rdtsc ticks to nanoseconds
  event_range<range_log_op::LOGGER_TIMESTAMP> markrange{
      {},
      std::nullopt,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count()};
}

void ranged_log_info::flush() const {
  global_ranged_log << timestamp_start << ',' << timestamp_end << ","
                    << uuids::to_string(id) << "," << tid << "," << start_cpu_id
                    << ',' << end_cpu_id << "," << static_cast<int>(op) << ','
                    << (pipeline_id.has_value()
                            ? uuids::to_string(pipeline_id.value())
                            : "")
                    << ',' << instance_id << "\n";
}
#endif
