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

#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#include <deque>

struct log_info;
struct ranged_log_info;

// <0  : point event
// >=0 : start/stop events
//     : % 2 == 0 ---> start
//     : % 2 == 1 ---> stop
enum log_op {
  MEMORY_MANAGER_ALLOC_GPU_START = 14,
  MEMORY_MANAGER_ALLOC_GPU_END = 15,
  EXCHANGE_INIT_CONS_START = 20,
  EXCHANGE_INIT_CONS_END = 21,
  MEMMOVE_OPEN_START = 22,
  MEMMOVE_OPEN_END = 23,
  MEMMOVE_CLOSE_START = 28,
  MEMMOVE_CLOSE_END = 29,
  MEMMOVE_CLOSE_CLEAN_UP_START = 30,
  MEMMOVE_CLOSE_CLEAN_UP_END = 31,
  CPU2GPU_OPEN_START = 32,
  CPU2GPU_OPEN_END = 33,
  CPU2GPU_CLOSE_START = 34,
  CPU2GPU_CLOSE_END = 35,
  LAST_LOG_OP,
};

enum class range_log_op {
  NON_RANGE = LAST_LOG_OP,
  CUPTI_GET_START_TIMESTAMP,
  LOGGER_TIMESTAMP,
  IB_BUFFS_GET_START_TIMESTAMP,
  IB_BUFFS_GET_END_TIMESTAMP,
  IB_LOCK_CONN,
  IB_CQ_PROCESSING_EVENT,
  IB_SENDING_BUFFERS,
  IB_SENDING_BUFFERS_WAITING,
  CPU2GPU_LAUNCH,
  EXCHANGE_INIT_CONS,
  EXCHANGE_CONSUMER_WAIT,
  EXCHANGE_PRODUCE,
  EXCHANGE_PRODUCE_PUSH,
  ROUTER_ACQUIRING_FREE_QUEUE_SLOT,
  ROUTER_WAITING_FOR_TASK,
  SPLIT_GET,
  SPLIT_RELEASE,
  MEMMOVE_OPEN,
  MEMMOVE_CONSUME,
  MEMMOVE_CLOSE,
  MEMMOVE_CLOSE_CLEAN_UP,
  AFFINITY_QUERY_DEVICE,
  UNPACK_OPEN,
};

// #define NLOG

#ifndef NLOG
class logger {
  std::deque<log_info> *data;

 public:
  logger();

  // ~logger();

  void log(void *dop, log_op op);
};

class ranged_logger {
  std::deque<ranged_log_info> *data;

 public:
  ranged_logger();

  // ~logger();
  struct start_rec {
    const void *dop;
    unsigned long long timestamp_start;
    int cpu_id;
    range_log_op op;
    void *pipeline_id;
    int64_t instance_id;  // Similar to PipelineGroupId
  };

  static start_rec log_start(const void *dop, range_log_op op,
                             void *pipeline_id, int64_t instance_id);
  void log(ranged_logger::start_rec r);
};
#else
class logger {
 public:
  inline void log(void *dop, log_op op){};
};

class ranged_logger {
 public:
  struct start_rec {};

  static start_rec log_start(void *dop, log_op op) { return {}; };
  inline void log(void *dop, log_op op){};
};
#endif

extern thread_local ranged_logger rangelogger;
extern thread_local logger eventlogger;

#ifndef NLOG
template <range_log_op op>
class [[nodiscard]] event_range {
  ranged_logger::start_rec r;

 public:
  static constexpr range_log_op event_type = op;

  constexpr explicit event_range(const void *t, void *pipeline_id = nullptr,
                                 int64_t instance_id = -1) noexcept
      : r(ranged_logger::log_start(t, op, pipeline_id, instance_id)) {}

  event_range(event_range &&) noexcept = delete;
  event_range &operator=(event_range &&) noexcept = delete;
  event_range(const event_range &) = delete;
  event_range &operator=(const event_range &) = delete;

  constexpr ~event_range() noexcept { rangelogger.log(r); }
};

#else

template <range_log_op op>
class [[nodiscard]] event_range {
 public:
  static constexpr range_log_op event_type = op;

  constexpr explicit event_range(const void *t, void *pipeline_id = nullptr,
                                 int64_t instance_id = -1) noexcept {}

  event_range(event_range &&) noexcept = delete;
  event_range &operator=(event_range &&) noexcept = delete;
  event_range(const event_range &) = delete;
  event_range &operator=(const event_range &) = delete;

  constexpr ~event_range() noexcept {}
};

#endif

#endif /* LOGGING_HPP_ */
