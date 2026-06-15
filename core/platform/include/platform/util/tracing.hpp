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

#ifndef PROTEUS_TRACING_HPP
#define PROTEUS_TRACING_HPP

#include <deque>
#include <stduuid/uuid.hpp>

/**
 * If you update the the loggers, please test and update the parser
 * (in tools/tracing) as necessary
 */

/**
 * To disable tracing at compile time, define NTRACE
 */
//#define NTRACE

struct log_info;
struct counter_log_info;
struct ranged_log_info;

/**
 * Point in time events
 * New events can be added without updating the parser
 */
enum log_op {
  LAST_LOG_OP,
};

/**
 * Range events
 * New range events can be added without updating the parser
 */
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
  CPU2GPU_OPEN,
  CPU2GPU_CLOSE,
  ROUTER_OPEN,
  ROUTER_CLOSE,
  ROUTER_CLOSE_JOIN_CONS,
  ROUTER_ACQUIRING_FREE_QUEUE_SLOT,
  ROUTER_WAITING_FOR_TASK,
  ROUTER_CONSUME,
  SPLIT_GET,
  SPLIT_RELEASE,
  MEMMOVE_OPEN,
  MEMMOVE_CONSUME,
  MEMMOVE_GET_PROPOGATED,
  MEMMOVE_CLOSE,
  MEMMOVE_CLOSE_CLEAN_UP,
  AFFINITY_QUERY_DEVICE,
  UNPACK_OPEN,
  UNPACK_CLOSE,
  GROUTER_OPEN,
  GROUTER_CREATE_QUEUES,
  GROUTER_ALLOC_QUEUE_BUFFS,
  GROUTER_CONS_FIRE,
  GROUTER_CLOSE,
  GROUTER_INIT_CONS,
  QUERY_EXECUTE,
  OPERATOR_PIPELINE_EXECUTE,
  OPERATOR_PIPELINE_OPEN,
  OPERATOR_PIPELINE_OPEN_INIT_STATE,
  OPERATOR_PIPELINE_OPENER,
  BUILDER_PREPARE,
  MEMORY_MANAGER_MALLOC_GPU,
  UNION_ALL_OPEN,
  UNION_ALL_CLOSE,
  BINARY_BLOCK_LOAD_DATA,
  BINARY_BLOCK_GET_FIELD_DATA,
  TOPO_CPU_ALLOC,
  CUFILE_READ,
  RUN_GPU_KERNEL
};

/**
 * Counter types
 * New counters can be added without updating the parser
 */
enum class counter_type { ROUTER_READY_QUEUE_SIZE, ROUTER_FREE_POOL_SIZE };

#ifndef NTRACE
class logger {
  std::deque<log_info> *data;

 public:
  logger();

  void log(uuids::uuid id, log_op op);
};

class counter_logger {
  std::deque<counter_log_info> *data;

 public:
  counter_logger();

  /**
   * @param id of object this counter is associated with
   * @param op counter type
   * @param value value to record for the counter
   * @param index optional index to support multiple counters of the same type.
   */
  void log(uuids::uuid id, counter_type op, int value, int index = 0);
};

class ranged_logger {
  std::deque<ranged_log_info> *data;

 public:
  ranged_logger();

  struct start_rec {
    const uuids::uuid id;
    unsigned long long timestamp_start;
    int cpu_id;
    range_log_op op;
    std::optional<const uuids::uuid> pipeline_id;
    int64_t instance_id;  // Similar to PipelineGroupId
  };

  static start_rec log_start(uuids::uuid id, range_log_op op,
                             std::optional<const uuids::uuid> pipeline_id,
                             int64_t instance_id);
  void log(ranged_logger::start_rec r);
};
#else

class logger {
 public:
  inline void log(uuids::uuid id, log_op op) {}
};

class counter_logger {
  //std::deque<counter_log_info> *data;

 public:
  counter_logger() {}
  inline void log(uuids::uuid id, counter_type op, int value, int index = 0) {}
};

class ranged_logger {
 public:
  struct start_rec {};

  static start_rec log_start(uuids::uuid id, range_log_op op,
                             std::optional<const uuids::uuid> pipeline_id,
                             int64_t instance_id) {
    return {};
  }
  inline void log(ranged_logger::start_rec r) {}
};
#endif

extern thread_local ranged_logger rangelogger;
extern thread_local counter_logger counterlogger;
extern thread_local logger eventlogger;

#ifndef NLOG
template <range_log_op op>
class [[nodiscard]] event_range {
  ranged_logger::start_rec r;

 public:
  static constexpr range_log_op event_type = op;

  /**
   *
   * @param id of object this counter is associated with. e.g.
   * Operator->getUUID()
   * @param pipeline_id unique id for a compiled pipeline. e.g.
   * PipelineGen->getUUID().
   * @param instance_id index of a pipeline instance created by the same
   * PipelineGen. e.g. Pipeline->GetGroup()
   */
  constexpr explicit event_range(
      const uuids::uuid id,
      std::optional<const uuids::uuid> pipeline_id = std::nullopt,
      int64_t instance_id = -1) noexcept
      : r(ranged_logger::log_start(id, op, pipeline_id, instance_id)) {}

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

  constexpr explicit event_range(
      const uuids::uuid id,
      std::optional<const uuids::uuid> pipeline_id = std::nullopt,
      int64_t instance_id = -1) noexcept {}

  event_range(event_range &&) noexcept = delete;
  event_range &operator=(event_range &&) noexcept = delete;
  event_range(const event_range &) = delete;
  event_range &operator=(const event_range &) = delete;

  constexpr ~event_range() noexcept {}
};

#endif

#endif /* PROTEUS_TRACING_HPP */
