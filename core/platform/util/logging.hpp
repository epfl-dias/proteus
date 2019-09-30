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

// <0  : point event
// >=0 : start/stop events
//     : % 2 == 0 ---> start
//     : % 2 == 1 ---> stop
enum log_op {
  EXCHANGE_PRODUCE = -1,
  EXCHANGE_CONSUME_OPEN_START = 0,
  EXCHANGE_CONSUME_OPEN_END = 1,
  EXCHANGE_CONSUME_CLOSE_START = 2,
  EXCHANGE_CONSUME_CLOSE_END = 3,
  EXCHANGE_CONSUME_START = 4,
  EXCHANGE_CONSUME_END = 5,
  EXCHANGE_CONSUMER_WAIT_START = 6,
  EXCHANGE_CONSUMER_WAIT_END = 7,
  EXCHANGE_PRODUCER_WAIT_START = 8,
  EXCHANGE_PRODUCER_WAIT_END = 9,
  EXCHANGE_PRODUCER_WAIT_FOR_FREE_START = 10,
  EXCHANGE_CONSUMER_WAIT_FOR_FREE_END = 11,
  MEMORY_MANAGER_ALLOC_PINNED_START = 12,
  MEMORY_MANAGER_ALLOC_PINNED_END = 13,
  MEMORY_MANAGER_ALLOC_GPU_START = 14,
  MEMORY_MANAGER_ALLOC_GPU_END = 15,
  EXCHANGE_PRODUCE_START = 16,
  EXCHANGE_PRODUCE_END = 17,
  EXCHANGE_PRODUCE_PUSH_START = 18,
  EXCHANGE_PRODUCE_PUSH_END = 19,
  EXCHANGE_INIT_CONS_START = 20,
  EXCHANGE_INIT_CONS_END = 21,
  MEMMOVE_OPEN_START = 22,
  MEMMOVE_OPEN_END = 23,
  MEMMOVE_CONSUME_WAIT_START = 24,
  MEMMOVE_CONSUME_WAIT_END = 25,
  MEMMOVE_CONSUME_START = 26,
  MEMMOVE_CONSUME_END = 27,
  MEMMOVE_CLOSE_START = 28,
  MEMMOVE_CLOSE_END = 29,
  MEMMOVE_CLOSE_CLEAN_UP_START = 30,
  MEMMOVE_CLOSE_CLEAN_UP_END = 31,
  CPU2GPU_OPEN_START = 32,
  CPU2GPU_OPEN_END = 33,
  CPU2GPU_CLOSE_START = 34,
  CPU2GPU_CLOSE_END = 35,
  EXCHANGE_JOIN_START = 36,
  EXCHANGE_JOIN_END = 37,
  KERNEL_LAUNCH_START = 38,
  KERNEL_LAUNCH_END = 39,
  THREADPOOL_THREAD_START = 40,
  THREADPOOL_THREAD_END = 41,
  BLOCK2TUPLES_OPEN_START = 42,
  BLOCK2TUPLES_OPEN_END = 43,
  IB_LOCK_CONN_START = 44,
  IB_LOCK_CONN_END = 45,
  IB_CQ_PROCESSING_EVENT_START = 46,
  IB_CQ_PROCESSING_EVENT_END = 47,
  IB_RDMA_WAIT_BUFFER_START = 48,
  IB_RDMA_WAIT_BUFFER_END = 49,
  IB_WAITING_DATA_START = 50,
  IB_WAITING_DATA_END = 51,
  IB_CREATE_RDMA_READ_START = 52,
  IB_CREATE_RDMA_READ_END = 53,
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
#else
class logger {
 public:
  inline void log(void *dop, log_op op){};
};
#endif

extern thread_local logger eventlogger;

#endif /* LOGGING_HPP_ */
