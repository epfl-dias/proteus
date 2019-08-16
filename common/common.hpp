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

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

//#JSON
#define JSMN_STRICT
//
//#define JSON_TIGHT
#include "jsmn/jsmn.h"

// Used to remove all logging messages at compile time and not affect
// performance Must be placed before glog include
/*Setting GOOGLE_STRIP_LOG to 1 or greater removes all log messages associated
 * with VLOGs
 * as well as INFO log statements. Setting it to two removes WARNING log
 * statements too. */
#ifdef NDEBUG
#define GOOGLE_STRIP_LOG 2
#define STRIP_LOG 2
#endif

#define TIMING

#include <glog/logging.h>

//#define DEBUG
// #define LOCAL_EXEC
//#undef DEBUG
#undef LOCAL_EXEC

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::list;
using std::map;
using std::multimap;
using std::ofstream;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

double diff(struct timespec st, struct timespec end);

[[noreturn, deprecated]] void fatal(const char *err);

[[noreturn, deprecated]] void exception(const char *err);

namespace llvm {
// forward declaration to avoid including the whole header
class AllocaInst;
class Value;
}  // namespace llvm

/**
 * Wrappers for LLVM Value and Alloca.
 * Maintain information such as whether the corresponding value is 'NULL'
 * LLVM's interpretation of 'NULL' for primitive types is not sufficient
 * (e.g., lvvm_null(int) = 0
 */
struct ProteusValueMemory {
  llvm::AllocaInst *mem;
  llvm::Value *isNull;
};

struct ProteusValue {
  llvm::Value *value;
  llvm::Value *isNull;
};

/*
 * Util Methods
 */
template <typename M, typename V>
void MapToVec(const M &m, V &v) {
  for (typename M::const_iterator it = m.begin(); it != m.end(); ++it) {
    v.push_back(it->second);
  }
}

typedef size_t vid_t;
typedef uint32_t cid_t;
typedef uint32_t sel_t;
typedef uint32_t cnt_t;

class time_block {
 private:
  std::chrono::time_point<std::chrono::system_clock> start;
  std::string text;

 public:
  inline time_block(std::string text = "")
      : text(text), start(std::chrono::system_clock::now()) {}

  inline ~time_block() {
    auto end = std::chrono::system_clock::now();
    std::cout << text;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
  }
};

size_t getFileSize(const char *filename);

enum data_loc {
  GPU_RESIDENT,
  PINNED,
  PAGEABLE,
  ALLSOCKETS,
  ALLGPUS,
  EVERYWHERE,
};

struct mmap_file {
 private:
  int fd;

  size_t filesize;
  void *data;
  void *gpu_data;
  data_loc loc;

  bool readonly;

 public:
  mmap_file(std::string name, data_loc loc = GPU_RESIDENT);
  mmap_file(std::string name, data_loc loc, size_t bytes, size_t offset);
  ~mmap_file();

  const void *getData() const;
  size_t getFileSize() const;
};

class bytes {
 private:
  size_t b;

 public:
  bytes(size_t b) : b(b) {}

  friend std::ostream &operator<<(std::ostream &out, const bytes &b);
};

template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v);
}

std::ostream &operator<<(std::ostream &out, const bytes &b);

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

namespace proteus {
void init(bool inc_buffers = false);
}

#endif /* COMMON_HPP_ */
