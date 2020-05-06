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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

#include "util/glog.hpp"

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

namespace std {
std::string to_string(const bytes &b);
}

template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v);
}

std::ostream &operator<<(std::ostream &out, const bytes &b);

namespace proteus {

class [[nodiscard]] platform {
 private:
  class impl;
  std::unique_ptr<impl> p_impl;

 public:
  platform(float gpu_mem_pool_percentage = 0.05,
           float cpu_mem_pool_percentage = 0.05, size_t log_buffers = 0);
  ~platform();
};

}  // namespace proteus

constexpr size_t operator"" _K(unsigned long long int x) { return x * 1024; }

constexpr size_t operator"" _M(unsigned long long int x) { return x * 1024_K; }

constexpr size_t operator"" _G(unsigned long long int x) { return x * 1024_M; }

#endif /* COMMON_HPP_ */
