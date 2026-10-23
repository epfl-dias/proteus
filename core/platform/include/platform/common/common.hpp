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

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <platform/util/glog.hpp>
#include <stdexcept>

// #define DEBUG
// #define LOCAL_EXEC
// #undef DEBUG
#undef LOCAL_EXEC

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

using std::cout;
using std::list;
using std::map;
using std::runtime_error;
using std::string;

double diff(struct timespec st, struct timespec end);

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

class bytes {
 private:
  size_t b;

 public:
  bytes(size_t b) : b(b) {}

  friend std::ostream &operator<<(std::ostream &out, const bytes &b);

  bytes &operator+=(const bytes &o) {
    b += o.b;
    return *this;
  }

  [[nodiscard]] explicit operator size_t() const { return b; }
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

constexpr size_t operator""_K(unsigned long long int x) { return x * 1024; }

constexpr size_t operator""_M(unsigned long long int x) { return x * 1024_K; }

constexpr size_t operator""_G(unsigned long long int x) { return x * 1024_M; }

#endif /* COMMON_HPP_ */
