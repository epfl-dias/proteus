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

#ifndef THREADSAFE_STACK_CUH_
#define THREADSAFE_STACK_CUH_

#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>

template <typename T, T invalid_value = std::numeric_limits<T>::max()>
class threadsafe_stack {
 private:
  std::vector<T> data;
  std::mutex m;
  std::condition_variable cv;
  size_t size;

 public:
  __host__ threadsafe_stack(size_t size, std::vector<T> fill) : size(size) {
    data.reserve(size);
    for (const auto &t : fill) data.push_back(t);
  }

  __host__ ~threadsafe_stack() {
    std::cout << "=======================================================>"
                 "host_stack: "
              << data.size() << " " << size << std::endl;
    // assert(data.size() == size);
  }

 public:
  __host__ void push(T v) {
    std::unique_lock<std::mutex> lock(m);
    data.push_back(v);
    cv.notify_all();
  }

  __host__ bool try_pop(T *ret) {  // blocking (as long as the stack is not
                                   // empty)
    std::unique_lock<std::mutex> lock(m);
    if (data.empty()) return false;
    *ret = data.back();
    data.pop_back();
    return true;
  }

  __host__ T pop() {  // blocking
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return !data.empty(); });
    T ret = data.back();
    data.pop_back();
    return ret;
  }

  __host__ __device__ static bool is_valid(T &x) { return x != invalid_value; }

  __host__ __device__ static T get_invalid() { return invalid_value; }
};

#endif /* THREADSAFE_STACK_CUH_ */
