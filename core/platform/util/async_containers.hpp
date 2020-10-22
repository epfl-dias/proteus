/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef ASYNC_CONTAINERS_HPP_
#define ASYNC_CONTAINERS_HPP_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>

#include "common/gpu/gpu-common.hpp"

template <typename T>
class AsyncStackSPSC {
 private:
  std::mutex m;
  std::condition_variable cv;

  std::vector<T> data;

  std::atomic<bool> terminating;

 public:
  AsyncStackSPSC() : terminating(false) {}

  void close() {
    nvtxRangePushA("AsyncStack_o");
    std::unique_lock<std::mutex> lock(m);

    terminating = true;
    cv.notify_all();

    nvtxRangePushA("AsyncStack_t");
    cv.wait(lock, [this]() { return data.empty(); });

    lock.unlock();
    nvtxRangePop();
    nvtxRangePop();
  }

  void push(const T x) {
    assert(!terminating);
    std::unique_lock<std::mutex> lock(m);
    data.push_back(x);
    cv.notify_all();
    lock.unlock();
  }

  bool pop(T &x) {
    std::unique_lock<std::mutex> lock(m);

    if (data.empty()) {
      cv.wait(lock, [this]() {
        return !data.empty() || (data.empty() && terminating);
      });
    }

    if (data.empty()) {
      assert(terminating);
      lock.unlock();

      cv.notify_all();
      return false;
    }

    x = data.back();
    data.pop_back();

    lock.unlock();
    return true;
  }

  T pop_unsafe() {
    T x = data.back();
    data.pop_back();
    return x;
  }

  bool empty_unsafe() { return data.empty(); }
};

template <typename T>
class AsyncQueueSPSC {
 public:
  typedef T value_type;

 private:
  std::mutex m;
  std::condition_variable cv;

  std::queue<T> data;

  std::atomic<bool> terminating;

  std::atomic<int> cache_size;
  T cache_data[8];

 public:
  AsyncQueueSPSC() : terminating(false), cache_size(0) {}

  void reset() {
    terminating = false;
    cache_size = 0;
  }

  void close() {
    // push(nullptr);
    nvtxRangePushA("AsyncQueue_o");
    std::unique_lock<std::mutex> lock(m);
    terminating = true;
    cv.notify_all();

    nvtxRangePushA("AsyncQueue_t");
    cv.wait(lock, [this]() { return (cache_size == 0) && data.empty(); });

    lock.unlock();
    nvtxRangePop();
    nvtxRangePop();
  }

  void push(const T x) {
    // nvtxRangePushA(("AsyncQueue_push" + std::to_string((uint64_t)
    // x)).c_str());
    assert(!terminating);
    std::unique_lock<std::mutex> lock(m);
    // nvtxRangePushA("AsyncQueue_push_w_lock");
    data.push(x);
    cv.notify_all();
    // nvtxRangePop();
    lock.unlock();
    // nvtxRangePop();
  }

  bool pop(T &x) {
    if (cache_size > 0) {
      x = cache_data[--cache_size];
      return true;
    }

    // nvtxRangePushA("AsyncQueue_pop");
    std::unique_lock<std::mutex> lock(m);

    // while (data.empty()){
    if (data.empty()) {
      cv.wait(lock, [this]() {
        return !data.empty() || (data.empty() && terminating);
      });
    }

    if (data.empty()) {
      assert(terminating);
      lock.unlock();

      cv.notify_all();
      // nvtxRangePop();
      return false;
    }

    x = data.front();
    // nvtxRangePushA(("AsyncQueue_pop" + std::to_string((uint64_t)
    // x)).c_str());
    data.pop();

    size_t to_retrieve = std::min(data.size(), (size_t)8);
    cache_size = to_retrieve;
    for (size_t i = 0; i < to_retrieve; ++i) {
      cache_data[to_retrieve - i - 1] = data.front();
      data.pop();
    }

    // assert(x != nullptr || data.empty());

    lock.unlock();
    // nvtxRangePop();
    // nvtxRangePop();
    return true;
    // return x != nullptr;
  }

  T pop_unsafe() {
    if (cache_size > 0) return cache_data[--cache_size];

    T x = data.front();
    data.pop();
    return x;
  }

  bool empty_unsafe() { return (cache_size == 0) && data.empty(); }
};

template <typename T>
using AsyncQueueMPSC = AsyncQueueSPSC<T>;

// template<typename T, size_t size>
// class AsyncQueueSPSC_spin{
// private:
//     volatile uint32_t front;
//     volatile uint32_t back ; //TODO: should we put them in different cache
//     lines ?

//     volatile bool     terminating;
//     volatile T data[size];
// public:
//     AsyncQueueSPSC(): front(0), back(-1), terminating(false){}

//     void close(){
//         nvtxRangePushA("AsyncQueue_o");
//         terminating = true;

//         while (f > back + 1);
//         nvtxRangePop();
//     }

//     void push(const T &x){
//         // assert(!terminating);
//         uint32_t f = front;
//         while (f - back - 1 >= size);

//         data[f % size] = x;

//         front = f + 1;
//     }

//     bool pop(T &x){
//         uint32_t b = back;

//         while (front <= b + 1 && !terminating);

//         x = data[b % size];

//         return true;
//     }

//     T pop_unsafe(){
//         T x = data.front();
//         data.pop();
//         return x;
//     }
// };

template <typename T, T eof = nullptr>
class AsyncQueueSPSC_lockfree {
  static constexpr int size = 32;

 private:
  // volatile uint32_t front;
  // volatile uint32_t back ; //TODO: should we put them in different cache
  // lines ?
  std::atomic<int> _tail;

  T _array[size];

  std::atomic<bool> terminating;

  std::atomic<int> _head;

 public:
  AsyncQueueSPSC_lockfree() : terminating(false), _tail(0), _head(0) {}

  void close(bool wait = true) {
    // push(nullptr);
    nvtxRangePushA("AsyncQueue_o");
    // std::unique_lock<std::mutex> lock(m);
    // terminating = true;
    push(eof);
    // cv.notify_all();

    // nvtxRangePushA("AsyncQueue_t");

    // cv.wait(lock, [this](){return data.empty();});

    if (wait)
      while (_head.load() != _tail.load())
        ;

    // lock.unlock();
    nvtxRangePop();
    nvtxRangePop();
  }

  // void push(const T x){
  //     nvtxRangePushA(("AsyncQueue_push" + std::to_string((uint64_t)
  //     x)).c_str()); assert(!terminating); std::unique_lock<std::mutex>
  //     lock(m); nvtxRangePushA("AsyncQueue_push_w_lock"); data.push(x);
  //     cv.notify_all();
  //     nvtxRangePop();
  //     lock.unlock();
  //     nvtxRangePop();
  // }

  void push(const T &x) {
    int current_tail = _tail.load(std::memory_order_relaxed);
    int next_tail = (current_tail + 1) % size;
    do {
    } while (next_tail == _head.load(std::memory_order_acquire));

    _array[current_tail] = x;
    _tail.store(next_tail, std::memory_order_release);
  }

  bool pop(T &item) {
    int current_head = _head.load(std::memory_order_relaxed);
    while (current_head == _tail.load(std::memory_order_acquire)) {
      // if (terminating.load(std::memory_order_acquire)){
      //     return false;
      // }
      // } return false; // empty queue
    }

    item = _array[current_head];
    _head.store((current_head + 1) % size, std::memory_order_release);
    return item != eof;
  }

  T pop_unsafe() {
    T x;
    pop(x);
    return x;
  }

  // bool pop(T &x){
  //     nvtxRangePushA("AsyncQueue_pop");
  //     std::unique_lock<std::mutex> lock(m);

  //     // while (data.empty()){
  //     if (data.empty()){
  //         cv.wait(lock, [this](){return !data.empty() || (data.empty() &&
  //         terminating);});
  //     }

  //     if (data.empty()){
  //         assert(terminating);
  //         lock.unlock();

  //         cv.notify_all();
  //         nvtxRangePop();
  //         return false;
  //     }

  //     x = data.front();
  //     nvtxRangePushA(("AsyncQueue_pop" + std::to_string((uint64_t)
  //     x)).c_str()); data.pop();

  //     // assert(x != nullptr || data.empty());

  //     lock.unlock();
  //     nvtxRangePop();
  //     nvtxRangePop();
  //     return true;
  //     // return x != nullptr;
  // }

  // T pop_unsafe(){
  //     T x = data.front();
  //     data.pop();
  //     return x;
  // }

  // bool empty_unsafe(){
  //     return data.empty();
  // }
};

#endif /* ASYNC_CONTAINERS_HPP_ */
