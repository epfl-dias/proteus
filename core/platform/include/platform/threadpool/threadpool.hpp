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

#ifndef THREADPOOL_HPP_
#define THREADPOOL_HPP_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <queue>
#include <thread>

namespace details {
namespace ThreadPool {
template <class F, class C, class... Args>
auto call(F &&f, C &&c, Args &&...args) -> decltype((
    std::forward<C>(c)->*std::forward<F>(f))(std::forward<Args>(args)...)) {
  return (std::forward<C>(c)->*(std::forward<F>(f)))(
      std::forward<Args>(args)...);
}

template <class F, class... Args>
auto call(F &&f, Args &&...args)
    -> decltype(std::forward<F>(f)(std::forward<Args>(args)...)) {
  return (std::forward<F>(f))(std::forward<Args>(args)...);
}
};  // namespace ThreadPool
};  // namespace details

/*
 * Based on: https://github.com/progschj/ThreadPool
 */
class ThreadPool {
 private:
  std::atomic<bool> terminate;

  std::mutex m;
  std::condition_variable cv;

  // Use deque that guarantees that objects are not copied/moved
  std::deque<std::thread> workers;

  std::queue<std::function<void()>> tasks;

  const bool elastic;
  std::atomic<int> idleWorkers;

 public:
  static ThreadPool &getInstance() {
    // Guaranteed-by-the-standard threadsafe initialization (and destruction)
    static ThreadPool instance{true};
    return instance;
  }

  void addThread() {
    workers.emplace_back(
        [this](size_t i) {
          set_exec_location_on_scope aff{
              topology::getInstance().getCpuNumaNodes()
                  [i % topology::getInstance().getCpuNumaNodeCount()]};
          // eventlogger.log(this, log_op::THREADPOOL_THREAD_START);
          while (true) {
            pthread_setname_np(pthread_self(), "idle (pool)");

            decltype(tasks)::value_type task;

            {
              std::unique_lock<std::mutex> lock(m);
              ++idleWorkers;
              cv.wait(lock, [this] { return terminate || !tasks.empty(); });
              --idleWorkers;

              if (terminate && tasks.empty()) break;

              task = std::move(tasks.front());
              tasks.pop();
            }

            pthread_setname_np(pthread_self(), "working");

            task();
          }
          // eventlogger.log(this, log_op::THREADPOOL_THREAD_END);
        },
        workers.size());
  }

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueue(
      F &&f, Args &&...args) {
    using packaged_task_t =
        std::packaged_task<typename std::result_of<F(Args...)>::type()>;

    class wrapper {
      std::decay_t<F> callable;
      std::tuple<std::decay_t<Args>...> _args;

     public:
      wrapper(F &&f, Args &&...args)
          : callable(std::forward<F>(f)),
            _args(std::make_tuple(std::forward<Args>(args)...)) {}

      auto operator()() {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) {
          return details::ThreadPool::call(
              std::move(callable), std::move(std::get<indices>(_args))...);
        }
        (std::make_index_sequence<sizeof...(Args)>{});
      }
    };

    auto w = std::make_shared<wrapper>(std::forward<F>(f),
                                       std::forward<Args>(args)...);

    auto task = std::make_shared<packaged_task_t>(std::bind(
        [](std::shared_ptr<wrapper> p) { return (*p)(); }, std::move(w)));

    auto res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(m);
      tasks.emplace([task]() { (*task)(); });
      if (elastic && idleWorkers < tasks.size()) addThread();
    }

    cv.notify_all();
    return res;
  }

  explicit ThreadPool(
      bool elastic = false,
      size_t initialNumberOfThreads = 4 * std::thread::hardware_concurrency())
      : terminate(false), elastic(elastic), idleWorkers(0) {
    for (size_t i = 0; i < initialNumberOfThreads; ++i) addThread();
  }

  ~ThreadPool() {
    terminate = true;
    cv.notify_all();
    for (auto &worker : workers) worker.join();
  }

 public:
  // Prevent copies
  ThreadPool(const ThreadPool &) = delete;
  void operator=(const ThreadPool &) = delete;

  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;
};

#endif /* THREADPOOL_HPP_ */
