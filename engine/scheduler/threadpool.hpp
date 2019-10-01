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
#include <queue>
#include <thread>

namespace scheduler {
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

 public:
  static ThreadPool &getInstance() {
    // Guaranteed-by-the-standard threadsafe initialization (and destruction)
    static ThreadPool instance;
    return instance;
  }

  void addThread() {
    workers.emplace_back([this] {
      // eventlogger.log(this, log_op::THREADPOOL_THREAD_START);
      while (true) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(m);
          cv.wait(lock, [this] { return terminate || !tasks.empty(); });

          if (terminate && tasks.empty()) break;

          task = std::move(tasks.front());
          tasks.pop();
        }

        task();
      }
      // eventlogger.log(this, log_op::THREADPOOL_THREAD_END);
    });
  }

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> enqueue(
      F &&f, Args &&... args) {
    using packaged_task_t =
        std::packaged_task<typename std::result_of<F(Args...)>::type()>;

    std::shared_ptr<packaged_task_t> task(new packaged_task_t(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)));

    auto res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(m);
      tasks.emplace([task]() { (*task)(); });
    }

    cv.notify_one();
    return res;
  }

 private:
  ThreadPool() : terminate(false) {
    size_t N = 4 * std::thread::hardware_concurrency();

    for (size_t i = 0; i < N; ++i) addThread();
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

}  // namespace scheduler

#endif /* THREADPOOL_HPP_ */
