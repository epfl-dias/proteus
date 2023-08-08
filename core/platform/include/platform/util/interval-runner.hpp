/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_INTERVAL_RUNNER_HPP
#define PROTEUS_INTERVAL_RUNNER_HPP

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <vector>
namespace proteus::utils {

/*
 * Calls a function every x ms intervals
 * Class: timed_func
 * Example Usage:
 *    timed_func::interval_runner(
 *      [] { DoSomething(); }, (interval in milliseconds));
 *
 * */
class timed_func {
 public:
  class Runner {
   public:
    explicit Runner(unsigned int interval_ms)
        : _interval_ms(interval_ms), _terminate_runner(false) {}

    void setInterval(unsigned int interval_ms) {
      this->_interval_ms = interval_ms;
    }
    [[nodiscard]] auto getInterval() const { return this->_interval_ms; }
    [[maybe_unused]] void terminate() { this->_terminate_runner = true; }
    [[nodiscard]] auto active() const { return !_terminate_runner; }

   private:
    unsigned int _interval_ms;
    bool _terminate_runner;

    friend class timed_func;
  };

 public:
  static void terminate_all_timed() { terminate_all = true; }

  static auto interval_runner(const std::function<void(void)>& func,
                              unsigned int interval_ms) {
    auto& runner = runners.emplace_back(interval_ms);

    std::thread([func, runner]() {
      while (true) {
        if (terminate_all || runner._terminate_runner) break;
        auto x = std::chrono::steady_clock::now();
        x += std::chrono::milliseconds(runner._interval_ms);

        func();

        std::this_thread::sleep_until(x);
      }
    }).detach();

    return runner;
  }

 private:
  static bool terminate_all;
  static std::vector<Runner> runners;
};

}  // namespace proteus::utils

#endif  // PROTEUS_INTERVAL_RUNNER_HPP
