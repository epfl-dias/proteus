/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_OLTP_UTILS_HPP
#define PROTEUS_OLTP_UTILS_HPP

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

// namespace utils {

#define FALSE 0
#define TRUE 1

//#include "transactions/transaction_manager.hpp"

// std::ostream& operator<<(std::ostream& o, const struct txn::TXN& a) {
//   o << "---TXN---\n";
//   for (int i = 0; i < a.n_ops; i++) {
//     o << "\tTXN[" << i << "]";
//     o << "\t\tOP: ";
//     switch (a.ops[i].op_type) {
//       case txn::OPTYPE_LOOKUP:
//         o << " LOOKUP";
//         break;
//       case txn::OPTYPE_INSERT:
//         o << " INSERT";
//         break;
//       case txn::OPTYPE_UPDATE:
//         o << " UPDATE";
//         break;
//       /*case txn::OP_TYPE_DELETE:
//         o << " DELETE";
//         break;*/
//       default:
//         o << " UNKNOWN";
//         break;
//     }
//     o << ", key: " << a.ops[i].key << std::endl;
//     ;
//   }

//   o << "---------\n";
//   return o;
// }

/*
  Calls a function every x(ms) intervals

*/

// template <class Duration>
// using sys_time = std::chrono::time_point<std::chrono::system_clock,
// Duration>; using sys_nanoseconds = sys_time<std::chrono::nanoseconds>;
// sys_nanoseconds now = std::chrono::system_clock::now();

/*
 Class: timed_func
 Example Usage:
    timed_func::interval_runner(
      [] { DoSomething(); }, (interval in milliseconds.));
 * */
class timed_func {
  static bool terminate;
  static int num_active_runners;

 public:
  static void init() { terminate = false; }
  static void terminate_all_timed() { terminate = true; }

  static void interval_runner(std::function<void(void)> func,
                              unsigned int interval) {
    std::thread([func, interval]() {
      while (true) {
        // std::cout << "HE" << std::endl;
        if (terminate) break;
        auto x = std::chrono::steady_clock::now();
        x += std::chrono::milliseconds(interval);

        func();

        std::this_thread::sleep_until(x);
      }
    }).detach();
    num_active_runners++;
  }

  /*template <class F, class... Args>
  static void interval_runner(F&& f, Args&&... args, unsigned int interval) {
    // using packaged_task_t =
    //   std::packaged_task<typename std::result_of<F(Args...)>::type()>;

    // packaged_task_t task(new packaged_task_t(
    //   std::bind(std::forward<F>(f), std::forward<Args>(args)...)));

    // auto res = task.get_future();

    std::thread([f, args, interval]() {
      while (true) {
        if (terminate) break;
        auto x = std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(interval);
        // task();
        f(args);
        std::this_thread::sleep_until(x);
      }
    }).detach();
    num_active_runners++;
  }*/
};

static inline int __attribute__((always_inline))
RAND(unsigned int *seed, int max) {
  // return rand_r(seed) % max;

  static thread_local std::mt19937 gen;
  return gen() % max;
}

static inline int __attribute__((always_inline))
URand(unsigned int *seed, int x, int y) {
  return x + RAND(seed, y - x + 1);
}

static inline int NURand(unsigned int *seed, int A, int x, int y) {
  static thread_local char C_255_init = FALSE;
  static thread_local char C_1023_init = FALSE;
  static thread_local char C_8191_init = FALSE;
  static thread_local int C_255, C_1023, C_8191;
  int C = 0;
  switch (A) {
    case 255:
      if (!C_255_init) {
        C_255 = URand(seed, 0, 255);
        C_255_init = TRUE;
      }
      C = C_255;
      break;
    case 1023:
      if (!C_1023_init) {
        C_1023 = URand(seed, 0, 1023);
        C_1023_init = TRUE;
      }
      C = C_1023;
      break;
    case 8191:
      if (!C_8191_init) {
        C_8191 = URand(seed, 0, 8191);
        C_8191_init = TRUE;
      }
      C = C_8191;
      break;
    default:
      assert(0);
      exit(-1);
  }
  return (((URand(seed, 0, A) | URand(seed, x, y)) + C) % (y - x + 1)) + x;
}

static inline int make_alpha_string(unsigned int *seed, int min, int max,
                                    char *str) {
  const char char_list[] = {
      '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
      'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
      'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
      'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
      'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
  int cnt = URand(seed, min, max);
  for (uint32_t i = 0; i < cnt; i++) str[i] = char_list[URand(seed, 0L, 60L)];

  for (int i = cnt; i < max; i++) str[i] = '\0';

  return cnt;
}

static inline int make_numeric_string(unsigned int *seed, int min, int max,
                                      char *str) {
  int cnt = URand(seed, min, max);

  for (int i = 0; i < cnt; i++) {
    int r = URand(seed, 0L, 9L);
    str[i] = '0' + r;
  }
  return cnt;
}

//};

#endif /* PROTEUS_OLTP_UTILS_HPP */
