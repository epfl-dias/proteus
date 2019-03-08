/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <unistd.h>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>

#include "transactions/transaction_manager.hpp"

std::ostream& operator<<(std::ostream& o, const struct txn::TXN& a) {
  o << "---TXN---\n";
  for (int i = 0; i < a.n_ops; i++) {
    o << "\tTXN[" << i << "]";
    o << "\t\tOP: ";
    switch (a.ops[i].op_type) {
      case txn::OPTYPE_LOOKUP:
        o << " LOOKUP";
        break;
      case txn::OPTYPE_INSERT:
        o << " INSERT";
        break;
      case txn::OPTYPE_UPDATE:
        o << " UPDATE";
        break;
      /*case txn::OP_TYPE_DELETE:
        o << " DELETE";
        break;*/
      default:
        o << " UNKNOWN";
        break;
    }
    o << ", key: " << a.ops[i].key << std::endl;
    ;
  }

  o << "---------\n";
  return o;
}

/*
  Calls a function every x(ms) intervals

*/

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
        if (terminate) break;
        auto x = std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(interval);
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

bool timed_func::terminate = false;
int timed_func::num_active_runners = 0;

#endif /* UTILS_HPP_ */
