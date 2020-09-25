/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_THREADVECTOR_HPP
#define PROTEUS_THREADVECTOR_HPP

#include <future>
#include <threadpool/threadpool.hpp>
#include <vector>

class threadvector {
  std::vector<std::future<void>> threads;

  static auto &getPool() {
    static ThreadPool pool{true};
    return pool;
  }

 public:
  template <class F, class... Args>
  void emplace_back(F &&f, Args &&...args) {
    threads.emplace_back(
        getPool().enqueue(std::forward<F>(f), std::forward<Args>(args)...));
  }

  [[nodiscard]] auto empty() const { return threads.empty(); }

  void clear() { threads.clear(); }

  [[nodiscard]] auto &operator[](size_t i) { return threads[i]; }

  [[nodiscard]] auto &operator[](size_t i) const { return threads[i]; }

  [[nodiscard]] auto size() const { return threads.size(); }

  [[nodiscard]] auto begin() const { return threads.begin(); }

  [[nodiscard]] auto end() const { return threads.end(); }

  [[nodiscard]] auto begin() { return threads.begin(); }

  [[nodiscard]] auto end() { return threads.end(); }
};

#endif /* PROTEUS_THREADVECTOR_HPP */
