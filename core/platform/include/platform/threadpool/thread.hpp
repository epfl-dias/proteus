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

#ifndef THREAD_HPP_
#define THREAD_HPP_

#include <platform/topology/affinity_manager.hpp>
#include <thread>

namespace proteus {

class thread {
 private:
  std::thread t;

 public:
  thread(thread&& other) noexcept : t(std::move(other.t)) {}

  template <class Function, class... Args>
  explicit thread(Function&& f, Args&&... args)
      : t(
            [](exec_location e, Function&& f_, Args&&... args_) {
              set_exec_location_on_scope aff{e};
              f_(args_...);
            },
            exec_location{}, f, args...) {}

  thread(const thread&) = delete;
  thread& operator=(thread&& other) noexcept = default;

  void join() { t.join(); }
  void detach() { t.detach(); }
  void swap(thread& other) noexcept { t.swap(other.t); }
  static auto hardware_concurrency() noexcept {
    return std::thread::hardware_concurrency();
  }
  auto native_handle() { return t.native_handle(); }
  [[nodiscard]] auto joinable() const noexcept { return t.joinable(); }
  [[nodiscard]] auto get_id() const noexcept { return t.get_id(); }
};

}  // namespace proteus

#endif /* THREAD_HPP_ */
