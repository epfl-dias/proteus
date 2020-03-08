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

#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <chrono>
#include <functional>
#include <string>
#include <util/glog.hpp>

class [[nodiscard]] time_block {
 private:
  std::function<void(std::chrono::milliseconds)> reg;
  std::chrono::time_point<std::chrono::system_clock> start;

 public:
  inline explicit time_block(decltype(reg) reg)
      : reg(std::move(reg)), start(std::chrono::system_clock::now()) {}

  inline explicit time_block(std::string text = "",
                             decltype(__builtin_FILE()) file = __builtin_FILE(),
                             decltype(__builtin_LINE()) line = __builtin_LINE())
      : time_block([text{std::move(text)}, file, line](const auto &t) {
          google::LogMessage(file, line, google::GLOG_INFO).stream()
              << text << t.count() << "ms";
        }) {}

  inline ~time_block() {
    auto end = std::chrono::system_clock::now();
    reg(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
  }
};

#endif /* TIMING_HPP_ */
