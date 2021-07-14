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
#include <platform/util/glog.hpp>
#include <platform/util/stacktrace.hpp>
#include <platform/util/time-registry.hpp>
#include <string>
#include <thread>

template <typename Tunit>
constexpr const char *asUnit();

template <>
constexpr const char *asUnit<std::chrono::milliseconds>() {
  return "ms";
}

template <>
constexpr const char *asUnit<std::chrono::microseconds>() {
  return "us";
}

template <>
constexpr const char *asUnit<std::chrono::nanoseconds>() {
  return "ns";
}

template <>
constexpr const char *asUnit<std::chrono::seconds>() {
  return "s";
}

template <class Rep, class Period>
std::ostream &operator<<(std::ostream &out,
                         const std::chrono::duration<Rep, Period> &duration) {
  out << duration.count() << asUnit<std::chrono::duration<Rep, Period>>();
  return out;
}

class nested_time_block {
 public:
  static size_t &getNestLevel() {
    static thread_local size_t nest_level = 0;
    return nest_level;
  }
};

class Glog {
 private:
  int severity;

 public:
  inline constexpr explicit Glog(int s) : severity(s) {}

  friend class Severity;
  template <typename Tduration, typename Tclock>
  friend class time_blockT;
};

class Severity {
 public:
  static constexpr Glog VVVERBOSE{-3};
  static constexpr Glog VVERBOSE{-2};
  static constexpr Glog VERBOSE{-1};
  static constexpr Glog INFO{google::GLOG_INFO};
  static constexpr Glog WARNING{google::GLOG_WARNING};
  static constexpr Glog ERROR{google::GLOG_ERROR};
  static constexpr Glog FATAL{google::GLOG_FATAL};
};

template <typename Tduration = std::chrono::milliseconds,
          typename Tclock = std::chrono::system_clock>
class [[nodiscard]] time_blockT {
 protected:
  using clock = Tclock;
  using dur = typename Tclock::duration;

  std::function<void(Tduration)> reg;
  std::chrono::time_point<clock> start;
  TimeRegistry::Key k;

  static_assert(dur{1} <= Tduration{1}, "clock not precise enough");

 public:
  inline explicit time_blockT(decltype(reg) reg,
                              TimeRegistry::Key k = TimeRegistry::Ignore)
      : reg(std::move(reg)), start(clock::now()), k(std::move(k)) {}

  inline explicit time_blockT(
      std::string text = "", TimeRegistry::Key k = TimeRegistry::Ignore,
      Glog glogSeverity = Severity::INFO,
      decltype(__builtin_FILE()) file = __builtin_FILE(),
      decltype(__builtin_LINE()) line = __builtin_LINE())
      : time_blockT(
            [text{std::move(text)}, file, line, glogSeverity](const auto &t) {
              auto s = --nested_time_block::getNestLevel();
              if (glogSeverity.severity >= 0 ||
                  VLOG_IS_ON(-glogSeverity.severity)) {
                google::LogMessage(file, line,
                                   std::max(glogSeverity.severity, 0))
                        .stream()
                    << '\t' << std::string(s, '|') << text << t;
              }
            },
            std::move(k)) {
    ++nested_time_block::getNestLevel();
  }

  inline explicit time_blockT(
      std::string text, Glog glogSeverity,
      decltype(__builtin_FILE()) file = __builtin_FILE(),
      decltype(__builtin_LINE()) line = __builtin_LINE())
      : time_blockT(std::move(text), TimeRegistry::Ignore,
                    std::move(glogSeverity), file, line) {}

  inline explicit time_blockT(TimeRegistry::Key k = TimeRegistry::Ignore)
      : reg([](const auto &t) {}), start(clock::now()), k(std::move(k)) {}

  inline ~time_blockT() {
    auto end = clock::now();
    auto d = std::chrono::duration_cast<Tduration>(end - start);
    TimeRegistry::getInstance().emplace(k, d);
    reg(d);
  }
};

class time_block : public time_blockT<std::chrono::milliseconds> {
  using time_blockT::time_blockT;
};

class time_block_us : public time_blockT<std::chrono::microseconds> {
  using time_blockT::time_blockT;
};

class time_block_ns : public time_blockT<std::chrono::nanoseconds,
                                         std::chrono::high_resolution_clock> {
  using time_blockT::time_blockT;
};

class [[nodiscard]] time_bomb {
 protected:
  class clock {
   public:
    proteus::stacktrace trace;
    bool escaped = false;
  };

 public:
  std::shared_ptr<clock> timer;

  template <class Rep, class Period>
  explicit time_bomb(std::chrono::duration<Rep, Period> duration,
                     decltype(__builtin_FILE()) file = __builtin_FILE(),
                     decltype(__builtin_LINE()) line = __builtin_LINE())
      : timer(std::make_shared<clock>()) {
    std::thread(
        [duration, clk = this->timer, file, line](proteus::stacktrace strace) {
          std::this_thread::sleep_for(duration);
          if (!clk->escaped) {
            google::LogMessage(file, line, google::GLOG_FATAL).stream()
                << "Time out " << strace;
          }
        },
        proteus::stacktrace{})
        .detach();
  }

  ~time_bomb() { timer->escaped = true; }
};

#endif /* TIMING_HPP_ */
