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

#ifndef PROTEUS_TIME_REGISTRY_HPP
#define PROTEUS_TIME_REGISTRY_HPP

#include <chrono>
#include <map>
#include <mutex>
#include <string>

class TimeRegistry {
 public:
  class [[nodiscard]] Key {
   private:
    std::string key;

   public:
    explicit Key(std::string key) : key(std::move(key)) {}

    [[nodiscard]] inline const std::string &getKey() const { return key; }

    [[nodiscard]] bool operator<(const Key &o) const { return key < o.key; }

    [[nodiscard]] bool operator==(const Key &o) const {
      if (this == &o) return true;
      return key == o.key;
    }
  };

  static const Key Ignore;

  static TimeRegistry &getInstance() {
    static TimeRegistry instance{};
    return instance;
  }

 private:
  std::mutex m;
  std::map<Key, std::chrono::nanoseconds> registry;

  TimeRegistry() = default;

 public:
  ~TimeRegistry();

  template <typename Rep, typename Dur>
  inline void emplace(const Key &k, const std::chrono::duration<Rep, Dur> &t) {
    if (k == Ignore) return;
    std::scoped_lock<std::mutex> lock{m};  // to protect insert/reference
    registry[k] += t;
  }
};

#endif /* PROTEUS_TIME_REGISTRY_HPP */
