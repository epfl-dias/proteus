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

#ifndef PROTEUS_PERCENTILE_HPP
#define PROTEUS_PERCENTILE_HPP

#include <cassert>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/memory/allocator.hpp>
#include <platform/util/rdtsc.hpp>
#include <utility>

namespace proteus::utils {

class threadLocal_percentile;
class percentile_point_rdtsc;

template <class T = std::chrono::nanoseconds>
class percentile_point_clock;

using percentile_point_ns = percentile_point_clock<std::chrono::nanoseconds>;
using percentile_point_us = percentile_point_clock<std::chrono::microseconds>;
using percentile_point_ms = percentile_point_clock<std::chrono::milliseconds>;
using percentile_point_s = percentile_point_clock<std::chrono::seconds>;

using percentile_point = percentile_point_rdtsc;

class [[nodiscard]] Percentile {
 public:
  /**
   * Construct a non-global Percentile and optional @param thread_safe
   * Percentile
   */
  explicit Percentile(bool thread_safe = false);
  /**
   * Construct a global Percentile that is accessible through @see
   * PercentileRegistry using
   * @param key
   * PercentileRegistry only holds a raw pointer to the newly constructed
   * Percentile. Constructs a threadsafe Percentile
   */
  explicit Percentile(const std::string& key);

  ~Percentile() = default;

  inline void add(size_t value) {
    if (m_threadsafe) {
      std::unique_lock<std::mutex> lock(m_points_mutex);
      m_points.push_back(value);
    } else {
      m_points.push_back(value);
    }
  }

  inline void add(const Percentile& p) {
    if (m_threadsafe) {
      std::unique_lock<std::mutex> lock(m_points_mutex);
      std::copy(p.m_points.begin(), p.m_points.end(),
                std::back_inserter(m_points));
    } else {
      std::copy(p.m_points.begin(), p.m_points.end(),
                std::back_inserter(m_points));
    }
  }

  inline void add(const std::vector<size_t>& v) {
    if (m_threadsafe) {
      std::unique_lock<std::mutex> lock(m_points_mutex);
      std::copy(v.begin(), v.end(), std::back_inserter(m_points));
    } else {
      std::copy(v.begin(), v.end(), std::back_inserter(m_points));
    }
  }

  size_t size() {
    if (m_threadsafe) {
      std::unique_lock<std::mutex> lock(m_points_mutex);
      return m_points.size();
    } else {
      return m_points.size();
    }
  }

  /**
   * Should not be on the critical path
   * @return return the @param n th percentile. Returns 0 if there are no points
   * recorded
   * @note not thread safe
   */
  size_t nth(double n);

  /**
   * Should not be on the critical path
   * @return the mean of the recorded points. Returns -1 if there are no points
   * recorded
   * @note not thread safe
   */
  double mean() const;

  void save_cdf(const std::string& out_path, size_t step = 1000);

  /**
   * @return return the nth percentile
   * @note not thread safe
   */
  size_t operator[](double n) {
    assert(n > 0 && n <= 100);
    return nth(n);
  }

 private:
  std::deque<size_t> m_points;
  std::mutex m_points_mutex;
  const bool m_threadsafe;
};

/**
 * Global singleton class that holds non-owning pointers to instances of @see
 * Percentile
 */
class [[nodiscard]] PercentileRegistry {
 public:
  /**
   * @param key key to register Percentile with
   * @param global_cdf Pointer to a Percentile to register
   * @return True if the Percentile was inserted, false if it replaced an
   * existing Percentile
   */
  static inline bool register_global(const std::string& key,
                                     Percentile* global_cdf) {
    LOG(INFO) << "registering global: " << key;
    std::unique_lock<std::mutex> lock(g_lock);
    return PercentileRegistry::global_registry.insert_or_assign(key, global_cdf)
        .second;
  }

  [[maybe_unused]] static inline Percentile* get_global(
      const std::string& key) {
    return PercentileRegistry::global_registry[key];
  }

  /**
   * not thread safe
   * @param f lambda to apply to each key-Percentile pair
   */
  [[maybe_unused]] static inline void for_each(void (*f)(std::string key,
                                                         Percentile* p)) {
    for (const auto& [k, val] : global_registry) {
      LOG(INFO) << "[GlobalPercentileRegistry][for_each] Key: " << k;
      f(k, val);
    }
  }

  [[maybe_unused]] static inline void for_each(
      void (*f)(std::string key, Percentile* p, void* args), void* args) {
    for (const auto& [k, val] : global_registry) {
      LOG(INFO) << "[GlobalPercentileRegistry][for_each] Key: " << k;
      f(k, val, args);
    }
  }

 private:
  PercentileRegistry() = default;

  static std::map<std::string, Percentile*> global_registry;
  static std::mutex g_lock;

  friend class threadLocal_percentile;
  friend class Percentile;
};

class [[nodiscard]] threadLocal_percentile {
 public:
  /**
   * A thread-local percentile that will add to the global percentile with
   * matching @param key on destruction.
   * The threadsafety assumes that no new keys will be inserted into the
   * PercentileRegistry when either the constructor or destructor if
   * threadLocal_percentile execute
   */
  explicit threadLocal_percentile(const std::string& key) : key(key) {
    LOG(INFO) << "threadLocal_percentile registered: " << key;
    if (PercentileRegistry::global_registry.find(key) ==
        PercentileRegistry::global_registry.end()) {
      throw std::runtime_error("global cdf not registered.");
    }
  }

  ~threadLocal_percentile() {
    // this is threadsafe if the Percentile in the registry is threadsafe
    // as we only read the global map
    PercentileRegistry::global_registry[this->key]->add(this->p);
  }

 private:
  Percentile p{};
  const std::string key;

  friend class percentile_point_parent;
};

class [[nodiscard]] percentile_point_parent {
 protected:
  explicit percentile_point_parent(Percentile& registry) : registry(registry) {}
  explicit percentile_point_parent(threadLocal_percentile& p_reg)
      : registry(p_reg.p) {}
  Percentile& registry;
};

template <typename T>
class [[nodiscard]] percentile_point_clock : public percentile_point_parent {
 public:
  inline explicit percentile_point_clock(Percentile& registry)
      : percentile_point_parent(registry),
        start(std::chrono::system_clock::now()) {}

  inline explicit percentile_point_clock(threadLocal_percentile& p_reg)
      : percentile_point_parent(p_reg),
        start(std::chrono::system_clock::now()) {}

  inline ~percentile_point_clock() {
    registry.add(
        std::chrono::duration_cast<T>(std::chrono::system_clock::now() - start)
            .count());
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start;
};

class [[nodiscard]] percentile_point_rdtsc : public percentile_point_parent {
 public:
  inline explicit percentile_point_rdtsc(Percentile& registry)
      : percentile_point_parent(registry), start(rdtsc()) {}

  inline explicit percentile_point_rdtsc(threadLocal_percentile& p_reg)
      : percentile_point_parent(p_reg), start(rdtsc()) {}

  inline ~percentile_point_rdtsc() { registry.add(rdtsc() - start); }

 private:
  const uint64_t start;
};
}  // namespace proteus::utils

#endif  // PROTEUS_PERCENTILE_HPP
