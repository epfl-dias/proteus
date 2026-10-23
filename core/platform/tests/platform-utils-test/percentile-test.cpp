/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <gtest/gtest.h>

#include <future>
#include <platform/util/percentile.hpp>

using namespace proteus::utils;

static void AddPercentilePoints(Percentile& p) {
  for (size_t i = 1000; i < 1100; i++) {
    p.add(i);
  }
}

static void AddPercentilePointClockGlobal(
    const std::string& registry_key, std::chrono::milliseconds sleep_time) {
  threadLocal_percentile lp(registry_key);
  percentile_point_ms cd(lp);
  std::this_thread::sleep_for(sleep_time);
}

static void AddPercentilePointClock(Percentile& p,
                                    std::chrono::milliseconds sleep_time) {
  percentile_point_ms cd(p);
  std::this_thread::sleep_for(sleep_time);
}

TEST(Percentile, threadlocal) {
  auto p = Percentile(false);

  AddPercentilePoints(p);

  EXPECT_EQ(p.nth(0.00000001), 1000);
  EXPECT_EQ(p.nth(100), 1099);
  EXPECT_FLOAT_EQ(p.mean(), 104950.0 / 100.0);
}

TEST(Percentile, threadsafe) {
  auto p = Percentile(true);

  std::vector<std::future<void>> futures;
  for (size_t j = 0; j < 10; j++) {
    futures.push_back(
        std::async(std::launch::async, &AddPercentilePoints, std::ref(p)));
  }
  for (size_t j = 0; j < 10; j++) {
    futures[j].wait();
  }
  EXPECT_EQ(p.nth(0.00000001), 1000);
  EXPECT_EQ(p.nth(100), 1099);
  EXPECT_FLOAT_EQ(p.mean(), 104950.0 / 100.0);
}

TEST(Percentile, PercentilePointClock) {
  auto p = Percentile(true);

  std::vector<std::future<void>> futures;
  for (size_t j = 1; j < 11; j++) {
    futures.push_back(std::async(std::launch::async, &AddPercentilePointClock,
                                 std::ref(p),
                                 std::chrono::milliseconds(j * 10)));
  }
  for (size_t j = 0; j < 10; j++) {
    futures[j].wait();
  }
  EXPECT_EQ(std::chrono::milliseconds(p.nth(0.00000001)).count(),
            std::chrono::milliseconds(10).count());
  EXPECT_EQ(std::chrono::milliseconds(p.nth(100)).count(),
            std::chrono::milliseconds(100).count());
}

TEST(Percentile, PercentileRegistry) {
  const std::string key = "test-percentile";
  auto p = Percentile(key);

  std::vector<std::future<void>> futures;
  for (size_t j = 1; j < 11; j++) {
    futures.push_back(std::async(std::launch::async,
                                 &AddPercentilePointClockGlobal, key,
                                 std::chrono::milliseconds(j * 10)));
  }
  for (size_t j = 0; j < 10; j++) {
    futures[j].wait();
  }

  EXPECT_EQ(std::chrono::milliseconds(p.nth(0.00000001)).count(),
            std::chrono::milliseconds(10).count());
  EXPECT_EQ(std::chrono::milliseconds(p.nth(100)).count(),
            std::chrono::milliseconds(100).count());

  auto* percentile_from_registry = PercentileRegistry::get_global(key);
  EXPECT_EQ(std::chrono::milliseconds(percentile_from_registry->nth(0.00000001))
                .count(),
            std::chrono::milliseconds(10).count());
  EXPECT_EQ(
      std::chrono::milliseconds(percentile_from_registry->nth(100)).count(),
      std::chrono::milliseconds(100).count());
}
