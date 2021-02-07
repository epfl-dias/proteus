/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include "oltp/util/interval-map.hpp"

// ::testing::Environment *const pools_env =
//     ::testing::AddGlobalTestEnvironment(new testing::TestEnvironment);

// class AtomicBitSetTest : public ::testing::Test {

// };
using IntervalMap = utils::IntervalMap<size_t, size_t>;

TEST(IntervalMap, Simple) {
  IntervalMap map;
  map.upsert(5, 1);

  for (uint i = 0; i < 5; i++) {
    EXPECT_EQ(map.getValue(i), 0);
  }
  EXPECT_EQ(map.getValue(5), 1);
  EXPECT_EQ(map.getValue(6), 0);

  map.upsert(5, 2);
  EXPECT_EQ(map.getValue(5), 2);

  for (uint i = 0; i < 5; i++) {
    EXPECT_EQ(map.getValue(i), 0);
  }
  EXPECT_EQ(map.getValue(5), 2);
  EXPECT_EQ(map.getValue(6), 0);

  map.upsert(5, 2);
  EXPECT_EQ(map.getValue(5), 2);

  map.upsert(1, 2);
  EXPECT_EQ(map.getValue(1), 2);
  EXPECT_EQ(map.getValue(2), 0);
  EXPECT_EQ(map.getValue(7), 0);
  EXPECT_EQ(map.getValue(0), 0);

  map.upsert(0, 5);
  map.upsert(5, 0);
  std::cout << map << std::endl;
  map.consolidate();
  std::cout << "Consolidated" << std::endl;
  std::cout << map << std::endl;
}

TEST(IntervalMap, UpdateByValue) {
  IntervalMap map;
  map.upsert(5, 1);

  for (uint i = 0; i < 5; i++) {
    EXPECT_EQ(map.getValue(i), 0);
  }
  EXPECT_EQ(map.getValue(5), 1);
  EXPECT_EQ(map.getValue(6), 0);

  map.upsert(5, 2);
  EXPECT_EQ(map.getValue(5), 2);

  for (uint i = 0; i < 5; i++) {
    EXPECT_EQ(map.getValue(i), 0);
  }
  EXPECT_EQ(map.getValue(5), 2);
  EXPECT_EQ(map.getValue(6), 0);

  map.upsert(5, 2);
  EXPECT_EQ(map.getValue(5), 2);

  map.upsert(1, 2);
  EXPECT_EQ(map.getValue(1), 2);
  EXPECT_EQ(map.getValue(2), 0);
  EXPECT_EQ(map.getValue(7), 0);
  EXPECT_EQ(map.getValue(0), 0);

  map.upsert(0, 5);

  map.upsert(7, 6);
  map.upsert(20, 6);
  map.upsert(50, 6);
  map.upsert(10, 6);

  std::cout << map << std::endl;
  auto count = map.updateByValue_withCount(6, 0);
  std::cout << "UpdateByValue count:" << count << std::endl;
  std::cout << map << std::endl;
  map.consolidate();
  std::cout << "Consolidated" << std::endl;
  std::cout << map << std::endl;
}
