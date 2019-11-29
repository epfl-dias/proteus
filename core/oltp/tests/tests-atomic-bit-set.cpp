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

#include <gtest/gtest.h>

#include "utils/atomic_bit_set.hpp"

// ::testing::Environment *const pools_env =
//     ::testing::AddGlobalTestEnvironment(new testing::TestEnvironment);

// class AtomicBitSetTest : public ::testing::Test {

// };

TEST(AtomicBitSet, Simple) {
  constexpr size_t kSize = 1024;
  utils::AtomicBitSet<kSize> bs;

  EXPECT_EQ(kSize, bs.size());

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FALSE(bs[i]);
  }

  bs.set(42);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(i == 42, bs[i]);
  }

  bs.set(43);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ((i == 42 || i == 43), bs[i]);
  }

  bs.reset(42);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ((i == 43), bs[i]);
  }

  bs.reset(43);
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FALSE(bs[i]);
  }
}

TEST(AtomicBitSet, Extensions) {
  constexpr size_t kSize = 1024;
  utils::AtomicBitSet<kSize> bs;

  EXPECT_EQ(bs.count(), 0);

  bs.set(512);

  EXPECT_TRUE(bs.any());
  EXPECT_EQ(bs.count(), 1);

  bs.reset(512);

  EXPECT_FALSE(bs.any());

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_FALSE(bs[i]);
    bs.set(i);
    EXPECT_EQ(bs.count(), i + 1);
  }

  EXPECT_EQ(bs.count(), kSize);
  EXPECT_TRUE(bs.all());

  bs.reset();

  EXPECT_FALSE(bs.any());
  EXPECT_FALSE(bs.all());
  EXPECT_EQ(bs.count(), 0);
}
