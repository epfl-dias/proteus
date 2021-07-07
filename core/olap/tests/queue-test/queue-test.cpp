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

#include <lib/util/datastructures/threadsafe-set.hpp>
#include <olap/test/timeout.hpp>

class QueueTest : public ::testing::Test {};

using namespace std::chrono_literals;

TEST_F(QueueTest, emplace_completes) {
  EXPECT_FINISHES(15s, {
    threadsafe_set<int> q;

    int val = 5;
    q.emplace(val);

    EXPECT_EQ(q.size_unsafe(), 1);
    EXPECT_FALSE(q.empty_unsafe());
  });
}

TEST_F(QueueTest, one_item_non_concurrent) {
  EXPECT_FINISHES(15s, {
    threadsafe_set<int> q;

    int val = 5;
    q.emplace(val);

    EXPECT_FALSE(q.empty_unsafe());
    EXPECT_EQ(q.pop(), val);
    EXPECT_TRUE(q.empty_unsafe());
  });
}
