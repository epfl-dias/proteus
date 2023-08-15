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

#include <platform/util/timing.hpp>

#include "gtest/gtest.h"
#include "oltp/index/ART/art.hpp"
#include "test-utils.hpp"

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class ARTIndexTest : public ::testing::Test {
 protected:
  ARTIndexTest() { LOG(INFO) << "ARTIndexTest()"; }
  ~ARTIndexTest() override { LOG(INFO) << "~ARTIndexTest()"; }

  //  void SetUp() override;
  //  void TearDown() override;

  const char *testPath = "/tests-index-art/";
  const char *catalogJSON = "inputs";

 public:
};

template <class K, class V>
void insertAndFind(indexes::ART<K, V> &art, K n) {
  for (K i = 0; i < n; i++) {
    art.insert(i, i);
    for (auto j = i; j > 0; j--) {
      EXPECT_EQ(art.find(j), j);
    }
  }
}

TEST(ARTIndexTest, UINT8_InsertAndFind) {
  indexes::ART<uint8_t, uint8_t> art("UINT8_InsertAndFind");
  insertAndFind<uint8_t, uint8_t>(art, UINT8_MAX);
}

TEST(ARTIndexTest, UINT16_InsertAndFind) {
  indexes::ART<uint16_t, uint16_t> art("UINT16_InsertAndFind");
  insertAndFind<uint16_t, uint16_t>(art, UINT16_MAX);
}
TEST(ARTIndexTest, UINT32_InsertAndFind) {
  indexes::ART<uint32_t, uint32_t> art("UINT32_InsertAndFind");
  insertAndFind<uint32_t, uint32_t>(art, UINT32_MAX);
}

TEST(ARTIndexTest, UINT64_InsertandFind) {
  const uint64_t n = 100_M;
  indexes::ART<uint64_t, uint64_t> art("UINT64_InsertAndFind");
  insertAndFind<uint64_t, uint64_t>(art, n);
}
