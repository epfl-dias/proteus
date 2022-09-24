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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include "gtest/gtest.h"

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>

// #include "oltp/index/ART/OptimisticLockCoupling/art.hpp"
#include "oltp/index/ART/art.hpp"
#include "platform/util/timing.hpp"
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

TEST(ARTIndexTest, UINT8_InsertandFind) {
  const uint8_t n = UINT8_MAX;
  art_index::ART<uint8_t, uint8_t> art(false);

  for (uint8_t i = 0; i < n; i++) {
    art.insert(i, i);
    for (uint8_t j = i; j > 0; j--) {
      EXPECT_EQ(art.find(j), j);
    }
  }
}

TEST(ARTIndexTest, UINT16_InsertandFind) {
  const uint16_t n = UINT16_MAX;

  art_index::ART<uint16_t, uint16_t> art(false);

  for (uint16_t i = 0; i < n; i++) {
    art.insert(i, i);
    for (uint16_t j = i; j > 0; j--) {
      EXPECT_EQ(art.find(j), j);
    }
  }
}
TEST(ARTIndexTest, UINT32_InsertandFind) {
  const uint32_t n = UINT32_MAX;
  art_index::ART<uint32_t, uint32_t> art(false);

  for (uint32_t i = 0; i < n; i++) {
    art.insert(i, i);
    EXPECT_EQ(art.find(0), 0);
  }
  for (uint64_t i = 0; i > 0; i++) {
    EXPECT_EQ(art.find(i), i);
  }
}

TEST(ARTIndexTest, UINT64_InsertandFind) {
  const uint64_t n = 100_M;
  art_index::ART<uint64_t, uint64_t> art(false);

  for (uint64_t i = 0; i < n; i++) {
    art.insert(i, i);
  }
  for (uint64_t i = 0; i > 0; i++) {
    EXPECT_EQ(art.find(i), i);
  }
}
