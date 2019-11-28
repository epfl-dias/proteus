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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include "gtest/gtest.h"
#include "memory/memory-manager.hpp"
#include "storage/storage-manager.hpp"
#include "test-utils.hpp"

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

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class PlanTest : public ::testing::Test {
 protected:
  virtual void TearDown() { StorageManager::unloadAll(); }

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-plan-parsing/";
  const char *catalogJSON = "inputs";

  void runAndVerify(const char *testLabel, const char *planPath,
                    bool unordered = true) {
    ::runAndVerify(testLabel, planPath, testPath, catalogJSON, unordered);
  }
};

/* SELECT COUNT(*) FROM SAILORS s; */
TEST_F(PlanTest, Scan) {
  const char *planPath = "inputs/plans/reduce-scan.json";
  const char *testLabel = "reduce-scan-log.json";

  runAndVerify(testLabel, planPath);
}

/* SELECT COUNT(*) as cnt, MAX(age) as max_age FROM SAILORS s; */
TEST_F(PlanTest, ScanTwoFields) {
  const char *planPath = "inputs/plans/reduce-twofields-scan.json";
  const char *testLabel = "reduce-twofields-scan-log.json";

  runAndVerify(testLabel, planPath);
}

/* SELECT COUNT(*) as cnt FROM employees e, unnest(e.children); */
TEST_F(PlanTest, Unnest) {
  const char *planPath = "inputs/plans/reduce-unnest-scan.json";
  const char *testLabel = "reduce-unnest-scan-log.json";

  runAndVerify(testLabel, planPath);
}

/* SELECT COUNT(*) FROM SAILORS s JOIN RESERVES r ON s.sid = r.sid; */
TEST_F(PlanTest, Join) {
  const char *planPath = "inputs/plans/reduce-join.json";
  const char *testLabel = "reduce-join-log.json";

  runAndVerify(testLabel, planPath);
}

/* SELECT COUNT(*) as count FROM RESERVES GROUP BY sid; */
TEST_F(PlanTest, Nest) {
  const char *planPath = "inputs/plans/reduce-nest.json";
  const char *testLabel = "reduce-nest-log.json";

  runAndVerify(testLabel, planPath);
}

/* SELECT COUNT(*) FROM RESERVES WHERE sid = 22; */
TEST_F(PlanTest, Select) {
  const char *planPath = "inputs/plans/reduce-select.json";
  const char *testLabel = "reduce-select-log.json";

  runAndVerify(testLabel, planPath);
}

/* Project out multiple cols:
 * SELECT COUNT(*) as outputCnt, MAX(bid) as outputMax FROM RESERVES GROUP BY
 * sid; */
TEST_F(PlanTest, MultiNest) {
  const char *planPath = "inputs/plans/reduce-multinest.json";
  const char *testLabel = "reduce-multinest-log.json";

  runAndVerify(testLabel, planPath);
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 * */
TEST_F(PlanTest, JoinRecord) {
  const char *planPath = "inputs/plans/reduce-join-record.json";
  const char *testLabel = "reduce-join-record-log.json";

  runAndVerify(testLabel, planPath);
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 *
 * [more results]
 * */
TEST_F(PlanTest, JoinRecordBNonselective) {
  // LSC: FIXME: Why this one alone uses other files? Can't we use the same
  //            inputs, or add a specific file for that test there instead
  //            of using a different catalog, and different test data?
  const char *catalogJSON = "inputs/parser";
  const char *planPath = "inputs/plans/reduce-join-record-nonselective.json";
  const char *testLabel = "reduce-join-record-nonselective-log.json";

  ::runAndVerify(testLabel, planPath, testPath, catalogJSON, true);
}

TEST_F(PlanTest, ScanBin) {
  const char *planPath = "inputs/plans/reduce-scan-bin.json";
  const char *testLabel = "reduce-scan-bin-log.json";

  runAndVerify(testLabel, planPath);
}
