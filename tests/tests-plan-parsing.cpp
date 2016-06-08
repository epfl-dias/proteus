/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2014
		Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "plan/plan-parser.hpp"
#include "common/common.hpp"

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

/* SELECT COUNT(*) FROM SAILORS s; */
TEST(Plan, Scan) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-scan.json";
	const char *testLabel = "reduce-scan-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/* SELECT COUNT(*), MAX(age) FROM SAILORS s; */
TEST(Plan, ScanTwoFields) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-twofields-scan.json";
	const char *testLabel = "reduce-twofields-scan-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

TEST(Plan, Unnest) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-unnest-scan.json";
	const char *testLabel = "reduce-unnest-scan-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/* SELECT COUNT(*) FROM SAILORS s JOIN RESERVES r ON s.sid = r.sid; */
TEST(Plan, Join) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-join.json";
	const char *testLabel = "reduce-join-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/* SELECT COUNT(*) FROM RESERVES GROUP BY sid; */
TEST(Plan, Nest) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-nest.json";
	const char *testLabel = "reduce-nest-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/* SELECT COUNT(*) FROM RESERVES WHERE sid = 22; */
TEST(Plan, Select) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-select.json";
	const char *testLabel = "reduce-select-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/* Project out multiple cols:
 * SELECT COUNT(*), MAX(bid) FROM RESERVES GROUP BY sid; */
TEST(Plan, MultiNest) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-multinest.json";
	const char *testLabel = "reduce-multinest-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 * */
TEST(Plan, JoinRecord) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/plans/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-join-record.json";
	const char *testLabel = "reduce-join-record-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 *
 * [more results]
 * */
TEST(Plan, JoinRecordBNonselective) {
	CachingService& caches = CachingService::getInstance();
	caches.clear();
	const char* catalogJSON = "inputs/parser/catalog.json";
	const char *testPath = "testResults/tests-plan-parsing/";
	//Test-specific
	const char* planPath = "inputs/plans/reduce-join-record-nonselective.json";
	const char *testLabel = "reduce-join-record-nonselective-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor(planPath,catalog,testLabel);

	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
}
