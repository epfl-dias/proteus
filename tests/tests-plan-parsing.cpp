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

#include "util/raw-memory-manager.hpp"
#include "storage/raw-storage-manager.hpp"

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

class RawTestEnvironment : public ::testing::Environment {
public:
	virtual void SetUp();
	virtual void TearDown();
};

::testing::Environment *const pools_env = ::testing::AddGlobalTestEnvironment(new RawTestEnvironment);

void RawTestEnvironment::SetUp(){
	google::InstallFailureSignalHandler();

	RawPipelineGen::init();
	RawMemoryManager::init();
}

void RawTestEnvironment::TearDown(){
	RawMemoryManager::destroy();
}

class PlanTest : public ::testing::Test {
protected:
	virtual void SetUp() {
		catalog = &RawCatalog::getInstance();
		caches = &CachingService::getInstance();
		catalog->clear();
		caches->clear();
	}

	virtual void TearDown() {}

	bool executePlan(const char * planPath, const char * testLabel) {
		std::vector<RawPipeline *> pipelines;
		{
			time_block t("Tcodegen: ");

			GpuRawContext * ctx   = new GpuRawContext(testLabel, false);
			CatalogParser catalog = CatalogParser(catalogJSON, ctx);
			PlanExecutor  exec    = PlanExecutor(planPath, catalog, testLabel, ctx);

			ctx->compileAndLoad();

			pipelines = ctx->getPipelines();
		}

		{
			time_block t("Texecute       : ");

			for (RawPipeline * p: pipelines) {
				{
					time_block t("T: ");

					p->open();
					p->consume(0);
					p->close();
				}
			}
		}

		bool res = verifyTestResult(testPath, testLabel);
		shm_unlink(testLabel);
		return res;
	}

	bool flushResults = true;
	const char * testPath = TEST_OUTPUTS "/tests-plan-parsing/";
	const char * catalogJSON = "inputs/plans/catalog.json";

private:
	RawCatalog * catalog;
	CachingService * caches;
};

/* SELECT COUNT(*) FROM SAILORS s; */
TEST_F(PlanTest, Scan) {
	const char* planPath = "inputs/plans/reduce-scan.json";
	const char *testLabel = "reduce-scan-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/* SELECT COUNT(*), MAX(age) FROM SAILORS s; */
TEST_F(PlanTest, ScanTwoFields) {
	const char* planPath = "inputs/plans/reduce-twofields-scan.json";
	const char *testLabel = "reduce-twofields-scan-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

TEST_F(PlanTest, Unnest) {
	const char* planPath = "inputs/plans/reduce-unnest-scan.json";
	const char *testLabel = "reduce-unnest-scan-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/* SELECT COUNT(*) FROM SAILORS s JOIN RESERVES r ON s.sid = r.sid; */
TEST_F(PlanTest, Join) {
	const char* planPath = "inputs/plans/reduce-join.json";
	const char *testLabel = "reduce-join-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/* SELECT COUNT(*) FROM RESERVES GROUP BY sid; */
TEST_F(PlanTest, Nest) {
	const char* planPath = "inputs/plans/reduce-nest.json";
	const char *testLabel = "reduce-nest-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/* SELECT COUNT(*) FROM RESERVES WHERE sid = 22; */
TEST_F(PlanTest, Select) {
	const char* planPath = "inputs/plans/reduce-select.json";
	const char *testLabel = "reduce-select-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/* Project out multiple cols:
 * SELECT COUNT(*), MAX(bid) FROM RESERVES GROUP BY sid; */
TEST_F(PlanTest, MultiNest) {
	const char* planPath = "inputs/plans/reduce-multinest.json";
	const char *testLabel = "reduce-multinest-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 * */
TEST_F(PlanTest, JoinRecord) {
	const char* planPath = "inputs/plans/reduce-join-record.json";
	const char *testLabel = "reduce-join-record-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}

/*
 * select A3,B3
 * From A, B
 * where A.A1 = B.B1 and A.A2 > 10 and B.B2 < 10;
 *
 * [more results]
 * */
TEST_F(PlanTest, JoinRecordBNonselective) {
	//LSC: FIXME: Why this one alone uses other files? Can't we use the same
	//            inputs, or add a specific file for that test there instead
	//            of using a different catalog, and different test data?
	const char* catalogJSON = "inputs/parser/catalog.json";
	const char* planPath = "inputs/plans/reduce-join-record-nonselective.json";
	const char *testLabel = "reduce-join-record-nonselective-log.json";

	CatalogParser catalog = CatalogParser(catalogJSON);
	PlanExecutor exec1 = PlanExecutor (planPath, catalog, testLabel);

	EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(PlanTest, ScanBin) {
	const char* planPath = "inputs/plans/reduce-scan-bin.json";
	const char *testLabel = "reduce-scan-bin-log.json";

	EXPECT_TRUE(executePlan(planPath, testLabel));
}
