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

#include "common/common.hpp"
#include "util/raw-context.hpp"
#include "util/raw-functions.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/radix-join.hpp"
#include "operators/reduce-opt.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"

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
//
//void verifyResult(const char *testLabel)	{
//	/* Compare with template answer */
//	/* correct */
//	struct stat statbuf;
//	string correctResult = string(TEST_OUTPUTS "/tests-sailors/") + testLabel;
//	stat(correctResult.c_str(), &statbuf);
//	size_t fsize1 = statbuf.st_size;
//	int fd1 = open(correctResult.c_str(), O_RDONLY);
//	if (fd1 == -1) {
//		throw runtime_error(string("csv.open: ")+correctResult);
//	}
//	char *correctBuf = (char*) mmap(NULL, fsize1, PROT_READ | PROT_WRITE,
//			MAP_PRIVATE, fd1, 0);
//
//	/* current */
//	stat(testLabel, &statbuf);
//	size_t fsize2 = statbuf.st_size;
//	int fd2 = open(testLabel, O_RDONLY);
//	if (fd2 == -1) {
//		throw runtime_error(string("csv.open: ")+testLabel);
//	}
//	char *currResultBuf = (char*) mmap(NULL, fsize2, PROT_READ | PROT_WRITE,
//			MAP_PRIVATE, fd2, 0);
//	cout << correctBuf << endl;
//	cout << currResultBuf << endl;
//	bool areEqual = (strcmp(correctBuf, currResultBuf) == 0) ? true : false;
//	EXPECT_TRUE(areEqual);
//
//	close(fd1);
//	munmap(correctBuf, fsize1);
//	close(fd2);
//	munmap(currResultBuf, fsize2);
//	if (remove(testLabel) != 0) {
//		throw runtime_error(string("Error deleting file"));
//	}
//}

//TEST(Sailors, Scan) {
//	const char *testPath = TEST_OUTPUTS "/tests-sailors/";
//	const char *testLabel = "sailorsScan.json";
//	bool flushResults = true;
//
//	RawContext& ctx = *prepareContext(testLabel);
//	RawCatalog& catalog = RawCatalog::getInstance();
//	CachingService& caches = CachingService::getInstance();
//	caches.clear();
//
//	/**
//	 * SCAN1
//	 */
//	string sailorsPath = string("inputs/sailors.csv");
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//	RecordAttribute* sid = new RecordAttribute(1, sailorsPath, string("sid"),
//			intType);
//	RecordAttribute* sname = new RecordAttribute(2, sailorsPath,
//			string("sname"), stringType);
//	RecordAttribute* rating = new RecordAttribute(3, sailorsPath,
//			string("rating"), intType);
//	RecordAttribute* age = new RecordAttribute(4, sailorsPath, string("age"),
//			floatType);
//
//	list<RecordAttribute*> sailorAtts;
//	sailorAtts.push_back(sid);
//	sailorAtts.push_back(sname);
//	sailorAtts.push_back(rating);
//	sailorAtts.push_back(age);
//	RecordType sailorRec = RecordType(sailorAtts);
//
//	vector<RecordAttribute*> sailorAttsToProject;
//	sailorAttsToProject.push_back(sid);
//	sailorAttsToProject.push_back(age); //Float
//
//	int linehint = 10;
//	int policy = 2;
//	pm::CSVPlugin* pgSailors = new pm::CSVPlugin(&ctx, sailorsPath, sailorRec,
//			sailorAttsToProject, ';', linehint, policy, false);
//	catalog.registerPlugin(sailorsPath, pgSailors);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//	/**
//	 * REDUCE
//	 */
//	list<RecordAttribute> projections = list<RecordAttribute>();
//	projections.push_back(*sid);
//	projections.push_back(*age);
//
//	expressions::Expression *arg = new expressions::InputArgument(&sailorRec, 0,
//			projections);
//	expressions::Expression *outputExpr = new expressions::RecordProjection(
//			intType, arg, *sid);
//	expressions::Expression *one = new expressions::IntConstant(1);
//
//	expressions::Expression *predicate = new expressions::BoolConstant(true);
//
//	vector<Monoid> accs;
//	vector<expressions::Expression*> exprs;
//	accs.push_back(MAX);
//	exprs.push_back(outputExpr);
//	/* Sanity checks*/
//	accs.push_back(SUM);
//	exprs.push_back(outputExpr);
//	accs.push_back(SUM);
//	exprs.push_back(one);
//	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &scanSailors, &ctx,
//			flushResults, testLabel);
//	scanSailors.setParent(&reduce);
//	reduce.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgSailors->finish();
//
//	catalog.clear();
//
//	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
//}

TEST(Sailors, Select) {
	const char *testPath = TEST_OUTPUTS "/tests-sailors/";
	const char *testLabel = "sailorsSel.json";
	bool flushResults = true;
	RawContext& ctx = *prepareContext(testLabel);
	RawCatalog& catalog = RawCatalog::getInstance();
	CachingService& caches = CachingService::getInstance();
	caches.clear();

	/**
	 * SCAN1
	 */
	string sailorsPath = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, sailorsPath, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, sailorsPath,
			string("sname"), stringType);
	RecordAttribute* rating = new RecordAttribute(3, sailorsPath,
			string("rating"), intType);
	RecordAttribute* age = new RecordAttribute(4, sailorsPath, string("age"),
			floatType);

	list<RecordAttribute*> sailorAtts;
	sailorAtts.push_back(sid);
	sailorAtts.push_back(sname);
	sailorAtts.push_back(rating);
	sailorAtts.push_back(age);
	RecordType sailorRec = RecordType(sailorAtts);

	vector<RecordAttribute*> sailorAttsToProject;
	sailorAttsToProject.push_back(sid);
	sailorAttsToProject.push_back(age); //Float

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pgSailors = new pm::CSVPlugin(&ctx, sailorsPath, sailorRec,
			sailorAttsToProject, ';', linehint, policy, false);
	catalog.registerPlugin(sailorsPath, pgSailors);
	Scan scanSailors = Scan(&ctx, *pgSailors);

	/**
	 * SELECT
	 */
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression *arg = new expressions::InputArgument(&sailorRec, 0,
			projections);

	expressions::Expression* lhs = new expressions::RecordProjection(
			new FloatType(), arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);
	Select sel = Select(predicate, &scanSailors);
	scanSailors.setParent(&sel);

	/**
	 * REDUCE
	 */
	expressions::Expression *outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);
	expressions::Expression *one = new expressions::IntConstant(1);

	expressions::Expression *predicateRed = new expressions::BoolConstant(true);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(MAX);
	exprs.push_back(outputExpr);
	/* Sanity checks*/
	accs.push_back(SUM);
	exprs.push_back(outputExpr);
	accs.push_back(SUM);
	exprs.push_back(one);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicateRed, &sel, &ctx,
			flushResults, testLabel);
	sel.setParent(&reduce);
	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgSailors->finish();

	catalog.clear();

	EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}


//TEST(Sailors, ScanBoats) {
//	const char *testPath = TEST_OUTPUTS "/tests-sailors/";
//	const char *testLabel = "sailorsScanBoats.json";
//	bool flushResults = true;
//
//	RawContext& ctx = *prepareContext(testLabel);
//	RawCatalog& catalog = RawCatalog::getInstance();
//	CachingService& caches = CachingService::getInstance();
//	caches.clear();
//
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//
//	string filenameBoats = string("inputs/boats.csv");
//	RecordAttribute* bidBoats = new RecordAttribute(1, filenameBoats,
//			string("bid"), intType);
//	RecordAttribute* bnameBoats = new RecordAttribute(2, filenameBoats,
//			string("bname"), stringType);
//	RecordAttribute* colorBoats = new RecordAttribute(3, filenameBoats,
//			string("color"), stringType);
//
//	list<RecordAttribute*> attrListBoats;
//	attrListBoats.push_back(bidBoats);
//	attrListBoats.push_back(bnameBoats);
//	attrListBoats.push_back(colorBoats);
//	RecordType recBoats = RecordType(attrListBoats);
//
//	vector<RecordAttribute*> whichFieldsBoats;
//	whichFieldsBoats.push_back(bidBoats);
//
//	int linehint = 4;
//	int policy = 2;
//	pm::CSVPlugin* pgBoats = new pm::CSVPlugin(&ctx, filenameBoats, recBoats,
//			whichFieldsBoats, ';', linehint, policy, false);
//	catalog.registerPlugin(filenameBoats, pgBoats);
//	Scan scanBoats = Scan(&ctx, *pgBoats);
//
//	/**
//	 * REDUCE
//	 */
//	list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
//	fieldsBoats.push_back(*bidBoats);
//	expressions::Expression *boatsArg = new expressions::InputArgument(intType,
//			1, fieldsBoats);
//
//	expressions::Expression* outputExpr = new expressions::RecordProjection(
//			intType, boatsArg, *bidBoats);
//	expressions::Expression *one = new expressions::IntConstant(1);
//
//	expressions::Expression *predicateRed = new expressions::BoolConstant(true);
//
//	vector<Monoid> accs;
//	vector<expressions::Expression*> exprs;
//	accs.push_back(MAX);
//	exprs.push_back(outputExpr);
//	/* Sanity checks*/
//	accs.push_back(SUM);
//	exprs.push_back(outputExpr);
//	accs.push_back(SUM);
//	exprs.push_back(one);
//	opt::Reduce reduce = opt::Reduce(accs, exprs, predicateRed, &scanBoats,
//			&ctx, flushResults, testLabel);
//	scanBoats.setParent(&reduce);
//	reduce.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgBoats->finish();
//	catalog.clear();
//	EXPECT_TRUE(true);
//}

//TEST(Sailors, JoinLeft3) {
//	const char *testPath = TEST_OUTPUTS "/tests-sailors/";
//	const char *testLabel = "sailorsJoinLeft3.json";
//	bool flushResults = true;
//
//
//	RawContext& ctx = *prepareContext(testLabel);
//	RawCatalog& catalog = RawCatalog::getInstance();
//	CachingService& caches = CachingService::getInstance();
//	caches.clear();
//
//	/**
//	 * SCAN1
//	 */
//	string sailorsPath = string("inputs/sailors.csv");
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//	RecordAttribute* sid = new RecordAttribute(1,sailorsPath,string("sid"),intType);
//	RecordAttribute* sname = new RecordAttribute(2,sailorsPath,string("sname"),stringType);
//	RecordAttribute* rating = new RecordAttribute(3,sailorsPath,string("rating"),intType);
//	RecordAttribute* age = new RecordAttribute(4,sailorsPath,string("age"),floatType);
//
//	list<RecordAttribute*> sailorAtts;
//	sailorAtts.push_back(sid);
//	sailorAtts.push_back(sname);
//	sailorAtts.push_back(rating);
//	sailorAtts.push_back(age);
//	RecordType sailorRec = RecordType(sailorAtts);
//
//	vector<RecordAttribute*> sailorAttsToProject;
//	sailorAttsToProject.push_back(sid);
//	sailorAttsToProject.push_back(age); //Float
//
//	int linehint = 10;
//	int policy = 2;
//	pm::CSVPlugin* pgSailors =
//			new pm::CSVPlugin(&ctx, sailorsPath, sailorRec, sailorAttsToProject, ';', linehint, policy, false);
//	catalog.registerPlugin(sailorsPath,pgSailors);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//	/**
//	 * SCAN2
//	 */
//	string reservesPath = string("inputs/reserves.csv");
//	RecordAttribute* sidReserves = new RecordAttribute(1,reservesPath,string("sid"),intType);
//	RecordAttribute* bidReserves = new RecordAttribute(2,reservesPath,string("bid"),intType);
//	RecordAttribute* day = new RecordAttribute(3,reservesPath,string("day"),stringType);
//
//	list<RecordAttribute*> reserveAtts;
//	reserveAtts.push_back(sidReserves);
//	reserveAtts.push_back(bidReserves);
//	reserveAtts.push_back(day);
//	RecordType reserveRec = RecordType(reserveAtts);
//	vector<RecordAttribute*> reserveAttsToProject;
//	reserveAttsToProject.push_back(sidReserves);
//	reserveAttsToProject.push_back(bidReserves);
//
//	linehint = 10;
//	policy = 2;
//	pm::CSVPlugin* pgReserves =
//			new pm::CSVPlugin(&ctx, reservesPath, reserveRec, reserveAttsToProject, ';', linehint, policy, false);
//	catalog.registerPlugin(reservesPath,pgReserves);
//	Scan scanReserves = Scan(&ctx, *pgReserves);
//
//	/**
//	 * JOIN
//	 */
//	/* Sailors: Left-side fields for materialization etc. */
//	RecordAttribute sailorOID = RecordAttribute(sailorsPath, activeLoop,
//			pgSailors->getOIDType());
//	list<RecordAttribute> sailorAttsForArg = list<RecordAttribute>();
//	sailorAttsForArg.push_back(sailorOID);
//	sailorAttsForArg.push_back(*sid);
//	sailorAttsForArg.push_back(*age);
//	expressions::Expression *sailorArg = new expressions::InputArgument(intType,
//			0, sailorAttsForArg);
//	expressions::Expression *sailorOIDProj = new expressions::RecordProjection(
//			intType, sailorArg, sailorOID);
//	expressions::Expression*sailorSIDProj = new expressions::RecordProjection(
//			intType, sailorArg, *sid);
//	expressions::Expression *sailorAgeProj = new expressions::RecordProjection(
//			floatType, sailorArg, *age);
//	vector<expressions::Expression*> exprsToMatSailor;
//	exprsToMatSailor.push_back(sailorOIDProj);
//	exprsToMatSailor.push_back(sailorSIDProj);
//	exprsToMatSailor.push_back(sailorAgeProj);
//	Materializer* matSailor = new Materializer(exprsToMatSailor);
//
//	/* Reserves: Right-side fields for materialization etc. */
//	RecordAttribute reservesOID = RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
//	list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
//	reserveAttsForArg.push_back(reservesOID);
//	reserveAttsForArg.push_back(*sidReserves);
//	reserveAttsForArg.push_back(*bidReserves);
//	expressions::Expression *reservesArg = new expressions::InputArgument(intType,
//			1, reserveAttsForArg);
//	expressions::Expression *reservesOIDProj = new expressions::RecordProjection(
//			pgReserves->getOIDType(), reservesArg, reservesOID);
//	expressions::Expression* reservesSIDProj = new expressions::RecordProjection(
//			intType, reservesArg, *sidReserves);
//	expressions::Expression* reservesBIDProj = new expressions::RecordProjection(
//				intType, reservesArg, *bidReserves);
//	vector<expressions::Expression*> exprsToMatReserves;
//	exprsToMatReserves.push_back(reservesOIDProj);
//	//exprsToMatRight.push_back(resevesSIDProj);
//	exprsToMatReserves.push_back(reservesBIDProj);
//
//	Materializer* matReserves = new Materializer(exprsToMatReserves);
//
//	expressions::BinaryExpression* joinPred =
//			new expressions::EqExpression(new BoolType(),sailorSIDProj,reservesSIDProj);
//
//	char joinLabel[] = "sailors_reserves";
//	RadixJoin join = RadixJoin(joinPred, &scanSailors, &scanReserves, &ctx, joinLabel, *matSailor, *matReserves);
//	scanSailors.setParent(&join);
//	scanReserves.setParent(&join);
//
//
//	//SCAN3: BOATS
//	string filenameBoats = string("inputs/boats.csv");
//	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats,string("bid"),intType);
//	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats,string("bname"),stringType);
//	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats,string("color"),stringType);
//
//	list<RecordAttribute*> attrListBoats;
//	attrListBoats.push_back(bidBoats);
//	attrListBoats.push_back(bnameBoats);
//	attrListBoats.push_back(colorBoats);
//	RecordType recBoats = RecordType(attrListBoats);
//
//	vector<RecordAttribute*> whichFieldsBoats;
//	whichFieldsBoats.push_back(bidBoats);
//
//	linehint = 4;
//	policy = 2;
//	pm::CSVPlugin* pgBoats = new pm::CSVPlugin(&ctx, filenameBoats, recBoats,
//			whichFieldsBoats, ';', linehint, policy, false);
//	catalog.registerPlugin(filenameBoats,pgBoats);
//	Scan scanBoats = Scan(&ctx, *pgBoats);
//
//	/**
//	 * JOIN2: BOATS
//	 */
//	expressions::Expression *previousJoinArg =
//			new expressions::InputArgument(intType,0,reserveAttsForArg);
//	expressions::Expression *previousJoinBIDProj =
//			new expressions::RecordProjection(intType,previousJoinArg,*bidReserves);
//	vector<expressions::Expression*> exprsToMatPreviousJoin;
//	exprsToMatPreviousJoin.push_back(sailorOIDProj);
//	exprsToMatPreviousJoin.push_back(reservesOIDProj);
//	exprsToMatPreviousJoin.push_back(sailorSIDProj);
//	Materializer* matPreviousJoin = new Materializer(exprsToMatPreviousJoin);
//
//	RecordAttribute projTupleBoat = RecordAttribute(filenameBoats, activeLoop, pgBoats->getOIDType());
//	list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
//	fieldsBoats.push_back(projTupleBoat);
//	fieldsBoats.push_back(*bidBoats);
//	expressions::Expression* boatsArg =
//			new expressions::InputArgument(intType,1,fieldsBoats);
//	expressions::Expression* boatsOIDProj =
//			new expressions::RecordProjection(pgBoats->getOIDType(),boatsArg,projTupleBoat);
//	expressions::Expression* boatsBIDProj =
//			new expressions::RecordProjection(intType,boatsArg,*bidBoats);
//
//	vector<expressions::Expression*> exprsToMatBoats;
//	exprsToMatBoats.push_back(boatsOIDProj);
//	exprsToMatBoats.push_back(boatsBIDProj);
//	Materializer* matBoats = new Materializer(exprsToMatBoats);
//
//	expressions::BinaryExpression* joinPred2 =
//			new expressions::EqExpression(new BoolType(),previousJoinBIDProj,boatsBIDProj);
//
//	char joinLabel2[] = "sailors_reserves_boats";
//	RadixJoin join2 = RadixJoin(joinPred2, &join, &scanBoats, &ctx, joinLabel2, *matPreviousJoin, *matBoats);
//	join.setParent(&join2);
//	scanBoats.setParent(&join2);
//
//	/**
//	 * REDUCE
//	 */
//	list<RecordAttribute> projections = list<RecordAttribute>();
//	projections.push_back(*sid);
//	projections.push_back(*age);
//
//	expressions::Expression *arg =
//			new expressions::InputArgument(&sailorRec, 0,projections);
//	expressions::Expression *outputExpr =
//			new expressions::RecordProjection(intType, arg, *sid);
//	expressions::Expression *one = new expressions::IntConstant(1);
//
//	expressions::Expression *predicate = new expressions::BoolConstant(true);
//
//	vector<Monoid> accs;
//	vector<expressions::Expression*> exprs;
//	accs.push_back(MAX);
//	exprs.push_back(outputExpr);
//	/* Sanity checks*/
//	accs.push_back(SUM);
//	exprs.push_back(outputExpr);
//	accs.push_back(SUM);
//	exprs.push_back(one);
//	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &join2, &ctx,
//			flushResults, testLabel);
//	join2.setParent(&reduce);
//	reduce.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgSailors->finish();
//	pgReserves->finish();
//	pgBoats->finish();
//	catalog.clear();
//
//	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
//}

//TEST(Sailors, JoinRight3) {
//	const char *testPath = TEST_OUTPUTS "/tests-sailors/";
//	const char *testLabel = "sailorsJoinRight3.json";
//	bool flushResults = true;
//	RawContext& ctx = *prepareContext(testLabel);
//	RawCatalog& catalog = RawCatalog::getInstance();
//	CachingService& caches = CachingService::getInstance();
//	caches.clear();
//
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//
//	/**
//	 * SCAN2
//	 */
//	string reservesPath = string("inputs/reserves.csv");
//	RecordAttribute* sidReserves = new RecordAttribute(1, reservesPath,
//			string("sid"), intType);
//	RecordAttribute* bidReserves = new RecordAttribute(2, reservesPath,
//			string("bid"), intType);
//	RecordAttribute* day = new RecordAttribute(3, reservesPath, string("day"),
//			stringType);
//
//	list<RecordAttribute*> reserveAtts;
//	reserveAtts.push_back(sidReserves);
//	reserveAtts.push_back(bidReserves);
//	reserveAtts.push_back(day);
//	RecordType reserveRec = RecordType(reserveAtts);
//	vector<RecordAttribute*> reserveAttsToProject;
//	reserveAttsToProject.push_back(sidReserves);
//	reserveAttsToProject.push_back(bidReserves);
//
//	int linehint = 10;
//	int policy = 2;
//	pm::CSVPlugin* pgReserves = new pm::CSVPlugin(&ctx, reservesPath,
//			reserveRec, reserveAttsToProject, ';', linehint, policy, false);
//	catalog.registerPlugin(reservesPath, pgReserves);
//	Scan scanReserves = Scan(&ctx, *pgReserves);
//
//	//SCAN3: BOATS
//	string filenameBoats = string("inputs/boats.csv");
//	RecordAttribute* bidBoats = new RecordAttribute(1, filenameBoats,
//			string("bid"), intType);
//	RecordAttribute* bnameBoats = new RecordAttribute(2, filenameBoats,
//			string("bname"), stringType);
//	RecordAttribute* colorBoats = new RecordAttribute(3, filenameBoats,
//			string("color"), stringType);
//
//	list<RecordAttribute*> attrListBoats;
//	attrListBoats.push_back(bidBoats);
//	attrListBoats.push_back(bnameBoats);
//	attrListBoats.push_back(colorBoats);
//	RecordType recBoats = RecordType(attrListBoats);
//
//	vector<RecordAttribute*> whichFieldsBoats;
//	whichFieldsBoats.push_back(bidBoats);
//
//	linehint = 4;
//	policy = 2;
//	pm::CSVPlugin* pgBoats = new pm::CSVPlugin(&ctx, filenameBoats, recBoats,
//			whichFieldsBoats, ';', linehint, policy, false);
//	catalog.registerPlugin(filenameBoats, pgBoats);
//	Scan scanBoats = Scan(&ctx, *pgBoats);
//
//	/**
//	 * JOIN: Reserves JOIN Boats
//	 */
//	/* Reserves: fields for materialization etc. */
//	RecordAttribute reservesOID = RecordAttribute(reservesPath, activeLoop,
//			pgReserves->getOIDType());
//	list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
//	reserveAttsForArg.push_back(reservesOID);
//	reserveAttsForArg.push_back(*sidReserves);
//	reserveAttsForArg.push_back(*bidReserves);
//	expressions::Expression *reservesArg = new expressions::InputArgument(
//			intType, 1, reserveAttsForArg);
//	expressions::Expression *reservesOIDProj =
//			new expressions::RecordProjection(pgReserves->getOIDType(),
//					reservesArg, reservesOID);
//	expressions::Expression* reservesSIDProj =
//			new expressions::RecordProjection(intType, reservesArg,
//					*sidReserves);
//	expressions::Expression* reservesBIDProj =
//			new expressions::RecordProjection(intType, reservesArg,
//					*bidReserves);
//	vector<expressions::Expression*> exprsToMatReserves;
//	exprsToMatReserves.push_back(reservesOIDProj);
//	exprsToMatReserves.push_back(reservesSIDProj);
//	exprsToMatReserves.push_back(reservesBIDProj);
//
//	Materializer* matReserves = new Materializer(exprsToMatReserves);
//
//	RecordAttribute projTupleBoat = RecordAttribute(filenameBoats, activeLoop,
//			pgBoats->getOIDType());
//	list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
//	fieldsBoats.push_back(projTupleBoat);
//	fieldsBoats.push_back(*bidBoats);
//	expressions::Expression* boatsArg = new expressions::InputArgument(intType,
//			1, fieldsBoats);
//	expressions::Expression* boatsOIDProj = new expressions::RecordProjection(
//			pgBoats->getOIDType(), boatsArg, projTupleBoat);
//	expressions::Expression* boatsBIDProj = new expressions::RecordProjection(
//			intType, boatsArg, *bidBoats);
//	vector<expressions::Expression*> exprsToMatBoats;
//	exprsToMatBoats.push_back(boatsOIDProj);
//	exprsToMatBoats.push_back(boatsBIDProj);
//	Materializer* matBoats = new Materializer(exprsToMatBoats);
//
//	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
//			new BoolType(), reservesBIDProj, boatsBIDProj);
//
//	char joinLabel2[] = "reserves_boats";
//	RadixJoin join2 = RadixJoin(joinPred2, &scanReserves, &scanBoats, &ctx,
//			joinLabel2, *matReserves, *matBoats);
//	scanReserves.setParent(&join2);
//			scanBoats.setParent(&join2);
//
//	/**
//	 * SCAN1
//	 */
//	string sailorsPath = string("inputs/sailors.csv");
//
//	RecordAttribute* sid = new RecordAttribute(1, sailorsPath, string("sid"),
//			intType);
//	RecordAttribute* sname = new RecordAttribute(2, sailorsPath,
//			string("sname"), stringType);
//	RecordAttribute* rating = new RecordAttribute(3, sailorsPath,
//			string("rating"), intType);
//	RecordAttribute* age = new RecordAttribute(4, sailorsPath, string("age"),
//			floatType);
//
//	list<RecordAttribute*> sailorAtts;
//	sailorAtts.push_back(sid);
//	sailorAtts.push_back(sname);
//	sailorAtts.push_back(rating);
//	sailorAtts.push_back(age);
//	RecordType sailorRec = RecordType(sailorAtts);
//
//	vector<RecordAttribute*> sailorAttsToProject;
//	sailorAttsToProject.push_back(sid);
//	sailorAttsToProject.push_back(age); //Float
//
//	linehint = 10;
//	policy = 2;
//	pm::CSVPlugin* pgSailors = new pm::CSVPlugin(&ctx, sailorsPath, sailorRec,
//			sailorAttsToProject, ';', linehint, policy, false);
//	catalog.registerPlugin(sailorsPath, pgSailors);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//	/**
//	 * JOIN: Sailors JOIN (Reserves JOIN Boats)
//	 */
//
//	/* Sailors: fields for materialization etc. */
//	RecordAttribute sailorOID = RecordAttribute(sailorsPath, activeLoop,
//			pgSailors->getOIDType());
//	list<RecordAttribute> sailorAttsForArg = list<RecordAttribute>();
//	sailorAttsForArg.push_back(sailorOID);
//	sailorAttsForArg.push_back(*sid);
//	sailorAttsForArg.push_back(*age);
//	expressions::Expression *sailorArg = new expressions::InputArgument(intType,
//			0, sailorAttsForArg);
//	expressions::Expression *sailorOIDProj = new expressions::RecordProjection(
//			intType, sailorArg, sailorOID);
//	expressions::Expression*sailorSIDProj = new expressions::RecordProjection(
//			intType, sailorArg, *sid);
//	vector<expressions::Expression*> exprsToMatSailor;
//	exprsToMatSailor.push_back(sailorOIDProj);
//	exprsToMatSailor.push_back(sailorSIDProj);
//	Materializer* matSailor = new Materializer(exprsToMatSailor);
//
//	expressions::Expression *previousJoinArg = new expressions::InputArgument(
//			intType, 0, reserveAttsForArg);
//
//	vector<expressions::Expression*> exprsToMatPreviousJoin;
//	exprsToMatPreviousJoin.push_back(reservesOIDProj);
//	exprsToMatPreviousJoin.push_back(reservesSIDProj);
//	Materializer* matPreviousJoin = new Materializer(exprsToMatPreviousJoin);
//
//	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
//			new BoolType(), sailorSIDProj, reservesSIDProj);
//
//	char joinLabel[] = "sailors_(reserves_boats)";
//	RadixJoin join = RadixJoin(joinPred, &scanSailors, &join2, &ctx, joinLabel,
//			*matSailor, *matPreviousJoin);
//	scanSailors.setParent(&join);
//	join2.setParent(&join);
//
//	/**
//	 * REDUCE
//	 */
//	list<RecordAttribute> projections = list<RecordAttribute>();
//	projections.push_back(*sid);
//
//	expressions::Expression *arg = new expressions::InputArgument(&sailorRec, 0,
//			projections);
//	expressions::Expression *outputExpr = new expressions::RecordProjection(
//			intType, arg, *sid);
//	expressions::Expression *one = new expressions::IntConstant(1);
//
//	expressions::Expression *predicate = new expressions::BoolConstant(true);
//
//	vector<Monoid> accs;
//	vector<expressions::Expression*> exprs;
//	accs.push_back(MAX);
//	exprs.push_back(outputExpr);
//	/* Sanity checks*/
//	accs.push_back(SUM);
//	exprs.push_back(outputExpr);
//	accs.push_back(SUM);
//	exprs.push_back(one);
//	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &join, &ctx,
//			flushResults, testLabel);
//	join.setParent(&reduce);
//	reduce.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgSailors->finish();
//	pgReserves->finish();
//	pgBoats->finish();
//	catalog.clear();
//
//	EXPECT_TRUE(verifyTestResult(testPath,testLabel));
//}
