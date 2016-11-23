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
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "operators/join.hpp"
#include "operators/unnest.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/radix-join.hpp"
#include "operators/radix-nest.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "common/tpch-config.hpp"


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

void verifyResult_(const char *testLabel)	{
	/* Compare with template answer */
	/* correct */
	struct stat statbuf;
	string correctResult = string(TEST_OUTPUTS "/tests-output/") + testLabel;
	stat(correctResult.c_str(), &statbuf);
	size_t fsize1 = statbuf.st_size;
	int fd1 = open(correctResult.c_str(), O_RDONLY);
	if (fd1 == -1) {
		throw runtime_error(string("csv.open: ")+correctResult);
	}
	char *correctBuf = (char*) mmap(NULL, fsize1, PROT_READ | PROT_WRITE,
			MAP_PRIVATE, fd1, 0);

	/* current */
	stat(testLabel, &statbuf);
	size_t fsize2 = statbuf.st_size;
	int fd2 = open(testLabel, O_RDONLY);
	if (fd2 == -1) {
		throw runtime_error(string("csv.open: ")+testLabel);
	}
	char *currResultBuf = (char*) mmap(NULL, fsize2, PROT_READ | PROT_WRITE,
			MAP_PRIVATE, fd2, 0);
	cout << correctBuf << endl;
	cout << currResultBuf << endl;
	bool areEqual = (strcmp(correctBuf, currResultBuf) == 0) ? true : false;
	EXPECT_TRUE(areEqual);

	close(fd1);
	munmap(correctBuf, fsize1);
	close(fd2);
	munmap(currResultBuf, fsize2);
	if (remove(testLabel) != 0) {
		throw runtime_error(string("Error deleting file"));
	}
}


TEST(Output, ReduceNumeric) {
	const char *testLabel = "reduceNumeric.json";
	bool flushResults = true;
	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
			floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop, pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression* outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(MAX);
	exprs.push_back(outputExpr);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &scan, &ctx, flushResults, testLabel);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();

	verifyResult_(testLabel);
}

TEST(Output, MultiReduceNumeric) {
	const char *testLabel = "multiReduceNumeric.json";
	bool flushResults = true;
	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType *intType = new IntType();
	PrimitiveType *floatType = new FloatType();
	PrimitiveType *stringType = new StringType();
	RecordAttribute *sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute *sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute *rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute *age = new RecordAttribute(4, filename, string("age"),
			floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop, pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression* outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(SUM);
	accs.push_back(MAX);
	exprs.push_back(outputExpr);
	exprs.push_back(outputExpr);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &scan, &ctx, flushResults, testLabel);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();

	verifyResult_(testLabel);
}

TEST(Output, ReduceBag) {
	const char *testLabel = "reduceBag.json";
	bool flushResults = true;
	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
			floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop, pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression* outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(BAGUNION);
	exprs.push_back(outputExpr);
	//	Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
	//	Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &scan, &ctx, flushResults, testLabel);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();

	verifyResult_(testLabel);
}

TEST(Output, ReduceBagRecord) {
	const char *testLabel = "reduceBagRecord.json";
	bool flushResults = true;
	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
			floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop, pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression *arg = new expressions::InputArgument(&rec1, 0,
			projections);

	/* CONSTRUCT OUTPUT RECORD */
	list<RecordAttribute*> newAttsTypes = list<RecordAttribute*>();
	newAttsTypes.push_back(sid);
	newAttsTypes.push_back(age);
	RecordType newRecType = RecordType(newAttsTypes);
	expressions::RecordProjection *projID = new expressions::RecordProjection(
			intType, arg, *sid);
	expressions::RecordProjection *projAge = new expressions::RecordProjection(
			floatType, arg, *age);

	expressions::AttributeConstruction attrExpr1 =
			expressions::AttributeConstruction("id", projID);
	expressions::AttributeConstruction attrExpr2 =
			expressions::AttributeConstruction("age", projAge);
	list<expressions::AttributeConstruction> newAtts = list<
			expressions::AttributeConstruction>();
	newAtts.push_back(attrExpr1);
	newAtts.push_back(attrExpr2);
	expressions::RecordConstruction *outputExpr =
			new expressions::RecordConstruction(&newRecType, newAtts);


	/* Construct filtering predicate */
	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(BAGUNION);
	exprs.push_back(outputExpr);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &scan, &ctx, flushResults, testLabel);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();

	verifyResult_(testLabel);
}

TEST(Output, NestBagTPCH) {
	const char *testLabel = "nestBagTPCH.json";
	bool flushResults = true;
	/* Bookkeeping */
	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();
	PrimitiveType *intType = new IntType();
	PrimitiveType *floatType = new FloatType();

	/* File Info */
	map<string, dataset> datasetCatalog;
	tpchSchemaCSV(datasetCatalog);
	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	string lineitemPath = string("inputs/tpch/lineitem10.csv");
	int linehint = 10;
	int policy = 5;
	char delimInner = '|';
	RecordType rec = lineitem.recType;

	/* Projections */
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
	RecordAttribute *l_quantity = argsLineitem["l_quantity"];

	projections.push_back(l_orderkey);
	projections.push_back(l_linenumber);
	projections.push_back(l_quantity);

	pm::CSVPlugin* pg = new pm::CSVPlugin(&ctx, lineitemPath, rec, projections,
			delimInner, linehint, policy, false);
	catalog.registerPlugin(lineitemPath, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	argProjections.push_back(*l_quantity);

	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(4);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: l_linenumber
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;

	nestProjections.push_back(*l_quantity);

	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);
	//f (& g)
	expressions::RecordProjection* f = new expressions::RecordProjection(
			l_linenumber->getOriginalType(), nestArg, *l_linenumber);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
//	vector<RecordAttribute*> fields;
//	vector<materialization_mode> outputModes;
//	fields.push_back(l_linenumber);
//	outputModes.insert(outputModes.begin(), EAGER);
//	fields.push_back(l_quantity);
//	outputModes.insert(outputModes.begin(), EAGER);
//	Materializer* mat = new Materializer(fields, outputModes);

	//new mat.
	RecordAttribute *oidAttr = new RecordAttribute(0,l_linenumber->getRelationName(),activeLoop,pg->getOIDType());
	expressions::Expression *oidToMat = new expressions::RecordProjection(
			oidAttr->getOriginalType(), nestArg, *oidAttr);
	expressions::Expression *toMat1 = new expressions::RecordProjection(
			l_linenumber->getOriginalType(), nestArg, *l_linenumber);
	expressions::Expression *toMat2 = new expressions::RecordProjection(
			l_quantity->getOriginalType(), nestArg, *l_quantity);
	vector<expressions::Expression*> exprsToMat;
	exprsToMat.push_back(oidToMat);
	exprsToMat.push_back(toMat1);
	exprsToMat.push_back(toMat2);
	Materializer* mat = new Materializer(exprsToMat);

	char nestLabel[] = "nest_lineitem";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: COUNT(*) */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	aggrField1 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* + Aggregate 2: MAX(l_quantity) */
	expressions::Expression* outputExpr2 = new expressions::RecordProjection(
			l_quantity->getOriginalType(), nestArg, *l_quantity);
	aggrField2 = string("_aggrMaxQuantity");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	/* CONSTRUCT OUTPUT RECORD */
	RecordAttribute* cnt = new RecordAttribute(1, nestLabel, string("_aggrCount"),
			intType);
	RecordAttribute* max_qty = new RecordAttribute(2, nestLabel,
			string("_aggrMaxQuantity"), floatType);
	/* Used for argument construction */
	list<RecordAttribute> newArgTypes;
	newArgTypes.push_back(*cnt);
	newArgTypes.push_back(*max_qty);

	/* Used for record type construction */
	list<RecordAttribute*> newAttsTypes = list<RecordAttribute*>();
	newAttsTypes.push_back(cnt);
	newAttsTypes.push_back(max_qty);
	RecordType newRecType = RecordType(newAttsTypes);

	expressions::Expression* nestResultArg = new expressions::InputArgument(
			&newRecType, 0, newArgTypes);
	expressions::RecordProjection *projCnt = new expressions::RecordProjection(
			intType, nestResultArg, *cnt);
	expressions::RecordProjection *projMax = new expressions::RecordProjection(
			floatType, nestResultArg, *max_qty);

	expressions::AttributeConstruction attrExpr1 =
				expressions::AttributeConstruction("cnt", projCnt);
	expressions::AttributeConstruction attrExpr2 =
			expressions::AttributeConstruction("max_qty", projMax);
	list<expressions::AttributeConstruction> newAtts = list<
			expressions::AttributeConstruction>();
	newAtts.push_back(attrExpr1);
	newAtts.push_back(attrExpr2);
	expressions::RecordConstruction *outputExpr =
			new expressions::RecordConstruction(&newRecType, newAtts);

	/* Construct filtering predicate */

	expressions::Expression *predicate = new expressions::BoolConstant(true);

	vector<Monoid> reduceAccs;
	vector<expressions::Expression*> exprs;
	reduceAccs.push_back(BAGUNION);
	exprs.push_back(outputExpr);
	opt::Reduce *reduceOp = new opt::Reduce(reduceAccs, exprs, predicate,
			nestOp, &ctx, flushResults, testLabel);
	nestOp->setParent(reduceOp);
	reduceOp->produce();

	/* Execute */
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	catalog.clear();

	pg->finish();
	catalog.clear();

	verifyResult_(testLabel);
}

TEST(Output, JoinLeft3) {

	const char *testLabel = "3wayJoin.json";
	bool flushResults = true;


	RawContext ctx = RawContext(testLabel);
	registerFunctions(ctx);
	RawCatalog& catalog = RawCatalog::getInstance();

	/**
	 * SCAN1
	 */
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(4,filename,string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);
	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age); //Float

	int linehint = 10;
	int policy = 2;
	pm::CSVPlugin* pgSailors =
			new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
	catalog.registerPlugin(filename,pgSailors);

	Scan scanSailors = Scan(&ctx, *pgSailors);

	/**
	 * SCAN2
	 */
	string filename2 = string("inputs/reserves.csv");
	RecordAttribute* sidReserves = new RecordAttribute(1,filename2,string("sid"),intType);
	RecordAttribute* bidReserves = new RecordAttribute(2,filename2,string("bid"),intType);
	RecordAttribute* day = new RecordAttribute(3,filename2,string("day"),stringType);

	list<RecordAttribute*> attrList2;
	attrList2.push_back(sidReserves);
	attrList2.push_back(bidReserves);
	attrList2.push_back(day);
	RecordType rec2 = RecordType(attrList2);
	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(sidReserves);
	whichFields2.push_back(bidReserves);

	linehint = 10;
	policy = 2;
	pm::CSVPlugin* pgReserves =
			new pm::CSVPlugin(&ctx, filename2, rec2, whichFields2, ';', linehint, policy, false);

	catalog.registerPlugin(filename2,pgReserves);
	Scan scanReserves = Scan(&ctx, *pgReserves);

	/**
	 * JOIN
	 */
	/* Sailors: Left-side fields for materialization etc. */
	RecordAttribute projTupleL = RecordAttribute(filename, activeLoop,
			pgSailors->getOIDType());
	list<RecordAttribute> projectionsL = list<RecordAttribute>();
	projectionsL.push_back(projTupleL);
	projectionsL.push_back(*sid);
	projectionsL.push_back(*age);
	expressions::Expression* leftArg = new expressions::InputArgument(intType,
			0, projectionsL);
	expressions::Expression* leftOidProj = new expressions::RecordProjection(
			intType, leftArg, projTupleL);
	expressions::Expression* leftSidProj = new expressions::RecordProjection(
			intType, leftArg, *sid);
	expressions::Expression* ageProj = new expressions::RecordProjection(
			floatType, leftArg, *age);
	vector<expressions::Expression*> exprsToMatLeft;
	exprsToMatLeft.push_back(leftOidProj);
	exprsToMatLeft.push_back(leftSidProj);
	exprsToMatLeft.push_back(ageProj);
	Materializer* matLeft = new Materializer(exprsToMatLeft);

	/* Reserves: Right-side fields for materialization etc. */
	RecordAttribute projTupleR = RecordAttribute(filename2, activeLoop, pgReserves->getOIDType());
	list<RecordAttribute> projectionsR = list<RecordAttribute>();
	projectionsR.push_back(projTupleR);
	projectionsR.push_back(*sidReserves);
	projectionsR.push_back(*bidReserves);
	expressions::Expression* rightArg = new expressions::InputArgument(intType,
			1, projectionsR);
	expressions::Expression* rightOidProj = new expressions::RecordProjection(
			pgReserves->getOIDType(), rightArg, projTupleR);
	expressions::Expression* rightSidProj = new expressions::RecordProjection(
			intType, rightArg, *sidReserves);
	expressions::Expression* rightBidProj = new expressions::RecordProjection(
				intType, rightArg, *bidReserves);
	vector<expressions::Expression*> exprsToMatRight;
	exprsToMatRight.push_back(rightOidProj);
	//exprsToMatRight.push_back(rightSidProj);
	exprsToMatRight.push_back(rightBidProj);

	Materializer* matRight = new Materializer(exprsToMatRight);

	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),leftSidProj,rightSidProj);

	char joinLabel[] = "sailors_reserves";
	RadixJoin join = RadixJoin(joinPred, &scanSailors, &scanReserves, &ctx, joinLabel, *matLeft, *matRight);
	scanSailors.setParent(&join);
	scanReserves.setParent(&join);


	//SCAN3: BOATS
	string filenameBoats = string("inputs/boats.csv");
	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats,string("bid"),intType);
	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats,string("bname"),stringType);
	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats,string("color"),stringType);

	list<RecordAttribute*> attrListBoats;
	attrListBoats.push_back(bidBoats);
	attrListBoats.push_back(bnameBoats);
	attrListBoats.push_back(colorBoats);
	RecordType recBoats = RecordType(attrListBoats);

	vector<RecordAttribute*> whichFieldsBoats;
	whichFieldsBoats.push_back(bidBoats);

	linehint = 4;
	policy = 2;
	pm::CSVPlugin* pgBoats = new pm::CSVPlugin(&ctx, filenameBoats, recBoats,
			whichFieldsBoats, ';', linehint, policy, false);
	catalog.registerPlugin(filenameBoats,pgBoats);
	Scan scanBoats = Scan(&ctx, *pgBoats);

	/**
	 * JOIN2: BOATS
	 */
	expressions::Expression* leftArg2 = new expressions::InputArgument(intType,0,projectionsR);
	expressions::Expression* left2 = new expressions::RecordProjection(intType,leftArg2,*bidReserves);
	vector<expressions::Expression*> exprsToMatPreviousJoin;
	exprsToMatPreviousJoin.push_back(leftOidProj);
	exprsToMatPreviousJoin.push_back(rightOidProj);
	exprsToMatPreviousJoin.push_back(leftSidProj);
	Materializer* matPreviousJoin = new Materializer(exprsToMatPreviousJoin);

	RecordAttribute projTupleBoat = RecordAttribute(filenameBoats, activeLoop, pgBoats->getOIDType());
	list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
	fieldsBoats.push_back(projTupleBoat);
	fieldsBoats.push_back(*bidBoats);
	expressions::Expression* rightArg2 = new expressions::InputArgument(intType,1,fieldsBoats);
	expressions::Expression* oidBoatsProj = new expressions::RecordProjection(pgBoats->getOIDType(),rightArg2,projTupleBoat);
	expressions::Expression* bidBoatsProj = new expressions::RecordProjection(intType,rightArg2,*bidBoats);

	vector<expressions::Expression*> exprsToMatBoats;
	exprsToMatBoats.push_back(oidBoatsProj);
	exprsToMatBoats.push_back(bidBoatsProj);
	Materializer* matBoats = new Materializer(exprsToMatBoats);

	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(new BoolType(),left2,bidBoatsProj);

	char joinLabel2[] = "sailors_reserves_boats";
	RadixJoin join2 = RadixJoin(joinPred2, &join, &scanBoats, &ctx, joinLabel2, *matPreviousJoin, *matBoats);
	join.setParent(&join2);
	scanBoats.setParent(&join2);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression *arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression *outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);
	expressions::Expression *one = new expressions::IntConstant(1);

	expressions::Expression *lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression *predicate = new expressions::BoolConstant(true);

	vector<Monoid> accs;
	vector<expressions::Expression*> exprs;
	accs.push_back(MAX);
	exprs.push_back(outputExpr);
	/* Sanity checks*/
	accs.push_back(SUM);
	exprs.push_back(outputExpr);
	accs.push_back(SUM);
	exprs.push_back(one);
	opt::Reduce reduce = opt::Reduce(accs, exprs, predicate, &join2, &ctx,
			flushResults, testLabel);
	join2.setParent(&reduce);
	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgSailors->finish();
	pgReserves->finish();
	pgBoats->finish();
	catalog.clear();

	verifyResult_(testLabel);
}
