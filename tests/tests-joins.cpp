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
#include "test-utils.hpp"

#include "common/common.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "operators/radix-join.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

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

class SailorsTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    catalog = &Catalog::getInstance();
    caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  virtual void TearDown() {}

  pm::CSVPlugin *openCSV(Context *const context, string &fname, RecordType &rec,
                         vector<RecordAttribute *> whichFields, char delimInner,
                         int lineHint, int policy, bool stringBrackets = true) {
    pm::CSVPlugin *plugin =
        new pm::CSVPlugin(context, fname, rec, whichFields, delimInner,
                          lineHint, policy, stringBrackets);
    catalog->registerPlugin(fname, plugin);
    return plugin;
  }

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-sailors/";

 private:
  Catalog *catalog;
  CachingService *caches;
};

TEST_F(SailorsTest, Scan) {
  const char *testLabel = "sailorsScan.json";
  Context &ctx = *prepareContext(testLabel);

  /**
   * SCAN1
   */
  string sailorsPath = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, sailorsPath, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, sailorsPath, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, sailorsPath, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, sailorsPath, string("age"), floatType);

  list<RecordAttribute *> sailorAtts;
  sailorAtts.push_back(sid);
  sailorAtts.push_back(sname);
  sailorAtts.push_back(rating);
  sailorAtts.push_back(age);
  RecordType sailorRec{sailorAtts};

  vector<RecordAttribute *> sailorAttsToProject;
  sailorAttsToProject.push_back(sid);
  sailorAttsToProject.push_back(age);  // Float

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pgSailors =
      openCSV(&ctx, sailorsPath, sailorRec, sailorAttsToProject, ';', linehint,
              policy, false);
  Scan scanSailors{&ctx, *pgSailors};

  /**
   * REDUCE
   */
  expressions::InputArgument arg{&sailorRec, 0, {*sid, *age}};
  expression_t outputExpr = expression_t{arg}[*sid];

  vector<Monoid> accs{MAX, SUM, SUM};
  vector<expression_t> exprs{outputExpr, /* sanity checks=> */ outputExpr, 1};
  opt::Reduce reduce{accs, exprs,        true,     &scanSailors,
                     &ctx, flushResults, testLabel};
  scanSailors.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgSailors->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, Select) {
  const char *testLabel = "sailorsSel.json";
  Context &ctx = *prepareContext(testLabel);

  /**
   * SCAN1
   */
  string sailorsPath = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, sailorsPath, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, sailorsPath, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, sailorsPath, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, sailorsPath, string("age"), floatType);

  list<RecordAttribute *> sailorAtts;
  sailorAtts.push_back(sid);
  sailorAtts.push_back(sname);
  sailorAtts.push_back(rating);
  sailorAtts.push_back(age);
  RecordType sailorRec = RecordType(sailorAtts);

  vector<RecordAttribute *> sailorAttsToProject;
  sailorAttsToProject.push_back(sid);
  sailorAttsToProject.push_back(age);  // Float

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pgSailors =
      openCSV(&ctx, sailorsPath, sailorRec, sailorAttsToProject, ';', linehint,
              policy, false);
  Scan scanSailors = Scan(&ctx, *pgSailors);

  /**
   * SELECT
   */
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(*sid);
  projections.push_back(*age);

  expressions::Expression *arg =
      new expressions::InputArgument(&sailorRec, 0, projections);

  expressions::Expression *lhs =
      new expressions::RecordProjection(new FloatType(), arg, *age);
  expressions::Expression *rhs = new expressions::FloatConstant(40);
  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);
  Select sel = Select(predicate, &scanSailors);
  scanSailors.setParent(&sel);

  /**
   * REDUCE
   */
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(intType, arg, *sid);
  expressions::Expression *one = new expressions::IntConstant(1);

  expressions::Expression *predicateRed = new expressions::BoolConstant(true);

  vector<Monoid> accs;
  vector<expression_t> exprs;
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

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgSailors->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, ScanBoats) {
  const char *testLabel = "sailorsScanBoats.json";
  Context &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();

  string filenameBoats = string("inputs/boats.csv");
  RecordAttribute *bidBoats =
      new RecordAttribute(1, filenameBoats, string("bid"), intType);
  RecordAttribute *bnameBoats =
      new RecordAttribute(2, filenameBoats, string("bname"), stringType);
  RecordAttribute *colorBoats =
      new RecordAttribute(3, filenameBoats, string("color"), stringType);

  list<RecordAttribute *> attrListBoats;
  attrListBoats.push_back(bidBoats);
  attrListBoats.push_back(bnameBoats);
  attrListBoats.push_back(colorBoats);
  RecordType recBoats = RecordType(attrListBoats);

  vector<RecordAttribute *> whichFieldsBoats;
  whichFieldsBoats.push_back(bidBoats);

  int linehint = 4;
  int policy = 2;
  pm::CSVPlugin *pgBoats =
      openCSV(&ctx, filenameBoats, recBoats, whichFieldsBoats, ';', linehint,
              policy, false);
  Scan scanBoats = Scan(&ctx, *pgBoats);

  /**
   * REDUCE
   */
  list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
  fieldsBoats.push_back(*bidBoats);
  expressions::Expression *boatsArg =
      new expressions::InputArgument(intType, 1, fieldsBoats);

  expressions::Expression *outputExpr =
      new expressions::RecordProjection(intType, boatsArg, *bidBoats);
  expressions::Expression *one = new expressions::IntConstant(1);

  expressions::Expression *predicateRed = new expressions::BoolConstant(true);

  vector<Monoid> accs;
  vector<expression_t> exprs;
  accs.push_back(MAX);
  exprs.push_back(outputExpr);
  /* Sanity checks*/
  accs.push_back(SUM);
  exprs.push_back(outputExpr);
  accs.push_back(SUM);
  exprs.push_back(one);
  opt::Reduce reduce = opt::Reduce(accs, exprs, predicateRed, &scanBoats, &ctx,
                                   flushResults, testLabel);
  scanBoats.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgBoats->finish();

  EXPECT_TRUE(true);
}

TEST_F(SailorsTest, JoinLeft3) {
  const char *testLabel = "sailorsJoinLeft3.json";
  Context &ctx = *prepareContext(testLabel);

  /**
   * SCAN1
   */
  string sailorsPath = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, sailorsPath, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, sailorsPath, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, sailorsPath, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, sailorsPath, string("age"), floatType);

  list<RecordAttribute *> sailorAtts;
  sailorAtts.push_back(sid);
  sailorAtts.push_back(sname);
  sailorAtts.push_back(rating);
  sailorAtts.push_back(age);
  RecordType sailorRec = RecordType(sailorAtts);

  vector<RecordAttribute *> sailorAttsToProject;
  sailorAttsToProject.push_back(sid);
  sailorAttsToProject.push_back(age);  // Float

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pgSailors =
      openCSV(&ctx, sailorsPath, sailorRec, sailorAttsToProject, ';', linehint,
              policy, false);
  Scan scanSailors = Scan(&ctx, *pgSailors);

  /**
   * SCAN2
   */
  string reservesPath = string("inputs/reserves.csv");
  RecordAttribute *sidReserves =
      new RecordAttribute(1, reservesPath, string("sid"), intType);
  RecordAttribute *bidReserves =
      new RecordAttribute(2, reservesPath, string("bid"), intType);
  RecordAttribute *day =
      new RecordAttribute(3, reservesPath, string("day"), stringType);

  list<RecordAttribute *> reserveAtts;
  reserveAtts.push_back(sidReserves);
  reserveAtts.push_back(bidReserves);
  reserveAtts.push_back(day);
  RecordType reserveRec = RecordType(reserveAtts);
  vector<RecordAttribute *> reserveAttsToProject;
  reserveAttsToProject.push_back(sidReserves);
  reserveAttsToProject.push_back(bidReserves);

  linehint = 10;
  policy = 2;
  pm::CSVPlugin *pgReserves =
      openCSV(&ctx, reservesPath, reserveRec, reserveAttsToProject, ';',
              linehint, policy, false);
  Scan scanReserves = Scan(&ctx, *pgReserves);

  /**
   * JOIN
   */
  /* Sailors: Left-side fields for materialization etc. */
  RecordAttribute sailorOID =
      RecordAttribute(sailorsPath, activeLoop, pgSailors->getOIDType());
  list<RecordAttribute> sailorAttsForArg = list<RecordAttribute>();
  sailorAttsForArg.push_back(sailorOID);
  sailorAttsForArg.push_back(*sid);
  sailorAttsForArg.push_back(*age);
  expressions::Expression *sailorArg =
      new expressions::InputArgument(intType, 0, sailorAttsForArg);
  expressions::Expression *sailorOIDProj =
      new expressions::RecordProjection(intType, sailorArg, sailorOID);
  expressions::Expression *sailorSIDProj =
      new expressions::RecordProjection(intType, sailorArg, *sid);
  expressions::Expression *sailorAgeProj =
      new expressions::RecordProjection(floatType, sailorArg, *age);
  vector<expression_t> exprsToMatSailor;
  exprsToMatSailor.push_back(sailorOIDProj);
  exprsToMatSailor.push_back(sailorSIDProj);
  exprsToMatSailor.push_back(sailorAgeProj);
  Materializer *matSailor = new Materializer(exprsToMatSailor);

  /* Reserves: Right-side fields for materialization etc. */
  RecordAttribute reservesOID =
      RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
  list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
  reserveAttsForArg.push_back(reservesOID);
  reserveAttsForArg.push_back(*sidReserves);
  reserveAttsForArg.push_back(*bidReserves);
  expressions::Expression *reservesArg =
      new expressions::InputArgument(intType, 1, reserveAttsForArg);
  expressions::Expression *reservesOIDProj = new expressions::RecordProjection(
      pgReserves->getOIDType(), reservesArg, reservesOID);
  expressions::Expression *reservesSIDProj =
      new expressions::RecordProjection(intType, reservesArg, *sidReserves);
  expressions::Expression *reservesBIDProj =
      new expressions::RecordProjection(intType, reservesArg, *bidReserves);
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  // exprsToMatRight.push_back(resevesSIDProj);
  exprsToMatReserves.push_back(reservesBIDProj);

  Materializer *matReserves = new Materializer(exprsToMatReserves);

  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(sailorSIDProj, reservesSIDProj);

  char joinLabel[] = "sailors_reserves";
  RadixJoin join(*joinPred, &scanSailors, &scanReserves, &ctx, joinLabel,
                 *matSailor, *matReserves);
  scanSailors.setParent(&join);
  scanReserves.setParent(&join);

  // SCAN3: BOATS
  string filenameBoats = string("inputs/boats.csv");
  RecordAttribute *bidBoats =
      new RecordAttribute(1, filenameBoats, string("bid"), intType);
  RecordAttribute *bnameBoats =
      new RecordAttribute(2, filenameBoats, string("bname"), stringType);
  RecordAttribute *colorBoats =
      new RecordAttribute(3, filenameBoats, string("color"), stringType);

  list<RecordAttribute *> attrListBoats;
  attrListBoats.push_back(bidBoats);
  attrListBoats.push_back(bnameBoats);
  attrListBoats.push_back(colorBoats);
  RecordType recBoats = RecordType(attrListBoats);

  vector<RecordAttribute *> whichFieldsBoats;
  whichFieldsBoats.push_back(bidBoats);

  linehint = 4;
  policy = 2;
  pm::CSVPlugin *pgBoats =
      openCSV(&ctx, filenameBoats, recBoats, whichFieldsBoats, ';', linehint,
              policy, false);
  Scan scanBoats = Scan(&ctx, *pgBoats);

  /**
   * JOIN2: BOATS
   */
  expressions::Expression *previousJoinArg =
      new expressions::InputArgument(intType, 0, reserveAttsForArg);
  expressions::Expression *previousJoinBIDProj =
      new expressions::RecordProjection(intType, previousJoinArg, *bidReserves);
  vector<expression_t> exprsToMatPreviousJoin;
  exprsToMatPreviousJoin.push_back(sailorOIDProj);
  exprsToMatPreviousJoin.push_back(reservesOIDProj);
  exprsToMatPreviousJoin.push_back(sailorSIDProj);
  Materializer *matPreviousJoin = new Materializer(exprsToMatPreviousJoin);

  RecordAttribute projTupleBoat =
      RecordAttribute(filenameBoats, activeLoop, pgBoats->getOIDType());
  list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
  fieldsBoats.push_back(projTupleBoat);
  fieldsBoats.push_back(*bidBoats);
  expressions::Expression *boatsArg =
      new expressions::InputArgument(intType, 1, fieldsBoats);
  expressions::Expression *boatsOIDProj = new expressions::RecordProjection(
      pgBoats->getOIDType(), boatsArg, projTupleBoat);
  expressions::Expression *boatsBIDProj =
      new expressions::RecordProjection(intType, boatsArg, *bidBoats);

  vector<expression_t> exprsToMatBoats;
  exprsToMatBoats.push_back(boatsOIDProj);
  exprsToMatBoats.push_back(boatsBIDProj);
  Materializer *matBoats = new Materializer(exprsToMatBoats);

  expressions::BinaryExpression *joinPred2 =
      new expressions::EqExpression(previousJoinBIDProj, boatsBIDProj);

  char joinLabel2[] = "sailors_reserves_boats";
  RadixJoin join2(*joinPred2, &join, &scanBoats, &ctx, joinLabel2,
                  *matPreviousJoin, *matBoats);
  join.setParent(&join2);
  scanBoats.setParent(&join2);

  /**
   * REDUCE
   */
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(*sid);
  projections.push_back(*age);

  expressions::Expression *arg =
      new expressions::InputArgument(&sailorRec, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(intType, arg, *sid);
  expressions::Expression *one = new expressions::IntConstant(1);

  expressions::Expression *predicate = new expressions::BoolConstant(true);

  vector<Monoid> accs;
  vector<expression_t> exprs;
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

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgSailors->finish();
  pgReserves->finish();
  pgBoats->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, JoinRight3) {
  const char *testLabel = "sailorsJoinRight3.json";
  Context &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();

  /**
   * SCAN2
   */
  string reservesPath = string("inputs/reserves.csv");
  RecordAttribute *sidReserves =
      new RecordAttribute(1, reservesPath, string("sid"), intType);
  RecordAttribute *bidReserves =
      new RecordAttribute(2, reservesPath, string("bid"), intType);
  RecordAttribute *day =
      new RecordAttribute(3, reservesPath, string("day"), stringType);

  list<RecordAttribute *> reserveAtts;
  reserveAtts.push_back(sidReserves);
  reserveAtts.push_back(bidReserves);
  reserveAtts.push_back(day);
  RecordType reserveRec = RecordType(reserveAtts);
  vector<RecordAttribute *> reserveAttsToProject;
  reserveAttsToProject.push_back(sidReserves);
  reserveAttsToProject.push_back(bidReserves);

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pgReserves =
      openCSV(&ctx, reservesPath, reserveRec, reserveAttsToProject, ';',
              linehint, policy, false);
  Scan scanReserves = Scan(&ctx, *pgReserves);

  // SCAN3: BOATS
  string filenameBoats = string("inputs/boats.csv");
  RecordAttribute *bidBoats =
      new RecordAttribute(1, filenameBoats, string("bid"), intType);
  RecordAttribute *bnameBoats =
      new RecordAttribute(2, filenameBoats, string("bname"), stringType);
  RecordAttribute *colorBoats =
      new RecordAttribute(3, filenameBoats, string("color"), stringType);

  list<RecordAttribute *> attrListBoats;
  attrListBoats.push_back(bidBoats);
  attrListBoats.push_back(bnameBoats);
  attrListBoats.push_back(colorBoats);
  RecordType recBoats = RecordType(attrListBoats);

  vector<RecordAttribute *> whichFieldsBoats;
  whichFieldsBoats.push_back(bidBoats);

  linehint = 4;
  policy = 2;
  pm::CSVPlugin *pgBoats =
      openCSV(&ctx, filenameBoats, recBoats, whichFieldsBoats, ';', linehint,
              policy, false);
  Scan scanBoats = Scan(&ctx, *pgBoats);

  /**
   * JOIN: Reserves JOIN Boats
   */
  /* Reserves: fields for materialization etc. */
  RecordAttribute reservesOID =
      RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
  list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
  reserveAttsForArg.push_back(reservesOID);
  reserveAttsForArg.push_back(*sidReserves);
  reserveAttsForArg.push_back(*bidReserves);
  expressions::Expression *reservesArg =
      new expressions::InputArgument(intType, 1, reserveAttsForArg);
  expressions::Expression *reservesOIDProj = new expressions::RecordProjection(
      pgReserves->getOIDType(), reservesArg, reservesOID);
  expressions::Expression *reservesSIDProj =
      new expressions::RecordProjection(intType, reservesArg, *sidReserves);
  expressions::Expression *reservesBIDProj =
      new expressions::RecordProjection(intType, reservesArg, *bidReserves);
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  exprsToMatReserves.push_back(reservesSIDProj);
  exprsToMatReserves.push_back(reservesBIDProj);

  Materializer *matReserves = new Materializer(exprsToMatReserves);

  RecordAttribute projTupleBoat =
      RecordAttribute(filenameBoats, activeLoop, pgBoats->getOIDType());
  list<RecordAttribute> fieldsBoats = list<RecordAttribute>();
  fieldsBoats.push_back(projTupleBoat);
  fieldsBoats.push_back(*bidBoats);
  expressions::Expression *boatsArg =
      new expressions::InputArgument(intType, 1, fieldsBoats);
  expressions::Expression *boatsOIDProj = new expressions::RecordProjection(
      pgBoats->getOIDType(), boatsArg, projTupleBoat);
  expressions::Expression *boatsBIDProj =
      new expressions::RecordProjection(intType, boatsArg, *bidBoats);
  vector<expression_t> exprsToMatBoats;
  exprsToMatBoats.push_back(boatsOIDProj);
  exprsToMatBoats.push_back(boatsBIDProj);
  Materializer *matBoats = new Materializer(exprsToMatBoats);

  expressions::BinaryExpression *joinPred2 =
      new expressions::EqExpression(reservesBIDProj, boatsBIDProj);

  char joinLabel2[] = "reserves_boats";
  RadixJoin join2(*joinPred2, &scanReserves, &scanBoats, &ctx, joinLabel2,
                  *matReserves, *matBoats);
  scanReserves.setParent(&join2);
  scanBoats.setParent(&join2);

  /**
   * SCAN1
   */
  string sailorsPath = string("inputs/sailors.csv");

  RecordAttribute *sid =
      new RecordAttribute(1, sailorsPath, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, sailorsPath, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, sailorsPath, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, sailorsPath, string("age"), floatType);

  list<RecordAttribute *> sailorAtts;
  sailorAtts.push_back(sid);
  sailorAtts.push_back(sname);
  sailorAtts.push_back(rating);
  sailorAtts.push_back(age);
  RecordType sailorRec = RecordType(sailorAtts);

  vector<RecordAttribute *> sailorAttsToProject;
  sailorAttsToProject.push_back(sid);
  sailorAttsToProject.push_back(age);  // Float

  linehint = 10;
  policy = 2;
  pm::CSVPlugin *pgSailors =
      openCSV(&ctx, sailorsPath, sailorRec, sailorAttsToProject, ';', linehint,
              policy, false);
  Scan scanSailors = Scan(&ctx, *pgSailors);

  /**
   * JOIN: Sailors JOIN (Reserves JOIN Boats)
   */

  /* Sailors: fields for materialization etc. */
  RecordAttribute sailorOID =
      RecordAttribute(sailorsPath, activeLoop, pgSailors->getOIDType());
  list<RecordAttribute> sailorAttsForArg = list<RecordAttribute>();
  sailorAttsForArg.push_back(sailorOID);
  sailorAttsForArg.push_back(*sid);
  sailorAttsForArg.push_back(*age);
  expressions::Expression *sailorArg =
      new expressions::InputArgument(intType, 0, sailorAttsForArg);
  expressions::Expression *sailorOIDProj =
      new expressions::RecordProjection(intType, sailorArg, sailorOID);
  expressions::Expression *sailorSIDProj =
      new expressions::RecordProjection(intType, sailorArg, *sid);
  vector<expression_t> exprsToMatSailor;
  exprsToMatSailor.push_back(sailorOIDProj);
  exprsToMatSailor.push_back(sailorSIDProj);
  Materializer *matSailor = new Materializer(exprsToMatSailor);

  expressions::Expression *previousJoinArg =
      new expressions::InputArgument(intType, 0, reserveAttsForArg);

  vector<expression_t> exprsToMatPreviousJoin;
  exprsToMatPreviousJoin.push_back(reservesOIDProj);
  exprsToMatPreviousJoin.push_back(reservesSIDProj);
  Materializer *matPreviousJoin = new Materializer(exprsToMatPreviousJoin);

  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(sailorSIDProj, reservesSIDProj);

  char joinLabel[] = "sailors_(reserves_boats)";
  RadixJoin join(*joinPred, &scanSailors, &join2, &ctx, joinLabel, *matSailor,
                 *matPreviousJoin);
  scanSailors.setParent(&join);
  join2.setParent(&join);

  /**
   * REDUCE
   */
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(*sid);

  expressions::Expression *arg =
      new expressions::InputArgument(&sailorRec, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(intType, arg, *sid);
  expressions::Expression *one = new expressions::IntConstant(1);

  expressions::Expression *predicate = new expressions::BoolConstant(true);

  vector<Monoid> accs;
  vector<expression_t> exprs;
  accs.push_back(MAX);
  exprs.push_back(outputExpr);
  /* Sanity checks*/
  accs.push_back(SUM);
  exprs.push_back(outputExpr);
  accs.push_back(SUM);
  exprs.push_back(one);
  opt::Reduce reduce =
      opt::Reduce(accs, exprs, predicate, &join, &ctx, flushResults, testLabel);
  join.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgSailors->finish();
  pgReserves->finish();
  pgBoats->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}
