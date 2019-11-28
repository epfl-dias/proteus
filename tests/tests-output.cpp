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
#include "common/common.hpp"
#include "common/tpch-config.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "gtest/gtest.h"
#include "operators/flush.hpp"
#include "operators/join.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/radix-join.hpp"
#include "operators/radix-nest.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "storage/storage-manager.hpp"
#include "test-utils.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/functions.hpp"
#include "util/parallel-context.hpp"
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

class OutputTest : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

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
  const char *testPath = TEST_OUTPUTS "/tests-output/";

  bool executePlan(ParallelContext &ctx, const char *testLabel,
                   std::vector<Plugin *> pgs) {
    ctx.compileAndLoad();
    auto pipelines = ctx.getPipelines();

    {
      time_block t("Texecute       : ");

      for (Pipeline *p : pipelines) {
        {
          time_block t("T: ");

          p->open();
          p->consume(0);
          p->close();
        }
      }
    }

    for (auto &pg : pgs) pg->finish();

    bool res = verifyTestResult(testPath, testLabel, true);
    shm_unlink(testLabel);
    return res;
  }

 private:
  Catalog *catalog;
  CachingService *caches;
};

void OutputTest::SetUp() {
  auto &topo = topology::getInstance();
  if (topo.getGpuCount() > 0) {
    exec_location{topo.getGpus()[0]}.activate();
  } else {
    exec_location{topo.getCpuNumaNodes()[0]}.activate();
  }

  catalog = &Catalog::getInstance();
  caches = &CachingService::getInstance();
  catalog->clear();
  caches->clear();
}

void OutputTest::TearDown() { StorageManager::unloadAll(); }

// works on new planner
// select max(sid) from sailors where age > 40 ;
TEST_F(OutputTest, ReduceNumeric) {
  const char *testLabel = "reduceNumeric.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  // SCAN1
  string filename = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, filename, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, filename, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, filename, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, filename, string("age"), floatType);

  list<RecordAttribute *> attrList;
  attrList.push_back(sid);
  attrList.push_back(sname);
  attrList.push_back(rating);
  attrList.push_back(age);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(sid);
  whichFields.push_back(age);

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pg =
      openCSV(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
  Scan scan = Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple{filename, activeLoop, pg->getOIDType()};

  list<RecordAttribute> projections{projTuple, *sid, *age};

  expressions::InputArgument arg{&rec1, 0, projections};
  auto outputExpr = expression_t{arg}[*sid];

  auto predicate = gt(expression_t{arg}[*age], 40.0);

  vector<Monoid> accs{MAX};
  vector<expression_t> exprs{outputExpr};
  opt::Reduce reduce =
      opt::Reduce(accs, exprs, predicate, &scan, &ctx, false, testLabel);
  scan.setParent(&reduce);

  Flush flush{exprs, &reduce, &ctx, testLabel};
  reduce.setParent(&flush);

  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

// works on new planner BUT planner does not request the output as json
// select sum(sid), max(sid) from sailors where age > 40 ;
TEST_F(OutputTest, MultiReduceNumeric) {
  const char *testLabel = "multiReduceNumeric.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  // SCAN1
  string filename = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, filename, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, filename, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, filename, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, filename, string("age"), floatType);

  list<RecordAttribute *> attrList;
  attrList.push_back(sid);
  attrList.push_back(sname);
  attrList.push_back(rating);
  attrList.push_back(age);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(sid);
  whichFields.push_back(age);

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pg =
      openCSV(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
  Scan scan = Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple{filename, activeLoop, pg->getOIDType()};
  list<RecordAttribute> projections{projTuple, *sid, *age};

  expressions::InputArgument arg{&rec1, 0, projections};
  auto outputExpr = expression_t{arg}[*sid];
  auto outputExpr2 = expression_t{arg}[*sid];
  outputExpr.registerAs(outputExpr.getRegisteredRelName(),
                        outputExpr.getProjectionName() + "$0");
  outputExpr2.registerAs(outputExpr.getRegisteredRelName(),
                         outputExpr.getProjectionName() + "$1");
  std::cout << outputExpr.getRegisteredRelName() << "+"
            << outputExpr.getRegisteredAttrName() << std::endl;
  auto predicate = gt(expression_t{arg}[*age], 40.0);

  vector<Monoid> accs{SUM, MAX};
  vector<expression_t> exprs{outputExpr, outputExpr2};
  opt::Reduce reduce{accs, exprs, predicate, &scan, &ctx, false, testLabel};
  scan.setParent(&reduce);

  vector<expression_t> exprs_flush{
      expression_t{arg}[outputExpr.getRegisteredAs()],
      expression_t{arg}[outputExpr2.getRegisteredAs()]};
  Flush flush{exprs_flush, &reduce, &ctx, testLabel};
  reduce.setParent(&flush);

  flush.produce();

  // Run function
  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

// works on new planner BUT planner does not request the output as json
// select sid from sailors where age > 40 ;
TEST_F(OutputTest, ReduceBag) {
  const char *testLabel = "reduceBag.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  // SCAN1
  string filename = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, filename, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, filename, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, filename, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, filename, string("age"), floatType);

  list<RecordAttribute *> attrList;
  attrList.push_back(sid);
  attrList.push_back(sname);
  attrList.push_back(rating);
  attrList.push_back(age);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(sid);
  whichFields.push_back(age);

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pg =
      openCSV(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
  Scan scan = Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple{filename, activeLoop, pg->getOIDType()};
  list<RecordAttribute> projections{projTuple, *sid, *age};

  expressions::InputArgument arg{&rec1, 0, projections};
  auto outputExpr = expression_t{arg}[*sid];
  auto predicate = gt(expression_t{arg}[*age], 40.0);

  vector<Monoid> accs{BAGUNION};
  vector<expression_t> exprs{outputExpr};

  Select filter{predicate, &scan};
  scan.setParent(&filter);

  //    Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
  //    Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);
  Flush flush{exprs, &filter, &ctx, testLabel};
  filter.setParent(&flush);

  flush.produce();

  // Run function
  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

// works on new planner BUT planner does not request the output as json
// select sid as id, age as age from sailors where age > 40 ;
TEST_F(OutputTest, ReduceBagRecord) {
  const char *testLabel = "reduceBagRecord.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  // SCAN1
  string filename = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, filename, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, filename, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, filename, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, filename, string("age"), floatType);

  list<RecordAttribute *> attrList;
  attrList.push_back(sid);
  attrList.push_back(sname);
  attrList.push_back(rating);
  attrList.push_back(age);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(sid);
  whichFields.push_back(age);

  int linehint = 10;
  int policy = 2;
  pm::CSVPlugin *pg =
      openCSV(&ctx, filename, rec1, whichFields, ';', linehint, policy, false);
  Scan scan = Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple{filename, activeLoop, pg->getOIDType()};
  list<RecordAttribute> projections{projTuple, *sid, *age};

  expressions::InputArgument arg{&rec1, 0, projections};

  /* CONSTRUCT OUTPUT RECORD */
  list<RecordAttribute *> newAttsTypes{sid, age};

  RecordType newRecType{newAttsTypes};
  auto projID = expression_t{arg}[*sid];
  auto projAge = expression_t{arg}[*age];

  expressions::AttributeConstruction attrExpr1{"id", projID};
  expressions::AttributeConstruction attrExpr2{"age", projAge};
  list<expressions::AttributeConstruction> newAtts;
  newAtts.push_back(attrExpr1);
  newAtts.push_back(attrExpr2);
  expressions::RecordConstruction outputExpr{newAtts};

  /* Construct filtering predicate */
  auto predicate = gt(projAge, 40.0);

  Select filter{predicate, &scan};
  scan.setParent(&filter);

  vector<expression_t> exprs{projID, projAge};
  //    Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
  //    Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);
  Flush flush{exprs, &filter, &ctx, testLabel};
  filter.setParent(&flush);

  flush.produce();

  // Run function
  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

// table not in catalog/repo
TEST_F(OutputTest, NestBagTPCH) {
  const char *testLabel = "nestBagTPCH.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();

  /* File Info */
  map<string, dataset> datasetCatalog;
  //    tpchSchemaCSV(datasetCatalog);
  //    string nameLineitem = string("lineitem");
  //    dataset lineitem = datasetCatalog[nameLineitem];
  //    map<string, RecordAttribute*> argsLineitem =
  //    lineitem.recType.getArgsMap();

  string lineitemPath = string("inputs/tpch/lineitem10.csv");
  list<RecordAttribute *> attsLineitem = list<RecordAttribute *>();
  RecordAttribute *l_orderkey =
      new RecordAttribute(1, lineitemPath, "l_orderkey", intType);
  attsLineitem.push_back(l_orderkey);
  RecordAttribute *l_partkey =
      new RecordAttribute(2, lineitemPath, "l_partkey", intType);
  attsLineitem.push_back(l_partkey);
  RecordAttribute *l_suppkey =
      new RecordAttribute(3, lineitemPath, "l_suppkey", intType);
  attsLineitem.push_back(l_suppkey);
  RecordAttribute *l_linenumber =
      new RecordAttribute(4, lineitemPath, "l_linenumber", intType);
  attsLineitem.push_back(l_linenumber);
  RecordAttribute *l_quantity =
      new RecordAttribute(5, lineitemPath, "l_quantity", floatType);
  attsLineitem.push_back(l_quantity);
  RecordAttribute *l_extendedprice =
      new RecordAttribute(6, lineitemPath, "l_extendedprice", floatType);
  attsLineitem.push_back(l_extendedprice);
  RecordAttribute *l_discount =
      new RecordAttribute(7, lineitemPath, "l_discount", floatType);
  attsLineitem.push_back(l_discount);
  RecordAttribute *l_tax =
      new RecordAttribute(8, lineitemPath, "l_tax", floatType);
  attsLineitem.push_back(l_tax);
  RecordAttribute *l_returnflag =
      new RecordAttribute(9, lineitemPath, "l_returnflag", stringType);
  attsLineitem.push_back(l_returnflag);
  RecordAttribute *l_linestatus =
      new RecordAttribute(10, lineitemPath, "l_linestatus", stringType);
  attsLineitem.push_back(l_linestatus);
  RecordAttribute *l_shipdate =
      new RecordAttribute(11, lineitemPath, "l_shipdate", stringType);
  attsLineitem.push_back(l_shipdate);
  RecordAttribute *l_commitdate =
      new RecordAttribute(12, lineitemPath, "l_commitdate", stringType);
  attsLineitem.push_back(l_commitdate);
  RecordAttribute *l_receiptdate =
      new RecordAttribute(13, lineitemPath, "l_receiptdate", stringType);
  attsLineitem.push_back(l_receiptdate);
  RecordAttribute *l_shipinstruct =
      new RecordAttribute(14, lineitemPath, "l_shipinstruct", stringType);
  attsLineitem.push_back(l_shipinstruct);
  RecordAttribute *l_shipmode =
      new RecordAttribute(15, lineitemPath, "l_shipmode", stringType);
  attsLineitem.push_back(l_shipmode);
  RecordAttribute *l_comment =
      new RecordAttribute(16, lineitemPath, "l_comment", stringType);
  attsLineitem.push_back(l_comment);

  RecordType rec = RecordType(attsLineitem);

  int linehint = 10;
  int policy = 5;
  char delimInner = '|';

  /* Projections */
  vector<RecordAttribute *> projections;

  projections.push_back(l_orderkey);
  projections.push_back(l_linenumber);
  projections.push_back(l_quantity);

  pm::CSVPlugin *pg = openCSV(&ctx, lineitemPath, rec, projections, delimInner,
                              linehint, policy, false);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*l_orderkey);
  argProjections.push_back(*l_quantity);

  expressions::InputArgument arg{&rec, 0, argProjections};

  auto pred = lt(expression_t{arg}[*l_orderkey], 4);

  Select *sel = new Select(pred, scan);
  scan->setParent(sel);

  /**
   * NEST
   * GroupBy: l_linenumber
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: COUNT() => SUM(1)
   */
  list<RecordAttribute> nestProjections;

  nestProjections.push_back(*l_quantity);

  expressions::InputArgument nestArg{&rec, 0, nestProjections};
  // f (& g)
  auto f = expression_t{nestArg}[*l_linenumber];
  // p
  auto predNest = eq(true, true);

  // mat.
  //    vector<RecordAttribute*> fields;
  //    vector<materialization_mode> outputModes;
  //    fields.push_back(l_linenumber);
  //    outputModes.insert(outputModes.begin(), EAGER);
  //    fields.push_back(l_quantity);
  //    outputModes.insert(outputModes.begin(), EAGER);
  //    Materializer* mat = new Materializer(fields, outputModes);

  // new mat.
  RecordAttribute *oidAttr = new RecordAttribute(
      0, l_linenumber->getRelationName(), activeLoop, pg->getOIDType());
  auto oidToMat = expression_t{nestArg}[*oidAttr];
  auto toMat1 = expression_t{nestArg}[*l_linenumber];
  auto toMat2 = expression_t{nestArg}[*l_quantity];
  vector<expression_t> exprsToMat{oidToMat, toMat1, toMat2};
  Materializer *mat = new Materializer(exprsToMat);

  char nestLabel[] = "nest_lineitem";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;

  /* Aggregate 1: COUNT(*) */
  std::string aggrField1{"_aggrCount"};
  accs.push_back(SUM);
  outputExprs.push_back(1);
  aggrLabels.push_back(aggrField1);

  /* + Aggregate 2: MAX(l_quantity) */
  auto outputExpr2 = expression_t{nestArg}[*l_quantity];
  std::string aggrField2{"_aggrMaxQuantity"};
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  /* CONSTRUCT OUTPUT RECORD */
  RecordAttribute *cnt =
      new RecordAttribute(1, nestLabel, string("_aggrCount"), intType);
  RecordAttribute *max_qty =
      new RecordAttribute(2, nestLabel, string("_aggrMaxQuantity"), floatType);
  /* Used for argument construction */
  list<RecordAttribute> newArgTypes;
  newArgTypes.push_back(*cnt);
  newArgTypes.push_back(*max_qty);

  /* Used for record type construction */
  list<RecordAttribute *> newAttsTypes;
  newAttsTypes.push_back(cnt);
  newAttsTypes.push_back(max_qty);
  RecordType newRecType = RecordType(newAttsTypes);

  expressions::InputArgument nestResultArg{&newRecType, 0, newArgTypes};
  auto projCnt = expression_t{nestResultArg}[*cnt];
  auto projMax = expression_t{nestResultArg}[*max_qty];

  // expressions::AttributeConstruction attrExpr1("cnt", projCnt);
  // expressions::AttributeConstruction attrExpr2("max_qty", projMax);

  Flush *flush = new Flush({projCnt, projMax}, nestOp, &ctx, testLabel);
  nestOp->setParent(flush);
  flush->produce();

  // Run function
  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

TEST_F(OutputTest, JoinLeft3) {
  const char *testLabel = "3wayJoin.json";
  ParallelContext &ctx = *prepareContext(testLabel);

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
  expression_t sailorArg{
      expressions::InputArgument{new RecordType{std::vector<RecordAttribute *>{
                                     new RecordAttribute{sailorOID}, sid, age}},
                                 0}};
  expression_t sailorOIDProj = sailorArg[sailorOID];
  expression_t sailorSIDProj = sailorArg[*sid];
  expression_t sailorAgeProj = sailorArg[*age];

  vector<expression_t> exprsToMatSailor{sailorOIDProj, sailorSIDProj,
                                        sailorAgeProj};
  Materializer matSailor{exprsToMatSailor};

  /* Reserves: Right-side fields for materialization etc. */
  RecordAttribute reservesOID(reservesPath, activeLoop,
                              pgReserves->getOIDType());
  expression_t reservesArg{expressions::InputArgument{
      new RecordType{std::vector<RecordAttribute *>{
          new RecordAttribute{reservesOID}, sidReserves, bidReserves}},
      1}};
  expression_t reservesOIDProj = reservesArg[reservesOID];
  expression_t reservesSIDProj = reservesArg[*sidReserves];
  expression_t reservesBIDProj = reservesArg[*bidReserves];
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  // exprsToMatRight.push_back(resevesSIDProj);
  exprsToMatReserves.push_back(reservesBIDProj);

  Materializer matReserves(exprsToMatReserves);

  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(sailorSIDProj, reservesSIDProj);

  RadixJoin join(*joinPred, &scanSailors, &scanReserves, &ctx,
                 "sailors_reserves", matSailor, matReserves);
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
  auto previousJoinArg = expression_t{expressions::InputArgument(
      new RecordType{std::vector<RecordAttribute *>{
          new RecordAttribute{reservesOID}, sidReserves, bidReserves}},
      0)};
  auto previousJoinBIDProj = previousJoinArg[*bidReserves];
  vector<expression_t> exprsToMatPreviousJoin;
  exprsToMatPreviousJoin.push_back(sailorOIDProj);
  exprsToMatPreviousJoin.push_back(reservesOIDProj);
  exprsToMatPreviousJoin.push_back(sailorSIDProj);
  Materializer matPreviousJoin{exprsToMatPreviousJoin};

  RecordAttribute projTupleBoat(filenameBoats, activeLoop,
                                pgBoats->getOIDType());
  list<RecordAttribute> fieldsBoats;
  fieldsBoats.push_back(projTupleBoat);
  fieldsBoats.push_back(*bidBoats);
  auto boatsArg = expression_t{expressions::InputArgument(
      new RecordType{std::vector<RecordAttribute *>{
          new RecordAttribute{projTupleBoat}, bidBoats}},
      1)};
  auto boatsOIDProj = boatsArg[projTupleBoat];
  auto boatsBIDProj = boatsArg[*bidBoats];

  assert(boatsBIDProj.isRegistered());

  vector<expression_t> exprsToMatBoats{boatsOIDProj, boatsBIDProj};
  assert(exprsToMatBoats.back().isRegistered());

  Materializer matBoats(exprsToMatBoats);
  std::cout << matBoats.getWantedExpressions().size() << std::endl;
  assert(matBoats.getWantedExpressions().back().isRegistered());

  expressions::BinaryExpression *joinPred2 =
      new expressions::EqExpression(previousJoinBIDProj, boatsBIDProj);

  char joinLabel2[] = "sailors_reserves_boats";
  RadixJoin join2(*joinPred2, &join, &scanBoats, &ctx, joinLabel2,
                  matPreviousJoin, matBoats);
  join.setParent(&join2);
  scanBoats.setParent(&join2);

  /**
   * REDUCE
   */
  list<RecordAttribute> projections{*sid, *age};

  auto arg =
      expression_t{expressions::InputArgument(&sailorRec, 0, projections)};
  auto outputExpr = arg[*sid];

  vector<Monoid> accs;
  vector<expression_t> exprs;
  accs.push_back(MAX);
  exprs.push_back(outputExpr);
  /* Sanity checks*/
  accs.push_back(SUM);
  exprs.push_back(outputExpr);
  accs.push_back(SUM);
  exprs.push_back(1);
  opt::Reduce reduce(accs, exprs, true, &join2, &ctx, flushResults, testLabel);
  join2.setParent(&reduce);
  reduce.produce();

  // Run function
  EXPECT_TRUE(executePlan(ctx, testLabel, {pgSailors, pgReserves, pgBoats}));
}

/* Corresponds to plan parser tests */
TEST_F(OutputTest, NestReserves) {
  const char *testLabel = "nestReserves.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();

  /**
   * SCAN RESERVES
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

  /*
   * NEST
   */

  /* Reserves: fields for materialization etc. */
  RecordAttribute *reservesOID =
      new RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
  list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
  reserveAttsForArg.push_back(*reservesOID);
  reserveAttsForArg.push_back(*sidReserves);

  /* constructing recType */
  list<RecordAttribute *> reserveAttsForRec = list<RecordAttribute *>();
  reserveAttsForRec.push_back(reservesOID);
  reserveAttsForRec.push_back(sidReserves);
  RecordType reserveRecType = RecordType(reserveAttsForRec);

  expressions::Expression *reservesArg =
      new expressions::InputArgument(&reserveRecType, 1, reserveAttsForArg);
  expressions::Expression *reservesOIDProj = new expressions::RecordProjection(
      pgReserves->getOIDType(), reservesArg, *reservesOID);
  expressions::Expression *reservesSIDProj =
      new expressions::RecordProjection(intType, reservesArg, *sidReserves);
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  exprsToMatReserves.push_back(reservesSIDProj);
  Materializer *matReserves = new Materializer(exprsToMatReserves);

  /* group-by expr */
  expressions::Expression *f = reservesSIDProj;
  /* null handling */
  expressions::Expression *g = reservesSIDProj;

  expressions::Expression *nestPred = new expressions::BoolConstant(true);

  /* output of nest */
  vector<Monoid> accsNest;
  vector<expression_t> exprsNest;
  vector<string> aggrLabels;
  expressions::Expression *one = new expressions::IntConstant(1);
  accsNest.push_back(SUM);
  exprsNest.push_back(one);
  aggrLabels.push_back("_groupCount");

  char nestLabel[] = "nest_reserves";
  radix::Nest nest =
      radix::Nest(&ctx, accsNest, exprsNest, aggrLabels, nestPred, f, g,
                  &scanReserves, nestLabel, *matReserves);
  scanReserves.setParent(&nest);

  /* REDUCE */
  RecordAttribute *cnt =
      new RecordAttribute(1, nestLabel, string("_groupCount"), intType);
  list<RecordAttribute *> newAttsTypes = list<RecordAttribute *>();
  newAttsTypes.push_back(cnt);
  RecordType newRecType = RecordType(newAttsTypes);

  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(*cnt);

  expressions::Expression *arg =
      new expressions::InputArgument(&newRecType, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(intType, arg, *cnt);

  expressions::Expression *predicate = new expressions::BoolConstant(true);

  vector<Monoid> accs;
  vector<expression_t> exprs;
  accs.push_back(BAGUNION);
  exprs.push_back(outputExpr);
  opt::Reduce reduce =
      opt::Reduce(accs, exprs, predicate, &nest, &ctx, flushResults, testLabel);
  nest.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgReserves->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(OutputTest, MultiNestReservesStaticAlloc) {
  const char *testLabel = "multinestReserves.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();

  /**
   * SCAN RESERVES
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

  /*
   * NEST
   */

  /* Reserves: fields for materialization etc. */
  RecordAttribute *reservesOID =
      new RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
  list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
  reserveAttsForArg.push_back(*reservesOID);
  reserveAttsForArg.push_back(*sidReserves);

  /* constructing recType */
  list<RecordAttribute *> reserveAttsForRec{reservesOID, sidReserves};
  RecordType reserveRecType(reserveAttsForRec);

  expressions::InputArgument reservesArg{&reserveRecType, 1};
  auto reservesOIDProj = reservesArg[*reservesOID];
  auto reservesSIDProj = reservesArg[*sidReserves];
  auto reservesBIDProj = reservesArg[*bidReserves];
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  exprsToMatReserves.push_back(reservesSIDProj);
  exprsToMatReserves.push_back(reservesBIDProj);
  Materializer *matReserves = new Materializer(exprsToMatReserves);

  /* group-by expr */
  auto f = reservesArg[*sidReserves];
  /* null handling */
  auto g = reservesArg[*sidReserves];

  /* output of nest */
  vector<Monoid> accsNest{SUM, MAX};
  vector<expression_t> exprsNest{1, reservesBIDProj};
  vector<string> aggrLabels{"_groupCount", "_groupMax"};

  char nestLabel[] = "nest_reserves";
  radix::Nest nest = radix::Nest(&ctx, accsNest, exprsNest, aggrLabels, true, f,
                                 g, &scanReserves, nestLabel, *matReserves);
  scanReserves.setParent(&nest);

  /* REDUCE */
  const char *outLabel = "output";
  RecordAttribute *newCnt =
      new RecordAttribute(1, outLabel, string("_outCount"), intType);
  RecordAttribute *newMax =
      new RecordAttribute(2, outLabel, string("_outMax"), intType);
  list<RecordAttribute *> newAttrTypes = list<RecordAttribute *>();
  newAttrTypes.push_back(newCnt);
  newAttrTypes.push_back(newMax);
  RecordType newRecType = RecordType(newAttrTypes);

  RecordAttribute *cnt =
      new RecordAttribute(1, nestLabel, string("_groupCount"), intType);
  RecordAttribute *max =
      new RecordAttribute(2, nestLabel, string("_groupMax"), intType);

  expressions::InputArgument arg{&newRecType, 0};

  expressions::AttributeConstruction constr1{"_outCount", arg[*cnt]};
  expressions::AttributeConstruction constr2{"_outMax", arg[*max]};

  expression_t newRec = expressions::RecordConstruction{{constr1, constr2}};
  newRec.registerAs(nestLabel, "_rec");

  opt::Reduce reduce({BAGUNION}, {newRec}, true, &nest, &ctx, flushResults,
                     testLabel);
  nest.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgReserves->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(OutputTest, MultiNestReservesDynAlloc) {
  const char *testLabel = "multinestReserves.json";
  ParallelContext &ctx = *prepareContext(testLabel);

  PrimitiveType *intType = new IntType();
  PrimitiveType *stringType = new StringType();

  /**
   * SCAN RESERVES
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

  /*
   * NEST
   */

  /* Reserves: fields for materialization etc. */
  RecordAttribute *reservesOID =
      new RecordAttribute(reservesPath, activeLoop, pgReserves->getOIDType());
  list<RecordAttribute> reserveAttsForArg = list<RecordAttribute>();
  reserveAttsForArg.push_back(*reservesOID);
  reserveAttsForArg.push_back(*sidReserves);

  /* constructing recType */
  list<RecordAttribute *> reserveAttsForRec = list<RecordAttribute *>();
  reserveAttsForRec.push_back(reservesOID);
  reserveAttsForRec.push_back(sidReserves);
  RecordType reserveRecType = RecordType(reserveAttsForRec);

  expressions::InputArgument reservesArg{&reserveRecType, 1};
  auto reservesOIDProj = reservesArg[*reservesOID];
  auto reservesSIDProj = reservesArg[*sidReserves];
  auto reservesBIDProj = reservesArg[*bidReserves];
  vector<expression_t> exprsToMatReserves;
  exprsToMatReserves.push_back(reservesOIDProj);
  exprsToMatReserves.push_back(reservesSIDProj);
  exprsToMatReserves.push_back(reservesBIDProj);
  Materializer *matReserves = new Materializer(exprsToMatReserves);

  /* group-by expr */
  auto f = reservesSIDProj;
  /* null handling */
  auto g = reservesSIDProj;

  /* output of nest */
  vector<Monoid> accsNest{SUM, MAX};
  vector<expression_t> exprsNest{1, reservesBIDProj};
  vector<string> aggrLabels{"_groupCount", "_groupMax"};

  char nestLabel[] = "nest_reserves";
  radix::Nest nest = radix::Nest(&ctx, accsNest, exprsNest, aggrLabels, true, f,
                                 g, &scanReserves, nestLabel, *matReserves);
  scanReserves.setParent(&nest);

  /* REDUCE */
  const char *outLabel = "output";
  RecordAttribute *newCnt =
      new RecordAttribute(1, outLabel, string("_outCount"), intType);
  RecordAttribute *newMax =
      new RecordAttribute(2, outLabel, string("_outMax"), intType);
  list<RecordAttribute *> *newAttrTypes = new list<RecordAttribute *>();
  newAttrTypes->push_back(newCnt);
  newAttrTypes->push_back(newMax);
  RecordType *newRecType = new RecordType(*newAttrTypes);

  RecordAttribute cnt{1, nestLabel, string("_groupCount"), intType};
  RecordAttribute max{2, nestLabel, string("_groupMax"), intType};

  expressions::InputArgument arg{newRecType, 0};
  auto outputExpr1 = arg[cnt];
  auto outputExpr2 = arg[max];

  expressions::AttributeConstruction constr1{"_outCount", outputExpr1};
  expressions::AttributeConstruction constr2{"_outMax", outputExpr2};
  list<expressions::AttributeConstruction> newAtts{constr1, constr2};

  expressions::RecordConstruction newRec{newAtts};

  opt::Reduce reduce({BAGUNION}, {newRec}, true, &nest, &ctx, flushResults,
                     testLabel);
  nest.setParent(&reduce);
  reduce.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pgReserves->finish();

  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}
