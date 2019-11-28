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

#include "common/common.hpp"
#include "engines/olap/util/context.hpp"
#include "engines/olap/util/functions.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "gtest/gtest.h"
#include "operators/flush.hpp"
#include "operators/join.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/relbuilder.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "test-utils.hpp"
#include "values/expressionTypes.hpp"

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class JSONTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    catalog = &Catalog::getInstance();
    caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  virtual void TearDown() {}

  jsonPipelined::JSONPlugin *openJSON(Context *const context, string &fname,
                                      ExpressionType *schema,
                                      size_t linehint = 1000) {
    jsonPipelined::JSONPlugin *plugin =
        new jsonPipelined::JSONPlugin(context, fname, schema, linehint);
    catalog->registerPlugin(fname, plugin);

    return plugin;
  }

  jsonPipelined::JSONPlugin *openJSON(Context *const context, string &fname,
                                      ExpressionType *schema, size_t linehint,
                                      jsmntok_t **tokens) {
    jsonPipelined::JSONPlugin *plugin =
        new jsonPipelined::JSONPlugin(context, fname, schema, linehint, tokens);
    catalog->registerPlugin(fname, plugin);
    return plugin;
  }

  bool reduceJSONMaxFlatCached(bool longRun, int lineHint, string fname,
                               jsmntok_t **tokens);

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-json/";

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

  bool executePlan(PreparedStatement &stmt, const char *testLabel,
                   std::vector<Plugin *> pgs) {
    stmt.execute();

    for (auto &pg : pgs) pg->finish();

    bool res = verifyTestResult(testPath, testLabel, true);
    shm_unlink(testLabel);
    return res;
  }

 private:
  Catalog *catalog;
  CachingService *caches;
};

TEST_F(JSONTest, String) {
  const char *testLabel = "String";
  auto &ctx = *prepareContext(testLabel);

  string fname = string("inputs/json/json-string.json");

  IntType intType = IntType();
  StringType stringType = StringType();

  string name = string("name");
  RecordAttribute field1 = RecordAttribute(1, fname, name, &stringType);
  string age = string("age");
  RecordAttribute field2 = RecordAttribute(2, fname, age, &intType);
  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&field1);
  atts.push_back(&field2);

  RecordType rec = RecordType(atts);
  ListType documentType = ListType(rec);

  int linehint = 3;
  jsonPipelined::JSONPlugin *pg =
      openJSON(&ctx, fname, &documentType, linehint);
  Scan scan = Scan(&ctx, *pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg->getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(field1);
  projections.push_back(field2);

  expressions::InputArgument lhsArg{&rec, 0};
  auto predicate = eq(lhsArg[field1], "Harry");

  Select sel(predicate, &scan);
  scan.setParent(&sel);

  /**
   * PRINT
   */
  Flush printOp({lhsArg[field2]}, &sel, &ctx, testLabel);
  sel.setParent(&printOp);
  printOp.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

TEST_F(JSONTest, ScanJSON) {
  const char *testLabel = "scanJSON.json";
  auto &ctx = *prepareContext(testLabel);

  string fname("inputs/json/jsmn-flat.json");

  IntType attrType;
  RecordAttribute attr(1, fname, "a", &attrType);
  RecordAttribute attr2(2, fname, "b", &attrType);

  list<RecordAttribute *> atts{&attr, &attr2};

  RecordType inner(std::list<RecordAttribute *>{&attr, &attr2});
  ListType documentType = ListType(inner);

  jsonPipelined::JSONPlugin *pg = openJSON(&ctx, fname, &documentType);
  Scan scan(&ctx, *pg);

  /* Reduce */
  expressions::InputArgument lhsArg(&inner, 0);

  Flush flush{{lhsArg[attr]}, &scan, &ctx, testLabel};
  scan.setParent(&flush);
  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

TEST_F(JSONTest, SelectJSON) {
  const char *testLabel = "selectJSON.json";
  auto &ctx = *prepareContext(testLabel);

  string fname = string("inputs/json/jsmn-flat.json");

  IntType attrType;
  RecordAttribute attr(1, fname, "a", &attrType);
  RecordAttribute attr2(2, fname, "b", &attrType);

  RecordType inner(std::list<RecordAttribute *>{&attr, &attr2});
  ListType documentType = ListType(inner);

  jsonPipelined::JSONPlugin *pg = openJSON(&ctx, fname, &documentType);
  Scan scan = Scan(&ctx, *pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple(fname, activeLoop, pg->getOIDType());

  expressions::InputArgument lhsArg(&inner, 0);
  Select sel{gt(lhsArg[attr2], 5), &scan};
  scan.setParent(&sel);

  /* Reduce */
  Flush flush{{lhsArg[attr]}, &sel, &ctx, testLabel};
  sel.setParent(&flush);
  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

TEST_F(JSONTest, unnestJSON) {
  const char *testLabel = "unnestJSONEmployees.json";
  auto &ctx = *prepareContext(testLabel);

  string fname("inputs/json/employees-flat.json");

  IntType intType;
  StringType stringType;

  string childName("name");
  RecordAttribute child1(1, fname, childName, &stringType);
  string childAge("age2");
  RecordAttribute child2(2, fname, childAge, &intType);
  list<RecordAttribute *> attsNested{&child1};
  RecordType nested(attsNested);
  ListType nestedCollection(nested);

  string empName("name");
  RecordAttribute emp1(1, fname, empName, &stringType);
  string empAge("age");
  RecordAttribute emp2(2, fname, empAge, &intType);
  string empChildren("children");
  RecordAttribute emp3(3, fname, empChildren, &nestedCollection);

  list<RecordAttribute *> atts{&emp1, &emp2, &emp3};

  RecordType inner(atts);
  ListType documentType(inner);

  jsonPipelined::JSONPlugin *pg = openJSON(&ctx, fname, &documentType);
  Scan scan(&ctx, *pg);

  expressions::InputArgument inputArg(&inner, 0);
  string nestedName = "c";
  auto proj = inputArg[emp3];
  Path path(nestedName, &proj);

  Unnest unnestOp(true, path, &scan);
  scan.setParent(&unnestOp);

  // New record type:
  string originalRecordName = "e";
  RecordAttribute recPrev(1, fname, originalRecordName, &inner);
  RecordAttribute recUnnested(2, fname, nestedName, &nested);
  list<RecordAttribute *> attsUnnested{&recPrev, &recUnnested};
  RecordType unnestedType(attsUnnested);

  expressions::InputArgument nestedArg(&unnestedType, 0);

  RecordAttribute toPrint(-1, fname + "." + empChildren, childAge, &intType);

  Flush flush({nestedArg[toPrint]}, &unnestOp, &ctx, testLabel);
  unnestOp.setParent(&flush);

  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

/* json plugin seems broken if linehint not provided */
TEST_F(JSONTest, reduceListObjectFlat) {
  const char *testLabel = "jsonFlushList.json";
  auto &ctx = *prepareContext(testLabel);

  string fname("inputs/json/jsmnDeeper-flat.json");

  IntType intType;

  string c1Name("c1");
  RecordAttribute c1(1, fname, c1Name, &intType);
  string c2Name("c2");
  RecordAttribute c2(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested{&c1, &c2};
  RecordType nested(attsNested);

  string attrName("a");
  string attrName2("b");
  string attrName3("c");
  RecordAttribute attr(1, fname, attrName, &intType);
  RecordAttribute attr2(2, fname, attrName2, &intType);
  RecordAttribute attr3(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts{&attr, &attr2, &attr3};

  RecordType inner(atts);
  ListType documentType(inner);
  int linehint = 10;
  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin *pg =
      openJSON(&ctx, fname, &documentType, linehint);
  Scan scan(&ctx, *pg);

  expressions::InputArgument arg{&inner, 0};

  Select sel{gt(arg[attr2], 43), &scan};
  scan.setParent(&sel);

  Flush flush{{arg[attr3]}, &sel, &ctx, testLabel};
  sel.setParent(&flush);

  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

/* SELECT MAX(obj.b) FROM jsonFile obj WHERE obj.b  > 43 */
TEST_F(JSONTest, reduceMax) {
  const char *testLabel = "reduceJSONMax.json";
  bool longRun = false;
  auto &ctx = *prepareContext(testLabel);

  string fname;
  size_t lineHint;
  if (!longRun) {
    fname = string("inputs/json/jsmnDeeper-flat.json");
    lineHint = 10;
  } else {
    //        fname = string("inputs/json/jsmnDeeper-flat1k.json");
    //        lineHint = 1000;

    //        fname = string("inputs/json/jsmnDeeper-flat100k.json");
    //        lineHint = 100000;

    //        fname = string("inputs/json/jsmnDeeper-flat1m.json");
    //        lineHint = 1000000;

    fname = string("inputs/json/jsmnDeeper-flat100m.json");
    lineHint = 100000000;

    //        fname = string("inputs/json/jsmnDeeper-flat200m.json");
    //        lineHint = 200000000;
  }

  std::string outRel{"output"};
  {
    RecordType r{};
    Plugin *newPg = new pm::CSVPlugin(&ctx, outRel, r, {}, ',', 10, 1, false);
    Catalog::getInstance().registerPlugin(*(new string(outRel)), newPg);
  }

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName("a");
  string attrName2("b");
  string attrName3("c");
  RecordAttribute attr(1, fname, attrName, &intType);
  RecordAttribute attr2(2, fname, attrName2, &intType);
  RecordAttribute attr3(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts{&attr, &attr2, &attr3};

  RecordType inner(atts);
  ListType documentType(inner);

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin *pg =
      openJSON(&ctx, fname, &documentType, lineHint);
  Scan scan(&ctx, *pg);

  /**
   * REDUCE
   */
  expressions::InputArgument arg{&inner, 0};

  RecordAttribute recOut{1, outRel, "b", &intType};
  opt::Reduce reduce{
      {MAX}, {arg[attr2].as(&recOut)}, gt(arg[attr2], 43), &scan, &ctx};
  scan.setParent(&reduce);

  RecordType outRec{list<RecordAttribute *>{&recOut}};
  expressions::InputArgument argout{&outRec, 0};
  Flush flush{{argout[recOut]}, &reduce, &ctx, testLabel};
  reduce.setParent(&flush);
  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {}));

  {
    const char *testLabel = "reduceJSONCached.json";
    auto &ctx = *prepareContext(testLabel);
    std::string outRel{"output2"};
    {
      RecordType r{};
      Plugin *newPg = new pm::CSVPlugin(&ctx, outRel, r, {}, ',', 10, 1, false);
      Catalog::getInstance().registerPlugin(*(new string(outRel)), newPg);
    }

    /**
     * SCAN
     */
    jsonPipelined::JSONPlugin *pgCached =
        openJSON(&ctx, fname, &documentType, lineHint, pg->getTokens());
    Scan scan(&ctx, *pgCached);

    /**
     * REDUCE
     */
    expressions::InputArgument arg{&inner, 0};

    RecordAttribute recOut{1, outRel, "b", &intType};
    opt::Reduce reduce{
        {MAX}, {arg[attr2].as(&recOut)}, gt(arg[attr2], 43), &scan, &ctx};
    scan.setParent(&reduce);

    RecordType outRec{list<RecordAttribute *>{&recOut}};
    expressions::InputArgument argout{&outRec, 0};
    Flush flush{{argout[recOut]}, &reduce, &ctx, testLabel};

    reduce.setParent(&flush);
    flush.produce();

    EXPECT_TRUE(executePlan(ctx, testLabel, {pgCached}));
  }
  pg->finish();
}

/* SELECT MAX(obj.c.c2) FROM jsonFile obj WHERE obj.b  > 43 */
TEST_F(JSONTest, reduceDeeperMax) {
  bool longRun = false;
  const char *testLabel = "reduceDeeperMax.json";
  auto &ctx = *prepareContext(testLabel);

  string fname;
  size_t lineHint;
  if (!longRun) {
    fname = string("inputs/json/jsmnDeeper-flat.json");
    lineHint = 10;
  } else {
    //        fname = string("inputs/json/jsmnDeeper-flat1m.json");
    //        lineHint = 1000000;

    fname = string("inputs/json/jsmnDeeper-flat100m.json");
    lineHint = 100000000;
  }

  std::string outRel{"output"};
  {
    RecordType r{};
    Plugin *newPg = new pm::CSVPlugin(&ctx, outRel, r, {}, ',', 10, 1, false);
    Catalog::getInstance().registerPlugin(*(new string(outRel)), newPg);
  }

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested{attsNested};

  string attrName{"a"};
  string attrName2{"b"};
  string attrName3{"c"};
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts{&attr, &attr2, &attr3};

  RecordType inner{atts};
  ListType documentType{inner};

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin *pg =
      openJSON(&ctx, fname, &documentType, lineHint);
  Scan scan = Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple(fname, activeLoop, pg->getOIDType());
  list<RecordAttribute> projection{projTuple, attr2, attr3};

  RecordAttribute recOut(outRel, "c2", &intType);

  expressions::InputArgument arg(&inner, 0);
  auto outputExpr_ = arg[attr3];
  auto outputExpr = expression_t{outputExpr_}[c2].as(&recOut);

  opt::Reduce reduce{{MAX}, {outputExpr}, gt(arg[attr2], 43), &scan, &ctx};
  scan.setParent(&reduce);

  RecordType outRec{list<RecordAttribute *>{&recOut}};
  expressions::InputArgument argout{&outRec, 0};
  Flush flush{{argout[recOut]}, &reduce, &ctx, testLabel};
  reduce.setParent(&flush);

  flush.produce();

  EXPECT_TRUE(executePlan(ctx, testLabel, {pg}));
}

/* SELECT age, age2 FROM employeesnum e, UNNEST (e.children); */
TEST_F(JSONTest, jsonRelBuilder) {
  const char *testLabel = "jsonRelBuilder.json";
  auto &ctx = *prepareContext(testLabel);

  string fname{"inputs/json/employees-numeric-only.json"};
  size_t lineHint = 3;

  {
    std::string outRel{"output"};
    RecordType r{};
    Plugin *newPg = new pm::CSVPlugin(&ctx, outRel, r, {}, ',', 10, 1, false);
    Catalog::getInstance().registerPlugin(*(new string(outRel)), newPg);
  }

  IntType intType;

  string c2Name("height2");
  RecordAttribute height2(1, fname + ".children", c2Name, &intType);
  string c1Name("age2");
  RecordAttribute age2(2, fname + ".children", c1Name, &intType);
  list<RecordAttribute *> attsNested{&age2, &height2};
  RecordType nested{attsNested};

  string attrName{"height"};
  string attrName2{"age"};
  string attrName3{"children"};
  RecordAttribute attr{1, fname, attrName, &intType};
  RecordAttribute attr2{2, fname, attrName2, &intType};
  RecordAttribute attr3{3, fname, attrName3, &nested};

  RecordAttribute nestedAs{5, fname, "c", &nested};

  list<RecordAttribute *> atts{&attr, &attr2, &attr3};

  RecordType inner{atts};
  ListType documentType{inner};

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin *pg =
      openJSON(&ctx, fname, &documentType, lineHint);

  auto stmnt = RelBuilder{&ctx}
                   .scan(*pg)
                   .unnest([&](const auto &arg) -> expression_t {
                     return arg[attr3].as(&nestedAs);
                   })
                   .print([&](const auto &arg,
                              std::string outrel) -> std::vector<expression_t> {
                     return {arg[attr2].as(outrel, attr2.getAttrName()),
                             arg[age2].as(outrel, age2.getAttrName())};
                   })
                   .prepare();

  EXPECT_TRUE(executePlan(stmnt, testLabel, {pg}));
}
