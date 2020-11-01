/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

#include <platform/common/common.hpp>

#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "gtest/gtest.h"
#include "operators/join.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/reduce.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

TEST(JSON, ScanJSON) {
  Context &ctx = *prepareContext("ScanJSON-flat");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/jsmn-flat.json");

  string attrName = string("a");
  string attrName2 = string("b");
  IntType attrType = IntType();
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &attrType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &attrType);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);

  Scan scan = Scan(&ctx, pg);

  // ROOT
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

TEST(JSON, SelectJSON) {
  Context &ctx = *prepareContext("SelectJSON-flat");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/jsmn-flat.json");

  string attrName = string("a");
  string attrName2 = string("b");
  IntType attrType = IntType();
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &attrType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &attrType);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);

  Scan scan = Scan(&ctx, pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr);
  projections.push_back(attr2);

  expressions::Expression *lhsArg =
      new expressions::InputArgument(&attrType, 0, projections);
  expressions::Expression *lhs =
      new expressions::RecordProjection(&attrType, lhsArg, attr2);
  expressions::Expression *rhs = new expressions::IntConstant(5);

  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);

  Select sel = Select(predicate, &scan);
  scan.setParent(&sel);

  /**
   * PRINT
   */
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&attrType, lhsArg, attr);
  Print printOp = Print(debugInt, proj, &sel);
  sel.setParent(&printOp);

  /**
   * ROOT
   */
  Root rootOp = Root(&printOp);
  printOp.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

TEST(JSON, unnestJSON) {
  Context &ctx = *prepareContext("unnestJSONFlat");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/employees-flat.json");

  IntType intType = IntType();
  StringType stringType = StringType();

  string childName = string("name");
  RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
  string childAge = string("age");
  RecordAttribute child2 = RecordAttribute(2, fname, childAge, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&child1);
  RecordType nested = RecordType(attsNested);
  ListType nestedCollection = ListType(nested);

  string empName = string("name");
  RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
  string empAge = string("age");
  RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
  string empChildren = string("children");
  RecordAttribute emp3 =
      RecordAttribute(3, fname, empChildren, &nestedCollection);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&emp1);
  atts.push_back(&emp2);
  atts.push_back(&emp3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  RecordAttribute proj1 =
      RecordAttribute(fname, empChildren, &nestedCollection);
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(proj1);

  expressions::Expression *inputArg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&nestedCollection, inputArg, emp3);
  string nestedName = "c";
  Path path = Path(nestedName, proj);

  expressions::Expression *lhs = new expressions::BoolConstant(true);
  expressions::Expression *rhs = new expressions::BoolConstant(true);
  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  Unnest unnestOp = Unnest(predicate, path, &scan);
  scan.setParent(&unnestOp);

  // New record type:
  string originalRecordName = "e";
  RecordAttribute recPrev =
      RecordAttribute(1, fname, originalRecordName, &inner);
  RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName, &nested);
  list<RecordAttribute *> attsUnnested = list<RecordAttribute *>();
  attsUnnested.push_back(&recPrev);
  attsUnnested.push_back(&recUnnested);
  RecordType unnestedType = RecordType(attsUnnested);

  // PRINT
  // a bit redundant, but 'new record construction can, in principle, cause new
  // aliases
  projections.push_back(recPrev);
  projections.push_back(recUnnested);
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::Expression *nestedArg =
      new expressions::InputArgument(&unnestedType, 0, projections);

  RecordAttribute toPrint =
      RecordAttribute(-1, fname + "." + empChildren, childAge, &intType);

  expressions::RecordProjection *projToPrint =
      new expressions::RecordProjection(&intType, nestedArg, toPrint);
  Print printOp = Print(debugInt, projToPrint, &unnestOp);
  unnestOp.setParent(&printOp);

  // ROOT
  Root rootOp = Root(&printOp);
  printOp.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

/* json plugin seems broken if linehint not provided */
TEST(JSON, reduceListObjectFlat) {
  Context &ctx = *prepareContext("Reduce-FlushListObject");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/jsmnDeeper-flat.json");

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName = string("a");
  string attrName2 = string("b");
  string attrName3 = string("c");
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);
  atts.push_back(&attr3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);
  int linehint = 10;
  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType, linehint);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr2);
  projections.push_back(attr3);

  expressions::Expression *arg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(&nested, arg, attr3);

  expressions::Expression *lhs =
      new expressions::RecordProjection(&intType, arg, attr2);
  expressions::Expression *rhs = new expressions::IntConstant(43.0);
  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);
  Reduce reduce = Reduce(UNION, outputExpr, predicate, &scan, &ctx);
  scan.setParent(&reduce);

  reduce.produce();
  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

void reduceJSONMaxFlatCached(bool longRun, int lineHint, string fname,
                             jsmntok_t **tokens) {
  Context &ctx = *prepareContext("Reduce-JSONMax");
  Catalog &catalog = Catalog::getInstance();

  cout << "Input: " << fname << endl;

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName = string("a");
  string attrName2 = string("b");
  string attrName3 = string("c");
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);
  atts.push_back(&attr3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType, lineHint, tokens);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr2);
  projections.push_back(attr3);

  expressions::Expression *arg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(&intType, arg, attr2);

  expressions::Expression *lhs =
      new expressions::RecordProjection(&intType, arg, attr2);
  expressions::Expression *rhs = new expressions::IntConstant(43.0);
  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);
  Reduce reduce = Reduce(MAX, outputExpr, predicate, &scan, &ctx);
  scan.setParent(&reduce);

  reduce.produce();
  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

/* SELECT MAX(obj.b) FROM jsonFile obj WHERE obj.b  > 43 */ TEST(JSON,
                                                                 reduceMax) {
  bool longRun = false;
  Context &ctx = *prepareContext("Reduce-JSONMax");
  Catalog &catalog = Catalog::getInstance();

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
  cout << "Input: " << fname << endl;

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName = string("a");
  string attrName2 = string("b");
  string attrName3 = string("c");
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);
  atts.push_back(&attr3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType, lineHint);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr2);
  projections.push_back(attr3);

  expressions::Expression *arg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(&intType, arg, attr2);

  expressions::Expression *lhs =
      new expressions::RecordProjection(&intType, arg, attr2);
  expressions::Expression *rhs = new expressions::IntConstant(43.0);
  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);
  Reduce reduce = Reduce(MAX, outputExpr, predicate, &scan, &ctx);
  scan.setParent(&reduce);

  reduce.produce();
  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  /**
   * CALL 2nd QUERY
   */
  reduceJSONMaxFlatCached(longRun, lineHint, fname, pg.getTokens());

  pg.finish();
  catalog.clear();
}

/* SELECT MAX(obj.c.c2) FROM jsonFile obj WHERE obj.b  > 43 */ TEST(
    JSON, reduceDeeperMax) {
  bool longRun = false;
  Context &ctx = *prepareContext("Reduce-DeeperJSONMax");
  Catalog &catalog = Catalog::getInstance();

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
  cout << "Input: " << fname << endl;

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName = string("a");
  string attrName2 = string("b");
  string attrName3 = string("c");
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);
  atts.push_back(&attr3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType, lineHint);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * REDUCE
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr2);
  projections.push_back(attr3);

  expressions::Expression *arg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::Expression *outputExpr_ =
      new expressions::RecordProjection(&nested, arg, attr3);
  expressions::Expression *outputExpr =
      new expressions::RecordProjection(&intType, outputExpr_, c2);

  expressions::Expression *lhs =
      new expressions::RecordProjection(&intType, arg, attr2);
  expressions::Expression *rhs = new expressions::IntConstant(43.0);
  expressions::Expression *predicate = new expressions::GtExpression(lhs, rhs);
  Reduce reduce = Reduce(MAX, outputExpr, predicate, &scan, &ctx);
  scan.setParent(&reduce);

  reduce.produce();
  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}
