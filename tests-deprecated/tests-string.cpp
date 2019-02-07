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

#include "gtest/gtest.h"

#include "common/common.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "operators/join.hpp"
#include "operators/nest-opt.hpp"
#include "operators/nest.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/radix-join.hpp"
#include "operators/radix-nest.hpp"
#include "operators/reduce-nopred.hpp"
#include "operators/reduce.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/binary-col-plugin.hpp"
#include "plugins/binary-row-plugin.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "util/caching.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

/*
 * Only including Radix variations
 * They are the only ones that perform both 'filtering' and 'refinement'
 * of matches (i.e., both hashKey and key checked) */

TEST(JSMN, String) {
  Context &ctx = *prepareContext("jsmnStringIngestion");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/jsmn-string.json");

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

  jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(field1);
  projections.push_back(field2);

  expressions::Expression *lhsArg =
      new expressions::InputArgument(&documentType, 0, projections);
  expressions::Expression *lhs =
      new expressions::RecordProjection(&stringType, lhsArg, field1);
  string neededName = string("Harry");
  expressions::Expression *rhs = new expressions::StringConstant(neededName);

  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  Select sel = Select(predicate, &scan);
  scan.setParent(&sel);

  /**
   * PRINT
   */
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&intType, lhsArg, field2);
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

TEST(JSON, String) {
  Context &ctx = *prepareContext("jsonStringIngestion");
  Catalog &catalog = Catalog::getInstance();

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
  projections.push_back(field1);
  projections.push_back(field2);

  expressions::Expression *lhsArg =
      new expressions::InputArgument(&documentType, 0, projections);
  expressions::Expression *lhs =
      new expressions::RecordProjection(&stringType, lhsArg, field1);
  string neededName = string("Harry");
  expressions::Expression *rhs = new expressions::StringConstant(neededName);

  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  Select sel = Select(predicate, &scan);
  scan.setParent(&sel);

  /**
   * PRINT
   */
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&intType, lhsArg, field2);
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

TEST(CSV, String) {
  Context &ctx = *prepareContext("csvStringIngestion");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/csv/csv-string.csv");

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
  vector<RecordAttribute *> whichFields;
  whichFields.push_back(&field1);
  whichFields.push_back(&field2);

  pm::CSVPlugin pg = pm::CSVPlugin(&ctx, fname, rec, whichFields, 3, 2);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(field1);
  projections.push_back(field2);

  expressions::Expression *lhsArg =
      new expressions::InputArgument(&documentType, 0, projections);
  expressions::Expression *lhs =
      new expressions::RecordProjection(&stringType, lhsArg, field1);
  string neededName = string("Harry");
  expressions::Expression *rhs = new expressions::StringConstant(neededName);

  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  Select sel = Select(predicate, &scan);
  scan.setParent(&sel);

  /**
   * PRINT
   */
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&intType, lhsArg, field2);
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
