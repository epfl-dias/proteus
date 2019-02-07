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

TEST(JSON, Nest) {
  Context &ctx = *prepareContext("nestRadixJSON");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/employees-more.json");

  IntType intType = IntType();
  // FloatType floatType = FloatType();
  StringType stringType = StringType();

  string childName = string("name");
  RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
  string childAge = string("age");
  RecordAttribute child2 = RecordAttribute(1, fname, childAge, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&child1);
  attsNested.push_back(&child2);
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

  /**
   * SCAN
   */
  jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * OUTER UNNEST
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  RecordAttribute proj1 =
      RecordAttribute(fname, empChildren, &nestedCollection);
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(proj1);
  expressions::Expression *inputArg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::RecordProjection *projChildren =
      new expressions::RecordProjection(&nestedCollection, inputArg, emp3);
  string nestedName = "c";
  Path path = Path(nestedName, projChildren);

  expressions::Expression *lhs = new expressions::BoolConstant(true);
  expressions::Expression *rhs = new expressions::BoolConstant(true);
  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  OuterUnnest unnestOp = OuterUnnest(predicate, path, &scan);
  scan.setParent(&unnestOp);

  // New record type:
  // XXX Makes no sense to come up with new bindingNames w/o having a way to
  // eval. them and ADD them in existing bindings!!!
  string originalRecordName = "e";
  RecordAttribute recPrev =
      RecordAttribute(1, fname, originalRecordName, &inner);
  RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName, &nested);
  list<RecordAttribute *> attsUnnested = list<RecordAttribute *>();
  attsUnnested.push_back(&recPrev);
  attsUnnested.push_back(&recUnnested);
  RecordType unnestedType = RecordType(attsUnnested);

  /**
   * NEST
   */
  // Output (e): SUM(children.age)
  // Have to model nested type too
  projections.push_back(recPrev);
  projections.push_back(recUnnested);
  expressions::Expression *nestedArg =
      new expressions::InputArgument(&unnestedType, 0, projections);
  RecordAttribute toAggr =
      RecordAttribute(-1, fname + "." + empChildren, childAge, &intType);
  expressions::RecordProjection *nestToAggr =
      new expressions::RecordProjection(&intType, nestedArg, toAggr);

  // Predicate (p): Ready from before

  // Grouping (f):
  /* Before (opt) */
  //    list<expressions::InputArgument> f;
  //    expressions::InputArgument f_arg = *(expressions::InputArgument*)
  //    inputArg; f.push_back(f_arg);
  /* After (radix) */
  expressions::RecordProjection *projName =
      new expressions::RecordProjection(&stringType, inputArg, emp1);
  expressions::Expression *f = projName;
  //    expressions::RecordProjection *projAge = new
  //    expressions::RecordProjection(
  //                    &intType, inputArg, emp2);
  //    expressions::Expression *f = projAge;
  // Specified inputArg

  // What to discard if null (g):
  // Ignoring for now

  // What to materialize (payload)
  // just currently active tuple ids should be enough

  vector<RecordAttribute *> whichFields;
  // Not added explicitly, bc noone materialized it before
  // whichFields.push_back(&emp2);
  vector<materialization_mode> outputModes;
  // outputModes.push_back(EAGER);

  Materializer *mat = new Materializer(whichFields, outputModes);

  char nestLabel[] = "nest_multiple";
  string aggrLabel = string(nestLabel);
  string aggrField1 = string("_aggrMax");
  string aggrField2 = string("_aggrSum");

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  /* Aggregate 1 */
  accs.push_back(MAX);
  outputExprs.push_back(nestToAggr);
  aggrLabels.push_back(aggrField1);
  /* Aggregate 2 */
  accs.push_back(SUM);
  outputExprs.push_back(nestToAggr);
  aggrLabels.push_back(aggrField2);

  radix::Nest nestOp = radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                   predicate, f, f, &unnestOp, nestLabel, *mat);
  unnestOp.setParent(&nestOp);

  // PRINT Field 1 (Max)
  llvm::Function *debugInt = ctx.getFunction("printi");
  RecordAttribute toOutput1 =
      RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestedArg, toOutput1);
  Print printOp1 = Print(debugInt, nestOutput1, &nestOp);
  nestOp.setParent(&printOp1);

  // PRINT Field 2 (Sum)
  RecordAttribute toOutput2 =
      RecordAttribute(2, aggrLabel, aggrField2, &intType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&intType, nestedArg, toOutput2);
  Print printOp2 = Print(debugInt, nestOutput2, &printOp1);
  printOp1.setParent(&printOp2);

  // ROOT
  Root rootOp = Root(&printOp2);
  printOp2.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

TEST(JSON, NestTwoKeys) {
  Context &ctx = *prepareContext("nestRadixJSON-TwoKeys");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/employees-more.json");

  IntType intType = IntType();
  // FloatType floatType = FloatType();
  StringType stringType = StringType();

  string childName = string("name");
  RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
  string childAge = string("age");
  RecordAttribute child2 = RecordAttribute(1, fname, childAge, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&child1);
  attsNested.push_back(&child2);
  RecordType nested = RecordType(attsNested);
  ListType nestedCollection = ListType(nested);

  string name = string("name");
  RecordAttribute empName = RecordAttribute(1, fname, name, &stringType);
  string age = string("age");
  RecordAttribute empAge = RecordAttribute(2, fname, age, &intType);
  string children = string("children");
  RecordAttribute empChildren =
      RecordAttribute(3, fname, children, &nestedCollection);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&empName);
  atts.push_back(&empAge);
  atts.push_back(&empChildren);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);

  /**
   * SCAN
   */
  jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * OUTER UNNEST
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  RecordAttribute proj1 = RecordAttribute(fname, children, &nestedCollection);
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(proj1);
  expressions::Expression *inputArg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::RecordProjection *projChildren =
      new expressions::RecordProjection(&nestedCollection, inputArg,
                                        empChildren);
  string nestedName = "c";
  Path path = Path(nestedName, projChildren);

  expressions::Expression *lhs = new expressions::BoolConstant(true);
  expressions::Expression *rhs = new expressions::BoolConstant(true);
  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  OuterUnnest unnestOp = OuterUnnest(predicate, path, &scan);
  scan.setParent(&unnestOp);

  // New record type:
  // XXX Makes no sense to come up with new bindingNames w/o having a way to
  // eval. them and ADD them in existing bindings!!!
  string originalRecordName = "e";
  RecordAttribute recPrev =
      RecordAttribute(1, fname, originalRecordName, &inner);
  RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName, &nested);
  list<RecordAttribute *> attsUnnested = list<RecordAttribute *>();
  attsUnnested.push_back(&recPrev);
  attsUnnested.push_back(&recUnnested);
  RecordType unnestedType = RecordType(attsUnnested);

  /**
   * NEST
   */
  // Output (e): SUM(children.age)
  // Have to model nested type too
  projections.push_back(recPrev);
  projections.push_back(recUnnested);
  expressions::Expression *nestedArg =
      new expressions::InputArgument(&unnestedType, 0, projections);
  RecordAttribute toAggr =
      RecordAttribute(-1, fname + "." + children, childAge, &intType);
  expressions::RecordProjection *nestToAggr =
      new expressions::RecordProjection(&intType, nestedArg, toAggr);

  // Predicate (p): Ready from before

  // Grouping (f):
  /* Before (opt) */
  //    list<expressions::InputArgument> f;
  //    expressions::InputArgument f_arg = *(expressions::InputArgument*)
  //    inputArg; f.push_back(f_arg);
  /* After (radix) */
  expressions::RecordProjection *projAge =
      new expressions::RecordProjection(&intType, inputArg, empAge);
  expressions::RecordProjection *projName =
      new expressions::RecordProjection(&stringType, inputArg, empName);

  list<RecordAttribute *> newAttsTypes = list<RecordAttribute *>();
  newAttsTypes.push_back(&empName);
  newAttsTypes.push_back(&empAge);
  RecordType newRecType = RecordType(newAttsTypes);

  expressions::AttributeConstruction attrExpr1 =
      expressions::AttributeConstruction("nameGroup", projName);
  expressions::AttributeConstruction attrExpr2 =
      expressions::AttributeConstruction("ageGroup", projAge);
  list<expressions::AttributeConstruction> newAtts =
      list<expressions::AttributeConstruction>();
  newAtts.push_back(attrExpr1);
  newAtts.push_back(attrExpr2);
  expressions::RecordConstruction *recGroup =
      new expressions::RecordConstruction(&newRecType, newAtts);

  //    expressions::Expression *f = projAge;
  //    expressions::Expression *f = projName;
  expressions::Expression *f = recGroup;
  // Specified inputArg

  // What to discard if null (g):
  // Ignoring for now

  // What to materialize (payload)
  // just currently active tuple ids should be enough

  vector<RecordAttribute *> whichFields;
  // Not added explicitly, bc noone materialized it before
  // whichFields.push_back(&empAge);
  vector<materialization_mode> outputModes;
  // outputModes.push_back(EAGER);

  Materializer *mat = new Materializer(whichFields, outputModes);

  char nestLabel[] = "nest_multiple";
  string aggrLabel = string(nestLabel);
  string aggrField1 = string("_aggrMax");
  string aggrField2 = string("_aggrSum");

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  /* Aggregate 1 */
  accs.push_back(MAX);
  outputExprs.push_back(nestToAggr);
  aggrLabels.push_back(aggrField1);
  /* Aggregate 2 */
  accs.push_back(SUM);
  outputExprs.push_back(nestToAggr);
  aggrLabels.push_back(aggrField2);

  radix::Nest nestOp = radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                   predicate, f, f, &unnestOp, nestLabel, *mat);
  unnestOp.setParent(&nestOp);

  // PRINT Field 1 (Max)
  llvm::Function *debugInt = ctx.getFunction("printi");
  RecordAttribute toOutput1 =
      RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestedArg, toOutput1);
  Print printOp1 = Print(debugInt, nestOutput1, &nestOp);
  nestOp.setParent(&printOp1);

  // PRINT Field 2 (Sum)
  RecordAttribute toOutput2 =
      RecordAttribute(2, aggrLabel, aggrField2, &intType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&intType, nestedArg, toOutput2);
  Print printOp2 = Print(debugInt, nestOutput2, &printOp1);
  printOp1.setParent(&printOp2);

  // ROOT
  Root rootOp = Root(&printOp2);
  printOp2.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}
