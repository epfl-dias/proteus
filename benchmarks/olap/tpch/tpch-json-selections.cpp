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
#include "common/tpch-config.hpp"
#include "engines/olap/util/caching.hpp"
#include "engines/olap/util/context.hpp"
#include "engines/olap/util/functions.hpp"
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
#include "operators/reduce-opt.hpp"
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
#include "values/expressionTypes.hpp"

void tpchSchema(map<string, dataset> &datasetCatalog) {
  tpchSchemaJSON(datasetCatalog);
}

/**
 * Selections
 * All should have the same result - what changes is the order of the predicates
 * and whether some are applied via a conjunction
 */
// 1-2-3-4
void tpchOrderSelection1(map<string, dataset> datasetCatalog,
                         vector<int> predicates);
/* Only use the rest in the case of 4 predicates! */
// 4-3-2-1
void tpchOrderSelection2(map<string, dataset> datasetCatalog,
                         vector<int> predicates);
//(1-2),(3-4)
void tpchOrderSelection3(map<string, dataset> datasetCatalog,
                         vector<int> predicates);
void tpchOrderSelection4(map<string, dataset> datasetCatalog,
                         vector<int> predicates);

int main() {
  map<string, dataset> datasetCatalog;
  tpchSchema(datasetCatalog);

  /* pred1 will be the one dictating query selectivity*/
  int pred1 = L_ORDERKEY_MAX;
  int pred2 = (int)L_QUANTITY_MAX;
  int pred3 = L_LINENUMBER_MAX;
  int pred4 = (int)L_EXTENDEDPRICE_MAX;

  for (int i = 0; i < 5; i++) {
    cout << "[tpch-json-selections: ] Run " << i + 1 << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(pred1 * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      vector<int> predicates;
      predicates.push_back(predicateVal);
      cout << "Query 0 (PM built if applicable)" << endl;
      tpchOrderSelection1(datasetCatalog, predicates);
      cout << "---" << endl;
      // 1 pred.
      cout << "Query 1a" << endl;
      tpchOrderSelection1(datasetCatalog, predicates);
      cout << "---" << endl;
      cout << "Query 1b" << endl;
      predicates.push_back(pred2);
      // 2 pred.
      tpchOrderSelection1(datasetCatalog, predicates);
      cout << "---" << endl;
      cout << "Query 1c" << endl;
      predicates.push_back(pred3);
      // 3 pred.
      tpchOrderSelection1(datasetCatalog, predicates);
      cout << "---" << endl;
      cout << "Query 1d" << endl;
      // 4 pred.
      predicates.push_back(pred4);
      tpchOrderSelection1(datasetCatalog, predicates);
      // Variations of last execution
      cout << "---" << endl;
      cout << "Query 2" << endl;
      tpchOrderSelection2(datasetCatalog, predicates);
      cout << "---" << endl;
      cout << "Query 3" << endl;
      tpchOrderSelection3(datasetCatalog, predicates);
      cout << "---" << endl;
      cout << "Query 4" << endl;
      tpchOrderSelection4(datasetCatalog, predicates);
      cout << "---" << endl;
    }
  }
}

void tpchOrderSelection1(map<string, dataset> datasetCatalog,
                         vector<int> predicates) {
  int predicatesNo = predicates.size();
  if (predicatesNo <= 0 || predicatesNo > 4) {
    throw runtime_error(string("Invalid no. of predicates requested: "));
  }
  Context &ctx = *prepareContext("tpch-csv-selection1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = lineitem.path;
  RecordType rec = lineitem.recType;
  int linehint = lineitem.linehint;
  RecordAttribute *orderkey = argsLineitem["orderkey"];
  RecordAttribute *linenumber = argsLineitem["linenumber"];
  RecordAttribute *quantity = argsLineitem["quantity"];
  RecordAttribute *extendedprice = argsLineitem["extendedprice"];

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT(S)
   * 1 to 4
   *
   * Lots of repetition..
   */
  Operator *lastSelectOp = nullptr;
  list<RecordAttribute> argProjections;
  if (predicatesNo == 1) {
    argProjections.push_back(*orderkey);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        orderkey->getOriginalType(), arg, *orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);
    lastSelectOp = sel1;
  } else if (predicatesNo == 2) {
    argProjections.push_back(*orderkey);
    argProjections.push_back(*quantity);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        orderkey->getOriginalType(), arg, *orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        quantity->getOriginalType(), arg, *quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    lastSelectOp = sel2;
  } else if (predicatesNo == 3) {
    argProjections.push_back(*orderkey);
    argProjections.push_back(*linenumber);
    argProjections.push_back(*quantity);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        orderkey->getOriginalType(), arg, *orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        quantity->getOriginalType(), arg, *quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        linenumber->getOriginalType(), arg, *linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);
    lastSelectOp = sel3;

  } else if (predicatesNo == 4) {
    argProjections.push_back(*orderkey);
    argProjections.push_back(*linenumber);
    argProjections.push_back(*quantity);
    argProjections.push_back(*extendedprice);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        orderkey->getOriginalType(), arg, *orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        quantity->getOriginalType(), arg, *quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        linenumber->getOriginalType(), arg, *linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);

    expressions::Expression *lhs4 = new expressions::RecordProjection(
        extendedprice->getOriginalType(), arg, *extendedprice);
    expressions::Expression *rhs4 =
        new expressions::FloatConstant(predicates.at(3));
    expressions::Expression *pred4 = new expressions::LtExpression(lhs4, rhs4);

    Select *sel4 = new Select(pred4, sel3);
    sel3->setParent(sel4);

    lastSelectOp = sel4;
  }

  /**
   * REDUCE
   * COUNT(*)
   */
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
  lastSelectOp->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

void tpchOrderSelection2(map<string, dataset> datasetCatalog,
                         vector<int> predicates) {
  int predicatesNo = predicates.size();
  if (predicatesNo != 4) {
    throw runtime_error(string("Invalid no. of predicates requested: "));
  }
  Context &ctx = *prepareContext("tpch-csv-selection2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = lineitem.path;
  RecordType rec = lineitem.recType;
  int linehint = lineitem.linehint;
  RecordAttribute *orderkey = argsLineitem["orderkey"];
  RecordAttribute *linenumber = argsLineitem["linenumber"];
  RecordAttribute *quantity = argsLineitem["quantity"];
  RecordAttribute *extendedprice = argsLineitem["extendedprice"];

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);
  /**
   * SELECT(S)
   */
  Operator *lastSelectOp;
  list<RecordAttribute> argProjections;
  argProjections.push_back(*orderkey);
  argProjections.push_back(*linenumber);
  argProjections.push_back(*quantity);
  argProjections.push_back(*extendedprice);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

  /* Predicates */
  expressions::Expression *lhs1 = new expressions::RecordProjection(
      orderkey->getOriginalType(), arg, *orderkey);
  expressions::Expression *rhs1 =
      new expressions::IntConstant(predicates.at(0));
  expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

  expressions::Expression *lhs2 = new expressions::RecordProjection(
      quantity->getOriginalType(), arg, *quantity);
  expressions::Expression *rhs2 =
      new expressions::FloatConstant(predicates.at(1));
  expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

  expressions::Expression *lhs3 = new expressions::RecordProjection(
      linenumber->getOriginalType(), arg, *linenumber);
  expressions::Expression *rhs3 =
      new expressions::IntConstant(predicates.at(2));
  expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

  expressions::Expression *lhs4 = new expressions::RecordProjection(
      extendedprice->getOriginalType(), arg, *extendedprice);
  expressions::Expression *rhs4 =
      new expressions::FloatConstant(predicates.at(3));
  expressions::Expression *pred4 = new expressions::LtExpression(lhs4, rhs4);

  /* Notice that we apply predicates in reverse order */
  Select *sel1 = new Select(pred4, scan);
  scan->setParent(sel1);

  Select *sel2 = new Select(pred3, sel1);
  sel1->setParent(sel2);

  Select *sel3 = new Select(pred2, sel2);
  sel2->setParent(sel3);

  Select *sel4 = new Select(pred1, sel3);
  sel3->setParent(sel4);

  lastSelectOp = sel4;

  /**
   * REDUCE
   * COUNT(*)
   */
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
  lastSelectOp->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

void tpchOrderSelection3(map<string, dataset> datasetCatalog,
                         vector<int> predicates) {
  int predicatesNo = predicates.size();
  if (predicatesNo != 4) {
    throw runtime_error(string("Invalid no. of predicates requested: "));
  }
  Context &ctx = *prepareContext("tpch-csv-selection3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = lineitem.path;
  RecordType rec = lineitem.recType;
  int linehint = lineitem.linehint;
  RecordAttribute *orderkey = argsLineitem["orderkey"];
  RecordAttribute *linenumber = argsLineitem["linenumber"];
  RecordAttribute *quantity = argsLineitem["quantity"];
  RecordAttribute *extendedprice = argsLineitem["extendedprice"];

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT(S)
   */
  Operator *lastSelectOp;
  list<RecordAttribute> argProjections;
  argProjections.push_back(*orderkey);
  argProjections.push_back(*linenumber);
  argProjections.push_back(*quantity);
  argProjections.push_back(*extendedprice);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

  /* Predicates */
  expressions::Expression *lhs1 = new expressions::RecordProjection(
      orderkey->getOriginalType(), arg, *orderkey);
  expressions::Expression *rhs1 =
      new expressions::IntConstant(predicates.at(0));
  expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

  expressions::Expression *lhs2 = new expressions::RecordProjection(
      quantity->getOriginalType(), arg, *quantity);
  expressions::Expression *rhs2 =
      new expressions::FloatConstant(predicates.at(1));
  expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

  expressions::Expression *lhs3 = new expressions::RecordProjection(
      linenumber->getOriginalType(), arg, *linenumber);
  expressions::Expression *rhs3 =
      new expressions::IntConstant(predicates.at(2));
  expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

  expressions::Expression *lhs4 = new expressions::RecordProjection(
      extendedprice->getOriginalType(), arg, *extendedprice);
  expressions::Expression *rhs4 =
      new expressions::FloatConstant(predicates.at(3));
  expressions::Expression *pred4 = new expressions::LtExpression(lhs4, rhs4);

  /* Two (2) composite predicates */
  expressions::Expression *predA = new expressions::AndExpression(pred1, pred2);
  expressions::Expression *predB = new expressions::AndExpression(pred3, pred4);

  Select *sel1 = new Select(predA, scan);
  scan->setParent(sel1);

  Select *sel2 = new Select(predB, sel1);
  sel1->setParent(sel2);

  lastSelectOp = sel2;

  /**
   * REDUCE
   * COUNT(*)
   */
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
  lastSelectOp->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

void tpchOrderSelection4(map<string, dataset> datasetCatalog,
                         vector<int> predicates) {
  int predicatesNo = predicates.size();
  if (predicatesNo != 4) {
    throw runtime_error(string("Invalid no. of predicates requested: "));
  }
  Context &ctx = *prepareContext("tpch-csv-selection4");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = lineitem.path;
  RecordType rec = lineitem.recType;
  int linehint = lineitem.linehint;
  RecordAttribute *orderkey = argsLineitem["orderkey"];
  RecordAttribute *linenumber = argsLineitem["linenumber"];
  RecordAttribute *quantity = argsLineitem["quantity"];
  RecordAttribute *extendedprice = argsLineitem["extendedprice"];

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT(S)
   */
  Operator *lastSelectOp;
  list<RecordAttribute> argProjections;
  argProjections.push_back(*orderkey);
  argProjections.push_back(*linenumber);
  argProjections.push_back(*quantity);
  argProjections.push_back(*extendedprice);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

  /* Predicates */
  expressions::Expression *lhs1 = new expressions::RecordProjection(
      orderkey->getOriginalType(), arg, *orderkey);
  expressions::Expression *rhs1 =
      new expressions::IntConstant(predicates.at(0));
  expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

  expressions::Expression *lhs2 = new expressions::RecordProjection(
      quantity->getOriginalType(), arg, *quantity);
  expressions::Expression *rhs2 =
      new expressions::FloatConstant(predicates.at(1));
  expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

  expressions::Expression *lhs3 = new expressions::RecordProjection(
      linenumber->getOriginalType(), arg, *linenumber);
  expressions::Expression *rhs3 =
      new expressions::IntConstant(predicates.at(2));
  expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

  expressions::Expression *lhs4 = new expressions::RecordProjection(
      extendedprice->getOriginalType(), arg, *extendedprice);
  expressions::Expression *rhs4 =
      new expressions::FloatConstant(predicates.at(3));
  expressions::Expression *pred4 = new expressions::LtExpression(lhs4, rhs4);

  /* One (1) final composite predicate */
  expressions::Expression *predA = new expressions::AndExpression(pred1, pred2);
  expressions::Expression *predB = new expressions::AndExpression(pred3, pred4);
  expressions::Expression *pred = new expressions::AndExpression(predA, predB);

  Select *sel1 = new Select(pred, scan);
  scan->setParent(sel1);

  lastSelectOp = sel1;

  /**
   * REDUCE
   * COUNT(*)
   */
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
  lastSelectOp->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}
