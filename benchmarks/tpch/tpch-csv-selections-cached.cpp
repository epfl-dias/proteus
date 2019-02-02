/*
    Proteus -- High-performance query processing on heterogeneous hardware.

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

#include "common/common.hpp"
#include "common/tpch-config.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "operators/join.hpp"
#include "operators/materializer-expr.hpp"
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
#include "util/caching.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

void tpchSchema(map<string, dataset> &datasetCatalog) {
  tpchSchemaCSV(datasetCatalog);
}

/**
 * Selections
 * All should have the same result - what changes is the order of the predicates
 * and whether some are applied via a conjunction
 */
// 1-2-3-4
/* Can exploit predefined schema */
void tpchOrderSelection1(map<string, dataset> datasetCatalog,
                         vector<int> predicates);

/* Caching the one field actually dictating selectivity -> orderkey */
void tpchOrderSelection1CachingPred(map<string, dataset> datasetCatalog,
                                    vector<int> predicates);
/* Caching the fields whose conversion is the most costly -> quantity, extended
 * price */
void tpchOrderSelection1CachingFloats(map<string, dataset> datasetCatalog,
                                      vector<int> predicates);
void tpchOrderSelection1CachingPredFloats(map<string, dataset> datasetCatalog,
                                          vector<int> predicates);

int main() {
  map<string, dataset> datasetCatalog;
  tpchSchema(datasetCatalog);

  /* pred1 will be the one dictating query selectivity*/
  int pred1 = L_ORDERKEY_MAX;
  int pred2 = (int)L_QUANTITY_MAX;
  int pred3 = L_LINENUMBER_MAX;
  int pred4 = (int)L_EXTENDEDPRICE_MAX;

  // cout << "Building PM" << endl;
  vector<int> predicates;
  predicates.push_back(pred1);
  predicates.push_back(pred2);
  predicates.push_back(pred3);
  predicates.push_back(pred4);

  for (int i = 0; i < 5; i++) {
    cout << "[tpch-csv-selections-cached: ] Run " << i + 1 << endl;
    /* Preparing cache (1) + pm */
    cout << "Preparing caches (&PM): Pred" << endl;
    tpchOrderSelection1CachingPred(datasetCatalog, predicates);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Pred" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(pred1 * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      vector<int> predicates;
      predicates.push_back(predicateVal);
      // 1 pred.
      predicates.push_back(pred2);
      // 2 pred.
      predicates.push_back(pred3);
      // 3 pred.
      // 4 pred.
      predicates.push_back(pred4);
      tpchOrderSelection1(datasetCatalog, predicates);
    }

    /* Clean */
    Catalog &rawCatalog = Catalog::getInstance();
    rawCatalog.clear();
    CachingService &cache = CachingService::getInstance();
    cache.clear();
    /* Caching again */
    cout << "Preparing caches: Floats" << endl;
    tpchOrderSelection1CachingFloats(datasetCatalog, predicates);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Floats" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(pred1 * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      vector<int> predicates;
      predicates.push_back(predicateVal);
      // 1 pred.
      predicates.push_back(pred2);
      // 2 pred.
      predicates.push_back(pred3);
      // 3 pred.
      // 4 pred.
      predicates.push_back(pred4);
      tpchOrderSelection1(datasetCatalog, predicates);
    }

    rawCatalog.clear();
    cache.clear();

    /* Caching again */
    cout << "Preparing caches: Pred +  Floats" << endl;
    tpchOrderSelection1CachingPredFloats(datasetCatalog, predicates);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Pred +  Floats" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(pred1 * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      vector<int> predicates;
      predicates.push_back(predicateVal);
      // 1 pred.
      predicates.push_back(pred2);
      // 2 pred.
      predicates.push_back(pred3);
      // 3 pred.
      // 4 pred.
      predicates.push_back(pred4);
      tpchOrderSelection1(datasetCatalog, predicates);
    }

    rawCatalog.clear();
    cache.clear();
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
  int policy = 5;
  int lineHint = lineitem.linehint;
  char delimInner = '|';
  vector<RecordAttribute *> projections;
  RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
  RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
  RecordAttribute *l_quantity = argsLineitem["l_quantity"];
  RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];

  if (predicatesNo == 1 || predicatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
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
    argProjections.push_back(*l_orderkey);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);
    lastSelectOp = sel1;
  } else if (predicatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    lastSelectOp = sel2;
  } else if (predicatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);
    lastSelectOp = sel3;

  } else if (predicatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
    expressions::Expression *arg =
        new expressions::InputArgument(&rec, 0, argProjections);

    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, scan);
    scan->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);

    expressions::Expression *lhs4 = new expressions::RecordProjection(
        l_extendedprice->getOriginalType(), arg, *l_extendedprice);
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

void tpchOrderSelection1CachingPred(map<string, dataset> datasetCatalog,
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
  int policy = 5;
  int lineHint = lineitem.linehint;
  char delimInner = '|';
  vector<RecordAttribute *> projections;
  RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
  RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
  RecordAttribute *l_quantity = argsLineitem["l_quantity"];
  RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];

  if (predicatesNo == 1 || predicatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  expressions::Expression *arg = nullptr;
  list<RecordAttribute> argProjections;
  if (predicatesNo == 1) {
    argProjections.push_back(*l_orderkey);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);

  } else if (predicatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  }
  /*
   * Materialize expression(s) here
   * l_orderkey the one cached
   */
  expressions::Expression *toMat = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);

  char matLabel[] = "orderkeyMaterializer";
  ExprMaterializer *mat =
      new ExprMaterializer(toMat, lineHint, scan, &ctx, matLabel);
  scan->setParent(mat);

  /**
   * SELECT(S)
   * 1 to 4
   *
   * Lots of repetition..
   */
  Operator *lastSelectOp = nullptr;

  if (predicatesNo == 1) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat);
    mat->setParent(sel1);
    lastSelectOp = sel1;
  } else if (predicatesNo == 2) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat);
    mat->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    lastSelectOp = sel2;
  } else if (predicatesNo == 3) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat);
    mat->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);
    lastSelectOp = sel3;

  } else if (predicatesNo == 4) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat);
    mat->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);

    expressions::Expression *lhs4 = new expressions::RecordProjection(
        l_extendedprice->getOriginalType(), arg, *l_extendedprice);
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

void tpchOrderSelection1CachingFloats(map<string, dataset> datasetCatalog,
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
  int policy = 5;
  int lineHint = lineitem.linehint;
  char delimInner = '|';
  vector<RecordAttribute *> projections;
  RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
  RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
  RecordAttribute *l_quantity = argsLineitem["l_quantity"];
  RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];

  if (predicatesNo == 1 || predicatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  expressions::Expression *arg = nullptr;
  list<RecordAttribute> argProjections;
  if (predicatesNo == 1) {
    argProjections.push_back(*l_orderkey);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);

  } else if (predicatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  }

  /*
   * Materialize expression(s) here
   * l_l_quantity the one cached
   */
  expressions::Expression *toMat1 = new expressions::RecordProjection(
      l_quantity->getOriginalType(), arg, *l_quantity);

  char matLabel[] = "l_quantityMaterializer";
  ExprMaterializer *mat1 =
      new ExprMaterializer(toMat1, lineHint, scan, &ctx, matLabel);
  scan->setParent(mat1);

  /*
   * Materialize (more) expression(s) here
   * l_l_extendedprice the one cached
   */
  expressions::Expression *toMat2 = new expressions::RecordProjection(
      l_extendedprice->getOriginalType(), arg, *l_extendedprice);
  char matLabel2[] = "priceMaterializer";
  ExprMaterializer *mat2 =
      new ExprMaterializer(toMat2, lineHint, mat1, &ctx, matLabel2);
  mat1->setParent(mat2);

  /**
   * SELECT(S)
   * 1 to 4
   *
   * Lots of repetition..
   */
  Operator *lastSelectOp = nullptr;

  if (predicatesNo == 1) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);
    lastSelectOp = sel1;
  } else if (predicatesNo == 2) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    lastSelectOp = sel2;
  } else if (predicatesNo == 3) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);
    lastSelectOp = sel3;

  } else if (predicatesNo == 4) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);

    expressions::Expression *lhs4 = new expressions::RecordProjection(
        l_extendedprice->getOriginalType(), arg, *l_extendedprice);
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

void tpchOrderSelection1CachingPredFloats(map<string, dataset> datasetCatalog,
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
  int policy = 5;
  int lineHint = lineitem.linehint;
  char delimInner = '|';
  vector<RecordAttribute *> projections;
  RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
  RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
  RecordAttribute *l_quantity = argsLineitem["l_quantity"];
  RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];

  if (predicatesNo == 1 || predicatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (predicatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  expressions::Expression *arg = nullptr;
  list<RecordAttribute> argProjections;
  if (predicatesNo == 1) {
    argProjections.push_back(*l_orderkey);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  } else if (predicatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    arg = new expressions::InputArgument(&rec, 0, argProjections);

  } else if (predicatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
    arg = new expressions::InputArgument(&rec, 0, argProjections);
  }

  /*
   * Materialize expression(s) here
   * l_orderkey (pred) the one cached
   */
  expressions::Expression *toMat0 = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);

  char matLabel0[] = "orderkeyMaterializer";
  ExprMaterializer *mat0 =
      new ExprMaterializer(toMat0, lineHint, scan, &ctx, matLabel0);
  scan->setParent(mat0);

  /*
   * Materialize expression(s) here
   * l_quantity the one cached
   */
  expressions::Expression *toMat1 = new expressions::RecordProjection(
      l_quantity->getOriginalType(), arg, *l_quantity);

  char matLabel[] = "quantityMaterializer";
  ExprMaterializer *mat1 =
      new ExprMaterializer(toMat1, lineHint, mat0, &ctx, matLabel);
  mat0->setParent(mat1);

  /*
   * Materialize (more) expression(s) here
   * l_extendedprice the one cached
   */
  expressions::Expression *toMat2 = new expressions::RecordProjection(
      l_extendedprice->getOriginalType(), arg, *l_extendedprice);
  char matLabel2[] = "priceMaterializer";
  ExprMaterializer *mat2 =
      new ExprMaterializer(toMat2, lineHint, mat1, &ctx, matLabel2);
  mat1->setParent(mat2);

  /**
   * SELECT(S)
   * 1 to 4
   *
   * Lots of repetition..
   */
  Operator *lastSelectOp = nullptr;

  if (predicatesNo == 1) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);
    lastSelectOp = sel1;
  } else if (predicatesNo == 2) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    lastSelectOp = sel2;
  } else if (predicatesNo == 3) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);
    lastSelectOp = sel3;

  } else if (predicatesNo == 4) {
    expressions::Expression *lhs1 = new expressions::RecordProjection(
        l_orderkey->getOriginalType(), arg, *l_orderkey);
    expressions::Expression *rhs1 =
        new expressions::IntConstant(predicates.at(0));
    expressions::Expression *pred1 = new expressions::LtExpression(lhs1, rhs1);

    Select *sel1 = new Select(pred1, mat2);
    mat2->setParent(sel1);

    expressions::Expression *lhs2 = new expressions::RecordProjection(
        l_quantity->getOriginalType(), arg, *l_quantity);
    expressions::Expression *rhs2 =
        new expressions::FloatConstant(predicates.at(1));
    expressions::Expression *pred2 = new expressions::LtExpression(lhs2, rhs2);

    Select *sel2 = new Select(pred2, sel1);
    sel1->setParent(sel2);

    expressions::Expression *lhs3 = new expressions::RecordProjection(
        l_linenumber->getOriginalType(), arg, *l_linenumber);
    expressions::Expression *rhs3 =
        new expressions::IntConstant(predicates.at(2));
    expressions::Expression *pred3 = new expressions::LtExpression(lhs3, rhs3);

    Select *sel3 = new Select(pred3, sel2);
    sel2->setParent(sel3);

    expressions::Expression *lhs4 = new expressions::RecordProjection(
        l_extendedprice->getOriginalType(), arg, *l_extendedprice);
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
