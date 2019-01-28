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

#include "common/common.hpp"
#include "common/tpch-config.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "operators/join.hpp"
#include "operators/materializer-expr.hpp"
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
#include "plugins/csv-plugin-pm.hpp"
#include "util/raw-caching.hpp"
#include "util/raw-context.hpp"
#include "util/raw-functions.hpp"
#include "values/expressionTypes.hpp"

void tpchSchema(map<string, dataset> &datasetCatalog) {
  tpchSchemaCSV(datasetCatalog);
}
/**
 * Projections
   SELECT COUNT(*)
   FROM lineitem
   WHERE l_orderkey < [X]
 */
/* 1-4 aggregates */
void tpchLineitemProjection3(map<string, dataset> datasetCatalog,
                             int predicateVal, int aggregatesNo);

/* 1-4 aggregates & Materializing predicate (l_orderkey) */
void tpchLineitemProjection3CachingPred(map<string, dataset> datasetCatalog,
                                        int predicateVal, int aggregatesNo);

/* 1-4 aggregates & Materializing floats (quantity & extendedprice) */
void tpchLineitemProjection3CachingAgg(map<string, dataset> datasetCatalog,
                                       int predicateVal, int aggregatesNo);

/* 1-4 aggregates & Materializing predicate (l_orderkey) *AND* floats (quantity
 * & extendedprice) */
void tpchLineitemProjection3CachingPredAgg(map<string, dataset> datasetCatalog,
                                           int predicateVal, int aggregatesNo);

int main() {
  cout << "Execution" << endl;
  map<string, dataset> datasetCatalog;
  tpchSchema(datasetCatalog);

  int predicateMax = L_ORDERKEY_MAX;

  for (int i = 0; i < 5; i++) {
    cout << "[tpch-csv-projections-cached: ] Run " << i + 1 << endl;
    cout << "Preparing caches (&PM): Pred" << endl;
    tpchLineitemProjection3CachingPred(datasetCatalog, predicateMax, 4);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Pred" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(predicateMax * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      tpchLineitemProjection3(datasetCatalog, predicateVal, 4);
      cout << "---" << endl;
    }

    RawCatalog &rawCatalog = RawCatalog::getInstance();
    rawCatalog.clear();
    CachingService &cache = CachingService::getInstance();
    cache.clear();

    cout << "Preparing caches: Agg" << endl;
    tpchLineitemProjection3CachingAgg(datasetCatalog, predicateMax, 4);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Agg" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(predicateMax * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      tpchLineitemProjection3(datasetCatalog, predicateVal, 4);
      cout << "---" << endl;
    }

    cout << "Preparing caches: Pred +  Agg" << endl;
    tpchLineitemProjection3CachingPredAgg(datasetCatalog, predicateMax, 4);
    cout << "CACHING (MATERIALIZING) INTO EFFECT: Pred + Agg" << endl;
    for (int i = 1; i <= 10; i++) {
      double ratio = (i / (double)10);
      double percentage = ratio * 100;
      int predicateVal = (int)ceil(predicateMax * ratio);
      cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
           << "%" << endl;
      tpchLineitemProjection3(datasetCatalog, predicateVal, 4);
      cout << "---" << endl;
    }
  }
}

void tpchLineitemProjection3(map<string, dataset> datasetCatalog,
                             int predicateVal, int aggregatesNo) {
  if (aggregatesNo <= 0 || aggregatesNo > 4) {
    throw runtime_error(string("Invalid aggregate no. requested: "));
  }
  RawContext &ctx = *prepareContext("tpch-csv-projection3");
  RawCatalog &rawCatalog = RawCatalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  string nameOrders = string("orders");
  dataset orders = datasetCatalog[nameOrders];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();
  map<string, RecordAttribute *> argsOrder = orders.recType.getArgsMap();

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

  if (aggregatesNo == 1 || aggregatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 4) {
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
   * REDUCE
   * MAX(l_quantity), [COUNT(*), [MAX(l_lineitem), [MAX(l_extendedprice)]]]
   * + predicate
   */
  list<RecordAttribute> argProjections;
  if (aggregatesNo == 1 || aggregatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
  }

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  switch (aggregatesNo) {
    case 1: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);
      break;
    }
    case 2: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);
      break;
    }
    case 3: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);
      break;
    }
    case 4: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);

      accs.push_back(MAX);
      expressions::Expression *outputExpr4 = new expressions::RecordProjection(
          l_extendedprice->getOriginalType(), arg, *l_extendedprice);
      outputExprs.push_back(outputExpr4);
      break;
    }
    default: {
      // Unreachable
      throw runtime_error(string("Invalid aggregate no. requested: "));
    }
  }

  /* Pred: */
  expressions::Expression *selOrderkey = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);
  expressions::Expression *val_key = new expressions::IntConstant(predicateVal);
  expressions::Expression *predicate =
      new expressions::LtExpression(selOrderkey, val_key);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
  scan->setParent(reduce);

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

void tpchLineitemProjection3CachingPred(map<string, dataset> datasetCatalog,
                                        int predicateVal, int aggregatesNo) {
  if (aggregatesNo <= 0 || aggregatesNo > 4) {
    throw runtime_error(string("Invalid aggregate no. requested: "));
  }
  RawContext &ctx = *prepareContext("tpch-csv-projection3");
  RawCatalog &rawCatalog = RawCatalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  string nameOrders = string("orders");
  dataset orders = datasetCatalog[nameOrders];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();
  map<string, RecordAttribute *> argsOrder = orders.recType.getArgsMap();

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

  if (aggregatesNo == 1 || aggregatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  list<RecordAttribute> argProjections;
  if (aggregatesNo == 1 || aggregatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
  }

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

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
   * REDUCE
   * MAX(l_quantity), [COUNT(*), [MAX(l_lineitem), [MAX(l_extendedprice)]]]
   * + predicate
   */

  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  switch (aggregatesNo) {
    case 1: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);
      break;
    }
    case 2: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);
      break;
    }
    case 3: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);
      break;
    }
    case 4: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);

      accs.push_back(MAX);
      expressions::Expression *outputExpr4 = new expressions::RecordProjection(
          l_extendedprice->getOriginalType(), arg, *l_extendedprice);
      outputExprs.push_back(outputExpr4);
      break;
    }
    default: {
      // Unreachable
      throw runtime_error(string("Invalid aggregate no. requested: "));
    }
  }

  /* Pred: */
  expressions::Expression *selOrderkey = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);
  expressions::Expression *val_key = new expressions::IntConstant(predicateVal);
  expressions::Expression *predicate =
      new expressions::LtExpression(selOrderkey, val_key);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
  mat->setParent(reduce);

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

void tpchLineitemProjection3CachingAgg(map<string, dataset> datasetCatalog,
                                       int predicateVal, int aggregatesNo) {
  if (aggregatesNo <= 0 || aggregatesNo > 4) {
    throw runtime_error(string("Invalid aggregate no. requested: "));
  }
  RawContext &ctx = *prepareContext("tpch-csv-projection3");
  RawCatalog &rawCatalog = RawCatalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  string nameOrders = string("orders");
  dataset orders = datasetCatalog[nameOrders];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();
  map<string, RecordAttribute *> argsOrder = orders.recType.getArgsMap();

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

  if (aggregatesNo == 1 || aggregatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  list<RecordAttribute> argProjections;
  if (aggregatesNo == 1 || aggregatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
  }

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

  /*
   * Materialize expression(s) here
   * l_quantity the one cached
   */
  expressions::Expression *toMat1 = new expressions::RecordProjection(
      l_quantity->getOriginalType(), arg, *l_quantity);

  char matLabel[] = "quantityMaterializer";
  ExprMaterializer *mat1 =
      new ExprMaterializer(toMat1, lineHint, scan, &ctx, matLabel);
  scan->setParent(mat1);

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
   * REDUCE
   * MAX(l_quantity), [COUNT(*), [MAX(l_lineitem), [MAX(l_extendedprice)]]]
   * + predicate
   */

  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  switch (aggregatesNo) {
    case 1: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);
      break;
    }
    case 2: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);
      break;
    }
    case 3: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);
      break;
    }
    case 4: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);

      accs.push_back(MAX);
      expressions::Expression *outputExpr4 = new expressions::RecordProjection(
          l_extendedprice->getOriginalType(), arg, *l_extendedprice);
      outputExprs.push_back(outputExpr4);
      break;
    }
    default: {
      // Unreachable
      throw runtime_error(string("Invalid aggregate no. requested: "));
    }
  }

  /* Pred: */
  expressions::Expression *selOrderkey = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);
  expressions::Expression *val_key = new expressions::IntConstant(predicateVal);
  expressions::Expression *predicate =
      new expressions::LtExpression(selOrderkey, val_key);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
  mat2->setParent(reduce);

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

void tpchLineitemProjection3CachingPredAgg(map<string, dataset> datasetCatalog,
                                           int predicateVal, int aggregatesNo) {
  if (aggregatesNo <= 0 || aggregatesNo > 4) {
    throw runtime_error(string("Invalid aggregate no. requested: "));
  }
  RawContext &ctx = *prepareContext("tpch-csv-projection3");
  RawCatalog &rawCatalog = RawCatalog::getInstance();

  string nameLineitem = string("lineitem");
  dataset lineitem = datasetCatalog[nameLineitem];
  string nameOrders = string("orders");
  dataset orders = datasetCatalog[nameOrders];
  map<string, RecordAttribute *> argsLineitem = lineitem.recType.getArgsMap();
  map<string, RecordAttribute *> argsOrder = orders.recType.getArgsMap();

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

  if (aggregatesNo == 1 || aggregatesNo == 2) {
    projections.push_back(l_orderkey);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 3) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
  } else if (aggregatesNo == 4) {
    projections.push_back(l_orderkey);
    projections.push_back(l_linenumber);
    projections.push_back(l_quantity);
    projections.push_back(l_extendedprice);
  }

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, lineHint, policy);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  list<RecordAttribute> argProjections;
  if (aggregatesNo == 1 || aggregatesNo == 2) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 3) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
  } else if (aggregatesNo == 4) {
    argProjections.push_back(*l_orderkey);
    argProjections.push_back(*l_linenumber);
    argProjections.push_back(*l_quantity);
    argProjections.push_back(*l_extendedprice);
  }

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);

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
   * REDUCE
   * MAX(l_quantity), [COUNT(*), [MAX(l_lineitem), [MAX(l_extendedprice)]]]
   * + predicate
   */

  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  switch (aggregatesNo) {
    case 1: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);
      break;
    }
    case 2: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);
      break;
    }
    case 3: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);
      break;
    }
    case 4: {
      accs.push_back(MAX);
      expressions::Expression *outputExpr1 = new expressions::RecordProjection(
          l_quantity->getOriginalType(), arg, *l_quantity);
      outputExprs.push_back(outputExpr1);

      accs.push_back(SUM);
      expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
      outputExprs.push_back(outputExpr2);

      accs.push_back(MAX);
      expressions::Expression *outputExpr3 = new expressions::RecordProjection(
          l_linenumber->getOriginalType(), arg, *l_linenumber);
      outputExprs.push_back(outputExpr3);

      accs.push_back(MAX);
      expressions::Expression *outputExpr4 = new expressions::RecordProjection(
          l_extendedprice->getOriginalType(), arg, *l_extendedprice);
      outputExprs.push_back(outputExpr4);
      break;
    }
    default: {
      // Unreachable
      throw runtime_error(string("Invalid aggregate no. requested: "));
    }
  }

  /* Pred: */
  expressions::Expression *selOrderkey = new expressions::RecordProjection(
      l_orderkey->getOriginalType(), arg, *l_orderkey);
  expressions::Expression *val_key = new expressions::IntConstant(predicateVal);
  expressions::Expression *predicate =
      new expressions::LtExpression(selOrderkey, val_key);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
  mat2->setParent(reduce);

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
