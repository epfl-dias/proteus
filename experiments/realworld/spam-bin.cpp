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
#include "common/symantec-config.hpp"
#include "engines/olap/util/caching.hpp"
#include "engines/olap/util/context.hpp"
#include "engines/olap/util/functions.hpp"
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
#include "plugins/json-plugin.hpp"
#include "values/expressionTypes.hpp"

/* SELECT MIN(p_event),MAX(p_event), COUNT(*) from symantecunordered where id >
 * 50000000 and id < 60000000; */
void symantecBin1(map<string, dataset> datasetCatalog);
void symantecBin2(map<string, dataset> datasetCatalog);
void symantecBin3(map<string, dataset> datasetCatalog);
void symantecBin4(map<string, dataset> datasetCatalog);
void symantecBin5(map<string, dataset> datasetCatalog);
void symantecBin6(map<string, dataset> datasetCatalog);
void symantecBin7(map<string, dataset> datasetCatalog);
void symantecBin8(map<string, dataset> datasetCatalog);

int main() {
  cout << "Execution" << endl;
  map<string, dataset> datasetCatalog;
  symantecBinSchema(datasetCatalog);

  cout << "SYMANTEC BIN 1" << endl;
  symantecBin1(datasetCatalog);
  cout << "SYMANTEC BIN 2" << endl;
  symantecBin2(datasetCatalog);
  cout << "SYMANTEC BIN 3" << endl;
  symantecBin3(datasetCatalog);
  cout << "SYMANTEC BIN 4" << endl;
  symantecBin4(datasetCatalog);
  cout << "SYMANTEC BIN 5" << endl;
  symantecBin5(datasetCatalog);
  cout << "SYMANTEC BIN 6" << endl;
  symantecBin6(datasetCatalog);
  cout << "**************" << endl;
  cout << "SYMANTEC BIN 7" << endl;
  symantecBin7(datasetCatalog);
  cout << "**************" << endl;
  cout << "SYMANTEC BIN 8" << endl;
  symantecBin8(datasetCatalog);
  cout << "**************" << endl;
}

void symantecBin1(map<string, dataset> datasetCatalog) {
  int idLow = 50000000;
  int idHigh = 60000000;
  Context &ctx = *prepareContext("symantec-bin-1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;

  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *p_event = argsSymantecBin["p_event"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(p_event);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*id);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = new expressions::RecordProjection(
      p_event->getOriginalType(), arg, *p_event);
  outputExprs.push_back(outputExpr1);

  accs.push_back(SUM);
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr2);
  /* Pred: */
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate1, predicate2);

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

void symantecBin2(map<string, dataset> datasetCatalog) {
  int idLow = 50000000;
  int idHigh = 60000000;
  int dimHigh = 3;

  Context &ctx = *prepareContext("symantec-bin-2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;

  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *p_event = argsSymantecBin["p_event"];
  RecordAttribute *size = argsSymantecBin["size"];
  RecordAttribute *slice_id = argsSymantecBin["slice_id"];
  RecordAttribute *value = argsSymantecBin["value"];
  RecordAttribute *dim = argsSymantecBin["dim"];
  RecordAttribute *mdc = argsSymantecBin["mdc"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(p_event);
  projections.push_back(size);
  projections.push_back(slice_id);
  projections.push_back(value);
  projections.push_back(dim);
  projections.push_back(mdc);
  projections.push_back(cluster);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*id);
  argProjections.push_back(*p_event);
  argProjections.push_back(*size);
  argProjections.push_back(*slice_id);
  argProjections.push_back(*value);
  argProjections.push_back(*dim);
  argProjections.push_back(*mdc);
  argProjections.push_back(*cluster);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = new expressions::RecordProjection(
      p_event->getOriginalType(), arg, *p_event);
  outputExprs.push_back(outputExpr2);

  accs.push_back(MAX);
  expressions::Expression *outputExpr3 =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  outputExprs.push_back(outputExpr3);

  accs.push_back(MAX);
  expressions::Expression *outputExpr4 = new expressions::RecordProjection(
      slice_id->getOriginalType(), arg, *slice_id);
  outputExprs.push_back(outputExpr4);

  accs.push_back(MAX);
  expressions::Expression *outputExpr5 =
      new expressions::RecordProjection(value->getOriginalType(), arg, *value);
  outputExprs.push_back(outputExpr5);

  accs.push_back(MAX);
  expressions::Expression *outputExpr6 =
      new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
  outputExprs.push_back(outputExpr6);

  accs.push_back(MAX);
  expressions::Expression *outputExpr7 =
      new expressions::RecordProjection(mdc->getOriginalType(), arg, *mdc);
  outputExprs.push_back(outputExpr7);

  accs.push_back(MAX);
  expressions::Expression *outputExpr8 = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);
  outputExprs.push_back(outputExpr8);

  accs.push_back(SUM);
  expressions::Expression *outputExpr9 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr9);
  /* Pred: */
  expressions::Expression *selID = outputExpr1;
  expressions::Expression *selDim = outputExpr6;
  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(dimHigh);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selDim, predExpr2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate1, predicate2);

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

void symantecBin3(map<string, dataset> datasetCatalog) {
  int idLow = 59000000;
  int idHigh = 63000000;
  int dimHigh = 3;
  int clusterNo = 500;
  Context &ctx = *prepareContext("symantec-bin-3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;

  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *p_event = argsSymantecBin["p_event"];
  RecordAttribute *dim = argsSymantecBin["dim"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(p_event);
  projections.push_back(dim);
  projections.push_back(cluster);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*id);
  argProjections.push_back(*p_event);
  argProjections.push_back(*cluster);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = new expressions::RecordProjection(
      p_event->getOriginalType(), arg, *p_event);
  outputExprs.push_back(outputExpr1);

  accs.push_back(SUM);
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr2);

  /* Pred: */
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selDim =
      new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
  expressions::Expression *selCluster = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 = new expressions::IntConstant(dimHigh);
  expressions::Expression *predExpr4 = new expressions::IntConstant(clusterNo);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::LtExpression(selDim, predExpr3);
  expressions::Expression *predicate4 =
      new expressions::EqExpression(selCluster, predExpr4);
  expressions::Expression *predicateAnd1 =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicateAnd2 =
      new expressions::AndExpression(predicate3, predicate4);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicateAnd1, predicateAnd2);

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

void symantecBin4(map<string, dataset> datasetCatalog) {
  int idLow = 70000000;
  int idHigh = 80000000;
  double p_eventLow = 0.7;
  double valueLow = 0.5;
  int clusterNo = 400;

  Context &ctx = *prepareContext("symantec-bin-4");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;

  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *p_event = argsSymantecBin["p_event"];
  RecordAttribute *dim = argsSymantecBin["dim"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];
  RecordAttribute *value = argsSymantecBin["value"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(p_event);
  projections.push_back(dim);
  projections.push_back(cluster);
  projections.push_back(value);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*id);
  argProjections.push_back(*p_event);
  argProjections.push_back(*value);
  argProjections.push_back(*dim);
  argProjections.push_back(*cluster);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 =
      new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
  outputExprs.push_back(outputExpr1);

  accs.push_back(SUM);
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr2);

  /* Pred: */
  /* id > 70000000 and id < 80000000 and (p_event > 0.7 OR value > 0.5) and
   * cluster = 400 */
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selEvent = new expressions::RecordProjection(
      p_event->getOriginalType(), arg, *p_event);
  expressions::Expression *selCluster = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);
  expressions::Expression *selValue =
      new expressions::RecordProjection(value->getOriginalType(), arg, *value);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 =
      new expressions::FloatConstant(p_eventLow);
  expressions::Expression *predExpr4 = new expressions::FloatConstant(valueLow);
  expressions::Expression *predExpr5 = new expressions::IntConstant(clusterNo);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::GtExpression(selEvent, predExpr3);
  expressions::Expression *predicate4 =
      new expressions::GtExpression(selValue, predExpr4);
  expressions::Expression *predicate5 =
      new expressions::EqExpression(selCluster, predExpr5);

  expressions::Expression *predicateAnd1 =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicateOr =
      new expressions::OrExpression(predicate3, predicate4);
  expressions::Expression *predicateAnd2 =
      new expressions::AndExpression(predicateAnd1, predicateOr);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicateAnd2, predicate5);

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

void symantecBin5(map<string, dataset> datasetCatalog) {
  int idLow = 380000000;
  int idHigh = 450000000;
  int sliceIdNo = 150;
  Context &ctx = *prepareContext("symantec-bin-5");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;

  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *slice_id = argsSymantecBin["slice_id"];
  RecordAttribute *neighbors = argsSymantecBin["neighbors"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(slice_id);
  projections.push_back(neighbors);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*id);
  argProjections.push_back(*slice_id);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = new expressions::RecordProjection(
      neighbors->getOriginalType(), arg, *neighbors);
  outputExprs.push_back(outputExpr1);

  accs.push_back(SUM);
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr2);

  /* Pred: */
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selSliceID = new expressions::RecordProjection(
      slice_id->getOriginalType(), arg, *slice_id);
  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 = new expressions::IntConstant(sliceIdNo);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::EqExpression(selSliceID, predExpr3);
  expressions::Expression *predicate_ =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate_, predicate3);

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

void symantecBin6(map<string, dataset> datasetCatalog) {
  int idLow = 380000000;
  int idHigh = 450000000;
  int clusterHigh = 10;
  Context &ctx = *prepareContext("symantec-bin-6(agg)");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;
  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *dim = argsSymantecBin["dim"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(dim);
  projections.push_back(cluster);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*dim);
  argSelections.push_back(*cluster);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selCluster = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 =
      new expressions::IntConstant(clusterHigh);
  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::LtExpression(selCluster, predExpr3);
  expressions::Expression *predicate_ =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate_, predicate3);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * NEST
   * GroupBy: cluster
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: MAX(dim), COUNT() => SUM(1)
   */
  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*cluster);
  nestProjections.push_back(*dim);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY cluster
  expressions::RecordProjection *f = new expressions::RecordProjection(
      cluster->getOriginalType(), nestArg, *cluster);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  fields.push_back(dim);
  outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(cluster);
  outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;

  /* Aggregate 1: MAX(dim) */
  expressions::Expression *aggrDim =
      new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
  expressions::Expression *outputExpr1 = aggrDim;
  aggrField1 = string("_maxDim");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  /* Aggregate 2: COUNT(*) */
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  aggrField2 = string("_aggrCount");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  // Function* debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  // FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  nestOp->setParent(printOp1);
  lastPrintOp = printOp1;

  RecordAttribute *toOutput2 =
      new RecordAttribute(2, aggrLabel, aggrField2, &intType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput2);
  Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
  printOp1->setParent(printOp2);
  lastPrintOp = printOp2;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

void symantecBin7(map<string, dataset> datasetCatalog) {
  int idLow = 59000000;
  int idHigh = 63000000;
  int dimHigh = 3;
  int clusterLow = 490;
  int clusterHigh = 500;
  Context &ctx = *prepareContext("symantec-bin-7(agg)");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;
  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *dim = argsSymantecBin["dim"];
  RecordAttribute *mdc = argsSymantecBin["mdc"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(dim);
  projections.push_back(mdc);
  projections.push_back(cluster);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   * id > 59000000 and id < 63000000 and dim < 3 AND cluster > 490 AND cluster
   * <= 500
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*dim);
  argSelections.push_back(*mdc);
  argSelections.push_back(*cluster);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selDim =
      new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
  expressions::Expression *selCluster = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 = new expressions::IntConstant(dimHigh);
  expressions::Expression *predExpr4 = new expressions::IntConstant(clusterLow);
  expressions::Expression *predExpr5 =
      new expressions::IntConstant(clusterHigh);

  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::LtExpression(selDim, predExpr3);
  expressions::Expression *predicate4 =
      new expressions::GtExpression(selCluster, predExpr4);
  expressions::Expression *predicate5 =
      new expressions::LeExpression(selCluster, predExpr5);

  expressions::Expression *predicateAnd1 =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicateAnd2 =
      new expressions::AndExpression(predicate3, predicate4);
  expressions::Expression *predicateAnd_ =
      new expressions::AndExpression(predicateAnd1, predicateAnd2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicateAnd_, predicate5);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * NEST
   * GroupBy: cluster
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: MAX(mdc), COUNT() => SUM(1)
   */
  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*cluster);
  nestProjections.push_back(*mdc);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY cluster
  expressions::RecordProjection *f = new expressions::RecordProjection(
      cluster->getOriginalType(), nestArg, *cluster);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  fields.push_back(mdc);
  outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(cluster);
  outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;

  /* Aggregate 1: MAX(mdc) */
  expressions::Expression *aggrMDC =
      new expressions::RecordProjection(mdc->getOriginalType(), arg, *mdc);
  expressions::Expression *outputExpr1 = aggrMDC;
  aggrField1 = string("_maxMDC");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  /* Aggregate 2: COUNT(*) */
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  aggrField2 = string("_aggrCount");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  // Function* debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  // FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  nestOp->setParent(printOp1);
  lastPrintOp = printOp1;

  RecordAttribute *toOutput2 =
      new RecordAttribute(2, aggrLabel, aggrField2, &intType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput2);
  Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
  printOp1->setParent(printOp2);
  lastPrintOp = printOp2;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

void symantecBin8(map<string, dataset> datasetCatalog) {
  int idLow = 70000000;
  int idHigh = 80000000;
  double p_eventLow = 0.7;
  double valueLow = 0.5;
  int clusterLow = 395;
  int clusterHigh = 405;
  Context &ctx = *prepareContext("symantec-bin-8(agg)");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  string fnamePrefix = symantecBin.path;
  RecordType rec = symantecBin.recType;
  int linehint = symantecBin.linehint;
  RecordAttribute *id = argsSymantecBin["id"];
  RecordAttribute *p_event = argsSymantecBin["p_event"];
  RecordAttribute *value = argsSymantecBin["value"];
  RecordAttribute *cluster = argsSymantecBin["cluster"];
  RecordAttribute *neighbors = argsSymantecBin["neighbors"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(p_event);
  projections.push_back(value);
  projections.push_back(cluster);
  projections.push_back(neighbors);

  BinaryColPlugin *pg =
      new BinaryColPlugin(&ctx, fnamePrefix, rec, projections);
  rawCatalog.registerPlugin(fnamePrefix, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   * id > 70000000  and id < 80000000 and (p_event > 0.7 OR value > 0.5) and
   * cluster > 395 and cluster <= 405
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*p_event);
  argSelections.push_back(*value);
  argSelections.push_back(*cluster);
  argSelections.push_back(*neighbors);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selEvent = new expressions::RecordProjection(
      p_event->getOriginalType(), arg, *p_event);
  expressions::Expression *selValue =
      new expressions::RecordProjection(value->getOriginalType(), arg, *value);
  expressions::Expression *selCluster = new expressions::RecordProjection(
      cluster->getOriginalType(), arg, *cluster);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr3 =
      new expressions::FloatConstant(p_eventLow);
  expressions::Expression *predExpr4 = new expressions::FloatConstant(valueLow);
  expressions::Expression *predExpr5 = new expressions::IntConstant(clusterLow);
  expressions::Expression *predExpr6 =
      new expressions::IntConstant(clusterHigh);

  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicateAnd1 =
      new expressions::AndExpression(predicate1, predicate2);

  expressions::Expression *predicate3 =
      new expressions::GtExpression(selEvent, predExpr3);
  expressions::Expression *predicate4 =
      new expressions::GtExpression(selValue, predExpr4);
  expressions::Expression *predicateOr =
      new expressions::OrExpression(predicate3, predicate4);

  expressions::Expression *predicate5 =
      new expressions::GtExpression(selCluster, predExpr5);
  expressions::Expression *predicate6 =
      new expressions::LeExpression(selCluster, predExpr6);
  expressions::Expression *predicateAnd2 =
      new expressions::AndExpression(predicate5, predicate6);

  expressions::Expression *predicateAnd =
      new expressions::AndExpression(predicateAnd1, predicateOr);

  expressions::Expression *predicate =
      new expressions::AndExpression(predicateAnd, predicateAnd2);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * NEST
   * GroupBy: cluster
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: MAX(neighbors), COUNT() => SUM(1)
   */
  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*cluster);
  nestProjections.push_back(*neighbors);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY cluster
  expressions::RecordProjection *f = new expressions::RecordProjection(
      cluster->getOriginalType(), nestArg, *cluster);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  fields.push_back(neighbors);
  outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(cluster);
  outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;

  /* Aggregate 1: MAX(neighbors) */
  expressions::Expression *aggrNeighbors = new expressions::RecordProjection(
      neighbors->getOriginalType(), arg, *neighbors);
  expressions::Expression *outputExpr1 = aggrNeighbors;
  aggrField1 = string("_maxMDC");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  /* Aggregate 2: COUNT(*) */
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  aggrField2 = string("_aggrCount");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  // Function* debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  // FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  nestOp->setParent(printOp1);
  lastPrintOp = printOp1;

  RecordAttribute *toOutput2 =
      new RecordAttribute(2, aggrLabel, aggrField2, &intType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput2);
  Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
  printOp1->setParent(printOp2);
  lastPrintOp = printOp2;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}
