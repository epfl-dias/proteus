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
#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "operators/join.hpp"
#include "operators/materializer-expr.hpp"
#include "operators/nest-opt.hpp"
#include "operators/nest.hpp"
#include "operators/null-filter.hpp"
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
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/json-plugin.hpp"
#include "util/caching.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

void symantecCSVWarmup(map<string, dataset> datasetCatalog);
void symantecJSONWarmup(map<string, dataset> datasetCatalog);
void symantecCSVJSON1(map<string, dataset> datasetCatalog);
/*
223
223.223000
64720
*/
void symantecCSVJSON2(map<string, dataset> datasetCatalog);
void symantecCSVJSON3(map<string, dataset> datasetCatalog);
void symantecCSVJSON4(map<string, dataset> datasetCatalog);
void symantecCSVJSON5(map<string, dataset> datasetCatalog);

int main() {
  cout << "Execution" << endl;
  map<string, dataset> datasetCatalog;
  symantecCSVSchema(datasetCatalog);

  symantecCoreIDDatesSchema(datasetCatalog);
  symantecJSONWarmup(datasetCatalog);
  symantecCSVWarmup(datasetCatalog);
  cout << "SYMANTEC CSV-JSON 1" << endl;
  symantecCSVJSON1(datasetCatalog);
  cout << "SYMANTEC CSV-JSON 2" << endl;
  symantecCSVJSON2(datasetCatalog);
  cout << "SYMANTEC CSV-JSON 3" << endl;
  symantecCSVJSON3(datasetCatalog);
  cout << "SYMANTEC CSV-JSON 4" << endl;
  symantecCSVJSON4(datasetCatalog);
  cout << "SYMANTEC CSV-JSON 5" << endl;
  symantecCSVJSON5(datasetCatalog);
}

void symantecCSVJSON1(map<string, dataset> datasetCatalog) {
  // csv
  int idLow = 5000000;
  int idHigh = 10000000;
  string botName = "Bobax";
  // JSON
  // id, again

  Context &ctx = *prepareContext("symantec-CSV-JSON-1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecCSV = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantecCSV];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  pm::CSVPlugin *pgCSV;
  Scan *scanCSV;
  RecordAttribute *idCSV;
  RecordAttribute *classa;
  RecordAttribute *classb;
  RecordAttribute *bot;
  RecordType recCSV = symantecCSV.recType;
  string fnameCSV = symantecCSV.path;
  int linehintCSV = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  idCSV = argsSymantecCSV["id"];
  classa = argsSymantecCSV["classa"];
  classb = argsSymantecCSV["classb"];
  bot = argsSymantecCSV["bot"];
  vector<RecordAttribute *> projectionsCSV;
  projectionsCSV.push_back(idCSV);
  projectionsCSV.push_back(classa);
  projectionsCSV.push_back(classb);
  projectionsCSV.push_back(bot);

  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
                            linehintCSV, policy, false);
  rawCatalog.registerPlugin(fnameCSV, pgCSV);
  scanCSV = new Scan(&ctx, *pgCSV);

  /*
   * SELECT CSV
   * id > 5000000 and id < 10000000 and bot = 'Bobax'
   */
  Select *selCSV;
  expressions::Expression *predicateCSV;
  {
    list<RecordAttribute> argProjectionsCSV;
    argProjectionsCSV.push_back(*idCSV);
    argProjectionsCSV.push_back(*bot);
    expressions::Expression *arg =
        new expressions::InputArgument(&recCSV, 0, argProjectionsCSV);
    expressions::Expression *selID = new expressions::RecordProjection(
        idCSV->getOriginalType(), arg, *idCSV);
    expressions::Expression *selBot =
        new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicate1, predicate2);

    Select *selNum = new Select(predicateNum, scanCSV);
    scanCSV->setParent(selNum);

    expressions::Expression *predExpr3 =
        new expressions::StringConstant(botName);
    expressions::Expression *predicateStr =
        new expressions::EqExpression(selBot, predExpr3);

    selCSV = new Select(predicateStr, selNum);
    selNum->setParent(selCSV);
  }

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int > 5000000 and (data->>'id')::int < 10000000
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selID = new expressions::RecordProjection(
        idJSON->getOriginalType(), arg, *idJSON);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    predicateJSON = new expressions::AndExpression(predicate1, predicate2);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idCSV);
  argProjectionsLeft.push_back(*classa);
  argProjectionsLeft.push_back(*classb);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recCSV, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);

  // RIGHT SIDE
  list<RecordAttribute> argProjectionsRight;
  argProjectionsRight.push_back(*idJSON);
  expressions::Expression *rightArg =
      new expressions::InputArgument(&recJSON, 1, argProjectionsRight);
  expressions::Expression *rightPred = new expressions::RecordProjection(
      idJSON->getOriginalType(), rightArg, *idJSON);

  /* join pred. */
  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(leftPred, rightPred);

  /* left materializer - dim needed */
  vector<RecordAttribute *> fieldsLeft;
  fieldsLeft.push_back(classa);
  fieldsLeft.push_back(classb);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnameCSV, activeLoop, pgCSV->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgCSV->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - no explicit field needed */
  vector<RecordAttribute *> fieldsRight;
  vector<materialization_mode> outputModesRight;

  /* explicit mention to right OID */
  RecordAttribute *projTupleR =
      new RecordAttribute(fnameJSON, activeLoop, pgJSON->getOIDType());
  vector<RecordAttribute *> OIDRight;
  OIDRight.push_back(projTupleR);
  expressions::Expression *exprRightOID = new expressions::RecordProjection(
      pgJSON->getOIDType(), rightArg, *projTupleR);
  vector<expression_t> expressionsRight;
  expressionsRight.push_back(exprRightOID);

  Materializer *matRight = new Materializer(fieldsRight, expressionsRight,
                                            OIDRight, outputModesRight);

  char joinLabel[] = "radixJoinCSVJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selCSV, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selCSV->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(classa), MAX(classb), COUNT(*)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*classa);
  argProjections.push_back(*classb);
  expressions::Expression *exprClassa = new expressions::RecordProjection(
      classa->getOriginalType(), leftArg, *classa);
  expressions::Expression *exprClassb = new expressions::RecordProjection(
      classb->getOriginalType(), leftArg, *classb);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprClassa;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprClassb;
  outputExprs.push_back(outputExpr2);

  accs.push_back(SUM);
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr3);
  /* Pred: Redundant */

  expressions::Expression *lhsRed = new expressions::BoolConstant(true);
  expressions::Expression *rhsRed = new expressions::BoolConstant(true);
  expressions::Expression *predRed =
      new expressions::EqExpression(lhsRed, rhsRed);

  opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgCSV->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecCSVJSON2(map<string, dataset> datasetCatalog) {
  // csv
  int idLow = 8000000;
  int idHigh = 10000000;
  string botName = "Bobax";
  // JSON
  // id, again
  int sizeLow = 1000;

  Context &ctx = *prepareContext("symantec-CSV-JSON-2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecCSV = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantecCSV];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  pm::CSVPlugin *pgCSV;
  Scan *scanCSV;
  RecordAttribute *idCSV;
  RecordAttribute *classa;
  RecordAttribute *classb;
  RecordAttribute *bot;
  RecordType recCSV = symantecCSV.recType;
  string fnameCSV = symantecCSV.path;
  int linehintCSV = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  idCSV = argsSymantecCSV["id"];
  classa = argsSymantecCSV["classa"];
  classb = argsSymantecCSV["classb"];
  bot = argsSymantecCSV["bot"];
  vector<RecordAttribute *> projectionsCSV;
  projectionsCSV.push_back(idCSV);
  projectionsCSV.push_back(classa);
  projectionsCSV.push_back(classb);
  projectionsCSV.push_back(bot);

  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
                            linehintCSV, policy, false);
  rawCatalog.registerPlugin(fnameCSV, pgCSV);
  scanCSV = new Scan(&ctx, *pgCSV);

  /*
   * SELECT CSV
   * id > 5000000 and id < 10000000 and bot = 'Bobax'
   */
  Select *selCSV;
  expressions::Expression *predicateCSV;
  {
    list<RecordAttribute> argProjectionsCSV;
    argProjectionsCSV.push_back(*idCSV);
    argProjectionsCSV.push_back(*bot);
    expressions::Expression *arg =
        new expressions::InputArgument(&recCSV, 0, argProjectionsCSV);
    expressions::Expression *selID = new expressions::RecordProjection(
        idCSV->getOriginalType(), arg, *idCSV);
    expressions::Expression *selBot =
        new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicate1, predicate2);

    Select *selNum = new Select(predicateNum, scanCSV);
    scanCSV->setParent(selNum);

    expressions::Expression *predExpr3 =
        new expressions::StringConstant(botName);
    expressions::Expression *predicateStr =
        new expressions::EqExpression(selBot, predExpr3);

    selCSV = new Select(predicateStr, selNum);
    selNum->setParent(selCSV);
  }

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000 and
   * (data->>'size')::int > 1000
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selID = new expressions::RecordProjection(
        idJSON->getOriginalType(), arg, *idJSON);
    expressions::Expression *selSize =
        new expressions::RecordProjection(size->getOriginalType(), arg, *size);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateAnd =
        new expressions::AndExpression(predicate1, predicate2);

    expressions::Expression *predExpr3 = new expressions::IntConstant(sizeLow);
    expressions::Expression *predicate3 =
        new expressions::GtExpression(selSize, predExpr3);

    predicateJSON = new expressions::AndExpression(predicateAnd, predicate3);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idCSV);
  argProjectionsLeft.push_back(*classa);
  argProjectionsLeft.push_back(*classb);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recCSV, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);

  // RIGHT SIDE
  list<RecordAttribute> argProjectionsRight;
  argProjectionsRight.push_back(*idJSON);
  expressions::Expression *rightArg =
      new expressions::InputArgument(&recJSON, 1, argProjectionsRight);
  expressions::Expression *rightPred = new expressions::RecordProjection(
      idJSON->getOriginalType(), rightArg, *idJSON);

  /* join pred. */
  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(leftPred, rightPred);

  /* left materializer - dim needed */
  vector<RecordAttribute *> fieldsLeft;
  fieldsLeft.push_back(classa);
  fieldsLeft.push_back(classb);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnameCSV, activeLoop, pgCSV->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgCSV->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - no explicit field needed */
  vector<RecordAttribute *> fieldsRight;
  vector<materialization_mode> outputModesRight;

  /* explicit mention to right OID */
  RecordAttribute *projTupleR =
      new RecordAttribute(fnameJSON, activeLoop, pgJSON->getOIDType());
  vector<RecordAttribute *> OIDRight;
  OIDRight.push_back(projTupleR);
  expressions::Expression *exprRightOID = new expressions::RecordProjection(
      pgJSON->getOIDType(), rightArg, *projTupleR);
  vector<expression_t> expressionsRight;
  expressionsRight.push_back(exprRightOID);

  Materializer *matRight = new Materializer(fieldsRight, expressionsRight,
                                            OIDRight, outputModesRight);

  char joinLabel[] = "radixJoinCSVJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selCSV, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selCSV->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(classa), MAX(classb), COUNT(*)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*classa);
  argProjections.push_back(*classb);
  expressions::Expression *exprClassa = new expressions::RecordProjection(
      classa->getOriginalType(), leftArg, *classa);
  expressions::Expression *exprClassb = new expressions::RecordProjection(
      classb->getOriginalType(), leftArg, *classb);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprClassa;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprClassb;
  outputExprs.push_back(outputExpr2);

  accs.push_back(SUM);
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr3);
  /* Pred: Redundant */

  expressions::Expression *lhsRed = new expressions::BoolConstant(true);
  expressions::Expression *rhsRed = new expressions::BoolConstant(true);
  expressions::Expression *predRed =
      new expressions::EqExpression(lhsRed, rhsRed);

  opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgCSV->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecCSVJSON3(map<string, dataset> datasetCatalog) {
  // csv
  int idLow = 8000000;
  int idHigh = 10000000;
  string botName = "Unclassified";
  // JSON
  // id, again
  int yearNo = 2012;

  Context &ctx = *prepareContext("symantec-CSV-JSON-3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecCSV = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantecCSV];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  pm::CSVPlugin *pgCSV;
  Scan *scanCSV;
  RecordAttribute *idCSV;
  RecordAttribute *classa;
  RecordAttribute *classb;
  RecordAttribute *bot;
  RecordType recCSV = symantecCSV.recType;
  string fnameCSV = symantecCSV.path;
  int linehintCSV = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  idCSV = argsSymantecCSV["id"];
  classa = argsSymantecCSV["classa"];
  classb = argsSymantecCSV["classb"];
  bot = argsSymantecCSV["bot"];
  vector<RecordAttribute *> projectionsCSV;
  projectionsCSV.push_back(idCSV);
  projectionsCSV.push_back(classa);
  projectionsCSV.push_back(classb);
  projectionsCSV.push_back(bot);

  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
                            linehintCSV, policy, false);
  rawCatalog.registerPlugin(fnameCSV, pgCSV);
  scanCSV = new Scan(&ctx, *pgCSV);

  /*
   * SELECT CSV
   * id > 5000000 and id < 10000000 and bot = 'Unclassified'
   */
  Select *selCSV;
  expressions::Expression *predicateCSV;
  {
    list<RecordAttribute> argProjectionsCSV;
    argProjectionsCSV.push_back(*idCSV);
    argProjectionsCSV.push_back(*bot);
    expressions::Expression *arg =
        new expressions::InputArgument(&recCSV, 0, argProjectionsCSV);
    expressions::Expression *selID = new expressions::RecordProjection(
        idCSV->getOriginalType(), arg, *idCSV);
    expressions::Expression *selBot =
        new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicate1, predicate2);

    Select *selNum = new Select(predicateNum, scanCSV);
    scanCSV->setParent(selNum);

    expressions::Expression *predExpr3 =
        new expressions::StringConstant(botName);
    expressions::Expression *predicateStr =
        new expressions::EqExpression(selBot, predExpr3);

    selCSV = new Select(predicateStr, selNum);
    selNum->setParent(selCSV);
  }

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *day = argsSymantecJSON["day"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000 and
   * (data->'day'->>'year')::int = 2012
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selID = new expressions::RecordProjection(
        idJSON->getOriginalType(), arg, *idJSON);
    expressions::Expression *selDay =
        new expressions::RecordProjection(day->getOriginalType(), arg, *day);
    IntType yearType = IntType();
    RecordAttribute *year =
        new RecordAttribute(2, fnameJSON, "year", &yearType);
    expressions::Expression *selYear =
        new expressions::RecordProjection(&yearType, selDay, *year);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateAnd =
        new expressions::AndExpression(predicate1, predicate2);

    expressions::Expression *predExpr3 = new expressions::IntConstant(yearNo);
    expressions::Expression *predicate3 =
        new expressions::EqExpression(selYear, predExpr3);

    predicateJSON = new expressions::AndExpression(predicateAnd, predicate3);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idCSV);
  argProjectionsLeft.push_back(*classa);
  argProjectionsLeft.push_back(*classb);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recCSV, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);

  // RIGHT SIDE
  list<RecordAttribute> argProjectionsRight;
  argProjectionsRight.push_back(*idJSON);
  expressions::Expression *rightArg =
      new expressions::InputArgument(&recJSON, 1, argProjectionsRight);
  expressions::Expression *rightPred = new expressions::RecordProjection(
      idJSON->getOriginalType(), rightArg, *idJSON);

  /* join pred. */
  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(leftPred, rightPred);

  /* left materializer - dim needed */
  vector<RecordAttribute *> fieldsLeft;
  fieldsLeft.push_back(classa);
  fieldsLeft.push_back(classb);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnameCSV, activeLoop, pgCSV->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgCSV->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - no explicit field needed */
  vector<RecordAttribute *> fieldsRight;
  vector<materialization_mode> outputModesRight;

  /* explicit mention to right OID */
  RecordAttribute *projTupleR =
      new RecordAttribute(fnameJSON, activeLoop, pgJSON->getOIDType());
  vector<RecordAttribute *> OIDRight;
  OIDRight.push_back(projTupleR);
  expressions::Expression *exprRightOID = new expressions::RecordProjection(
      pgJSON->getOIDType(), rightArg, *projTupleR);
  vector<expression_t> expressionsRight;
  expressionsRight.push_back(exprRightOID);

  Materializer *matRight = new Materializer(fieldsRight, expressionsRight,
                                            OIDRight, outputModesRight);

  char joinLabel[] = "radixJoinCSVJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selCSV, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selCSV->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(classa), MAX(classb), COUNT(*)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*classa);
  argProjections.push_back(*classb);
  expressions::Expression *exprClassa = new expressions::RecordProjection(
      classa->getOriginalType(), leftArg, *classa);
  expressions::Expression *exprClassb = new expressions::RecordProjection(
      classb->getOriginalType(), leftArg, *classb);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprClassa;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprClassb;
  outputExprs.push_back(outputExpr2);

  accs.push_back(SUM);
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr3);
  /* Pred: Redundant */

  expressions::Expression *lhsRed = new expressions::BoolConstant(true);
  expressions::Expression *rhsRed = new expressions::BoolConstant(true);
  expressions::Expression *predRed =
      new expressions::EqExpression(lhsRed, rhsRed);

  opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgCSV->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecCSVJSON4(map<string, dataset> datasetCatalog) {
  // csv
  int idLow = 9000000;
  int idHigh = 10000000;
  string botName = "Unclassified";
  int classaHigh = 5;
  // JSON
  // id, again
  int yearNo = 2012;

  Context &ctx = *prepareContext("symantec-CSV-JSON-3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecCSV = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantecCSV];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  pm::CSVPlugin *pgCSV;
  Scan *scanCSV;
  RecordAttribute *idCSV;
  RecordAttribute *classa;
  RecordAttribute *classb;
  RecordAttribute *bot;
  RecordType recCSV = symantecCSV.recType;
  string fnameCSV = symantecCSV.path;
  int linehintCSV = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  idCSV = argsSymantecCSV["id"];
  classa = argsSymantecCSV["classa"];
  classb = argsSymantecCSV["classb"];
  bot = argsSymantecCSV["bot"];
  vector<RecordAttribute *> projectionsCSV;
  projectionsCSV.push_back(idCSV);
  projectionsCSV.push_back(classa);
  projectionsCSV.push_back(classb);
  projectionsCSV.push_back(bot);

  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
                            linehintCSV, policy, false);
  rawCatalog.registerPlugin(fnameCSV, pgCSV);
  scanCSV = new Scan(&ctx, *pgCSV);

  /*
   * SELECT CSV
   * id > 9000000 and id < 10000000 and bot = 'Unclassified' and classa < 5
   */
  Select *selCSV;
  expressions::Expression *predicateCSV;
  {
    list<RecordAttribute> argProjectionsCSV;
    argProjectionsCSV.push_back(*idCSV);
    argProjectionsCSV.push_back(*classa);
    argProjectionsCSV.push_back(*bot);
    expressions::Expression *arg =
        new expressions::InputArgument(&recCSV, 0, argProjectionsCSV);
    expressions::Expression *selID = new expressions::RecordProjection(
        idCSV->getOriginalType(), arg, *idCSV);
    expressions::Expression *selClassa = new expressions::RecordProjection(
        classa->getOriginalType(), arg, *classa);
    expressions::Expression *selBot =
        new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predExpr3 =
        new expressions::IntConstant(classaHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicate3 =
        new expressions::LtExpression(selClassa, predExpr3);

    expressions::Expression *predicateNum_ =
        new expressions::AndExpression(predicate1, predicate2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicateNum_, predicate3);

    Select *selNum = new Select(predicateNum, scanCSV);
    scanCSV->setParent(selNum);

    expressions::Expression *predExpr4 =
        new expressions::StringConstant(botName);
    expressions::Expression *predicateStr =
        new expressions::EqExpression(selBot, predExpr4);

    selCSV = new Select(predicateStr, selNum);
    selNum->setParent(selCSV);
  }

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *day = argsSymantecJSON["day"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(day);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int > 9000000 and (data->>'id')::int < 10000000 and
   * (data->'day'->>'year')::int = 2012
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selID = new expressions::RecordProjection(
        idJSON->getOriginalType(), arg, *idJSON);
    expressions::Expression *selDay =
        new expressions::RecordProjection(day->getOriginalType(), arg, *day);
    IntType yearType = IntType();
    RecordAttribute *year =
        new RecordAttribute(2, fnameJSON, "year", &yearType);
    expressions::Expression *selYear =
        new expressions::RecordProjection(&yearType, selDay, *year);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateAnd =
        new expressions::AndExpression(predicate1, predicate2);

    expressions::Expression *predExpr3 = new expressions::IntConstant(yearNo);
    expressions::Expression *predicate3 =
        new expressions::EqExpression(selYear, predExpr3);

    predicateJSON = new expressions::AndExpression(predicateAnd, predicate3);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idCSV);
  argProjectionsLeft.push_back(*classa);
  argProjectionsLeft.push_back(*classb);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recCSV, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);

  // RIGHT SIDE
  list<RecordAttribute> argProjectionsRight;
  argProjectionsRight.push_back(*idJSON);
  expressions::Expression *rightArg =
      new expressions::InputArgument(&recJSON, 1, argProjectionsRight);
  expressions::Expression *rightPred = new expressions::RecordProjection(
      idJSON->getOriginalType(), rightArg, *idJSON);

  /* join pred. */
  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(leftPred, rightPred);

  /* left materializer - dim needed */
  vector<RecordAttribute *> fieldsLeft;
  fieldsLeft.push_back(classa);
  fieldsLeft.push_back(classb);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnameCSV, activeLoop, pgCSV->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgCSV->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - no explicit field needed */
  vector<RecordAttribute *> fieldsRight;
  vector<materialization_mode> outputModesRight;

  /* explicit mention to right OID */
  RecordAttribute *projTupleR =
      new RecordAttribute(fnameJSON, activeLoop, pgJSON->getOIDType());
  vector<RecordAttribute *> OIDRight;
  OIDRight.push_back(projTupleR);
  expressions::Expression *exprRightOID = new expressions::RecordProjection(
      pgJSON->getOIDType(), rightArg, *projTupleR);
  vector<expression_t> expressionsRight;
  expressionsRight.push_back(exprRightOID);

  Materializer *matRight = new Materializer(fieldsRight, expressionsRight,
                                            OIDRight, outputModesRight);

  char joinLabel[] = "radixJoinCSVJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selCSV, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selCSV->setParent(join);
  selJSON->setParent(join);

  /**
   * NEST
   * GroupBy: classa
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: max(classb), count(*)
   */
  expressions::Expression *selClassa = new expressions::RecordProjection(
      classa->getOriginalType(), leftArg, *classa);

  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*classa);
  nestProjections.push_back(*classb);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&recCSV, 0, nestProjections);

  // f (& g) -> GROUPBY cluster
  expressions::RecordProjection *f = new expressions::RecordProjection(
      classa->getOriginalType(), nestArg, *classa);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  fields.push_back(classa);
  fields.push_back(classb);
  outputModes.insert(outputModes.begin(), EAGER);
  outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;

  /* Aggregate 1: MAX(classb) */
  expressions::Expression *aggrSize = new expressions::RecordProjection(
      classb->getOriginalType(), leftArg, *classb);
  expressions::Expression *outputExpr1 = aggrSize;
  aggrField1 = string("_maxClassb");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  expressions::Expression *aggrCount = new expressions::IntConstant(1);
  expressions::Expression *outputExpr2 = aggrCount;
  aggrField2 = string("_Cnt");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, join, nestLabel, *mat);
  join->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  llvm::Function *debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &floatType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&floatType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugFloat, nestOutput1, nestOp);
  nestOp->setParent(printOp1);

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
  pgCSV->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecCSVJSON5(map<string, dataset> datasetCatalog) {
  // csv
  int idHigh = 100000;
  // JSON
  // id, again

  Context &ctx = *prepareContext("symantec-CSV-JSON-5");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecCSV = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantecCSV];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  pm::CSVPlugin *pgCSV;
  Scan *scanCSV;
  RecordAttribute *idCSV;
  RecordAttribute *classa;
  RecordType recCSV = symantecCSV.recType;
  string fnameCSV = symantecCSV.path;
  int linehintCSV = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  idCSV = argsSymantecCSV["id"];
  classa = argsSymantecCSV["classa"];
  vector<RecordAttribute *> projectionsCSV;
  projectionsCSV.push_back(idCSV);
  projectionsCSV.push_back(classa);

  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
                            linehintCSV, policy, false);
  rawCatalog.registerPlugin(fnameCSV, pgCSV);
  scanCSV = new Scan(&ctx, *pgCSV);

  /*
   * SELECT CSV
   * id < 100000
   */
  Select *selCSV;
  expressions::Expression *predicateCSV;
  {
    list<RecordAttribute> argProjectionsCSV;
    argProjectionsCSV.push_back(*idCSV);
    expressions::Expression *arg =
        new expressions::InputArgument(&recCSV, 0, argProjectionsCSV);
    expressions::Expression *selID = new expressions::RecordProjection(
        idCSV->getOriginalType(), arg, *idCSV);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    predicateCSV = new expressions::LtExpression(selID, predExpr1);

    selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);
  }

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *uri = argsSymantecJSON["uri"];
  RecordAttribute *size = argsSymantecJSON["size"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int < 100000
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selID = new expressions::RecordProjection(
        idJSON->getOriginalType(), arg, *idJSON);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    predicateJSON = new expressions::LtExpression(selID, predExpr1);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idCSV);
  argProjectionsLeft.push_back(*classa);
  argProjectionsLeft.push_back(*size);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recCSV, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);

  // RIGHT SIDE
  list<RecordAttribute> argProjectionsRight;
  argProjectionsRight.push_back(*idJSON);
  expressions::Expression *rightArg =
      new expressions::InputArgument(&recJSON, 1, argProjectionsRight);
  expressions::Expression *rightPred = new expressions::RecordProjection(
      idJSON->getOriginalType(), rightArg, *idJSON);

  /* join pred. */
  expressions::BinaryExpression *joinPred =
      new expressions::EqExpression(leftPred, rightPred);

  /* left materializer - dim needed */
  vector<RecordAttribute *> fieldsLeft;
  fieldsLeft.push_back(classa);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnameCSV, activeLoop, pgCSV->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgCSV->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idCSV->getOriginalType(), leftArg, *idCSV);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - no explicit field needed */
  vector<RecordAttribute *> fieldsRight;
  vector<materialization_mode> outputModesRight;

  /* explicit mention to right OID */
  RecordAttribute *projTupleR =
      new RecordAttribute(fnameJSON, activeLoop, pgJSON->getOIDType());
  vector<RecordAttribute *> OIDRight;
  OIDRight.push_back(projTupleR);
  expressions::Expression *exprRightOID = new expressions::RecordProjection(
      pgJSON->getOIDType(), rightArg, *projTupleR);
  vector<expression_t> expressionsRight;
  expressionsRight.push_back(exprRightOID);

  Materializer *matRight = new Materializer(fieldsRight, expressionsRight,
                                            OIDRight, outputModesRight);

  char joinLabel[] = "radixJoinCSVJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selCSV, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selCSV->setParent(join);
  selJSON->setParent(join);

  /* OUTER Unnest -> some entries have no URIs */
  list<RecordAttribute> unnestProjections = list<RecordAttribute>();
  unnestProjections.push_back(*uri);

  expressions::Expression *outerArg =
      new expressions::InputArgument(&recJSON, 0, unnestProjections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(uri->getOriginalType(), outerArg, *uri);
  string nestedName = activeLoop;
  Path path = Path(nestedName, proj);

  expressions::Expression *lhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *rhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *predUnnest =
      new expressions::EqExpression(lhsUnnest, rhsUnnest);

  OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, join);
  join->setParent(unnestOp);

  /*
   * NULL FILTER!
   * Acts as mini-nest operator
   */
  StringType nestedType = StringType();
  RecordAttribute recUnnested =
      RecordAttribute(2, fnameJSON + ".uri", nestedName, &nestedType);
  list<RecordAttribute *> attsUnnested = list<RecordAttribute *>();
  attsUnnested.push_back(&recUnnested);
  RecordType unnestedType = RecordType(attsUnnested);

  list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
  nullFilterProjections.push_back(recUnnested);

  expressions::InputArgument *nullFilterArg =
      new expressions::InputArgument(&nestedType, 0, nullFilterProjections);
  NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
  unnestOp->setParent(nullFilter);

  /**
   * REDUCE
   * MAX(size), MAX(classa), COUNT(*)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*size);
  argProjections.push_back(*classa);
  expressions::Expression *exprClassa = new expressions::RecordProjection(
      classa->getOriginalType(), leftArg, *classa);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), leftArg, *size);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprSize;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprClassa;
  outputExprs.push_back(outputExpr2);

  accs.push_back(SUM);
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr3);

  /* Pred: Redundant */

  expressions::Expression *lhsRed = new expressions::BoolConstant(true);
  expressions::Expression *rhsRed = new expressions::BoolConstant(true);
  expressions::Expression *predRed =
      new expressions::EqExpression(lhsRed, rhsRed);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predRed, nullFilter, &ctx);
  nullFilter->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgCSV->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecJSONWarmup(map<string, dataset> datasetCatalog) {
  Context &ctx = *prepareContext("symantec-json-warmup");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantec = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantec = symantec.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = symantec.path;
  RecordType rec = symantec.recType;
  int linehint = symantec.linehint;

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   * (SUM 1)
   */
  list<RecordAttribute> argProjections;

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);

  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, scan, &ctx);
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

void symantecCSVWarmup(map<string, dataset> datasetCatalog) {
  Context &ctx = *prepareContext("symantec-csv-warmup");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecCSV");
  dataset symantec = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantec = symantec.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = symantec.path;
  RecordType rec = symantec.recType;
  int linehint = symantec.linehint;
  int policy = 5;
  char delimInner = ';';

  vector<RecordAttribute *> projections;

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, linehint, policy, false);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   * (SUM 1)
   */
  list<RecordAttribute> argProjections;

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  expressions::Expression *outputExpr = new expressions::IntConstant(1);

  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, scan, &ctx);
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
