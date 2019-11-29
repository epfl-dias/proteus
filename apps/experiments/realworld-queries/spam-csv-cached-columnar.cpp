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

#include "experiments/realworld-queries/spam-csv-cached-columnar.hpp"

void symantecCSV3v1(map<string, dataset> datasetCatalog) {
  string botName = "DARKMAILER3";
  int idLow = 100000000;
  int idHigh = 200000000;

  Context &ctx = *prepareContext("symantec-csv-3v1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  string fname = symantecCSV.path;
  RecordType rec = symantecCSV.recType;
  int linehint = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';

  RecordAttribute *classb = argsSymantecCSV["classb"];
  RecordAttribute *bot = argsSymantecCSV["bot"];
  RecordAttribute *id = argsSymantecCSV["id"];

  vector<RecordAttribute *> projections;
  projections.push_back(bot);
  //    projections.push_back(classb);

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, linehint, policy, false);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  expressions::Expression *argSel =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), argSel, *id);
  expressions::Expression *predExprNum1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExprNum2 = new expressions::IntConstant(idHigh);
  expressions::Expression *predicateNum1 =
      new expressions::GtExpression(selID, predExprNum1);
  expressions::Expression *predicateNum2 =
      new expressions::LtExpression(selID, predExprNum2);
  expressions::Expression *predicateSel =
      new expressions::AndExpression(predicateNum1, predicateNum2);

  Select *sel = new Select(predicateSel, scan);
  scan->setParent(sel);

  /**
   * REDUCE
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*classb);
  argProjections.push_back(*bot);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = new expressions::RecordProjection(
      classb->getOriginalType(), arg, *classb);
  outputExprs.push_back(outputExpr1);

  accs.push_back(SUM);
  expressions::Expression *outputExpr2 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr2);

  /* Pred: */
  expressions::Expression *selBot =
      new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);
  expressions::Expression *predExpr1 = new expressions::StringConstant(botName);
  expressions::Expression *predicate =
      new expressions::EqExpression(selBot, predExpr1);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, sel, &ctx);
  sel->setParent(reduce);

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

// XXX FIXME Crashes
// select max(classa), max(classb), count(*) as COUNTER from spamsclasses400m
// where id > 40000000 and id < 50000000 group by country_code;
void symantecCSV4v1(map<string, dataset> datasetCatalog) {
  int idLow = 40000000;
  int idHigh = 50000000;
  //    string botName = "Bobax";
  Context &ctx = *prepareContext("symantec-csv-4(agg)");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  string fname = symantecCSV.path;
  RecordType rec = symantecCSV.recType;
  int linehint = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  RecordAttribute *id = argsSymantecCSV["id"];
  RecordAttribute *classa = argsSymantecCSV["classa"];
  RecordAttribute *classb = argsSymantecCSV["classb"];
  //    RecordAttribute *bot = argsSymantecCSV["bot"];
  RecordAttribute *country_code = argsSymantecCSV["country_code"];

  vector<RecordAttribute *> projections;
  //    projections.push_back(id);
  //    projections.push_back(classa);
  //    projections.push_back(classb);
  projections.push_back(country_code);
  //    projections.push_back(bot);

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, linehint, policy, false);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   * id > 40000000 and id < 50000000 and bot = 'Bobax'
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  //    argSelections.push_back(*bot);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  //    expressions::Expression* selBot      =
  //                    new
  //                    expressions::RecordProjection(bot->getOriginalType(),arg,*bot);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
  //    expressions::Expression* predExpr3 = new expressions::StringConstant(
  //            botName);

  expressions::Expression *predicate1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicateNum =
      new expressions::AndExpression(predicate1, predicate2);

  //    expressions::Expression* predicateStr = new expressions::EqExpression(
  //                selBot, predExpr3);

  Select *sel = new Select(predicateNum, scan);
  scan->setParent(sel);

  //    //rest of preds
  //    Select *sel = new Select(predicateStr, selNum);
  //    selNum->setParent(sel);

  /**
   * NEST
   * GroupBy: country_code
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: max(classa), max(classb), count(*)
   */
  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*classa);
  nestProjections.push_back(*classb);
  nestProjections.push_back(*country_code);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY country_code
  expressions::RecordProjection *f = new expressions::RecordProjection(
      country_code->getOriginalType(), nestArg, *country_code);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  fields.push_back(classa);
  outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(classb);
  outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(country_code);
  outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;
  string aggrField3;

  /* Aggregate 1: MAX(classa) */
  expressions::Expression *aggrClassa = new expressions::RecordProjection(
      classa->getOriginalType(), arg, *classa);
  expressions::Expression *outputExpr1 = aggrClassa;
  aggrField1 = string("_maxClassA");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  /* Aggregate 2: MAX(classb) */
  expressions::Expression *outputExpr2 = new expressions::RecordProjection(
      classb->getOriginalType(), arg, *classb);
  aggrField2 = string("_maxClassB");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  /* Aggregate 3: COUNT(*) */
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  aggrField3 = string("_aggrCount");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr3);
  aggrLabels.push_back(aggrField3);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  llvm::Function *debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  nestOp->setParent(printOp1);

  RecordAttribute *toOutput2 =
      new RecordAttribute(2, aggrLabel, aggrField2, &floatType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&floatType, nestArg, *toOutput2);
  Print *printOp2 = new Print(debugFloat, nestOutput2, printOp1);
  printOp1->setParent(printOp2);

  RecordAttribute *toOutput3 =
      new RecordAttribute(3, aggrLabel, aggrField3, &intType);
  expressions::RecordProjection *nestOutput3 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput3);
  Print *printOp3 = new Print(debugInt, nestOutput3, printOp2);
  printOp2->setParent(printOp3);
  lastPrintOp = printOp3;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce();
  //    reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}

// select max(classb), count(*) as COUNTER from spamsclasses400m  where id >
// 40000000 and id < 50000000 and classa > 105 and classa < 125 and bot =
// 'Lethic' group by classa;
void symantecCSV4v2(map<string, dataset> datasetCatalog) {
  int idLow = 40000000;
  int idHigh = 50000000;
  int classaLow = 105;
  int classaHigh = 125;
  string botName = "Lethic";
  Context &ctx = *prepareContext("symantec-csv-4(agg)");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecCSV");
  dataset symantecCSV = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecCSV =
      symantecCSV.recType.getArgsMap();

  /**
   * SCAN CSV FILE
   */
  string fname = symantecCSV.path;
  RecordType rec = symantecCSV.recType;
  int linehint = symantecCSV.linehint;
  int policy = 5;
  char delimInner = ';';
  RecordAttribute *id = argsSymantecCSV["id"];
  RecordAttribute *classa = argsSymantecCSV["classa"];
  RecordAttribute *classb = argsSymantecCSV["classb"];
  RecordAttribute *bot = argsSymantecCSV["bot"];
  //    RecordAttribute *country_code = argsSymantecCSV["country_code"];

  vector<RecordAttribute *> projections;
  //    projections.push_back(id);
  //    projections.push_back(classa);
  //    projections.push_back(classb);
  //    projections.push_back(country_code);
  projections.push_back(bot);

  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
                                        delimInner, linehint, policy, false);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * SELECT
   * id > 40000000 and id < 50000000 and bot = 'Bobax'
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*bot);
  argSelections.push_back(*classa);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selClassa = new expressions::RecordProjection(
      classa->getOriginalType(), arg, *classa);
  expressions::Expression *selBot =
      new expressions::RecordProjection(bot->getOriginalType(), arg, *bot);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);

  expressions::Expression *predExpr3 = new expressions::IntConstant(classaLow);
  expressions::Expression *predExpr4 = new expressions::IntConstant(classaHigh);

  expressions::Expression *predExpr5 = new expressions::StringConstant(botName);

  expressions::Expression *predicateNum1 =
      new expressions::GtExpression(selID, predExpr1);
  expressions::Expression *predicateNum2 =
      new expressions::LtExpression(selID, predExpr2);
  expressions::Expression *predicateNum3 =
      new expressions::GtExpression(selClassa, predExpr3);
  expressions::Expression *predicateNum4 =
      new expressions::LtExpression(selClassa, predExpr4);

  Select *selNum1 = new Select(predicateNum1, scan);
  scan->setParent(selNum1);

  Select *selNum2 = new Select(predicateNum2, selNum1);
  selNum1->setParent(selNum2);

  Select *selNum3 = new Select(predicateNum3, selNum2);
  selNum2->setParent(selNum3);

  Select *selNum4 = new Select(predicateNum4, selNum3);
  selNum3->setParent(selNum4);

  expressions::Expression *predicateStr =
      new expressions::EqExpression(selBot, predExpr5);

  // rest of preds
  Select *sel = new Select(predicateStr, selNum4);
  selNum4->setParent(sel);

  /**
   * NEST
   * GroupBy: classa
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: max(classa), max(classb), count(*)
   */
  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*classa);
  nestProjections.push_back(*classb);
  //    nestProjections.push_back(*country_code);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY country_code
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
  //    fields.push_back(classa);
  //    outputModes.insert(outputModes.begin(), EAGER);
  fields.push_back(classb);
  outputModes.insert(outputModes.begin(), EAGER);
  //    fields.push_back(country_code);
  //    outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;
  string aggrField3;

  /* Aggregate 1: MAX(classa) */
  //    expressions::Expression* aggrClassa = new expressions::RecordProjection(
  //            classa->getOriginalType(), arg, *classa);
  //    expressions::Expression* outputExpr1 = aggrClassa;
  //    aggrField1 = string("_maxClassA");
  //    accs.push_back(MAX);
  //    outputExprs.push_back(outputExpr1);
  //    aggrLabels.push_back(aggrField1);

  /* Aggregate 2: MAX(classb) */
  expressions::Expression *outputExpr2 = new expressions::RecordProjection(
      classb->getOriginalType(), arg, *classb);
  aggrField2 = string("_maxClassB");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr2);
  aggrLabels.push_back(aggrField2);

  /* Aggregate 3: COUNT(*) */
  expressions::Expression *outputExpr3 = new expressions::IntConstant(1);
  aggrField3 = string("_aggrCount");
  accs.push_back(SUM);
  outputExprs.push_back(outputExpr3);
  aggrLabels.push_back(aggrField3);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  llvm::Function *debugFloat = ctx.getFunction("printFloat");
  IntType intType = IntType();
  FloatType floatType = FloatType();

  /* OUTPUT */
  Operator *lastPrintOp;
  //    RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel,
  //    aggrField1,
  //            &intType);
  //    expressions::RecordProjection* nestOutput1 =
  //            new expressions::RecordProjection(&intType, nestArg,
  //            *toOutput1);
  //    Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  //    nestOp->setParent(printOp1);

  RecordAttribute *toOutput2 =
      new RecordAttribute(2, aggrLabel, aggrField2, &floatType);
  expressions::RecordProjection *nestOutput2 =
      new expressions::RecordProjection(&floatType, nestArg, *toOutput2);
  Print *printOp2 = new Print(debugFloat, nestOutput2, nestOp);
  nestOp->setParent(printOp2);

  RecordAttribute *toOutput3 =
      new RecordAttribute(3, aggrLabel, aggrField3, &intType);
  expressions::RecordProjection *nestOutput3 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput3);
  Print *printOp3 = new Print(debugInt, nestOutput3, printOp2);
  printOp2->setParent(printOp3);
  lastPrintOp = printOp3;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce();
  //    reduce->produce();
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
