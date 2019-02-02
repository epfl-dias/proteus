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

#include "experiments/realworld-vldb/spam-bin-json.hpp"

void symantecBinJSON1(map<string, dataset> datasetCatalog) {
  // bin
  int idLow = 100000000;
  int idHigh = 200000000;
  int dimHigh = 3;
  int clusterLow = 490;
  int clusterHighIncl = 500;
  // JSON
  // id, again

  Context &ctx = *prepareContext("symantec-bin-JSON-1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *dim;
  RecordAttribute *cluster;
  RecordAttribute *mdc;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  cluster = argsSymantecBin["cluster"];
  dim = argsSymantecBin["dim"];
  mdc = argsSymantecBin["mdc"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(cluster);
  projectionsBin.push_back(dim);
  projectionsBin.push_back(mdc);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /*
   * SELECT BINARY
   * id > 5000000 and id < 10000000 and dim < 3 AND cluster > 490 AND cluster <=
   * 500
   */
  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    argProjectionsBin.push_back(*cluster);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selID = new expressions::RecordProjection(
        idBin->getOriginalType(), arg, *idBin);
    expressions::Expression *selDim =
        new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
    expressions::Expression *selCluster = new expressions::RecordProjection(
        cluster->getOriginalType(), arg, *cluster);

    // id
    expressions::Expression *predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression *predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::GtExpression(selID, predExpr1);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selID, predExpr2);
    expressions::Expression *predicateAnd1 =
        new expressions::AndExpression(predicate1, predicate2);

    expressions::Expression *predExpr3 = new expressions::IntConstant(dimHigh);
    expressions::Expression *predExpr4 =
        new expressions::IntConstant(clusterLow);
    expressions::Expression *predExpr5 =
        new expressions::IntConstant(clusterHighIncl);

    // dim
    expressions::Expression *predicate3 =
        new expressions::LtExpression(selDim, predExpr3);
    // cluster
    expressions::Expression *predicate4 =
        new expressions::GtExpression(selCluster, predExpr4);
    expressions::Expression *predicateAnd2 =
        new expressions::AndExpression(predicate3, predicate4);

    expressions::Expression *predicate5 =
        new expressions::LeExpression(selCluster, predExpr5);

    expressions::Expression *predicateAnd3 =
        new expressions::AndExpression(predicateAnd2, predicate5);

    predicateBin = new expressions::AndExpression(predicateAnd1, predicateAnd3);
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

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
  projectionsJSON.push_back(size);

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
  argProjectionsLeft.push_back(*idBin);
  argProjectionsLeft.push_back(*mdc);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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
  fieldsLeft.push_back(mdc);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(mdc), MAX(size)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*mdc);
  expressions::Expression *exprMDC =
      new expressions::RecordProjection(mdc->getOriginalType(), leftArg, *mdc);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), rightArg, *size);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprMDC;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprSize;
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
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON2(map<string, dataset> datasetCatalog) {
  // bin
  int idHigh = 100000000;
  int dimHigh = 3;
  int clusterLow = 490;
  int clusterHighIncl = 500;
  // JSON
  // id, again

  Context &ctx = *prepareContext("symantec-bin-JSON-2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *dim;
  RecordAttribute *cluster;
  RecordAttribute *mdc;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  cluster = argsSymantecBin["cluster"];
  dim = argsSymantecBin["dim"];
  mdc = argsSymantecBin["mdc"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(cluster);
  projectionsBin.push_back(dim);
  projectionsBin.push_back(mdc);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /*
   * SELECT BINARY
   * id > 5000000 and id < 10000000 and dim < 3 AND cluster > 490 AND cluster <=
   * 500
   */
  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    argProjectionsBin.push_back(*cluster);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selID = new expressions::RecordProjection(
        idBin->getOriginalType(), arg, *idBin);
    expressions::Expression *selDim =
        new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);
    expressions::Expression *selCluster = new expressions::RecordProjection(
        cluster->getOriginalType(), arg, *cluster);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::LtExpression(selID, predExpr1);

    expressions::Expression *predExpr3 = new expressions::IntConstant(dimHigh);
    expressions::Expression *predExpr4 =
        new expressions::IntConstant(clusterLow);
    expressions::Expression *predExpr5 =
        new expressions::IntConstant(clusterHighIncl);

    expressions::Expression *predicate3 =
        new expressions::LtExpression(selDim, predExpr3);
    expressions::Expression *predicate4 =
        new expressions::GtExpression(selCluster, predExpr4);
    expressions::Expression *predicateAnd2 =
        new expressions::AndExpression(predicate3, predicate4);

    expressions::Expression *predicate5 =
        new expressions::LeExpression(selCluster, predExpr5);

    expressions::Expression *predicateAnd3 =
        new expressions::AndExpression(predicateAnd2, predicate5);

    predicateBin = new expressions::AndExpression(predicate1, predicateAnd3);
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

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
  projectionsJSON.push_back(size);

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
  argProjectionsLeft.push_back(*idBin);
  argProjectionsLeft.push_back(*mdc);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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
  fieldsLeft.push_back(mdc);
  vector<materialization_mode> outputModesLeft;
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(mdc), MAX(size)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*mdc);
  expressions::Expression *exprMDC =
      new expressions::RecordProjection(mdc->getOriginalType(), leftArg, *mdc);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), rightArg, *size);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(MAX);
  expressions::Expression *outputExpr1 = exprMDC;
  outputExprs.push_back(outputExpr1);

  accs.push_back(MAX);
  expressions::Expression *outputExpr2 = exprSize;
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
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON3(map<string, dataset> datasetCatalog) {
  // bin: Nothing yet
  // JSON
  int idHigh = 1000000;
  int sizeHigh = 1000;
  string langName = "german";

  Context &ctx = *prepareContext("symantec-bin-JSON-3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  RecordAttribute *lang = argsSymantecJSON["lang"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(size);
  projectionsJSON.push_back(lang);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int < 1000000 and (data->>'size')::int < 1000 and
   * (data->>'lang') = 'german'
   */
  Select *selJSON;
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
    expressions::Expression *selLang =
        new expressions::RecordProjection(lang->getOriginalType(), arg, *lang);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::LtExpression(selID, predExpr1);

    expressions::Expression *predExpr2 = new expressions::IntConstant(sizeHigh);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selSize, predExpr2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicate1, predicate2);

    expressions::Expression *predExpr3 =
        new expressions::StringConstant(langName);
    expressions::Expression *predicateStr =
        new expressions::EqExpression(selLang, predExpr3);
    Select *selNum = new Select(predicateNum, scanJSON);
    scanJSON->setParent(selNum);

    selJSON = new Select(predicateStr, selNum);
    selNum->setParent(selJSON);
  }

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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

  /* left materializer - no fields needed */
  vector<RecordAttribute *> fieldsLeft;
  vector<materialization_mode> outputModesLeft;

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, scanBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  scanBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * COUNT(*)
   */
  list<RecordAttribute> argProjections;

  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON3v1(map<string, dataset> datasetCatalog) {
  // bin: Nothing yet
  // JSON
  int idHigh = 160000000;
  int sizeHigh = 1000;
  //    string langName = "german";

  Context &ctx = *prepareContext("symantec-bin-JSON-3v1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  //    RecordAttribute *lang = argsSymantecJSON["lang"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(size);
  //    projectionsJSON.push_back(lang);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int < 1000000 and (data->>'size')::int < 1000 and
   * (data->>'lang') = 'german'
   */
  Select *selJSON;
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
    //        expressions::Expression* selLang = new
    //        expressions::RecordProjection(
    //                                lang->getOriginalType(), arg, *lang);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::LtExpression(selID, predExpr1);

    expressions::Expression *predExpr2 = new expressions::IntConstant(sizeHigh);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selSize, predExpr2);
    expressions::Expression *predicateNum =
        new expressions::AndExpression(predicate1, predicate2);

    //        expressions::Expression* predExpr3 = new
    //        expressions::StringConstant(
    //                                langName);
    //                expressions::Expression* predicateStr = new
    //                expressions::EqExpression(
    //                                selLang, predExpr3);
    Select *selNum = new Select(predicateNum, scanJSON);
    scanJSON->setParent(selNum);

    //        selJSON = new Select(predicateStr, selNum);
    //        selNum->setParent(selJSON);
    selJSON = selNum;
  }

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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

  /* left materializer - no fields needed */
  vector<RecordAttribute *> fieldsLeft;
  vector<materialization_mode> outputModesLeft;

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, scanBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  scanBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * COUNT(*)
   */
  list<RecordAttribute> argProjections;

  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

/* more stuff projected / more selections compared to v1 */
// select max(mdc), max(size), count(*)
// FROM symantecunordered st, spamscoreiddates28m sj
// where st.id = (data->>'id')::int and st.id < 8000000 and (data->>'id')::int <
// 8000000 and (data->>'size')::int < 1000;
void symantecBinJSON3v2(map<string, dataset> datasetCatalog) {
  // bin: Nothing yet
  // JSON
  int idHigh = 160000000;
  int sizeHigh = 1000;
  //    string langName = "german";

  Context &ctx = *prepareContext("symantec-bin-JSON-3v2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *mdc;
  RecordAttribute *sizeBin;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  sizeBin = argsSymantecBin["size"];
  mdc = argsSymantecBin["mdc"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(sizeBin);
  projectionsBin.push_back(mdc);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selID = new expressions::RecordProjection(
        idBin->getOriginalType(), arg, *idBin);

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::LtExpression(selID, predExpr1);

    predicateBin = predicate1;
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  //    RecordAttribute *lang = argsSymantecJSON["lang"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(size);
  //    projectionsJSON.push_back(lang);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->>'id')::int < 8000000 and (data->>'size')::int < 1000
   */
  Select *selJSON;
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

    expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
    expressions::Expression *predicate1 =
        new expressions::LtExpression(selID, predExpr1);

    expressions::Expression *predExpr2 = new expressions::IntConstant(sizeHigh);
    expressions::Expression *predicate2 =
        new expressions::LtExpression(selSize, predExpr2);

    /* size is more selective here */
    Select *selNum1 = new Select(predicate2, scanJSON);
    scanJSON->setParent(selNum1);

    Select *selNum2 = new Select(predicate1, selNum1);
    selNum1->setParent(selNum2);

    selJSON = selNum2;
  }

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  argProjectionsLeft.push_back(*mdc);
  argProjectionsLeft.push_back(*sizeBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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

  /* left materializer - no fields needed */
  vector<RecordAttribute *> fieldsLeft;
  vector<materialization_mode> outputModesLeft;
  fieldsLeft.push_back(mdc);
  fieldsLeft.push_back(sizeBin);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
  outputModesLeft.insert(outputModesLeft.begin(), EAGER);

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * COUNT(*)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*mdc);
  argProjections.push_back(*sizeBin);
  expressions::Expression *exprMDC =
      new expressions::RecordProjection(mdc->getOriginalType(), leftArg, *mdc);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      sizeBin->getOriginalType(), leftArg, *sizeBin);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  //    accs.push_back(MAX);
  //    expressions::Expression* outputExpr1 = exprMDC;
  //    outputExprs.push_back(outputExpr1);
  //
  //    accs.push_back(MAX);
  //    expressions::Expression* outputExpr2 = exprSize;
  //    outputExprs.push_back(outputExpr2);

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
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON4(map<string, dataset> datasetCatalog) {
  // bin
  int dimNo = 7;
  // JSON
  int monthNo = 12;

  Context &ctx = *prepareContext("symantec-bin-JSON-4");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *dim;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  dim = argsSymantecBin["dim"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(dim);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /*
   * SELECT BINARY
   * dim = 7
   */
  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*dim);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selDim =
        new expressions::RecordProjection(dim->getOriginalType(), arg, *dim);

    expressions::Expression *predExpr1 = new expressions::IntConstant(dimNo);
    predicateBin = new expressions::EqExpression(selDim, predExpr1);
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *day = argsSymantecJSON["day"];
  RecordAttribute *size = argsSymantecJSON["size"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(day);
  projectionsJSON.push_back(size);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->'day'->>'month')::int = 12
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*day);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selDay =
        new expressions::RecordProjection(day->getOriginalType(), arg, *day);
    IntType monthType = IntType();
    RecordAttribute *month =
        new RecordAttribute(2, fnameJSON, "month", &monthType);
    expressions::Expression *selMonth =
        new expressions::RecordProjection(&monthType, selDay, *month);

    expressions::Expression *predExpr1 = new expressions::IntConstant(monthNo);
    predicateJSON = new expressions::EqExpression(selMonth, predExpr1);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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
  vector<materialization_mode> outputModesLeft;

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(size)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*size);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), rightArg, *size);

  ReduceNoPred *reduce = new ReduceNoPred(MAX, exprSize, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON5(map<string, dataset> datasetCatalog) {
  // bin
  int sizeLow = 1000;
  // JSON
  int yearNo = 2010;

  Context &ctx = *prepareContext("symantec-bin-JSON-5");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *sizeBin;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  sizeBin = argsSymantecBin["size"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(sizeBin);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /*
   * SELECT BINARY
   * size > 1000
   */
  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*sizeBin);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selSize = new expressions::RecordProjection(
        sizeBin->getOriginalType(), arg, *sizeBin);

    expressions::Expression *predExpr1 = new expressions::IntConstant(sizeLow);
    predicateBin = new expressions::GtExpression(selSize, predExpr1);
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *day = argsSymantecJSON["day"];
  RecordAttribute *size = argsSymantecJSON["size"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(size);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->'day'->>'year')::int = 2010
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    argProjectionsJSON.push_back(*day);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selDay =
        new expressions::RecordProjection(day->getOriginalType(), arg, *day);
    IntType yearType = IntType();
    RecordAttribute *year =
        new RecordAttribute(2, fnameJSON, "year", &yearType);
    expressions::Expression *selYear =
        new expressions::RecordProjection(&yearType, selDay, *year);

    expressions::Expression *predExpr1 = new expressions::IntConstant(yearNo);
    predicateJSON = new expressions::EqExpression(selYear, predExpr1);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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
  vector<materialization_mode> outputModesLeft;

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(size)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*size);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), rightArg, *size);

  ReduceNoPred *reduce = new ReduceNoPred(MAX, exprSize, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}

void symantecBinJSON5v1(map<string, dataset> datasetCatalog) {
  // bin
  // XXX Filter was way too permissive
  //    int sizeLow = 1000;
  int sizeLow = 900000000;
  // JSON
  int yearNo = 2010;

  Context &ctx = *prepareContext("symantec-bin-JSON-5");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantecBin = string("symantecBin");
  dataset symantecBin = datasetCatalog[nameSymantecBin];
  map<string, RecordAttribute *> argsSymantecBin =
      symantecBin.recType.getArgsMap();

  string nameSymantecJSON = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantecJSON];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN BINARY FILE
   */
  BinaryColPlugin *pgBin;
  Scan *scanBin;
  RecordAttribute *idBin;
  RecordAttribute *sizeBin;
  RecordType recBin = symantecBin.recType;
  string fnamePrefixBin = symantecBin.path;
  int linehintBin = symantecBin.linehint;
  idBin = argsSymantecBin["id"];
  sizeBin = argsSymantecBin["size"];
  vector<RecordAttribute *> projectionsBin;
  projectionsBin.push_back(idBin);
  projectionsBin.push_back(sizeBin);

  pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
  rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
  scanBin = new Scan(&ctx, *pgBin);

  /*
   * SELECT BINARY
   * size > 1000
   */
  expressions::Expression *predicateBin;
  {
    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*sizeBin);
    expressions::Expression *arg =
        new expressions::InputArgument(&recBin, 0, argProjectionsBin);
    expressions::Expression *selSize = new expressions::RecordProjection(
        sizeBin->getOriginalType(), arg, *sizeBin);

    expressions::Expression *predExpr1 = new expressions::IntConstant(sizeLow);
    predicateBin = new expressions::GtExpression(selSize, predExpr1);
  }

  Select *selBin = new Select(predicateBin, scanBin);
  scanBin->setParent(selBin);

  /**
   * SCAN JSON FILE
   */
  string fnameJSON = symantecJSON.path;
  RecordType recJSON = symantecJSON.recType;
  int linehintJSON = symantecJSON.linehint;

  RecordAttribute *idJSON = argsSymantecJSON["id"];
  RecordAttribute *day = argsSymantecJSON["day"];
  RecordAttribute *size = argsSymantecJSON["size"];

  vector<RecordAttribute *> projectionsJSON;
  projectionsJSON.push_back(idJSON);
  projectionsJSON.push_back(size);

  ListType *documentType = new ListType(recJSON);
  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(
      &ctx, fnameJSON, documentType, linehintJSON);
  rawCatalog.registerPlugin(fnameJSON, pgJSON);
  Scan *scanJSON = new Scan(&ctx, *pgJSON);

  /*
   * SELECT JSON
   * (data->'day'->>'year')::int = 2010
   */
  expressions::Expression *predicateJSON;
  {
    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    argProjectionsJSON.push_back(*day);
    expressions::Expression *arg =
        new expressions::InputArgument(&recJSON, 0, argProjectionsJSON);
    expressions::Expression *selDay =
        new expressions::RecordProjection(day->getOriginalType(), arg, *day);
    IntType yearType = IntType();
    RecordAttribute *year =
        new RecordAttribute(2, fnameJSON, "year", &yearType);
    expressions::Expression *selYear =
        new expressions::RecordProjection(&yearType, selDay, *year);

    expressions::Expression *predExpr1 = new expressions::IntConstant(yearNo);
    predicateJSON = new expressions::EqExpression(selYear, predExpr1);
  }

  Select *selJSON = new Select(predicateJSON, scanJSON);
  scanJSON->setParent(selJSON);

  /*
   * JOIN
   * st.id = sj.id
   */

  // LEFT SIDE
  list<RecordAttribute> argProjectionsLeft;
  argProjectionsLeft.push_back(*idBin);
  expressions::Expression *leftArg =
      new expressions::InputArgument(&recBin, 0, argProjectionsLeft);
  expressions::Expression *leftPred = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);

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

  /* left materializer - no field needed */
  vector<RecordAttribute *> fieldsLeft;
  vector<materialization_mode> outputModesLeft;

  /* explicit mention to left OID */
  RecordAttribute *projTupleL =
      new RecordAttribute(fnamePrefixBin, activeLoop, pgBin->getOIDType());
  vector<RecordAttribute *> OIDLeft;
  OIDLeft.push_back(projTupleL);
  expressions::Expression *exprLeftOID = new expressions::RecordProjection(
      pgBin->getOIDType(), leftArg, *projTupleL);
  expressions::Expression *exprLeftKey = new expressions::RecordProjection(
      idBin->getOriginalType(), leftArg, *idBin);
  vector<expression_t> expressionsLeft;
  expressionsLeft.push_back(exprLeftOID);
  expressionsLeft.push_back(exprLeftKey);

  Materializer *matLeft =
      new Materializer(fieldsLeft, expressionsLeft, OIDLeft, outputModesLeft);

  /* right materializer - XXX mat. size (json) */
  vector<RecordAttribute *> fieldsRight;
  fieldsRight.push_back(size);
  vector<materialization_mode> outputModesRight;
  outputModesRight.insert(outputModesRight.begin(), EAGER);

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

  char joinLabel[] = "radixJoinBinJSON";
  RadixJoin *join = new RadixJoin(*joinPred, selBin, selJSON, &ctx, joinLabel,
                                  *matLeft, *matRight);
  selBin->setParent(join);
  selJSON->setParent(join);

  /**
   * REDUCE
   * MAX(size)
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*size);
  expressions::Expression *exprSize = new expressions::RecordProjection(
      size->getOriginalType(), rightArg, *size);

  ReduceNoPred *reduce = new ReduceNoPred(MAX, exprSize, join, &ctx);
  join->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce();
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pgBin->finish();
  pgJSON->finish();
  rawCatalog.clear();
}
