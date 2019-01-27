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
#include "util/raw-context.hpp"
#include "util/raw-functions.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/join.hpp"
#include "operators/radix-join.hpp"
#include "operators/unnest.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "operators/reduce.hpp"
#include "operators/reduce-nopred.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/nest.hpp"
#include "operators/nest-opt.hpp"
#include "operators/radix-nest.hpp"
#include "operators/materializer-expr.hpp"
#include "operators/null-filter.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/binary-row-plugin.hpp"
#include "plugins/binary-col-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "expressions/expressions-hasher.hpp"
#include "util/raw-caching.hpp"
#include "common/symantec-config.hpp"
#include "operators/materializer-expr.hpp"

void symantecCSVWarmup(map<string,dataset> datasetCatalog);
void symantecJSONWarmup(map<string,dataset> datasetCatalog);

void symantecBinCSVJSON0(map<string,dataset> datasetCatalog);

void symantecBinCSVJSON1(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON2(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON3(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON4(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON5(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON6(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON7(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON8(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON9(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON10(map<string,dataset> datasetCatalog);


void symantecBinCSVJSON2Debug1(map<string, dataset> datasetCatalog) {

    //bin
    float p_eventHigh = 0.7;
    //csv
    int classaLow = 70;
    //json
    int sizeLow = 10000;


    RawContext& ctx = *prepareContext("symantec-bin-csv-json-2-dbg");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *p_event;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    p_event = argsSymantecBin["p_event"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(p_event);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.p_event < 0.7
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selEvent = new expressions::RecordProjection(
            p_event->getOriginalType(), arg, *p_event);
    expressions::Expression* predExpr = new expressions::FloatConstant(
            p_eventHigh);
    predicateBin = new expressions::LtExpression(selEvent, predExpr);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

//  /**
//   * SCAN CSV FILE
//   */
//  pm::CSVPlugin* pgCSV;
//  Scan *scanCSV;
//  RecordType recCSV = symantecCSV.recType;
//  string fnameCSV = symantecCSV.path;
//  RecordAttribute *idCSV;
//  RecordAttribute *classa;
//  int linehintCSV = symantecCSV.linehint;
//  int policy = 5;
//  char delimInner = ';';
//
//  idCSV = argsSymantecCSV["id"];
//  classa = argsSymantecCSV["classa"];
//
//  vector<RecordAttribute*> projections;
//  projections.push_back(idCSV);
//  projections.push_back(classa);
//
//  pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
//          linehintCSV, policy, false);
//  rawCatalog.registerPlugin(fnameCSV, pgCSV);
//  scanCSV = new Scan(&ctx, *pgCSV);
//
//  /*
//   * SELECT CSV
//   * sc.classa > 70
//   */
//  expressions::Expression* predicateCSV;
//  {
//      list<RecordAttribute> argProjectionsCSV;
//      argProjectionsCSV.push_back(*classa);
//      expressions::Expression* arg = new expressions::InputArgument(&recCSV,
//              0, argProjectionsCSV);
//      expressions::Expression* selClassa = new expressions::RecordProjection(
//              classa->getOriginalType(), arg, *classa);
//
//      expressions::Expression* predExpr = new expressions::IntConstant(
//              classaLow);
//      predicateCSV = new expressions::GtExpression(
//              selClassa, predExpr);
//  }
//  Select *selCSV = new Select(predicateCSV, scanCSV);
//  scanCSV->setParent(selCSV);
//
//  /*
//   * JOIN
//   * st.id = sc.id
//   */
//
//  //LEFT SIDE
//  list<RecordAttribute> argProjectionsLeft;
//  argProjectionsLeft.push_back(*idBin);
//  argProjectionsLeft.push_back(*dim);
//  expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
//          0, argProjectionsLeft);
//  expressions::Expression* leftPred = new expressions::RecordProjection(
//          idBin->getOriginalType(), leftArg, *idBin);
//
//  //RIGHT SIDE
//  list<RecordAttribute> argProjectionsRight;
//  argProjectionsRight.push_back(*idCSV);
//  expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
//          1, argProjectionsRight);
//  expressions::Expression* rightPred = new expressions::RecordProjection(
//          idCSV->getOriginalType(), rightArg, *idCSV);
//
//  /* join pred. */
//  expressions::BinaryExpression* joinPred = new expressions::EqExpression(
//          leftPred, rightPred);
//
//  /* left materializer - dim needed */
//  vector<RecordAttribute*> fieldsLeft;
//  fieldsLeft.push_back(idBin);
//  fieldsLeft.push_back(dim);
//  vector<materialization_mode> outputModesLeft;
//  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
//  outputModesLeft.insert(outputModesLeft.begin(), EAGER);
//
//  /* explicit mention to left OID */
//  RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
//          activeLoop, pgBin->getOIDType());
//  vector<RecordAttribute*> OIDLeft;
//  OIDLeft.push_back(projTupleL);
//  expressions::Expression* exprLeftOID = new expressions::RecordProjection(
//          pgBin->getOIDType(), leftArg, *projTupleL);
//  expressions::Expression* exprLeftKey = new expressions::RecordProjection(
//          idBin->getOriginalType(), leftArg, *idBin);
//  vector<expression_t> expressionsLeft;
//  expressionsLeft.push_back(exprLeftOID);
//  expressionsLeft.push_back(exprLeftKey);
//
//  Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
//          OIDLeft, outputModesLeft);
//
//  /* right materializer - no explicit field needed */
//  vector<RecordAttribute*> fieldsRight;
//  vector<materialization_mode> outputModesRight;
//
//  /* explicit mention to right OID */
//  RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
//          pgCSV->getOIDType());
//  vector<RecordAttribute*> OIDRight;
//  OIDRight.push_back(projTupleR);
//  expressions::Expression* exprRightOID = new expressions::RecordProjection(
//          pgCSV->getOIDType(), rightArg, *projTupleR);
//  vector<expression_t> expressionsRight;
//  expressionsRight.push_back(exprRightOID);
//
//  Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
//          OIDRight, outputModesRight);
//
//  char joinLabel[] = "radixJoinBinCSV";
//  RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
//          *matLeft, *matRight);
//  selBin->setParent(joinBinCSV);
//  selCSV->setParent(joinBinCSV);
//
//  /**
//   * SCAN JSON FILE
//   */
//  string fnameJSON = symantecJSON.path;
//  RecordType recJSON = symantecJSON.recType;
//  int linehintJSON = symantecJSON.linehint;
//
//  RecordAttribute *idJSON = argsSymantecJSON["id"];
//  RecordAttribute *size = argsSymantecJSON["size"];
//
//  vector<RecordAttribute*> projectionsJSON;
//  projectionsJSON.push_back(idJSON);
//  projectionsJSON.push_back(size);
//
//  ListType *documentType = new ListType(recJSON);
//  jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
//          fnameJSON, documentType, linehintJSON);
//  rawCatalog.registerPlugin(fnameJSON, pgJSON);
//  Scan *scanJSON = new Scan(&ctx, *pgJSON);
//
//  /*
//   * SELECT JSON
//   * (data->>'size')::int > 10000
//   */
//  expressions::Expression* predicateJSON;
//  {
//      list<RecordAttribute> argProjectionsJSON;
//      argProjectionsJSON.push_back(*idJSON);
//      expressions::Expression* arg = new expressions::InputArgument(&recJSON,
//              0, argProjectionsJSON);
//      expressions::Expression* selSize = new expressions::RecordProjection(
//              size->getOriginalType(), arg, *size);
//
//      expressions::Expression* predExpr1 = new expressions::IntConstant(
//              sizeLow);
//      predicateJSON = new expressions::GtExpression(
//              selSize, predExpr1);
//  }
//  Select *selJSON = new Select(predicateJSON, scanJSON);
//  scanJSON->setParent(selJSON);
//
//  /*
//   * JOIN II
//   */
//  RadixJoin *joinJSON;
//  //LEFT SIDE
//  list<RecordAttribute> argProjectionsLeft2;
//  argProjectionsLeft2.push_back(*idBin);
//  argProjectionsLeft2.push_back(*dim);
//  expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
//          argProjectionsLeft2);
//  expressions::Expression* leftPred2 = new expressions::RecordProjection(
//          idBin->getOriginalType(), leftArg2, *idBin);
//
//  //RIGHT SIDE
//  list<RecordAttribute> argProjectionsRight2;
//  argProjectionsRight2.push_back(*idJSON);
//  expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
//          1, argProjectionsRight2);
//  expressions::Expression* rightPred2 = new expressions::RecordProjection(
//          idJSON->getOriginalType(), rightArg2, *idJSON);
//
//  /* join pred. */
//  expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
//          leftPred2, rightPred2);
//
//  /* explicit mention to left OIDs */
//  RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
//          activeLoop, pgBin->getOIDType());
//  RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
//              activeLoop, pgCSV->getOIDType());
//  vector<RecordAttribute*> OIDLeft2;
//  OIDLeft2.push_back(projTupleL2a);
//  OIDLeft2.push_back(projTupleL2b);
//  expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
//          pgBin->getOIDType(), leftArg2, *projTupleL2a);
//  expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
//              pgBin->getOIDType(), leftArg2, *projTupleL2b);
//  expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
//          idBin->getOriginalType(), leftArg2, *idBin);
//  vector<expression_t> expressionsLeft2;
//  expressionsLeft2.push_back(exprLeftOID2a);
//  expressionsLeft2.push_back(exprLeftOID2b);
//  expressionsLeft2.push_back(exprLeftKey2);
//
//  /* left materializer - dim needed */
//  vector<RecordAttribute*> fieldsLeft2;
//  fieldsLeft2.push_back(dim);
//  vector<materialization_mode> outputModesLeft2;
//  outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);
//
//  Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
//          OIDLeft2, outputModesLeft2);
//
//  /* right materializer - no explicit field needed */
//  vector<RecordAttribute*> fieldsRight2;
//  vector<materialization_mode> outputModesRight2;
//
//  /* explicit mention to right OID */
//  RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
//          pgJSON->getOIDType());
//  vector<RecordAttribute*> OIDRight2;
//  OIDRight2.push_back(projTupleR2);
//  expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
//          pgJSON->getOIDType(), rightArg2, *projTupleR2);
//  vector<expression_t> expressionsRight2;
//  expressionsRight2.push_back(exprRightOID2);
//
//  Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
//          OIDRight2, outputModesRight2);
//
//  char joinLabel2[] = "radixJoinIntermediateJSON";
//  joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
//          *matLeft2, *matRight2);
//  joinBinCSV->setParent(joinJSON);
//  selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, selBin,
            &ctx);
    selBin->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
//  pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON2Debug2(map<string, dataset> datasetCatalog) {

    //bin
    float p_eventHigh = 0.7;
    //csv
    int classaLow = 70;
    //json
    int sizeLow = 10000;


    RawContext& ctx = *prepareContext("symantec-bin-csv-json-2-dbg2");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();


    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * sc.classa > 70
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr = new expressions::IntConstant(
                classaLow);
        predicateCSV = new expressions::GtExpression(
                selClassa, predExpr);
    }
    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);


    /**
     * REDUCE
     * COUNT(*)
     */

    list<RecordAttribute> argProjections;
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, selCSV,
            &ctx);
    selCSV->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON2Debug3(map<string, dataset> datasetCatalog) {

    //bin
    float p_eventHigh = 0.7;
    //csv
    int classaLow = 70;
    //json
    int sizeLow = 10000;


    RawContext& ctx = *prepareContext("symantec-bin-csv-json-2-dbg3");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];
    RecordAttribute *size = argsSymantecJSON["size"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);
    projectionsJSON.push_back(size);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->>'size')::int > 10000
     */
    expressions::Expression* predicateJSON;

    list<RecordAttribute> argProjectionsJSON;
    argProjectionsJSON.push_back(*idJSON);
    expressions::Expression* arg = new expressions::InputArgument(&recJSON, 0,
            argProjectionsJSON);
    expressions::Expression* selSize = new expressions::RecordProjection(
            size->getOriginalType(), arg, *size);
    expressions::Expression* selID = new expressions::RecordProjection(
                idJSON->getOriginalType(), arg, *idJSON);

    expressions::Expression* predExpr1 = new expressions::IntConstant(sizeLow);
    predicateJSON = new expressions::GtExpression(selSize, predExpr1);

//  expressions::Expression* predExprA = new expressions::IntConstant(55007);
//  expressions::Expression* predExprB = new expressions::IntConstant(1926807);
//  expressions::Expression* predExprC = new expressions::IntConstant(1930155);
//  expressions::Expression* predExprD = new expressions::IntConstant(1930229);
//  expressions::Expression* predExprE = new expressions::IntConstant(1930235);
//  expressions::Expression* predExprF = new expressions::IntConstant(1979928);
//  expressions::Expression* predExprG = new expressions::IntConstant(2346485);
//  expressions::Expression* predExprH = new expressions::IntConstant(2346648);
//  expressions::Expression* predExprI = new expressions::IntConstant(2346705);
//  expressions::Expression* predExprJ = new expressions::IntConstant(3977494);
//  expressions::Expression* predicateA = new expressions::EqExpression(selID, predExprA);
//  expressions::Expression* predicateB = new expressions::EqExpression(selID, predExprB);
//  expressions::Expression* predicateC = new expressions::EqExpression(selID, predExprC);
//  expressions::Expression* predicateD = new expressions::EqExpression(selID, predExprD);
//
//  expressions::Expression* predicateOr1 = new expressions::OrExpression(predicateA,predicateB);
//  expressions::Expression* predicateOr2 = new expressions::OrExpression(predicateC,predicateD);
//  predicateJSON = new expressions::OrExpression(predicateOr1,predicateOr2);



    Select *selJSON = new Select(predicateJSON, scanJSON);
    scanJSON->setParent(selJSON);
    /**
     * REDUCE
     * COUNT(*)
     */

    list<RecordAttribute> argProjections;
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

//  accs.push_back(MAX);
//  expressions::Expression* outputExpr2 = new expressions::RecordProjection(
//          idJSON->getOriginalType(), arg, *idJSON);
//  outputExprs.push_back(outputExpr2);
//  expressions::Expression* outputExpr2 = new expressions::RecordProjection(
//              size->getOriginalType(), arg, *size);
//      outputExprs.push_back(outputExpr2);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, selJSON,
            &ctx);
    selJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgJSON->finish();
    rawCatalog.clear();
}

int main()  {
    cout << "Execution" << endl;
    map<string,dataset> datasetCatalog;
    symantecBinSchema(datasetCatalog);
    symantecCSVSchema(datasetCatalog);
    symantecCoreIDDatesSchema(datasetCatalog);

//  symantecCSVWarmup(datasetCatalog);
//  symantecJSONWarmup(datasetCatalog);

//  cout << "SYMANTEC BIN-CSV-JSON 0" << endl;
//  symantecBinCSVJSON0(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 1" << endl;
//  symantecBinCSVJSON1(datasetCatalog);
    cout << "SYMANTEC BIN-CSV-JSON 2 DEBUG" << endl;
//  symantecBinCSVJSON2Debug1(datasetCatalog);
//  symantecBinCSVJSON2Debug2(datasetCatalog);
    symantecBinCSVJSON2Debug3(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 3" << endl;
//  symantecBinCSVJSON3(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 4" << endl;
//  symantecBinCSVJSON4(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 5" << endl;
//  symantecBinCSVJSON5(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 6" << endl;
//  symantecBinCSVJSON6(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 7" << endl;
//  symantecBinCSVJSON7(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 8" << endl;
//  symantecBinCSVJSON8(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 9" << endl;
//  cout << "SYMANTEC BIN-CSV-JSON 9" << endl;
//  symantecBinCSVJSON9(datasetCatalog);
//  cout << "SYMANTEC BIN-CSV-JSON 10" << endl;
//  symantecBinCSVJSON10(datasetCatalog);
}

void symantecBinCSVJSON0(map<string, dataset> datasetCatalog) {

    //bin
    int idLow = 8000000;
    int idHigh = 10000000;
    int clusterHigh = 200;
    //csv
    int classaHigh = 90;
    string botName = "Bobax";

    RawContext& ctx = *prepareContext("symantec-bin-csv-1");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *cluster;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    cluster = argsSymantecBin["cluster"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(cluster);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.cluster < 200 and st.id > 8000000 and st.id < 10000000
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selID = new expressions::RecordProjection(
            idBin->getOriginalType(), arg, *idBin);
    expressions::Expression* selCluster = new expressions::RecordProjection(
            cluster->getOriginalType(), arg, *cluster);

    expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression* predicate1 = new expressions::GtExpression(
            selID, predExpr1);
    expressions::Expression* predicate2 = new expressions::LtExpression(
            selID, predExpr2);
    expressions::Expression* predicateAnd1 = new expressions::AndExpression(
            predicate1, predicate2);

    expressions::Expression* predExpr3 = new expressions::IntConstant(
            clusterHigh);
    expressions::Expression* predicate3 = new expressions::LtExpression(
            selCluster, predExpr3);

    predicateBin = new expressions::AndExpression(predicateAnd1, predicate3);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * classa < 19 and sc.id > 70000000  and sc.id < 80000000;
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*idCSV);
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selID = new expressions::RecordProjection(
                idCSV->getOriginalType(), arg, *idCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::GtExpression(
                selID, predExpr1);
        expressions::Expression* predicate2 = new expressions::LtExpression(
                selID, predExpr2);
        expressions::Expression* predicateAnd1 = new expressions::AndExpression(
                predicate1, predicate2);

        expressions::Expression* predExpr3 = new expressions::IntConstant(
                classaHigh);
        expressions::Expression* predicate3 = new expressions::LtExpression(
                selClassa, predExpr3);

        predicateCSV = new expressions::AndExpression(predicateAnd1, predicate3);
    }

    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);


    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(join);
    scanCSV->setParent(join);

    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
//  expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
//              0, argProjections);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join,
            &ctx);
    join->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON1(map<string, dataset> datasetCatalog) {

    //bin
    int idLow = 8000000;
    int idHigh = 10000000;
    int clusterHigh = 200;
    //csv
    int classaHigh = 90;
    string botName = "Bobax";

    RawContext& ctx = *prepareContext("symantec-bin-csv-json-1");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *cluster;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    cluster = argsSymantecBin["cluster"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(cluster);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.cluster < 200 and st.id > 8000000 and st.id < 10000000
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selID = new expressions::RecordProjection(
            idBin->getOriginalType(), arg, *idBin);
    expressions::Expression* selCluster = new expressions::RecordProjection(
            cluster->getOriginalType(), arg, *cluster);

    expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression* predicate1 = new expressions::GtExpression(
            selID, predExpr1);
    expressions::Expression* predicate2 = new expressions::LtExpression(
            selID, predExpr2);
    expressions::Expression* predicateAnd1 = new expressions::AndExpression(
            predicate1, predicate2);

    expressions::Expression* predExpr3 = new expressions::IntConstant(
            clusterHigh);
    expressions::Expression* predicate3 = new expressions::LtExpression(
            selCluster, predExpr3);

    predicateBin = new expressions::AndExpression(predicateAnd1, predicate3);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * classa < 19 and sc.id > 70000000  and sc.id < 80000000;
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*idCSV);
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selID = new expressions::RecordProjection(
                idCSV->getOriginalType(), arg, *idCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::GtExpression(
                selID, predExpr1);
        expressions::Expression* predicate2 = new expressions::LtExpression(
                selID, predExpr2);
        expressions::Expression* predicateAnd1 = new expressions::AndExpression(
                predicate1, predicate2);

        expressions::Expression* predExpr3 = new expressions::IntConstant(
                classaHigh);
        expressions::Expression* predicate3 = new expressions::LtExpression(
                selClassa, predExpr3);

        predicateCSV = new expressions::AndExpression(predicateAnd1, predicate3);
    }

    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);


    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000
     */
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selID = new expressions::RecordProjection(
                idJSON->getOriginalType(), arg, *idJSON);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::LtExpression(
                selID, predExpr1);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predicate2 = new expressions::GtExpression(
                selID, predExpr2);
        predicateJSON = new expressions::AndExpression(predicate1, predicate2);
    }
    Select *selJSON = new Select(predicateJSON, scanJSON);
    scanJSON->setParent(selJSON);

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON2(map<string, dataset> datasetCatalog) {

    //bin
    float p_eventHigh = 0.7;
    //csv
    int classaLow = 70;
    //json
    int sizeLow = 10000;


    RawContext& ctx = *prepareContext("symantec-bin-csv-json-2");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *p_event;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    p_event = argsSymantecBin["p_event"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(p_event);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.p_event < 0.7
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selEvent = new expressions::RecordProjection(
            p_event->getOriginalType(), arg, *p_event);
    expressions::Expression* predExpr = new expressions::FloatConstant(
            p_eventHigh);
    predicateBin = new expressions::LtExpression(selEvent, predExpr);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * sc.classa > 70
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr = new expressions::IntConstant(
                classaLow);
        predicateCSV = new expressions::GtExpression(selClassa, predExpr);
    }
    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);

    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];
    RecordAttribute *size = argsSymantecJSON["size"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);
    projectionsJSON.push_back(size);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->>'size')::int > 10000
     */
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selSize = new expressions::RecordProjection(
                size->getOriginalType(), arg, *size);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                sizeLow);
        predicateJSON = new expressions::GtExpression(
                selSize, predExpr1);
    }
    Select *selJSON = new Select(predicateJSON, scanJSON);
    scanJSON->setParent(selJSON);

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON3(map<string, dataset> datasetCatalog) {

    //bin
    float p_eventLow = 0.8;
    //csv
    int classaLow = 80;
    //json
    int sizeLow = 10000;


    RawContext& ctx = *prepareContext("symantec-bin-csv-json-3");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *p_event;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    p_event = argsSymantecBin["p_event"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(p_event);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.p_event < 0.7
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selEvent = new expressions::RecordProjection(
            p_event->getOriginalType(), arg, *p_event);
    expressions::Expression* predExpr = new expressions::FloatConstant(
            p_eventLow);
    predicateBin = new expressions::GtExpression(selEvent, predExpr);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * sc.classa > 70
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr = new expressions::IntConstant(
                classaLow);
        predicateCSV = new expressions::GtExpression(selClassa, predExpr);
    }
    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);

    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];
    RecordAttribute *size = argsSymantecJSON["size"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);
    projectionsJSON.push_back(size);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->>'size')::int > 10000
     */
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selSize = new expressions::RecordProjection(
                size->getOriginalType(), arg, *size);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                sizeLow);
        predicateJSON = new expressions::GtExpression(
                selSize, predExpr1);
    }
    Select *selJSON = new Select(predicateJSON, scanJSON);
    scanJSON->setParent(selJSON);

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON4(map<string, dataset> datasetCatalog) {

    //bin
    int idLow = 7000000;
    int idHigh = 9000000;
    int clusterHigh = 20;

    RawContext& ctx = *prepareContext("symantec-bin-csv-json-1");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *cluster;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    cluster = argsSymantecBin["cluster"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(cluster);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.cluster < 20 and st.id > 7000000 and st.id < 9000000
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selID = new expressions::RecordProjection(
            idBin->getOriginalType(), arg, *idBin);
    expressions::Expression* selCluster = new expressions::RecordProjection(
            cluster->getOriginalType(), arg, *cluster);

    expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression* predicate1 = new expressions::GtExpression(
            selID, predExpr1);
    expressions::Expression* predicate2 = new expressions::LtExpression(
            selID, predExpr2);
    expressions::Expression* predicateAnd1 = new expressions::AndExpression(
            predicate1, predicate2);

    expressions::Expression* predExpr3 = new expressions::IntConstant(
            clusterHigh);
    expressions::Expression* predicate3 = new expressions::LtExpression(
            selCluster, predExpr3);

    predicateBin = new expressions::AndExpression(predicateAnd1, predicate3);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * sc.id > 7000000 and sc.id < 9000000
     */
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*idCSV);
        argProjectionsCSV.push_back(*classa);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selID = new expressions::RecordProjection(
                idCSV->getOriginalType(), arg, *idCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                classa->getOriginalType(), arg, *classa);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::GtExpression(
                selID, predExpr1);
        expressions::Expression* predicate2 = new expressions::LtExpression(
                selID, predExpr2);
        expressions::Expression* predicateAnd1 = new expressions::AndExpression(
                predicate1, predicate2);

        predicateCSV = predicateAnd1;
    }

    Select *selCSV = new Select(predicateCSV, scanCSV);
    scanCSV->setParent(selCSV);


    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->>'id')::int > 7000000 and (data->>'id')::int < 9000000
     */
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selID = new expressions::RecordProjection(
                idJSON->getOriginalType(), arg, *idJSON);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::LtExpression(
                selID, predExpr1);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predicate2 = new expressions::GtExpression(
                selID, predExpr2);
        predicateJSON = new expressions::AndExpression(predicate1, predicate2);
    }
    Select *selJSON = new Select(predicateJSON, scanJSON);
    scanJSON->setParent(selJSON);

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON5(map<string, dataset> datasetCatalog) {

    //bin
    int idLow = 7000000;
    int idHigh = 10000000;
    int clusterHigh = 30;
    //csv
    string botName1 = "Bobax";
    string botName2 = "UNCLASSIFIED";
    string botName3 = "Unclassified";
    //json
    int yearLow = 2010;

    RawContext& ctx = *prepareContext("symantec-bin-csv-json-5");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *cluster;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    cluster = argsSymantecBin["cluster"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(cluster);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.cluster < 30 and st.id > 7000000 and st.id < 10000000
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selID = new expressions::RecordProjection(
            idBin->getOriginalType(), arg, *idBin);
    expressions::Expression* selCluster = new expressions::RecordProjection(
            cluster->getOriginalType(), arg, *cluster);

    expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression* predicate1 = new expressions::GtExpression(
            selID, predExpr1);
    expressions::Expression* predicate2 = new expressions::LtExpression(
            selID, predExpr2);
    expressions::Expression* predicateAnd1 = new expressions::AndExpression(
            predicate1, predicate2);

    expressions::Expression* predExpr3 = new expressions::IntConstant(
            clusterHigh);
    expressions::Expression* predicate3 = new expressions::LtExpression(
            selCluster, predExpr3);

    predicateBin = new expressions::AndExpression(predicateAnd1,
            predicate3);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *bot;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    bot = argsSymantecCSV["bot"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(bot);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * (bot = 'Bobax' OR bot = 'UNCLASSIFIED' or bot = 'Unclassified') and sc.id > 7000000  and sc.id < 10000000;
     */
    Select *selCSV;
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*idCSV);
        argProjectionsCSV.push_back(*bot);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selID = new expressions::RecordProjection(
                idCSV->getOriginalType(), arg, *idCSV);
        expressions::Expression* selBot = new expressions::RecordProjection(
                bot->getOriginalType(), arg, *bot);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::GtExpression(
                selID, predExpr1);
        expressions::Expression* predicate2 = new expressions::LtExpression(
                selID, predExpr2);
        expressions::Expression* predicateNum = new expressions::AndExpression(
                predicate1, predicate2);

        Select *selNum = new Select(predicateNum,scanCSV);
        scanCSV->setParent(selNum);

        expressions::Expression* predExprStr1 = new expressions::StringConstant(
                botName1);
        expressions::Expression* predExprStr2 = new expressions::StringConstant(
                        botName2);
        expressions::Expression* predExprStr3 = new expressions::StringConstant(
                        botName3);
        expressions::Expression* predicateStr1 = new expressions::EqExpression(
                selBot, predExprStr1);
        expressions::Expression* predicateStr2 = new expressions::EqExpression(
                        selBot, predExprStr2);
        expressions::Expression* predicateStr3 = new expressions::EqExpression(
                        selBot, predExprStr3);
        expressions::Expression* predicateOr1 = new expressions::OrExpression(
                predicateStr1, predicateStr2);

        predicateCSV = new expressions::OrExpression(
                predicateOr1, predicateStr3);
        selCSV = new Select(predicateCSV, selNum);
        selNum->setParent(selCSV);
    }

    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];
    RecordAttribute *day = argsSymantecJSON["day"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);
    projectionsJSON.push_back(day);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->'day'->>'year')::int > 2010 and (data->>'id')::int > 7000000 and (data->>'id')::int < 10000000
     */
    Select *selJSON;
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selID = new expressions::RecordProjection(
                idJSON->getOriginalType(), arg, *idJSON);

        expressions::Expression* selDay = new expressions::RecordProjection(
                day->getOriginalType(), arg, *day);
        IntType yearType = IntType();
        RecordAttribute *year = new RecordAttribute(1, fnameJSON, "year",
                &yearType);
        expressions::Expression* selYear = new expressions::RecordProjection(
                &yearType, selDay, *year);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::LtExpression(
                selID, predExpr1);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predicate2 = new expressions::GtExpression(
                selID, predExpr2);
        expressions::Expression* predicateNum = new expressions::AndExpression(
                predicate1, predicate2);

        Select *selNum = new Select(predicateNum, scanJSON);
        scanJSON->setParent(selNum);

        expressions::Expression* predExprYear = new expressions::IntConstant(
                yearLow);
        expressions::Expression* predicateYear = new expressions::GtExpression(
                selYear, predExprYear);

        selJSON = new Select(predicateYear, selNum);
        selNum->setParent(selJSON);
    }

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}

void symantecBinCSVJSON6(map<string, dataset> datasetCatalog) {

    //bin
    int idLow = 5000000;
    int idHigh = 10000000;
    int clusterLow = 200;
    //csv
    int classaHigh = 75;
    string botName1 = "Bobax";
    string botName2 = "UNCLASSIFIED";
    string botName3 = "Unclassified";
    //json
    int yearLow = 2010;

    RawContext& ctx = *prepareContext("symantec-bin-csv-json-6");
    RawCatalog& rawCatalog = RawCatalog::getInstance();

    string nameSymantecBin = string("symantecBin");
    dataset symantecBin = datasetCatalog[nameSymantecBin];
    map<string, RecordAttribute*> argsSymantecBin =
            symantecBin.recType.getArgsMap();

    string nameSymantecCSV = string("symantecCSV");
    dataset symantecCSV = datasetCatalog[nameSymantecCSV];
    map<string, RecordAttribute*> argsSymantecCSV =
            symantecCSV.recType.getArgsMap();

    string nameSymantecJSON = string("symantecIDDates");
    dataset symantecJSON = datasetCatalog[nameSymantecJSON];
    map<string, RecordAttribute*> argsSymantecJSON =
            symantecJSON.recType.getArgsMap();

    /**
     * SCAN BINARY FILE
     */
    BinaryColPlugin *pgBin;
    Scan *scanBin;
    RecordAttribute *idBin;
    RecordAttribute *cluster;
    RecordAttribute *dim;
    RecordType recBin = symantecBin.recType;
    string fnamePrefixBin = symantecBin.path;
    int linehintBin = symantecBin.linehint;
    idBin = argsSymantecBin["id"];
    cluster = argsSymantecBin["cluster"];
    dim = argsSymantecBin["dim"];
    vector<RecordAttribute*> projectionsBin;
    projectionsBin.push_back(idBin);
    projectionsBin.push_back(cluster);
    projectionsBin.push_back(dim);

    pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
    rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
    scanBin = new Scan(&ctx, *pgBin);

    /*
     * SELECT BINARY
     * st.cluster < 30 and st.id > 7000000 and st.id < 10000000
     */
    expressions::Expression* predicateBin;

    list<RecordAttribute> argProjectionsBin;
    argProjectionsBin.push_back(*idBin);
    argProjectionsBin.push_back(*dim);
    expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
            argProjectionsBin);
    expressions::Expression* selID = new expressions::RecordProjection(
            idBin->getOriginalType(), arg, *idBin);
    expressions::Expression* selCluster = new expressions::RecordProjection(
            cluster->getOriginalType(), arg, *cluster);

    expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
    expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
    expressions::Expression* predicate1 = new expressions::GtExpression(
            selID, predExpr1);
    expressions::Expression* predicate2 = new expressions::LtExpression(
            selID, predExpr2);
    expressions::Expression* predicateAnd1 = new expressions::AndExpression(
            predicate1, predicate2);

    expressions::Expression* predExpr3 = new expressions::IntConstant(
            clusterLow);
    expressions::Expression* predicate3 = new expressions::GtExpression(
            selCluster, predExpr3);

    predicateBin = new expressions::AndExpression(predicateAnd1,
            predicate3);

    Select *selBin = new Select(predicateBin, scanBin);
    scanBin->setParent(selBin);

    /**
     * SCAN CSV FILE
     */
    pm::CSVPlugin* pgCSV;
    Scan *scanCSV;
    RecordType recCSV = symantecCSV.recType;
    string fnameCSV = symantecCSV.path;
    RecordAttribute *idCSV;
    RecordAttribute *classa;
    RecordAttribute *bot;
    int linehintCSV = symantecCSV.linehint;
    int policy = 5;
    char delimInner = ';';

    idCSV = argsSymantecCSV["id"];
    classa = argsSymantecCSV["classa"];
    bot = argsSymantecCSV["bot"];

    vector<RecordAttribute*> projections;
    projections.push_back(idCSV);
    projections.push_back(classa);
    projections.push_back(bot);

    pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
            linehintCSV, policy, false);
    rawCatalog.registerPlugin(fnameCSV, pgCSV);
    scanCSV = new Scan(&ctx, *pgCSV);

    /*
     * SELECT CSV
     * (classA < 75 OR (bot = 'Bobax' OR bot = 'UNCLASSIFIED' or bot = 'Unclassified')) and sc.id > 5000000 and sc.id < 10000000
     */
    Select *selCSV;
    expressions::Expression* predicateCSV;
    {
        list<RecordAttribute> argProjectionsCSV;
        argProjectionsCSV.push_back(*idCSV);
        argProjectionsCSV.push_back(*bot);
        expressions::Expression* arg = new expressions::InputArgument(&recCSV,
                0, argProjectionsCSV);
        expressions::Expression* selID = new expressions::RecordProjection(
                idCSV->getOriginalType(), arg, *idCSV);
        expressions::Expression* selClassa = new expressions::RecordProjection(
                        classa->getOriginalType(), arg, *classa);
        expressions::Expression* selBot = new expressions::RecordProjection(
                bot->getOriginalType(), arg, *bot);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::GtExpression(
                selID, predExpr1);
        expressions::Expression* predicate2 = new expressions::LtExpression(
                selID, predExpr2);
        expressions::Expression* predicateNum = new expressions::AndExpression(
                predicate1, predicate2);

        Select *selNum = new Select(predicateNum,scanCSV);
        scanCSV->setParent(selNum);


        expressions::Expression* predExpr3 = new expressions::IntConstant(
                        classaHigh);
        expressions::Expression* predicate3 = new expressions::LtExpression(
                        selClassa, predExpr3);

        expressions::Expression* predExprStr1 = new expressions::StringConstant(
                botName1);
        expressions::Expression* predExprStr2 = new expressions::StringConstant(
                        botName2);
        expressions::Expression* predExprStr3 = new expressions::StringConstant(
                        botName3);
        expressions::Expression* predicateStr1 = new expressions::EqExpression(
                selBot, predExprStr1);
        expressions::Expression* predicateStr2 = new expressions::EqExpression(
                        selBot, predExprStr2);
        expressions::Expression* predicateStr3 = new expressions::EqExpression(
                        selBot, predExprStr3);
        expressions::Expression* predicateOr1 = new expressions::OrExpression(
                predicateStr1, predicateStr2);
        expressions::Expression* predicateOr2 = new expressions::OrExpression(
                        predicateOr1, predicateStr3);

        predicateCSV = new expressions::OrExpression(
                predicateOr2, predicate3);
        selCSV = new Select(predicateCSV, selNum);
        selNum->setParent(selCSV);
    }

    /*
     * JOIN
     * st.id = sc.id
     */

    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft;
    argProjectionsLeft.push_back(*idBin);
    argProjectionsLeft.push_back(*dim);
    expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
            0, argProjectionsLeft);
    expressions::Expression* leftPred = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight;
    argProjectionsRight.push_back(*idCSV);
    expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
            1, argProjectionsRight);
    expressions::Expression* rightPred = new expressions::RecordProjection(
            idCSV->getOriginalType(), rightArg, *idCSV);

    /* join pred. */
    expressions::BinaryExpression* joinPred = new expressions::EqExpression(
            leftPred, rightPred);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft;
    fieldsLeft.push_back(idBin);
    fieldsLeft.push_back(dim);
    vector<materialization_mode> outputModesLeft;
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);
    outputModesLeft.insert(outputModesLeft.begin(), EAGER);

    /* explicit mention to left OID */
    RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    vector<RecordAttribute*> OIDLeft;
    OIDLeft.push_back(projTupleL);
    expressions::Expression* exprLeftOID = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg, *projTupleL);
    expressions::Expression* exprLeftKey = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg, *idBin);
    vector<expression_t> expressionsLeft;
    expressionsLeft.push_back(exprLeftOID);
    expressionsLeft.push_back(exprLeftKey);

    Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
            OIDLeft, outputModesLeft);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight;
    vector<materialization_mode> outputModesRight;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
            pgCSV->getOIDType());
    vector<RecordAttribute*> OIDRight;
    OIDRight.push_back(projTupleR);
    expressions::Expression* exprRightOID = new expressions::RecordProjection(
            pgCSV->getOIDType(), rightArg, *projTupleR);
    vector<expression_t> expressionsRight;
    expressionsRight.push_back(exprRightOID);

    Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
            OIDRight, outputModesRight);

    char joinLabel[] = "radixJoinBinCSV";
    RadixJoin *joinBinCSV = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
            *matLeft, *matRight);
    selBin->setParent(joinBinCSV);
    selCSV->setParent(joinBinCSV);

    /**
     * SCAN JSON FILE
     */
    string fnameJSON = symantecJSON.path;
    RecordType recJSON = symantecJSON.recType;
    int linehintJSON = symantecJSON.linehint;

    RecordAttribute *idJSON = argsSymantecJSON["id"];
    RecordAttribute *day = argsSymantecJSON["day"];

    vector<RecordAttribute*> projectionsJSON;
    projectionsJSON.push_back(idJSON);
    projectionsJSON.push_back(day);

    ListType *documentType = new ListType(recJSON);
    jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
            fnameJSON, documentType, linehintJSON);
    rawCatalog.registerPlugin(fnameJSON, pgJSON);
    Scan *scanJSON = new Scan(&ctx, *pgJSON);

    /*
     * SELECT JSON
     * (data->'day'->>'year')::int > 2010 and (data->>'id')::int > 5000000 and (data->>'id')::int < 10000000
     */
    Select *selJSON;
    expressions::Expression* predicateJSON;
    {
        list<RecordAttribute> argProjectionsJSON;
        argProjectionsJSON.push_back(*idJSON);
        expressions::Expression* arg = new expressions::InputArgument(&recJSON,
                0, argProjectionsJSON);
        expressions::Expression* selID = new expressions::RecordProjection(
                idJSON->getOriginalType(), arg, *idJSON);

        expressions::Expression* selDay = new expressions::RecordProjection(
                day->getOriginalType(), arg, *day);
        IntType yearType = IntType();
        RecordAttribute *year = new RecordAttribute(1, fnameJSON, "year",
                &yearType);
        expressions::Expression* selYear = new expressions::RecordProjection(
                &yearType, selDay, *year);

        expressions::Expression* predExpr1 = new expressions::IntConstant(
                idHigh);
        expressions::Expression* predicate1 = new expressions::LtExpression(
                selID, predExpr1);
        expressions::Expression* predExpr2 = new expressions::IntConstant(
                idLow);
        expressions::Expression* predicate2 = new expressions::GtExpression(
                selID, predExpr2);
        expressions::Expression* predicateNum = new expressions::AndExpression(
                predicate1, predicate2);

        Select *selNum = new Select(predicateNum, scanJSON);
        scanJSON->setParent(selNum);

        expressions::Expression* predExprYear = new expressions::IntConstant(
                yearLow);
        expressions::Expression* predicateYear = new expressions::GtExpression(
                selYear, predExprYear);

        selJSON = new Select(predicateYear, selNum);
        selNum->setParent(selJSON);
    }

    /*
     * JOIN II
     */
    RadixJoin *joinJSON;
    //LEFT SIDE
    list<RecordAttribute> argProjectionsLeft2;
    argProjectionsLeft2.push_back(*idBin);
    argProjectionsLeft2.push_back(*dim);
    expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
            argProjectionsLeft2);
    expressions::Expression* leftPred2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);

    //RIGHT SIDE
    list<RecordAttribute> argProjectionsRight2;
    argProjectionsRight2.push_back(*idJSON);
    expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
            1, argProjectionsRight2);
    expressions::Expression* rightPred2 = new expressions::RecordProjection(
            idJSON->getOriginalType(), rightArg2, *idJSON);

    /* join pred. */
    expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
            leftPred2, rightPred2);

    /* explicit mention to left OIDs */
    RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
            activeLoop, pgBin->getOIDType());
    RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
                activeLoop, pgCSV->getOIDType());
    vector<RecordAttribute*> OIDLeft2;
    OIDLeft2.push_back(projTupleL2a);
    OIDLeft2.push_back(projTupleL2b);
    expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
            pgBin->getOIDType(), leftArg2, *projTupleL2a);
    expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
                pgBin->getOIDType(), leftArg2, *projTupleL2b);
    expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
            idBin->getOriginalType(), leftArg2, *idBin);
    vector<expression_t> expressionsLeft2;
    expressionsLeft2.push_back(exprLeftOID2a);
    expressionsLeft2.push_back(exprLeftOID2b);
    expressionsLeft2.push_back(exprLeftKey2);

    /* left materializer - dim needed */
    vector<RecordAttribute*> fieldsLeft2;
    fieldsLeft2.push_back(dim);
    vector<materialization_mode> outputModesLeft2;
    outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

    Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
            OIDLeft2, outputModesLeft2);

    /* right materializer - no explicit field needed */
    vector<RecordAttribute*> fieldsRight2;
    vector<materialization_mode> outputModesRight2;

    /* explicit mention to right OID */
    RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
            pgJSON->getOIDType());
    vector<RecordAttribute*> OIDRight2;
    OIDRight2.push_back(projTupleR2);
    expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
            pgJSON->getOIDType(), rightArg2, *projTupleR2);
    vector<expression_t> expressionsRight2;
    expressionsRight2.push_back(exprRightOID2);

    Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
            OIDRight2, outputModesRight2);

    char joinLabel2[] = "radixJoinIntermediateJSON";
    joinJSON = new RadixJoin(*joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
            *matLeft2, *matRight2);
    joinBinCSV->setParent(joinJSON);
    selJSON->setParent(joinJSON);


    /**
     * REDUCE
     * MAX(dim), COUNT(*)
     */

    list<RecordAttribute> argProjections;
    argProjections.push_back(*dim);
    expressions::Expression* exprDim = new expressions::RecordProjection(
            dim->getOriginalType(), leftArg2, *dim);
    /* Output: */
    vector<Monoid> accs;
    vector<expression_t> outputExprs;

    accs.push_back(MAX);
    expressions::Expression* outputExpr1 = exprDim;
    outputExprs.push_back(outputExpr1);

    accs.push_back(SUM);
    expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
    outputExprs.push_back(outputExpr2);
    /* Pred: Redundant */

    expressions::Expression* lhsRed = new expressions::BoolConstant(true);
    expressions::Expression* rhsRed = new expressions::BoolConstant(true);
    expressions::Expression* predRed = new expressions::EqExpression(
            lhsRed, rhsRed);

    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
            &ctx);
    joinJSON->setParent(reduce);

    //Run function
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    reduce->produce();
    ctx.prepareFunction(ctx.getGlobalFunction());
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("Execution took %f seconds\n", diff(t0, t1));

    //Close all open files & clear
    pgBin->finish();
    pgCSV->finish();
    rawCatalog.clear();
}
