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

#include <platform/common/common.hpp>

#include "common/symantec-config.hpp"
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
#include "util/caching.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

/* COUNT() */
void symantecProjection1(map<string, dataset> datasetCatalog, int predicateVal);

int main() {
  cout << "Execution" << endl;
  map<string, dataset> datasetCatalog;
  symantecCoreSchema(datasetCatalog);

  /* Filtering on email size */
  int predicateVal = 5000;
  //    int predicateVal = 600;
  symantecProjection1(datasetCatalog, predicateVal);
  symantecProjection1(datasetCatalog, predicateVal);
}

// void symantecProjection1(map<string,dataset> datasetCatalog, int
// predicateVal)    {
//
//    Context& ctx = *prepareContext("symantec-json-projection1");
//    Catalog& rawCatalog = Catalog::getInstance();
//
//    string nameSymantec = string("symantec");
//    dataset symantec = datasetCatalog[nameSymantec];
//    string nameOrders = string("orders");
//    dataset orders = datasetCatalog[nameOrders];
//    map<string, RecordAttribute*> argsLineitem     =
//            symantec.recType.getArgsMap();
//    map<string, RecordAttribute*> argsOrder        =
//            orders.recType.getArgsMap();
//
//    /**
//     * SCAN
//     */
//    string fname = symantec.path;
//    RecordType rec = symantec.recType;
//    int linehint = symantec.linehint;
//    RecordAttribute *size = argsLineitem["size"];
//
//    ListType *documentType = new ListType(rec);
//    jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
//            documentType,linehint);
//
//    rawCatalog.registerPlugin(fname, pg);
//    Scan *scan = new Scan(&ctx, *pg);
//
//    /**
//     * REDUCE
//     * (SUM 1)
//     * + predicate
//     */
//    list<RecordAttribute> argProjections;
//    argProjections.push_back(*size);
//    expressions::Expression* arg             =
//                new expressions::InputArgument(&rec,0,argProjections);
//    /* Output: */
//    vector<Monoid> accs;
//    vector<expression_t> outputExprs;
//    accs.push_back(SUM);
//    expressions::Expression* outputExpr = new expressions::IntConstant(1);
//    outputExprs.push_back(outputExpr);
//    /* Pred: */
//    expressions::Expression* selOrderkey      =
//            new
//            expressions::RecordProjection(size->getOriginalType(),arg,*size);
//    expressions::Expression* vakey = new
//    expressions::IntConstant(predicateVal); expressions::Expression* predicate
//    = new expressions::LtExpression(
//                selOrderkey, vakey);
//
//    opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan,
//    &ctx); scan->setParent(reduce);
//
//    //Run function
//    struct timespec t0, t1;
//    clock_gettime(CLOCK_REALTIME, &t0);
//    reduce->produce(context);
//    ctx.prepareFunction(ctx.getGlobalFunction());
//    clock_gettime(CLOCK_REALTIME, &t1);
//    printf("Execution took %f seconds\n", diff(t0, t1));
//
//    //Close all open files & clear
//    pg->finish();
//    rawCatalog.clear();
//}

void symantecProjection1(map<string, dataset> datasetCatalog,
                         int predicateVal) {
  Context &ctx = *prepareContext("symantec-json-projection1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantec");
  dataset symantec = datasetCatalog[nameSymantec];
  string nameOrders = string("orders");
  dataset orders = datasetCatalog[nameOrders];
  map<string, RecordAttribute *> argsLineitem = symantec.recType.getArgsMap();
  map<string, RecordAttribute *> argsOrder = orders.recType.getArgsMap();

  /**
   * SCAN
   */
  string fname = symantec.path;
  RecordType rec = symantec.recType;
  int linehint = symantec.linehint;
  RecordAttribute *size = argsLineitem["size"];

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);

  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   * (SUM 1)
   * + predicate
   */
  list<RecordAttribute> argProjections;
  argProjections.push_back(*size);
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argProjections);
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  accs.push_back(SUM);
  expressions::Expression *outputExpr = new expressions::IntConstant(1);
  //            new
  //            expressions::RecordProjection(size->getOriginalType(),arg,*size);
  //    new expressions::RecordProjection(size->getOriginalType(),arg,*size);
  outputExprs.push_back(outputExpr);
  /* Pred: */
  expressions::Expression *selOrderkey =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  expressions::Expression *vakey = new expressions::IntConstant(predicateVal);
  expressions::Expression *vakey2 = new expressions::IntConstant(0);
  expressions::Expression *predicate1 =
      new expressions::LtExpression(selOrderkey, vakey);
  expressions::Expression *predicate2 =
      new expressions::GtExpression(selOrderkey, vakey2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate1, predicate2);

  opt::Reduce *reduce =
      new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
  scan->setParent(reduce);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  reduce->produce(context);
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}
