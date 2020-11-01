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
#include "plugins/binary-col-plugin.hpp"
#include "plugins/binary-row-plugin.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "util/caching.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

/* SELECT MIN(p_event),MAX(p_event), COUNT(*) from symantecunordered where id >
 * 50000000 and id < 60000000; */
void symantecJSON1(map<string, dataset> datasetCatalog);
void symantecJSON2(map<string, dataset> datasetCatalog);
void symantecJSON3(map<string, dataset> datasetCatalog);
void symantecJSON4(map<string, dataset> datasetCatalog);
void symantecJSON5(map<string, dataset> datasetCatalog);
void symantecJSON6(map<string, dataset> datasetCatalog);
void symantecJSON7(map<string, dataset> datasetCatalog);
void symantecJSON8(map<string, dataset> datasetCatalog);
void symantecJSON9(map<string, dataset> datasetCatalog);
void symantecJSON10(map<string, dataset> datasetCatalog);
void symantecJSON11(map<string, dataset> datasetCatalog);
void symantecJSONWarmup(map<string, dataset> datasetCatalog);
void nesting();

int main() {
  cout << "Execution" << endl;
  map<string, dataset> datasetCatalog;
  symantecCoreIDDatesSchema(datasetCatalog);
  symantecJSONWarmup(datasetCatalog);
  cout << "SYMANTEC JSON 1" << endl;
  symantecJSON1(datasetCatalog);
  cout << "SYMANTEC JSON 2" << endl;
  symantecJSON2(datasetCatalog);
  cout << "SYMANTEC JSON 3" << endl;
  symantecJSON3(datasetCatalog);
  cout << "SYMANTEC JSON 4" << endl;
  symantecJSON4(datasetCatalog);
  cout << "SYMANTEC JSON 5" << endl;
  symantecJSON5(datasetCatalog);
  cout << "SYMANTEC JSON 6" << endl;
  symantecJSON6(datasetCatalog);
  cout << "SYMANTEC JSON 7" << endl;
  symantecJSON7(datasetCatalog);
  cout << "SYMANTEC JSON 9" << endl;
  symantecJSON9(datasetCatalog);
  cout << "SYMANTEC JSON 10" << endl;
  symantecJSON10(datasetCatalog);
  // XXX 11 Cannot run locally!
  cout << "SYMANTEC JSON 11" << endl;
  symantecJSON11(datasetCatalog);
  // XXX Long-running (and actually crashing with caches..?)
  //    cout << "SYMANTEC JSON 8" << endl;
  //    symantecJSON8(datasetCatalog);

  //    nesting();
}

void nesting() {
  Context &ctx = *prepareContext("Hierarchical query");
  Catalog &catalog = Catalog::getInstance();

  string fname = string("inputs/json/jsmnDeeper-flat.json");

  IntType intType = IntType();

  string c1Name = string("c1");
  RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
  string c2Name = string("c2");
  RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
  list<RecordAttribute *> attsNested = list<RecordAttribute *>();
  attsNested.push_back(&c1);
  attsNested.push_back(&c2);
  RecordType nested = RecordType(attsNested);

  string attrName = string("a");
  string attrName2 = string("b");
  string attrName3 = string("c");
  RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
  RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
  RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

  list<RecordAttribute *> atts = list<RecordAttribute *>();
  atts.push_back(&attr);
  atts.push_back(&attr2);
  atts.push_back(&attr3);

  RecordType inner = RecordType(atts);
  ListType documentType = ListType(inner);
  int linehint = 10;
  /**
   * SCAN
   */
  jsonPipelined::JSONPlugin pg =
      jsonPipelined::JSONPlugin(&ctx, fname, &documentType, linehint);
  catalog.registerPlugin(fname, &pg);
  Scan scan = Scan(&ctx, pg);

  /**
   * SELECT
   */
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg.getOIDType());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(attr);
  projections.push_back(attr2);
  projections.push_back(attr3);
  expressions::Expression *lhsArg =
      new expressions::InputArgument(&inner, 0, projections);
  expressions::Expression *lhs_ =
      new expressions::RecordProjection(&nested, lhsArg, attr3);
  expressions::Expression *lhs =
      new expressions::RecordProjection(&intType, lhs_, c2);
  expressions::Expression *rhs = new expressions::IntConstant(114);

  // obj.c.c2 = 114 --> Only 1 must qualify
  expressions::Expression *predicate = new expressions::EqExpression(lhs, rhs);

  Select sel = Select(predicate, &scan);
  scan.setParent(&sel);

  // PRINT
  llvm::Function *debugInt = ctx.getFunction("printi");
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(&intType, lhsArg, attr);
  Print printOp = Print(debugInt, proj, &sel);
  sel.setParent(&printOp);

  // ROOT
  Root rootOp = Root(&printOp);
  printOp.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  pg.finish();
  catalog.clear();
}

void symantecJSON1(map<string, dataset> datasetCatalog) {
  int sizeHigh = 1500;
  string charsetType = "iso-8859-1";
  string cteType = "7bit";

  Context &ctx = *prepareContext("symantec-json-1");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *size = argsSymantecJSON["size"];
  RecordAttribute *charset = argsSymantecJSON["charset"];
  RecordAttribute *cte = argsSymantecJSON["cte"];

  vector<RecordAttribute *> projections;
  projections.push_back(size);
  projections.push_back(charset);
  projections.push_back(cte);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  vector<Monoid> accs;
  vector<expression_t> outputExprs;

  accs.push_back(SUM);
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  outputExprs.push_back(outputExpr1);

  /* Pred: */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*size);
  argSelections.push_back(*charset);
  argSelections.push_back(*cte);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selSize =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  expressions::Expression *selCharset = new expressions::RecordProjection(
      charset->getOriginalType(), arg, *charset);
  expressions::Expression *selCte =
      new expressions::RecordProjection(cte->getOriginalType(), arg, *cte);
  expressions::Expression *predExpr1 = new expressions::IntConstant(sizeHigh);
  expressions::Expression *predExpr2 =
      new expressions::StringConstant(charsetType);
  expressions::Expression *predExpr3 = new expressions::StringConstant(cteType);
  expressions::Expression *predicate1 =
      new expressions::LtExpression(selSize, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::EqExpression(selCharset, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::EqExpression(selCte, predExpr3);
  expressions::Expression *predicateAnd =
      new expressions::AndExpression(predicate1, predicate2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicateAnd, predicate3);

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

void symantecJSON2(map<string, dataset> datasetCatalog) {
  int idHigh = 1000000;
  int sizeHigh = 1000;
  string langType = "german";

  Context &ctx = *prepareContext("symantec-json-2");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  RecordAttribute *lang = argsSymantecJSON["lang"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(size);
  projections.push_back(lang);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 1000000 and (data->>'size')::int < 1000 and
   * (data->>'lang') = 'german'
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*size);
  argSelections.push_back(*lang);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selSize =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  expressions::Expression *selLang =
      new expressions::RecordProjection(lang->getOriginalType(), arg, *lang);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr2 = new expressions::IntConstant(sizeHigh);
  expressions::Expression *predExpr3 =
      new expressions::StringConstant(langType);

  expressions::Expression *predicate1 =
      new expressions::LtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::LtExpression(selSize, predExpr2);
  expressions::Expression *predicateNum =
      new expressions::AndExpression(predicate1, predicate2);

  Select *selNum = new Select(predicateNum, scan);
  scan->setParent(selNum);

  expressions::Expression *predicateStr =
      new expressions::EqExpression(selLang, predExpr3);
  Select *sel = new Select(predicateStr, selNum);
  selNum->setParent(sel);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
  sel->setParent(reduce);

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

void symantecJSON3(map<string, dataset> datasetCatalog) {
  int idHigh = 1000000;
  string country_codeType = "GR";
  string cityType = "Athens";

  Context &ctx = *prepareContext("symantec-json-3");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];
  RecordAttribute *city = argsSymantecJSON["city"];
  RecordAttribute *country_code = argsSymantecJSON["country_code"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(city);
  projections.push_back(country_code);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 1000000 and (data->>'country_code') = 'GR' and
   * (data->>'city') = 'Athens'
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*city);
  argSelections.push_back(*country_code);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *selCountryCode = new expressions::RecordProjection(
      country_code->getOriginalType(), arg, *country_code);
  expressions::Expression *selCity =
      new expressions::RecordProjection(city->getOriginalType(), arg, *city);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr2 =
      new expressions::StringConstant(country_codeType);
  expressions::Expression *predExpr3 =
      new expressions::StringConstant(cityType);

  expressions::Expression *predicate1 =
      new expressions::LtExpression(selID, predExpr1);

  Select *selNum = new Select(predicate1, scan);
  scan->setParent(selNum);

  expressions::Expression *predicate2 =
      new expressions::EqExpression(selCountryCode, predExpr2);
  expressions::Expression *predicate3 =
      new expressions::EqExpression(selCity, predExpr3);

  expressions::Expression *predicateStr =
      new expressions::AndExpression(predicate2, predicate3);

  Select *sel = new Select(predicateStr, selNum);
  selNum->setParent(sel);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
  sel->setParent(reduce);

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

void symantecJSON4(map<string, dataset> datasetCatalog) {
  int idHigh = 3000000;
  int idLow = 1000000;

  Context &ctx = *prepareContext("symantec-json-4");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 3000000 AND (data->>'id')::int > 1000000
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);

  expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
  expressions::Expression *predExpr2 = new expressions::IntConstant(idLow);

  expressions::Expression *predicate1 =
      new expressions::LtExpression(selID, predExpr1);
  expressions::Expression *predicate2 =
      new expressions::GtExpression(selID, predExpr2);
  expressions::Expression *predicate =
      new expressions::AndExpression(predicate1, predicate2);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
  sel->setParent(reduce);

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

void symantecJSON5(map<string, dataset> datasetCatalog) {
  int idHigh = 1000000;

  Context &ctx = *prepareContext("symantec-json-5");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];
  RecordAttribute *uri = argsSymantecJSON["uri"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 1000000
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*uri);

  /* Filtering: (data->>'id')::int < 1000000 */
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
  expressions::Expression *predicate =
      new expressions::LtExpression(selID, predExpr1);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /* OUTER Unnest -> some entries have no URIs */
  list<RecordAttribute> unnestProjections = list<RecordAttribute>();
  unnestProjections.push_back(*uri);

  expressions::Expression *outerArg =
      new expressions::InputArgument(&rec, 0, unnestProjections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(uri->getOriginalType(), outerArg, *uri);
  string nestedName = activeLoop;
  Path path = Path(nestedName, proj);

  expressions::Expression *lhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *rhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *predUnnest =
      new expressions::EqExpression(lhsUnnest, rhsUnnest);

  OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
  //    Unnest *unnestOp = new Unnest(predicate, path, scan);
  sel->setParent(unnestOp);

  /*
   * NULL FILTER!
   * Acts as mini-nest operator
   */
  StringType nestedType = StringType();
  //            &rec);
  RecordAttribute recUnnested =
      RecordAttribute(2, fname + ".uri", nestedName, &nestedType);
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
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, nullFilter, &ctx);
  nullFilter->setParent(reduce);
  //    ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, unnestOp,
  //    &ctx); unnestOp->setParent(reduce);

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

void symantecJSON6(map<string, dataset> datasetCatalog) {
  string cityType = "Athens";

  Context &ctx = *prepareContext("symantec-json-6");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *city = argsSymantecJSON["city"];
  RecordAttribute *uri = argsSymantecJSON["uri"];

  vector<RecordAttribute *> projections;
  projections.push_back(city);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'city') = 'Athens'
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*city);
  argSelections.push_back(*uri);

  /* Filtering: (data->>'id')::int < 1000000 */
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selCity =
      new expressions::RecordProjection(city->getOriginalType(), arg, *city);
  expressions::Expression *predExpr1 =
      new expressions::StringConstant(cityType);
  expressions::Expression *predicate =
      new expressions::EqExpression(selCity, predExpr1);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /* OUTER Unnest -> some entries have no URIs */
  list<RecordAttribute> unnestProjections = list<RecordAttribute>();
  unnestProjections.push_back(*uri);

  expressions::Expression *outerArg =
      new expressions::InputArgument(&rec, 0, unnestProjections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(uri->getOriginalType(), outerArg, *uri);
  string nestedName = activeLoop;
  Path path = Path(nestedName, proj);

  expressions::Expression *lhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *rhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *predUnnest =
      new expressions::EqExpression(lhsUnnest, rhsUnnest);

  OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
  //    Unnest *unnestOp = new Unnest(predicate, path, scan);
  sel->setParent(unnestOp);

  /*
   * NULL FILTER!
   * Acts as mini-nest operator
   */
  StringType nestedType = StringType();
  //            &rec);
  RecordAttribute recUnnested =
      RecordAttribute(2, fname + ".uri", nestedName, &nestedType);
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
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, nullFilter, &ctx);
  nullFilter->setParent(reduce);

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

void symantecJSON7(map<string, dataset> datasetCatalog) {
  int idHigh = 1000000;

  Context &ctx = *prepareContext("symantec-json-7");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  RecordAttribute *uri = argsSymantecJSON["uri"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(size);
  projections.push_back(uri);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 1000000
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*id);
  argSelections.push_back(*size);
  argSelections.push_back(*uri);

  /* Filtering: (data->>'id')::int < 1000000 */
  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selID =
      new expressions::RecordProjection(id->getOriginalType(), arg, *id);
  expressions::Expression *predExpr1 = new expressions::IntConstant(idHigh);
  expressions::Expression *predicate =
      new expressions::LtExpression(selID, predExpr1);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /* OUTER Unnest -> some entries have no URIs */
  list<RecordAttribute> unnestProjections = list<RecordAttribute>();
  unnestProjections.push_back(*uri);

  expressions::Expression *outerArg =
      new expressions::InputArgument(&rec, 0, unnestProjections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(uri->getOriginalType(), outerArg, *uri);
  string nestedName = activeLoop;
  Path path = Path(nestedName, proj);

  expressions::Expression *lhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *rhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *predUnnest =
      new expressions::EqExpression(lhsUnnest, rhsUnnest);

  OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
  //    Unnest *unnestOp = new Unnest(predicate, path, scan);
  sel->setParent(unnestOp);

  /*
   * NULL FILTER!
   * Acts as mini-nest operator
   */
  StringType nestedType = StringType();
  //            &rec);
  RecordAttribute recUnnested =
      RecordAttribute(2, fname + ".uri", nestedName, &nestedType);
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
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  ReduceNoPred *reduce = new ReduceNoPred(MAX, outputExpr1, nullFilter, &ctx);
  nullFilter->setParent(reduce);

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

// Takes a very long time to evaluate string predicate
void symantecJSON8(map<string, dataset> datasetCatalog) {
  //    string botName = "GRUM";
  string langName = "german";

  Context &ctx = *prepareContext("symantec-json-8");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  //    RecordAttribute *bot = argsSymantecJSON["bot"];
  RecordAttribute *lang = argsSymantecJSON["lang"];
  RecordAttribute *uri = argsSymantecJSON["uri"];

  vector<RecordAttribute *> projections;
  //    projections.push_back(bot);
  projections.push_back(lang);
  projections.push_back(uri);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'bot') = 'GRUM' OR (data->>'lang') = 'german' )
   */
  list<RecordAttribute> argSelections;
  //    argSelections.push_back(*bot);
  argSelections.push_back(*lang);
  argSelections.push_back(*uri);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  //    expressions::Expression* selBot = new expressions::RecordProjection(
  //            bot->getOriginalType(), arg, *bot);
  expressions::Expression *selLang =
      new expressions::RecordProjection(lang->getOriginalType(), arg, *lang);
  //    expressions::Expression* predExpr1 = new
  //    expressions::StringConstant(botName);
  expressions::Expression *predExpr2 =
      new expressions::StringConstant(langName);
  //    expressions::Expression* predicate1 = new expressions::EqExpression(
  //            selBot, predExpr1);
  expressions::Expression *predicate =
      new expressions::EqExpression(selLang, predExpr2);
  //    expressions::Expression* predicate = new expressions::OrExpression(
  //                    predicate1, predicate2);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /* OUTER Unnest -> some entries have no URIs */
  list<RecordAttribute> unnestProjections = list<RecordAttribute>();
  unnestProjections.push_back(*uri);

  expressions::Expression *outerArg =
      new expressions::InputArgument(&rec, 0, unnestProjections);
  expressions::RecordProjection *proj =
      new expressions::RecordProjection(uri->getOriginalType(), outerArg, *uri);
  string nestedName = activeLoop;
  Path path = Path(nestedName, proj);

  expressions::Expression *lhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *rhsUnnest = new expressions::BoolConstant(true);
  expressions::Expression *predUnnest =
      new expressions::EqExpression(lhsUnnest, rhsUnnest);

  OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
  sel->setParent(unnestOp);

  /*
   * NULL FILTER!
   * Acts as mini-nest operator
   */
  StringType nestedType = StringType();
  RecordAttribute recUnnested =
      RecordAttribute(2, fname + ".uri", nestedName, &nestedType);
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
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, nullFilter, &ctx);
  nullFilter->setParent(reduce);

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

void symantecJSON9(map<string, dataset> datasetCatalog) {
  int yearNo = 2010;

  Context &ctx = *prepareContext("symantec-json-9");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *day = argsSymantecJSON["day"];

  vector<RecordAttribute *> projections;
  projections.push_back(day);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->'day'->>'year')::int = 2010
   */
  list<RecordAttribute> argSelections;

  // Don't think this is needed
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, pg->getOIDType());
  argSelections.push_back(projTuple);

  argSelections.push_back(*day);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selDay =
      new expressions::RecordProjection(day->getOriginalType(), arg, *day);
  IntType yearType = IntType();
  RecordAttribute *year = new RecordAttribute(1, fname, "year", &yearType);
  expressions::Expression *selYear =
      new expressions::RecordProjection(&yearType, selDay, *year);
  expressions::Expression *predExpr1 = new expressions::IntConstant(yearNo);

  expressions::Expression *predicate =
      new expressions::EqExpression(selYear, predExpr1);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
  sel->setParent(reduce);

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

void symantecJSON10(map<string, dataset> datasetCatalog) {
  int monthNo = 12;

  Context &ctx = *prepareContext("symantec-json-10");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *day = argsSymantecJSON["day"];

  vector<RecordAttribute *> projections;
  projections.push_back(day);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->'day'->>'month')::int = 12
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*day);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
  expressions::Expression *selDay =
      new expressions::RecordProjection(day->getOriginalType(), arg, *day);
  IntType monthType = IntType();
  RecordAttribute *month = new RecordAttribute(2, fname, "month", &monthType);
  expressions::Expression *selMonth =
      new expressions::RecordProjection(&monthType, selDay, *month);

  expressions::Expression *predExpr1 = new expressions::IntConstant(monthNo);

  expressions::Expression *predicate =
      new expressions::EqExpression(selMonth, predExpr1);

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * REDUCE
   * Count(*)
   */
  /* Output: */
  expressions::Expression *outputExpr1 = new expressions::IntConstant(1);
  ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
  sel->setParent(reduce);

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

void symantecJSON11(map<string, dataset> datasetCatalog) {
  int idLow = 1000000;
  int idHigh = 3000000;

  Context &ctx = *prepareContext("symantec-json-11");
  Catalog &rawCatalog = Catalog::getInstance();

  string nameSymantec = string("symantecIDDates");
  dataset symantecJSON = datasetCatalog[nameSymantec];
  map<string, RecordAttribute *> argsSymantecJSON =
      symantecJSON.recType.getArgsMap();

  /**
   * SCAN JSON FILE
   */
  string fname = symantecJSON.path;
  RecordType rec = symantecJSON.recType;
  int linehint = symantecJSON.linehint;

  RecordAttribute *id = argsSymantecJSON["id"];
  RecordAttribute *size = argsSymantecJSON["size"];
  RecordAttribute *day = argsSymantecJSON["day"];

  vector<RecordAttribute *> projections;
  projections.push_back(id);
  projections.push_back(day);

  ListType *documentType = new ListType(rec);
  jsonPipelined::JSONPlugin *pg =
      new jsonPipelined::JSONPlugin(&ctx, fname, documentType, linehint);
  rawCatalog.registerPlugin(fname, pg);
  Scan *scan = new Scan(&ctx, *pg);

  /*
   * SELECT: Splitting preds in numeric and non-numeric
   * (data->>'id')::int < 3000000 AND (data->>'id')::int > 1000000
   */
  list<RecordAttribute> argSelections;
  argSelections.push_back(*day);

  expressions::Expression *arg =
      new expressions::InputArgument(&rec, 0, argSelections);
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

  Select *sel = new Select(predicate, scan);
  scan->setParent(sel);

  /**
   * NEST
   * GroupBy: (data->'day'->>'year')::int
   * Pred: Redundant (true == true)
   *         -> I wonder whether it gets statically removed..
   * Output: max((data->>'size')::int)
   */
  expressions::Expression *selDay =
      new expressions::RecordProjection(day->getOriginalType(), arg, *day);
  IntType yearType = IntType();
  RecordAttribute *year = new RecordAttribute(2, fname, "year", &yearType);
  expressions::Expression *selYear =
      new expressions::RecordProjection(&yearType, selDay, *year);

  list<RecordAttribute> nestProjections;
  nestProjections.push_back(*size);
  nestProjections.push_back(*year);
  expressions::Expression *nestArg =
      new expressions::InputArgument(&rec, 0, nestProjections);

  // f (& g) -> GROUPBY cluster
  expressions::RecordProjection *f = new expressions::RecordProjection(
      year->getOriginalType(), nestArg, *year);
  // p
  expressions::Expression *lhsNest = new expressions::BoolConstant(true);
  expressions::Expression *rhsNest = new expressions::BoolConstant(true);
  expressions::Expression *predNest =
      new expressions::EqExpression(lhsNest, rhsNest);

  // mat.
  vector<RecordAttribute *> fields;
  vector<materialization_mode> outputModes;
  //    fields.push_back(year);
  //    outputModes.insert(outputModes.begin(), EAGER);
  //    fields.push_back(size);
  //    outputModes.insert(outputModes.begin(), EAGER);

  Materializer *mat = new Materializer(fields, outputModes);

  char nestLabel[] = "nest_cluster";
  string aggrLabel = string(nestLabel);

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  vector<string> aggrLabels;
  string aggrField1;
  string aggrField2;

  /* Aggregate 1: MAX(size) */
  expressions::Expression *aggrSize =
      new expressions::RecordProjection(size->getOriginalType(), arg, *size);
  expressions::Expression *outputExpr1 = aggrSize;
  aggrField1 = string("_maxSize");
  accs.push_back(MAX);
  outputExprs.push_back(outputExpr1);
  aggrLabels.push_back(aggrField1);

  radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
                                        predNest, f, f, sel, nestLabel, *mat);
  sel->setParent(nestOp);

  llvm::Function *debugInt = ctx.getFunction("printi");
  IntType intType = IntType();

  /* OUTPUT */
  Operator *lastPrintOp;
  RecordAttribute *toOutput1 =
      new RecordAttribute(1, aggrLabel, aggrField1, &intType);
  expressions::RecordProjection *nestOutput1 =
      new expressions::RecordProjection(&intType, nestArg, *toOutput1);
  Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
  nestOp->setParent(printOp1);
  lastPrintOp = printOp1;

  Root *rootOp = new Root(lastPrintOp);
  lastPrintOp->setParent(rootOp);

  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  rootOp->produce(context);
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
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
  reduce->produce(context);
  ctx.prepareFunction(ctx.getGlobalFunction());
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("Execution took %f seconds\n", diff(t0, t1));

  // Close all open files & clear
  pg->finish();
  rawCatalog.clear();
}
