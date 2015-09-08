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
#include "operators/null-filter.hpp"


/* SELECT MIN(p_event),MAX(p_event), COUNT(*) from symantecunordered where id > 50000000 and id < 60000000; */
void symantecJSON1(map<string,dataset> datasetCatalog);
void symantecJSON2(map<string,dataset> datasetCatalog);
void symantecJSON3(map<string,dataset> datasetCatalog);
void symantecJSON4(map<string,dataset> datasetCatalog);
void symantecJSON5(map<string,dataset> datasetCatalog);
void symantecJSON6(map<string,dataset> datasetCatalog);
void symantecJSON7(map<string,dataset> datasetCatalog);
//void symantecCSV5(map<string,dataset> datasetCatalog);
//void symantecCSV6(map<string,dataset> datasetCatalog);
//void symantecCSV7(map<string,dataset> datasetCatalog);


RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{
	cout << "Execution" << endl;
	map<string,dataset> datasetCatalog;
	symantecCoreIDDatesSchema(datasetCatalog);
	symantecJSON1(datasetCatalog);
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
}

void symantecJSON1(map<string, dataset> datasetCatalog) {

	int sizeHigh = 1500;
	string charsetType = "iso-8859-1";
	string cteType = "7bit";

	RawContext ctx = prepareContext("symantec-json-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
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

	vector<RecordAttribute*> projections;
	projections.push_back(size);
	projections.push_back(charset);
	projections.push_back(cte);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(SUM);
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr1);

	/* Pred: */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*size);
	argSelections.push_back(*charset);
	argSelections.push_back(*cte);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selSize = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	expressions::Expression* selCharset = new expressions::RecordProjection(
			charset->getOriginalType(), arg, *charset);
	expressions::Expression* selCte = new expressions::RecordProjection(
			cte->getOriginalType(), arg, *cte);
	expressions::Expression* predExpr1 = new expressions::IntConstant(sizeHigh);
	expressions::Expression* predExpr2 = new expressions::StringConstant(
			charsetType);
	expressions::Expression* predExpr3 = new expressions::StringConstant(
			cteType);
	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selSize, predExpr1);
	expressions::Expression* predicate2 = new expressions::EqExpression(
			new BoolType(), selCharset, predExpr2);
	expressions::Expression* predicate3 = new expressions::EqExpression(
			new BoolType(), selCte, predExpr3);
	expressions::Expression* predicateAnd = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicate = new expressions::AndExpression(
			new BoolType(), predicateAnd, predicate3);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan,
			&ctx);
	scan->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON2(map<string, dataset> datasetCatalog) {

	int idHigh = 1000000;
	int sizeHigh = 1000;
	string langType = "german";

	RawContext ctx = prepareContext("symantec-json-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
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

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(size);
	projections.push_back(lang);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->>'id')::int < 1000000 and (data->>'size')::int < 1000 and (data->>'lang') = 'german'
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*size);
	argSelections.push_back(*lang);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* selSize = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	expressions::Expression* selLang = new expressions::RecordProjection(
			lang->getOriginalType(), arg, *lang);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr2 = new expressions::IntConstant(sizeHigh);
	expressions::Expression* predExpr3 = new expressions::StringConstant(
			langType);

	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selSize, predExpr2);
	expressions::Expression* predicateNum = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	Select *selNum = new Select(predicateNum,scan);
	scan->setParent(selNum);

	expressions::Expression* predicateStr = new expressions::EqExpression(
				new BoolType(), selLang, predExpr3);
	Select *sel = new Select(predicateStr,selNum);
	selNum->setParent(sel);


	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
	sel->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON3(map<string, dataset> datasetCatalog) {

	int idHigh = 1000000;
	string country_codeType = "GR";
	string cityType = "Athens";

	RawContext ctx = prepareContext("symantec-json-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
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

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(city);
	projections.push_back(country_code);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->>'id')::int < 1000000 and (data->>'country_code') = 'GR' and (data->>'city') = 'Athens'
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*city);
	argSelections.push_back(*country_code);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* selCountryCode = new expressions::RecordProjection(
				country_code->getOriginalType(), arg, *country_code);
	expressions::Expression* selCity = new expressions::RecordProjection(
			city->getOriginalType(), arg, *city);


	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr2 = new expressions::StringConstant(country_codeType);
	expressions::Expression* predExpr3 = new expressions::StringConstant(cityType);

	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	Select *selNum = new Select(predicate1,scan);
		scan->setParent(selNum);

	expressions::Expression* predicate2 = new expressions::EqExpression(
			new BoolType(), selCountryCode, predExpr2);
	expressions::Expression* predicate3 = new expressions::EqExpression(
				new BoolType(), selCity, predExpr3);

	expressions::Expression* predicateStr = new expressions::AndExpression(
				new BoolType(), predicate2, predicate3);

	Select *sel = new Select(predicateStr,selNum);
	selNum->setParent(sel);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
	sel->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON4(map<string, dataset> datasetCatalog) {

	int idHigh = 3000000;
	int idLow = 1000000;

	RawContext ctx = prepareContext("symantec-json-4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN JSON FILE
	 */
	string fname = symantecJSON.path;
	RecordType rec = symantecJSON.recType;
	int linehint = symantecJSON.linehint;

	RecordAttribute *id = argsSymantecJSON["id"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->>'id')::int < 3000000 AND (data->>'id')::int > 1000000
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idLow);


	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
	expressions::Expression* predicate = new expressions::AndExpression(
						new BoolType(), predicate1, predicate2);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, sel, &ctx);
	sel->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON5(map<string, dataset> datasetCatalog) {

	int idHigh = 1000000;

	RawContext ctx = prepareContext("symantec-json-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN JSON FILE
	 */
	string fname = symantecJSON.path;
	RecordType rec = symantecJSON.recType;
	int linehint = symantecJSON.linehint;

	RecordAttribute *id = argsSymantecJSON["id"];
	RecordAttribute *uri = argsSymantecJSON["uri"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
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
	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&rec, 0,
			unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
//	Unnest *unnestOp = new Unnest(predicate, path, scan);
	sel->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
//			&rec);
	RecordAttribute recUnnested = RecordAttribute(2, fname+".uri", nestedName,
			&nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
				&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, nullFilter, &ctx);
	nullFilter->setParent(reduce);
//	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, unnestOp, &ctx);
//	unnestOp->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON6(map<string, dataset> datasetCatalog) {

	string cityType = "Athens";

	RawContext ctx = prepareContext("symantec-json-6");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN JSON FILE
	 */
	string fname = symantecJSON.path;
	RecordType rec = symantecJSON.recType;
	int linehint = symantecJSON.linehint;

	RecordAttribute *city = argsSymantecJSON["city"];
	RecordAttribute *uri = argsSymantecJSON["uri"];

	vector<RecordAttribute*> projections;
	projections.push_back(city);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
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
	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selCity = new expressions::RecordProjection(
			city->getOriginalType(), arg, *city);
	expressions::Expression* predExpr1 = new expressions::StringConstant(cityType);
	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), selCity, predExpr1);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&rec, 0,
			unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
//	Unnest *unnestOp = new Unnest(predicate, path, scan);
	sel->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
//			&rec);
	RecordAttribute recUnnested = RecordAttribute(2, fname+".uri", nestedName,
			&nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
				&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr1, nullFilter, &ctx);
	nullFilter->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON7(map<string, dataset> datasetCatalog) {

	int idHigh = 1000000;

	RawContext ctx = prepareContext("symantec-json-7");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecJSON =
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

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(size);
	projections.push_back(uri);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
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
	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&rec, 0,
			unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, sel);
//	Unnest *unnestOp = new Unnest(predicate, path, scan);
	sel->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
//			&rec);
	RecordAttribute recUnnested = RecordAttribute(2, fname+".uri", nestedName,
			&nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
				&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * REDUCE
	 * Count(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	ReduceNoPred *reduce = new ReduceNoPred(MAX, outputExpr1, nullFilter, &ctx);
	nullFilter->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}
