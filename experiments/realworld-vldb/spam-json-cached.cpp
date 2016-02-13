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

#include "experiments/realworld-vldb/spam-json-cached.hpp"

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

void symantecJSON1Caching(map<string, dataset> datasetCatalog) {

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

	RecordAttribute *id = argsSymantecJSON["id"];
	RecordAttribute *size = argsSymantecJSON["size"];
	RecordAttribute *charset = argsSymantecJSON["charset"];
	RecordAttribute *cte = argsSymantecJSON["cte"];
	RecordAttribute *day = argsSymantecJSON["day"];

	vector<RecordAttribute*> projections;
	projections.push_back(size);
	projections.push_back(charset);
	projections.push_back(cte);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * Materialize expression(s) here:
	 * id, size cached (...+year/month)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	argProjections.push_back(*size);
	argProjections.push_back(*day);
	expressions::Expression* argMat = new expressions::InputArgument(&rec, 0,
			argProjections);

	expressions::Expression* toMat1 = new expressions::RecordProjection(
			id->getOriginalType(), argMat, *id);
	char matLabel1[] = "idMaterializer";
	ExprMaterializer *mat1 = new ExprMaterializer(toMat1, linehint, scan, &ctx,
			matLabel1);
	scan->setParent(mat1);

	expressions::Expression* toMat2 = new expressions::RecordProjection(
			size->getOriginalType(), argMat, *size);
	char matLabel2[] = "sizeMaterializer";
	ExprMaterializer *mat2 = new ExprMaterializer(toMat2, linehint, mat1, &ctx,
			matLabel2);
	mat1->setParent(mat2);

	expressions::Expression* matDay = new expressions::RecordProjection(
			day->getOriginalType(), argMat, *day);
	IntType yearType = IntType();
	RecordAttribute *year = new RecordAttribute(1, fname, "year", &yearType);
	expressions::Expression* toMat3 = new expressions::RecordProjection(
			&yearType, matDay, *year);
	char matLabel3[] = "yearMaterializer";
	ExprMaterializer *mat3 = new ExprMaterializer(toMat3, linehint, mat2, &ctx,
			matLabel3);
	mat2->setParent(mat3);

	IntType monthType = IntType();
	RecordAttribute *month = new RecordAttribute(2, fname, "month", &monthType);
	expressions::Expression* toMat4 = new expressions::RecordProjection(
			&monthType, matDay, *month);
	char matLabel4[] = "monthMaterializer";
	ExprMaterializer *mat4 = new ExprMaterializer(toMat4, linehint, mat3, &ctx,
			matLabel4);
	mat3->setParent(mat4);

	ExprMaterializer *lastMat = mat4;

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

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, lastMat,
			&ctx);
	lastMat->setParent(reduce);

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

	int idHigh = 20000000;
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

void symantecJSON2v1(map<string, dataset> datasetCatalog) {

	int idHigh = 160000000;
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

	expressions::Expression* predExpr1 = new expressions::IntConstant(sizeHigh);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::StringConstant(
			langType);

	expressions::Expression* predicateNum1 = new expressions::LtExpression(
				new BoolType(), selSize, predExpr1);
	expressions::Expression* predicateNum2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);

//	expressions::Expression* predicateNum = new expressions::AndExpression(
//			new BoolType(), predicate1, predicate2);

	Select *selNum1 = new Select(predicateNum1,scan);
	scan->setParent(selNum1);

	Select *selNum2 = new Select(predicateNum2,selNum1);
	selNum1->setParent(selNum2);

	expressions::Expression* predicateStr = new expressions::EqExpression(
				new BoolType(), selLang, predExpr3);
	Select *sel = new Select(predicateStr,selNum2);
	selNum2->setParent(sel);


	/**
	 * REDUCE
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

	int idHigh = 20000000;
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

//select max((data->>'size')::int), count(*) from spamscoreiddates28m where (data->>'id')::int < 30000000 and (data->>'country_code') = 'US';
void symantecJSON3v1(map<string, dataset> datasetCatalog) {

	int idHigh = 600000000;
	string country_codeType = "US";
//	string cityType = "Athens";

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
	RecordAttribute *size = argsSymantecJSON["size"];
//	RecordAttribute *city = argsSymantecJSON["city"];
	RecordAttribute *country_code = argsSymantecJSON["country_code"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
//	projections.push_back(city);
	projections.push_back(country_code);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
//	argSelections.push_back(*city);
	argSelections.push_back(*country_code);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* selCountryCode = new expressions::RecordProjection(
				country_code->getOriginalType(), arg, *country_code);
//	expressions::Expression* selCity = new expressions::RecordProjection(
//			city->getOriginalType(), arg, *city);


	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr2 = new expressions::StringConstant(country_codeType);
//	expressions::Expression* predExpr3 = new expressions::StringConstant(cityType);

	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	Select *selNum = new Select(predicate1,scan);
		scan->setParent(selNum);

	expressions::Expression* predicate2 = new expressions::EqExpression(
			new BoolType(), selCountryCode, predExpr2);
//	expressions::Expression* predicate3 = new expressions::EqExpression(
//				new BoolType(), selCity, predExpr3);
//
//	expressions::Expression* predicateStr = new expressions::AndExpression(
//				new BoolType(), predicate2, predicate3);
	expressions::Expression* predicateStr = predicate2;
	Select *sel = new Select(predicateStr,selNum);
	selNum->setParent(sel);

	/**
	 * REDUCE
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

	int idHigh = 60000000;
	int idLow = 20000000;

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

void symantecJSON4v1(map<string, dataset> datasetCatalog) {

	int idHigh = 60000000;
	int idLow = 20000000;

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
	RecordAttribute *size = argsSymantecJSON["size"];

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
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);

	accs.push_back(MAX);
	expressions::Expression* exprID = new expressions::RecordProjection(
			id->getOriginalType(), argRed, *id);
	expressions::Expression* outputExpr3 = exprID;
	outputExprs.push_back(outputExpr3);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

	int idHigh = 20000000;

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

//select max((data->>'size')::int), count(*)
//from (select (data->>'id')::int, json_array_elements((data->>'uri')::json)
//      from spamscoreiddates28m
//      WHERE (data->>'id')::int < 1000000 ) internal;
void symantecJSON5v1(map<string, dataset> datasetCatalog) {

	int idHigh = 20000000;

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
	RecordAttribute *size = argsSymantecJSON["size"];
	RecordAttribute *uri = argsSymantecJSON["uri"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(size);

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
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed,
			nullFilter, &ctx);
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

//select max((data->>'size')::int), count(*)
//from (select (data->>'id')::int, json_array_elements((data->>'uri')::json)
//      from spamscoreiddates28m
//      WHERE (data->>'country_code') = 'RU') internal;
void symantecJSON6v1(map<string, dataset> datasetCatalog) {

//	string cityType = "Athens";
	string country_codeType = "RU";

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

//	RecordAttribute *city = argsSymantecJSON["city"];
	RecordAttribute *country_code = argsSymantecJSON["country_code"];
	RecordAttribute *size = argsSymantecJSON["size"];
	RecordAttribute *uri = argsSymantecJSON["uri"];

	vector<RecordAttribute*> projections;
//	projections.push_back(city);
	projections.push_back(country_code);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT
	 */
	list<RecordAttribute> argSelections;
//	argSelections.push_back(*city);
	argSelections.push_back(*country_code);
	argSelections.push_back(*uri);

	/* Filtering: (data->>'id')::int < 1000000 */
	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
//	expressions::Expression* selCity = new expressions::RecordProjection(
//			city->getOriginalType(), arg, *city);
	expressions::Expression* selCountryCode = new expressions::RecordProjection(
				country_code->getOriginalType(), arg, *country_code);
//	expressions::Expression* predExpr1 = new expressions::StringConstant(cityType);
	expressions::Expression* predExpr1 = new expressions::StringConstant(country_codeType);
//	expressions::Expression* predicate = new expressions::EqExpression(
//			new BoolType(), selCity, predExpr1);
	expressions::Expression* predicate = new expressions::EqExpression(
				new BoolType(), selCountryCode, predExpr1);

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
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed,
			nullFilter, &ctx);
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

	int idHigh = 40000000;

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

void symantecJSON7v1(map<string, dataset> datasetCatalog) {

	int idHigh = 40000000;

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
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed,
			nullFilter, &ctx);
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


//Takes a very long time to evaluate string predicate
void symantecJSON8(map<string, dataset> datasetCatalog) {

//	string botName = "GRUM";
	string langName = "german";

	RawContext ctx = prepareContext("symantec-json-8");
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

//	RecordAttribute *bot = argsSymantecJSON["bot"];
	RecordAttribute *lang = argsSymantecJSON["lang"];
	RecordAttribute *uri = argsSymantecJSON["uri"];

	vector<RecordAttribute*> projections;
//	projections.push_back(bot);
	projections.push_back(lang);
	projections.push_back(uri);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->>'bot') = 'GRUM' OR (data->>'lang') = 'german' )
	 */
	list<RecordAttribute> argSelections;
//	argSelections.push_back(*bot);
	argSelections.push_back(*lang);
	argSelections.push_back(*uri);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
//	expressions::Expression* selBot = new expressions::RecordProjection(
//			bot->getOriginalType(), arg, *bot);
	expressions::Expression* selLang = new expressions::RecordProjection(
				lang->getOriginalType(), arg, *lang);
//	expressions::Expression* predExpr1 = new expressions::StringConstant(botName);
	expressions::Expression* predExpr2 = new expressions::StringConstant(langName);
//	expressions::Expression* predicate1 = new expressions::EqExpression(
//			new BoolType(), selBot, predExpr1);
	expressions::Expression* predicate = new expressions::EqExpression(
				new BoolType(), selLang, predExpr2);
//	expressions::Expression* predicate = new expressions::OrExpression(
//					new BoolType(), predicate1, predicate2);

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
	sel->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
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

void symantecJSON9(map<string, dataset> datasetCatalog) {

	int yearNo = 2010;

	RawContext ctx = prepareContext("symantec-json-9");
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

	RecordAttribute *day = argsSymantecJSON["day"];


	vector<RecordAttribute*> projections;
	projections.push_back(day);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->'day'->>'year')::int = 2010
	 */
	list<RecordAttribute> argSelections;

	//Don't think this is needed
	RecordAttribute projTuple = RecordAttribute(fname, activeLoop,pg->getOIDType());
	argSelections.push_back(projTuple);

	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType yearType = IntType();
	RecordAttribute *year = new RecordAttribute(1, fname, "year", &yearType);
	expressions::Expression* selYear = new expressions::RecordProjection(
			&yearType, selDay, *year);
	expressions::Expression* predExpr1 = new expressions::IntConstant(yearNo);

	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), selYear, predExpr1);

	Select *sel = new Select(predicate,scan);
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

void symantecJSON9v1(map<string, dataset> datasetCatalog) {

	int yearNo = 2010;

	RawContext ctx = prepareContext("symantec-json-9");
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
	RecordAttribute *day = argsSymantecJSON["day"];

	vector<RecordAttribute*> projections;
	projections.push_back(day);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->'day'->>'year')::int = 2010
	 */
	list<RecordAttribute> argSelections;

	//Don't think this is needed
	RecordAttribute projTuple = RecordAttribute(fname, activeLoop,pg->getOIDType());
	argSelections.push_back(projTuple);

	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType yearType = IntType();
	RecordAttribute *year = new RecordAttribute(1, fname, "year", &yearType);
	expressions::Expression* selYear = new expressions::RecordProjection(
			&yearType, selDay, *year);
	expressions::Expression* predExpr1 = new expressions::IntConstant(yearNo);

	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), selYear, predExpr1);

	Select *sel = new Select(predicate,scan);
	scan->setParent(sel);

	/**
	 * REDUCE
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

void symantecJSON10(map<string, dataset> datasetCatalog) {

	int monthNo = 12;

	RawContext ctx = prepareContext("symantec-json-10");
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

	RecordAttribute *day = argsSymantecJSON["day"];


	vector<RecordAttribute*> projections;
	projections.push_back(day);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->'day'->>'month')::int = 12
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType monthType = IntType();
	RecordAttribute *month = new RecordAttribute(2, fname, "month", &monthType);
	expressions::Expression* selMonth = new expressions::RecordProjection(
			&monthType, selDay, *month);

	expressions::Expression* predExpr1 = new expressions::IntConstant(monthNo);

	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), selMonth, predExpr1);

	Select *sel = new Select(predicate,scan);
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

void symantecJSON10v1(map<string, dataset> datasetCatalog) {

	int monthNo = 12;

	RawContext ctx = prepareContext("symantec-json-10");
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

	RecordAttribute *day = argsSymantecJSON["day"];
	RecordAttribute *size = argsSymantecJSON["size"];


	vector<RecordAttribute*> projections;
	projections.push_back(day);

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECT: Splitting preds in numeric and non-numeric
	 * (data->'day'->>'month')::int = 12
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType monthType = IntType();
	RecordAttribute *month = new RecordAttribute(2, fname, "month", &monthType);
	expressions::Expression* selMonth = new expressions::RecordProjection(
			&monthType, selDay, *month);

	expressions::Expression* predExpr1 = new expressions::IntConstant(monthNo);

	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), selMonth, predExpr1);

	Select *sel = new Select(predicate,scan);
	scan->setParent(sel);

	/**
	 * REDUCE
	 * MAX(size), Count(*)
	 */
	list<RecordAttribute> argsReduce;
	argsReduce.push_back(*size);
	expressions::Expression* argRed = new expressions::InputArgument(&rec, 0,
			argsReduce);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), argRed, *size);

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

void symantecJSON11(map<string, dataset> datasetCatalog) {

	int idLow  = 20000000;
	int idHigh = 60000000;

	RawContext ctx = prepareContext("symantec-json-11");
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
	RecordAttribute *day = argsSymantecJSON["day"];


	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(day);

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
	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
				id->getOriginalType(), arg, *id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);

	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selID, predExpr2);
	expressions::Expression* predicate = new expressions::AndExpression(
						new BoolType(), predicate1, predicate2);

	Select *sel = new Select(predicate,scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: (data->'day'->>'year')::int
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max((data->>'size')::int)
	 */
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType yearType = IntType();
	RecordAttribute *year = new RecordAttribute(2, fname, "year", &yearType);
	expressions::Expression* selYear = new expressions::RecordProjection(
			&yearType, selDay, *year);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*size);
	nestProjections.push_back(*year);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			year->getOriginalType(), nestArg, *year);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
//	fields.push_back(year);
//	outputModes.insert(outputModes.begin(), EAGER);
//	fields.push_back(size);
//	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(size) */
	expressions::Expression* aggrSize = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	expressions::Expression* outputExpr1 = aggrSize;
	aggrField1 = string("_maxSize");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	Root *rootOp = new Root(lastPrintOp);
	lastPrintOp->setParent(rootOp);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	rootOp->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSON11v1(map<string, dataset> datasetCatalog) {

	int idLow  = 20000000;
	int idHigh = 60000000;

	RawContext ctx = prepareContext("symantec-json-11");
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
	RecordAttribute *day = argsSymantecJSON["day"];


	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(day);

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
	argSelections.push_back(*day);

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argSelections);
	expressions::Expression* selID = new expressions::RecordProjection(
				id->getOriginalType(), arg, *id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);

	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selID, predExpr2);
	expressions::Expression* predicate = new expressions::AndExpression(
						new BoolType(), predicate1, predicate2);

	Select *sel = new Select(predicate,scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: (data->'day'->>'year')::int
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max((data->>'size')::int)
	 */
	expressions::Expression* selDay = new expressions::RecordProjection(
			day->getOriginalType(), arg, *day);
	IntType yearType = IntType();
	RecordAttribute *year = new RecordAttribute(2, fname, "year", &yearType);
	expressions::Expression* selYear = new expressions::RecordProjection(
			&yearType, selDay, *year);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*size);
	nestProjections.push_back(*year);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			year->getOriginalType(), nestArg, *year);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
//	fields.push_back(year);
//	outputModes.insert(outputModes.begin(), EAGER);
//	fields.push_back(size);
//	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(size) */
	expressions::Expression* aggrSize = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	expressions::Expression* outputExpr1 = aggrSize;
	aggrField1 = string("_maxSize");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_Cnt");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);

	lastPrintOp = printOp2;

	Root *rootOp = new Root(lastPrintOp);
	lastPrintOp->setParent(rootOp);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	rootOp->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pg->finish();
	rawCatalog.clear();
}

void symantecJSONWarmup(map<string,dataset> datasetCatalog)	{

	RawContext ctx = prepareContext("symantec-json-warmup");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantec = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantec 	=
			symantec.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = symantec.path;
	RecordType rec = symantec.recType;
	int linehint = symantec.linehint;

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType,linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 */
	list<RecordAttribute> argProjections;

	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	expressions::Expression* outputExpr =
			new expressions::IntConstant(1);

	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, scan, &ctx);
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
