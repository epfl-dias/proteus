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
#include "common/tpch-config.hpp"
#include "operators/materializer-expr.hpp"

void tpchSchema(map<string,dataset>& datasetCatalog)	{
	tpchSchemaJSON(datasetCatalog);
}

/**
 * Projections
   SELECT COUNT(*)
   FROM lineitem
   WHERE l_orderkey < [X]
 */
/* COUNT() */
void tpchLineitemProjection1(map<string,dataset> datasetCatalog, int predicateVal);
/* 1 x MAX() */
void tpchLineitemProjection2(map<string,dataset> datasetCatalog, int predicateVal);
/* 1-4 aggregates */
void tpchLineitemProjection3(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo);

/* 1-4 aggregates & Materializing l_orderkey */
void tpchLineitemProjection3Caching(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{
	cout << "Execution" << endl;
	map<string,dataset> datasetCatalog;
	tpchSchema(datasetCatalog);


//	tpchLineitemProjection3(datasetCatalog, predicateMax, 4);
//	cout << "---" << endl;
//	tpchLineitemProjection3Caching(datasetCatalog,predicateMax,4);
//	cout << "---" << endl;
//	tpchLineitemProjection3(datasetCatalog, predicateMax, 4);
//	cout << "---" << endl;
	int predicateMax = L_ORDERKEY_MAX;
	for (int i = 1; i <= 10; i++) {
		double ratio = (i / (double) 10);
		double percentage = ratio * 100;
		int predicateVal = (int) ceil(predicateMax * ratio);
		cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
				<< "%" << endl;
		cout << "Query 0 (PM built if applicable)" << endl;
		tpchLineitemProjection1(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 1" << endl;
		tpchLineitemProjection1(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 2" << endl;
		tpchLineitemProjection2(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 3" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 1);
		cout << "---" << endl;
		cout << "Query 4" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 2);
		cout << "---" << endl;
		cout << "Query 5" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 3);
		cout << "---" << endl;
		cout << "Query 6" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 4);
		cout << "---" << endl;
	}

	tpchLineitemProjection3Caching(datasetCatalog, predicateMax, 4);

	cout << "CACHING (MATERIALIZING) INTO EFFECT" << endl;
	for (int i = 1; i <= 10; i++) {
		double ratio = (i / (double) 10);
		double percentage = ratio * 100;
		int predicateVal = (int) ceil(predicateMax * ratio);
		cout << "SELECTIVITY FOR key < " << predicateVal << ": " << percentage
				<< "%" << endl;
		cout << "Query 0 (PM built if applicable)" << endl;
		tpchLineitemProjection1(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 1" << endl;
		tpchLineitemProjection1(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 2" << endl;
		tpchLineitemProjection2(datasetCatalog, predicateVal);
		cout << "---" << endl;
		cout << "Query 3" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 1);
		cout << "---" << endl;
		cout << "Query 4" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 2);
		cout << "---" << endl;
		cout << "Query 5" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 3);
		cout << "---" << endl;
		cout << "Query 6" << endl;
		tpchLineitemProjection3(datasetCatalog, predicateVal, 4);
		cout << "---" << endl;
	}
}

void tpchLineitemProjection1(map<string,dataset> datasetCatalog, int predicateVal)	{

	RawContext ctx = prepareContext("tpch-json-projection1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder		=
			orders.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = lineitem.path;
	RecordType rec = lineitem.recType;
	int linehint = lineitem.linehint;
	RecordAttribute *orderkey = argsLineitem["orderkey"];

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType,linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*orderkey);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(SUM);
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr);
	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(orderkey->getOriginalType(),arg,*orderkey);
	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, vakey);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
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

/*
 SELECT MAX(quantity)
 FROM lineitem
 WHERE orderkey < [X]
 * atm: X = 3
 * quantity: float
 */
void tpchLineitemProjection2(map<string,dataset> datasetCatalog, int predicateVal)	{

	RawContext ctx = prepareContext("tpch-json-projection2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder		=
			orders.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = lineitem.path;
	RecordType rec = lineitem.recType;
	int linehint = lineitem.linehint;
	RecordAttribute *orderkey = argsLineitem["orderkey"];
	RecordAttribute *quantity = argsLineitem["quantity"];

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (MAX quantity)
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*orderkey);
	argProjections.push_back(*quantity);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(MAX);
	expressions::Expression* outputExpr =
			new expressions::RecordProjection(quantity->getOriginalType(), arg, *quantity);
	outputExprs.push_back(outputExpr);

	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(orderkey->getOriginalType(),arg,*orderkey);
	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, vakey);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
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

void tpchLineitemProjection3(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo)	{

	if(aggregatesNo <= 0 || aggregatesNo > 4)	{
		throw runtime_error(string("Invalid aggregate no. requested: "));
	}
	RawContext ctx = prepareContext("tpch-json-projection3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder		=
			orders.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = lineitem.path;
	RecordType rec = lineitem.recType;
	int linehint = lineitem.linehint;

	vector<RecordAttribute*> projections;
	RecordAttribute *orderkey = argsLineitem["orderkey"];
	RecordAttribute *linenumber = argsLineitem["linenumber"];
	RecordAttribute *quantity = argsLineitem["quantity"];
	RecordAttribute *extendedprice = argsLineitem["extendedprice"];

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * MAX(quantity), [COUNT(*), [MAX(lineitem), [MAX(extendedprice)]]]
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	if (aggregatesNo == 1 || aggregatesNo == 2) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*quantity);
	} else if (aggregatesNo == 3) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*linenumber);
		argProjections.push_back(*quantity);
	} else if (aggregatesNo == 4) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*linenumber);
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
	}

	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	switch (aggregatesNo) {
	case 1: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);
		break;
	}
	case 2: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);
		break;
	}
	case 3: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						linenumber->getOriginalType(), arg, *linenumber);
		outputExprs.push_back(outputExpr3);
		break;
	}
	case 4: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						linenumber->getOriginalType(), arg, *linenumber);
		outputExprs.push_back(outputExpr3);

		accs.push_back(MAX);
		expressions::Expression* outputExpr4 =
				new expressions::RecordProjection(
						extendedprice->getOriginalType(), arg,
						*extendedprice);
		outputExprs.push_back(outputExpr4);
		break;
	}
	default: {
		//Unreachable
		throw runtime_error(string("Invalid aggregate no. requested: "));
	}
	}

	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(orderkey->getOriginalType(),arg,*orderkey);
	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, vakey);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
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


void tpchLineitemProjection3Caching(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo)	{

	if(aggregatesNo <= 0 || aggregatesNo > 4)	{
			throw runtime_error(string("Invalid aggregate no. requested: "));
		}
		RawContext ctx = prepareContext("tpch-json-projection3");
		RawCatalog& rawCatalog = RawCatalog::getInstance();

		string nameLineitem = string("lineitem");
		dataset lineitem = datasetCatalog[nameLineitem];
		string nameOrders = string("orders");
		dataset orders = datasetCatalog[nameOrders];
		map<string, RecordAttribute*> argsLineitem 	=
				lineitem.recType.getArgsMap();
		map<string, RecordAttribute*> argsOrder		=
				orders.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = lineitem.path;
	RecordType rec = lineitem.recType;
	int linehint = lineitem.linehint;

	vector<RecordAttribute*> projections;
	RecordAttribute *orderkey = argsLineitem["orderkey"];
	RecordAttribute *linenumber = argsLineitem["linenumber"];
	RecordAttribute *quantity = argsLineitem["quantity"];
	RecordAttribute *extendedprice = argsLineitem["extendedprice"];

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	list<RecordAttribute> argProjections;
	if (aggregatesNo == 1 || aggregatesNo == 2) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*quantity);
	} else if (aggregatesNo == 3) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*linenumber);
		argProjections.push_back(*quantity);
	} else if (aggregatesNo == 4) {
		argProjections.push_back(*orderkey);
		argProjections.push_back(*linenumber);
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
	}

	expressions::Expression* arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	/*
	 * Materialize expression(s) here
	 * l_orderkey the one cached
	 */
//	RecordAttribute projTuple = RecordAttribute(fname, activeLoop, pg->getOIDType());
//		list<RecordAttribute> projections = list<RecordAttribute>();
//		projections.push_back(projTuple);
//		projections.push_back(attr);
//		projections.push_back(attr2);
//		expressions::Expression* lhsArg = new expressions::InputArgument(&attrType,
//				0, projections);
	expressions::Expression* toMat = new expressions::RecordProjection(
			orderkey->getOriginalType(), arg, *orderkey);

	char matLabel[] = "orderkeyMaterializer";
	ExprMaterializer *mat = new ExprMaterializer(toMat, linehint, scan, &ctx, matLabel);
	scan->setParent(mat);

	/**
	 * REDUCE
	 * MAX(quantity), [COUNT(*), [MAX(lineitem), [MAX(extendedprice)]]]
	 * + predicate
	 */

	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	switch (aggregatesNo) {
	case 1: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);
		break;
	}
	case 2: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);
		break;
	}
	case 3: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						linenumber->getOriginalType(), arg, *linenumber);
		outputExprs.push_back(outputExpr3);
		break;
	}
	case 4: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						arg, *quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						linenumber->getOriginalType(), arg, *linenumber);
		outputExprs.push_back(outputExpr3);

		accs.push_back(MAX);
		expressions::Expression* outputExpr4 =
				new expressions::RecordProjection(
						extendedprice->getOriginalType(), arg,
						*extendedprice);
		outputExprs.push_back(outputExpr4);
		break;
	}
	default: {
		//Unreachable
		throw runtime_error(string("Invalid aggregate no. requested: "));
	}
	}

	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(orderkey->getOriginalType(),arg,*orderkey);
	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, vakey);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
	mat->setParent(reduce);

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

