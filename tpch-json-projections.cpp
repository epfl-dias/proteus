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

typedef struct dataset	{
	string path;
	RecordType recType;
	int linehint;
} dataset;

void tpchSchemaJSON(map<string,dataset>& datasetCatalog);

/**
 * Projections
 */
/* COUNT() */
void tpchOrderProjection1(map<string,dataset> datasetCatalog, int predicateVal);
/* 1 x MAX() */
void tpchOrderProjection2(map<string,dataset> datasetCatalog, int predicateVal);
/* 1-4 aggregates */
void tpchOrderProjection3(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchemaJSON(datasetCatalog);

	int predicateVal = 3;
	tpchOrderProjection1(datasetCatalog, 3);
	tpchOrderProjection2(datasetCatalog, 3);
	tpchOrderProjection3(datasetCatalog, 3, 1);
	tpchOrderProjection3(datasetCatalog, 3, 2);
	tpchOrderProjection3(datasetCatalog, 3, 3);
	tpchOrderProjection3(datasetCatalog, 3, 4);

}

/*
 SELECT COUNT(*)
 FROM lineitem
 WHERE orderkey < [X]
 * atm: X = 3
 */
void tpchOrderProjection1(map<string,dataset> datasetCatalog, int predicateVal)	{

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
void tpchOrderProjection2(map<string,dataset> datasetCatalog, int predicateVal)	{

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

void tpchOrderProjection3(map<string,dataset> datasetCatalog, int predicateVal, int aggregatesNo)	{

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

void tpchSchemaJSON(map<string,dataset>& datasetCatalog)	{
	IntType *intType 		= new IntType();
	FloatType *floatType 	= new FloatType();
	StringType *stringType 	= new StringType();

	/* Lineitem */
	string lineitemPath = string("inputs/tpch/json/lineitem10.json");

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey =
			new RecordAttribute(1, lineitemPath, "orderkey",intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *partkey =
			new RecordAttribute(2, lineitemPath, "partkey", intType);
	attsLineitem.push_back(partkey);
	RecordAttribute *suppkey =
			new RecordAttribute(3, lineitemPath, "suppkey", intType);
	attsLineitem.push_back(suppkey);
	RecordAttribute *linenumber =
			new RecordAttribute(4, lineitemPath, "linenumber",intType);
	attsLineitem.push_back(linenumber);
	RecordAttribute *quantity =
			new RecordAttribute(5, lineitemPath, "quantity", floatType);
	attsLineitem.push_back(quantity);
	RecordAttribute *extendedprice =
			new RecordAttribute(6, lineitemPath,"extendedprice", floatType);
	attsLineitem.push_back(extendedprice);
	RecordAttribute *discount =
			new RecordAttribute(7, lineitemPath, "discount",	floatType);
	attsLineitem.push_back(discount);
	RecordAttribute *tax =
			new RecordAttribute(8, lineitemPath, "tax", floatType);
	attsLineitem.push_back(tax);
	RecordAttribute *returnflag =
			new RecordAttribute(9, lineitemPath, "returnflag", stringType);
	attsLineitem.push_back(returnflag);
	RecordAttribute *linestatus =
			new RecordAttribute(10, lineitemPath, "linestatus", stringType);
	attsLineitem.push_back(linestatus);
	RecordAttribute *shipdate =
			new RecordAttribute(11, lineitemPath, "shipdate", stringType);
	attsLineitem.push_back(shipdate);
	RecordAttribute *commitdate =
			new RecordAttribute(12, lineitemPath, "commitdate",stringType);
	attsLineitem.push_back(commitdate);
	RecordAttribute *receiptdate =
			new RecordAttribute(13, lineitemPath, "receiptdate",stringType);
	attsLineitem.push_back(receiptdate);
	RecordAttribute *shipinstruct =
			new RecordAttribute(14, lineitemPath, "shipinstruct", stringType);
	attsLineitem.push_back(shipinstruct);
	RecordAttribute *shipmode =
			new RecordAttribute(15, lineitemPath, "shipmode", stringType);
	attsLineitem.push_back(shipmode);
	RecordAttribute *l_comment =
			new RecordAttribute(16, lineitemPath, "comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);

	/* Orders */
	string ordersPath = string("inputs/tpch/json/orders10.json");

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey =
			new RecordAttribute(1, ordersPath, "orderkey",intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *custkey =
			new RecordAttribute(2, ordersPath, "custkey", intType);
	attsOrder.push_back(custkey);
	RecordAttribute *orderstatus =
			new RecordAttribute(3, ordersPath, "orderstatus", stringType);
	attsOrder.push_back(orderstatus);
	RecordAttribute *totalprice =
			new RecordAttribute(4, ordersPath, "totalprice",floatType);
	attsOrder.push_back(totalprice);
	RecordAttribute *orderdate =
			new RecordAttribute(5, ordersPath, "orderdate", stringType);
	attsOrder.push_back(orderdate);
	RecordAttribute *orderpriority =
			new RecordAttribute(6, ordersPath,"orderpriority", stringType);
	attsOrder.push_back(orderpriority);
	RecordAttribute *clerk =
			new RecordAttribute(7, ordersPath, "clerk",	stringType);
	attsOrder.push_back(clerk);
	RecordAttribute *shippriority =
			new RecordAttribute(8, ordersPath, "shippriority", intType);
	attsOrder.push_back(shippriority);
	RecordAttribute *o_comment =
			new RecordAttribute(9, ordersPath, "comment", stringType);
	attsOrder.push_back(o_comment);

	RecordType ordersRec = RecordType(attsOrder);

	dataset lineitem;
	lineitem.path = lineitemPath;
	lineitem.recType = lineitemRec;
	lineitem.linehint = 10;

	dataset orders;
	orders.path = ordersPath;
	orders.recType = ordersRec;
	orders.linehint = 10;

	datasetCatalog["lineitem"] = lineitem;
	datasetCatalog["orders"] 	= orders;
}

