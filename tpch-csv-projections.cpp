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
#include "plugins/json-jsmn-plugin.hpp"
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

void tpchSchema(map<string,dataset>& datasetCatalog);

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
	tpchSchema(datasetCatalog);

	int predicateVal = 3;
	cout << "Query 0 (PM built if applicable)" << endl;
	tpchOrderProjection1(datasetCatalog, 3);
	cout << "---" << endl;
	cout << "Query 1" << endl;
	tpchOrderProjection1(datasetCatalog, 3);
	cout << "---" << endl;
	cout << "Query 2" << endl;
	tpchOrderProjection2(datasetCatalog, 3);
	cout << "---" << endl;
	cout << "Query 3" << endl;
	tpchOrderProjection3(datasetCatalog, 3, 1);
	cout << "---" << endl;
	cout << "Query 4" << endl;
	tpchOrderProjection3(datasetCatalog, 3, 2);
	cout << "---" << endl;
	cout << "Query 5" << endl;
	tpchOrderProjection3(datasetCatalog, 3, 3);
	cout << "---" << endl;
	cout << "Query 6" << endl;
	tpchOrderProjection3(datasetCatalog, 3, 4);
	cout << "---" << endl;

}

/*
 SELECT COUNT(*)
 FROM lineitem
 WHERE l_orderkey < [X]
 * atm: X = 3
 */
void tpchOrderProjection1(map<string,dataset> datasetCatalog, int predicateVal)	{

	RawContext ctx = prepareContext("tpch-csv-projection1");
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
	int policy = 5;
	int lineHint = lineitem.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	projections.push_back(l_orderkey);


	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
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
			new expressions::RecordProjection(l_orderkey->getOriginalType(),arg,*l_orderkey);
	expressions::Expression* val_key = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, val_key);

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
 SELECT MAX(l_quantity)
 FROM lineitem
 WHERE l_orderkey < [X]
 * atm: X = 3
 * l_quantity: float
 */
void tpchOrderProjection2(map<string,dataset> datasetCatalog, int predicateVal)	{

	RawContext ctx = prepareContext("tpch-csv-projection2");
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
	int policy = 5;
	int lineHint = lineitem.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	RecordAttribute *l_quantity = argsLineitem["l_quantity"];
	projections.push_back(l_orderkey);
	projections.push_back(l_quantity);


	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (MAX l_quantity)
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	argProjections.push_back(*l_quantity);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(MAX);
	expressions::Expression* outputExpr =
			new expressions::RecordProjection(l_quantity->getOriginalType(), arg, *l_quantity);
	outputExprs.push_back(outputExpr);

	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(l_orderkey->getOriginalType(),arg,*l_orderkey);
	expressions::Expression* val_key = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, val_key);

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
	RawContext ctx = prepareContext("tpch-csv-projection3");
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
	int policy = 5;
	int lineHint = lineitem.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
	RecordAttribute *l_quantity = argsLineitem["l_quantity"];
	RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];

	if (aggregatesNo == 1 || aggregatesNo == 2) {
		projections.push_back(l_orderkey);
		projections.push_back(l_quantity);
	} else if (aggregatesNo == 3) {
		projections.push_back(l_orderkey);
		projections.push_back(l_linenumber);
		projections.push_back(l_quantity);
	} else if (aggregatesNo == 4) {
		projections.push_back(l_orderkey);
		projections.push_back(l_linenumber);
		projections.push_back(l_quantity);
		projections.push_back(l_extendedprice);
	}

	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * MAX(l_quantity), [COUNT(*), [MAX(l_lineitem), [MAX(l_extendedprice)]]]
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	if (aggregatesNo == 1 || aggregatesNo == 2) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_quantity);
	} else if (aggregatesNo == 3) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_linenumber);
		argProjections.push_back(*l_quantity);
	} else if (aggregatesNo == 4) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_linenumber);
		argProjections.push_back(*l_quantity);
		argProjections.push_back(*l_extendedprice);
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
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						arg, *l_quantity);
		outputExprs.push_back(outputExpr1);
		break;
	}
	case 2: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						arg, *l_quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);
		break;
	}
	case 3: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						arg, *l_quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						l_linenumber->getOriginalType(), arg, *l_linenumber);
		outputExprs.push_back(outputExpr3);
		break;
	}
	case 4: {
		accs.push_back(MAX);
		expressions::Expression* outputExpr1 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						arg, *l_quantity);
		outputExprs.push_back(outputExpr1);

		accs.push_back(SUM);
		expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
		outputExprs.push_back(outputExpr2);

		accs.push_back(MAX);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						l_linenumber->getOriginalType(), arg, *l_linenumber);
		outputExprs.push_back(outputExpr3);

		accs.push_back(MAX);
		expressions::Expression* outputExpr4 =
				new expressions::RecordProjection(
						l_extendedprice->getOriginalType(), arg,
						*l_extendedprice);
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
			new expressions::RecordProjection(l_orderkey->getOriginalType(),arg,*l_orderkey);
	expressions::Expression* val_key = new expressions::IntConstant(predicateVal);
	expressions::Expression* predicate = new expressions::LtExpression(
				new BoolType(), selOrderkey, val_key);

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

void tpchSchema(map<string,dataset>& datasetCatalog)	{
	IntType *intType 		= new IntType();
	FloatType *floatType 	= new FloatType();
	StringType *stringType 	= new StringType();

	/* Lineitem */
	string lineitemPath = string("inputs/tpch/lineitem10.csv");

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey =
			new RecordAttribute(1, lineitemPath, "l_orderkey",intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *l_partkey =
			new RecordAttribute(2, lineitemPath, "l_partkey", intType);
	attsLineitem.push_back(l_partkey);
	RecordAttribute *l_suppkey =
			new RecordAttribute(3, lineitemPath, "l_suppkey", intType);
	attsLineitem.push_back(l_suppkey);
	RecordAttribute *l_linenumber =
			new RecordAttribute(4, lineitemPath, "l_linenumber",intType);
	attsLineitem.push_back(l_linenumber);
	RecordAttribute *l_quantity =
			new RecordAttribute(5, lineitemPath, "l_quantity", floatType);
	attsLineitem.push_back(l_quantity);
	RecordAttribute *l_extendedprice =
			new RecordAttribute(6, lineitemPath,"l_extendedprice", floatType);
	attsLineitem.push_back(l_extendedprice);
	RecordAttribute *l_discount =
			new RecordAttribute(7, lineitemPath, "l_discount",	floatType);
	attsLineitem.push_back(l_discount);
	RecordAttribute *l_tax =
			new RecordAttribute(8, lineitemPath, "l_tax", floatType);
	attsLineitem.push_back(l_tax);
	RecordAttribute *l_returnflag =
			new RecordAttribute(9, lineitemPath, "l_returnflag", stringType);
	attsLineitem.push_back(l_returnflag);
	RecordAttribute *l_linestatus =
			new RecordAttribute(10, lineitemPath, "l_linestatus", stringType);
	attsLineitem.push_back(l_linestatus);
	RecordAttribute *l_shipdate =
			new RecordAttribute(11, lineitemPath, "l_shipdate", stringType);
	attsLineitem.push_back(l_shipdate);
	RecordAttribute *l_commitdate =
			new RecordAttribute(12, lineitemPath, "l_commitdate",stringType);
	attsLineitem.push_back(l_commitdate);
	RecordAttribute *l_receiptdate =
			new RecordAttribute(13, lineitemPath, "l_receiptdate",stringType);
	attsLineitem.push_back(l_receiptdate);
	RecordAttribute *l_shipinstruct =
			new RecordAttribute(14, lineitemPath, "l_shipinstruct", stringType);
	attsLineitem.push_back(l_shipinstruct);
	RecordAttribute *l_shipmode =
			new RecordAttribute(15, lineitemPath, "l_shipmode", stringType);
	attsLineitem.push_back(l_shipmode);
	RecordAttribute *l_comment =
			new RecordAttribute(16, lineitemPath, "l_comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);

	/* Orders */
	string ordersPath = string("inputs/tpch/orders10.csv");

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey =
			new RecordAttribute(1, ordersPath, "o_orderkey",intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *o_custkey =
			new RecordAttribute(2, ordersPath, "o_custkey", intType);
	attsOrder.push_back(o_custkey);
	RecordAttribute *o_orderstatus =
			new RecordAttribute(3, ordersPath, "o_orderstatus", stringType);
	attsOrder.push_back(o_orderstatus);
	RecordAttribute *o_totalprice =
			new RecordAttribute(4, ordersPath, "o_totalprice",floatType);
	attsOrder.push_back(o_totalprice);
	RecordAttribute *o_orderdate =
			new RecordAttribute(5, ordersPath, "o_orderdate", stringType);
	attsOrder.push_back(o_orderdate);
	RecordAttribute *o_orderpriority =
			new RecordAttribute(6, ordersPath,"o_orderpriority", stringType);
	attsOrder.push_back(o_orderpriority);
	RecordAttribute *o_clerk =
			new RecordAttribute(7, ordersPath, "o_clerk",	stringType);
	attsOrder.push_back(o_clerk);
	RecordAttribute *o_shippriority =
			new RecordAttribute(8, ordersPath, "o_shippriority", intType);
	attsOrder.push_back(o_shippriority);
	RecordAttribute *o_comment =
			new RecordAttribute(9, ordersPath, "o_comment", stringType);
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

