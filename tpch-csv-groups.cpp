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
#include "operators/reduce-opt-nopred.hpp"
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

/* Numbers of lineitems per order
   SELECT COUNT(*), [...]
   FROM lineitem
   WHERE l_orderkey < [X]
   GROUP BY l_orderkey
 */
void tpchGroup(map<string,dataset> datasetCatalog, int predicate, int aggregatesNo);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchema(datasetCatalog);

	cout << "Query 0 (PM + Side built if applicable)" << endl;
	tpchGroup(datasetCatalog,0,4);
	cout << "---" << endl;
	cout << "Query 1 (aggr.)" << endl;
	tpchGroup(datasetCatalog,0,1);
	cout << "---" << endl;
	cout << "Query 2 (aggr.)" << endl;
	tpchGroup(datasetCatalog,0,2);
	cout << "---" << endl;
	cout << "Query 3 (aggr.)" << endl;
	tpchGroup(datasetCatalog,0,3);
	cout << "---" << endl;
	cout << "Query 4 (aggr.)" << endl;
	tpchGroup(datasetCatalog,0,4);
	cout << "---" << endl;
}


void tpchGroup(map<string, dataset> datasetCatalog, int predicate, int aggregatesNo) {

	if(aggregatesNo < 1 || aggregatesNo > 4)	{
		throw runtime_error(string("Invalid no. of aggregates requested: "));
	}
	RawContext ctx = prepareContext("tpch-csv-selection1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = lineitem.path;
	RecordType rec = lineitem.recType;
	int policy = 5;
	int lineHint = lineitem.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];//NULL;
	RecordAttribute *l_quantity = NULL;
	RecordAttribute *l_extendedprice = NULL;
	RecordAttribute *l_tax = NULL;

//	l_orderkey = argsLineitem["l_orderkey"];
	projections.push_back(l_orderkey);
	if (aggregatesNo == 2) {
		l_quantity = argsLineitem["l_quantity"];
		projections.push_back(l_quantity);
	}
	if (aggregatesNo == 3) {
		l_quantity = argsLineitem["l_quantity"];
		projections.push_back(l_quantity);
		l_extendedprice = argsLineitem["l_extendedprice"];
		projections.push_back(l_extendedprice);
	}
	if (aggregatesNo == 4) {
		l_quantity = argsLineitem["l_quantity"];
		projections.push_back(l_quantity);
		l_extendedprice = argsLineitem["l_extendedprice"];
		projections.push_back(l_extendedprice);
		l_tax = argsLineitem["l_tax"];
		projections.push_back(l_tax);
	}

	pm::CSVPlugin* pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
			delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	if (aggregatesNo == 2) {
		argProjections.push_back(*l_quantity);
	}
	if (aggregatesNo == 3) {
		argProjections.push_back(*l_quantity);
		argProjections.push_back(*l_extendedprice);
	}
	if (aggregatesNo == 4) {
		argProjections.push_back(*l_quantity);
		argProjections.push_back(*l_extendedprice);
		argProjections.push_back(*l_tax);
	}
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: l_orderkey
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*l_orderkey);
	if (aggregatesNo == 2) {
		nestProjections.push_back(*l_quantity);
	}
	if (aggregatesNo == 3) {
		nestProjections.push_back(*l_quantity);
		nestProjections.push_back(*l_extendedprice);
	}
	if (aggregatesNo == 4) {
		nestProjections.push_back(*l_quantity);
		nestProjections.push_back(*l_extendedprice);
		nestProjections.push_back(*l_tax);
	}
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);
	//f (& g)
	expressions::RecordProjection* f = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), nestArg, *l_orderkey);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(l_orderkey);
	outputModes.insert(outputModes.begin(), EAGER);


	if (aggregatesNo == 2) {
		fields.push_back(l_quantity);
		outputModes.insert(outputModes.begin(), EAGER);
	}
	if (aggregatesNo == 3) {
		fields.push_back(l_quantity);
		outputModes.insert(outputModes.begin(), EAGER);
		fields.push_back(l_extendedprice);
		outputModes.insert(outputModes.begin(), EAGER);
	}
	if (aggregatesNo == 4) {
		fields.push_back(l_quantity);
		outputModes.insert(outputModes.begin(), EAGER);
		fields.push_back(l_extendedprice);
		outputModes.insert(outputModes.begin(), EAGER);
		fields.push_back(l_tax);
		outputModes.insert(outputModes.begin(), EAGER);
	}
	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_lineitem";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;
	string aggrField3;
	string aggrField4;

	/* Aggregate 1: COUNT(*) */
	expressions::Expression* outputExpr1 = new expressions::IntConstant(1);
	aggrField1 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);
	if (aggregatesNo == 2) {
		/* + Aggregate 2: MAX(l_quantity) */
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						nestArg, *l_quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);

	}
	if (aggregatesNo == 3) {
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						nestArg, *l_quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);
		/* + Aggregate 3: MAX(l_extendedprice) */
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						l_extendedprice->getOriginalType(), nestArg,
						*l_extendedprice);
		aggrField3 = string("_aggrMaxExtPrice");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr3);
		aggrLabels.push_back(aggrField3);
	}
	if (aggregatesNo == 4) {
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(l_quantity->getOriginalType(),
						nestArg, *l_quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						l_extendedprice->getOriginalType(), nestArg,
						*l_extendedprice);
		aggrField3 = string("_aggrMaxExtPrice");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr3);
		aggrLabels.push_back(aggrField3);
		/* + Aggregate 4: MAX(l_extendedprice) */
		expressions::Expression* outputExpr4 =
				new expressions::RecordProjection(l_tax->getOriginalType(),
						nestArg, *l_tax);
		aggrField4 = string("_aggrMaxTax");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr4);
		aggrLabels.push_back(aggrField4);
	}

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	FloatType floatType = FloatType();

	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;
	if (aggregatesNo == 2) {
		RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
				&floatType);
		expressions::RecordProjection* nestOutput2 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput2);
		Print *printOp2 = new Print(debugFloat, nestOutput2, printOp1);
		printOp1->setParent(printOp2);
		lastPrintOp = printOp2;
	}
	if (aggregatesNo == 3) {
		RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel,
				aggrField2, &floatType);
		expressions::RecordProjection* nestOutput2 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput2);
		Print *printOp2 = new Print(debugFloat, nestOutput2, printOp1);
		printOp1->setParent(printOp2);

		RecordAttribute *toOutput3 = new RecordAttribute(3, aggrLabel,
				aggrField3, &floatType);
		expressions::RecordProjection* nestOutput3 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput3);
		Print *printOp3 = new Print(debugFloat, nestOutput3, printOp2);
		printOp2->setParent(printOp3);
		lastPrintOp = printOp3;
	}
	if (aggregatesNo == 4) {
		RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel,
				aggrField2, &floatType);
		expressions::RecordProjection* nestOutput2 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput2);
		Print *printOp2 = new Print(debugFloat, nestOutput2, printOp1);
		printOp1->setParent(printOp2);

		RecordAttribute *toOutput3 = new RecordAttribute(3, aggrLabel,
				aggrField3, &floatType);
		expressions::RecordProjection* nestOutput3 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput3);
		Print *printOp3 = new Print(debugFloat, nestOutput3, printOp2);
		printOp2->setParent(printOp3);

		RecordAttribute *toOutput4 = new RecordAttribute(4, aggrLabel,
				aggrField4, &floatType);
		expressions::RecordProjection* nestOutput4 =
				new expressions::RecordProjection(&floatType, nestArg,
						*toOutput4);
		Print *printOp4 = new Print(debugFloat, nestOutput4, printOp3);
		printOp3->setParent(printOp4);

		lastPrintOp = printOp4;
	}

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
