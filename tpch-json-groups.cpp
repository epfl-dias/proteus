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
#include "operators/radix-join.hpp"
#include "operators/unnest.hpp"
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "operators/reduce.hpp"
#include "operators/reduce-nopred.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/reduce-opt-nopred.hpp"
#include "operators/radix-nest.hpp"
#include "plugins/json-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "util/raw-caching.hpp"

typedef struct dataset	{
	string path;
	RecordType recType;
	int linehint;
} dataset;

void tpchSchemaJSON(map<string,dataset>& datasetCatalog);

/* Numbers of lineitems per order
   SELECT COUNT(*)
   FROM lineitem
   WHERE l_orderkey < [X]
   GROUP BY l_orderkey < [X]
 */
void tpchGroup(map<string,dataset> datasetCatalog, int predicate, int aggregatesNo);

/* FIXME Need a case featuring Unnest too */

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchemaJSON(datasetCatalog);

	tpchGroup(datasetCatalog,0,1);
	tpchGroup(datasetCatalog,0,2);
	tpchGroup(datasetCatalog,0,3);
	tpchGroup(datasetCatalog,0,4);
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
	int lineHint = lineitem.linehint;

	vector<RecordAttribute*> projections;
	RecordAttribute *orderkey = argsLineitem["orderkey"];
	RecordAttribute *quantity = NULL;
	RecordAttribute *extendedprice = NULL;
	RecordAttribute *tax = NULL;

	projections.push_back(orderkey);
	if (aggregatesNo == 2) {
		quantity = argsLineitem["quantity"];
		projections.push_back(quantity);
	}
	if (aggregatesNo == 3) {
		quantity = argsLineitem["quantity"];
		projections.push_back(quantity);
		extendedprice = argsLineitem["extendedprice"];
		projections.push_back(extendedprice);
	}
	if (aggregatesNo == 4) {
		quantity = argsLineitem["quantity"];
		projections.push_back(quantity);
		extendedprice = argsLineitem["extendedprice"];
		projections.push_back(extendedprice);
		tax = argsLineitem["tax"];
		projections.push_back(tax);
	}

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, lineHint);
//	jsmn::JSONPlugin *pg = new jsmn::JSONPlugin(&ctx, fname, documentType);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*orderkey);
	if (aggregatesNo == 2) {
		argProjections.push_back(*quantity);
	}
	if (aggregatesNo == 3) {
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
	}
	if (aggregatesNo == 4) {
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
		argProjections.push_back(*tax);
	}
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	expressions::Expression *lhs = new expressions::RecordProjection(
			orderkey->getOriginalType(), arg, *orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: orderkey
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*orderkey);
	if (aggregatesNo == 2) {
		nestProjections.push_back(*quantity);
	}
	if (aggregatesNo == 3) {
		nestProjections.push_back(*quantity);
		nestProjections.push_back(*extendedprice);
	}
	if (aggregatesNo == 4) {
		nestProjections.push_back(*quantity);
		nestProjections.push_back(*extendedprice);
		nestProjections.push_back(*tax);
	}
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);
	//f (& g)
	expressions::RecordProjection* f = new expressions::RecordProjection(
			orderkey->getOriginalType(), nestArg, *orderkey);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	//fields.push_back(orderkey);
	//outputModes.insert(outputModes.begin(), EAGER);


//	if (aggregatesNo == 2) {
//		fields.push_back(quantity);
//		outputModes.insert(outputModes.begin(), EAGER);
//	}
//	if (aggregatesNo == 3) {
//		fields.push_back(quantity);
//		outputModes.insert(outputModes.begin(), EAGER);
//		fields.push_back(extendedprice);
//		outputModes.insert(outputModes.begin(), EAGER);
//	}
//	if (aggregatesNo == 4) {
//		fields.push_back(quantity);
//		outputModes.insert(outputModes.begin(), EAGER);
//		fields.push_back(extendedprice);
//		outputModes.insert(outputModes.begin(), EAGER);
//		fields.push_back(tax);
//		outputModes.insert(outputModes.begin(), EAGER);
//	}
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
		/* + Aggregate 2: MAX(quantity) */
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						nestArg, *quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);

	}
	if (aggregatesNo == 3) {
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						nestArg, *quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);
		/* + Aggregate 3: MAX(extendedprice) */
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						extendedprice->getOriginalType(), nestArg,
						*extendedprice);
		aggrField3 = string("_aggrMaxExtPrice");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr3);
		aggrLabels.push_back(aggrField3);
	}
	if (aggregatesNo == 4) {
		expressions::Expression* outputExpr2 =
				new expressions::RecordProjection(quantity->getOriginalType(),
						nestArg, *quantity);
		aggrField2 = string("_aggrMaxQuantity");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr2);
		aggrLabels.push_back(aggrField2);
		expressions::Expression* outputExpr3 =
				new expressions::RecordProjection(
						extendedprice->getOriginalType(), nestArg,
						*extendedprice);
		aggrField3 = string("_aggrMaxExtPrice");
		accs.push_back(MAX);
		outputExprs.push_back(outputExpr3);
		aggrLabels.push_back(aggrField3);
		/* + Aggregate 4: MAX(extendedprice) */
		expressions::Expression* outputExpr4 =
				new expressions::RecordProjection(tax->getOriginalType(),
						nestArg, *tax);
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


void tpchSchemaJSON(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	/* Lineitem */
	/* FIXME once hash for json pg is also fixed */
	string lineitemPath = string("inputs/tpch/json/lineitem10.json");

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey = new RecordAttribute(1, lineitemPath,
			"orderkey", intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *partkey = new RecordAttribute(2, lineitemPath, "partkey",
			intType);
	attsLineitem.push_back(partkey);
	RecordAttribute *suppkey = new RecordAttribute(3, lineitemPath, "suppkey",
			intType);
	attsLineitem.push_back(suppkey);
	RecordAttribute *linenumber = new RecordAttribute(4, lineitemPath,
			"linenumber", intType);
	attsLineitem.push_back(linenumber);
	RecordAttribute *quantity = new RecordAttribute(5, lineitemPath, "quantity",
			floatType);
	attsLineitem.push_back(quantity);
	RecordAttribute *extendedprice = new RecordAttribute(6, lineitemPath,
			"extendedprice", floatType);
	attsLineitem.push_back(extendedprice);
	RecordAttribute *discount = new RecordAttribute(7, lineitemPath, "discount",
			floatType);
	attsLineitem.push_back(discount);
	RecordAttribute *tax = new RecordAttribute(8, lineitemPath, "tax",
			floatType);
	attsLineitem.push_back(tax);
	RecordAttribute *returnflag = new RecordAttribute(9, lineitemPath,
			"returnflag", stringType);
	attsLineitem.push_back(returnflag);
	RecordAttribute *linestatus = new RecordAttribute(10, lineitemPath,
			"linestatus", stringType);
	attsLineitem.push_back(linestatus);
	RecordAttribute *shipdate = new RecordAttribute(11, lineitemPath,
			"shipdate", stringType);
	attsLineitem.push_back(shipdate);
	RecordAttribute *commitdate = new RecordAttribute(12, lineitemPath,
			"commitdate", stringType);
	attsLineitem.push_back(commitdate);
	RecordAttribute *receiptdate = new RecordAttribute(13, lineitemPath,
			"receiptdate", stringType);
	attsLineitem.push_back(receiptdate);
	RecordAttribute *shipinstruct = new RecordAttribute(14, lineitemPath,
			"shipinstruct", stringType);
	attsLineitem.push_back(shipinstruct);
	RecordAttribute *shipmode = new RecordAttribute(15, lineitemPath,
			"shipmode", stringType);
	attsLineitem.push_back(shipmode);
	RecordAttribute *l_comment = new RecordAttribute(16, lineitemPath,
			"comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);

	/* Orders */
	string ordersPath = string("inputs/tpch/json/orders10.json");

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey = new RecordAttribute(1, ordersPath, "orderkey",
			intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *custkey = new RecordAttribute(2, ordersPath, "custkey",
			intType);
	attsOrder.push_back(custkey);
	RecordAttribute *orderstatus = new RecordAttribute(3, ordersPath,
			"orderstatus", stringType);
	attsOrder.push_back(orderstatus);
	RecordAttribute *totalprice = new RecordAttribute(4, ordersPath,
			"totalprice", floatType);
	attsOrder.push_back(totalprice);
	RecordAttribute *orderdate = new RecordAttribute(5, ordersPath, "orderdate",
			stringType);
	attsOrder.push_back(orderdate);
	RecordAttribute *orderpriority = new RecordAttribute(6, ordersPath,
			"orderpriority", stringType);
	attsOrder.push_back(orderpriority);
	RecordAttribute *clerk = new RecordAttribute(7, ordersPath, "clerk",
			stringType);
	attsOrder.push_back(clerk);
	RecordAttribute *shippriority = new RecordAttribute(8, ordersPath,
			"shippriority", intType);
	attsOrder.push_back(shippriority);
	RecordAttribute *o_comment = new RecordAttribute(9, ordersPath, "comment",
			stringType);
	attsOrder.push_back(o_comment);

	RecordType ordersRec = RecordType(attsOrder);

	/* XXX ERROR PRONE...
	 * outer and nested attributes have the same relationName
	 * Don't know how caches will work in this scenario.
	 *
	 * The rest of the code should be sanitized,
	 * since readPath is applied in steps.
	 */

	/* OrdersLineitems
	 * i.e., pre-materialized join */
	string ordersLineitemsPath = string(
			"inputs/tpch/json/ordersLineitems10.json");

	/*
	 * The lineitem entries in the ordersLineitems objects
	 * do not contain the orderkey (again)
	 */
	list<RecordAttribute*> attsLineitemNested = list<RecordAttribute*>();
	{
		RecordAttribute *partkey = new RecordAttribute(1, ordersLineitemsPath,
				"partkey", intType);
		attsLineitemNested.push_back(partkey);
		RecordAttribute *suppkey = new RecordAttribute(2, ordersLineitemsPath,
				"suppkey", intType);
		attsLineitemNested.push_back(suppkey);
		RecordAttribute *linenumber = new RecordAttribute(3,
				ordersLineitemsPath, "linenumber", intType);
		attsLineitemNested.push_back(linenumber);
		RecordAttribute *quantity = new RecordAttribute(4, ordersLineitemsPath,
				"quantity", floatType);
		attsLineitemNested.push_back(quantity);
		RecordAttribute *extendedprice = new RecordAttribute(5,
				ordersLineitemsPath, "extendedprice", floatType);
		attsLineitemNested.push_back(extendedprice);
		RecordAttribute *discount = new RecordAttribute(6, ordersLineitemsPath,
				"discount", floatType);
		attsLineitemNested.push_back(discount);
		RecordAttribute *tax = new RecordAttribute(7, ordersLineitemsPath,
				"tax", floatType);
		attsLineitemNested.push_back(tax);
		RecordAttribute *returnflag = new RecordAttribute(8,
				ordersLineitemsPath, "returnflag", stringType);
		attsLineitemNested.push_back(returnflag);
		RecordAttribute *linestatus = new RecordAttribute(9,
				ordersLineitemsPath, "linestatus", stringType);
		attsLineitemNested.push_back(linestatus);
		RecordAttribute *shipdate = new RecordAttribute(10, ordersLineitemsPath,
				"shipdate", stringType);
		attsLineitemNested.push_back(shipdate);
		RecordAttribute *commitdate = new RecordAttribute(11,
				ordersLineitemsPath, "commitdate", stringType);
		attsLineitemNested.push_back(commitdate);
		RecordAttribute *receiptdate = new RecordAttribute(12,
				ordersLineitemsPath, "receiptdate", stringType);
		attsLineitemNested.push_back(receiptdate);
		RecordAttribute *shipinstruct = new RecordAttribute(13,
				ordersLineitemsPath, "shipinstruct", stringType);
		attsLineitemNested.push_back(shipinstruct);
		RecordAttribute *shipmode = new RecordAttribute(14, ordersLineitemsPath,
				"shipmode", stringType);
		attsLineitemNested.push_back(shipmode);
		RecordAttribute *l_comment = new RecordAttribute(15,
				ordersLineitemsPath, "comment", stringType);
		attsLineitemNested.push_back(l_comment);
	}
	RecordType *lineitemNestedRec = new RecordType(attsLineitemNested);

	list<RecordAttribute*> attsOrdersLineitems = list<RecordAttribute*>();
	{
		RecordAttribute *o_orderkey = new RecordAttribute(1,
				ordersLineitemsPath, "orderkey", intType);
		attsOrdersLineitems.push_back(o_orderkey);
		RecordAttribute *custkey = new RecordAttribute(2, ordersLineitemsPath,
				"custkey", intType);
		attsOrdersLineitems.push_back(custkey);
		RecordAttribute *orderstatus = new RecordAttribute(3,
				ordersLineitemsPath, "orderstatus", stringType);
		attsOrdersLineitems.push_back(orderstatus);
		RecordAttribute *totalprice = new RecordAttribute(4,
				ordersLineitemsPath, "totalprice", floatType);
		attsOrdersLineitems.push_back(totalprice);
		RecordAttribute *orderdate = new RecordAttribute(5, ordersLineitemsPath,
				"orderdate", stringType);
		attsOrdersLineitems.push_back(orderdate);
		RecordAttribute *orderpriority = new RecordAttribute(6,
				ordersLineitemsPath, "orderpriority", stringType);
		attsOrdersLineitems.push_back(orderpriority);
		RecordAttribute *clerk = new RecordAttribute(7, ordersLineitemsPath,
				"clerk", stringType);
		attsOrdersLineitems.push_back(clerk);
		RecordAttribute *shippriority = new RecordAttribute(8,
				ordersLineitemsPath, "shippriority", intType);
		attsOrdersLineitems.push_back(shippriority);
		RecordAttribute *o_comment = new RecordAttribute(9, ordersLineitemsPath,
				"comment", stringType);
		attsOrdersLineitems.push_back(o_comment);
		RecordAttribute *lineitems = new RecordAttribute(10,
				ordersLineitemsPath, "lineitems", lineitemNestedRec);
		attsOrdersLineitems.push_back(lineitems);
	}
	RecordType ordersLineitemsRec = RecordType(attsOrdersLineitems);

	dataset lineitem;
	lineitem.path = lineitemPath;
	lineitem.recType = lineitemRec;
	lineitem.linehint = 10;

	dataset orders;
	orders.path = ordersPath;
	orders.recType = ordersRec;
	orders.linehint = 10;

	dataset ordersLineitems;
	ordersLineitems.path = ordersLineitemsPath;
	ordersLineitems.recType = ordersLineitemsRec;
	ordersLineitems.linehint = 10;

	datasetCatalog["lineitem"] = lineitem;
	datasetCatalog["orders"] = orders;
	datasetCatalog["ordersLineitems"] = ordersLineitems;
}
