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

void tpchSchemaJSON(map<string,dataset>& datasetCatalog);

/* These versions will be fully lazy -> always rely on OID */
/*
   SELECT COUNT(*)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND l_orderkey < [X]
 */
void tpchJoin1a(map<string,dataset> datasetCatalog, int predicate);
/*
   SELECT COUNT(*)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND o_orderkey < [X]
 */
void tpchJoin1b(map<string,dataset> datasetCatalog, int predicate);

/*
 * Using UNNEST instead of JOIN:
   SELECT COUNT(lineitem)
   FROM ordersLineitems
   WHERE o_orderkey < [X]
 *
 * XXX Data cannot be used for this atm..
 * Reason: Instead of being
 * {order1 , lineitems: [{},{},...,{}]},
 * they look like
 * {order1 , lineitems: {}}
 * {order1 , lineitems: {}}
 * {order1 , lineitems: {}}
 * etc.
 */
void tpchJoin1c(map<string,dataset> datasetCatalog, int predicate);

/*
   SELECT MAX(o_orderkey)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND l_orderkey < [X]
 */
void tpchJoin2a(map<string,dataset> datasetCatalog, int predicate);

/*
   SELECT MAX(l_orderkey)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND o_orderkey < [X]
 */
void tpchJoin2b(map<string,dataset> datasetCatalog, int predicate);

/*
   SELECT MAX(o_orderkey) , MAX(o_totalprice)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND l_orderkey < [X]
 */
void tpchJoin3(map<string,dataset> datasetCatalog, int predicate);

/*
   SELECT MAX(l_orderkey) , MAX(l_extendedprice)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND l_orderkey < [X]
 */
void tpchJoin4(map<string,dataset> datasetCatalog, int predicate);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchemaJSON(datasetCatalog);

	/* Make sure sides are materialized */
	cout << "Query 0a (PM + Side built if applicable)" << endl;
	tpchJoin1a(datasetCatalog,2);
	cout << "---" << endl;
	cout << "Query 0b (PM + Side built if applicable)" << endl;
	tpchJoin1b(datasetCatalog,2);
	cout << "---" << endl;

	cout << "Query 1a" << endl;
	tpchJoin1a(datasetCatalog,2);
	cout << "---" << endl;
	cout << "Query 1b" << endl;
	tpchJoin1b(datasetCatalog,2);
	cout << "---" << endl;
	cout << "Query 1c" << endl;
	tpchJoin1c(datasetCatalog, 2);
	cout << "---" << endl;
	cout << "Query 2a" << endl;
	tpchJoin2a(datasetCatalog,3);
	cout << "---" << endl;
	cout << "Query 2b" << endl;
	tpchJoin2b(datasetCatalog,3);
	cout << "---" << endl;

	/* Make sure sides are materialized */
	cout << "Query 0c (Side built if applicable)" << endl;
	tpchJoin3(datasetCatalog, 3);
	cout << "---" << endl;
	cout << "Query 0d (Side built if applicable)" << endl;
	tpchJoin4(datasetCatalog, 3);
	cout << "---" << endl;

	cout << "Query 3" << endl;
	tpchJoin3(datasetCatalog, 3);
	cout << "---" << endl;
	cout << "Query 4" << endl;
	tpchJoin4(datasetCatalog, 3);
	cout << "---" << endl;
}

void tpchJoin1a(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join1a");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	/*
	 * SELECT on LINEITEM
	 */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);

	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanLineitem);
	scanLineitem->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);
	scanOrders->setParent(join);
	sel->setParent(join);

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, join, &ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
	rawCatalog.clear();
}

void tpchJoin1b(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join1b");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	/*
	 * SELECT on ORDERS
	 */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);
	expressions::Expression *lhs = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanOrders);
	scanOrders->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);
	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
//	fieldsRight.push_back(l_orderkey);
	vector<materialization_mode> outputModesRight;
//	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
//	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
//			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);
//	expressionsRight.push_back(exprOrderkey);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, sel, scanLineitem, &ctx,
			joinLabel, *matLeft, *matRight);

	sel->setParent(join);
	scanLineitem->setParent(join);

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, join, &ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
	rawCatalog.clear();
}


void tpchJoin1c(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json/unnest-join1c");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameOrdersLineitems = string("ordersLineitems");
	dataset ordersLineitems = datasetCatalog[nameOrdersLineitems];
	map<string, RecordAttribute*> argsOrdersLineitems =
			ordersLineitems.recType.getArgsMap();

	/**
	 * SCAN : ordersLineitems
	 */
	string fileName = ordersLineitems.path;
	RecordType rec = ordersLineitems.recType;
	int linehint = ordersLineitems.linehint;


	ListType *ordersDocType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx,
			fileName, ordersDocType, linehint);

	rawCatalog.registerPlugin(fileName, pg);
	Scan *scan = new Scan(&ctx, *pg);
	cout << "3. MADE IT HERE" << endl;

	/**
	 * UNNEST:
	 */
	RecordAttribute *lineitems = argsOrdersLineitems["lineitems"];
	RecordAttribute *o_orderkey = argsOrdersLineitems["orderkey"];
//	RecordAttribute *oid = new RecordAttribute(fileName, activeLoop,
//			pg->getOIDType());
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
//	unnestProjections.push_back(*oid);
	unnestProjections.push_back(*o_orderkey);
	unnestProjections.push_back(*lineitems);

//	cout << "3a. MADE IT HERE" << endl;
	expressions::Expression* outerArg = new expressions::InputArgument(&rec, 0,
			unnestProjections);
//	cout << "3b. MADE IT HERE" << endl;
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			lineitems->getOriginalType(), outerArg, *lineitems);
//	cout << "3c. MADE IT HERE" << endl;
	string nestedName = "items";
	Path path = Path(nestedName, proj);
//	cout << "4. MADE IT HERE" << endl;

	/* Filtering on orderkey */
	expressions::Expression *lhs = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), outerArg, *o_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Unnest *unnestOp = new Unnest(pred, path, scan);
	scan->setParent(unnestOp);
//	cout << "5. MADE IT HERE" << endl;

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, unnestOp, &ctx);
	unnestOp->setParent(reduce);
//	cout << "6. MADE IT HERE" << endl;

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

void tpchJoin2a(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join2a");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	/*
	 * SELECT on LINEITEM
	 */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);
	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanLineitem);
	scanLineitem->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);
	scanOrders->setParent(join);
	sel->setParent(join);

	/**
	 * REDUCE
	 * MAX(orderkey)
	 */
	/* Output: */
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	expressions::Expression* outputExpr = exprOrderkey;
	ReduceNoPred *reduce = new ReduceNoPred(MAX, outputExpr, join, &ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
	rawCatalog.clear();
}

void tpchJoin2b(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join2b");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	//	/*
	//	 * SELECT on ORDERS
	//	 */
	//	list<RecordAttribute> argProjectionsLeft;
	//	argProjectionsLeft.push_back(*o_orderkey);
	//	expressions::Expression* leftArg = new expressions::InputArgument(
	//			&recOrders, 0, argProjectionsLeft);
	//	expressions::Expression *lhs = new expressions::RecordProjection(
	//			o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	//	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	//	expressions::Expression* pred = new expressions::LtExpression(
	//			new BoolType(), lhs, rhs);
	//
	//	Select *sel = new Select(pred, scanOrders);
	//	scanOrders->setParent(sel);

	/*
	 * SELECT on LINEITEM
	 */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);
	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanLineitem);
	scanLineitem->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);

	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	//	list<RecordAttribute> argProjectionsRight;
	//	argProjectionsRight.push_back(*l_orderkey);
	//	expressions::Expression* rightArg = new expressions::InputArgument(
	//			&recLineitem, 1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);

	scanOrders->setParent(join);
	sel->setParent(join);

	/**
	 * REDUCE
	 * MAX(orderkey)
	 */
	/* Output: */
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
				l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* outputExpr = exprOrderkey;
	ReduceNoPred *reduce = new ReduceNoPred(MAX, outputExpr, join, &ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
	rawCatalog.clear();
}

void tpchJoin3(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];
	RecordAttribute *o_totalprice = argsOrder["totalprice"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	/*
	 * SELECT on LINEITEM
	 */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);
	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanLineitem);
	scanLineitem->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);
	scanOrders->setParent(join);
	sel->setParent(join);

	/**
	 * REDUCE
	 * MAX(o_orderkey), MAX(o_totalprice)
	 */
	/* Output: */
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	expressions::Expression* exprTotalprice = new expressions::RecordProjection(
			o_totalprice->getOriginalType(), leftArg, *o_totalprice);
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(MAX);
	accs.push_back(MAX);
	outputExprs.push_back(exprOrderkey);
	outputExprs.push_back(exprTotalprice);

	expressions::Expression* outputExpr = exprOrderkey;
	opt::ReduceNoPred *reduce =
			new opt::ReduceNoPred(accs, outputExprs, join,&ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
	rawCatalog.clear();
}

void tpchJoin4(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-json-join4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

	/**
	 * SCAN 1: Orders
	 */
	string ordersName = orders.path;
	RecordType recOrders = orders.recType;
	int ordersLinehint = orders.linehint;
	RecordAttribute *o_orderkey = argsOrder["orderkey"];

	ListType *ordersDocType = new ListType(recOrders);
	jsonPipelined::JSONPlugin *pgOrders = new jsonPipelined::JSONPlugin(&ctx,
			ordersName, ordersDocType, ordersLinehint);

	rawCatalog.registerPlugin(ordersName, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemName = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	int lineitemLinehint = lineitem.linehint;
	RecordAttribute *l_orderkey = argsLineitem["orderkey"];
	RecordAttribute *l_extendedprice = argsLineitem["extendedprice"];

	ListType *lineitemDocType = new ListType(recLineitem);
	jsonPipelined::JSONPlugin *pgLineitem = new jsonPipelined::JSONPlugin(&ctx,
			lineitemName, lineitemDocType, lineitemLinehint);

	rawCatalog.registerPlugin(lineitemName, pgLineitem);
	Scan *scanLineitem = new Scan(&ctx, *pgLineitem);

	//	/*
	//	 * SELECT on ORDERS
	//	 */
	//	list<RecordAttribute> argProjectionsLeft;
	//	argProjectionsLeft.push_back(*o_orderkey);
	//	expressions::Expression* leftArg = new expressions::InputArgument(
	//			&recOrders, 0, argProjectionsLeft);
	//	expressions::Expression *lhs = new expressions::RecordProjection(
	//			o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	//	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	//	expressions::Expression* pred = new expressions::LtExpression(
	//			new BoolType(), lhs, rhs);
	//
	//	Select *sel = new Select(pred, scanOrders);
	//	scanOrders->setParent(sel);

	/*
	 * SELECT on LINEITEM
	 */
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*l_orderkey);
	expressions::Expression* rightArg = new expressions::InputArgument(
			&recLineitem, 1, argProjectionsRight);
	expressions::Expression *lhs = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scanLineitem);
	scanLineitem->setParent(sel);

	/**
	 * JOIN
	 */
	/* join key - orders */
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*o_orderkey);
	expressions::Expression* leftArg = new expressions::InputArgument(
			&recOrders, 0, argProjectionsLeft);

	expressions::Expression* leftPred = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), leftArg, *o_orderkey);

	/* join key - lineitem */
	//	list<RecordAttribute> argProjectionsRight;
	//	argProjectionsRight.push_back(*l_orderkey);
	//	expressions::Expression* rightArg = new expressions::InputArgument(
	//			&recLineitem, 1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersName, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemName, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);

	scanOrders->setParent(join);
	sel->setParent(join);

	/**
	 * REDUCE
	 * MAX(orderkey)
	 */
	/* Output: */
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* exprExtendedprice =
			new expressions::RecordProjection(
					l_extendedprice->getOriginalType(), rightArg,
					*l_extendedprice);
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(MAX);
	accs.push_back(MAX);
	outputExprs.push_back(exprOrderkey);
	outputExprs.push_back(exprExtendedprice);

	opt::ReduceNoPred *reduce = new opt::ReduceNoPred(accs, outputExprs, join, &ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgOrders->finish();
	pgLineitem->finish();
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


	/* XXX ERROR PRONE...
	 * outer and nested attributes have the same relationName
	 * Don't know how caches will work in this scenario.
	 *
	 * The rest of the code should be sanitized,
	 * since readPath is applied in steps.
	 */

	/* OrdersLineitems
	 * i.e., pre-materialized join */
	string ordersLineitemsPath =
			string("inputs/tpch/json/ordersLineitemsArray10.json");

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
	ListType *lineitemArray = new ListType(*lineitemNestedRec);

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
				ordersLineitemsPath, "lineitems", lineitemArray);
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
