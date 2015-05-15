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
#include "common/tpch-config.hpp"

void tpchSchema(map<string,dataset>& datasetCatalog)	{
	tpchSchemaJSON(datasetCatalog);
}
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


//int main()	{
//
//	map<string,dataset> datasetCatalog;
//	tpchSchemaJSON(datasetCatalog);
//
//	/* Make sure sides are materialized */
//	cout << "Query 0a (PM + Side built if applicable)" << endl;
//	tpchJoin1a(datasetCatalog,2);
//	cout << "---" << endl;
//	cout << "Query 0b (PM + Side built if applicable)" << endl;
//	tpchJoin1b(datasetCatalog,2);
//	cout << "---" << endl;
//
//	cout << "Query 1a" << endl;
//	tpchJoin1a(datasetCatalog,2);
//	cout << "---" << endl;
//	cout << "Query 1b" << endl;
//	tpchJoin1b(datasetCatalog,2);
//	cout << "---" << endl;
//	cout << "Query 1c" << endl;
//	tpchJoin1c(datasetCatalog, 2);
//	cout << "---" << endl;
//	cout << "Query 2a" << endl;
//	tpchJoin2a(datasetCatalog,3);
//	cout << "---" << endl;
//	cout << "Query 2b" << endl;
//	tpchJoin2b(datasetCatalog,3);
//	cout << "---" << endl;
//
//	/* Make sure sides are materialized */
//	cout << "Query 0c (Side built if applicable)" << endl;
//	tpchJoin3(datasetCatalog, 3);
//	cout << "---" << endl;
//	cout << "Query 0d (Side built if applicable)" << endl;
//	tpchJoin4(datasetCatalog, 3);
//	cout << "---" << endl;
//
//	cout << "Query 3" << endl;
//	tpchJoin3(datasetCatalog, 3);
//	cout << "---" << endl;
//	cout << "Query 4" << endl;
//	tpchJoin4(datasetCatalog, 3);
//	cout << "---" << endl;
//}

///* Same as in other plugins */
//int main()	{
//
//	map<string,dataset> datasetCatalog;
//	tpchSchema(datasetCatalog);
//
//	/* Make sure sides are materialized */
////	cout << "Query 0a (PM + Side built if applicable)" << endl;
////	tpchJoin1a(datasetCatalog,2);
////	tpchJoin1a(datasetCatalog,2);
////	cout << "---" << endl;
////	cout << "Query 0b (PM + Side built if applicable)" << endl;
////	tpchJoin1b(datasetCatalog,2);
////	tpchJoin1b(datasetCatalog,2);
////	cout << "---" << endl;
////
//	cout << "Query 1a" << endl;
//	tpchJoin1a(datasetCatalog,2);
//	cout << "---" << endl;
//	cout << "Query 1b" << endl;
//	tpchJoin1b(datasetCatalog,2);
//	cout << "---" << endl;
//	cout << "Query 2a" << endl;
//	tpchJoin2a(datasetCatalog,3);
//	cout << "---" << endl;
//	cout << "Query 2b" << endl;
//	tpchJoin2b(datasetCatalog,3);
//	cout << "---" << endl;
//
//	/* Make sure sides are materialized */
//	cout << "Query 0c (Side built if applicable)" << endl;
//	tpchJoin3(datasetCatalog, 3);
//	cout << "---" << endl;
//	cout << "Query 0d (Side built if applicable)" << endl;
//	tpchJoin4(datasetCatalog, 3);
//	cout << "---" << endl;
//
////	cout << "Query 3" << endl;
////	tpchJoin3(datasetCatalog, 3);
////	cout << "---" << endl;
////	cout << "Query 4" << endl;
////	tpchJoin4(datasetCatalog, 3);
////	cout << "---" << endl;
//}

///* Testing (very) wide json files with nesting
// * ordersLineitemArray.json */
//int main()	{
//
//	map<string,dataset> datasetCatalog;
//	tpchSchema(datasetCatalog);
//
//	/* Returned 20760489 in 30 sec.*/
//	tpchJoin1c(datasetCatalog,20000000);
//	tpchJoin1c(datasetCatalog,20000000);
//
////	tpchJoin1a(datasetCatalog,3);
////	tpchJoin1b(datasetCatalog,3);
//
//	/* Returns 1 more result! */
//	//Correct: 25
//	//tpchJoin1c(datasetCatalog,10);
//	//Correct: 55
//	//tpchJoin1c(datasetCatalog,40);
//
//	tpchJoin1c(datasetCatalog,30000000);
//	tpchJoin1c(datasetCatalog,30000000);
//
//}

int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchema(datasetCatalog);

	int runs = 5;
	int selectivityShifts = 10;
	int predicateMax = O_ORDERKEY_MAX;

	cout << "[tpch-json-joins: ] Warmup (PM)" << endl;
	cout << "-> tpchJoinWarmupPM1" << endl;
	tpchJoin1a(datasetCatalog, predicateMax);
//	cout << "-> tpchJoinWarmupPM2" << endl;
//	tpchJoin1c(datasetCatalog, predicateMax);
	cout << "[tpch-json-joins: ] End of Warmup (PM)" << endl;

	CachingService& cache = CachingService::getInstance();
	RawCatalog& rawCatalog = RawCatalog::getInstance();
	rawCatalog.clear();
	cache.clear();

	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-joins: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1a)" << endl;
			tpchJoin1a(datasetCatalog, predicateVal);
			rawCatalog.clear();
			cache.clear();

			cout << "1b)" << endl;
			tpchJoin1b(datasetCatalog, predicateVal);
			rawCatalog.clear();
			cache.clear();

//			cout << "1c)" << endl;
//			tpchJoin1c(datasetCatalog, predicateVal);
//			rawCatalog.clear();
//			cache.clear();

			cout << "2a)" << endl;
			tpchJoin2a(datasetCatalog, predicateVal);
			rawCatalog.clear();
			cache.clear();

			cout << "2b)" << endl;
			tpchJoin2b(datasetCatalog, predicateVal);
			rawCatalog.clear();
			cache.clear();

			cout << "3)" << endl;
			tpchJoin3(datasetCatalog, predicateMax);
			rawCatalog.clear();
			cache.clear();

			cout << "4)" << endl;
			tpchJoin4(datasetCatalog, predicateMax);
			rawCatalog.clear();
			cache.clear();
		}
	}
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
	//cout << "3. MADE IT HERE" << endl;

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
