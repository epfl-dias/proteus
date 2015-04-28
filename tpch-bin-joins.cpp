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
   AND l_orderkey < [X]
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
	tpchSchema(datasetCatalog);

	tpchJoin1a(datasetCatalog,2);
	tpchJoin1a(datasetCatalog,2);
	tpchJoin2a(datasetCatalog,3);
	tpchJoin2b(datasetCatalog,3);

	tpchJoin3(datasetCatalog,3);
	tpchJoin3(datasetCatalog,3);
	tpchJoin4(datasetCatalog,3);
	tpchJoin4(datasetCatalog,3);

}

void tpchJoin1a(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-csv-join1a");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	orderProjections.push_back(o_orderkey);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	lineitemProjections.push_back(l_orderkey);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
			expressions::Expression* rhs = new expressions::IntConstant(
					predicate);
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
	fieldsLeft.push_back(o_orderkey);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
				o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprOrderkey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
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

	RawContext ctx = prepareContext("tpch-csv-join1b");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	orderProjections.push_back(o_orderkey);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	lineitemProjections.push_back(l_orderkey);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
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
	fieldsRight.push_back(l_orderkey);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);
	expressionsRight.push_back(exprOrderkey);

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

void tpchJoin2a(map<string, dataset> datasetCatalog, int predicate) {

	RawContext ctx = prepareContext("tpch-csv-join2a");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	orderProjections.push_back(o_orderkey);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	lineitemProjections.push_back(l_orderkey);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
	fieldsLeft.push_back(o_orderkey);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
				o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprOrderkey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
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

	RawContext ctx = prepareContext("tpch-csv-join2b");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	orderProjections.push_back(o_orderkey);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	lineitemProjections.push_back(l_orderkey);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
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
	fieldsRight.push_back(l_orderkey);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);

	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);
	expressionsRight.push_back(exprOrderkey);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoin";
	RadixJoin *join = new RadixJoin(joinPred, scanOrders, sel, &ctx,
			joinLabel, *matLeft, *matRight);

	sel->setParent(join);
	scanOrders->setParent(join);

	/**
	 * REDUCE
	 * MAX(orderkey)
	 */
	/* Output: */
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

	RawContext ctx = prepareContext("tpch-csv-join3");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	RecordAttribute *o_totalprice = argsOrder["o_totalprice"];
	orderProjections.push_back(o_orderkey);
	orderProjections.push_back(o_totalprice);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	lineitemProjections.push_back(l_orderkey);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
	fieldsLeft.push_back(o_orderkey);
	fieldsLeft.push_back(o_totalprice);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
			pgOrders->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgOrders->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
				o_orderkey->getOriginalType(), leftArg, *o_orderkey);
	expressions::Expression* exprTotalprice = new expressions::RecordProjection(
					o_totalprice->getOriginalType(), leftArg, *o_totalprice);
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprOrderkey);
	expressionsLeft.push_back(exprTotalprice);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
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

	RawContext ctx = prepareContext("tpch-csv-join4");
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
	string ordersNamePrefix = orders.path;
	RecordType recOrders = orders.recType;
	int policy = 5;
	int lineHint = orders.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> orderProjections;
	RecordAttribute *o_orderkey = argsOrder["o_orderkey"];
	orderProjections.push_back(o_orderkey);

	BinaryColPlugin *pgOrders = new BinaryColPlugin(&ctx, ordersNamePrefix,
			recOrders, orderProjections);
	rawCatalog.registerPlugin(ordersNamePrefix, pgOrders);
	Scan *scanOrders = new Scan(&ctx, *pgOrders);

	/**
	 * SCAN 2: Lineitem
	 */
	string lineitemNamePrefix = lineitem.path;
	RecordType recLineitem = lineitem.recType;
	policy = 5;
	lineHint = lineitem.linehint;
	delimInner = '|';
	vector<RecordAttribute*> lineitemProjections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];
	RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];
	lineitemProjections.push_back(l_orderkey);
	lineitemProjections.push_back(l_extendedprice);

	BinaryColPlugin *pgLineitem = new BinaryColPlugin(&ctx, lineitemNamePrefix,
			recLineitem, lineitemProjections);
	rawCatalog.registerPlugin(lineitemNamePrefix, pgLineitem);
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
	RecordAttribute *projTupleL = new RecordAttribute(ordersNamePrefix, activeLoop,
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
	fieldsRight.push_back(l_orderkey);
	fieldsRight.push_back(l_extendedprice);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(lineitemNamePrefix, activeLoop,
			pgLineitem->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgLineitem->getOIDType(), rightArg, *projTupleR);
	expressions::Expression* exprOrderkey = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), rightArg, *l_orderkey);
	expressions::Expression* exprExtendedprice = new expressions::RecordProjection(
				l_extendedprice->getOriginalType(), rightArg, *l_extendedprice);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);
	expressionsRight.push_back(exprOrderkey);
	expressionsRight.push_back(exprExtendedprice);

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

void tpchSchema(map<string, dataset>& datasetCatalog) {
	IntType *intType = new IntType();
	FloatType *floatType = new FloatType();
	StringType *stringType = new StringType();

	/* Lineitem */
	string lineitemPath = string("inputs/tpch/col10/lineitem");

	list<RecordAttribute*> attsLineitem = list<RecordAttribute*>();
	RecordAttribute *l_orderkey = new RecordAttribute(1, lineitemPath,
			"l_orderkey", intType);
	attsLineitem.push_back(l_orderkey);
	RecordAttribute *l_partkey = new RecordAttribute(2, lineitemPath,
			"l_partkey", intType);
	attsLineitem.push_back(l_partkey);
	RecordAttribute *l_suppkey = new RecordAttribute(3, lineitemPath,
			"l_suppkey", intType);
	attsLineitem.push_back(l_suppkey);
	RecordAttribute *l_linenumber = new RecordAttribute(4, lineitemPath,
			"l_linenumber", intType);
	attsLineitem.push_back(l_linenumber);
	RecordAttribute *l_quantity = new RecordAttribute(5, lineitemPath,
			"l_quantity", floatType);
	attsLineitem.push_back(l_quantity);
	RecordAttribute *l_extendedprice = new RecordAttribute(6, lineitemPath,
			"l_extendedprice", floatType);
	attsLineitem.push_back(l_extendedprice);
	RecordAttribute *l_discount = new RecordAttribute(7, lineitemPath,
			"l_discount", floatType);
	attsLineitem.push_back(l_discount);
	RecordAttribute *l_tax = new RecordAttribute(8, lineitemPath, "l_tax",
			floatType);
	attsLineitem.push_back(l_tax);
	RecordAttribute *l_returnflag = new RecordAttribute(9, lineitemPath,
			"l_returnflag", stringType);
	attsLineitem.push_back(l_returnflag);
	RecordAttribute *l_linestatus = new RecordAttribute(10, lineitemPath,
			"l_linestatus", stringType);
	attsLineitem.push_back(l_linestatus);
	RecordAttribute *l_shipdate = new RecordAttribute(11, lineitemPath,
			"l_shipdate", stringType);
	attsLineitem.push_back(l_shipdate);
	RecordAttribute *l_commitdate = new RecordAttribute(12, lineitemPath,
			"l_commitdate", stringType);
	attsLineitem.push_back(l_commitdate);
	RecordAttribute *l_receiptdate = new RecordAttribute(13, lineitemPath,
			"l_receiptdate", stringType);
	attsLineitem.push_back(l_receiptdate);
	RecordAttribute *l_shipinstruct = new RecordAttribute(14, lineitemPath,
			"l_shipinstruct", stringType);
	attsLineitem.push_back(l_shipinstruct);
	RecordAttribute *l_shipmode = new RecordAttribute(15, lineitemPath,
			"l_shipmode", stringType);
	attsLineitem.push_back(l_shipmode);
	RecordAttribute *l_comment = new RecordAttribute(16, lineitemPath,
			"l_comment", stringType);
	attsLineitem.push_back(l_comment);

	RecordType lineitemRec = RecordType(attsLineitem);

	/* Orders */
	string ordersPath = string("inputs/tpch/col10/orders");

	list<RecordAttribute*> attsOrder = list<RecordAttribute*>();
	RecordAttribute *o_orderkey = new RecordAttribute(1, ordersPath,
			"o_orderkey", intType);
	attsOrder.push_back(o_orderkey);
	RecordAttribute *o_custkey = new RecordAttribute(2, ordersPath, "o_custkey",
			intType);
	attsOrder.push_back(o_custkey);
	RecordAttribute *o_orderstatus = new RecordAttribute(3, ordersPath,
			"o_orderstatus", stringType);
	attsOrder.push_back(o_orderstatus);
	RecordAttribute *o_totalprice = new RecordAttribute(4, ordersPath,
			"o_totalprice", floatType);
	attsOrder.push_back(o_totalprice);
	RecordAttribute *o_orderdate = new RecordAttribute(5, ordersPath,
			"o_orderdate", stringType);
	attsOrder.push_back(o_orderdate);
	RecordAttribute *o_orderpriority = new RecordAttribute(6, ordersPath,
			"o_orderpriority", stringType);
	attsOrder.push_back(o_orderpriority);
	RecordAttribute *o_clerk = new RecordAttribute(7, ordersPath, "o_clerk",
			stringType);
	attsOrder.push_back(o_clerk);
	RecordAttribute *o_shippriority = new RecordAttribute(8, ordersPath,
			"o_shippriority", intType);
	attsOrder.push_back(o_shippriority);
	RecordAttribute *o_comment = new RecordAttribute(9, ordersPath, "o_comment",
			stringType);
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
	datasetCatalog["orders"] = orders;
}
