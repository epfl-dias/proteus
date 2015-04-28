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
 * Selections
 * All should have the same result - what changes is the order of the predicates
 * and whether some are applied via a conjunction
 */
//1-2-3-4
void tpchOrderSelection1(map<string,dataset> datasetCatalog, vector<int> predicates);
/* Only use the rest in the case of 4 predicates! */
//4-3-2-1
void tpchOrderSelection2(map<string,dataset> datasetCatalog, vector<int> predicates);
//(1-2),(3-4)
void tpchOrderSelection3(map<string,dataset> datasetCatalog, vector<int> predicates);
void tpchOrderSelection4(map<string,dataset> datasetCatalog, vector<int> predicates);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{

	map<string,dataset> datasetCatalog;
	tpchSchema(datasetCatalog);

	vector<int> predicates;
	predicates.push_back(2);
	cout << "Query 0 (PM built if applicable)" << endl;
	tpchOrderSelection1(datasetCatalog, predicates);
	cout << "---" << endl;
	//1 pred.
	cout << "Query 1a" << endl;
	tpchOrderSelection1(datasetCatalog, predicates);
	cout << "---" << endl;
	cout << "Query 1b" << endl;
	predicates.push_back(0);
	//2 pred.
	tpchOrderSelection1(datasetCatalog, predicates);
	cout << "---" << endl;
	cout << "Query 1c" << endl;
	predicates.push_back(0);
	//3 pred.
	tpchOrderSelection1(datasetCatalog, predicates);
	cout << "---" << endl;
	cout << "Query 1d" << endl;
	//4 pred.
	predicates.push_back(0);
	tpchOrderSelection1(datasetCatalog, predicates);
	//Variations of last execution
	cout << "---" << endl;
	cout << "Query 2" << endl;
	tpchOrderSelection2(datasetCatalog, predicates);
	cout << "---" << endl;
	cout << "Query 3" << endl;
	tpchOrderSelection3(datasetCatalog, predicates);
	cout << "---" << endl;
	cout << "Query 4" << endl;
	tpchOrderSelection4(datasetCatalog, predicates);
	cout << "---" << endl;
}

void tpchOrderSelection1(map<string,dataset> datasetCatalog, vector<int> predicates)	{

	int predicatesNo = predicates.size();
	if(predicatesNo <= 0 || predicatesNo > 4)	{
		throw runtime_error(string("Invalid no. of predicates requested: "));
	}
	RawContext ctx = prepareContext("tpch-csv-selection1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();

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

	if (predicatesNo == 1 || predicatesNo == 2) {
		projections.push_back(l_orderkey);
		projections.push_back(l_quantity);
	} else if (predicatesNo == 3) {
		projections.push_back(l_orderkey);
		projections.push_back(l_linenumber);
		projections.push_back(l_quantity);
	} else if (predicatesNo == 4) {
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
	 * SELECT(S)
	 * 1 to 4
	 *
	 * Lots of repetition..
	 */
	RawOperator *lastSelectOp;
	list<RecordAttribute> argProjections;
	if (predicatesNo == 1) {
		argProjections.push_back(*l_orderkey);
		expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
				argProjections);

		expressions::Expression *lhs1 = new expressions::RecordProjection(
				l_orderkey->getOriginalType(), arg, *l_orderkey);
		expressions::Expression* rhs1 = new expressions::IntConstant(
				predicates.at(0));
		expressions::Expression* pred1 = new expressions::GtExpression(
				new BoolType(), lhs1, rhs1);

		Select *sel1 = new Select(pred1, scan);
		scan->setParent(sel1);
		lastSelectOp = sel1;
	} else if (predicatesNo == 2) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_quantity);
		expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
				argProjections);

		expressions::Expression *lhs1 = new expressions::RecordProjection(
				l_orderkey->getOriginalType(), arg, *l_orderkey);
		expressions::Expression* rhs1 = new expressions::IntConstant(
				predicates.at(0));
		expressions::Expression* pred1 = new expressions::GtExpression(
				new BoolType(), lhs1, rhs1);

		Select *sel1 = new Select(pred1, scan);
		scan->setParent(sel1);

		expressions::Expression *lhs2 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		expressions::Expression* rhs2 = new expressions::FloatConstant(
				predicates.at(1));
		expressions::Expression* pred2 = new expressions::GtExpression(
				new BoolType(), lhs2, rhs2);

		Select *sel2 = new Select(pred2, sel1);
		sel1->setParent(sel2);

		lastSelectOp = sel2;
	} else if (predicatesNo == 3) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_linenumber);
		argProjections.push_back(*l_quantity);
		expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
				argProjections);

		expressions::Expression *lhs1 = new expressions::RecordProjection(
				l_orderkey->getOriginalType(), arg, *l_orderkey);
		expressions::Expression* rhs1 = new expressions::IntConstant(
				predicates.at(0));
		expressions::Expression* pred1 = new expressions::GtExpression(
				new BoolType(), lhs1, rhs1);

		Select *sel1 = new Select(pred1, scan);
		scan->setParent(sel1);

		expressions::Expression *lhs2 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		expressions::Expression* rhs2 = new expressions::FloatConstant(
				predicates.at(1));
		expressions::Expression* pred2 = new expressions::GtExpression(
				new BoolType(), lhs2, rhs2);

		Select *sel2 = new Select(pred2, sel1);
		sel1->setParent(sel2);

		expressions::Expression *lhs3 = new expressions::RecordProjection(
				l_linenumber->getOriginalType(), arg, *l_linenumber);
		expressions::Expression* rhs3 = new expressions::IntConstant(
				predicates.at(2));
		expressions::Expression* pred3 = new expressions::GtExpression(
				new BoolType(), lhs3, rhs3);

		Select *sel3 = new Select(pred3, sel2);
		sel2->setParent(sel3);
		lastSelectOp = sel3;

	} else if (predicatesNo == 4) {
		argProjections.push_back(*l_orderkey);
		argProjections.push_back(*l_linenumber);
		argProjections.push_back(*l_quantity);
		argProjections.push_back(*l_extendedprice);
		expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
				argProjections);

		expressions::Expression *lhs1 = new expressions::RecordProjection(
				l_orderkey->getOriginalType(), arg, *l_orderkey);
		expressions::Expression* rhs1 = new expressions::IntConstant(
				predicates.at(0));
		expressions::Expression* pred1 = new expressions::GtExpression(
				new BoolType(), lhs1, rhs1);

		Select *sel1 = new Select(pred1, scan);
		scan->setParent(sel1);

		expressions::Expression *lhs2 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		expressions::Expression* rhs2 = new expressions::FloatConstant(
				predicates.at(1));
		expressions::Expression* pred2 = new expressions::GtExpression(
				new BoolType(), lhs2, rhs2);

		Select *sel2 = new Select(pred2, sel1);
		sel1->setParent(sel2);

		expressions::Expression *lhs3 = new expressions::RecordProjection(
				l_linenumber->getOriginalType(), arg, *l_linenumber);
		expressions::Expression* rhs3 = new expressions::IntConstant(
				predicates.at(2));
		expressions::Expression* pred3 = new expressions::GtExpression(
				new BoolType(), lhs3, rhs3);

		Select *sel3 = new Select(pred3, sel2);
		sel2->setParent(sel3);

		expressions::Expression *lhs4 = new expressions::RecordProjection(
				l_extendedprice->getOriginalType(), arg, *l_extendedprice);
		expressions::Expression* rhs4 = new expressions::FloatConstant(
				predicates.at(3));
		expressions::Expression* pred4 = new expressions::GtExpression(
				new BoolType(), lhs4, rhs4);

		Select *sel4 = new Select(pred4, sel3);
		sel3->setParent(sel4);

		lastSelectOp = sel4;
	}

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
	lastSelectOp->setParent(reduce);

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

void tpchOrderSelection2(map<string,dataset> datasetCatalog, vector<int> predicates)	{

	int predicatesNo = predicates.size();
	if(predicatesNo != 4)	{
		throw runtime_error(string("Invalid no. of predicates requested: "));
	}
	RawContext ctx = prepareContext("tpch-csv-selection2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();

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

	projections.push_back(l_orderkey);
	projections.push_back(l_linenumber);
	projections.push_back(l_quantity);
	projections.push_back(l_extendedprice);

	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT(S)
	 */
	RawOperator *lastSelectOp;
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	argProjections.push_back(*l_linenumber);
	argProjections.push_back(*l_quantity);
	argProjections.push_back(*l_extendedprice);
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	/* Predicates */
	expressions::Expression *lhs1 = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	expressions::Expression* rhs1 = new expressions::IntConstant(
			predicates.at(0));
	expressions::Expression* pred1 = new expressions::GtExpression(
			new BoolType(), lhs1, rhs1);

	expressions::Expression *lhs2 = new expressions::RecordProjection(
			l_quantity->getOriginalType(), arg, *l_quantity);
	expressions::Expression* rhs2 = new expressions::FloatConstant(
			predicates.at(1));
	expressions::Expression* pred2 = new expressions::GtExpression(
			new BoolType(), lhs2, rhs2);

	expressions::Expression *lhs3 = new expressions::RecordProjection(
			l_linenumber->getOriginalType(), arg, *l_linenumber);
	expressions::Expression* rhs3 = new expressions::IntConstant(
			predicates.at(2));
	expressions::Expression* pred3 = new expressions::GtExpression(
			new BoolType(), lhs3, rhs3);

	expressions::Expression *lhs4 = new expressions::RecordProjection(
			l_extendedprice->getOriginalType(), arg, *l_extendedprice);
	expressions::Expression* rhs4 = new expressions::FloatConstant(
			predicates.at(3));
	expressions::Expression* pred4 = new expressions::GtExpression(
			new BoolType(), lhs4, rhs4);

	/* Notice that we apply predicates in reverse order */
	Select *sel1 = new Select(pred4, scan);
	scan->setParent(sel1);

	Select *sel2 = new Select(pred3, sel1);
	sel1->setParent(sel2);

	Select *sel3 = new Select(pred2, sel2);
	sel2->setParent(sel3);

	Select *sel4 = new Select(pred1, sel3);
	sel3->setParent(sel4);

	lastSelectOp = sel4;

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
	lastSelectOp->setParent(reduce);

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

void tpchOrderSelection3(map<string,dataset> datasetCatalog, vector<int> predicates)	{

	int predicatesNo = predicates.size();
	if(predicatesNo != 4)	{
		throw runtime_error(string("Invalid no. of predicates requested: "));
	}
	RawContext ctx = prepareContext("tpch-csv-selection3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();

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

	projections.push_back(l_orderkey);
	projections.push_back(l_linenumber);
	projections.push_back(l_quantity);
	projections.push_back(l_extendedprice);

	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT(S)
	 */
	RawOperator *lastSelectOp;
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	argProjections.push_back(*l_linenumber);
	argProjections.push_back(*l_quantity);
	argProjections.push_back(*l_extendedprice);
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	/* Predicates */
	expressions::Expression *lhs1 = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	expressions::Expression* rhs1 = new expressions::IntConstant(
			predicates.at(0));
	expressions::Expression* pred1 = new expressions::GtExpression(
			new BoolType(), lhs1, rhs1);

	expressions::Expression *lhs2 = new expressions::RecordProjection(
			l_quantity->getOriginalType(), arg, *l_quantity);
	expressions::Expression* rhs2 = new expressions::FloatConstant(
			predicates.at(1));
	expressions::Expression* pred2 = new expressions::GtExpression(
			new BoolType(), lhs2, rhs2);

	expressions::Expression *lhs3 = new expressions::RecordProjection(
			l_linenumber->getOriginalType(), arg, *l_linenumber);
	expressions::Expression* rhs3 = new expressions::IntConstant(
			predicates.at(2));
	expressions::Expression* pred3 = new expressions::GtExpression(
			new BoolType(), lhs3, rhs3);

	expressions::Expression *lhs4 = new expressions::RecordProjection(
			l_extendedprice->getOriginalType(), arg, *l_extendedprice);
	expressions::Expression* rhs4 = new expressions::FloatConstant(
			predicates.at(3));
	expressions::Expression* pred4 = new expressions::GtExpression(
			new BoolType(), lhs4, rhs4);

	/* Two (2) composite predicates */
	expressions::Expression* predA = new expressions::AndExpression(
			new BoolType(), pred1, pred2);
	expressions::Expression* predB = new expressions::AndExpression(
			new BoolType(), pred3, pred4);

	Select *sel1 = new Select(predA, scan);
	scan->setParent(sel1);

	Select *sel2 = new Select(predB, sel1);
	sel1->setParent(sel2);

	lastSelectOp = sel2;

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
	lastSelectOp->setParent(reduce);

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

void tpchOrderSelection4(map<string,dataset> datasetCatalog, vector<int> predicates)	{

	int predicatesNo = predicates.size();
	if(predicatesNo != 4)	{
		throw runtime_error(string("Invalid no. of predicates requested: "));
	}
	RawContext ctx = prepareContext("tpch-csv-selection4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	map<string, RecordAttribute*> argsLineitem 	=
			lineitem.recType.getArgsMap();

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

	projections.push_back(l_orderkey);
	projections.push_back(l_linenumber);
	projections.push_back(l_quantity);
	projections.push_back(l_extendedprice);

	pm::CSVPlugin* pg =
			new pm::CSVPlugin(&ctx, fname, rec, projections, delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT(S)
	 */
	RawOperator *lastSelectOp;
	list<RecordAttribute> argProjections;
	argProjections.push_back(*l_orderkey);
	argProjections.push_back(*l_linenumber);
	argProjections.push_back(*l_quantity);
	argProjections.push_back(*l_extendedprice);
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	/* Predicates */
	expressions::Expression *lhs1 = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	expressions::Expression* rhs1 = new expressions::IntConstant(
			predicates.at(0));
	expressions::Expression* pred1 = new expressions::GtExpression(
			new BoolType(), lhs1, rhs1);

	expressions::Expression *lhs2 = new expressions::RecordProjection(
			l_quantity->getOriginalType(), arg, *l_quantity);
	expressions::Expression* rhs2 = new expressions::FloatConstant(
			predicates.at(1));
	expressions::Expression* pred2 = new expressions::GtExpression(
			new BoolType(), lhs2, rhs2);

	expressions::Expression *lhs3 = new expressions::RecordProjection(
			l_linenumber->getOriginalType(), arg, *l_linenumber);
	expressions::Expression* rhs3 = new expressions::IntConstant(
			predicates.at(2));
	expressions::Expression* pred3 = new expressions::GtExpression(
			new BoolType(), lhs3, rhs3);

	expressions::Expression *lhs4 = new expressions::RecordProjection(
			l_extendedprice->getOriginalType(), arg, *l_extendedprice);
	expressions::Expression* rhs4 = new expressions::FloatConstant(
			predicates.at(3));
	expressions::Expression* pred4 = new expressions::GtExpression(
			new BoolType(), lhs4, rhs4);

	/* One (1) final composite predicate */
	expressions::Expression* predA = new expressions::AndExpression(
			new BoolType(), pred1, pred2);
	expressions::Expression* predB = new expressions::AndExpression(
			new BoolType(), pred3, pred4);
	expressions::Expression* pred = new expressions::AndExpression(
			new BoolType(), predA, predB);

	Select *sel1 = new Select(pred, scan);
	scan->setParent(sel1);

	lastSelectOp = sel1;

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastSelectOp, &ctx);
	lastSelectOp->setParent(reduce);

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
