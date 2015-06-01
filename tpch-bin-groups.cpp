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
	tpchSchemaBin(datasetCatalog);
}

/* Numbers of lineitems per order
   SELECT COUNT(*), [...]
   FROM lineitem
   WHERE l_orderkey < [X]
   GROUP BY l_linenumber
 */
void tpchGroup(map<string,dataset> datasetCatalog, int predicate, int aggregatesNo);

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


//int main()	{
//
//	map<string,dataset> datasetCatalog;
//	tpchSchema(datasetCatalog);
//
//	int predicate = 3;
//	cout << "Query 0 (PM + Side built if applicable)" << endl;
//	tpchGroup(datasetCatalog, predicate, 4);
//	cout << "---" << endl;
//	cout << "Query 1 (aggr.)" << endl;
//	tpchGroup(datasetCatalog, predicate, 1);
//	cout << "---" << endl;
//	cout << "Query 2 (aggr.)" << endl;
//	tpchGroup(datasetCatalog, predicate, 2);
//	cout << "---" << endl;
//	cout << "Query 3 (aggr.)" << endl;
//	tpchGroup(datasetCatalog, predicate, 3);
//	cout << "---" << endl;
//	cout << "Query 4 (aggr.)" << endl;
//	tpchGroup(datasetCatalog, predicate, 4);
//	cout << "---" << endl;
//}

int main(int argc, char **argv) {

	map<string, dataset> datasetCatalog;
	tpchSchema(datasetCatalog);

	int runs = 5;
	int selectivityShifts = 10;
	int predicateMax = O_ORDERKEY_MAX;

	if (argc != 2) {
		return -1;
	}
	int mode = atoi(argv[1]);

	switch (mode) {
	case 1: {
		CachingService& cache = CachingService::getInstance();
		RawCatalog& rawCatalog = RawCatalog::getInstance();
		cout << "Query 0 (OS caches Warmup - no PM applicable)" << endl;
		tpchGroup(datasetCatalog, predicateMax, 4);
		rawCatalog.clear();
		cache.clear();
		for (int i = 0; i < runs; i++) {
			cout << "[tpch-bin-groups: ] Run " << i + 1 << endl;
			for (int i = 1; i <= selectivityShifts; i++) {
				double ratio = (i / (double) 10);
				double percentage = ratio * 100;

				int predicateVal = (int) ceil(predicateMax * ratio);
				cout << "SELECTIVITY FOR key < " << predicateVal << ": "
						<< percentage << "%" << endl;

				cout << "1)" << endl;
				tpchGroup(datasetCatalog, predicateVal, 1);
				rawCatalog.clear();
				cache.clear();
			}
		}
		break;
	}
	case 2: {
		CachingService& cache = CachingService::getInstance();
		RawCatalog& rawCatalog = RawCatalog::getInstance();
		cout << "Query 0 (OS caches Warmup - no PM applicable)" << endl;
		tpchGroup(datasetCatalog, predicateMax, 4);
		rawCatalog.clear();
		cache.clear();
		for (int i = 0; i < runs; i++) {
			cout << "[tpch-bin-groups: ] Run " << i + 1 << endl;
			for (int i = 1; i <= selectivityShifts; i++) {
				double ratio = (i / (double) 10);
				double percentage = ratio * 100;

				int predicateVal = (int) ceil(predicateMax * ratio);
				cout << "SELECTIVITY FOR key < " << predicateVal << ": "
						<< percentage << "%" << endl;

				cout << "2)" << endl;
				tpchGroup(datasetCatalog, predicateVal, 2);
				rawCatalog.clear();
				cache.clear();

			}
		}
		break;
	}
	case 3: {
		CachingService& cache = CachingService::getInstance();
		RawCatalog& rawCatalog = RawCatalog::getInstance();
		cout << "Query 0 (OS caches Warmup - no PM applicable)" << endl;
		tpchGroup(datasetCatalog, predicateMax, 4);
		rawCatalog.clear();
		cache.clear();
		for (int i = 0; i < runs; i++) {
			cout << "[tpch-bin-groups: ] Run " << i + 1 << endl;
			for (int i = 1; i <= selectivityShifts; i++) {
				double ratio = (i / (double) 10);
				double percentage = ratio * 100;

				int predicateVal = (int) ceil(predicateMax * ratio);
				cout << "SELECTIVITY FOR key < " << predicateVal << ": "
						<< percentage << "%" << endl;

				cout << "3)" << endl;
				tpchGroup(datasetCatalog, predicateVal, 3);
				rawCatalog.clear();
				cache.clear();

			}
		}
		break;
	}
	case 4: {
		CachingService& cache = CachingService::getInstance();
		RawCatalog& rawCatalog = RawCatalog::getInstance();
		cout << "Query 0 (OS caches Warmup - no PM applicable)" << endl;
		tpchGroup(datasetCatalog, predicateMax, 4);
		rawCatalog.clear();
		cache.clear();
		for (int i = 0; i < runs; i++) {
			cout << "[tpch-bin-groups: ] Run " << i + 1 << endl;
			for (int i = 1; i <= selectivityShifts; i++) {
				double ratio = (i / (double) 10);
				double percentage = ratio * 100;

				int predicateVal = (int) ceil(predicateMax * ratio);
				cout << "SELECTIVITY FOR key < " << predicateVal << ": "
						<< percentage << "%" << endl;

				cout << "4)" << endl;
				tpchGroup(datasetCatalog, predicateVal, 4);
				rawCatalog.clear();
				cache.clear();
			}
		}
		break;
	}
	default: {
		cout << "Please use mode 1-4 next time" << endl;
		return -1;
	}
	}

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
	string fnamePrefix = lineitem.path;
	RecordType rec = lineitem.recType;
	int policy = 5;
	int lineHint = lineitem.linehint;
	char delimInner = '|';
	vector<RecordAttribute*> projections;
	RecordAttribute *l_orderkey = argsLineitem["l_orderkey"];//NULL;
	RecordAttribute *l_linenumber = argsLineitem["l_linenumber"];
	RecordAttribute *l_quantity = NULL;
	RecordAttribute *l_extendedprice = NULL;
	RecordAttribute *l_tax = NULL;

	projections.push_back(l_orderkey);
	projections.push_back(l_linenumber);
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

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
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
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Select *sel = new Select(pred, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: l_linenumber
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
//	nestProjections.push_back(*l_orderkey);
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
			l_linenumber->getOriginalType(), nestArg, *l_linenumber);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(l_linenumber);
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
