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
#include "operators/materializer-expr.hpp"
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

/*
 * Using UNNEST instead of JOIN:
   SELECT COUNT(lineitem)
   FROM ordersLineitems
   WHERE o_orderkey < [X]
 */
void tpchUnnest(map<string,dataset> datasetCatalog, int predicate, bool exploitSchema);

void tpchUnnestCachingPred(map<string,dataset> datasetCatalog, int predicate, bool exploitSchema);

/*
   SELECT MAX(o_orderkey)
   FROM orders
   INNER JOIN lineitem ON (o_orderkey = l_orderkey)
   AND l_orderkey < [X]
 */

/* XXX Must define JSON_TPCH_WIDE before executing!!! */
int main()	{

	CachingService& cache = CachingService::getInstance();
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	map<string,dataset> datasetCatalog;
	tpchSchema(datasetCatalog);

	int runs = 5;
	int selectivityShifts = 10;
	int predicateMax = O_ORDERKEY_MAX;

	cout << "[tpch-json-unnests: ] Warmup (PM)" << endl;
	cout << "-> tpchUnnestWarmup" << endl;
	tpchUnnest(datasetCatalog, predicateMax, false);
	cout << "[tpch-json-unnests: ] End of Warmup (PM)" << endl;

	rawCatalog.clear();
	cache.clear();

	/* Baseline */
	bool exploitSchema = false;
	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-unnests-noschema: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1c)" << endl;
			tpchUnnest(datasetCatalog, predicateVal, exploitSchema);
			rawCatalog.clear();
			cache.clear();
		}
	}

	cout << "[tpch-json-unnests: ] Caching predicate" << endl;
	tpchUnnestCachingPred(datasetCatalog, predicateMax, exploitSchema);
	cout << "[tpch-json-unnests: ] End of Caching predicate" << endl;

	/* Baseline + Caching */
	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-unnests-noschema-cached: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1c)" << endl;
			tpchUnnest(datasetCatalog, predicateVal, exploitSchema);
		}
	}

	rawCatalog.clear();
	cache.clear();

	/* Exploiting schema during codegen. */
	exploitSchema = true;
	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-unnests-schema: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1c)" << endl;
			tpchUnnest(datasetCatalog, predicateVal, exploitSchema);
			rawCatalog.clear();
			cache.clear();

		}
	}

	/* Schema + Caching */
	cout << "[tpch-json-unnests: ] Caching predicate while exploiting schema" << endl;
	tpchUnnestCachingPred(datasetCatalog, predicateMax, exploitSchema);
	cout << "[tpch-json-unnests: ] End of Caching predicate while exploiting schema" << endl;

	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-unnests-schema-cached: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1c)" << endl;
			tpchUnnest(datasetCatalog, predicateVal, exploitSchema);
		}
	}
}

void tpchUnnestCachingPred(map<string, dataset> datasetCatalog, int predicate, bool exploitSchema) {

	RawContext& ctx = *prepareContext("tpch-json/unnest-cachingPred");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameOrdersLineitems = string("ordersLineitems");
	dataset ordersLineitems = datasetCatalog[nameOrdersLineitems];
	map<string, RecordAttribute*> argsOrdersLineitems =
			ordersLineitems.recType.getArgsMap();

	RecordAttribute *lineitems = argsOrdersLineitems["lineitems"];
	RecordAttribute *o_orderkey = argsOrdersLineitems["orderkey"];

	/**
	 * SCAN : ordersLineitems
	 */
	string fileName = ordersLineitems.path;
	RecordType rec = ordersLineitems.recType;
	int linehint = ordersLineitems.linehint;


	ListType *ordersDocType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx,
			fileName, ordersDocType, linehint,exploitSchema);

	rawCatalog.registerPlugin(fileName, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/* Preparing for unnest */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*o_orderkey);
	unnestProjections.push_back(*lineitems);

	expressions::Expression* outerArg = new expressions::InputArgument(&rec, 0,
			unnestProjections);

	/*
	 * Materialize expression(s) here
	 * l_orderkey the one cached
	 */
	expressions::Expression* toMat = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), outerArg, *o_orderkey);

	char matLabel[] = "orderkeyMaterializer";
	ExprMaterializer *mat = new ExprMaterializer(toMat, linehint, scan, &ctx,
			matLabel);
	scan->setParent(mat);

	/**
	 * UNNEST:
	 */

	expressions::RecordProjection* proj = new expressions::RecordProjection(
			lineitems->getOriginalType(), outerArg, *lineitems);
	string nestedName = "items";
	Path path = Path(nestedName, proj);

	/* Filtering on orderkey */
	expressions::Expression *lhs = new expressions::RecordProjection(
			o_orderkey->getOriginalType(), outerArg, *o_orderkey);
	expressions::Expression* rhs = new expressions::IntConstant(predicate);
	expressions::Expression* pred = new expressions::LtExpression(
			new BoolType(), lhs, rhs);

	Unnest *unnestOp = new Unnest(pred, path, mat);
	mat->setParent(unnestOp);

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

void tpchUnnest(map<string, dataset> datasetCatalog, int predicate, bool exploitSchema) {

	RawContext& ctx = *prepareContext("tpch-json/unnest");
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
			fileName, ordersDocType, linehint,exploitSchema);

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
