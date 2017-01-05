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
#include "operators/materializer-expr.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/json-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "util/raw-caching.hpp"
#include "common/tpch-config.hpp"

void tpchSchema(map<string,dataset>& datasetCatalog)	{
	tpchSchemaJSON(datasetCatalog);
}

void tpchMatCSV(map<string,dataset> datasetCatalog, int predicate, int aggregatesNo);
void tpchMatJSON(map<string,dataset> datasetCatalog, int predicate, int aggregatesNo);

/* FIXME Need a case featuring Unnest too */


int main()	{

	int runs = 5;
	int selectivityShifts = 10;
//	int runs = 1;
//	int selectivityShifts = 1;
	int predicateMax = O_ORDERKEY_MAX;

	CachingService& cache = CachingService::getInstance();
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	/* CSV */
	map<string,dataset> datasetCatalogCSV;
	tpchSchemaCSV(datasetCatalogCSV);

	cout << "Query 0 (OS caches Warmup + PM)" << endl;
	tpchMatCSV(datasetCatalogCSV, predicateMax, 4);
	rawCatalog.clear();
	cache.clear();
	for (int i = 0; i < runs; i++) {
		cout << "[tpch-csv-mat: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1)" << endl;
			tpchMatCSV(datasetCatalogCSV, predicateVal, 1);
			rawCatalog.clear();
			cache.clear();

			cout << "2)" << endl;
			tpchMatCSV(datasetCatalogCSV, predicateVal, 2);
			rawCatalog.clear();
			cache.clear();

			cout << "3)" << endl;
			tpchMatCSV(datasetCatalogCSV, predicateVal, 3);
			rawCatalog.clear();
			cache.clear();

			cout << "4)" << endl;
			tpchMatCSV(datasetCatalogCSV, predicateVal, 4);
			rawCatalog.clear();
			cache.clear();
		}
	}


	map<string, dataset> datasetCatalogJSON;
	tpchSchemaJSON(datasetCatalogJSON);

	cout << "Query 0 (OS caches Warmup + PM)" << endl;
	tpchMatJSON(datasetCatalogJSON, predicateMax, 4);
	rawCatalog.clear();
	cache.clear();
	for (int i = 0; i < runs; i++) {
		cout << "[tpch-json-mat: ] Run " << i + 1 << endl;
		for (int i = 1; i <= selectivityShifts; i++) {
			double ratio = (i / (double) 10);
			double percentage = ratio * 100;

			int predicateVal = (int) ceil(predicateMax * ratio);
			cout << "SELECTIVITY FOR key < " << predicateVal << ": "
					<< percentage << "%" << endl;

			cout << "1)" << endl;
			tpchMatJSON(datasetCatalogJSON, predicateVal, 1);
			rawCatalog.clear();
			cache.clear();

			cout << "2)" << endl;
			tpchMatJSON(datasetCatalogJSON, predicateVal, 2);
			rawCatalog.clear();
			cache.clear();

			cout << "3)" << endl;
			tpchMatJSON(datasetCatalogJSON, predicateVal, 3);
			rawCatalog.clear();
			cache.clear();

			cout << "4)" << endl;
			tpchMatJSON(datasetCatalogJSON, predicateVal, 4);
			rawCatalog.clear();
			cache.clear();
		}
	}


}

void tpchMatCSV(map<string, dataset> datasetCatalog, int predicate, int aggregatesNo) {

	if (aggregatesNo <= 0 || aggregatesNo > 4) {
		throw runtime_error(string("Invalid aggregate no. requested: "));
	}
	RawContext& ctx = *prepareContext("tpch-csv-projection3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameLineitem = string("lineitem");
	dataset lineitem = datasetCatalog[nameLineitem];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem = lineitem.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder = orders.recType.getArgsMap();

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
	RecordAttribute *l_extendedprice = argsLineitem["l_extendedprice"];
	RecordAttribute *l_tax = argsLineitem["l_tax"];

	if (aggregatesNo == 1)
	{
		projections.push_back(l_orderkey);
	}
	else if (aggregatesNo == 2) {
		projections.push_back(l_orderkey);
		projections.push_back(l_quantity);
	} else if (aggregatesNo == 3) {
		projections.push_back(l_orderkey);
		projections.push_back(l_quantity);
		projections.push_back(l_extendedprice);
	} else if (aggregatesNo == 4) {
		projections.push_back(l_orderkey);
		projections.push_back(l_quantity);
		projections.push_back(l_extendedprice);
		projections.push_back(l_tax);
	}

	pm::CSVPlugin* pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
			delimInner, lineHint, policy);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

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

	/* Expr Materializers */

	expressions::Expression* toMat0 = new expressions::RecordProjection(
			l_orderkey->getOriginalType(), arg, *l_orderkey);
	char matLabel0[] = "orderkeyMaterializer";
	ExprMaterializer *mat0 = new ExprMaterializer(toMat0, lineHint, scan, &ctx,
			matLabel0);
	scan->setParent(mat0);
	ExprMaterializer *lastMat = mat0;
	if (aggregatesNo == 2) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);
		lastMat = mat1;
	}
	if (aggregatesNo == 3) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);

		expressions::Expression* toMat2 = new expressions::RecordProjection(
				l_extendedprice->getOriginalType(), arg, *l_extendedprice);
		char matLabel2[] = "extpriceMaterializer";
		ExprMaterializer *mat2 = new ExprMaterializer(toMat2, lineHint, mat1,
				&ctx, matLabel2);
		mat1->setParent(mat2);
		lastMat = mat2;

	}
	if (aggregatesNo == 4) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				l_quantity->getOriginalType(), arg, *l_quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);

		expressions::Expression* toMat2 = new expressions::RecordProjection(
				l_extendedprice->getOriginalType(), arg, *l_extendedprice);
		char matLabel2[] = "extpriceMaterializer";
		ExprMaterializer *mat2 = new ExprMaterializer(toMat2, lineHint, mat1,
				&ctx, matLabel2);
		mat1->setParent(mat2);

		expressions::Expression* toMat3 = new expressions::RecordProjection(
				l_tax->getOriginalType(), arg, *l_tax);
		char matLabel3[] = "taxMaterializer";
		ExprMaterializer *mat3 = new ExprMaterializer(toMat3, lineHint, mat2,
				&ctx, matLabel3);
		mat2->setParent(mat3);
		lastMat = mat3;
	}

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastMat, &ctx);
	lastMat->setParent(reduce);

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


void tpchMatJSON(map<string, dataset> datasetCatalog, int predicate, int aggregatesNo) {

	if(aggregatesNo < 1 || aggregatesNo > 4)	{
		throw runtime_error(string("Invalid no. of aggregates requested: "));
	}
	RawContext& ctx = *prepareContext("tpch-csv-selection1");
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

	RecordAttribute *orderkey = argsLineitem["orderkey"];
	RecordAttribute *linenumber = argsLineitem["linenumber"];
	RecordAttribute *quantity = NULL;
	RecordAttribute *extendedprice = NULL;
	RecordAttribute *tax = NULL;

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType, lineHint);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	list<RecordAttribute> argProjections;
	argProjections.push_back(*orderkey);
	if (aggregatesNo == 2) {
		quantity = argsLineitem["quantity"];
		argProjections.push_back(*quantity);
	}
	if (aggregatesNo == 3) {
		quantity = argsLineitem["quantity"];
		extendedprice = argsLineitem["extendedprice"];
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
	}
	if (aggregatesNo == 4) {
		quantity = argsLineitem["quantity"];
		extendedprice = argsLineitem["extendedprice"];
		tax = argsLineitem["tax"];
		argProjections.push_back(*quantity);
		argProjections.push_back(*extendedprice);
		argProjections.push_back(*tax);
	}
	expressions::Expression *arg = new expressions::InputArgument(&rec, 0,
			argProjections);

	/* Expr Materializers */

	expressions::Expression* toMat0 = new expressions::RecordProjection(
			orderkey->getOriginalType(), arg, *orderkey);
	char matLabel0[] = "orderkeyMaterializer";
	ExprMaterializer *mat0 = new ExprMaterializer(toMat0, lineHint, scan, &ctx,
			matLabel0);
	scan->setParent(mat0);
	ExprMaterializer *lastMat = mat0;
	if (aggregatesNo == 2) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				quantity->getOriginalType(), arg, *quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);
		lastMat = mat1;
	}
	if (aggregatesNo == 3) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				quantity->getOriginalType(), arg, *quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);

		expressions::Expression* toMat2 = new expressions::RecordProjection(
				extendedprice->getOriginalType(), arg, *extendedprice);
		char matLabel2[] = "extpriceMaterializer";
		ExprMaterializer *mat2 = new ExprMaterializer(toMat2, lineHint, mat1,
				&ctx, matLabel2);
		mat1->setParent(mat2);
		lastMat = mat2;

	}
	if (aggregatesNo == 4) {
		expressions::Expression* toMat1 = new expressions::RecordProjection(
				quantity->getOriginalType(), arg, *quantity);
		char matLabel1[] = "quantityMaterializer";
		ExprMaterializer *mat1 = new ExprMaterializer(toMat1, lineHint, mat0,
				&ctx, matLabel1);
		mat0->setParent(mat1);

		expressions::Expression* toMat2 = new expressions::RecordProjection(
				extendedprice->getOriginalType(), arg, *extendedprice);
		char matLabel2[] = "extpriceMaterializer";
		ExprMaterializer *mat2 = new ExprMaterializer(toMat2, lineHint, mat1,
				&ctx, matLabel2);
		mat1->setParent(mat2);

		expressions::Expression* toMat3 = new expressions::RecordProjection(
				tax->getOriginalType(), arg, *tax);
		char matLabel3[] = "taxMaterializer";
		ExprMaterializer *mat3 = new ExprMaterializer(toMat3, lineHint, mat2,
				&ctx, matLabel3);
		mat2->setParent(mat3);
		lastMat = mat3;
	}

	/**
	 * REDUCE
	 * COUNT(*)
	 */
	/* Output: */
	expressions::Expression* outputExpr = new expressions::IntConstant(1);
	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, lastMat, &ctx);
	lastMat->setParent(reduce);

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
