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
#include "operators/materializer-expr.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/binary-row-plugin.hpp"
#include "plugins/binary-col-plugin.hpp"
#include "plugins/json-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "expressions/expressions-hasher.hpp"
#include "util/raw-caching.hpp"
#include "common/symantec-config.hpp"
#include "operators/materializer-expr.hpp"


/* COUNT() */
void symantecProjection1(map<string,dataset> datasetCatalog, int predicateVal);

//RawContext prepareContext(string moduleName)	{
//	RawContext ctx = RawContext(moduleName);
//	registerFunctions(ctx);
//	return ctx;
//}


int main()	{
	cout << "Execution" << endl;
	map<string,dataset> datasetCatalog;
	symantecCoreSchema(datasetCatalog);

	/* Filtering on email size */
	int predicateVal = 5000;
//	int predicateVal = 600;
	symantecProjection1(datasetCatalog, predicateVal);
	symantecProjection1(datasetCatalog, predicateVal);
}

//void symantecProjection1(map<string,dataset> datasetCatalog, int predicateVal)	{
//
//	RawContext ctx = prepareContext("symantec-json-projection1");
//	RawCatalog& rawCatalog = RawCatalog::getInstance();
//
//	string nameSymantec = string("symantec");
//	dataset symantec = datasetCatalog[nameSymantec];
//	string nameOrders = string("orders");
//	dataset orders = datasetCatalog[nameOrders];
//	map<string, RecordAttribute*> argsLineitem 	=
//			symantec.recType.getArgsMap();
//	map<string, RecordAttribute*> argsOrder		=
//			orders.recType.getArgsMap();
//
//	/**
//	 * SCAN
//	 */
//	string fname = symantec.path;
//	RecordType rec = symantec.recType;
//	int linehint = symantec.linehint;
//	RecordAttribute *size = argsLineitem["size"];
//
//	ListType *documentType = new ListType(rec);
//	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
//			documentType,linehint);
//
//	rawCatalog.registerPlugin(fname, pg);
//	Scan *scan = new Scan(&ctx, *pg);
//
//	/**
//	 * REDUCE
//	 * (SUM 1)
//	 * + predicate
//	 */
//	list<RecordAttribute> argProjections;
//	argProjections.push_back(*size);
//	expressions::Expression* arg 			=
//				new expressions::InputArgument(&rec,0,argProjections);
//	/* Output: */
//	vector<Monoid> accs;
//	vector<expressions::Expression*> outputExprs;
//	accs.push_back(SUM);
//	expressions::Expression* outputExpr = new expressions::IntConstant(1);
//	outputExprs.push_back(outputExpr);
//	/* Pred: */
//	expressions::Expression* selOrderkey  	=
//			new expressions::RecordProjection(size->getOriginalType(),arg,*size);
//	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
//	expressions::Expression* predicate = new expressions::LtExpression(
//				new BoolType(), selOrderkey, vakey);
//
//	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan, &ctx);
//	scan->setParent(reduce);
//
//	//Run function
//	struct timespec t0, t1;
//	clock_gettime(CLOCK_REALTIME, &t0);
//	reduce->produce();
//	ctx.prepareFunction(ctx.getGlobalFunction());
//	clock_gettime(CLOCK_REALTIME, &t1);
//	printf("Execution took %f seconds\n", diff(t0, t1));
//
//	//Close all open files & clear
//	pg->finish();
//	rawCatalog.clear();
//}

void symantecProjection1(map<string,dataset> datasetCatalog, int predicateVal)	{

	RawContext ctx = prepareContext("symantec-json-projection1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantec");
	dataset symantec = datasetCatalog[nameSymantec];
	string nameOrders = string("orders");
	dataset orders = datasetCatalog[nameOrders];
	map<string, RecordAttribute*> argsLineitem 	=
			symantec.recType.getArgsMap();
	map<string, RecordAttribute*> argsOrder		=
			orders.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = symantec.path;
	RecordType rec = symantec.recType;
	int linehint = symantec.linehint;
	RecordAttribute *size = argsLineitem["size"];

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType,linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 * + predicate
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*size);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	accs.push_back(SUM);
	expressions::Expression* outputExpr =
			new expressions::IntConstant(1);
//			new expressions::RecordProjection(size->getOriginalType(),arg,*size);
//	new expressions::RecordProjection(size->getOriginalType(),arg,*size);
	outputExprs.push_back(outputExpr);
	/* Pred: */
	expressions::Expression* selOrderkey  	=
			new expressions::RecordProjection(size->getOriginalType(),arg,*size);
	expressions::Expression* vakey = new expressions::IntConstant(predicateVal);
	expressions::Expression* vakey2 = new expressions::IntConstant(0);
	expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selOrderkey, vakey);
	expressions::Expression* predicate2 = new expressions::GtExpression(
					new BoolType(), selOrderkey, vakey2);
	expressions::Expression* predicate = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

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

