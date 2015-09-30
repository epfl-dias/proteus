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

#include "experiments/realworld-queries/spam-bin.hpp"

void symantecBin1(map<string,dataset> datasetCatalog)	{

	int idLow = 50000000;
	int idHigh = 60000000;
	RawContext ctx = prepareContext("symantec-bin-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			p_event->getOriginalType(), arg, *p_event);
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: */
	expressions::Expression* selID  	=
			new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selID, predExpr2);
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

void symantecBin2(map<string,dataset> datasetCatalog)	{

	int idLow = 50000000;
	int idHigh = 60000000;
	int dimHigh = 3;

	RawContext ctx = prepareContext("symantec-bin-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];
	RecordAttribute *size = argsSymantecBin["size"];
	RecordAttribute *slice_id = argsSymantecBin["slice_id"];
	RecordAttribute *value = argsSymantecBin["value"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *mdc = argsSymantecBin["mdc"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);
	projections.push_back(size);
	projections.push_back(slice_id);
	projections.push_back(value);
	projections.push_back(dim);
	projections.push_back(mdc);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	argProjections.push_back(*p_event);
	argProjections.push_back(*size);
	argProjections.push_back(*slice_id);
	argProjections.push_back(*value);
	argProjections.push_back(*dim);
	argProjections.push_back(*mdc);
	argProjections.push_back(*cluster);

	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = new expressions::RecordProjection(
			p_event->getOriginalType(), arg, *p_event);
	outputExprs.push_back(outputExpr2);

	accs.push_back(MAX);
	expressions::Expression* outputExpr3 = new expressions::RecordProjection(
			size->getOriginalType(), arg, *size);
	outputExprs.push_back(outputExpr3);

	accs.push_back(MAX);
	expressions::Expression* outputExpr4 = new expressions::RecordProjection(
			slice_id->getOriginalType(), arg, *slice_id);
	outputExprs.push_back(outputExpr4);

	accs.push_back(MAX);
	expressions::Expression* outputExpr5 = new expressions::RecordProjection(
			value->getOriginalType(), arg, *value);
	outputExprs.push_back(outputExpr5);

	accs.push_back(MAX);
	expressions::Expression* outputExpr6 = new expressions::RecordProjection(
			dim->getOriginalType(), arg, *dim);
	outputExprs.push_back(outputExpr6);

	accs.push_back(MAX);
	expressions::Expression* outputExpr7 = new expressions::RecordProjection(
			mdc->getOriginalType(), arg, *mdc);
	outputExprs.push_back(outputExpr7);

	accs.push_back(MAX);
	expressions::Expression* outputExpr8 = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);
	outputExprs.push_back(outputExpr8);

	accs.push_back(SUM);
	expressions::Expression* outputExpr9 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr9);
	/* Pred: */
	expressions::Expression* selID = outputExpr1;
	expressions::Expression* selDim = outputExpr6;
	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(dimHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selDim, predExpr2);
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

void symantecBin3(map<string,dataset> datasetCatalog)	{

	int idLow = 59000000;
	int idHigh = 63000000;
	int dimHigh = 3;
	int clusterNo = 500;
	RawContext ctx = prepareContext("symantec-bin-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);
	projections.push_back(dim);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	argProjections.push_back(*p_event);
	argProjections.push_back(*cluster);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			p_event->getOriginalType(), arg, *p_event);
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);

	/* Pred: */
	expressions::Expression* selID  	=
			new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selDim  	=
				new expressions::RecordProjection(dim->getOriginalType(),arg,*dim);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);


	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(dimHigh);
	expressions::Expression* predExpr4 = new expressions::IntConstant(clusterNo);
	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::LtExpression(
						new BoolType(), selDim, predExpr3);
	expressions::Expression* predicate4 = new expressions::EqExpression(
						new BoolType(), selCluster, predExpr4);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
				new BoolType(), predicate3, predicate4);
	expressions::Expression* predicate = new expressions::AndExpression(
				new BoolType(), predicateAnd1, predicateAnd2);

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

void symantecBin4(map<string,dataset> datasetCatalog)	{

	int idLow = 70000000;
	int idHigh = 80000000;
	double p_eventLow = 0.7;
	double valueLow = 0.5;
	int clusterNo = 400;

	RawContext ctx = prepareContext("symantec-bin-4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];
	RecordAttribute *value = argsSymantecBin["value"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);
	projections.push_back(dim);
	projections.push_back(cluster);
	projections.push_back(value);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	argProjections.push_back(*p_event);
	argProjections.push_back(*value);
	argProjections.push_back(*dim);
	argProjections.push_back(*cluster);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			dim->getOriginalType(), arg, *dim);
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);

	/* Pred: */
	/* id > 70000000 and id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster = 400 */
	expressions::Expression* selID = new expressions::RecordProjection(
			id->getOriginalType(), arg, *id);
	expressions::Expression* selEvent = new expressions::RecordProjection(
			p_event->getOriginalType(), arg, *p_event);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);
	expressions::Expression* selValue = new expressions::RecordProjection(value->getOriginalType(),arg,*value);


	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::FloatConstant(
			p_eventLow);
	expressions::Expression* predExpr4 = new expressions::FloatConstant(valueLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
			clusterNo);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::GtExpression(
			new BoolType(), selEvent, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
			new BoolType(), selValue, predExpr4);
	expressions::Expression* predicate5 = new expressions::EqExpression(
			new BoolType(), selCluster, predExpr5);

	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicateOr = new expressions::OrExpression(
			new BoolType(), predicate3, predicate4);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
			new BoolType(), predicateAnd1, predicateOr);
	expressions::Expression* predicate = new expressions::AndExpression(
			new BoolType(), predicateAnd2, predicate5);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predicate, scan,
			&ctx);
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

//SELECT MAX(dim), COUNT(*)
//FROM symantecunordered
//where id > 70000000  and id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster = 400;
void symantecBin4v1(map<string,dataset> datasetCatalog)	{

	int idLow = 70000000;
	int idHigh = 80000000;
	double p_eventLow = 0.7;
	double valueLow = 0.5;
	int clusterNo = 400;

	RawContext ctx = prepareContext("symantec-bin-4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];
	RecordAttribute *value = argsSymantecBin["value"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);
	projections.push_back(dim);
	projections.push_back(cluster);
	projections.push_back(value);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/*
	 * SELECTS
	 */

	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*p_event);
	argSelections.push_back(*value);
	argSelections.push_back(*cluster);

	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);
	expressions::Expression* selEvent  	=
						new expressions::RecordProjection(p_event->getOriginalType(),arg,*p_event);
	expressions::Expression* selValue  	=
							new expressions::RecordProjection(value->getOriginalType(),arg,*value);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::FloatConstant(
			p_eventLow);
	expressions::Expression* predExpr4 = new expressions::FloatConstant(
			valueLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
			clusterNo);

	//cluster equality
	expressions::Expression* predicate5 = new expressions::EqExpression(
				new BoolType(), selCluster, predExpr5);
	Select *sel1 = new Select(predicate5, scan);
	scan->setParent(sel1);

	//id < ...
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	Select *sel2 = new Select(predicate2, sel1);
	sel1->setParent(sel2);

	//OR
	expressions::Expression* predicate3 = new expressions::GtExpression(
			new BoolType(), selEvent, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
			new BoolType(), selValue, predExpr4);
	expressions::Expression* predicateOr = new expressions::OrExpression(
			new BoolType(), predicate3, predicate4);
	Select *sel3 = new Select(predicateOr, sel2);
	sel2->setParent(sel3);

	//id > ...
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	Select *sel = new Select(predicate1, sel3);
	sel3->setParent(sel);


	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* argProj 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			dim->getOriginalType(), argProj, *dim);
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);

	/* Pred: Redundant */
	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, sel,
			&ctx);
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

void symantecBin5(map<string,dataset> datasetCatalog)	{

	int idLow = 380000000;
	int idHigh = 450000000;
	int sliceIdNo = 150;
	RawContext ctx = prepareContext("symantec-bin-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
			symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;


	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *slice_id = argsSymantecBin["slice_id"];
	RecordAttribute *neighbors = argsSymantecBin["neighbors"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(slice_id);
	projections.push_back(neighbors);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
				projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*id);
	argProjections.push_back(*slice_id);
	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = new expressions::RecordProjection(
			neighbors->getOriginalType(), arg, *neighbors);
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);

	/* Pred: */
	expressions::Expression* selID  	=
			new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selSliceID  	=
				new expressions::RecordProjection(slice_id->getOriginalType(),arg,*slice_id);
	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(sliceIdNo);
	expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
					new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::EqExpression(
						new BoolType(), selSliceID, predExpr3);
	expressions::Expression* predicate_ = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicate = new expressions::AndExpression(
				new BoolType(), predicate_, predicate3);

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

void symantecBin6(map<string, dataset> datasetCatalog) {

	int idLow = 380000000;
	int idHigh = 450000000;
	int clusterHigh = 10;
	RawContext ctx = prepareContext("symantec-bin-6(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(dim);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*dim);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);
	expressions::Expression* predicate_ = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicate = new expressions::AndExpression(
			new BoolType(), predicate_, predicate3);


	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(dim), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*dim);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(dim);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(dim) */
	expressions::Expression* aggrDim = new expressions::RecordProjection(
			dim->getOriginalType(), arg, *dim);
	expressions::Expression* outputExpr1 = aggrDim;
	aggrField1 = string("_maxDim");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin6v1(map<string, dataset> datasetCatalog) {

//	int idLow = 380000000;
//	int idHigh = 450000000;
	int clusterHigh = 10;
	RawContext ctx = prepareContext("symantec-bin-6(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
//	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
//	projections.push_back(id);
	projections.push_back(dim);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argSelections;
//	argSelections.push_back(*id);
//	argSelections.push_back(*dim);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
//	expressions::Expression* selID  	=
//				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

//	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
//	expressions::Expression* predicate1 = new expressions::GtExpression(
//			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);
//	expressions::Expression* predicate_ = new expressions::AndExpression(
//			new BoolType(), predicate1, predicate2);
//	expressions::Expression* predicate = new expressions::AndExpression(
//			new BoolType(), predicate_, predicate3);
	expressions::Expression* predicate = predicate3;


	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(dim), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*dim);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(dim);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(dim) */
	expressions::Expression* aggrDim = new expressions::RecordProjection(
			dim->getOriginalType(), arg, *dim);
	expressions::Expression* outputExpr1 = aggrDim;
	aggrField1 = string("_maxDim");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin6v2(map<string, dataset> datasetCatalog) {

	int idLow = 1;
//	int idHigh = 450000000;
	int clusterHigh = 10;
	RawContext ctx = prepareContext("symantec-bin-6(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(dim);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
//	argSelections.push_back(*dim);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);
//	expressions::Expression* predicate_ = new expressions::AndExpression(
//			new BoolType(), predicate1, predicate2);
//	expressions::Expression* predicate = new expressions::AndExpression(
//			new BoolType(), predicate_, predicate3);
	Select *sel1 = new Select(predicate3, scan);
	scan->setParent(sel1);

	expressions::Expression* predicate = predicate1;

	Select *sel = new Select(predicate, sel1);
	sel1->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(dim), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*dim);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(dim);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(dim) */
	expressions::Expression* aggrDim = new expressions::RecordProjection(
			dim->getOriginalType(), arg, *dim);
	expressions::Expression* outputExpr1 = aggrDim;
	aggrField1 = string("_maxDim");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin7(map<string, dataset> datasetCatalog) {

	int idLow = 59000000;
	int idHigh = 63000000;
	int dimHigh = 3;
	int clusterLow = 490;
	int clusterHigh = 500;
	RawContext ctx = prepareContext("symantec-bin-7(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *mdc = argsSymantecBin["mdc"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(dim);
	projections.push_back(mdc);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 59000000 and id < 63000000 and dim < 3 AND cluster > 490 AND cluster <= 500
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*dim);
	argSelections.push_back(*mdc);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selDim  	=
					new expressions::RecordProjection(dim->getOriginalType(),arg,*dim);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::IntConstant(
			dimHigh);
	expressions::Expression* predExpr4 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterHigh);


	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selDim, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
				new BoolType(), selCluster, predExpr4);
	expressions::Expression* predicate5 = new expressions::LeExpression(
				new BoolType(), selCluster, predExpr5);


	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
			new BoolType(), predicate3, predicate4);
	expressions::Expression* predicateAnd_ = new expressions::AndExpression(
				new BoolType(), predicateAnd1, predicateAnd2);
	expressions::Expression* predicate = new expressions::AndExpression(
					new BoolType(), predicateAnd_, predicate5);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(mdc), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*mdc);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(mdc);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(mdc) */
	expressions::Expression* aggrMDC = new expressions::RecordProjection(
			mdc->getOriginalType(), arg, *mdc);
	expressions::Expression* outputExpr1 = aggrMDC;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin7v1(map<string, dataset> datasetCatalog) {

//	int idLow = 59000000;
//	int idHigh = 63000000;
//	int dimHigh = 3;
	int clusterLow = 490;
	int clusterHigh = 500;
	RawContext ctx = prepareContext("symantec-bin-7(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
//	RecordAttribute *id = argsSymantecBin["id"];
//	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *mdc = argsSymantecBin["mdc"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
//	projections.push_back(id);
//	projections.push_back(dim);
	projections.push_back(mdc);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 59000000 and id < 63000000 and dim < 3 AND cluster > 490 AND cluster <= 500
	 */
	list<RecordAttribute> argSelections;
//	argSelections.push_back(*id);
//	argSelections.push_back(*dim);
	argSelections.push_back(*mdc);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
//	expressions::Expression* selID  	=
//				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
//	expressions::Expression* selDim  	=
//					new expressions::RecordProjection(dim->getOriginalType(),arg,*dim);
	expressions::Expression* selCluster  	=
			new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

//	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
//	expressions::Expression* predExpr3 = new expressions::IntConstant(
//			dimHigh);
	expressions::Expression* predExpr4 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterHigh);


//	expressions::Expression* predicate1 = new expressions::GtExpression(
//			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
//	expressions::Expression* predicate3 = new expressions::LtExpression(
//			new BoolType(), selDim, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
				new BoolType(), selCluster, predExpr4);
	expressions::Expression* predicate5 = new expressions::LeExpression(
				new BoolType(), selCluster, predExpr5);
//
//
//	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
//			new BoolType(), predicate1, predicate2);
//	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
//			new BoolType(), predicate3, predicate4);
//	expressions::Expression* predicateAnd_ = new expressions::AndExpression(
//				new BoolType(), predicateAnd1, predicateAnd2);
//	expressions::Expression* predicate = new expressions::AndExpression(
//					new BoolType(), predicateAnd_, predicate5);
	expressions::Expression* predicate = new expressions::AndExpression(
						new BoolType(), predicate4, predicate5);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(mdc), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*mdc);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(mdc);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(mdc) */
	expressions::Expression* aggrMDC = new expressions::RecordProjection(
			mdc->getOriginalType(), arg, *mdc);
	expressions::Expression* outputExpr1 = aggrMDC;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin7v2(map<string, dataset> datasetCatalog) {

	int idLow = 1;
//	int idHigh = 63000000;
//	int dimHigh = 3;
	int clusterLow = 490;
	int clusterHigh = 500;
	RawContext ctx = prepareContext("symantec-bin-7(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
//	RecordAttribute *dim = argsSymantecBin["dim"];
	RecordAttribute *mdc = argsSymantecBin["mdc"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
//	projections.push_back(dim);
	projections.push_back(mdc);
	projections.push_back(cluster);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 59000000 and id < 63000000 and dim < 3 AND cluster > 490 AND cluster <= 500
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
//	argSelections.push_back(*dim);
	argSelections.push_back(*mdc);
	argSelections.push_back(*cluster);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
//	expressions::Expression* selDim  	=
//					new expressions::RecordProjection(dim->getOriginalType(),arg,*dim);
	expressions::Expression* selCluster  	=
			new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
//	expressions::Expression* predExpr3 = new expressions::IntConstant(
//			dimHigh);
	expressions::Expression* predExpr4 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterHigh);


	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
//	expressions::Expression* predicate3 = new expressions::LtExpression(
//			new BoolType(), selDim, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
				new BoolType(), selCluster, predExpr4);
	expressions::Expression* predicate5 = new expressions::LeExpression(
				new BoolType(), selCluster, predExpr5);
//
//
//	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
//			new BoolType(), predicate1, predicate2);
//	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
//			new BoolType(), predicate3, predicate4);
//	expressions::Expression* predicateAnd_ = new expressions::AndExpression(
//				new BoolType(), predicateAnd1, predicateAnd2);
//	expressions::Expression* predicate = new expressions::AndExpression(
//					new BoolType(), predicateAnd_, predicate5);
	expressions::Expression* predicate = new expressions::AndExpression(
						new BoolType(), predicate4, predicate5);

	Select *sel1 = new Select(predicate, scan);
	scan->setParent(sel1);

	Select *sel = new Select(predicate1, sel1);
	sel1->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(mdc), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*mdc);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(mdc);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(mdc) */
	expressions::Expression* aggrMDC = new expressions::RecordProjection(
			mdc->getOriginalType(), arg, *mdc);
	expressions::Expression* outputExpr1 = aggrMDC;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin8(map<string, dataset> datasetCatalog) {

	int idLow = 70000000;
	int idHigh = 80000000;
	double p_eventLow = 0.7;
	double valueLow = 0.5;
	int clusterLow = 395;
	int clusterHigh = 405;
	RawContext ctx = prepareContext("symantec-bin-8(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
	RecordAttribute *p_event = argsSymantecBin["p_event"];
	RecordAttribute *value = argsSymantecBin["value"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];
	RecordAttribute *neighbors = argsSymantecBin["neighbors"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
	projections.push_back(p_event);
	projections.push_back(value);
	projections.push_back(cluster);
	projections.push_back(neighbors);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 70000000  and id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster > 395 and cluster <= 405
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
	argSelections.push_back(*p_event);
	argSelections.push_back(*value);
	argSelections.push_back(*cluster);
	argSelections.push_back(*neighbors);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
	expressions::Expression* selEvent  	=
						new expressions::RecordProjection(p_event->getOriginalType(),arg,*p_event);
	expressions::Expression* selValue  	=
						new expressions::RecordProjection(value->getOriginalType(),arg,*value);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predExpr3 = new expressions::FloatConstant(
			p_eventLow);
	expressions::Expression* predExpr4 = new expressions::FloatConstant(
				valueLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr6 = new expressions::IntConstant(
					clusterHigh);

	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

	expressions::Expression* predicate3 = new expressions::GtExpression(
			new BoolType(), selEvent, predExpr3);
	expressions::Expression* predicate4 = new expressions::GtExpression(
				new BoolType(), selValue, predExpr4);
	expressions::Expression* predicateOr = new expressions::OrExpression(
					new BoolType(), predicate3, predicate4);

	expressions::Expression* predicate5 = new expressions::GtExpression(
			new BoolType(), selCluster, predExpr5);
	expressions::Expression* predicate6 = new expressions::LeExpression(
			new BoolType(), selCluster, predExpr6);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
			new BoolType(), predicate5, predicate6);

	expressions::Expression* predicateAnd = new expressions::AndExpression(
				new BoolType(), predicateAnd1, predicateOr);

	expressions::Expression* predicate = new expressions::AndExpression(
			new BoolType(), predicateAnd, predicateAnd2);

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(neighbors), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*neighbors);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(neighbors);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(neighbors) */
	expressions::Expression* aggrNeighbors = new expressions::RecordProjection(
			neighbors->getOriginalType(), arg, *neighbors);
	expressions::Expression* outputExpr1 = aggrNeighbors;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin8v1(map<string, dataset> datasetCatalog) {

//	int idLow = 70000000;
//	int idHigh = 80000000;
//	double p_eventLow = 0.7;
//	double valueLow = 0.5;
	int clusterLow = 395;
	int clusterHigh = 405;
	RawContext ctx = prepareContext("symantec-bin-8(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
//	RecordAttribute *id = argsSymantecBin["id"];
//	RecordAttribute *p_event = argsSymantecBin["p_event"];
//	RecordAttribute *value = argsSymantecBin["value"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];
	RecordAttribute *neighbors = argsSymantecBin["neighbors"];

	vector<RecordAttribute*> projections;
//	projections.push_back(id);
//	projections.push_back(p_event);
//	projections.push_back(value);
	projections.push_back(cluster);
	projections.push_back(neighbors);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 70000000  and id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster > 395 and cluster <= 405
	 */
	list<RecordAttribute> argSelections;
//	argSelections.push_back(*id);
//	argSelections.push_back(*p_event);
//	argSelections.push_back(*value);
	argSelections.push_back(*cluster);
	argSelections.push_back(*neighbors);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
//	expressions::Expression* selID  	=
//				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
//	expressions::Expression* selEvent  	=
//						new expressions::RecordProjection(p_event->getOriginalType(),arg,*p_event);
//	expressions::Expression* selValue  	=
//						new expressions::RecordProjection(value->getOriginalType(),arg,*value);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

//	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
//	expressions::Expression* predExpr3 = new expressions::FloatConstant(
//			p_eventLow);
//	expressions::Expression* predExpr4 = new expressions::FloatConstant(
//				valueLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr6 = new expressions::IntConstant(
					clusterHigh);

//	expressions::Expression* predicate1 = new expressions::GtExpression(
//			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
//	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
//				new BoolType(), predicate1, predicate2);
//
//	expressions::Expression* predicate3 = new expressions::GtExpression(
//			new BoolType(), selEvent, predExpr3);
//	expressions::Expression* predicate4 = new expressions::GtExpression(
//				new BoolType(), selValue, predExpr4);
//	expressions::Expression* predicateOr = new expressions::OrExpression(
//					new BoolType(), predicate3, predicate4);

	expressions::Expression* predicate5 = new expressions::GtExpression(
			new BoolType(), selCluster, predExpr5);
	expressions::Expression* predicate6 = new expressions::LeExpression(
			new BoolType(), selCluster, predExpr6);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
			new BoolType(), predicate5, predicate6);

//	expressions::Expression* predicateAnd = new expressions::AndExpression(
//				new BoolType(), predicateAnd1, predicateOr);
//
//	expressions::Expression* predicate = new expressions::AndExpression(
//			new BoolType(), predicateAnd, predicateAnd2);

	expressions::Expression* predicate = predicateAnd2;

	Select *sel = new Select(predicate, scan);
	scan->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(neighbors), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*neighbors);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(neighbors);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(neighbors) */
	expressions::Expression* aggrNeighbors = new expressions::RecordProjection(
			neighbors->getOriginalType(), arg, *neighbors);
	expressions::Expression* outputExpr1 = aggrNeighbors;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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

void symantecBin8v2(map<string, dataset> datasetCatalog) {

	int idLow = 1;
//	int idHigh = 80000000;
//	double p_eventLow = 0.7;
//	double valueLow = 0.5;
	int clusterLow = 395;
	int clusterHigh = 405;
	RawContext ctx = prepareContext("symantec-bin-8(agg)");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantecBin 	=
				symantecBin.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	string fnamePrefix = symantecBin.path;
	RecordType rec = symantecBin.recType;
	int linehint = symantecBin.linehint;
	RecordAttribute *id = argsSymantecBin["id"];
//	RecordAttribute *p_event = argsSymantecBin["p_event"];
//	RecordAttribute *value = argsSymantecBin["value"];
	RecordAttribute *cluster = argsSymantecBin["cluster"];
	RecordAttribute *neighbors = argsSymantecBin["neighbors"];

	vector<RecordAttribute*> projections;
	projections.push_back(id);
//	projections.push_back(p_event);
//	projections.push_back(value);
	projections.push_back(cluster);
	projections.push_back(neighbors);

	BinaryColPlugin *pg = new BinaryColPlugin(&ctx, fnamePrefix, rec,
			projections);
	rawCatalog.registerPlugin(fnamePrefix, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * SELECT
	 * id > 70000000  and id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster > 395 and cluster <= 405
	 */
	list<RecordAttribute> argSelections;
	argSelections.push_back(*id);
//	argSelections.push_back(*p_event);
//	argSelections.push_back(*value);
	argSelections.push_back(*cluster);
	argSelections.push_back(*neighbors);
	expressions::Expression* arg 			=
					new expressions::InputArgument(&rec,0,argSelections);
	expressions::Expression* selID  	=
				new expressions::RecordProjection(id->getOriginalType(),arg,*id);
//	expressions::Expression* selEvent  	=
//						new expressions::RecordProjection(p_event->getOriginalType(),arg,*p_event);
//	expressions::Expression* selValue  	=
//						new expressions::RecordProjection(value->getOriginalType(),arg,*value);
	expressions::Expression* selCluster  	=
					new expressions::RecordProjection(cluster->getOriginalType(),arg,*cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
//	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
//	expressions::Expression* predExpr3 = new expressions::FloatConstant(
//			p_eventLow);
//	expressions::Expression* predExpr4 = new expressions::FloatConstant(
//				valueLow);
	expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterLow);
	expressions::Expression* predExpr6 = new expressions::IntConstant(
					clusterHigh);

	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
//	expressions::Expression* predicate2 = new expressions::LtExpression(
//			new BoolType(), selID, predExpr2);
//	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
//				new BoolType(), predicate1, predicate2);
//
//	expressions::Expression* predicate3 = new expressions::GtExpression(
//			new BoolType(), selEvent, predExpr3);
//	expressions::Expression* predicate4 = new expressions::GtExpression(
//				new BoolType(), selValue, predExpr4);
//	expressions::Expression* predicateOr = new expressions::OrExpression(
//					new BoolType(), predicate3, predicate4);

	expressions::Expression* predicate5 = new expressions::GtExpression(
			new BoolType(), selCluster, predExpr5);
	expressions::Expression* predicate6 = new expressions::LeExpression(
			new BoolType(), selCluster, predExpr6);
	expressions::Expression* predicateAnd2 = new expressions::AndExpression(
			new BoolType(), predicate5, predicate6);

//	expressions::Expression* predicateAnd = new expressions::AndExpression(
//				new BoolType(), predicateAnd1, predicateOr);
//
//	expressions::Expression* predicate = new expressions::AndExpression(
//			new BoolType(), predicateAnd, predicateAnd2);

	expressions::Expression* predicate = predicateAnd2;

	Select *sel1 = new Select(predicate, scan);
	scan->setParent(sel1);

	Select *sel = new Select(predicate1, sel1);
	sel1->setParent(sel);

	/**
	 * NEST
	 * GroupBy: cluster
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: MAX(neighbors), COUNT() => SUM(1)
	 */
	list<RecordAttribute> nestProjections;
		nestProjections.push_back(*cluster);
		nestProjections.push_back(*neighbors);
	expressions::Expression* nestArg = new expressions::InputArgument(&rec, 0,
			nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			cluster->getOriginalType(), nestArg, *cluster);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);


	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(neighbors);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(cluster);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);


	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(neighbors) */
	expressions::Expression* aggrNeighbors = new expressions::RecordProjection(
			neighbors->getOriginalType(), arg, *neighbors);
	expressions::Expression* outputExpr1 = aggrNeighbors;
	aggrField1 = string("_maxMDC");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: COUNT(*) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_aggrCount");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);



	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, sel, nestLabel, *mat);
	sel->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	//Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	//FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
	nestOp->setParent(printOp1);
	lastPrintOp = printOp1;

	RecordAttribute *toOutput2 = new RecordAttribute(2, aggrLabel, aggrField2,
			&intType);
	expressions::RecordProjection* nestOutput2 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput2);
	Print *printOp2 = new Print(debugInt, nestOutput2, printOp1);
	printOp1->setParent(printOp2);
	lastPrintOp = printOp2;


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
