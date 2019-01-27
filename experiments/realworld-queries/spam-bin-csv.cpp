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

#include "experiments/realworld-queries/spam-bin-csv.hpp"


void symantecBinCSV1(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 70000000;
	int idHigh = 80000000;
	int clusterLow = 400;
	float valueLow = 0.5;
	float p_eventLow = 0.7;
	//csv
	int classaHigh = 19;

	RawContext& ctx = *prepareContext("symantec-bin-csv-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *p_event;
	RecordAttribute *cluster;
	RecordAttribute *value;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	p_event = argsSymantecBin["p_event"];
	value = argsSymantecBin["value"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(p_event);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);
	projectionsBin.push_back(value);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);


	/*
	 * SELECT BINARY
	 * st.id > 70000000  and st.id < 80000000 and (p_event > 0.7 OR value > 0.5) and cluster > 400
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		argProjectionsBin.push_back(*p_event);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);
		expressions::Expression* selEvent = new expressions::RecordProjection(
				p_event->getOriginalType(), arg, *p_event);
		expressions::Expression* selCluster = new expressions::RecordProjection(
				cluster->getOriginalType(), arg, *cluster);
		expressions::Expression* selValue = new expressions::RecordProjection(
						value->getOriginalType(), arg, *value);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::FloatConstant(
				p_eventLow);
		expressions::Expression* predExpr4 = new expressions::FloatConstant(
				valueLow);
		expressions::Expression* predicate3 = new expressions::GtExpression(
				selEvent, predExpr3);
		expressions::Expression* predicate4 = new expressions::GtExpression(
				selValue, predExpr4);
		expressions::Expression* predicateOr = new expressions::OrExpression(
				predicate3, predicate4);

		expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterLow);
		expressions::Expression* predicate5 = new expressions::GtExpression(
				selCluster, predExpr5);

		expressions::Expression* predicateAnd2 = new expressions::AndExpression(
				predicateOr, predicate5);

		predicateBin = new expressions::AndExpression(
				predicateAnd1, predicateAnd2);
	}

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
//	projections.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * classa < 19 and sc.id > 70000000  and sc.id < 80000000;
	 */
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				selClassa, predExpr3);

		predicateCSV = new expressions::AndExpression(
				predicateAnd1, predicate3);

	}

	Select *selCSV = new Select(predicateCSV, scanCSV);
	scanCSV->setParent(selCSV);

	/*
	 * JOIN
	 * st.id = sc.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idBin);
	argProjectionsLeft.push_back(*dim);
	expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idCSV);
	expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);
	vector<expression_t> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expression_t> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(join);
	selCSV->setParent(join);

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expression_t> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprDim;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join,
			&ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgBin->finish();
	pgCSV->finish();
	rawCatalog.clear();
}

void symantecBinCSV2(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 60000000;
	int idHigh = 70000000;
	int clusterLow = 400;
	float valueLow = 0.5;
	float p_eventLow = 0.7;
	//csv
	int classaHigh = 19;

	RawContext& ctx = *prepareContext("symantec-bin-csv-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *p_event;
	RecordAttribute *cluster;
	RecordAttribute *value;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	p_event = argsSymantecBin["p_event"];
	value = argsSymantecBin["value"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(p_event);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);
	projectionsBin.push_back(value);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);


	/*
	 * SELECT BINARY
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		argProjectionsBin.push_back(*p_event);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);
		expressions::Expression* selEvent = new expressions::RecordProjection(
				p_event->getOriginalType(), arg, *p_event);
		expressions::Expression* selCluster = new expressions::RecordProjection(
				cluster->getOriginalType(), arg, *cluster);
		expressions::Expression* selValue = new expressions::RecordProjection(
						value->getOriginalType(), arg, *value);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::FloatConstant(
				p_eventLow);
		expressions::Expression* predExpr4 = new expressions::FloatConstant(
				valueLow);
		expressions::Expression* predicate3 = new expressions::GtExpression(
				selEvent, predExpr3);
		expressions::Expression* predicate4 = new expressions::GtExpression(
				selValue, predExpr4);

		expressions::Expression* predicateOr = new expressions::OrExpression(
				predicate3, predicate4);

		expressions::Expression* predExpr5 = new expressions::IntConstant(
				clusterLow);
		expressions::Expression* predicate5 = new expressions::GtExpression(
				selCluster, predExpr5);

		expressions::Expression* predicateAnd2 = new expressions::AndExpression(
				predicateOr, predicate5);

		predicateBin = new expressions::AndExpression(
				predicateAnd1, predicateAnd2);
	}

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
//	projections.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 */
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				selClassa, predExpr3);

		predicateCSV = new expressions::AndExpression(
				predicateAnd1, predicate3);

	}

	Select *selCSV = new Select(predicateCSV, scanCSV);
	scanCSV->setParent(selCSV);

	/*
	 * JOIN
	 * st.id = sc.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idBin);
	argProjectionsLeft.push_back(*dim);
	expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idCSV);
	expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);
	vector<expression_t> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expression_t> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(join);
	selCSV->setParent(join);

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expression_t> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprDim;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join,
			&ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgBin->finish();
	pgCSV->finish();
	rawCatalog.clear();
}

void symantecBinCSV3(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 97000000;
	int idHigh = 100000000;
	//csv
	string botName = "UNCLASSIFIED";

	RawContext& ctx = *prepareContext("symantec-bin-csv-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *neighbors;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	neighbors = argsSymantecBin["neighbors"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(neighbors);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);


	/*
	 * SELECT BINARY
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		predicateBin = new expressions::AndExpression(
				predicate1, predicate2);
	}

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	RecordAttribute *idCSV;
	RecordAttribute *bot;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
	projections.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 */
	expressions::Expression* predicateCSV;
	Select *selCSV;

	list<RecordAttribute> argProjectionsCSV;
	argProjectionsCSV.push_back(*idCSV);
	argProjectionsCSV.push_back(*bot);
	expressions::Expression* arg = new expressions::InputArgument(&recCSV, 0,
			argProjectionsCSV);
	expressions::Expression* selID = new expressions::RecordProjection(
			idCSV->getOriginalType(), arg, *idCSV);
	expressions::Expression* selBot = new expressions::RecordProjection(
			bot->getOriginalType(), arg, *bot);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			selID, predExpr2);
	expressions::Expression* predicateNum = new expressions::AndExpression(
			predicate1, predicate2);

	Select *selNum = new Select(predicateNum, scanCSV);
	scanCSV->setParent(selNum);

	expressions::Expression* predExpr3 = new expressions::StringConstant(
			botName);
	expressions::Expression* predicateStr = new expressions::EqExpression(
			selBot, predExpr3);

	selCSV = new Select(predicateStr, selNum);
	selNum->setParent(selCSV);


	/*
	 * JOIN
	 * st.id = sc.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idBin);
	argProjectionsLeft.push_back(*neighbors);
	expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idCSV);
	expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - neighbors needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(neighbors);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);
	vector<expression_t> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expression_t> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(join);
	selCSV->setParent(join);

	/**
	 * REDUCE
	 * MAX(neighbors), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*neighbors);
	expressions::Expression* exprNeighbors = new expressions::RecordProjection(
			neighbors->getOriginalType(), leftArg, *neighbors);
	/* Output: */
	vector<Monoid> accs;
	vector<expression_t> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprNeighbors;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join,
			&ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgBin->finish();
	pgCSV->finish();
	rawCatalog.clear();
}

void symantecBinCSV4(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 97000000;
	int idHigh = 100000000;
	//csv
	string botName = "UNCLASSIFIED";
	string countryCodeName = "RU";

	RawContext& ctx = *prepareContext("symantec-bin-csv-4");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *neighbors;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	neighbors = argsSymantecBin["neighbors"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);

	projectionsBin.push_back(neighbors);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);


	/*
	 * SELECT BINARY
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		predicateBin = new expressions::AndExpression(
				predicate1, predicate2);
	}

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	RecordAttribute *idCSV;
	RecordAttribute *bot;
	RecordAttribute *country_code;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	bot = argsSymantecCSV["bot"];
	country_code = argsSymantecCSV["country_code"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
	projections.push_back(bot);
	projections.push_back(country_code);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 */
	Select *selCSV;

	list<RecordAttribute> argProjectionsCSV;
	argProjectionsCSV.push_back(*idCSV);
	argProjectionsCSV.push_back(*bot);
	expressions::Expression* arg = new expressions::InputArgument(&recCSV, 0,
			argProjectionsCSV);
	expressions::Expression* selID = new expressions::RecordProjection(
			idCSV->getOriginalType(), arg, *idCSV);
	expressions::Expression* selBot = new expressions::RecordProjection(
			bot->getOriginalType(), arg, *bot);
	expressions::Expression* selCountryCode = new expressions::RecordProjection(
			country_code->getOriginalType(), arg, *country_code);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			selID, predExpr2);
	expressions::Expression* predicateNum = new expressions::AndExpression(
			predicate1, predicate2);

	Select *selNum = new Select(predicateNum, scanCSV);
	scanCSV->setParent(selNum);

	expressions::Expression* predExpr3 = new expressions::StringConstant(
			botName);
	expressions::Expression* predExpr4 = new expressions::StringConstant(
			countryCodeName);
	expressions::Expression* predicateStr1 = new expressions::EqExpression(
			selBot, predExpr3);
	expressions::Expression* predicateStr2 = new expressions::EqExpression(
			selCountryCode, predExpr4);
	expressions::Expression* predicateStr = new expressions::AndExpression(
			predicateStr1, predicateStr2);

	selCSV = new Select(predicateStr, selNum);
	selNum->setParent(selCSV);


	/*
	 * JOIN
	 * st.id = sc.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idBin);
	argProjectionsLeft.push_back(*neighbors);
	expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idCSV);
	expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - neighbors needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(neighbors);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);
	vector<expression_t> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expression_t> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(join);
	selCSV->setParent(join);

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*neighbors);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			neighbors->getOriginalType(), leftArg, *neighbors);
	/* Output: */
	vector<Monoid> accs;
	vector<expression_t> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprDim;
	outputExprs.push_back(outputExpr1);

	accs.push_back(SUM);
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr2);
	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, join,
			&ctx);
	join->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgBin->finish();
	pgCSV->finish();
	rawCatalog.clear();
}

void symantecBinCSV5(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 350000000;
	int idHigh = 400000000;
	//csv
	string botName = "DARKMAILER3";
	string countryCodeName = "IN";
	int classaHigh = 20;

	RawContext& ctx = *prepareContext("symantec-bin-csv-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);


	/*
	 * SELECT BINARY
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		predicateBin = new expressions::AndExpression(
				predicate1, predicate2);
	}

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordAttribute *classb;
	RecordAttribute *bot;
	RecordAttribute *country_code;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	classb = argsSymantecCSV["classb"];
	country_code = argsSymantecCSV["country_code"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
//	projections.push_back(classa);
//	projections.push_back(classb);
	projections.push_back(country_code);
	projections.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * sc.bot = 'DARKMAILER3' and sc.id > 350000000 and sc.id < 400000000 and sc.country_code = 'IN' and sc.classa < 20
	 */
	expressions::Expression* predicateCSV;
	Select *selCSV;

	list<RecordAttribute> argProjectionsCSV;
	argProjectionsCSV.push_back(*idCSV);
	argProjectionsCSV.push_back(*bot);
	expressions::Expression* arg = new expressions::InputArgument(&recCSV, 0,
			argProjectionsCSV);
	expressions::Expression* selID = new expressions::RecordProjection(
			idCSV->getOriginalType(), arg, *idCSV);
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), arg, *classa);
	expressions::Expression* selBot = new expressions::RecordProjection(
			bot->getOriginalType(), arg, *bot);
	expressions::Expression* selCountryCode = new expressions::RecordProjection(
			country_code->getOriginalType(), arg, *country_code);

	expressions::Expression* predExprNum1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExprNum2 = new expressions::IntConstant(
			idHigh);
	expressions::Expression* predExprNum3 = new expressions::IntConstant(
			classaHigh);
	expressions::Expression* predicateNum1 = new expressions::GtExpression(
			selID, predExprNum1);
	expressions::Expression* predicateNum2 = new expressions::LtExpression(
			selID, predExprNum2);
	expressions::Expression* predicateNum3 = new expressions::LtExpression(
			selClassa, predExprNum3);

	expressions::Expression* predicateNum_ = new expressions::AndExpression(
			predicateNum1, predicateNum2);
	expressions::Expression* predicateNum = new expressions::AndExpression(
			predicateNum_, predicateNum3);

	Select *selNum = new Select(predicateNum, scanCSV);
	scanCSV->setParent(selNum);

	expressions::Expression* predExpr3 = new expressions::StringConstant(
			botName);
	expressions::Expression* predExpr4 = new expressions::StringConstant(
			countryCodeName);
	expressions::Expression* predicateStr1 = new expressions::EqExpression(
			selBot, predExpr3);
	expressions::Expression* predicateStr2 = new expressions::EqExpression(
			selCountryCode, predExpr4);
	expressions::Expression* predicateStr = new expressions::AndExpression(
			predicateStr1, predicateStr2);

	selCSV = new Select(predicateStr, selNum);
	selNum->setParent(selCSV);


	/*
	 * JOIN
	 * st.id = sc.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idBin);
	expressions::Expression* leftArg = new expressions::InputArgument(&recBin,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idCSV);
	expressions::Expression* rightArg = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - neighbors needed */
	vector<RecordAttribute*> fieldsLeft;
	vector<materialization_mode> outputModesLeft;

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg, *idBin);
	vector<expression_t> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	fieldsRight.push_back(classb);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expression_t> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *join = new RadixJoin(*joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(join);
	selCSV->setParent(join);

	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(classb), count(*)
	 */
	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*classa);
	nestProjections.push_back(*classb);

	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY country_code
	expressions::RecordProjection* f = new expressions::RecordProjection(
			classa->getOriginalType(), nestArg, *classa);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(classa);
	outputModes.insert(outputModes.begin(), EAGER);
	fields.push_back(classb);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_classa";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expression_t> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(classb) */
	expressions::Expression* aggrClassb = new expressions::RecordProjection(
			classb->getOriginalType(), rightArg, *classb);
	expressions::Expression* outputExpr1 = aggrClassb;
	aggrField1 = string("_maxClassb");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	/* Aggregate 2: MAX(size) */
	expressions::Expression* outputExpr2 = new expressions::IntConstant(1);
	aggrField2 = string("_count");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, join, nestLabel, *mat);
	join->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	Function* debugFloat = ctx.getFunction("printFloat");
	IntType intType = IntType();
	FloatType floatType = FloatType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&floatType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&floatType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugFloat, nestOutput1, nestOp);
	nestOp->setParent(printOp1);

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
	pgBin->finish();
	pgCSV->finish();
	rawCatalog.clear();
}
