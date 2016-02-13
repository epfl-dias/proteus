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

#include "experiments/realworld-vldb/spam-bin-csv-json-v2.hpp"

void symantecBinCSVJSON1(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 160000000;
	int idHigh = 200000000;
	int clusterHigh = 200;
	//csv
	int classaHigh = 90;
//	string botName = "Bobax";
	string botName = "GHEG";

	RawContext ctx = prepareContext("symantec-bin-csv-json-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 200 and st.id > 8000000 and st.id < 10000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);

	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd1,
			predicate3);

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
	RecordAttribute *bot;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
//	projections.push_back(classa);
	projections.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * classA < 90 and bot = 'Bobax' and sc.id > 8000000 and sc.id < 10000000
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*classa);
		argProjectionsCSV.push_back(*bot);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);
		expressions::Expression* selBot = new expressions::RecordProjection(
				bot->getOriginalType(), arg, *bot);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				new BoolType(), selClassa, predExpr3);

		expressions::Expression* predicateNum = new expressions::AndExpression(
				new BoolType(), predicateAnd1, predicate3);

		Select *selNum = new Select(predicateNum, scanCSV);
		scanCSV->setParent(selNum);

		expressions::Expression* predExpr4 = new expressions::StringConstant(
				botName);
		predicateCSV = new expressions::EqExpression(new BoolType(), selBot,
				predExpr4);

		selCSV = new Select(predicateCSV, selNum);
		selNum->setParent(selCSV);
	}
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON2(map<string, dataset> datasetCatalog) {

	//bin
	float clusterHigh = 700;
	//csv
	int classaLow = 70;
	//json
	int sizeLow = 10000;


	RawContext ctx = prepareContext("symantec-bin-csv-json-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.p_event < 0.7
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);
	expressions::Expression* predExpr = new expressions::IntConstant(
			clusterHigh);
	predicateBin = new expressions::LtExpression(new BoolType(), selCluster, predExpr);

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
	 * sc.classa > 70
	 */
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr = new expressions::IntConstant(
				classaLow);
		predicateCSV = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'size')::int > 10000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				sizeLow);
		predicateJSON = new expressions::GtExpression(
				new BoolType(), selSize, predExpr1);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON2v1(map<string, dataset> datasetCatalog) {

	//bin
	float clusterHigh = 700;
	//csv
	int classaLow = 70;
	//json
	int sizeLow = 10000;


	RawContext ctx = prepareContext("symantec-bin-csv-json-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.p_event < 0.7
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);
	expressions::Expression* predExpr = new expressions::IntConstant(
			clusterHigh);
	predicateBin = new expressions::LtExpression(new BoolType(), selCluster, predExpr);

	Select *selBin = new Select(predicateBin, scanBin);
	scanBin->setParent(selBin);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'size')::int > 10000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				sizeLow);
		predicateJSON = new expressions::GtExpression(new BoolType(), selSize,
				predExpr1);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

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
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	vector<materialization_mode> outputModesRight;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinJSON";
	RadixJoin *joinBinJSON = new RadixJoin(joinPred, selBin, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinJSON);
	selJSON->setParent(joinBinJSON);

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
	 * sc.classa > 70
	 */
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr = new expressions::IntConstant(
				classaLow);
		predicateCSV = new expressions::GtExpression(new BoolType(), selClassa,
				predExpr);
	}
	Select *selCSV = new Select(predicateCSV, scanCSV);
	scanCSV->setParent(selCSV);

	/*
	 * JOIN II
	 */
	RadixJoin *joinCSV;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idCSV);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recCSV,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idCSV->getOriginalType(), rightArg2, *idCSV);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameJSON,
				activeLoop, pgJSON->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinCSV = new RadixJoin(joinPred2, joinBinJSON, selCSV, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinJSON->setParent(joinCSV);
	selCSV->setParent(joinCSV);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinCSV,
			&ctx);
	joinCSV->setParent(reduce);

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

void symantecBinCSVJSON3(map<string, dataset> datasetCatalog) {

	//bin
	float clusterLow = 800;
	//csv
	int classaLow = 80;
	//json
	int sizeLow = 10000;


	RawContext ctx = prepareContext("symantec-bin-csv-json-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.p_event < 0.7
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);
	expressions::Expression* predExpr = new expressions::IntConstant(
			clusterLow);
	predicateBin = new expressions::GtExpression(new BoolType(), selCluster, predExpr);

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
	 * sc.classa > 70
	 */
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr = new expressions::IntConstant(
				classaLow);
		predicateCSV = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'size')::int > 10000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				sizeLow);
		predicateJSON = new expressions::GtExpression(
				new BoolType(), selSize, predExpr1);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON4(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 140000000;
	int idHigh = 180000000;
	int clusterHigh = 20;

	RawContext ctx = prepareContext("symantec-bin-csv-json-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 20 and st.id > 7000000 and st.id < 9000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);

	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd1,
			predicate3);

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

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * sc.id > 7000000 and sc.id < 9000000
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
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		predicateCSV = predicateAnd1;
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int > 7000000 and (data->>'id')::int < 9000000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON5(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 140000000;
	int idHigh = 200000000;
	int clusterHigh = 30;
	//csv
	string botName1 = "Bobax";
	string botName2 = "UNCLASSIFIED";
	string botName3 = "Unclassified";
	//json
	int yearLow = 2010;

	RawContext ctx = prepareContext("symantec-bin-csv-json-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 30 and st.id > 7000000 and st.id < 10000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);

	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd1,
			predicate3);

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
	 * (bot = 'Bobax' OR bot = 'UNCLASSIFIED' or bot = 'Unclassified') and sc.id > 7000000  and sc.id < 10000000;
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*bot);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selBot = new expressions::RecordProjection(
				bot->getOriginalType(), arg, *bot);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		Select *selNum = new Select(predicateNum,scanCSV);
		scanCSV->setParent(selNum);

		expressions::Expression* predExprStr1 = new expressions::StringConstant(
				botName1);
		expressions::Expression* predExprStr2 = new expressions::StringConstant(
						botName2);
		expressions::Expression* predExprStr3 = new expressions::StringConstant(
						botName3);
		expressions::Expression* predicateStr1 = new expressions::EqExpression(
				new BoolType(), selBot, predExprStr1);
		expressions::Expression* predicateStr2 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr2);
		expressions::Expression* predicateStr3 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr3);
		expressions::Expression* predicateOr1 = new expressions::OrExpression(
				new BoolType(), predicateStr1, predicateStr2);

		predicateCSV = new expressions::OrExpression(new BoolType(),
				predicateOr1, predicateStr3);
		selCSV = new Select(predicateCSV, selNum);
		selNum->setParent(selCSV);
	}

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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *day = argsSymantecJSON["day"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(day);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->'day'->>'year')::int > 2010 and (data->>'id')::int > 7000000 and (data->>'id')::int < 10000000
	 */
	Select *selJSON;
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* selDay = new expressions::RecordProjection(
				day->getOriginalType(), arg, *day);
		IntType yearType = IntType();
		RecordAttribute *year = new RecordAttribute(1, fnameJSON, "year",
				&yearType);
		expressions::Expression* selYear = new expressions::RecordProjection(
				&yearType, selDay, *year);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);

		Select *selNum = new Select(predicateNum, scanJSON);
		scanJSON->setParent(selNum);

		expressions::Expression* predExprYear = new expressions::IntConstant(
				yearLow);
		expressions::Expression* predicateYear = new expressions::GtExpression(
				new BoolType(), selYear, predExprYear);

		selJSON = new Select(predicateYear, selNum);
		selNum->setParent(selJSON);
	}

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON6v1(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 140000000;
	int idHigh = 200000000;
	int clusterHigh = 30;
	//csv
	string botName1 = "FESTI";
	string botName2 = "UNCLASSIFIED";
	string botName3 = "GHEG";
	//json
	int yearLow = 2010;

	RawContext ctx = prepareContext("symantec-bin-csv-json-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 30 and st.id > 7000000 and st.id < 10000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);

	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd1,
			predicate3);

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
	 * (bot = 'Bobax' OR bot = 'UNCLASSIFIED' or bot = 'Unclassified') and sc.id > 7000000  and sc.id < 10000000;
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*bot);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selBot = new expressions::RecordProjection(
				bot->getOriginalType(), arg, *bot);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		Select *selNum = new Select(predicateNum,scanCSV);
		scanCSV->setParent(selNum);

		expressions::Expression* predExprStr1 = new expressions::StringConstant(
				botName1);
		expressions::Expression* predExprStr2 = new expressions::StringConstant(
						botName2);
		expressions::Expression* predExprStr3 = new expressions::StringConstant(
						botName3);
		expressions::Expression* predicateStr1 = new expressions::EqExpression(
				new BoolType(), selBot, predExprStr1);
		expressions::Expression* predicateStr2 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr2);
		expressions::Expression* predicateStr3 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr3);
		expressions::Expression* predicateOr1 = new expressions::OrExpression(
				new BoolType(), predicateStr1, predicateStr2);

		predicateCSV = new expressions::OrExpression(new BoolType(),
				predicateOr1, predicateStr3);
		selCSV = new Select(predicateCSV, selNum);
		selNum->setParent(selCSV);
	}

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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *day = argsSymantecJSON["day"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(day);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->'day'->>'year')::int > 2010 and (data->>'id')::int > 7000000 and (data->>'id')::int < 10000000
	 */
	Select *selJSON;
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* selDay = new expressions::RecordProjection(
				day->getOriginalType(), arg, *day);
		IntType yearType = IntType();
		RecordAttribute *year = new RecordAttribute(1, fnameJSON, "year",
				&yearType);
		expressions::Expression* selYear = new expressions::RecordProjection(
				&yearType, selDay, *year);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);

		Select *selNum = new Select(predicateNum, scanJSON);
		scanJSON->setParent(selNum);

		expressions::Expression* predExprYear = new expressions::IntConstant(
				yearLow);
		expressions::Expression* predicateYear = new expressions::GtExpression(
				new BoolType(), selYear, predExprYear);

		selJSON = new Select(predicateYear, selNum);
		selNum->setParent(selJSON);
	}

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON6(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 100000000;
	int idHigh = 200000000;
	int clusterLow = 200;
	//csv
	int classaHigh = 75;
	string botName1 = "Bobax";
	string botName2 = "UNCLASSIFIED";
	string botName3 = "Unclassified";
	//json
	int yearLow = 2010;

	RawContext ctx = prepareContext("symantec-bin-csv-json-6");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *cluster;
	RecordAttribute *dim;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	cluster = argsSymantecBin["cluster"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster > 200 and st.id > 5000000 and st.id < 10000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idLow);
	expressions::Expression* predExpr2 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::GtExpression(
			new BoolType(), selID, predExpr1);
	expressions::Expression* predicate2 = new expressions::LtExpression(
			new BoolType(), selID, predExpr2);
	expressions::Expression* predicateAnd1 = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterLow);
	expressions::Expression* predicate3 = new expressions::GtExpression(
			new BoolType(), selCluster, predExpr3);

	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd1,
			predicate3);

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
	RecordAttribute *bot;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projections;
//	projections.push_back(idCSV);
//	projections.push_back(classa);
	projections.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projections, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * (classA < 75 OR (bot = 'Bobax' OR bot = 'UNCLASSIFIED' or bot = 'Unclassified')) and sc.id > 5000000 and sc.id < 10000000
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*bot);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
						classa->getOriginalType(), arg, *classa);
		expressions::Expression* selBot = new expressions::RecordProjection(
				bot->getOriginalType(), arg, *bot);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		Select *selNum = new Select(predicateNum,scanCSV);
		scanCSV->setParent(selNum);


		expressions::Expression* predExpr3 = new expressions::IntConstant(
						classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
						new BoolType(), selClassa, predExpr3);

		expressions::Expression* predExprStr1 = new expressions::StringConstant(
				botName1);
		expressions::Expression* predExprStr2 = new expressions::StringConstant(
						botName2);
		expressions::Expression* predExprStr3 = new expressions::StringConstant(
						botName3);
		expressions::Expression* predicateStr1 = new expressions::EqExpression(
				new BoolType(), selBot, predExprStr1);
		expressions::Expression* predicateStr2 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr2);
		expressions::Expression* predicateStr3 = new expressions::EqExpression(
						new BoolType(), selBot, predExprStr3);
		expressions::Expression* predicateOr1 = new expressions::OrExpression(
				new BoolType(), predicateStr1, predicateStr2);
		expressions::Expression* predicateOr2 = new expressions::OrExpression(
						new BoolType(), predicateOr1, predicateStr3);

		predicateCSV = new expressions::OrExpression(new BoolType(),
				predicate3,predicateOr2);
		selCSV = new Select(predicateCSV, selNum);
		selNum->setParent(selCSV);
	}

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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
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
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *day = argsSymantecJSON["day"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(day);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->'day'->>'year')::int > 2010 and (data->>'id')::int > 5000000 and (data->>'id')::int < 10000000
	 */
	Select *selJSON;
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* selDay = new expressions::RecordProjection(
				day->getOriginalType(), arg, *day);
		IntType yearType = IntType();
		RecordAttribute *year = new RecordAttribute(1, fnameJSON, "year",
				&yearType);
		expressions::Expression* selYear = new expressions::RecordProjection(
				&yearType, selDay, *year);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateNum = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);

		Select *selNum = new Select(predicateNum, scanJSON);
		scanJSON->setParent(selNum);

		expressions::Expression* predExprYear = new expressions::IntConstant(
				yearLow);
		expressions::Expression* predicateYear = new expressions::GtExpression(
				new BoolType(), selYear, predExprYear);

		selJSON = new Select(predicateYear, selNum);
		selNum->setParent(selJSON);
	}

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */

	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), leftArg2, *dim);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

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
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed, joinJSON,
			&ctx);
	joinJSON->setParent(reduce);

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

void symantecBinCSVJSON7(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 2000000;

	RawContext ctx = prepareContext("symantec-bin-csv-json-7");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

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
	 * st.id < 100000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	predicateBin = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
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
	 * sc.id < 100000;
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

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		predicateCSV = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *uri = argsSymantecJSON["uri"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(uri);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int < 100000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		predicateJSON = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&recJSON,
			0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, joinJSON);
	joinJSON->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
	RecordAttribute recUnnested = RecordAttribute(2, fnameJSON + ".uri",
			nestedName, &nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
			&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * REDUCE
	 * MAX(size), MAX(classa), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*size);
	argProjections.push_back(*classa);
	expressions::Expression* exprClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), leftArg, *size);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = exprClassa;
	outputExprs.push_back(outputExpr2);

	accs.push_back(SUM);
	expressions::Expression* outputExpr3 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr3);

	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed,
			nullFilter, &ctx);
	nullFilter->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

//--7
//--NEW in v1: predicates
//select max(size) , max(classa), count(*)
//from (select (data->>'id')::int, json_array_elements((data->>'uri')::json), (data->>'size')::int AS size , classa
//      FROM symantecunordered st
//      JOIN spamsclasses400m sc ON (st.id = sc.id)
//	  JOIN spamscoreiddates28m sj ON (sc.id = (data->>'id')::int)
//      WHERE (data->>'id')::int < 10000000 and (data->>'size')::int > 5000
//            and sc.id < 10000000
//            and st.id < 10000000) internal;
void symantecBinCSVJSON7v1(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 200000000;
	//json
	int sizeLow = 5000;

	RawContext ctx = prepareContext("symantec-bin-csv-json-7");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

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
	 * st.id < 200000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	predicateBin = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
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
	 * sc.id < 200000000;
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

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		predicateCSV = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *uri = argsSymantecJSON["uri"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(uri);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				sizeLow);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selSize, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&recJSON,
			0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, joinJSON);
	joinJSON->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
	RecordAttribute recUnnested = RecordAttribute(2, fnameJSON + ".uri",
			nestedName, &nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
			&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * REDUCE
	 * MAX(size), MAX(classa), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*size);
	argProjections.push_back(*classa);
	expressions::Expression* exprClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);
	expressions::Expression* exprSize = new expressions::RecordProjection(
			size->getOriginalType(), leftArg, *size);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprSize;
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = exprClassa;
	outputExprs.push_back(outputExpr2);

	accs.push_back(SUM);
	expressions::Expression* outputExpr3 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr3);

	/* Pred: Redundant */

	expressions::Expression* lhsRed = new expressions::BoolConstant(true);
	expressions::Expression* rhsRed = new expressions::BoolConstant(true);
	expressions::Expression* predRed = new expressions::EqExpression(
			new BoolType(), lhsRed, rhsRed);

	opt::Reduce *reduce = new opt::Reduce(accs, outputExprs, predRed,
			nullFilter, &ctx);
	nullFilter->setParent(reduce);

	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	reduce->produce();
	ctx.prepareFunction(ctx.getGlobalFunction());
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Execution took %f seconds\n", diff(t0, t1));

	//Close all open files & clear
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecBinCSVJSON8(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 2000000;
	//csv
	int classaLow = 80;
	int classaHigh = 100;

	RawContext ctx = prepareContext("symantec-bin-csv-json-8");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

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
	 * st.id < 100000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	predicateBin = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
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
	 * sc.id < 100000;
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
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				classaLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				new BoolType(), selClassa, predExpr3);

		expressions::Expression* predicateAnd = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);
		predicateCSV = new expressions::AndExpression(new BoolType(),
				predicateAnd, predicate3);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *uri = argsSymantecJSON["uri"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(uri);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int < 100000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		predicateJSON = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&recJSON,
			0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, joinJSON);
	joinJSON->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
	RecordAttribute recUnnested = RecordAttribute(2, fnameJSON + ".uri",
			nestedName, &nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
			&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(size), count(*)
	 */
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*size);
	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			classa->getOriginalType(), nestArg, *classa);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(classa);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(size) */
	expressions::Expression* aggrSize = new expressions::RecordProjection(
			size->getOriginalType(), leftArg, *size);
	expressions::Expression* outputExpr1 = aggrSize;
	aggrField1 = string("_maxSize");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	expressions::Expression* aggrCount = new expressions::IntConstant(1);
	expressions::Expression* outputExpr2 = aggrCount;
	aggrField2 = string("_Cnt");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, nullFilter, nestLabel, *mat);
	nullFilter->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
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
	pgJSON->finish();
	rawCatalog.clear();
}

//--8
//--NEW in v1: predicates
//select classa, max(size), count(*)
//from (select (data->>'id')::int, json_array_elements((data->>'uri')::json), (data->>'size')::int AS size , classa
//      FROM symantecunordered st
//      JOIN spamsclasses400m sc ON (st.id = sc.id)
//	  JOIN spamscoreiddates28m sj ON (sc.id = (data->>'id')::int)
//      WHERE (data->>'id')::int < 10000000 and (data->>'size')::int > 5000
//            and sc.id < 10000000 and classa > 80 and classa < 100
//            and st.id < 10000000) internal
//group by classa;
void symantecBinCSVJSON8v1(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 200000000;
	//csv
	int classaLow = 80;
	int classaHigh = 100;
	//json
	int sizeLow = 5000;

	RawContext ctx = prepareContext("symantec-bin-csv-json-8");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

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
	 * st.id < idHigh
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	predicateBin = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);
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
	 * sc.id < idHigh;
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
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				classaLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				new BoolType(), selClassa, predExpr3);

		expressions::Expression* predicateAnd = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);
		predicateCSV = new expressions::AndExpression(new BoolType(),
				predicateAnd, predicate3);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];
	RecordAttribute *uri = argsSymantecJSON["uri"];
	RecordAttribute *size = argsSymantecJSON["size"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);
	projectionsJSON.push_back(uri);
	projectionsJSON.push_back(size);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				sizeLow);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selSize, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(&recJSON,
			0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			new BoolType(), lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, joinJSON);
	joinJSON->setParent(unnestOp);

	/*
	 * NULL FILTER!
	 * Acts as mini-nest operator
	 */
	StringType nestedType = StringType();
	RecordAttribute recUnnested = RecordAttribute(2, fnameJSON + ".uri",
			nestedName, &nestedType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	list<RecordAttribute> nullFilterProjections = list<RecordAttribute>();
	nullFilterProjections.push_back(recUnnested);

	expressions::InputArgument* nullFilterArg = new expressions::InputArgument(
			&nestedType, 0, nullFilterProjections);
	NullFilter *nullFilter = new NullFilter(nullFilterArg, unnestOp);
	unnestOp->setParent(nullFilter);

	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(size), count(*)
	 */
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*size);
	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			classa->getOriginalType(), nestArg, *classa);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(classa);
	outputModes.insert(outputModes.begin(), EAGER);

	Materializer* mat = new Materializer(fields, outputModes);

	char nestLabel[] = "nest_cluster";
	string aggrLabel = string(nestLabel);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	string aggrField1;
	string aggrField2;

	/* Aggregate 1: MAX(size) */
	expressions::Expression* aggrSize = new expressions::RecordProjection(
			size->getOriginalType(), leftArg, *size);
	expressions::Expression* outputExpr1 = aggrSize;
	aggrField1 = string("_maxSize");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	expressions::Expression* aggrCount = new expressions::IntConstant(1);
	expressions::Expression* outputExpr2 = aggrCount;
	aggrField2 = string("_Cnt");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, nullFilter, nestLabel, *mat);
	nullFilter->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
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
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecBinCSVJSON9(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 180000000;
	int idLow = 140000000;
	int clusterHigh = 100;
	//csv
	int classaLow = 100;
	int classaHigh = 120;

	RawContext ctx = prepareContext("symantec-bin-csv-json-8");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *dim;
	RecordAttribute *cluster;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	dim = argsSymantecBin["dim"];
	cluster = argsSymantecBin["cluster"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 100 and st.id > 7000000 and st.id < 9000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*cluster);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	expressions::Expression* predExpr2 = new expressions::IntConstant(idLow);
	expressions::Expression* predicate2 = new expressions::GtExpression(
			new BoolType(), selID, predExpr2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterHigh);
	expressions::Expression* predicate3 = new expressions::LtExpression(
			new BoolType(), selCluster, predExpr3);

	expressions::Expression* predicateAnd = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd, predicate3);

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
	 * classA > 100 and classA < 120 and sc.id > 7000000 and sc.id < 9000000
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
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaLow);
		expressions::Expression* predicate3 = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr3);

		expressions::Expression* predExpr4 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate4 = new expressions::LtExpression(
				new BoolType(), selClassa, predExpr4);

		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);
		expressions::Expression* predicateAnd2 = new expressions::AndExpression(
				new BoolType(), predicate3, predicate4);
		predicateCSV = new expressions::AndExpression(new BoolType(),
				predicateAnd1, predicateAnd2);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int > 7000000 and (data->>'id')::int < 9000000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(dim), count(*)
	 */
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*dim);
	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			classa->getOriginalType(), nestArg, *classa);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(dim);
	fields.push_back(classa);
	outputModes.insert(outputModes.begin(), EAGER);
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
			dim->getOriginalType(), leftArg, *dim);
	expressions::Expression* outputExpr1 = aggrDim;
	aggrField1 = string("_maxDim");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	expressions::Expression* aggrCount = new expressions::IntConstant(1);
	expressions::Expression* outputExpr2 = aggrCount;
	aggrField2 = string("_Cnt");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, joinJSON, nestLabel, *mat);
	joinJSON->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
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
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecBinCSVJSON10(map<string, dataset> datasetCatalog) {

	//bin
	int idHigh = 170000000;
	int idLow =  160000000;
	int clusterLow = 100;
	//csv
	int classaLow = 80;
	int classaHigh = 100;

	RawContext ctx = prepareContext("symantec-bin-csv-json-8");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecBin = string("symantecBin");
	dataset symantecBin = datasetCatalog[nameSymantecBin];
	map<string, RecordAttribute*> argsSymantecBin =
			symantecBin.recType.getArgsMap();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN BINARY FILE
	 */
	BinaryColPlugin *pgBin;
	Scan *scanBin;
	RecordAttribute *idBin;
	RecordAttribute *dim;
	RecordAttribute *cluster;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	dim = argsSymantecBin["dim"];
	cluster = argsSymantecBin["cluster"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(cluster);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.cluster < 100 and st.id > 7000000 and st.id < 9000000
	 */
	expressions::Expression* predicateBin;

	list<RecordAttribute> argProjectionsBin;
	argProjectionsBin.push_back(*idBin);
	argProjectionsBin.push_back(*cluster);
	argProjectionsBin.push_back(*dim);
	expressions::Expression* arg = new expressions::InputArgument(&recBin, 0,
			argProjectionsBin);
	expressions::Expression* selID = new expressions::RecordProjection(
			idBin->getOriginalType(), arg, *idBin);
	expressions::Expression* selCluster = new expressions::RecordProjection(
			cluster->getOriginalType(), arg, *cluster);

	expressions::Expression* predExpr1 = new expressions::IntConstant(idHigh);
	expressions::Expression* predicate1 = new expressions::LtExpression(
			new BoolType(), selID, predExpr1);

	expressions::Expression* predExpr2 = new expressions::IntConstant(idLow);
	expressions::Expression* predicate2 = new expressions::GtExpression(
			new BoolType(), selID, predExpr2);

	expressions::Expression* predExpr3 = new expressions::IntConstant(
			clusterLow);
	expressions::Expression* predicate3 = new expressions::GtExpression(
			new BoolType(), selCluster, predExpr3);

	expressions::Expression* predicateAnd = new expressions::AndExpression(
			new BoolType(), predicate1, predicate2);
	predicateBin = new expressions::AndExpression(new BoolType(), predicateAnd, predicate3);

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
	 * classA > 100 and classA < 120 and sc.id > 7000000 and sc.id < 9000000
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
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				classaLow);
		expressions::Expression* predicate3 = new expressions::GtExpression(
				new BoolType(), selClassa, predExpr3);

		expressions::Expression* predExpr4 = new expressions::IntConstant(
				classaHigh);
		expressions::Expression* predicate4 = new expressions::LtExpression(
				new BoolType(), selClassa, predExpr4);

		expressions::Expression* predicateAnd1 = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);
		expressions::Expression* predicateAnd2 = new expressions::AndExpression(
				new BoolType(), predicate3, predicate4);
		predicateCSV = new expressions::AndExpression(new BoolType(),
				predicateAnd1, predicateAnd2);
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
			new BoolType(), leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(idBin);
	fieldsLeft.push_back(dim);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
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
	vector<expressions::Expression*> expressionsLeft;
	expressionsLeft.push_back(exprLeftOID);
	expressionsLeft.push_back(exprLeftKey);

	Materializer* matLeft = new Materializer(fieldsLeft, expressionsLeft,
			OIDLeft, outputModesLeft);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight;
	fieldsRight.push_back(classa);
	vector<materialization_mode> outputModesRight;
	outputModesRight.insert(outputModesRight.begin(), EAGER);

	/* explicit mention to right OID */
	RecordAttribute *projTupleR = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDRight;
	OIDRight.push_back(projTupleR);
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), rightArg, *projTupleR);
	vector<expressions::Expression*> expressionsRight;
	expressionsRight.push_back(exprRightOID);

	Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
			OIDRight, outputModesRight);

	char joinLabel[] = "radixJoinBinCSV";
	RadixJoin *joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
			*matLeft, *matRight);
	selBin->setParent(joinBinCSV);
	selCSV->setParent(joinBinCSV);

	/**
	 * SCAN JSON FILE
	 */
	string fnameJSON = symantecJSON.path;
	RecordType recJSON = symantecJSON.recType;
	int linehintJSON = symantecJSON.linehint;

	RecordAttribute *idJSON = argsSymantecJSON["id"];

	vector<RecordAttribute*> projectionsJSON;
	projectionsJSON.push_back(idJSON);

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);

	/*
	 * SELECT JSON
	 * (data->>'id')::int > 7000000 and (data->>'id')::int < 9000000
	 */
	expressions::Expression* predicateJSON;
	{
		list<RecordAttribute> argProjectionsJSON;
		argProjectionsJSON.push_back(*idJSON);
		expressions::Expression* arg = new expressions::InputArgument(&recJSON,
				0, argProjectionsJSON);
		expressions::Expression* selID = new expressions::RecordProjection(
				idJSON->getOriginalType(), arg, *idJSON);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				new BoolType(), selID, predExpr1);

		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				new BoolType(), selID, predExpr2);
		predicateJSON = new expressions::AndExpression(new BoolType(),
				predicate1, predicate2);
	}
	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN II
	 */
	RadixJoin *joinJSON;
	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft2;
	argProjectionsLeft2.push_back(*idBin);
	argProjectionsLeft2.push_back(*dim);
	argProjectionsLeft2.push_back(*classa);
	expressions::Expression* leftArg2 = new expressions::InputArgument(&recBin, 0,
			argProjectionsLeft2);
	expressions::Expression* leftPred2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight2;
	argProjectionsRight2.push_back(*idJSON);
	expressions::Expression* rightArg2 = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight2);
	expressions::Expression* rightPred2 = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg2, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(
			new BoolType(), leftPred2, rightPred2);

	/* explicit mention to left OIDs */
	RecordAttribute *projTupleL2a = new RecordAttribute(fnamePrefixBin,
			activeLoop, pgBin->getOIDType());
	RecordAttribute *projTupleL2b = new RecordAttribute(fnameCSV,
				activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft2;
	OIDLeft2.push_back(projTupleL2a);
	OIDLeft2.push_back(projTupleL2b);
	expressions::Expression* exprLeftOID2a = new expressions::RecordProjection(
			pgBin->getOIDType(), leftArg2, *projTupleL2a);
	expressions::Expression* exprLeftOID2b = new expressions::RecordProjection(
				pgBin->getOIDType(), leftArg2, *projTupleL2b);
	expressions::Expression* exprLeftKey2 = new expressions::RecordProjection(
			idBin->getOriginalType(), leftArg2, *idBin);
	vector<expressions::Expression*> expressionsLeft2;
	expressionsLeft2.push_back(exprLeftOID2a);
	expressionsLeft2.push_back(exprLeftOID2b);
	expressionsLeft2.push_back(exprLeftKey2);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft2;
	fieldsLeft2.push_back(dim);
	fieldsLeft2.push_back(classa);
	vector<materialization_mode> outputModesLeft2;
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);
	outputModesLeft2.insert(outputModesLeft2.begin(), EAGER);

	Materializer* matLeft2 = new Materializer(fieldsLeft2, expressionsLeft2,
			OIDLeft2, outputModesLeft2);

	/* right materializer - no explicit field needed */
	vector<RecordAttribute*> fieldsRight2;
	vector<materialization_mode> outputModesRight2;

	/* explicit mention to right OID */
	RecordAttribute *projTupleR2 = new RecordAttribute(fnameJSON, activeLoop,
			pgJSON->getOIDType());
	vector<RecordAttribute*> OIDRight2;
	OIDRight2.push_back(projTupleR2);
	expressions::Expression* exprRightOID2 = new expressions::RecordProjection(
			pgJSON->getOIDType(), rightArg2, *projTupleR2);
	vector<expressions::Expression*> expressionsRight2;
	expressionsRight2.push_back(exprRightOID2);

	Materializer* matRight2 = new Materializer(fieldsRight2, expressionsRight2,
			OIDRight2, outputModesRight2);

	char joinLabel2[] = "radixJoinIntermediateJSON";
	joinJSON = new RadixJoin(joinPred2, joinBinCSV, selJSON, &ctx, joinLabel2,
			*matLeft2, *matRight2);
	joinBinCSV->setParent(joinJSON);
	selJSON->setParent(joinJSON);


	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(dim), count(*)
	 */
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*dim);
	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY cluster
	expressions::RecordProjection* f = new expressions::RecordProjection(
			classa->getOriginalType(), nestArg, *classa);
	//p
	expressions::Expression* lhsNest = new expressions::BoolConstant(true);
	expressions::Expression* rhsNest = new expressions::BoolConstant(true);
	expressions::Expression* predNest = new expressions::EqExpression(
			new BoolType(), lhsNest, rhsNest);

	//mat.
	vector<RecordAttribute*> fields;
	vector<materialization_mode> outputModes;
	fields.push_back(dim);
	fields.push_back(classa);
	outputModes.insert(outputModes.begin(), EAGER);
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
			dim->getOriginalType(), leftArg, *dim);
	expressions::Expression* outputExpr1 = aggrDim;
	aggrField1 = string("_maxDim");
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr1);
	aggrLabels.push_back(aggrField1);

	expressions::Expression* aggrCount = new expressions::IntConstant(1);
	expressions::Expression* outputExpr2 = aggrCount;
	aggrField2 = string("_Cnt");
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr2);
	aggrLabels.push_back(aggrField2);

	radix::Nest *nestOp = new radix::Nest(&ctx, accs, outputExprs, aggrLabels,
			predNest, f, f, joinJSON, nestLabel, *mat);
	joinJSON->setParent(nestOp);

	Function* debugInt = ctx.getFunction("printi");
	IntType intType = IntType();

	/* OUTPUT */
	RawOperator *lastPrintOp;
	RecordAttribute *toOutput1 = new RecordAttribute(1, aggrLabel, aggrField1,
			&intType);
	expressions::RecordProjection* nestOutput1 =
			new expressions::RecordProjection(&intType, nestArg, *toOutput1);
	Print *printOp1 = new Print(debugInt, nestOutput1, nestOp);
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
	pgJSON->finish();
	rawCatalog.clear();
}
