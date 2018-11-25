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

#include "experiments/realworld-vldb/spam-csv-json.hpp"

void symantecCSVJSON1(map<string, dataset> datasetCatalog) {

	//csv
	int idLow = 100000000;
	int idHigh = 200000000;
	//string botName = "Bobax";
	string botName = "GHEG";
	//JSON
	//id, again

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-1");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordAttribute *classb;
	RecordAttribute *bot;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	classb = argsSymantecCSV["classb"];
	bot = argsSymantecCSV["bot"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);
//	projectionsCSV.push_back(classb);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * id > 5000000 and id < 10000000 and bot = 'Bobax'
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

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
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

		selCSV = new Select(predicateStr,selNum);
		selNum->setParent(selCSV);
	}

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
	 * (data->>'id')::int > 5000000 and (data->>'id')::int < 10000000
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
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		predicateJSON = new expressions::AndExpression(
				predicate1, predicate2);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*classb);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	fieldsLeft.push_back(classb);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV,
			activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/**
	 * REDUCE
	 * MAX(classa), MAX(classb), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*classa);
	argProjections.push_back(*classb);
	expressions::Expression* exprClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);
	expressions::Expression* exprClassb = new expressions::RecordProjection(
				classb->getOriginalType(), leftArg, *classb);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprClassa;
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = exprClassb;
	outputExprs.push_back(outputExpr2);

	accs.push_back(SUM);
	expressions::Expression* outputExpr3 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr3);
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
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecCSVJSON2(map<string, dataset> datasetCatalog) {

	//csv
	int idLow =  160000000;
	int idHigh = 200000000;
//	string botName = "Bobax";
	string botName = "BAGLE-CB";

	//JSON
	//id, again
	int sizeLow = 1000;

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-2");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordAttribute *classb;
	RecordAttribute *bot;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	classb = argsSymantecCSV["classb"];
	bot = argsSymantecCSV["bot"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);
//	projectionsCSV.push_back(classb);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * id > 5000000 and id < 10000000 and bot = 'Bobax'
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

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
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

		selCSV = new Select(predicateStr,selNum);
		selNum->setParent(selCSV);
	}

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

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);


	/*
	 * SELECT JSON
	 * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000 and (data->>'size')::int > 1000
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
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
						sizeLow);
		expressions::Expression* predicate3 = new expressions::GtExpression(
						selSize, predExpr3);

		predicateJSON = new expressions::AndExpression(
						predicateAnd, predicate3);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*classb);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	fieldsLeft.push_back(classb);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV,
			activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/**
	 * REDUCE
	 * MAX(classa), MAX(classb), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*classa);
	argProjections.push_back(*classb);
	expressions::Expression* exprClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);
	expressions::Expression* exprClassb = new expressions::RecordProjection(
				classb->getOriginalType(), leftArg, *classb);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprClassa;
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = exprClassb;
	outputExprs.push_back(outputExpr2);

	accs.push_back(SUM);
	expressions::Expression* outputExpr3 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr3);
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
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecCSVJSON3(map<string, dataset> datasetCatalog) {

	//csv
	int idLow = 160000000;
	int idHigh = 200000000;
//	string botName = "Unclassified";
	string botName = "FESTI";
	//JSON
	//id, again
	int yearNo = 2012;

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordAttribute *classb;
	RecordAttribute *bot;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	classb = argsSymantecCSV["classb"];
	bot = argsSymantecCSV["bot"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);
//	projectionsCSV.push_back(classb);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * id > 5000000 and id < 10000000 and bot = 'Unclassified'
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

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
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

		selCSV = new Select(predicateStr,selNum);
		selNum->setParent(selCSV);
	}

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

	ListType *documentType = new ListType(recJSON);
	jsonPipelined::JSONPlugin *pgJSON = new jsonPipelined::JSONPlugin(&ctx,
			fnameJSON, documentType, linehintJSON);
	rawCatalog.registerPlugin(fnameJSON, pgJSON);
	Scan *scanJSON = new Scan(&ctx, *pgJSON);


	/*
	 * SELECT JSON
	 * (data->>'id')::int > 8000000 and (data->>'id')::int < 10000000 and (data->'day'->>'year')::int = 2012
	 */
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
		RecordAttribute *year = new RecordAttribute(2, fnameJSON, "year",
				&yearType);
		expressions::Expression* selYear = new expressions::RecordProjection(
				&yearType, selDay, *year);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
						yearNo);
		expressions::Expression* predicate3 = new expressions::EqExpression(
						selYear, predExpr3);

		predicateJSON = new expressions::AndExpression(
						predicateAnd, predicate3);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*classb);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	fieldsLeft.push_back(classb);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV,
			activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/**
	 * REDUCE
	 * MAX(classa), MAX(classb), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*classa);
	argProjections.push_back(*classb);
	expressions::Expression* exprClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);
	expressions::Expression* exprClassb = new expressions::RecordProjection(
				classb->getOriginalType(), leftArg, *classb);
	/* Output: */
	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;

	accs.push_back(MAX);
	expressions::Expression* outputExpr1 = exprClassa;
	outputExprs.push_back(outputExpr1);

	accs.push_back(MAX);
	expressions::Expression* outputExpr2 = exprClassb;
	outputExprs.push_back(outputExpr2);

	accs.push_back(SUM);
	expressions::Expression* outputExpr3 = new expressions::IntConstant(1);
	outputExprs.push_back(outputExpr3);
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
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecCSVJSON4(map<string, dataset> datasetCatalog) {

	//csv
	int idLow =  180000000;
	int idHigh = 200000000;
	//string botName = "Unclassified";
	string botName = "FESTI";
	int classaHigh = 5;
	//JSON
	//id, again
	int yearNo = 2012;

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-3");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordAttribute *classb;
	RecordAttribute *bot;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	classb = argsSymantecCSV["classb"];
	bot = argsSymantecCSV["bot"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);
//	projectionsCSV.push_back(classb);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);


	/*
	 * SELECT CSV
	 * id > 9000000 and id < 10000000 and bot = 'Unclassified' and classa < 5
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

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predExpr3 = new expressions::IntConstant(
						classaHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicate3 = new expressions::LtExpression(
						selClassa, predExpr3);

		expressions::Expression* predicateNum_ = new expressions::AndExpression(
				predicate1, predicate2);
		expressions::Expression* predicateNum = new expressions::AndExpression(
						predicateNum_, predicate3);

		Select *selNum = new Select(predicateNum, scanCSV);
		scanCSV->setParent(selNum);

		expressions::Expression* predExpr4 = new expressions::StringConstant(
						botName);
		expressions::Expression* predicateStr = new expressions::EqExpression(
				selBot, predExpr4);

		selCSV = new Select(predicateStr,selNum);
		selNum->setParent(selCSV);
	}

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
	 * (data->>'id')::int > 9000000 and (data->>'id')::int < 10000000 and (data->'day'->>'year')::int = 2012
	 */
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
		RecordAttribute *year = new RecordAttribute(2, fnameJSON, "year",
				&yearType);
		expressions::Expression* selYear = new expressions::RecordProjection(
				&yearType, selDay, *year);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				selID, predExpr2);
		expressions::Expression* predicateAnd = new expressions::AndExpression(
				predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
						yearNo);
		expressions::Expression* predicate3 = new expressions::EqExpression(
						selYear, predExpr3);

		predicateJSON = new expressions::AndExpression(
						predicateAnd, predicate3);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*classb);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	fieldsLeft.push_back(classb);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);

	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV,
			activeLoop, pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/**
	 * NEST
	 * GroupBy: classa
	 * Pred: Redundant (true == true)
	 * 		-> I wonder whether it gets statically removed..
	 * Output: max(classb), count(*)
	 */
	expressions::Expression* selClassa = new expressions::RecordProjection(
			classa->getOriginalType(), leftArg, *classa);

	list<RecordAttribute> nestProjections;
	nestProjections.push_back(*classa);
	nestProjections.push_back(*classb);
	expressions::Expression* nestArg = new expressions::InputArgument(&recCSV,
			0, nestProjections);

	//f (& g) -> GROUPBY cluster
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
	fields.push_back(classb);
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

	/* Aggregate 1: MAX(classb) */
	expressions::Expression* aggrSize = new expressions::RecordProjection(
			classb->getOriginalType(), leftArg, *classb);
	expressions::Expression* outputExpr1 = aggrSize;
	aggrField1 = string("_maxClassb");
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
	pgCSV->finish();
	pgJSON->finish();
	rawCatalog.clear();
}

void symantecCSVJSON5(map<string, dataset> datasetCatalog) {

	//csv
	int idHigh = 2000000;
	//JSON
	//id, again

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * id < 100000
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		predicateCSV = new expressions::LtExpression(selID,
				predExpr1);

		selCSV = new Select(predicateCSV, scanCSV);
		scanCSV->setParent(selCSV);
	}

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
		predicateJSON = new expressions::LtExpression(selID,
				predExpr1);

	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*size);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);


	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(
			&recJSON, 0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, join);
	join->setParent(unnestOp);

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
			lhsRed, rhsRed);

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

void symantecCSVJSON5v1(map<string, dataset> datasetCatalog) {

	//csv
	int idHigh = 200000000;
	//JSON
	//id, again
	int sizeLow = 5000;

	RawContext& ctx = *prepareContext("symantec-CSV-JSON-5");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantecCSV = string("symantecCSV");
	dataset symantecCSV = datasetCatalog[nameSymantecCSV];
	map<string, RecordAttribute*> argsSymantecCSV =
			symantecCSV.recType.getArgsMap();

	string nameSymantecJSON = string("symantecIDDates");
	dataset symantecJSON = datasetCatalog[nameSymantecJSON];
	map<string, RecordAttribute*> argsSymantecJSON =
			symantecJSON.recType.getArgsMap();

	/**
	 * SCAN CSV FILE
	 */
	pm::CSVPlugin* pgCSV;
	Scan *scanCSV;
	RecordAttribute *idCSV;
	RecordAttribute *classa;
	RecordType recCSV = symantecCSV.recType;
	string fnameCSV = symantecCSV.path;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';
	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	vector<RecordAttribute*> projectionsCSV;
//	projectionsCSV.push_back(idCSV);
//	projectionsCSV.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV,
			delimInner, linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * id < 100000
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selID = new expressions::RecordProjection(
				idCSV->getOriginalType(), arg, *idCSV);

		//id
		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		predicateCSV = new expressions::LtExpression(selID,
				predExpr1);

		selCSV = new Select(predicateCSV, scanCSV);
		scanCSV->setParent(selCSV);
	}

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
		expressions::Expression* selSize = new expressions::RecordProjection(
				size->getOriginalType(), arg, *size);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				sizeLow);
		expressions::Expression* predicate1 = new expressions::LtExpression(
				selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::GtExpression(
				selSize, predExpr2);
		predicateJSON = new expressions::AndExpression(predicate1, predicate2);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	/*
	 * JOIN
	 * st.id = sj.id
	 */

	//LEFT SIDE
	list<RecordAttribute> argProjectionsLeft;
	argProjectionsLeft.push_back(*idCSV);
	argProjectionsLeft.push_back(*classa);
	argProjectionsLeft.push_back(*size);
	expressions::Expression* leftArg = new expressions::InputArgument(&recCSV,
			0, argProjectionsLeft);
	expressions::Expression* leftPred = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);

	//RIGHT SIDE
	list<RecordAttribute> argProjectionsRight;
	argProjectionsRight.push_back(*idJSON);
	expressions::Expression* rightArg = new expressions::InputArgument(&recJSON,
			1, argProjectionsRight);
	expressions::Expression* rightPred = new expressions::RecordProjection(
			idJSON->getOriginalType(), rightArg, *idJSON);

	/* join pred. */
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			leftPred, rightPred);

	/* left materializer - dim needed */
	vector<RecordAttribute*> fieldsLeft;
	fieldsLeft.push_back(classa);
	vector<materialization_mode> outputModesLeft;
	outputModesLeft.insert(outputModesLeft.begin(), EAGER);


	/* explicit mention to left OID */
	RecordAttribute *projTupleL = new RecordAttribute(fnameCSV, activeLoop,
			pgCSV->getOIDType());
	vector<RecordAttribute*> OIDLeft;
	OIDLeft.push_back(projTupleL);
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			pgCSV->getOIDType(), leftArg, *projTupleL);
	expressions::Expression* exprLeftKey = new expressions::RecordProjection(
			idCSV->getOriginalType(), leftArg, *idCSV);
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

	char joinLabel[] = "radixJoinCSVJSON";
	RadixJoin *join = new RadixJoin(joinPred, selCSV, selJSON, &ctx, joinLabel,
			*matLeft, *matRight);
	selCSV->setParent(join);
	selJSON->setParent(join);

	/* OUTER Unnest -> some entries have no URIs */
	list<RecordAttribute> unnestProjections = list<RecordAttribute>();
	unnestProjections.push_back(*uri);

	expressions::Expression* outerArg = new expressions::InputArgument(
			&recJSON, 0, unnestProjections);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			uri->getOriginalType(), outerArg, *uri);
	string nestedName = activeLoop;
	Path path = Path(nestedName, proj);

	expressions::Expression* lhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* rhsUnnest = new expressions::BoolConstant(true);
	expressions::Expression* predUnnest = new expressions::EqExpression(
			lhsUnnest, rhsUnnest);

	OuterUnnest *unnestOp = new OuterUnnest(predUnnest, path, join);
	join->setParent(unnestOp);

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
			lhsRed, rhsRed);

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
