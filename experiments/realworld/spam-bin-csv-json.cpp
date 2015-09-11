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

void symantecCSVWarmup(map<string,dataset> datasetCatalog);
void symantecJSONWarmup(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON1(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON2(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON3(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON4(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON5(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON6(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON7(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON8(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON9(map<string,dataset> datasetCatalog);
void symantecBinCSVJSON10(map<string,dataset> datasetCatalog);


RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}


int main()	{
	cout << "Execution" << endl;
	map<string,dataset> datasetCatalog;
	symantecBinSchema(datasetCatalog);
	symantecCSVSchema(datasetCatalog);
	symantecCoreIDDatesSchema(datasetCatalog);

	symantecCSVWarmup(datasetCatalog);
	symantecJSONWarmup(datasetCatalog);

	cout << "SYMANTEC BIN-CSV-JSON 1" << endl;
	symantecBinCSVJSON1(datasetCatalog);
	cout << "SYMANTEC BIN-CSV-JSON 2" << endl;
	symantecBinCSVJSON2(datasetCatalog);
	cout << "SYMANTEC BIN-CSV-JSON 3" << endl;
	symantecBinCSVJSON3(datasetCatalog);
	cout << "SYMANTEC BIN-CSV-JSON 4" << endl;
	symantecBinCSVJSON4(datasetCatalog);
//	cout << "SYMANTEC BIN-CSV-JSON 5" << endl;
//	symantecBinCSVJSON5(datasetCatalog);
}

void symantecBinCSVJSON1(map<string, dataset> datasetCatalog) {

	//bin
	int idLow = 8000000;
	int idHigh = 10000000;
	int clusterHigh = 200;
	//csv
	int classaHigh = 90;
	string botName = "Bobax";
	//json
	//id

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
	 * st.cluster < 200 and st.id > 8000000 and st.id < 10000000;
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		argProjectionsBin.push_back(*cluster);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);
		expressions::Expression* selCluster = new expressions::RecordProjection(
				cluster->getOriginalType(), arg, *cluster);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateAnd = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				clusterHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				new BoolType(), selCluster, predExpr3);

		predicateBin = new expressions::AndExpression(new BoolType(),
				predicateAnd, predicate3);
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
	RecordAttribute *bot;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projectionsCSV;
	projectionsCSV.push_back(idCSV);
	projectionsCSV.push_back(classa);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
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
	RadixJoin *joinBinCSV;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idCSV);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recCSV, 1, argProjectionsRight);
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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgCSV->getOIDType(), rightArg,
						*projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinBinCSV";
		joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
				*matLeft, *matRight);
		selBin->setParent(joinBinCSV);
		selCSV->setParent(joinBinCSV);
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
		predicateJSON = new expressions::LtExpression(new BoolType(), selID,
				predExpr1);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	RadixJoin *joinJSON;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idJSON);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recJSON, 1, argProjectionsRight);
		expressions::Expression* rightPred = new expressions::RecordProjection(
				idJSON->getOriginalType(), rightArg, *idJSON);

		/* join pred. */
		expressions::BinaryExpression* joinPred = new expressions::EqExpression(
				new BoolType(), leftPred, rightPred);

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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgJSON->getOIDType(),
						rightArg, *projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinIntermediateJSON";
		joinJSON = new RadixJoin(joinPred, joinBinCSV, selJSON, &ctx, joinLabel,
				*matLeft, *matRight);
		joinBinCSV->setParent(joinJSON);
		selJSON->setParent(joinJSON);
	}

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* inputArg = new expressions::InputArgument(&recBin,
			0, argProjections);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), inputArg, *dim);
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
	float p_eventHigh = 0.7;
	//csv
	int classaLow = 70;
	//json
	int sizeLow = 1000;
	//id

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
	RecordAttribute *dim;
	RecordAttribute *p_event;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	p_event = argsSymantecBin["p_event"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(p_event);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.p_event < 0.7
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*p_event);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selEvent = new expressions::RecordProjection(
				p_event->getOriginalType(), arg, *p_event);

		expressions::Expression* predExpr1 = new expressions::FloatConstant(
				p_eventHigh);
		predicateBin = new expressions::LtExpression(new BoolType(), selEvent,
				predExpr1);
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

	vector<RecordAttribute*> projectionsCSV;
	projectionsCSV.push_back(idCSV);
	projectionsCSV.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * sc.classa > 70
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				classaLow);

		predicateCSV = new expressions::GtExpression(new BoolType(), selClassa,
				predExpr1);

		selCSV = new Select(predicateCSV, scanCSV);
		scanCSV->setParent(selCSV);
	}

	/*
	 * JOIN
	 * st.id = sc.id
	 */
	RadixJoin *joinBinCSV;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idCSV);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recCSV, 1, argProjectionsRight);
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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgCSV->getOIDType(), rightArg,
						*projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinBinCSV";
		joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
				*matLeft, *matRight);
		selBin->setParent(joinBinCSV);
		selCSV->setParent(joinBinCSV);
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

	RadixJoin *joinJSON;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idJSON);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recJSON, 1, argProjectionsRight);
		expressions::Expression* rightPred = new expressions::RecordProjection(
				idJSON->getOriginalType(), rightArg, *idJSON);

		/* join pred. */
		expressions::BinaryExpression* joinPred = new expressions::EqExpression(
				new BoolType(), leftPred, rightPred);

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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgJSON->getOIDType(),
						rightArg, *projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinIntermediateJSON";
		joinJSON = new RadixJoin(joinPred, joinBinCSV, selJSON, &ctx, joinLabel,
				*matLeft, *matRight);
		joinBinCSV->setParent(joinJSON);
		selJSON->setParent(joinJSON);
	}

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* inputArg = new expressions::InputArgument(&recBin,
			0, argProjections);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), inputArg, *dim);
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


void symantecBinCSVJSON3(map<string, dataset> datasetCatalog) {

	//bin
	float p_eventLow = 0.8;
	//csv
	int classaLow = 80;
	//json
	int sizeLow = 1000;
	//id

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
	RecordAttribute *dim;
	RecordAttribute *p_event;
	RecordType recBin = symantecBin.recType;
	string fnamePrefixBin = symantecBin.path;
	int linehintBin = symantecBin.linehint;
	idBin = argsSymantecBin["id"];
	p_event = argsSymantecBin["p_event"];
	dim = argsSymantecBin["dim"];
	vector<RecordAttribute*> projectionsBin;
	projectionsBin.push_back(idBin);
	projectionsBin.push_back(p_event);
	projectionsBin.push_back(dim);

	pgBin = new BinaryColPlugin(&ctx, fnamePrefixBin, recBin, projectionsBin);
	rawCatalog.registerPlugin(fnamePrefixBin, pgBin);
	scanBin = new Scan(&ctx, *pgBin);

	/*
	 * SELECT BINARY
	 * st.p_event > 0.8
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*p_event);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selEvent = new expressions::RecordProjection(
				p_event->getOriginalType(), arg, *p_event);

		expressions::Expression* predExpr1 = new expressions::FloatConstant(p_eventLow);
		predicateBin = new expressions::GtExpression(new BoolType(), selEvent,
						predExpr1);
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

	vector<RecordAttribute*> projectionsCSV;
	projectionsCSV.push_back(idCSV);
	projectionsCSV.push_back(classa);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
			linehintCSV, policy, false);
	rawCatalog.registerPlugin(fnameCSV, pgCSV);
	scanCSV = new Scan(&ctx, *pgCSV);

	/*
	 * SELECT CSV
	 * sc.classa > 80
	 */
	Select *selCSV;
	expressions::Expression* predicateCSV;
	{
		list<RecordAttribute> argProjectionsCSV;
		argProjectionsCSV.push_back(*idCSV);
		argProjectionsCSV.push_back(*classa);
		expressions::Expression* arg = new expressions::InputArgument(&recCSV,
				0, argProjectionsCSV);
		expressions::Expression* selClassa = new expressions::RecordProjection(
				classa->getOriginalType(), arg, *classa);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				classaLow);

		predicateCSV = new expressions::GtExpression(new BoolType(), selClassa,
				predExpr1);

		selCSV = new Select(predicateCSV, scanCSV);
		scanCSV->setParent(selCSV);
	}

	/*
	 * JOIN
	 * st.id = sc.id
	 */
	RadixJoin *joinBinCSV;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idCSV);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recCSV, 1, argProjectionsRight);
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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgCSV->getOIDType(), rightArg,
						*projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinBinCSV";
		joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
				*matLeft, *matRight);
		selBin->setParent(joinBinCSV);
		selCSV->setParent(joinBinCSV);
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

	RadixJoin *joinJSON;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idJSON);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recJSON, 1, argProjectionsRight);
		expressions::Expression* rightPred = new expressions::RecordProjection(
				idJSON->getOriginalType(), rightArg, *idJSON);

		/* join pred. */
		expressions::BinaryExpression* joinPred = new expressions::EqExpression(
				new BoolType(), leftPred, rightPred);

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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgJSON->getOIDType(),
						rightArg, *projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinIntermediateJSON";
		joinJSON = new RadixJoin(joinPred, joinBinCSV, selJSON, &ctx, joinLabel,
				*matLeft, *matRight);
		joinBinCSV->setParent(joinJSON);
		selJSON->setParent(joinJSON);
	}

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* inputArg = new expressions::InputArgument(&recBin,
			0, argProjections);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), inputArg, *dim);
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
	int idLow = 8000000;
	int idHigh = 10000000;
	int clusterHigh = 200;
	//csv
	int classaHigh = 90;
	string botName = "Bobax";
	//json
	//id

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
	 * st.cluster < 200 and st.id > 8000000 and st.id < 10000000;
	 */
	expressions::Expression* predicateBin;
	{
		list<RecordAttribute> argProjectionsBin;
		argProjectionsBin.push_back(*idBin);
		argProjectionsBin.push_back(*cluster);
		expressions::Expression* arg = new expressions::InputArgument(&recBin,
				0, argProjectionsBin);
		expressions::Expression* selID = new expressions::RecordProjection(
				idBin->getOriginalType(), arg, *idBin);
		expressions::Expression* selCluster = new expressions::RecordProjection(
				cluster->getOriginalType(), arg, *cluster);

		expressions::Expression* predExpr1 = new expressions::IntConstant(
				idLow);
		expressions::Expression* predExpr2 = new expressions::IntConstant(
				idHigh);
		expressions::Expression* predicate1 = new expressions::GtExpression(
				new BoolType(), selID, predExpr1);
		expressions::Expression* predicate2 = new expressions::LtExpression(
				new BoolType(), selID, predExpr2);
		expressions::Expression* predicateAnd = new expressions::AndExpression(
				new BoolType(), predicate1, predicate2);

		expressions::Expression* predExpr3 = new expressions::IntConstant(
				clusterHigh);
		expressions::Expression* predicate3 = new expressions::LtExpression(
				new BoolType(), selCluster, predExpr3);

		predicateBin = new expressions::AndExpression(new BoolType(),
				predicateAnd, predicate3);
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
	RecordAttribute *bot;
	int linehintCSV = symantecCSV.linehint;
	int policy = 5;
	char delimInner = ';';

	idCSV = argsSymantecCSV["id"];
	classa = argsSymantecCSV["classa"];
	bot = argsSymantecCSV["bot"];

	vector<RecordAttribute*> projectionsCSV;
	projectionsCSV.push_back(idCSV);
	projectionsCSV.push_back(classa);
	projectionsCSV.push_back(bot);

	pgCSV = new pm::CSVPlugin(&ctx, fnameCSV, recCSV, projectionsCSV, delimInner,
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
	RadixJoin *joinBinCSV;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idCSV);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recCSV, 1, argProjectionsRight);
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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgCSV->getOIDType(), rightArg,
						*projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinBinCSV";
		joinBinCSV = new RadixJoin(joinPred, selBin, selCSV, &ctx, joinLabel,
				*matLeft, *matRight);
		selBin->setParent(joinBinCSV);
		selCSV->setParent(joinBinCSV);
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
		predicateJSON = new expressions::LtExpression(new BoolType(), selID,
				predExpr1);
	}

	Select *selJSON = new Select(predicateJSON, scanJSON);
	scanJSON->setParent(selJSON);

	RadixJoin *joinJSON;
	{
		//LEFT SIDE
		list<RecordAttribute> argProjectionsLeft;
		argProjectionsLeft.push_back(*idBin);
		argProjectionsLeft.push_back(*dim);
		expressions::Expression* leftArg = new expressions::InputArgument(
				&recBin, 0, argProjectionsLeft);
		expressions::Expression* leftPred = new expressions::RecordProjection(
				idBin->getOriginalType(), leftArg, *idBin);

		//RIGHT SIDE
		list<RecordAttribute> argProjectionsRight;
		argProjectionsRight.push_back(*idJSON);
		expressions::Expression* rightArg = new expressions::InputArgument(
				&recJSON, 1, argProjectionsRight);
		expressions::Expression* rightPred = new expressions::RecordProjection(
				idJSON->getOriginalType(), rightArg, *idJSON);

		/* join pred. */
		expressions::BinaryExpression* joinPred = new expressions::EqExpression(
				new BoolType(), leftPred, rightPred);

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
		expressions::Expression* exprLeftOID =
				new expressions::RecordProjection(pgBin->getOIDType(), leftArg,
						*projTupleL);
		expressions::Expression* exprLeftKey =
				new expressions::RecordProjection(idBin->getOriginalType(),
						leftArg, *idBin);
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
		expressions::Expression* exprRightOID =
				new expressions::RecordProjection(pgJSON->getOIDType(),
						rightArg, *projTupleR);
		vector<expressions::Expression*> expressionsRight;
		expressionsRight.push_back(exprRightOID);

		Materializer* matRight = new Materializer(fieldsRight, expressionsRight,
				OIDRight, outputModesRight);

		char joinLabel[] = "radixJoinIntermediateJSON";
		joinJSON = new RadixJoin(joinPred, joinBinCSV, selJSON, &ctx, joinLabel,
				*matLeft, *matRight);
		joinBinCSV->setParent(joinJSON);
		selJSON->setParent(joinJSON);
	}

	/**
	 * REDUCE
	 * MAX(dim), COUNT(*)
	 */
	list<RecordAttribute> argProjections;
	argProjections.push_back(*dim);
	expressions::Expression* inputArg = new expressions::InputArgument(&recBin,
			0, argProjections);
	expressions::Expression* exprDim = new expressions::RecordProjection(
			dim->getOriginalType(), inputArg, *dim);
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


void symantecCSVWarmup(map<string,dataset> datasetCatalog)	{

	RawContext ctx = prepareContext("symantec-csv-warmup");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecCSV");
	dataset symantec = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantec 	=
			symantec.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = symantec.path;
	RecordType rec = symantec.recType;
	int linehint = symantec.linehint;
	int policy = 5;
	char delimInner = ';';

	vector<RecordAttribute*> projections;

	pm::CSVPlugin* pg = new pm::CSVPlugin(&ctx, fname, rec, projections,
			delimInner, linehint, policy, false);
	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 */
	list<RecordAttribute> argProjections;

	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	expressions::Expression* outputExpr =
			new expressions::IntConstant(1);

	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, scan, &ctx);
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

void symantecJSONWarmup(map<string,dataset> datasetCatalog)	{

	RawContext ctx = prepareContext("symantec-json-warmup");
	RawCatalog& rawCatalog = RawCatalog::getInstance();

	string nameSymantec = string("symantecIDDates");
	dataset symantec = datasetCatalog[nameSymantec];
	map<string, RecordAttribute*> argsSymantec 	=
			symantec.recType.getArgsMap();

	/**
	 * SCAN
	 */
	string fname = symantec.path;
	RecordType rec = symantec.recType;
	int linehint = symantec.linehint;

	ListType *documentType = new ListType(rec);
	jsonPipelined::JSONPlugin *pg = new jsonPipelined::JSONPlugin(&ctx, fname,
			documentType,linehint);

	rawCatalog.registerPlugin(fname, pg);
	Scan *scan = new Scan(&ctx, *pg);

	/**
	 * REDUCE
	 * (SUM 1)
	 */
	list<RecordAttribute> argProjections;

	expressions::Expression* arg 			=
				new expressions::InputArgument(&rec,0,argProjections);
	/* Output: */
	expressions::Expression* outputExpr =
			new expressions::IntConstant(1);

	ReduceNoPred *reduce = new ReduceNoPred(SUM, outputExpr, scan, &ctx);
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

