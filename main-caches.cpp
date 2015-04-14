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
#include "operators/nest.hpp"
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

/* Expression matching microbenchmarks */
void expressionMap();
void expressionMapVertical();
void joinQueryRelationalRadixCache();
void materializer();
/* Use it to cross-compare with materializer */
void selectionJSONFlat();

RawContext prepareContext(string moduleName)	{
	RawContext ctx = RawContext(moduleName);
	registerFunctions(ctx);
	return ctx;
}

int main() {

//	expressionMap();
//	expressionMapVertical();
//
//	joinQueryRelationalRadixCache();
//	joinQueryRelationalRadixCache();

//	selectionJSONFlat();
	materializer();

}

void expressionMap() {
	map<expressions::Expression*, int, less_map> mapTest;

	expressions::IntConstant val_int = expressions::IntConstant(88);
	expressions::FloatConstant val_float = expressions::FloatConstant(90.4);
	expressions::IntConstant key = expressions::IntConstant(88);

	mapTest[&val_int] = 9;
	mapTest[&val_float] = 7;

	IntType intType = IntType();
	StringType stringType = StringType();
	string fname = "file.json";
	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	RecordAttribute projTuple = RecordAttribute(fname, activeLoop);
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	expressions::Expression *inputArg = new expressions::InputArgument(&inner,
			0, projections);
	mapTest[inputArg] = 111;

	/* 2nd Input Arg */
	projections.push_back(projTuple);
	expressions::Expression *inputArg2 = new expressions::InputArgument(&inner,
			0, projections);
	mapTest[inputArg2] = 181;

	{
		map<expressions::Expression*, int, less_map>::iterator it =
				mapTest.find(&key);
		if (it != mapTest.end()) {
			cout << "Found! " << it->second << endl;
		}
	}

	{
		map<expressions::Expression*, int, less_map>::iterator it =
				mapTest.find(inputArg2);
		if (it != mapTest.end()) {
			cout << "Found! " << it->second << endl;
		}
	}
}

void expressionMapVertical() {
	map<expressions::Expression*, int, less_map> mapTest;

	/* Using constants as smoke test. Not to be matched */
	expressions::IntConstant val_int = expressions::IntConstant(88);
	expressions::FloatConstant val_float = expressions::FloatConstant(90.4);
	mapTest[&val_int] = 9;
	mapTest[&val_float] = 7;

	/* Preparing an Input Argument to insert */
	IntType intType = IntType();
	StringType stringType = StringType();
	string fname = "file.csv";
	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	string empSalary = string("salary");
	RecordAttribute emp3 = RecordAttribute(3, fname, empSalary, &intType);
	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);
	atts.push_back(&emp3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	RecordAttribute projTuple = RecordAttribute(fname, activeLoop);
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	expressions::Expression *inputArg = new expressions::InputArgument(&inner,
			0, projections);
	mapTest[inputArg] = 111;

	/* Also inserting the Argument's projections as standalone */
	expressions::RecordProjection* proj1 = new expressions::RecordProjection(
			&stringType, inputArg, emp1);
	expressions::RecordProjection* proj2 = new expressions::RecordProjection(
			&intType, inputArg, emp2);
	expressions::RecordProjection* proj3 = new expressions::RecordProjection(
			&intType, inputArg, emp3);
	mapTest[proj1] = 112;
	mapTest[proj2] = 113;
	mapTest[proj3] = 114;

	/* 2nd Input Arg as smoke test. Not to be matched */
	projections.push_back(projTuple);
	expressions::Expression *inputArg2 = new expressions::InputArgument(&inner,
			0, projections);
	mapTest[inputArg2] = 181;

	{
		map<expressions::Expression*, int, less_map>::iterator it =
				mapTest.find(inputArg);
		if (it != mapTest.end()) {
			cout << "Found! " << it->second << endl;
		}
	}

	{
		map<expressions::Expression*, int, less_map>::iterator it =
				mapTest.find(proj2);
		if (it != mapTest.end()) {
			cout << "Found! " << it->second << endl;
		}
	}
}

void joinQueryRelationalRadixCache() {
	RawContext ctx = prepareContext("testFunction-RadixJoinCSV");
	RawCatalog& catalog = RawCatalog::getInstance();

	/**
	 * SCAN1
	 */
	string filename = string("inputs/input.csv");
	PrimitiveType* intType = new IntType();
	RecordAttribute* attr1 = new RecordAttribute(1, filename, string("att1"),
			intType);
	RecordAttribute* attr2 = new RecordAttribute(2, filename, string("att2"),
			intType);
	RecordAttribute* attr3 = new RecordAttribute(3, filename, string("att3"),
			intType);
	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);
	attrList.push_back(attr3);

	RecordType rec1 = RecordType(attrList);
	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

//	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	pm::CSVPlugin* pg = new pm::CSVPlugin(&ctx, filename, rec1, whichFields, 3,
			2);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	RecordAttribute projTupleL = RecordAttribute(filename, activeLoop);
	list<RecordAttribute> projectionsL = list<RecordAttribute>();
	projectionsL.push_back(projTupleL);
	projectionsL.push_back(*attr1);
	projectionsL.push_back(*attr2);

	/**
	 * SCAN2
	 */
	string filename2 = string("inputs/input2.csv");
	RecordAttribute* attr1_f2 = new RecordAttribute(1, filename2,
			string("att1_r"), intType);
	RecordAttribute* attr2_f2 = new RecordAttribute(2, filename2,
			string("att2_r"), intType);
	RecordAttribute* attr3_f2 = new RecordAttribute(3, filename2,
			string("att3_r"), intType);

	list<RecordAttribute*> attrList2;
	attrList2.push_back(attr1_f2);
	attrList2.push_back(attr2_f2);
	attrList2.push_back(attr3_f2);
	RecordType rec2 = RecordType(attrList2);

	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(attr1_f2);
	whichFields2.push_back(attr2_f2);

//	CSVPlugin* pg2 = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	pm::CSVPlugin* pg2 = new pm::CSVPlugin(&ctx, filename2, rec2, whichFields2,
			2, 2);
	catalog.registerPlugin(filename2, pg2);
	Scan scan2 = Scan(&ctx, *pg2);
	LOG(INFO)<<"Right:"<<&scan2;

	RecordAttribute projTupleR = RecordAttribute(filename2, activeLoop);
	list<RecordAttribute> projectionsR = list<RecordAttribute>();
	projectionsR.push_back(projTupleR);
	projectionsR.push_back(*attr1_f2);
	projectionsR.push_back(*attr2_f2);

	/**
	 * JOIN
	 */
	expressions::Expression* leftArg = new expressions::InputArgument(intType,
			0, projectionsL);
	expressions::Expression* left = new expressions::RecordProjection(intType,
			leftArg, *attr2);
	expressions::Expression* rightArg = new expressions::InputArgument(intType,
			1, projectionsR);
	expressions::Expression* right = new expressions::RecordProjection(intType,
			rightArg, *attr2_f2);
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(
			new BoolType(), left, right);
	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(), EAGER);
	outputModes.insert(outputModes.begin(), EAGER);

	/* XXX Updated Materializer requires 'expressions to be cached'*/
	expressions::Expression* exprLeftOID = new expressions::RecordProjection(
			intType, leftArg, projTupleL);
	expressions::Expression* exprLeftMat1 = new expressions::RecordProjection(
			intType, leftArg, *attr1);
	expressions::Expression* exprLeftMat2 = new expressions::RecordProjection(
			intType, leftArg, *attr2);
	vector<expressions::Expression*> whichExpressionsLeft;
	whichExpressionsLeft.push_back(exprLeftOID);
	whichExpressionsLeft.push_back(exprLeftMat1);
	whichExpressionsLeft.push_back(exprLeftMat2);

	Materializer* matLeft = new Materializer(whichFields, whichExpressionsLeft,
			outputModes);

	vector<materialization_mode> outputModes2;
	//active loop too
	outputModes2.insert(outputModes2.begin(), EAGER);
	outputModes2.insert(outputModes2.begin(), EAGER);
	outputModes2.insert(outputModes2.begin(), EAGER);

	/* XXX Updated Materializer requires 'expressions to be cached'*/
	expressions::Expression* exprRightOID = new expressions::RecordProjection(
			intType, rightArg, projTupleR);
	expressions::Expression* exprRightMat1 = new expressions::RecordProjection(
			intType, rightArg, *attr1_f2);
	expressions::Expression* exprRightMat2 = new expressions::RecordProjection(
			intType, rightArg, *attr2_f2);
	vector<expressions::Expression*> whichExpressionsRight;
	whichExpressionsRight.push_back(exprRightOID);
	whichExpressionsRight.push_back(exprRightMat1);
	whichExpressionsRight.push_back(exprRightMat2);

	Materializer* matRight = new Materializer(whichFields2,
			whichExpressionsRight, outputModes2);

	char joinLabel[] = "radixJoin1";
	RadixJoin join = RadixJoin(joinPred, scan, scan2, &ctx, joinLabel, *matLeft,
			*matRight);
	scan.setParent(&join);
	scan2.setParent(&join);

	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	//To be 100% correct, this proj should be over a new InputArg that only exposes the new bindings
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			new IntType(), leftArg, *attr1);

	Print printOp = Print(debugInt, proj, &join);
	join.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	pg2->finish();
	catalog.clear();
	cout << "End of execution" << endl;
}

void materializer()
{
	RawContext ctx = prepareContext("materializer");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/json/jsmn-flat.json");

	string attrName = string("a");
	string attrName2 = string("b");
	IntType attrType = IntType();
	RecordAttribute attr = RecordAttribute(1, fname, attrName, &attrType);
	RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &attrType);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&attr);
	atts.push_back(&attr2);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsonPipelined::JSONPlugin pg = jsonPipelined::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);

	Scan scan = Scan(&ctx, pg);

	/*
	 * Materialize expression(s) here
	 * rec.b the one cached
	 */
	RecordAttribute projTuple = RecordAttribute(fname, activeLoop);
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(attr);
	projections.push_back(attr2);
	expressions::Expression* lhsArg = new expressions::InputArgument(&attrType,
			0, projections);
	expressions::Expression* lhs = new expressions::RecordProjection(&attrType,
			lhsArg, attr2);

	char matLabel[] = "materializer";
	ExprMaterializer mat = ExprMaterializer(lhs, &scan,&ctx, matLabel);
	scan.setParent(&mat);

	/**
	 * SELECT
	 */
	expressions::Expression* rhs = new expressions::IntConstant(5);

	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);
	Select sel = Select(predicate, &mat);
	mat.setParent(&sel);

	/**
	 * PRINT
	 */
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&attrType, lhsArg, attr);
	Print printOp = Print(debugInt, proj, &sel);
	sel.setParent(&printOp);

	/**
	 * ROOT
	 */
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();
}

void selectionJSONFlat()
{
	RawContext ctx = prepareContext("testFunction-SelectJSON-flat");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/json/jsmn-flat.json");

	string attrName = string("a");
	string attrName2 = string("b");
	IntType attrType = IntType();
	RecordAttribute attr = RecordAttribute(1, fname, attrName, &attrType);
	RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &attrType);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&attr);
	atts.push_back(&attr2);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsonPipelined::JSONPlugin pg = jsonPipelined::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);

	Scan scan = Scan(&ctx, pg);

	/**
	 * SELECT
	 */
	RecordAttribute projTuple = RecordAttribute(fname, activeLoop);
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(attr);
	projections.push_back(attr2);

	expressions::Expression* lhsArg = new expressions::InputArgument(&attrType,
			0, projections);
	expressions::Expression* lhs = new expressions::RecordProjection(&attrType,
			lhsArg, attr2);
	expressions::Expression* rhs = new expressions::IntConstant(5);

	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	Select sel = Select(predicate, &scan);
	scan.setParent(&sel);

	/**
	 * PRINT
	 */
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&attrType, lhsArg, attr);
	Print printOp = Print(debugInt, proj, &sel);
	sel.setParent(&printOp);

	/**
	 * ROOT
	 */
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();
}
