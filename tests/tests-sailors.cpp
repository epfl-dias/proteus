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


// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include "gtest/gtest.h"

#include "common/common.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/join.hpp"
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "plugins/csv-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>


TEST(Sailors, Scan) {
	RawContext ctx = RawContext("Sailors-Scan");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(3,filename,string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);
	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);
	Scan scan = Scan(&ctx, *pg);

	Root rootOp = Root(&scan);
	scan.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, Select) {
	RawContext ctx = RawContext("Sailors-Select");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(4,filename,string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);
	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * SELECT
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop);
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);
	expressions::Expression* lhsArg = new expressions::InputArgument(new FloatType(),0,projections);
	expressions::Expression* lhs = new expressions::RecordProjection(new FloatType(),lhsArg,*age);
	expressions::Expression* rhs = new expressions::FloatConstant(40);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);
	Select sel = Select(predicate,&scan);
	scan.setParent(&sel);

	/**
	 * PRINT
	 */
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),lhsArg,*sid);
	Print printOpProj = Print(debugInt,argProj,&sel);
	sel.setParent(&printOpProj);

	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, ScanBoats) {
	RawContext ctx = RawContext("Sailors-ScanBoats");
	RawCatalog& catalog = RawCatalog::getInstance();


	//SCAN1
	string filenameBoats = string("inputs/boats.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new StringType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats,string("bid"),intType);
	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats,string("bname"),stringType);
	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats,string("color"),stringType);

	list<RecordAttribute*> attrListBoats;
	attrListBoats.push_back(bidBoats);
	attrListBoats.push_back(bnameBoats);
	attrListBoats.push_back(colorBoats);
	RecordType recBoats = RecordType(attrListBoats);

	vector<RecordAttribute*> whichFieldsBoats;
	whichFieldsBoats.push_back(bidBoats);

	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
	catalog.registerPlugin(filenameBoats,pgBoats);
	Scan scanBoats = Scan(&ctx, *pgBoats);

	Root rootOp = Root(&scanBoats);
	scanBoats.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgBoats->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, JoinLeft3) {
	RawContext ctx = RawContext("Sailors-JoinLeft3");
	RawCatalog& catalog = RawCatalog::getInstance();

	/**
	 * SCAN1
	 */
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(3,filename,string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);
	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	//whichFields.push_back(rating); //Int
	whichFields.push_back(age); //Float

	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pgSailors);

	Scan scanSailors = Scan(&ctx, *pgSailors);

	/**
	 * SCAN2
	 */
	string filename2 = string("inputs/reserves.csv");
	RecordAttribute* sidReserves = new RecordAttribute(1,filename2,string("sid"),intType);
	RecordAttribute* bidReserves = new RecordAttribute(2,filename2,string("bid"),intType);
	RecordAttribute* day = new RecordAttribute(3,filename2,string("day"),stringType);

	list<RecordAttribute*> attrList2;
	attrList2.push_back(sidReserves);
	attrList2.push_back(bidReserves);
	attrList2.push_back(day);
	RecordType rec2 = RecordType(attrList2);
	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(sidReserves);
	whichFields2.push_back(bidReserves);

	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	catalog.registerPlugin(filename2,pgReserves);
	Scan scanReserves = Scan(&ctx, *pgReserves);

	/**
	 * JOIN
	 */
	RecordAttribute projTupleL = RecordAttribute(filename, activeLoop);
	list<RecordAttribute> projectionsL = list<RecordAttribute>();
	projectionsL.push_back(projTupleL);
	projectionsL.push_back(*sid);
	projectionsL.push_back(*age);
	expressions::Expression* leftArg = new expressions::InputArgument(intType,0,projectionsL);
	expressions::Expression* left = new expressions::RecordProjection(intType,leftArg,*sid);

	RecordAttribute projTupleR = RecordAttribute(filename2, activeLoop);
	list<RecordAttribute> projectionsR = list<RecordAttribute>();
	projectionsR.push_back(projTupleR);
	projectionsR.push_back(*sidReserves);
	projectionsR.push_back(*bidReserves);
	expressions::Expression* rightArg = new expressions::InputArgument(intType,1,projectionsR);
	expressions::Expression* right = new expressions::RecordProjection(intType,rightArg,*sidReserves);

	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);

	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(),EAGER);
	outputModes.insert(outputModes.begin(),EAGER);
	Materializer* mat = new Materializer(whichFields,outputModes);

	char joinLabel[] = "join1";
	Join join = Join(joinPred,scanSailors,scanReserves,joinLabel,*mat);
	scanSailors.setParent(&join);
	scanReserves.setParent(&join);


	//SCAN3
	string filenameBoats = string("inputs/boats.csv");
	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats,string("bid"),intType);
	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats,string("bname"),stringType);
	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats,string("color"),stringType);

	list<RecordAttribute*> attrListBoats;
	attrListBoats.push_back(bidBoats);
	attrListBoats.push_back(bnameBoats);
	attrListBoats.push_back(colorBoats);
	RecordType recBoats = RecordType(attrListBoats);

	vector<RecordAttribute*> whichFieldsBoats;
	whichFieldsBoats.push_back(bidBoats);

	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
	catalog.registerPlugin(filenameBoats,pgBoats);
	Scan scanBoats = Scan(&ctx, *pgBoats);

	/**
	 * JOIN2
	 */
	expressions::Expression* leftArg2 = new expressions::InputArgument(intType,0,projectionsR);
	expressions::Expression* left2 = new expressions::RecordProjection(intType,leftArg2,*bidReserves);

	RecordAttribute projTupleBoat = RecordAttribute(filenameBoats, activeLoop);
	list<RecordAttribute> projectionsBoats = list<RecordAttribute>();
	projectionsBoats.push_back(projTupleBoat);
	projectionsBoats.push_back(*bidBoats);
	expressions::Expression* rightArg2 = new expressions::InputArgument(intType,1,projectionsBoats);
	expressions::Expression* right2 = new expressions::RecordProjection(intType,rightArg2,*bidBoats);

	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(new BoolType(),left2,right2);
	vector<materialization_mode> outputModes2;
	outputModes2.insert(outputModes2.begin(),EAGER);
	outputModes2.insert(outputModes2.begin(),EAGER);
	Materializer* mat2 = new Materializer(whichFields2,outputModes2);

	char joinLabel2[] = "join2";
	Join join2 = Join(joinPred2,join,scanBoats, joinLabel2, *mat2);
	join.setParent(&join2);
	scanBoats.setParent(&join2);


	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),leftArg2,*bidBoats);
	Print printOpProj = Print(debugInt,argProj,&join2);
	join2.setParent(&printOpProj);


	//ROOT
	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgSailors->finish();
	pgReserves->finish();
	pgBoats->finish();

	catalog.clear();
	EXPECT_TRUE(true);
}

//Just like previous one, but with permuted operators
TEST(Sailors, JoinRight3) {
	RawContext ctx = RawContext("Sailors-JoinRight3");
	RawCatalog& catalog = RawCatalog::getInstance();

	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();

	//SCAN2
	string filename2 = string("inputs/reserves.csv");
	RecordAttribute* sidReserves = new RecordAttribute(1,filename2,string("sid"),intType);
	RecordAttribute* bidReserves = new RecordAttribute(2,filename2,string("bid"),intType);
	RecordAttribute* day = new RecordAttribute(3,filename2,string("day"),stringType);
	list<RecordAttribute*> attrList2;
	attrList2.push_back(sidReserves);
	attrList2.push_back(bidReserves);
	attrList2.push_back(day);
	RecordType rec2 = RecordType(attrList2);

	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(sidReserves);
	whichFields2.push_back(bidReserves);

	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	catalog.registerPlugin(filename2,pgReserves);

	Scan scanReserves = Scan(&ctx, *pgReserves);


	//SCAN3!!
	string filenameBoats = string("inputs/boats.csv");
	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats,string("bid"),intType);
	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats,string("bname"),stringType);
	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats,string("color"),stringType);
	list<RecordAttribute*> attrListBoats;
	attrListBoats.push_back(bidBoats);
	attrListBoats.push_back(bnameBoats);
	attrListBoats.push_back(colorBoats);
	RecordType recBoats = RecordType(attrListBoats);

	vector<RecordAttribute*> whichFieldsBoats;
	whichFieldsBoats.push_back(bidBoats);
	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
	catalog.registerPlugin(filenameBoats,pgBoats);

	Scan scanBoats = Scan(&ctx, *pgBoats);


	//JOIN2
	RecordAttribute projTupleReserves = RecordAttribute(filename2, activeLoop);
	list<RecordAttribute> projectionsReserves = list<RecordAttribute>();
	projectionsReserves.push_back(projTupleReserves);
	projectionsReserves.push_back(*sidReserves);
	projectionsReserves.push_back(*bidReserves);
	expressions::Expression* leftArg2 = new expressions::InputArgument(intType,0,projectionsReserves);
	expressions::Expression* left2 = new expressions::RecordProjection(intType,leftArg2,*bidReserves);

	RecordAttribute projTupleBoats = RecordAttribute(filenameBoats, activeLoop);
	list<RecordAttribute> projectionsBoats = list<RecordAttribute>();
	projectionsBoats.push_back(projTupleBoats);
	projectionsBoats.push_back(*bidBoats);
	expressions::Expression* rightArg2 = new expressions::InputArgument(intType,1,projectionsBoats);
	expressions::Expression* right2 = new expressions::RecordProjection(intType,rightArg2,*bidBoats);

	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(new BoolType(),left2,right2);
	vector<materialization_mode> outputModes2;
	outputModes2.insert(outputModes2.begin(),EAGER);
	outputModes2.insert(outputModes2.begin(),EAGER);
	Materializer* mat2 = new Materializer(whichFields2,outputModes2);

	char joinLabel2[] = "join2";
	Join join2 = Join(joinPred2,scanReserves,scanBoats, joinLabel2, *mat2);
	scanReserves.setParent(&join2);
	scanBoats.setParent(&join2);


	//SCAN1
	string filename = string("inputs/sailors.csv");
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(3,filename,string("age"),floatType);
	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);
	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	//whichFields.push_back(rating); //Int
	whichFields.push_back(age); //Float

	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pgSailors);

	Scan scanSailors = Scan(&ctx, *pgSailors);


	//JOIN
	RecordAttribute projTupleSailors = RecordAttribute(filename, activeLoop);
	list<RecordAttribute> projectionsSailors = list<RecordAttribute>();
	projectionsSailors.push_back(projTupleSailors);
	projectionsSailors.push_back(*sid);
	projectionsSailors.push_back(*age);
	expressions::Expression* leftArg = new expressions::InputArgument(intType,
			0, projectionsSailors);
	expressions::Expression* left = new expressions::RecordProjection(intType,
			leftArg, *sid);


	//For 100% correctness, I should have a new projections list, and not just the ones from reserves
	expressions::Expression* rightArg = new expressions::InputArgument(intType,1,projectionsReserves);
	expressions::Expression* right = new expressions::RecordProjection(intType,rightArg,*sidReserves);
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);
	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(),EAGER);
	outputModes.insert(outputModes.begin(),EAGER);
	Materializer* mat = new Materializer(whichFields,outputModes);

	char joinLabel[] = "join1";
	Join join = Join(joinPred,scanSailors,join2, joinLabel, *mat);
	scanSailors.setParent(&join);
	join2.setParent(&join);


	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),leftArg2,*bidBoats);
	Print printOpProj = Print(debugInt,argProj,&join);
	join.setParent(&printOpProj);

	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);

	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());


	//Close all open files
	pgReserves->finish();
	pgBoats->finish();

	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, Join) {
	RawContext ctx = RawContext("Sailors-Join");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename,string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename,string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename,string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(3,filename,string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	//whichFields.push_back(rating); //Int
	whichFields.push_back(age); //Float

	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pgSailors);
	Scan scanSailors = Scan(&ctx, *pgSailors);


	//SCAN2
	string filename2 = string("inputs/reserves.csv");
	RecordAttribute* sidReserves = new RecordAttribute(1,filename2,string("sid"),intType);
	RecordAttribute* bidReserves = new RecordAttribute(2,filename2,string("bid"),intType);
	RecordAttribute* day = new RecordAttribute(3,filename2,string("day"),stringType);
	list<RecordAttribute*> attrList2;
	attrList2.push_back(sidReserves);
	attrList2.push_back(bidReserves);
	attrList2.push_back(day);
	RecordType rec2 = RecordType(attrList2);

	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(sidReserves);
	whichFields2.push_back(bidReserves);

	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	catalog.registerPlugin(filename2,pgReserves);
	Scan scanReserves = Scan(&ctx, *pgReserves);

	//JOIN
	string argLeft = filename+"_"+string("sid");
	string argRight = filename2+"_"+string("sid");

	RecordAttribute projTupleSailors = RecordAttribute(filename, activeLoop);
	list<RecordAttribute> projectionsSailors = list<RecordAttribute>();
	projectionsSailors.push_back(projTupleSailors);
	projectionsSailors.push_back(*sid);
	projectionsSailors.push_back(*age);
	expressions::Expression* leftArg = new expressions::InputArgument(intType,
			0, projectionsSailors);
	expressions::Expression* left = new expressions::RecordProjection(intType,
			leftArg, *sid);

	RecordAttribute projTupleReserves = RecordAttribute(filename2, activeLoop);
	list<RecordAttribute> projectionsReserves = list<RecordAttribute>();
	projectionsReserves.push_back(projTupleReserves);
	projectionsReserves.push_back(*sidReserves);
	projectionsReserves.push_back(*bidReserves);
	expressions::Expression* rightArg = new expressions::InputArgument(intType,
			1, projectionsReserves);
	expressions::Expression* right = new expressions::RecordProjection(intType,rightArg,*sidReserves);
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);
	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(),EAGER);
	outputModes.insert(outputModes.begin(),EAGER);
	Materializer* mat = new Materializer(whichFields,outputModes);

	char joinLabel[] = "join1";
	Join join = Join(joinPred,scanSailors,scanReserves, joinLabel, *mat);
	scanSailors.setParent(&join);
	scanReserves.setParent(&join);


	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),leftArg,*sid);
	Print printOpProj = Print(debugInt,argProj,&join);
	join.setParent(&printOpProj);


	//ROOT
	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgSailors->finish();
	pgReserves->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

