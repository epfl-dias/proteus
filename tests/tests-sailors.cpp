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

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename+"_"+string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename+"_"+string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename+"_"+string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(3,filename+"_"+string("age"),floatType);

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
	Scan scan = Scan(&ctx, *pg);

	Root rootOp = Root(&scan);
	scan.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, Select) {
	RawContext ctx = RawContext("Sailors-Select");

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename+"_"+string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename+"_"+string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename+"_"+string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(4,filename+"_"+string("age"),floatType);

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
	Scan scan = Scan(&ctx, *pg);

	//Selection
	string argName = filename+"_"+string("age");
	expressions::Expression* lhsArg = new expressions::InputArgument(new FloatType(),0);
	expressions::Expression* lhs = new expressions::RecordProjection(new FloatType(),lhsArg,argName.c_str());
	expressions::Expression* rhs = new expressions::FloatConstant(40);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);
	Select sel = Select(predicate,&scan,pg);
	scan.setParent(&sel);

	//'Print' operator - used for debug purposes
	Function* debugInt = ctx.getFunction("printi");
	string argName_ = filename+"_"+string("sid");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),lhsArg,argName_.c_str());
	Print printOpProj = Print(debugInt,argProj,&sel,pg);
	sel.setParent(&printOpProj);

	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Sailors, ScanBoats) {
	RawContext ctx = RawContext("Sailors-ScanBoats");

	//SCAN1
	string filenameBoats = string("inputs/boats.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new StringType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats+"_"+string("bid"),intType);
	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats+"_"+string("bname"),stringType);
	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats+"_"+string("color"),stringType);

	list<RecordAttribute*> attrListBoats;
	attrListBoats.push_back(bidBoats);
	attrListBoats.push_back(bnameBoats);
	attrListBoats.push_back(colorBoats);
	RecordType recBoats = RecordType(attrListBoats);

	vector<RecordAttribute*> whichFieldsBoats;
	whichFieldsBoats.push_back(bidBoats);

	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
	Scan scanBoats = Scan(&ctx, *pgBoats);

	Root rootOp = Root(&scanBoats);
	scanBoats.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pgBoats->finish();
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.clear();
	EXPECT_TRUE(true);
}

//'Complex' join queries break with the current plugins/bindings handling
//TEST(Sailors, JoinLeft3) {
//	RawContext ctx = RawContext("Sailors-JoinLeft3");
//
//	//SCAN1
//	string filename = string("inputs/sailors.csv");
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//	RecordAttribute* sid = new RecordAttribute(1,filename+"_"+string("sid"),intType);
//	RecordAttribute* sname = new RecordAttribute(2,filename+"_"+string("sname"),stringType);
//	RecordAttribute* rating = new RecordAttribute(3,filename+"_"+string("rating"),intType);
//	RecordAttribute* age = new RecordAttribute(3,filename+"_"+string("age"),floatType);
//
//	list<RecordAttribute*> attrList;
//	attrList.push_back(sid);
//	attrList.push_back(sname);
//	attrList.push_back(rating);
//	attrList.push_back(age);
//	RecordType rec1 = RecordType(attrList);
//
//	vector<RecordAttribute*> whichFields;
//	whichFields.push_back(sid);
//	//whichFields.push_back(rating); //Int
//	whichFields.push_back(age); //Float
//
//	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//
//	//SCAN2
//	string filename2 = string("inputs/reserves.csv");
//	RecordAttribute* sidReserves = new RecordAttribute(1,filename2+"_"+string("sid"),intType);
//	RecordAttribute* bidReserves = new RecordAttribute(2,filename2+"_"+string("bid"),intType);
//	RecordAttribute* day = new RecordAttribute(3,filename2+"_"+string("day"),stringType);
//
//	list<RecordAttribute*> attrList2;
//	attrList2.push_back(sidReserves);
//	attrList2.push_back(bidReserves);
//	attrList2.push_back(day);
//	RecordType rec2 = RecordType(attrList2);
//	vector<RecordAttribute*> whichFields2;
//	whichFields2.push_back(sidReserves);
//	whichFields2.push_back(bidReserves);
//
//	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
//	Scan scanReserves = Scan(&ctx, *pgReserves);
//
//
//	//JOIN
//	string argLeft = filename+"_"+string("sid");
//	string argRight = filename2+"_"+string("sid");
//
//	expressions::Expression* leftArg = new expressions::InputArgument(intType,0);
//	expressions::Expression* left = new expressions::RecordProjection(intType,leftArg,argLeft.c_str());
//	expressions::Expression* rightArg = new expressions::InputArgument(intType,1);
//	expressions::Expression* right = new expressions::RecordProjection(intType,rightArg,argRight.c_str());
//
//	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);
//
//	vector<materialization_mode> outputModes;
//	outputModes.insert(outputModes.begin(),EAGER);
//	outputModes.insert(outputModes.begin(),EAGER);
//	Materializer* mat = new Materializer(whichFields,outputModes);
//
//	Join join = Join(joinPred,scanSailors,scanReserves,"join1",*mat,pgSailors,pgReserves);
//	scanSailors.setParent(&join);
//	scanReserves.setParent(&join);
//
//
//	//SCAN3
//	string filenameBoats = string("inputs/boats.csv");
//	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats+"_"+string("bid"),intType);
//	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats+"_"+string("bname"),stringType);
//	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats+"_"+string("color"),stringType);
//
//	list<RecordAttribute*> attrListBoats;
//	attrListBoats.push_back(bidBoats);
//	attrListBoats.push_back(bnameBoats);
//	attrListBoats.push_back(colorBoats);
//	RecordType recBoats = RecordType(attrListBoats);
//
//	vector<RecordAttribute*> whichFieldsBoats;
//	whichFieldsBoats.push_back(bidBoats);
//
//	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
//	Scan scanBoats = Scan(&ctx, *pgBoats);
//
//
//	//JOIN2
//	string argLeft2 = filename2+"_"+string("bid");
//	string argRight2 = filenameBoats+"_"+string("bid");
//
//
//	expressions::Expression* leftArg2 = new expressions::InputArgument(intType,0);
//	expressions::Expression* left2 = new expressions::RecordProjection(intType,leftArg2,argLeft2.c_str());
//	expressions::Expression* rightArg2 = new expressions::InputArgument(intType,1);
//	expressions::Expression* right2 = new expressions::RecordProjection(intType,rightArg2,argRight2.c_str());
//
//	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(new BoolType(),left2,right2,???);
//	vector<materialization_mode> outputModes2;
//	outputModes2.insert(outputModes2.begin(),EAGER);
//	outputModes2.insert(outputModes2.begin(),EAGER);
//	Materializer* mat2 = new Materializer(whichFields2,outputModes2);
//
//	Join join2 = Join(joinPred2,join,scanBoats, "join2", *mat2);
//	join.setParent(&join2);
//	scanBoats.setParent(&join2);
//
//
//	//PRINT
//	Function* debugInt = ctx.getFunction("printi");
//	string argName_ = filenameBoats+"_"+string("bid");
//	expressions::InputArgument* argProj = new expressions::InputArgument(new IntType(),1,argName_);
//	Print printOpProj = Print(debugInt,argProj,&join2);
//	join2.setParent(&printOpProj);
//
//
//	//ROOT
//	Root rootOp = Root(&printOpProj);
//	printOpProj.setParent(&rootOp);
//	rootOp.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgSailors->finish();
//	pgReserves->finish();
//	pgBoats->finish();
//
//	RawCatalog& catalog = RawCatalog::getInstance();
//	catalog.clear();
//	EXPECT_TRUE(true);
//}

////Just like previous one, but with permuted operators
//TEST(Sailors, JoinRight3) {
//	RawContext ctx = RawContext("Sailors-JoinRight3");
//
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//
//	//SCAN2
//	string filename2 = string("inputs/reserves.csv");
//	RecordAttribute* sidReserves = new RecordAttribute(1,filename2+"_"+string("sid"),intType);
//	RecordAttribute* bidReserves = new RecordAttribute(2,filename2+"_"+string("bid"),intType);
//	RecordAttribute* day = new RecordAttribute(3,filename2+"_"+string("day"),stringType);
//	list<RecordAttribute*> attrList2;
//	attrList2.push_back(sidReserves);
//	attrList2.push_back(bidReserves);
//	attrList2.push_back(day);
//	RecordType rec2 = RecordType(attrList2);
//
//	vector<RecordAttribute*> whichFields2;
//	whichFields2.push_back(sidReserves);
//	whichFields2.push_back(bidReserves);
//
//	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
//	Scan scanReserves = Scan(&ctx, *pgReserves);
//
//
//	//SCAN3!!
//	string filenameBoats = string("inputs/boats.csv");
//	RecordAttribute* bidBoats = new RecordAttribute(1,filenameBoats+"_"+string("bid"),intType);
//	RecordAttribute* bnameBoats = new RecordAttribute(2,filenameBoats+"_"+string("bname"),stringType);
//	RecordAttribute* colorBoats = new RecordAttribute(3,filenameBoats+"_"+string("color"),stringType);
//	list<RecordAttribute*> attrListBoats;
//	attrListBoats.push_back(bidBoats);
//	attrListBoats.push_back(bnameBoats);
//	attrListBoats.push_back(colorBoats);
//	RecordType recBoats = RecordType(attrListBoats);
//
//	vector<RecordAttribute*> whichFieldsBoats;
//	whichFieldsBoats.push_back(bidBoats);
//
//	CSVPlugin* pgBoats = new CSVPlugin(&ctx, filenameBoats, recBoats, whichFieldsBoats);
//	Scan scanBoats = Scan(&ctx, *pgBoats);
//
//
//	//JOIN2
//	string argLeft2 = filename2+"_"+string("bid");
//	string argRight2 = filenameBoats+"_"+string("bid");
//	expressions::InputArgument* left2 = new expressions::InputArgument(new IntType(),1,argLeft2);
//	expressions::InputArgument* right2 = new expressions::InputArgument(new IntType(),1,argRight2);
//	expressions::BinaryExpression* joinPred2 = new expressions::EqExpression(new BoolType(),left2,right2);
//	vector<materialization_mode> outputModes2;
//	outputModes2.insert(outputModes2.begin(),EAGER);
//	outputModes2.insert(outputModes2.begin(),EAGER);
//	Materializer* mat2 = new Materializer(whichFields2,outputModes2);
//
//	Join join2 = Join(joinPred2,scanReserves,scanBoats, "join2", *mat2);
//	scanReserves.setParent(&join2);
//	scanBoats.setParent(&join2);
//
//
//	//SCAN1
//	string filename = string("inputs/sailors.csv");
//	RecordAttribute* sid = new RecordAttribute(1,filename+"_"+string("sid"),intType);
//	RecordAttribute* sname = new RecordAttribute(2,filename+"_"+string("sname"),stringType);
//	RecordAttribute* rating = new RecordAttribute(3,filename+"_"+string("rating"),intType);
//	RecordAttribute* age = new RecordAttribute(3,filename+"_"+string("age"),floatType);
//	list<RecordAttribute*> attrList;
//	attrList.push_back(sid);
//	attrList.push_back(sname);
//	attrList.push_back(rating);
//	attrList.push_back(age);
//	RecordType rec1 = RecordType(attrList);
//
//	vector<RecordAttribute*> whichFields;
//	whichFields.push_back(sid);
//	//whichFields.push_back(rating); //Int
//	whichFields.push_back(age); //Float
//
//	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//
//	//JOIN
//	string argLeft = filename+"_"+string("sid");
//	string argRight = filename2+"_"+string("sid");
//	expressions::InputArgument* left = new expressions::InputArgument(new IntType(),1,argLeft);
//	expressions::InputArgument* right = new expressions::InputArgument(new IntType(),1,argRight);
//	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);
//	vector<materialization_mode> outputModes;
//	outputModes.insert(outputModes.begin(),EAGER);
//	outputModes.insert(outputModes.begin(),EAGER);
//	Materializer* mat = new Materializer(whichFields,outputModes);
//
//	Join join = Join(joinPred,scanSailors,join2, "join1", *mat);
//	scanSailors.setParent(&join);
//	join2.setParent(&join);
//
//
//	//PRINT
//	Function* debugInt = ctx.getFunction("printi");
//	string argName_ = filenameBoats+"_"+string("bid");
//	expressions::InputArgument* argProj = new expressions::InputArgument(new IntType(),1,argName_);
//	Print printOpProj = Print(debugInt,argProj,&join);
//	join.setParent(&printOpProj);
//
//	Root rootOp = Root(&printOpProj);
//	printOpProj.setParent(&rootOp);
//
//	rootOp.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//
//	//Close all open files
//	pgReserves->finish();
//	pgBoats->finish();
//
//	RawCatalog& catalog = RawCatalog::getInstance();
//	catalog.clear();
//	EXPECT_TRUE(true);
//}
//
//TEST(Sailors, Join) {
//	RawContext ctx = RawContext("Sailors-Join");
//
//	//SCAN1
//	string filename = string("inputs/sailors.csv");
//	PrimitiveType* intType = new IntType();
//	PrimitiveType* floatType = new FloatType();
//	PrimitiveType* stringType = new StringType();
//	RecordAttribute* sid = new RecordAttribute(1,filename+"_"+string("sid"),intType);
//	RecordAttribute* sname = new RecordAttribute(2,filename+"_"+string("sname"),stringType);
//	RecordAttribute* rating = new RecordAttribute(3,filename+"_"+string("rating"),intType);
//	RecordAttribute* age = new RecordAttribute(3,filename+"_"+string("age"),floatType);
//
//	list<RecordAttribute*> attrList;
//	attrList.push_back(sid);
//	attrList.push_back(sname);
//	attrList.push_back(rating);
//	attrList.push_back(age);
//
//	RecordType rec1 = RecordType(attrList);
//
//	vector<RecordAttribute*> whichFields;
//	whichFields.push_back(sid);
//	//whichFields.push_back(rating); //Int
//	whichFields.push_back(age); //Float
//
//	CSVPlugin* pgSailors = new CSVPlugin(&ctx, filename, rec1, whichFields);
//	Scan scanSailors = Scan(&ctx, *pgSailors);
//
//
//	//SCAN2
//	string filename2 = string("inputs/reserves.csv");
//	RecordAttribute* sidReserves = new RecordAttribute(1,filename2+"_"+string("sid"),intType);
//	RecordAttribute* bidReserves = new RecordAttribute(2,filename2+"_"+string("bid"),intType);
//	RecordAttribute* day = new RecordAttribute(3,filename2+"_"+string("day"),stringType);
//	list<RecordAttribute*> attrList2;
//	attrList2.push_back(sidReserves);
//	attrList2.push_back(bidReserves);
//	attrList2.push_back(day);
//	RecordType rec2 = RecordType(attrList2);
//
//	vector<RecordAttribute*> whichFields2;
//	whichFields2.push_back(sidReserves);
//	whichFields2.push_back(bidReserves);
//
//	CSVPlugin* pgReserves = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
//	Scan scanReserves = Scan(&ctx, *pgReserves);
//
//
//	//JOIN
//	string argLeft = filename+"_"+string("sid");
//	string argRight = filename2+"_"+string("sid");
//	expressions::InputArgument* left = new expressions::InputArgument(new IntType(),1,argLeft);
//	expressions::InputArgument* right = new expressions::InputArgument(new IntType(),1,argRight);
//	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);
//	vector<materialization_mode> outputModes;
//	outputModes.insert(outputModes.begin(),EAGER);
//	outputModes.insert(outputModes.begin(),EAGER);
//	Materializer* mat = new Materializer(whichFields,outputModes);
//
//	Join join = Join(joinPred,scanSailors,scanReserves, "join1", *mat);
//	scanSailors.setParent(&join);
//	scanReserves.setParent(&join);
//
//
//	//PRINT
//	Function* debugInt = ctx.getFunction("printi");
//	string argName_ = filename+"_"+string("sid");
//	expressions::InputArgument* argProj = new expressions::InputArgument(new IntType(),1,argName_);
//	Print printOpProj = Print(debugInt,argProj,&join);
//	join.setParent(&printOpProj);
//
//
//	//ROOT
//	Root rootOp = Root(&printOpProj);
//	printOpProj.setParent(&rootOp);
//	rootOp.produce();
//
//	//Run function
//	ctx.prepareFunction(ctx.getGlobalFunction());
//
//	//Close all open files & clear
//	pgSailors->finish();
//	pgReserves->finish();
//	RawCatalog& catalog = RawCatalog::getInstance();
//	catalog.clear();
//	EXPECT_TRUE(true);
//}

