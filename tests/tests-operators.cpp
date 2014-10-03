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
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "operators/join.hpp"
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

TEST(Relational, Scan) {
	RawContext ctx = RawContext("RelationalScan");

	//SCAN1
	string filename = string("inputs/input.csv");
	PrimitiveType* intType = new IntType();//PrimitiveType(Int);
	RecordAttribute* attr1 = new RecordAttribute(1,filename+"_"+string("att1"),intType);
	RecordAttribute* attr2 = new RecordAttribute(2,filename+"_"+string("att2"),intType);
	RecordAttribute* attr3 = new RecordAttribute(3,filename+"_"+string("att3"),intType);

	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);
	attrList.push_back(attr3);
	RecordType rec1 = RecordType(attrList);
	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

	CSVPlugin *pg = new CSVPlugin(&ctx, filename, rec1, whichFields);

	Scan scan = Scan(&ctx, *pg);

	Root rootOp = Root(&scan);
	scan.setParent(&rootOp);
	rootOp.produce();

	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Relational, SPJ) {

	RawContext ctx = RawContext("RelationalSPJ");

	//SCAN1
	string filename = string("inputs/input.csv");
	PrimitiveType* intType = new IntType();//PrimitiveType(Int);
	RecordAttribute* attr1 = new RecordAttribute(1,filename+"_"+string("att1"),intType);
	RecordAttribute* attr2 = new RecordAttribute(2,filename+"_"+string("att2"),intType);
	RecordAttribute* attr3 = new RecordAttribute(3,filename+"_"+string("att3"),intType);
	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);
	attrList.push_back(attr3);

	RecordType rec1 = RecordType(attrList);
	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);

	Scan scan = Scan(&ctx, *pg);


	//SELECT
	string argName = filename+"_"+string("att1");
	expressions::Expression* lhs = new expressions::InputArgument(new IntType(),2,argName);
	expressions::Expression* rhs = new expressions::IntConstant(555);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);

	Select sel = Select(predicate,&scan);
	scan.setParent(&sel);


	//SCAN2
	string filename2 = string("inputs/input2.csv");
	RecordAttribute* attr1_f2 = new RecordAttribute(1,filename2+"_"+string("att1"),intType);
	RecordAttribute* attr2_f2 = new RecordAttribute(2,filename2+"_"+string("att2"),intType);
	RecordAttribute* attr3_f2 = new RecordAttribute(3,filename2+"_"+string("att3"),intType);
	list<RecordAttribute*> attrList2;
	attrList2.push_back(attr1_f2);
	attrList2.push_back(attr1_f2);
	attrList2.push_back(attr1_f2);
	RecordType rec2 = RecordType(attrList2);
	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(attr1_f2);
	whichFields2.push_back(attr2_f2);

	CSVPlugin* pg2 = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	Scan scan2 = Scan(&ctx, *pg2);


	//JOIN
	string argName2 = filename2+"_"+string("att2");
	string argName_ = filename+"_"+string("att2");
	expressions::InputArgument* left = new expressions::InputArgument(new IntType(),2,argName_);
	expressions::InputArgument* right = new expressions::InputArgument(new IntType(),2,argName2);
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);

	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(),EAGER);
	outputModes.insert(outputModes.begin(),EAGER);
	Materializer* mat = new Materializer(whichFields,outputModes);

	Join join = Join(joinPred,sel,scan2, "join1", *mat);
	sel.setParent(&join);
	scan2.setParent(&join);


	//PRINT
	string argNameProj = filename+"_"+string("att1");
	Function* debugInt = ctx.getFunction("printi");
	expressions::InputArgument* argProj = new expressions::InputArgument(new IntType(),1,argNameProj);
	Print printOpProj = Print(debugInt,argProj,&join);
	join.setParent(&printOpProj);


	//ROOT
	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);

	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	pg2->finish();
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.clear();
	EXPECT_TRUE(true);
}

// Step 3. Call RUN_ALL_TESTS() in main().
//
// We do this by linking in src/gtest_main.cc file, which consists of
// a main() function which calls RUN_ALL_TESTS() for us.
//
// This runs all the tests you've defined, prints the result, and
// returns 0 if successful, or 1 otherwise.
//
// Did you notice that we didn't register the tests?  The
// RUN_ALL_TESTS() macro magically knows about all the tests we
// defined.  Isn't this convenient?
