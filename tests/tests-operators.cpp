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
#include "util/raw-context.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/print.hpp"
#include "operators/root.hpp"
#include "operators/join.hpp"
#include "operators/unnest.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/reduce.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
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
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/input.csv");
	PrimitiveType* intType = new IntType();
	RecordAttribute* attr1 = new RecordAttribute(1,filename,string("att1"),intType);
	RecordAttribute* attr2 = new RecordAttribute(2,filename,string("att2"),intType);
	RecordAttribute* attr3 = new RecordAttribute(3,filename,string("att3"),intType);

	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);
	attrList.push_back(attr3);
	RecordType rec1 = RecordType(attrList);
	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

	CSVPlugin *pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);

	Scan scan = Scan(&ctx, *pg);

	Root rootOp = Root(&scan);
	scan.setParent(&rootOp);
	rootOp.produce();

	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Relational, SPJ) {

	RawContext ctx = RawContext("RelationalSPJ");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/input.csv");
	PrimitiveType* intType = new IntType();//PrimitiveType(Int);
	RecordAttribute* attr1 = new RecordAttribute(1,filename,string("att1"),intType);
	RecordAttribute* attr2 = new RecordAttribute(2,filename,string("att2"),intType);
	RecordAttribute* attr3 = new RecordAttribute(3,filename,string("att3"),intType);
	list<RecordAttribute*> attrList;
	attrList.push_back(attr1);
	attrList.push_back(attr2);
	attrList.push_back(attr3);

	RecordType rec1 = RecordType(attrList);
	vector<RecordAttribute*> whichFields;
	whichFields.push_back(attr1);
	whichFields.push_back(attr2);

	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);

	Scan scan = Scan(&ctx, *pg);

	//SELECT
	expressions::Expression* lhsArg = new expressions::InputArgument(new IntType(),0);
	expressions::Expression* lhs = new expressions::RecordProjection(new IntType(),lhsArg,*attr1);
	expressions::Expression* rhs = new expressions::IntConstant(555);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);
	Select sel = Select(predicate,&scan);
	scan.setParent(&sel);


	//SCAN2
	string filename2 = string("inputs/input2.csv");
	RecordAttribute* attr1_f2 = new RecordAttribute(1,filename2,string("att1"),intType);
	RecordAttribute* attr2_f2 = new RecordAttribute(2,filename2,string("att2"),intType);
	RecordAttribute* attr3_f2 = new RecordAttribute(3,filename2,string("att3"),intType);
	list<RecordAttribute*> attrList2;
	attrList2.push_back(attr1_f2);
	attrList2.push_back(attr1_f2);
	attrList2.push_back(attr1_f2);
	RecordType rec2 = RecordType(attrList2);
	vector<RecordAttribute*> whichFields2;
	whichFields2.push_back(attr1_f2);
	whichFields2.push_back(attr2_f2);

	CSVPlugin* pg2 = new CSVPlugin(&ctx, filename2, rec2, whichFields2);
	catalog.registerPlugin(filename2,pg2);

	Scan scan2 = Scan(&ctx, *pg2);


	//JOIN
	expressions::Expression* leftArg = new expressions::InputArgument(intType,0);
	expressions::Expression* left = new expressions::RecordProjection(intType,leftArg,*attr2);
	expressions::Expression* rightArg = new expressions::InputArgument(intType,1);
	expressions::Expression* right = new expressions::RecordProjection(intType,rightArg,*attr2_f2);
	expressions::BinaryExpression* joinPred = new expressions::EqExpression(new BoolType(),left,right);

	vector<materialization_mode> outputModes;
	outputModes.insert(outputModes.begin(),EAGER);
	outputModes.insert(outputModes.begin(),EAGER);
	Materializer* mat = new Materializer(whichFields,outputModes);

	Join join = Join(joinPred,sel,scan2, "join1", *mat);
	sel.setParent(&join);
	scan2.setParent(&join);


	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* argProj = new expressions::RecordProjection(new IntType(),leftArg,*attr1);
	Print printOpProj = Print(debugInt,argProj,&join);
	join.setParent(&printOpProj);


	//ROOT
	Root rootOp = Root(&printOpProj);
	printOpProj.setParent(&rootOp);

	ctx.prepareFunction(ctx.getGlobalFunction());

	//Close all open files & clear
	pg->finish();
	pg2->finish();
	catalog.clear();
	EXPECT_TRUE(true);
}

TEST(Hierarchical, TwoProjections) {
	RawContext ctx = RawContext("testFunction-ScanJSON-jsmn");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/jsmnDeeper.json");

	IntType intType = IntType();

	string c1Name = string("c1");
	RecordAttribute c1 = RecordAttribute(1, fname, c1Name, &intType);
	string c2Name = string("c2");
	RecordAttribute c2 = RecordAttribute(2, fname, c2Name, &intType);
	list<RecordAttribute*> attsNested = list<RecordAttribute*>();
	attsNested.push_back(&c1);
	attsNested.push_back(&c2);
	RecordType nested = RecordType(attsNested);

	string attrName = string("a");
	string attrName2 = string("b");
	string attrName3 = string("c");
	RecordAttribute attr = RecordAttribute(1, fname, attrName, &intType);
	RecordAttribute attr2 = RecordAttribute(2, fname, attrName2, &intType);
	RecordAttribute attr3 = RecordAttribute(3, fname, attrName3, &nested);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&attr);
	atts.push_back(&attr2);
	atts.push_back(&attr3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);
	Scan scan = Scan(&ctx, pg);

	//SELECT
	expressions::Expression* lhsArg = new expressions::InputArgument(&inner, 0);
	expressions::Expression* lhs_ = new expressions::RecordProjection(&nested,
			lhsArg, attr3);
	expressions::Expression* lhs = new expressions::RecordProjection(&intType,
			lhs_, c2);
	expressions::Expression* rhs = new expressions::IntConstant(110);

	//obj.c.c2 > 110 --> Only 1 must qualify
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	Select sel = Select(predicate, &scan);
	scan.setParent(&sel);

	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&intType, lhsArg, attr);
	Print printOp = Print(debugInt, proj, &sel);
	sel.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();

	EXPECT_TRUE(true);
}

TEST(Hierarchical, Unnest) {
	RawContext ctx = RawContext("testFunction-unnestJSON");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/employees.json");

	IntType intType = IntType();
	StringType stringType = StringType();

	string childName = string("name");
	RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
	string childAge = string("age");
	RecordAttribute child2 = RecordAttribute(1, fname, childAge, &intType);
	list<RecordAttribute*> attsNested = list<RecordAttribute*>();
	attsNested.push_back(&child1);
	RecordType nested = RecordType(attsNested);
	ListType nestedCollection = ListType(nested);

	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	string empChildren = string("children");
	RecordAttribute emp3 = RecordAttribute(3, fname, empChildren,
			&nestedCollection);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);
	atts.push_back(&emp3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);
	Scan scan = Scan(&ctx, pg);

	expressions::Expression* inputArg = new expressions::InputArgument(&inner,
			0);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&stringType, inputArg, emp3);
	string nestedName = "c";
	Path path = Path(nestedName, proj);

	expressions::Expression* lhs = new expressions::BoolConstant(true);
	expressions::Expression* rhs = new expressions::BoolConstant(true);
	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), lhs, rhs);

	Unnest unnestOp = Unnest(predicate, path, &scan);
	scan.setParent(&unnestOp);

	//New record type:
	string originalRecordName = "e";
	RecordAttribute recPrev = RecordAttribute(1, fname, originalRecordName,
			&inner);
	RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName,
			&nested);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recPrev);
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::Expression* nestedArg = new expressions::InputArgument(
			&unnestedType, 0);

	RecordAttribute toPrint = RecordAttribute(-1, fname + "." + empChildren,
			childAge, &intType);

	expressions::RecordProjection* projToPrint =
			new expressions::RecordProjection(&intType, nestedArg, toPrint);
	Print printOp = Print(debugInt, projToPrint, &unnestOp);
	unnestOp.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();

	EXPECT_TRUE(true);
}

TEST(Hierarchical, UnnestDeeper) {
	RawContext ctx = RawContext("testFunction-unnestJSON");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/employeesDeeper.json");

	IntType intType = IntType();
	StringType stringType = StringType();

	/**
	 * SCHEMA
	 */
	string ages = string("ages");
	ListType childrenAgesType = ListType(intType);
	RecordAttribute childrenAges = RecordAttribute(1, fname, ages, &childrenAgesType);
	list<RecordAttribute*> attsChildren = list<RecordAttribute*>();
	attsChildren.push_back(&childrenAges);
	RecordType children = RecordType(attsChildren);

	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	string empChildren = string("children");
	RecordAttribute emp3 = RecordAttribute(3, fname, empChildren, &children);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);
	atts.push_back(&emp3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	/**
	 * SCAN
	 */
	jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname,&pg);
	Scan scan = Scan(&ctx,pg);

	/**
	 * UNNEST
	 */
	expressions::Expression* inputArg = new expressions::InputArgument(&inner, 0);
	expressions::RecordProjection* proj = new expressions::RecordProjection(&children,inputArg,emp3);
	expressions::RecordProjection* projDeeper = new expressions::RecordProjection(&childrenAgesType,proj,childrenAges);
	string nestedName = "c";
	Path path = Path(nestedName,projDeeper);

	expressions::Expression* lhs = new expressions::BoolConstant(true);
	expressions::Expression* rhs = new expressions::BoolConstant(true);
	expressions::Expression* predicate = new expressions::EqExpression(new BoolType(),lhs,rhs);

	Unnest unnestOp = Unnest(predicate,path,&scan);
	scan.setParent(&unnestOp);

	//New record type:
	string originalRecordName = "e";
	RecordAttribute recPrev = RecordAttribute(1, fname, originalRecordName, &inner);
	RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName, &intType);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recPrev);
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	expressions::Expression* nestedArg = new expressions::InputArgument(&unnestedType, 0);

	RecordAttribute toPrint = RecordAttribute(2,
			fname+"."+empChildren+"."+ages,
			activeLoop,
			&intType);

	expressions::RecordProjection* projToPrint = new expressions::RecordProjection(&intType,nestedArg,toPrint);
	Print printOp = Print(debugInt,projToPrint,&unnestOp);
	unnestOp.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();

	EXPECT_TRUE(true);
}

TEST(Hierarchical, UnnestFiltering) {
	RawContext ctx = RawContext("testFunction-unnestJSON");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/employees.json");

	IntType intType = IntType();
	StringType stringType = StringType();

	string childName = string("name");
	RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
	string childAge = string("age");
	RecordAttribute child2 = RecordAttribute(1, fname, childAge, &intType);
	list<RecordAttribute*> attsNested = list<RecordAttribute*>();
	attsNested.push_back(&child1);
	RecordType nested = RecordType(attsNested);
	ListType nestedCollection = ListType(nested);

	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	string empChildren = string("children");
	RecordAttribute emp3 = RecordAttribute(3, fname, empChildren,
			&nestedCollection);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);
	atts.push_back(&emp3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);
	Scan scan = Scan(&ctx, pg);

	/**
	 * UNNEST
	 */
	expressions::Expression* inputArg = new expressions::InputArgument(&inner,
			0);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&stringType, inputArg, emp3);
	string nestedName = "c";
	Path path = Path(nestedName, proj);

	//New record type as the result of unnest:
	string originalRecordName = "e";
	RecordAttribute recPrev = RecordAttribute(1, fname, originalRecordName,
			&inner);
	RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName,
			&nested);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recPrev);
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);
	expressions::Expression* nestedArg = new expressions::InputArgument(
			&unnestedType, 0);
	RecordAttribute toFilter = RecordAttribute(-1, fname + "." + empChildren,
			childAge, &intType);
	expressions::RecordProjection* projToFilter =
			new expressions::RecordProjection(&intType, nestedArg, toFilter);
	expressions::Expression* lhs = projToFilter;
	expressions::Expression* rhs = new expressions::IntConstant(20);
	expressions::Expression* predicate = new expressions::GtExpression(
			new BoolType(), lhs, rhs);

	Unnest unnestOp = Unnest(predicate, path, &scan);
	scan.setParent(&unnestOp);

	//PRINT
	Function* debugInt = ctx.getFunction("printi");
	Print printOp = Print(debugInt, projToFilter, &unnestOp);
	unnestOp.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();

	EXPECT_TRUE(true);
}

TEST(Generic, ReduceNumeric) {
	RawContext ctx = RawContext("reduceNumeric");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1,filename, string("sid"),intType);
	RecordAttribute* sname = new RecordAttribute(2,filename, string("sname"),stringType);
	RecordAttribute* rating = new RecordAttribute(3,filename, string("rating"),intType);
	RecordAttribute* age = new RecordAttribute(4,filename, string("age"),floatType);

	list<RecordAttribute*> attrList;
	attrList.push_back(sid);
	attrList.push_back(sname);
	attrList.push_back(rating);
	attrList.push_back(age);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(sid);
	whichFields.push_back(age);

	CSVPlugin* pg = new CSVPlugin(&ctx,filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);
	Scan scan = Scan(&ctx,*pg);

	/**
	 * REDUCE
	 */
	expressions::Expression* arg = new expressions::InputArgument(&rec1,0);
	expressions::Expression* outputExpr = new expressions::RecordProjection(intType,arg,*sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,arg,*age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);
//	Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
//	Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);
	Reduce reduce = Reduce(MAX, outputExpr, predicate, &scan, &ctx);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();
}

TEST(Generic, ReduceBoolean) {
	RawContext ctx = RawContext("reduceBoolean");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/bills.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* boolType = new BoolType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* category = new RecordAttribute(1,filename,string("category"),stringType);
	RecordAttribute* amount = new RecordAttribute(2,filename,string("amount"),intType);
	RecordAttribute* isPaid = new RecordAttribute(3,filename,string("isPaid"),boolType);

	list<RecordAttribute*> attrList;
	attrList.push_back(category);
	attrList.push_back(amount);
	attrList.push_back(isPaid);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(amount);
	whichFields.push_back(isPaid);

	CSVPlugin* pg = new CSVPlugin(&ctx,filename, rec1, whichFields);
	catalog.registerPlugin(filename,pg);
	Scan scan = Scan(&ctx,*pg);

	/**
	 * REDUCE
	 */
	expressions::Expression* arg = new expressions::InputArgument(&rec1,0);
	expressions::Expression* outputExpr = new expressions::RecordProjection(boolType,arg,*isPaid);

	expressions::Expression* lhs = new expressions::RecordProjection(intType,arg,*amount);
	expressions::Expression* rhs = new expressions::IntConstant(1400);
	expressions::Expression* predicate = new expressions::GtExpression(new BoolType(),lhs,rhs);
	Reduce reduce = Reduce(AND, outputExpr, predicate, &scan, &ctx);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();
}

TEST(Generic, IfThenElse) {
	RawContext ctx = RawContext("ifThenElseExpr");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/bills.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* boolType = new BoolType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* category = new RecordAttribute(1, filename,
			string("category"), stringType);
	RecordAttribute* amount = new RecordAttribute(2, filename, string("amount"),
			intType);
	RecordAttribute* isPaid = new RecordAttribute(3, filename, string("isPaid"),
			boolType);

	list<RecordAttribute*> attrList;
	attrList.push_back(category);
	attrList.push_back(amount);
	attrList.push_back(isPaid);

	RecordType rec1 = RecordType(attrList);

	vector<RecordAttribute*> whichFields;
	whichFields.push_back(amount);

	CSVPlugin* pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	expressions::Expression* trueCons = new expressions::BoolConstant(true);
	expressions::Expression* falseCons = new expressions::BoolConstant(false);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0);
	expressions::Expression* ifLhs = new expressions::RecordProjection(boolType,
			arg, *amount);
	expressions::Expression* ifRhs = new expressions::IntConstant(200);
	expressions::Expression* ifCond = new expressions::GtExpression(boolType,
			ifLhs, ifRhs);
	expressions::Expression* ifElse = new expressions::IfThenElse(boolType,
			ifCond, trueCons, falseCons);

	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), trueCons, trueCons);
	Reduce reduce = Reduce(AND, ifElse, predicate, &scan, &ctx);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();
}

TEST(Hierarchical,OuterUnnest1)
{
	RawContext ctx = RawContext("testFunction-outerUnnestJSON");
	RawCatalog& catalog = RawCatalog::getInstance();

	string fname = string("inputs/employees.json");

	IntType intType = IntType();
	//FloatType floatType = FloatType();
	StringType stringType = StringType();

	string childName = string("name");
	RecordAttribute child1 = RecordAttribute(1, fname, childName, &stringType);
	string childAge = string("age");
	RecordAttribute child2 = RecordAttribute(1, fname, childAge, &intType);
	list<RecordAttribute*> attsNested = list<RecordAttribute*>();
	attsNested.push_back(&child1);
	RecordType nested = RecordType(attsNested);
	ListType nestedCollection = ListType(nested);

	string empName = string("name");
	RecordAttribute emp1 = RecordAttribute(1, fname, empName, &stringType);
	string empAge = string("age");
	RecordAttribute emp2 = RecordAttribute(2, fname, empAge, &intType);
	string empChildren = string("children");
	RecordAttribute emp3 = RecordAttribute(3, fname, empChildren,
			&nestedCollection);

	list<RecordAttribute*> atts = list<RecordAttribute*>();
	atts.push_back(&emp1);
	atts.push_back(&emp2);
	atts.push_back(&emp3);

	RecordType inner = RecordType(atts);
	ListType documentType = ListType(inner);

	jsmn::JSONPlugin pg = jsmn::JSONPlugin(&ctx, fname, &documentType);
	catalog.registerPlugin(fname, &pg);
	Scan scan = Scan(&ctx, pg);

	expressions::Expression* inputArg = new expressions::InputArgument(&inner,
			0);
	expressions::RecordProjection* proj = new expressions::RecordProjection(
			&nestedCollection, inputArg, emp3);
	string nestedName = "c";
	Path path = Path(nestedName, proj);

	expressions::Expression* lhs = new expressions::BoolConstant(true);
	expressions::Expression* rhs = new expressions::BoolConstant(true);
	expressions::Expression* predicate = new expressions::EqExpression(
			new BoolType(), lhs, rhs);

	OuterUnnest unnestOp = OuterUnnest(predicate, path, &scan);
	scan.setParent(&unnestOp);

	//New record type:
	string originalRecordName = "e";
	RecordAttribute recPrev = RecordAttribute(1, fname, originalRecordName,
			&inner);
	RecordAttribute recUnnested = RecordAttribute(2, fname, nestedName,
			&nested);
	list<RecordAttribute*> attsUnnested = list<RecordAttribute*>();
	attsUnnested.push_back(&recPrev);
	attsUnnested.push_back(&recUnnested);
	RecordType unnestedType = RecordType(attsUnnested);

	//PRINT
	//Function* debugInt = ctx.getFunction("printFloat");
	Function* debugInt = ctx.getFunction("printi");
	expressions::Expression* nestedArg = new expressions::InputArgument(
			&unnestedType, 0);

	RecordAttribute toPrint = RecordAttribute(-1, fname + "." + empChildren,
			childAge, &intType);

	expressions::RecordProjection* projToPrint =
			new expressions::RecordProjection(&intType, nestedArg, toPrint);
	Print printOp = Print(debugInt, projToPrint, &unnestOp);
	unnestOp.setParent(&printOp);

	//ROOT
	Root rootOp = Root(&printOp);
	printOp.setParent(&rootOp);
	rootOp.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg.finish();
	catalog.clear();
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
