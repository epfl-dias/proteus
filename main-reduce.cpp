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


/* Sanity check: Previous reduce() */
void reduceNumeric();
/* New version */
void reduceMultipleNumeric();

int main() {

	reduceMultipleNumeric();
	reduceNumeric();

}

void reduceMultipleNumeric()
{
	RawContext& ctx = *prepareContext("reduceNumeric");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
			floatType);

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
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop,pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression* outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			lhs, rhs);
//	Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
//	Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	vector<string> aggrLabels;
	/* Aggregate 1 */
	accs.push_back(MAX);
	outputExprs.push_back(outputExpr);
	/* Aggregate 2 */
	accs.push_back(SUM);
	outputExprs.push_back(outputExpr);
	opt::Reduce reduce = opt::Reduce(accs, outputExprs, predicate, &scan, &ctx);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();
}

void reduceNumeric()
{
	RawContext& ctx = *prepareContext("reduceNumeric");
	RawCatalog& catalog = RawCatalog::getInstance();

	//SCAN1
	string filename = string("inputs/sailors.csv");
	PrimitiveType* intType = new IntType();
	PrimitiveType* floatType = new FloatType();
	PrimitiveType* stringType = new StringType();
	RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
			intType);
	RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
			stringType);
	RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
			intType);
	RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
			floatType);

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
	catalog.registerPlugin(filename, pg);
	Scan scan = Scan(&ctx, *pg);

	/**
	 * REDUCE
	 */
	RecordAttribute projTuple = RecordAttribute(filename, activeLoop,pg->getOIDType());
	list<RecordAttribute> projections = list<RecordAttribute>();
	projections.push_back(projTuple);
	projections.push_back(*sid);
	projections.push_back(*age);

	expressions::Expression* arg = new expressions::InputArgument(&rec1, 0,
			projections);
	expressions::Expression* outputExpr = new expressions::RecordProjection(
			intType, arg, *sid);

	expressions::Expression* lhs = new expressions::RecordProjection(floatType,
			arg, *age);
	expressions::Expression* rhs = new expressions::FloatConstant(40.0);
	expressions::Expression* predicate = new expressions::GtExpression(
			lhs, rhs);
	Reduce reduce = Reduce(SUM, outputExpr, predicate, &scan, &ctx);
//	Reduce reduce = Reduce(MULTIPLY, outputExpr, predicate, &scan, &ctx);
//	Reduce reduce = Reduce(MAX, outputExpr, predicate, &scan, &ctx);
	scan.setParent(&reduce);

	reduce.produce();

	//Run function
	ctx.prepareFunction(ctx.getGlobalFunction());

	pg->finish();
	catalog.clear();
}
