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

#ifndef EXPRESSIONS_HASHER_VISITOR_HPP_
#define EXPRESSIONS_HASHER_VISITOR_HPP_

#include "common/common.hpp"
#include "plugins/plugins.hpp"
#include "util/raw-functions.hpp"
#include "expressions/expressions-generator.hpp"

#ifdef DEBUG
#define DEBUG_HASH
#endif

//XXX Is a visitor pattern overkill? The HasherVisitor does not recurse
//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for generating the appropriate hash for an Expression
//===---------------------------------------------------------------------------===//
class ExpressionHasherVisitor: public ExprVisitor
{
public:
	ExpressionHasherVisitor(RawContext* const context,
			const OperatorState& currState) :
			context(context), currState(currState),
			activeRelation("")											{}
	ExpressionHasherVisitor(RawContext* const context,
			const OperatorState& currState, string activeRelation) :
			context(context), currState(currState),
			activeRelation(activeRelation)								{}
	RawValue visit(expressions::IntConstant *e);
	RawValue visit(expressions::FloatConstant *e);
	RawValue visit(expressions::BoolConstant *e);
	RawValue visit(expressions::StringConstant *e);
	RawValue visit(expressions::InputArgument *e);
	RawValue visit(expressions::RecordProjection *e);
	RawValue visit(expressions::IfThenElse *e);
	//XXX Do binary operators require explicit handling of NULL?
	RawValue visit(expressions::EqExpression *e);
	RawValue visit(expressions::NeExpression *e);
	RawValue visit(expressions::GeExpression *e);
	RawValue visit(expressions::GtExpression *e);
	RawValue visit(expressions::LeExpression *e);
	RawValue visit(expressions::LtExpression *e);
	RawValue visit(expressions::AddExpression *e);
	RawValue visit(expressions::SubExpression *e);
	RawValue visit(expressions::MultExpression *e);
	RawValue visit(expressions::DivExpression *e);
	RawValue visit(expressions::AndExpression *e);
	RawValue visit(expressions::OrExpression *e);
	RawValue visit(expressions::RecordConstruction *e);

	void setActiveRelation(string relName)		{ activeRelation = relName; }
	string getActiveRelation(string relName)	{ return activeRelation; }
private:
	RawContext* const context;
	const OperatorState& currState;

	string activeRelation;

};

#endif /* EXPRESSIONS_HASHER_VISITOR_HPP_ */
