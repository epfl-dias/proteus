/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Laboratory (DIAS)
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
#include "expressions/expressions-generator.hpp"
#include "plugins/plugins.hpp"
#include "util/functions.hpp"

#ifdef DEBUG
#define DEBUG_HASH
#endif

// XXX Is a visitor pattern overkill? The HasherVisitor does not recurse
//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for generating the appropriate hash for an
// Expression
//===---------------------------------------------------------------------------===//
class ExpressionHasherVisitor : public ExprVisitor {
 public:
  ExpressionHasherVisitor(Context *const context,
                          const OperatorState &currState)
      : context(context), currState(currState), activeRelation("") {}
  ExpressionHasherVisitor(Context *const context,
                          const OperatorState &currState, string activeRelation)
      : context(context),
        currState(currState),
        activeRelation(activeRelation) {}
  ProteusValue visit(const expressions::IntConstant *e);
  ProteusValue visit(const expressions::Int64Constant *e);
  ProteusValue visit(const expressions::DateConstant *e);
  ProteusValue visit(const expressions::FloatConstant *e);
  ProteusValue visit(const expressions::BoolConstant *e);
  ProteusValue visit(const expressions::StringConstant *e);
  ProteusValue visit(const expressions::DStringConstant *e);
  ProteusValue visit(const expressions::InputArgument *e);
  ProteusValue visit(const expressions::RecordProjection *e);
  ProteusValue visit(const expressions::IfThenElse *e);
  // XXX Do binary operators require explicit handling of NULL?
  ProteusValue visit(const expressions::EqExpression *e);
  ProteusValue visit(const expressions::NeExpression *e);
  ProteusValue visit(const expressions::GeExpression *e);
  ProteusValue visit(const expressions::GtExpression *e);
  ProteusValue visit(const expressions::LeExpression *e);
  ProteusValue visit(const expressions::LtExpression *e);
  ProteusValue visit(const expressions::AddExpression *e);
  ProteusValue visit(const expressions::SubExpression *e);
  ProteusValue visit(const expressions::MultExpression *e);
  ProteusValue visit(const expressions::DivExpression *e);
  ProteusValue visit(const expressions::ModExpression *e);
  ProteusValue visit(const expressions::AndExpression *e);
  ProteusValue visit(const expressions::OrExpression *e);
  ProteusValue visit(const expressions::RecordConstruction *e);
  ProteusValue visit(const expressions::ProteusValueExpression *e);
  ProteusValue visit(const expressions::MinExpression *e);
  ProteusValue visit(const expressions::MaxExpression *e);
  ProteusValue visit(const expressions::HashExpression *e);
  ProteusValue visit(const expressions::NegExpression *e);
  ProteusValue visit(const expressions::ExtractExpression *e);
  ProteusValue visit(const expressions::TestNullExpression *e);
  ProteusValue visit(const expressions::CastExpression *e);

  void setActiveRelation(string relName) { activeRelation = relName; }
  string getActiveRelation(string relName) { return activeRelation; }

 private:
  Context *const context;
  const OperatorState &currState;

  string activeRelation;

  ProteusValue hashInt32(const expressions::Expression *e);
  ProteusValue hashPrimitive(const expressions::Expression *e);
};

ProteusValue hashInt32(ProteusValue v, Context *context);
ProteusValue hashPrimitive(ProteusValue v, typeID type, Context *context);

#endif /* EXPRESSIONS_HASHER_VISITOR_HPP_ */
