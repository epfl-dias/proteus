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

#include <platform/common/common.hpp>

#include "expressions-generator.hpp"
#include "lib/util/functions.hpp"
#include "olap/plugins/plugins.hpp"

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
  ProteusValue visit(const expressions::IntConstant *e) override;
  ProteusValue visit(const expressions::Int64Constant *e) override;
  ProteusValue visit(const expressions::DateConstant *e) override;
  ProteusValue visit(const expressions::FloatConstant *e) override;
  ProteusValue visit(const expressions::BoolConstant *e) override;
  ProteusValue visit(const expressions::StringConstant *e) override;
  ProteusValue visit(const expressions::DStringConstant *e) override;
  ProteusValue visit(const expressions::InputArgument *e) override;
  ProteusValue visit(const expressions::RecordProjection *e) override;
  ProteusValue visit(const expressions::IfThenElse *e) override;
  // XXX Do binary operators require explicit handling of nullptr?
  ProteusValue visit(const expressions::EqExpression *e) override;
  ProteusValue visit(const expressions::NeExpression *e) override;
  ProteusValue visit(const expressions::GeExpression *e) override;
  ProteusValue visit(const expressions::GtExpression *e) override;
  ProteusValue visit(const expressions::LeExpression *e) override;
  ProteusValue visit(const expressions::LtExpression *e) override;
  ProteusValue visit(const expressions::AddExpression *e) override;
  ProteusValue visit(const expressions::SubExpression *e) override;
  ProteusValue visit(const expressions::MultExpression *e) override;
  ProteusValue visit(const expressions::DivExpression *e) override;
  ProteusValue visit(const expressions::ModExpression *e) override;
  ProteusValue visit(const expressions::AndExpression *e) override;
  ProteusValue visit(const expressions::OrExpression *e) override;
  ProteusValue visit(const expressions::RecordConstruction *e) override;
  ProteusValue visit(const expressions::ProteusValueExpression *e) override;
  ProteusValue visit(const expressions::MinExpression *e) override;
  ProteusValue visit(const expressions::MaxExpression *e) override;
  ProteusValue visit(const expressions::RandExpression *e) override;
  ProteusValue visit(const expressions::HintExpression *e) override;
  ProteusValue visit(const expressions::HashExpression *e) override;
  ProteusValue visit(const expressions::RefExpression *e) override;
  ProteusValue visit(const expressions::AssignExpression *e) override;
  ProteusValue visit(const expressions::NegExpression *e) override;
  ProteusValue visit(const expressions::ExtractExpression *e) override;
  ProteusValue visit(const expressions::TestNullExpression *e) override;
  ProteusValue visit(const expressions::CastExpression *e) override;

  ProteusValue visit(const expressions::ShiftLeftExpression *e) override;
  ProteusValue visit(
      const expressions::LogicalShiftRightExpression *e) override;
  ProteusValue visit(
      const expressions::ArithmeticShiftRightExpression *e) override;
  ProteusValue visit(const expressions::XORExpression *e) override;

  void setActiveRelation(string relName) { activeRelation = relName; }
  string getActiveRelation(string relName) { return activeRelation; }

 private:
  Context *const context;
  const OperatorState &currState;

  string activeRelation;

  ProteusValue hashInt32(const expressions::Expression *e);
  ProteusValue hashInt64(const expressions::Expression *e);
  ProteusValue hashPrimitive(const expressions::Expression *e);
};

ProteusValue hashInt32(ProteusValue v, Context *context);
ProteusValue hashInt64(ProteusValue v, Context *context);
ProteusValue hashPrimitive(ProteusValue v, typeID type, Context *context);

#endif /* EXPRESSIONS_HASHER_VISITOR_HPP_ */
