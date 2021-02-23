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

#ifndef EXPRESSIONS_DOT_VISITOR_HPP_
#define EXPRESSIONS_DOT_VISITOR_HPP_

#include <platform/common/common.hpp>

#include "expressions-generator.hpp"
#include "expressions-hasher.hpp"
#include "lib/util/functions.hpp"
#include "olap/plugins/plugins.hpp"
//#include "values/expressionTypes.hpp"

//#ifdef DEBUG
#define DEBUG_DOT
//#endif

//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for evaluating dot equality
//===---------------------------------------------------------------------------===//
class ExpressionDotVisitor : public ExprTandemVisitor {
 public:
  ExpressionDotVisitor(Context *const context,
                       const OperatorState &currStateLeft,
                       const OperatorState &currStateRight)
      : context(context),
        currStateLeft(currStateLeft),
        currStateRight(currStateRight) {}
  ProteusValue visit(const expressions::IntConstant *e1,
                     const expressions::IntConstant *e2) override;
  ProteusValue visit(const expressions::Int64Constant *e1,
                     const expressions::Int64Constant *e2) override;
  ProteusValue visit(const expressions::DateConstant *e1,
                     const expressions::DateConstant *e2) override;
  ProteusValue visit(const expressions::FloatConstant *e1,
                     const expressions::FloatConstant *e2) override;
  ProteusValue visit(const expressions::BoolConstant *e1,
                     const expressions::BoolConstant *e2) override;
  ProteusValue visit(const expressions::StringConstant *e1,
                     const expressions::StringConstant *e2) override;
  ProteusValue visit(const expressions::DStringConstant *e1,
                     const expressions::DStringConstant *e2) override;
  ProteusValue visit(const expressions::InputArgument *e1,
                     const expressions::InputArgument *e2) override;
  ProteusValue visit(const expressions::PlaceholderExpression *e1,
                     const expressions::PlaceholderExpression *e2) override;
  ProteusValue visit(const expressions::ProteusValueExpression *e1,
                     const expressions::ProteusValueExpression *e2) override;
  ProteusValue visit(const expressions::RecordProjection *e1,
                     const expressions::RecordProjection *e2) override;
  ProteusValue visit(const expressions::IfThenElse *e1,
                     const expressions::IfThenElse *e2) override;
  ProteusValue visit(const expressions::EqExpression *e1,
                     const expressions::EqExpression *e2) override;
  ProteusValue visit(const expressions::NeExpression *e1,
                     const expressions::NeExpression *e2) override;
  ProteusValue visit(const expressions::GeExpression *e1,
                     const expressions::GeExpression *e2) override;
  ProteusValue visit(const expressions::GtExpression *e1,
                     const expressions::GtExpression *e2) override;
  ProteusValue visit(const expressions::LeExpression *e1,
                     const expressions::LeExpression *e2) override;
  ProteusValue visit(const expressions::LtExpression *e1,
                     const expressions::LtExpression *e2) override;
  ProteusValue visit(const expressions::AddExpression *e1,
                     const expressions::AddExpression *e2) override;
  ProteusValue visit(const expressions::SubExpression *e1,
                     const expressions::SubExpression *e2) override;
  ProteusValue visit(const expressions::MultExpression *e1,
                     const expressions::MultExpression *e2) override;
  ProteusValue visit(const expressions::DivExpression *e1,
                     const expressions::DivExpression *e2) override;
  ProteusValue visit(const expressions::ModExpression *e1,
                     const expressions::ModExpression *e2) override;
  ProteusValue visit(const expressions::AndExpression *e1,
                     const expressions::AndExpression *e2) override;
  ProteusValue visit(const expressions::OrExpression *e1,
                     const expressions::OrExpression *e2) override;
  ProteusValue visit(const expressions::RecordConstruction *e1,
                     const expressions::RecordConstruction *e2) override;
  ProteusValue visit(const expressions::MaxExpression *e1,
                     const expressions::MaxExpression *e2) override;
  ProteusValue visit(const expressions::MinExpression *e1,
                     const expressions::MinExpression *e2) override;
  ProteusValue visit(const expressions::RandExpression *e1,
                     const expressions::RandExpression *e2) override;
  ProteusValue visit(const expressions::HintExpression *e1,
                     const expressions::HintExpression *e2) override;
  ProteusValue visit(const expressions::HashExpression *e1,
                     const expressions::HashExpression *e2) override;
  ProteusValue visit(const expressions::RefExpression *e1,
                     const expressions::RefExpression *e2) override;
  ProteusValue visit(const expressions::AssignExpression *e1,
                     const expressions::AssignExpression *e2) override;
  ProteusValue visit(const expressions::NegExpression *e1,
                     const expressions::NegExpression *e2) override;
  ProteusValue visit(const expressions::ExtractExpression *e1,
                     const expressions::ExtractExpression *e2) override;
  ProteusValue visit(const expressions::TestNullExpression *e1,
                     const expressions::TestNullExpression *e2) override;
  ProteusValue visit(const expressions::CastExpression *e1,
                     const expressions::CastExpression *e2) override;

  ProteusValue visit(const expressions::ShiftLeftExpression *e1,
                     const expressions::ShiftLeftExpression *e2) override;
  ProteusValue visit(
      const expressions::LogicalShiftRightExpression *e1,
      const expressions::LogicalShiftRightExpression *e2) override;
  ProteusValue visit(
      const expressions::ArithmeticShiftRightExpression *e1,
      const expressions::ArithmeticShiftRightExpression *e2) override;
  ProteusValue visit(const expressions::XORExpression *e1,
                     const expressions::XORExpression *e2) override;

 private:
  Context *const context;
  const OperatorState &currStateLeft;
  const OperatorState &currStateRight;
  ProteusValue compareThroughEvaluation(const expressions::Expression *e1,
                                        const expressions::Expression *e2);
};

#endif /* EXPRESSIONS_DOT_VISITOR_HPP_ */
