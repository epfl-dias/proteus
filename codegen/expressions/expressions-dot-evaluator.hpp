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

#ifndef EXPRESSIONS_DOT_VISITOR_HPP_
#define EXPRESSIONS_DOT_VISITOR_HPP_

#include "common/common.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-hasher.hpp"
#include "plugins/plugins.hpp"
#include "util/raw-functions.hpp"
//#include "values/expressionTypes.hpp"

//#ifdef DEBUG
#define DEBUG_DOT
//#endif

//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for evaluating dot equality
//===---------------------------------------------------------------------------===//
class ExpressionDotVisitor : public ExprTandemVisitor {
 public:
  ExpressionDotVisitor(RawContext *const context,
                       const OperatorState &currStateLeft,
                       const OperatorState &currStateRight)
      : context(context),
        currStateLeft(currStateLeft),
        currStateRight(currStateRight) {}
  RawValue visit(const expressions::IntConstant *e1,
                 const expressions::IntConstant *e2);
  RawValue visit(const expressions::Int64Constant *e1,
                 const expressions::Int64Constant *e2);
  RawValue visit(const expressions::DateConstant *e1,
                 const expressions::DateConstant *e2);
  RawValue visit(const expressions::FloatConstant *e1,
                 const expressions::FloatConstant *e2);
  RawValue visit(const expressions::BoolConstant *e1,
                 const expressions::BoolConstant *e2);
  RawValue visit(const expressions::StringConstant *e1,
                 const expressions::StringConstant *e2);
  RawValue visit(const expressions::DStringConstant *e1,
                 const expressions::DStringConstant *e2);
  RawValue visit(const expressions::InputArgument *e1,
                 const expressions::InputArgument *e2);
  RawValue visit(const expressions::RawValueExpression *e1,
                 const expressions::RawValueExpression *e2);
  RawValue visit(const expressions::RecordProjection *e1,
                 const expressions::RecordProjection *e2);
  RawValue visit(const expressions::IfThenElse *e1,
                 const expressions::IfThenElse *e2);
  RawValue visit(const expressions::EqExpression *e1,
                 const expressions::EqExpression *e2);
  RawValue visit(const expressions::NeExpression *e1,
                 const expressions::NeExpression *e2);
  RawValue visit(const expressions::GeExpression *e1,
                 const expressions::GeExpression *e2);
  RawValue visit(const expressions::GtExpression *e1,
                 const expressions::GtExpression *e2);
  RawValue visit(const expressions::LeExpression *e1,
                 const expressions::LeExpression *e2);
  RawValue visit(const expressions::LtExpression *e1,
                 const expressions::LtExpression *e2);
  RawValue visit(const expressions::AddExpression *e1,
                 const expressions::AddExpression *e2);
  RawValue visit(const expressions::SubExpression *e1,
                 const expressions::SubExpression *e2);
  RawValue visit(const expressions::MultExpression *e1,
                 const expressions::MultExpression *e2);
  RawValue visit(const expressions::DivExpression *e1,
                 const expressions::DivExpression *e2);
  RawValue visit(const expressions::AndExpression *e1,
                 const expressions::AndExpression *e2);
  RawValue visit(const expressions::OrExpression *e1,
                 const expressions::OrExpression *e2);
  RawValue visit(const expressions::RecordConstruction *e1,
                 const expressions::RecordConstruction *e2);
  RawValue visit(const expressions::MaxExpression *e1,
                 const expressions::MaxExpression *e2);
  RawValue visit(const expressions::MinExpression *e1,
                 const expressions::MinExpression *e2);
  RawValue visit(const expressions::HashExpression *e1,
                 const expressions::HashExpression *e2);
  RawValue visit(const expressions::NegExpression *e1,
                 const expressions::NegExpression *e2);
  RawValue visit(const expressions::ExtractExpression *e1,
                 const expressions::ExtractExpression *e2);
  RawValue visit(const expressions::TestNullExpression *e1,
                 const expressions::TestNullExpression *e2);
  RawValue visit(const expressions::CastExpression *e1,
                 const expressions::CastExpression *e2);

 private:
  RawContext *const context;
  const OperatorState &currStateLeft;
  const OperatorState &currStateRight;
};

#endif /* EXPRESSIONS_DOT_VISITOR_HPP_ */
