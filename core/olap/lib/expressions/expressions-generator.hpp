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

#ifndef EXPRESSIONS_VISITOR_HPP_
#define EXPRESSIONS_VISITOR_HPP_

#include <platform/common/common.hpp>
#include <utility>

#include "lib/util/caching.hpp"
#include "lib/util/catalog.hpp"
#include "olap/plugins/plugins.hpp"

/**
 * "Visitor(s)" responsible for generating the appropriate code per Expression
 * 'node'
 * @see Karpathiotakis et al, VLDB2016
 */
class ExpressionGeneratorVisitor : public ExprVisitor {
 public:
  ExpressionGeneratorVisitor(Context *const context,
                             const OperatorState &currState,
                             std::string activeRelation = "")
      : context(context),
        currState(currState),
        activeRelation(std::move(activeRelation)) {}
  ProteusValue visit(const expressions::IntConstant *e) override;
  ProteusValue visit(const expressions::Int64Constant *e) override;
  ProteusValue visit(const expressions::DateConstant *e) override;
  ProteusValue visit(const expressions::FloatConstant *e) override;
  ProteusValue visit(const expressions::BoolConstant *e) override;
  ProteusValue visit(const expressions::StringConstant *e) override;
  ProteusValue visit(const expressions::DStringConstant *e) override;
  ProteusValue visit(const expressions::InputArgument *e) override;
  ProteusValue visit(const expressions::RecordProjection *e) override;
  /*
   * XXX How is nullptr propagated? What if one of the attributes is nullptr;
   * XXX Did not have to test it yet -> Applicable to output only
   */
  ProteusValue visit(const expressions::RecordConstruction *e) override;
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
  ProteusValue visit(const expressions::PlaceholderExpression *e) override;
  ProteusValue visit(const expressions::ProteusValueExpression *e) override;
  ProteusValue visit(const expressions::MinExpression *e) override;
  ProteusValue visit(const expressions::MaxExpression *e) override;
  ProteusValue visit(const expressions::HashExpression *e) override;
  ProteusValue visit(const expressions::RandExpression *e) override;
  ProteusValue visit(const expressions::HintExpression *e) override;
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

  void setActiveRelation(std::string relName) {
    activeRelation = std::move(relName);
  }
  std::string getActiveRelation() { return activeRelation; }

 private:
  Context *const context;
  const OperatorState &currState;

  std::string activeRelation;
};

static_assert(!std::is_abstract<ExpressionGeneratorVisitor>(),
              "ExpressionGeneratorVisitor should be non-abstract");

#endif /* EXPRESSIONS_VISITOR_HPP_ */
