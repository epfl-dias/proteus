/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "expressions/expressions/ref-expression.hpp"

#include <values/indexed-seq.hpp>

namespace expressions {
RefExpression::RefExpression(expression_t ptr)
    : ExpressionCRTP(
          // FIXME: this would normally require a clone, now it's unsafe
          &dynamic_cast<const type::IndexedSeq &>(*ptr.getExpressionType())
               .getNestedType()),
      ptr(std::move(ptr)) {}

const expression_t &RefExpression::getExpr() const { return ptr; }

AssignExpression RefExpression::assign(expression_t e) const {
  return AssignExpression{*this, std::move(e)};
}

AssignExpression::AssignExpression(RefExpression ref, expression_t v)
    : ExpressionCRTP(v.getExpressionType()),
      ref(std::move(ref)),
      v(std::move(v)) {
  assert(this->ref.getExpressionType()->getTypeID() ==
         this->v.getExpressionType()->getTypeID());
}

const expression_t &AssignExpression::getExpr() const { return v; }
const RefExpression &AssignExpression::getRef() const { return ref; };
}  // namespace expressions
