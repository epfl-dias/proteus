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

#ifndef PROTEUS_AGG_T_HPP
#define PROTEUS_AGG_T_HPP

#include <olap/expressions/expressions.hpp>
#include <olap/operators/monoids.hpp>

class agg_t {
  expression_t e;
  Monoid m;

 public:
  agg_t(expression_t e, Monoid m) : e(std::move(e)), m(m) {}

  [[nodiscard]] const expression_t &getExpression() const { return e; }

  [[nodiscard]] auto getExpressionType() const { return e.getExpressionType(); }

  [[nodiscard]] auto getRegisteredAs() const { return e.getRegisteredAs(); }

  [[nodiscard]] const Monoid &getMonoid() const { return m; }

  [[nodiscard]] expression_t toReduceExpression(expression_t acc) const {
    return toExpression(m, std::move(acc), e);
  }
};

agg_t sum(expression_t e);

agg_t count();

agg_t max(expression_t e);

agg_t min(expression_t e);

#endif /* PROTEUS_AGG_T_HPP */
