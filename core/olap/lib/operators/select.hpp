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

#include "olap/expressions/expressions.hpp"
#include "operators.hpp"

class Select : public experimental::UnaryOperator {
 public:
  Select(expression_t expr, Operator *child)
      : UnaryOperator(child), expr(std::move(expr)) {}

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return true; }

  [[nodiscard]] RecordType getRowType() const override {
    return getChild()->getRowType();
  }

  [[nodiscard]] expression_t getFilter() const { return expr; }

 private:
  expression_t expr;
};
