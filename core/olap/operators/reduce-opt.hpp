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

#ifndef REDUCE_OPT_HPP_
#define REDUCE_OPT_HPP_

#include <utility>

#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"

namespace opt {
//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

class agg_t {
  expression_t e;
  Monoid m;

 public:
  agg_t(expression_t e, Monoid m) : e(std::move(e)), m(std::move(m)) {}

  [[nodiscard]] const expression_t &getExpression() const { return e; }

  [[nodiscard]] auto getExpressionType() const { return e.getExpressionType(); }

  [[nodiscard]] auto getRegisteredAs() const { return e.getRegisteredAs(); }

  [[nodiscard]] const Monoid &getMonoid() const { return m; }

  [[nodiscard]] expression_t toReduceExpression(expression_t acc) const {
    return toExpression(m, std::move(acc), e);
  }
};

class sum : agg_t {
 public:
  sum(expression_t e) : agg_t(std::move(e), SUM) {}
};

/* MULTIPLE ACCUMULATORS SUPPORTED */
class Reduce : public experimental::UnaryOperator {
 public:
  Reduce(std::vector<agg_t> aggs, expression_t pred, Operator *const child)
      : UnaryOperator(child), aggs(std::move(aggs)), pred(std::move(pred)) {}

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return true; }

  [[nodiscard]] RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : aggs) {
      attrs.emplace_back(new RecordAttribute{attr.getRegisteredAs()});
    }
    return attrs;
  }

 protected:
  std::vector<agg_t> aggs;
  expression_t pred;
  std::vector<StateVar> mem_accumulators;

 protected:
  virtual void generate_flush(ParallelContext *context);

  virtual StateVar resetAccumulator(const agg_t &agg, bool is_first = false,
                                    bool is_last = false,
                                    ParallelContext *context = nullptr) const;

  // Used to enable chaining with subsequent operators
  virtual void generateBagUnion(expression_t outputExpr, Context *const context,
                                const OperatorState &state,
                                llvm::Value *cnt_mem) const;
};
}  // namespace opt

#endif /* REDUCE_HPP_ */
