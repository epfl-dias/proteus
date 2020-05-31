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

#include <olap/operators/agg-t.hpp>
#include <utility>

#include "lib/expressions/expressions-flusher.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "olap/operators/monoids.hpp"
#include "operators.hpp"

namespace opt {

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

  virtual StateVar resetAccumulator(const agg_t &agg, bool is_first,
                                    bool is_last,
                                    ParallelContext *context) const;

  // Used to enable chaining with subsequent operators
  virtual void generateBagUnion(const expression_t &outputExpr,
                                ParallelContext *context,
                                const OperatorState &state,
                                llvm::Value *cnt_mem) const;
};
}  // namespace opt

#endif /* REDUCE_HPP_ */
