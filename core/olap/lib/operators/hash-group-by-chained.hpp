/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef HASH_GROUP_BY_CHAINED_HPP_
#define HASH_GROUP_BY_CHAINED_HPP_

#include "lib/util/jit/pipeline.hpp"
#include "olap/expressions/expressions.hpp"
#include "olap/operators/gpu-aggr-mat-expr.hpp"
#include "olap/operators/monoids.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class HashGroupByChained : public experimental::UnaryOperator {
 public:
  HashGroupByChained(std::vector<GpuAggrMatExpr> agg_exprs,
                     std::vector<expression_t> key_expr, Operator *child,

                     int hash_bits,

                     size_t maxInputSize, std::string opLabel = "gb_chained");

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] bool isFiltering() const override { return true; }

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  [[nodiscard]] RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : key_expr) {
      attrs.emplace_back(new RecordAttribute{attr.getRegisteredAs()});
    }
    for (const auto &attr : agg_exprs) {
      attrs.emplace_back(new RecordAttribute{attr.expr.getRegisteredAs()});
    }
    return attrs;
  }

 protected:
  virtual void prepareDescription(ParallelContext *context);
  virtual void generate_build(ParallelContext *context,
                              const OperatorState &childState);
  virtual void generate_scan(ParallelContext *context);
  virtual void buildHashTableFormat(ParallelContext *context);
  virtual llvm::Value *hash(const std::vector<expression_t> &exprs,
                            ParallelContext *context,
                            const OperatorState &childState);

  std::vector<GpuAggrMatExpr> agg_exprs;
  std::vector<size_t> packet_widths;
  std::vector<expression_t> key_expr;
  std::vector<llvm::Type *> ptr_types;

  StateVar head_param_id;
  std::vector<StateVar> out_param_ids;
  StateVar cnt_param_id;

  unsigned int hash_bits;
  size_t maxInputSize;

  std::string opLabel;
};

#endif /* HASH_GROUP_BY_CHAINED_HPP_ */
