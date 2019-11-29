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

#include "expressions/expressions.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"
#include "util/jit/pipeline.hpp"
#include "util/parallel-context.hpp"

struct GpuAggrMatExpr {
 public:
  expression_t expr;
  size_t packet;
  size_t bitoffset;
  size_t packind;
  Monoid m;
  bool is_m;

  GpuAggrMatExpr(expression_t expr, size_t packet, size_t bitoffset, Monoid m)
      : expr(expr),
        packet(packet),
        bitoffset(bitoffset),
        packind(-1),
        m(m),
        is_m(true) {}

  GpuAggrMatExpr(expression_t expr, size_t packet, size_t bitoffset)
      : expr(expr),
        packet(packet),
        bitoffset(bitoffset),
        packind(-1),
        m(SUM),
        is_m(false) {}

  bool is_aggregation() { return is_m; }
};

class HashGroupByChained : public UnaryOperator {
 public:
  HashGroupByChained(const std::vector<GpuAggrMatExpr> &agg_exprs,
                     const std::vector<expression_t> key_expr,
                     Operator *const child,

                     int hash_bits,

                     ParallelContext *context, size_t maxInputSize,
                     string opLabel = "gb_chained");
  virtual ~HashGroupByChained() {
    LOG(INFO) << "Collapsing HashGroupByChained operator";
  }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);

  virtual bool isFiltering() const { return true; }

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  virtual RecordType getRowType() const {
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
  void prepareDescription();
  virtual void generate_build(ParallelContext *const context,
                              const OperatorState &childState);
  virtual void generate_scan();
  virtual void buildHashTableFormat();
  virtual llvm::Value *hash(const std::vector<expression_t> &exprs,
                            Context *const context,
                            const OperatorState &childState);

  std::vector<GpuAggrMatExpr> agg_exprs;
  std::vector<size_t> packet_widths;
  std::vector<expression_t> key_expr;
  std::vector<llvm::Type *> ptr_types;

  int head_param_id;
  std::vector<int> out_param_ids;
  int cnt_param_id;

  int hash_bits;
  size_t maxInputSize;

  PipelineGen *probe_gen;

  ParallelContext *context;
  string opLabel;
};

#endif /* HASH_GROUP_BY_CHAINED_HPP_ */
