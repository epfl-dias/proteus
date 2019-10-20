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

#ifndef HASH_JOIN_CHAINED_HPP_
#define HASH_JOIN_CHAINED_HPP_

#include <unordered_map>

#include "codegen/util/parallel-context.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/operators.hpp"

class HashJoinChained : public BinaryOperator {
 public:
  HashJoinChained(const std::vector<GpuMatExpr> &build_mat_exprs,
                  const std::vector<size_t> &build_packet_widths,
                  expression_t build_keyexpr, Operator *const build_child,
                  const std::vector<GpuMatExpr> &probe_mat_exprs,
                  const std::vector<size_t> &probe_packet_widths,
                  expression_t probe_keyexpr, Operator *const probe_child,
                  int hash_bits, ParallelContext *context,
                  size_t maxBuildInputSize, string opLabel = "hj_chained");
  virtual ~HashJoinChained() {
    LOG(INFO) << "Collapsing HashJoinChained operator";
  }

  virtual void produce();
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  virtual void consume(Context *const context, const OperatorState &childState);

  virtual void open_probe(Pipeline *pip);
  virtual void open_build(Pipeline *pip);
  virtual void close_probe(Pipeline *pip);
  virtual void close_build(Pipeline *pip);

  virtual bool isFiltering() const { return true; }

  virtual RecordType getRowType() const {
    std::vector<RecordAttribute *> ret;

    for (const GpuMatExpr &mexpr : build_mat_exprs) {
      if (mexpr.packet == 0 && mexpr.packind == 0) continue;
      ret.emplace_back(new RecordAttribute(mexpr.expr.getRegisteredAs()));
    }

    for (const GpuMatExpr &mexpr : probe_mat_exprs) {
      if (mexpr.packet == 0 && mexpr.packind == 0) continue;
      ret.emplace_back(new RecordAttribute(mexpr.expr.getRegisteredAs()));
    }

    if (build_keyexpr.isRegistered()) {
      ret.emplace_back(new RecordAttribute(build_keyexpr.getRegisteredAs()));
    }

    if (build_keyexpr.getExpressionType()->getTypeID() == RECORD) {
      auto rc = dynamic_cast<const expressions::RecordConstruction *>(
          build_keyexpr.getUnderlyingExpression());

      for (const auto &a : rc->getAtts()) {
        auto e = a.getExpression();
        if (e.isRegistered()) {
          ret.emplace_back(new RecordAttribute(e.getRegisteredAs()));
        }
      }
    }

    if (probe_keyexpr.isRegistered()) {
      ret.emplace_back(new RecordAttribute(probe_keyexpr.getRegisteredAs()));
    }

    if (probe_keyexpr.getExpressionType()->getTypeID() == RECORD) {
      auto rc = dynamic_cast<const expressions::RecordConstruction *>(
          probe_keyexpr.getUnderlyingExpression());

      for (const auto &a : rc->getAtts()) {
        auto e = a.getExpression();
        if (e.isRegistered()) {
          ret.emplace_back(new RecordAttribute(e.getRegisteredAs()));
        }
      }
    }

    return ret;
  }

 protected:
  void generate_build(ParallelContext *context,
                      const OperatorState &childState);
  void generate_probe(ParallelContext *context,
                      const OperatorState &childState);
  void buildHashTableFormat();
  void probeHashTableFormat();

  llvm::Value *hash(expression_t exprs, Context *const context,
                    const OperatorState &childState);

  std::vector<GpuMatExpr> build_mat_exprs;
  std::vector<GpuMatExpr> probe_mat_exprs;
  std::vector<size_t> build_packet_widths;
  expression_t build_keyexpr;

  expression_t probe_keyexpr;

  std::vector<size_t> packet_widths;

  int head_param_id;
  std::vector<int> out_param_ids;
  std::vector<int> in_param_ids;
  int cnt_param_id;

  int probe_head_param_id;

  int hash_bits;
  size_t maxBuildInputSize;

  // GpuExprMaterializer *   build_mat  ;
  // GpuExprMaterializer *   probe_mat  ;
  ParallelContext *context;

  string opLabel;

  // std::unordered_map<int32_t, std::vector<void *>> confs;
  std::vector<void *> confs[128];
};

#endif /* HASH_JOIN_CHAINED_HPP_ */
