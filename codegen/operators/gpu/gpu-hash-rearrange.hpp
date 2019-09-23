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
#ifndef GPU_HASH_REARRANGE_HPP_
#define GPU_HASH_REARRANGE_HPP_

#include "codegen/util/parallel-context.hpp"
#include "memory/block-manager.hpp"
#include "operators/operators.hpp"
// #include "operators/hash-rearrange.hpp"
// #include "operators/gpu/gpu-materializer-expr.hpp"

class GpuHashRearrange : public UnaryOperator {
 public:
  GpuHashRearrange(Operator *const child, ParallelContext *const context,
                   int numOfBuckets, const std::vector<expression_t> &matExpr,
                   expression_t hashExpr,
                   RecordAttribute *hashProject = nullptr)
      : UnaryOperator(child),
        context(context),
        numOfBuckets(numOfBuckets),
        matExpr(matExpr),
        hashExpr(std::move(hashExpr)),
        hashProject(hashProject),
        blockSize(BlockManager::block_size) {
    // packet_widths(packet_widths){
  }  // FIMXE: default blocksize...

  virtual ~GpuHashRearrange() {
    LOG(INFO) << "Collapsing GpuHashRearrange operator";
  }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const {
    std::vector<RecordAttribute *> attr;
    for (const auto &t : matExpr) {
      attr.emplace_back(new RecordAttribute(t.getRegisteredAs(), true));
    }
    if (hashExpr.isRegistered()) {
      attr.emplace_back(new RecordAttribute(hashExpr.getRegisteredAs()));
    }
    return attr;
  }

 protected:
  virtual void consume_flush(llvm::IntegerType *target_type);

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  llvm::Value *hash(const std::vector<expression_t> &exprs,
                    Context *const context, const OperatorState &childState);

  std::vector<expression_t> matExpr;
  const int numOfBuckets;
  RecordAttribute *hashProject;

  expression_t hashExpr;
  expressions::RecordConstruction *mexpr;

  PipelineGen *closingPip;
  llvm::Function *flushingFunc;

  std::vector<size_t> buffVar_id;
  size_t cntVar_id;
  size_t oidVar_id;
  size_t wcntVar_id;

  size_t blockSize;  // bytes

  int64_t cap;

  ParallelContext *const context;

  // std::vector<size_t>                     packet_widths   ;
};

#endif /* GPU_HASH_REARRANGE_HPP_ */
