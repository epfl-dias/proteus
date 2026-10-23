/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <platform/memory/block-manager.hpp>

#include "lib/operators/operators.hpp"
#include "olap/util/parallel-context.hpp"

class GpuHashRearrange : public experimental::UnaryOperator {
 public:
  GpuHashRearrange(Operator *const child, OlapParallelContext *const context,
                   int numOfBuckets, const std::vector<expression_t> &matExpr,
                   expression_t hashExpr,
                   RecordAttribute *hashProject = nullptr)
      : UnaryOperator(child),
        numOfBuckets(numOfBuckets),
        matExpr(matExpr),
        hashExpr(std::move(hashExpr)),
        hashProject(hashProject),
        blockSize(BlockManager::block_size) {
    attr_size = new size_t[matExpr.size()];
    for (size_t attr_i = 0; attr_i < matExpr.size(); ++attr_i) {
      attr_size[attr_i] =
          context->getSizeOf(matExpr[attr_i].getExpressionType()->getLLVMType(
              context->getLLVMContext()));
    }
    // GPU hash-rearrange produces packed block attributes. Appending an
    // auto-generated scalar hash attribute here would make downstream
    // MemMoveDevice treat a scalar like a block pointer and crash.
    // Keep hashProject only when explicitly provided by the caller.
    // packet_widths(packet_widths){
  }  // FIMXE: default blocksize...

  ~GpuHashRearrange() override {
    LOG(INFO) << "Collapsing GpuHashRearrange operator";
  }

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override {
    std::vector<RecordAttribute *> attr;
    for (const auto &t : matExpr) {
      attr.emplace_back(new RecordAttribute(t.getRegisteredAs(), true));
    }
    if (hashProject) attr.emplace_back(hashProject);
    return attr;
  }

  [[nodiscard]] bool isPacked() const override {
    assert(!getChild()->isPacked());
    return true;
  }

 protected:
  virtual void consume_flush(OlapParallelContext *context,
                             llvm::IntegerType *target_type);

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  static llvm::Value *hash(const std::vector<expression_t> &exprs,
                           OlapParallelContext *context,
                           const OperatorState &childState);

  std::vector<expression_t> matExpr;
  const int numOfBuckets;
  RecordAttribute *hashProject;

  expression_t hashExpr;
  expressions::RecordConstruction *mexpr;

  PipelineGen *closingPip;
  llvm::Function *flushingFunc;

  std::vector<StateVar> buffVar_id;
  StateVar cntVar_id;
  StateVar oidVar_id;
  StateVar wcntVar_id;

  size_t blockSize;  // bytes

  int64_t cap;
  size_t *attr_size;

  // Overflow buffers allocated in open() are tracked per pipeline and freed in
  // close(). This avoids relying on buffer-manager internals from this TU.
  std::unordered_map<const Pipeline *, std::unordered_set<void *>>
      overflow_buffers_by_pipeline;
  std::mutex overflow_buffers_mu;

  // std::vector<size_t>                     packet_widths   ;
};

#endif /* GPU_HASH_REARRANGE_HPP_ */
