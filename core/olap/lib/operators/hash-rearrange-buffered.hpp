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
#ifndef HASH_REARRANGE_BUFFERED_HPP_
#define HASH_REARRANGE_BUFFERED_HPP_

#include <platform/memory/block-manager.hpp>

#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class HashRearrangeBuffered : public UnaryOperator {
 public:
  HashRearrangeBuffered(Operator *const child, ParallelContext *const context,
                        int numOfBuckets,
                        const vector<expressions::Expression *> &wantedFields,
                        expression_t hashExpr,
                        RecordAttribute *hashProject = nullptr)
      : UnaryOperator(child),
        context(context),
        numOfBuckets(numOfBuckets),
        wantedFields(wantedFields),
        hashExpr(std::move(hashExpr)),
        hashProject(hashProject),
        blockSize(BlockManager::block_size) {}  // FIMXE: default blocksize...

  ~HashRearrangeBuffered() override {
    LOG(INFO) << "Collapsing HashRearrangeBuffered operator";
  }

  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  bool isFiltering() const override { return false; }

  llvm::Value *hash(llvm::Value *key, llvm::Value *old_seed = nullptr);
  llvm::Value *hash(const std::vector<expression_t> &exprs,
                    Context *const context, const OperatorState &childState);

 protected:
  virtual void consume_flush();
  void consume_flush1();

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  const vector<expressions::Expression *> wantedFields;
  const int numOfBuckets;
  RecordAttribute *hashProject;

  expression_t hashExpr;

  // void *                          flushFunc   ;

  StateVar blkVar_id;
  StateVar cntVar_id;
  StateVar oidVar_id;

  StateVar cache_Var_id;
  StateVar cache_cnt_Var_id;

  size_t blockSize;  // bytes

  int64_t cap;

  ParallelContext *const context;

  PipelineGen *closingPip;
  PipelineGen *closingPip1;
  llvm::Function *flushingFunc1;
  llvm::Function *flushingFunc2;

  std::vector<size_t> wfSizes;
};

#endif /* HASH_REARRANGE_HPP_ */
