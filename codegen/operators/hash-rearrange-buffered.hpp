/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class HashRearrangeBuffered : public UnaryRawOperator {
 public:
  HashRearrangeBuffered(RawOperator *const child, GpuRawContext *const context,
                        int numOfBuckets,
                        const vector<expressions::Expression *> &wantedFields,
                        expression_t hashExpr,
                        RecordAttribute *hashProject = NULL)
      : UnaryRawOperator(child),
        context(context),
        numOfBuckets(numOfBuckets),
        wantedFields(wantedFields),
        hashExpr(std::move(hashExpr)),
        hashProject(hashProject),
        blockSize(h_vector_size * sizeof(int32_t)) {
  }  // FIMXE: default blocksize...

  virtual ~HashRearrangeBuffered() {
    LOG(INFO) << "Collapsing HashRearrangeBuffered operator";
  }

  virtual void produce();
  virtual void consume(RawContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  llvm::Value *hash(llvm::Value *key, llvm::Value *old_seed = NULL);
  llvm::Value *hash(const std::vector<expression_t> &exprs,
                    RawContext *const context, const OperatorState &childState);

 protected:
  virtual void consume_flush();
  void consume_flush1();

  virtual void open(RawPipeline *pip);
  virtual void close(RawPipeline *pip);

  const vector<expressions::Expression *> wantedFields;
  const int numOfBuckets;
  RecordAttribute *hashProject;

  expression_t hashExpr;

  // void *                          flushFunc   ;

  size_t blkVar_id;
  size_t cntVar_id;
  size_t oidVar_id;

  size_t cache_Var_id;
  size_t cache_cnt_Var_id;

  size_t blockSize;  // bytes

  int64_t cap;

  GpuRawContext *const context;

  RawPipelineGen *closingPip;
  RawPipelineGen *closingPip1;
  Function *flushingFunc1;
  Function *flushingFunc2;

  std::vector<size_t> wfSizes;
};

#endif /* HASH_REARRANGE_HPP_ */
