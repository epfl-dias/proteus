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
#ifndef HASH_REARRANGE_HPP_
#define HASH_REARRANGE_HPP_

#include "codegen/util/parallel-context.hpp"
#include "operators/operators.hpp"

class HashRearrange : public UnaryOperator {
 public:
  HashRearrange(Operator *const child, ParallelContext *const context,
                int numOfBuckets, const std::vector<expression_t> &wantedFields,
                expression_t hashExpr, RecordAttribute *hashProject = NULL)
      : UnaryOperator(child),
        context(context),
        numOfBuckets(numOfBuckets),
        wantedFields(wantedFields),
        hashExpr(std::move(hashExpr)),
        hashProject(hashProject),
        blockSize(h_vector_size * sizeof(int32_t)) {
  }  // FIMXE: default blocksize...

  virtual ~HashRearrange() { LOG(INFO) << "Collapsing HashRearrange operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  llvm::Value *hash(const std::vector<expression_t> &exprs,
                    Context *const context, const OperatorState &childState);

 protected:
  virtual void consume_flush();

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  const int numOfBuckets;
  RecordAttribute *hashProject;

  expression_t hashExpr;

  // void *                          flushFunc   ;

  size_t blkVar_id;
  size_t cntVar_id;
  size_t oidVar_id;

  size_t blockSize;  // bytes

  int64_t cap;

  ParallelContext *const context;

  PipelineGen *closingPip;
  llvm::Function *flushingFunc;

  std::vector<size_t> wfSizes;
};

#endif /* HASH_REARRANGE_HPP_ */
