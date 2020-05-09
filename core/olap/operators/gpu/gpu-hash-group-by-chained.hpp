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

#ifndef GPU_HASH_GROUP_BY_CHAINED_HPP_
#define GPU_HASH_GROUP_BY_CHAINED_HPP_

#include "expressions/expressions.hpp"
#include "operators/hash-group-by-chained.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"
#include "util/jit/pipeline.hpp"
#include "util/parallel-context.hpp"

class GpuHashGroupByChained : public HashGroupByChained {
 public:
  // inherit constructor
  using HashGroupByChained::HashGroupByChained;

  void produce_(ParallelContext *context) override;

  void open(Pipeline *pip) override;
  void close(Pipeline *pip) override;

 private:
  void generate_build(ParallelContext *context,
                      const OperatorState &childState) override;
  void buildHashTableFormat(ParallelContext *context) override;
  // llvm::Value *hash(llvm::Value *key);
  // llvm::Value *hash(llvm::Value *old_seed, llvm::Value *key);
  PipelineGen *probe_gen;
};

#endif /* GPU_HASH_GROUP_BY_CHAINED_HPP_ */
