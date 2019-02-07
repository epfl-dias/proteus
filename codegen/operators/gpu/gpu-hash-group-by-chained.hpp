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

#include "codegen/util/jit/pipeline.hpp"
#include "codegen/util/parallel-context.hpp"
#include "expressions/expressions.hpp"
#include "operators/hash-group-by-chained.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"

class GpuHashGroupByChained : public HashGroupByChained {
 public:
  // inherit constructor
  using HashGroupByChained::HashGroupByChained;

  virtual void produce();
  // virtual void consume(Context *const context,
  //                      const OperatorState &childState);

  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

 private:
  void generate_build(ParallelContext *const context,
                      const OperatorState &childState);
  void buildHashTableFormat();
  // llvm::Value *hash(llvm::Value *key);
  // llvm::Value *hash(llvm::Value *old_seed, llvm::Value *key);
};

#endif /* GPU_HASH_GROUP_BY_CHAINED_HPP_ */
