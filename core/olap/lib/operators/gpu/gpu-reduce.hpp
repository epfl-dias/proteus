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

#ifndef GPU_REDUCE_HPP_
#define GPU_REDUCE_HPP_

#include "lib/expressions/expressions-flusher.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "lib/operators/operators.hpp"
#include "lib/operators/reduce-opt.hpp"
#include "olap/operators/monoids.hpp"
#include "olap/util/parallel-context.hpp"

namespace opt {
class GpuReduce : public Reduce {
 public:
  GpuReduce(std::vector<agg_t> accs, expression_t pred, Operator *child);
  void consume(ParallelContext *context,
               const OperatorState &childState) override;

 protected:
  void generateBagUnion(const expression_t &outputExpr,
                        ParallelContext *context, const OperatorState &state,
                        llvm::Value *cnt_mem) const override;

 private:
  void generate(const agg_t &agg, ParallelContext *context,
                const OperatorState &childState, llvm::Value *mem_accumulating,
                llvm::Value *global_accumulator_ptr) const;
};
}  // namespace opt

#endif /* GPU_REDUCE_HPP_ */
