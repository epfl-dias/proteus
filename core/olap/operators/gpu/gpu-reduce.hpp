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

#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"
#include "operators/reduce-opt.hpp"
#include "util/parallel-context.hpp"

namespace opt {
//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

/* MULTIPLE ACCUMULATORS SUPPORTED */
class GpuReduce : public Reduce {
 public:
  // FIXME get read of parameter global_accumulator_ptr, it should be created
  // inside this class and the materializer should be responsible to write it to
  // the host
  GpuReduce(vector<Monoid> accs, vector<expression_t> outputExprs,
            expression_t pred, Operator *const child, ParallelContext *context);
  virtual ~GpuReduce() { LOG(INFO) << "Collapsing GpuReduce operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);

  // virtual void open (Pipeline * pip) const;
  // virtual void close(Pipeline * pip) const;

 protected:
  virtual StateVar resetAccumulator(expression_t outputExpr, Monoid acc,
                                    bool flushDelim, bool is_first,
                                    bool is_last) const;

 private:
  void generate(Context *const context, const OperatorState &childState) const;
  void generate(const Monoid &m, expression_t outputExpr,
                ParallelContext *const context, const OperatorState &childState,
                llvm::Value *mem_accumulating,
                llvm::Value *global_accumulator_ptr) const;

  std::vector<llvm::Value *> global_acc_ptr;

  std::vector<StateVar> out_ids;
};
}  // namespace opt

#endif /* GPU_REDUCE_HPP_ */
