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

#ifndef GPU_REDUCE_HPP_
#define GPU_REDUCE_HPP_

#include "operators/operators.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"
#include "util/gpu/gpu-raw-context.hpp"

namespace opt {
//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

/* MULTIPLE ACCUMULATORS SUPPORTED */
class GpuReduce: public Reduce {
public:
//FIXME get read of parameter global_accumulator_ptr, it should be created inside this class and the materializer should be responsible to write it to the host
    GpuReduce(vector<Monoid>                    accs,
            vector<expressions::Expression *>   outputExprs,
            expressions::Expression *           pred,
            RawOperator * const                 child,
            GpuRawContext *                     context);
    virtual ~GpuReduce() { LOG(INFO)<< "Collapsing GpuReduce operator";}
    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);
private:
    void generate(RawContext* const context, const OperatorState& childState) const;
    void generate(const Monoid &m, expressions::Expression* outputExpr, GpuRawContext* const context, const OperatorState& childState, AllocaInst *mem_accumulating, Argument *global_accumulator_ptr) const;
    
    std::vector<llvm::Value *> global_acc_ptr;

    std::vector<int> out_ids;
};
}

#endif /* GPU_REDUCE_HPP_ */
