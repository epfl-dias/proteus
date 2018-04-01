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

#ifndef GPU_SORT_HPP_
#define GPU_SORT_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "operators/sort.hpp"

class GpuSort : public UnaryRawOperator {
public:
    GpuSort(RawOperator  * const                     child,
            GpuRawContext * const                    context,
            const vector<expressions::Expression *> &orderByFields,
            const vector<direction                > &dirs);

    virtual ~GpuSort()                                             { LOG(INFO)<<"Collapsing GpuSort operator";}

    virtual void produce();
    virtual void consume(RawContext    * const context, const OperatorState& childState);
    virtual void consume(GpuRawContext * const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

protected:
    // virtual void open (RawPipeline * pip);
    // virtual void close(RawPipeline * pip);

    virtual void flush_sorted();

    virtual void call_sort(llvm::Value * mem, llvm::Value * N);

    std::vector<expressions::Expression *> orderByFields;
    const vector<direction               > dirs         ;

    expressions::Expression *       outputExpr  ;
    std::string                     relName     ;

    // size_t                          width       ;

    size_t                          cntVar_id   ;
    size_t                          memVar_id   ;

    llvm::Type                    * mem_type    ;

    GpuRawContext * const           context     ;

    std::string                     suffix      ;
};

#endif /* GPU_SORT_HPP_ */
