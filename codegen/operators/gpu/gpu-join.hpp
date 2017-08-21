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

#ifndef GPU_JOIN_HPP_
#define GPU_JOIN_HPP_

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class GpuJoin : public BinaryRawOperator {
public:
    GpuJoin(
            const std::vector<GpuMatExpr>      &build_mat_exprs, 
            const std::vector<size_t>          &build_mat_packet_widths,
            RawOperator * const                 build_child,
            const std::vector<GpuMatExpr>      &probe_mat_exprs, 
            const std::vector<size_t>          &probe_mat_packet_widths,
            RawOperator * const                 probe_child,
            GpuRawContext *                     context);
    virtual ~GpuJoin() { LOG(INFO)<< "Collapsing GpuJoin operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);

    CUfunction probe_kernel;


    virtual bool isFiltering() const{
        return true;
    }
private:
    void generate_build(RawContext* const context, const OperatorState& childState) const;
    void generate_probe(RawContext* const context, const OperatorState& childState) const;


    GpuExprMaterializer *   build_mat  ;
    GpuExprMaterializer *   probe_mat  ;
    GpuRawContext *         context;
};

#endif /* GPU_JOIN_HPP_ */
