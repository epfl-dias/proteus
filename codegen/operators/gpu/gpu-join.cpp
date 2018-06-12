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

#include "operators/gpu/gpu-join.hpp"
#include "operators/gpu/gmonoids.hpp"

GpuJoin::GpuJoin(
            const std::vector<GpuMatExpr>      &build_mat_exprs, 
            const std::vector<size_t>          &build_mat_packet_widths,
            RawOperator * const                 build_child,

            const std::vector<GpuMatExpr>      &probe_mat_exprs, 
            const std::vector<size_t>          &probe_mat_packet_widths,
            RawOperator * const                 probe_child,
            
            GpuRawContext *                     context): 
                BinaryRawOperator(build_child, probe_child), 
                context(context){
    build_mat = new GpuExprMaterializer(build_mat_exprs, 
                                        build_mat_packet_widths,
                                        build_child,
                                        context,
                                        "join_build");

    probe_mat = new GpuExprMaterializer(probe_mat_exprs, 
                                        probe_mat_packet_widths,
                                        probe_child,
                                        context,
                                        "join_probe");
}

void GpuJoin::produce() {
    context->pushPipeline();

    build_mat->produce();

    // context->compileAndLoad(); //FIXME: Remove!!!! causes an extra compilation! this compile will be done again later!
    // Get kernel function
    // probe_kernel = context->getKernel();
    context->popPipeline();

    probe_mat->produce();
}

void GpuJoin::consume(RawContext* const context, const OperatorState& childState) {
    const RawOperator& caller = childState.getProducer();

    if(caller == *(getLeftChild())){
        generate_build(context, childState);
    } else {
        generate_probe(context, childState);
    }
}

void GpuJoin::generate_build(RawContext* const context, const OperatorState& childState) const {
    build_mat->consume(context, childState);
}

void GpuJoin::generate_probe(RawContext* const context, const OperatorState& childState) const {
    probe_mat->consume(context, childState);
}
