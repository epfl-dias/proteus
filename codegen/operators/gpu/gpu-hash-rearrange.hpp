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
#ifndef GPU_HASH_REARRANGE_HPP_
#define GPU_HASH_REARRANGE_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"
// #include "operators/hash-rearrange.hpp"
// #include "operators/gpu/gpu-materializer-expr.hpp"

class GpuHashRearrange : public UnaryRawOperator {
public:
    GpuHashRearrange(RawOperator  * const                           child,
                    GpuRawContext * const                           context,
                    int                                             numOfBuckets,
                    const std::vector<expressions::Expression *>  & matExpr,
                    expressions::Expression                       * hashExpr,
                    RecordAttribute                               * hashProject = NULL) :
                        UnaryRawOperator(child), 
                        context(context), 
                        numOfBuckets(numOfBuckets),
                        matExpr(matExpr),
                        hashExpr(hashExpr),
                        hashProject(hashProject),
                        blockSize(h_vector_size * sizeof(int32_t)){
                        // packet_widths(packet_widths){
    }//FIMXE: default blocksize...

    virtual ~GpuHashRearrange()                                             { LOG(INFO)<<"Collapsing GpuHashRearrange operator";}

    virtual void produce();
    virtual void consume(RawContext    * const context, const OperatorState& childState);
    virtual void consume(GpuRawContext * const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

protected:
    virtual void consume_flush();

    virtual void open (RawPipeline * pip);
    virtual void close(RawPipeline * pip);

    std::vector<expressions::Expression *>  matExpr         ;
    const int                               numOfBuckets    ;
    RecordAttribute                       * hashProject     ;

    expressions::Expression               * hashExpr        ;

    RawPipelineGen                        * closingPip      ;

    std::vector<size_t>                     buffVar_id      ;
    size_t                                  cntVar_id       ;
    size_t                                  oidVar_id       ;
    size_t                                  wcntVar_id      ;

    size_t                                  blockSize       ; //bytes

    GpuRawContext * const                   context         ;

    // std::vector<size_t>                     packet_widths   ;
};

#endif /* GPU_HASH_REARRANGE_HPP_ */
