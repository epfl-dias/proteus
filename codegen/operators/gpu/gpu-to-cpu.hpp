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
#ifndef GPU_TO_CPU_HPP_
#define GPU_TO_CPU_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class GpuToCpu : public UnaryRawOperator {
public:
    GpuToCpu(   RawOperator   * const            child,
                GpuRawContext * const            context,
                const vector<RecordAttribute *> &wantedFields,
                size_t                           size) :
                    UnaryRawOperator(child), 
                    context(context), 
                    wantedFields(wantedFields),
                    size(size){}

    virtual ~GpuToCpu(){ LOG(INFO)<<"Collapsing GpuToCpu operator";}

    virtual void produce();
    virtual void consume(RawContext    * const context, const OperatorState& childState);
    virtual void consume(GpuRawContext * const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

private:
    void generate_catch();

    void open (RawPipeline * pip);
    void close(RawPipeline * pip);



    const vector<RecordAttribute *> wantedFields        ;

    GpuRawContext * const           context             ;

    RawPipelineGen *                cpu_pip             ;

    llvm::Type                    * params_type         ;

    size_t                          lockVar_id          ;
    size_t                          lastVar_id          ;
    size_t                          flagsVar_id         ;
    size_t                          storeVar_id         ;
    size_t                          threadVar_id        ;
    size_t                          eofVar_id           ;

    size_t                          flagsVar_id_catch   ;
    size_t                          storeVar_id_catch   ;
    size_t                          eofVar_id_catch     ;

    size_t                          size                ;
};

#endif /* GPU_TO_CPU_HPP_ */
