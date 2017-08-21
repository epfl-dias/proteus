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

#ifndef RAW_PIPELINE_HPP_
#define RAW_PIPELINE_HPP_

#include <vector>
#include "util/raw-context.hpp"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "common/gpu/gpu-common.hpp"

class RawPipeline;

class RawPipelineGen {
private:
    std::vector<llvm::Type *>   inputs          ;
    std::vector<bool     >      inputs_noalias  ;
    std::vector<bool     >      inputs_readonly ;

    std::vector<llvm::Type *>   state_vars      ;
    std::vector<Argument *>     args            ;

    std::vector<std::function<void (RawPipeline * pip)>> openers;
    std::vector<std::function<void (RawPipeline * pip)>> closers;

    std::string                 pipName         ;
    RawContext                * context         ;
public:
    Function *                  F               ;

    RawPipelineGen(RawContext * context, std::string pipName = "pip"): F(nullptr), pipName(pipName), context(context){};
    // ~RawPipelineGen();

    virtual size_t appendParameter(llvm::Type * ptype, bool noalias  = false, bool readonly = false);
    virtual size_t appendStateVar (llvm::Type * ptype);

    virtual llvm::Argument* getArgument(size_t id) const;
    virtual llvm::Value   * getStateVar(size_t id) const;

    Function              * prepare();
    RawPipeline           * getPipeline(CUmodule cudaModule) const;

    void registerOpen (std::function<void (RawPipeline * pip)> open );
    void registerClose(std::function<void (RawPipeline * pip)> close);
    

    [[deprecated]] Function              * getFunction() const;
};

class RawPipeline{
private:
    CUfunction          cons      ;
    llvm::StructType *  state_type;
    const int32_t       group_id  ;

    std::vector<std::function<void (RawPipeline * pip)>> openers;
    std::vector<std::function<void (RawPipeline * pip)>> closers;

    RawPipeline(CUfunction cons, RawContext * context, llvm::StructType * state_type,
        const std::vector<std::function<void (RawPipeline * pip)>> &openers,
        const std::vector<std::function<void (RawPipeline * pip)>> &closers,
        int32_t group_id = 0); //FIXME: group id should be handled to comply with the requirements!

    friend class RawPipelineGen;
public:
    void     * state;

    ~RawPipeline();

    template<typename T>
    void setStateVar(RawContext * context, size_t state_id, const T &value){
        const DataLayout &layout = context->getModule()->getDataLayout();
        size_t offset = layout.getStructLayout(state_type)->getElementOffset(state_id);

        *((T *) (((char *) state) + offset)) = value;
    }

    template<typename T>
    T getStateVar(RawContext * context, size_t state_id){
        const DataLayout &layout = context->getModule()->getDataLayout();
        size_t offset = layout.getStructLayout(state_type)->getElementOffset(state_id);

        return *((T *) (((char *) state) + offset));
    }

    int32_t getGroup() const;

    void open();
    
    template<typename... Tin>
    void consume(size_t N, const Tin * ... src){ //FIXME: cleanup + remove synchronization
        void *KernelParams[] = {(&src)...,
                                &N,
                                state};

        launch_kernel(cons, KernelParams);

        gpu_run(cudaDeviceSynchronize());
    }//;// cnt_t N, vid_t v, cid_t c){

    void close();
};


#endif /* RAW_PIPELINE_HPP_ */