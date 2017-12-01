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
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef RAW_GPU_PIPELINE_HPP_
#define RAW_GPU_PIPELINE_HPP_

#include "util/raw-pipeline.hpp"
#include "util/jit/raw-gpu-module.hpp"
#include "util/jit/raw-cpu-module.hpp"

class RawGpuPipelineGen: public RawPipelineGen {
protected:
    RawGpuModule                                        module              ;
    RawCpuModule                                        wrapper_module      ;

    bool                                                wrapperModuleActive ;
    size_t                                              kernel_id           ;
    size_t                                              strm_id             ;

    Function                                          * Fconsume            ;
    Function                                          * subpipelineSync     ;
    map<string, Function*>                              availableWrapperFunctions   ;
public:
    RawGpuPipelineGen(  RawContext        * context                 , 
                        std::string         pipName         = "pip" , 
                        RawPipelineGen    * copyStateFrom   = NULL  );

    virtual void                    compileAndLoad();

    virtual Function              * prepare();

    virtual RawPipeline           * getPipeline(int group_id = 0);
    virtual void                  * getKernel  () const;

    // virtual size_t appendStateVar (llvm::Type * ptype);
    // virtual size_t appendStateVar (llvm::Type * ptype, std::function<init_func_t> init, std::function<deinit_func_t> deinit);
    virtual Module                * getModule () const {
        if (wrapperModuleActive) return wrapper_module.getModule();
        return module.getModule();
    }

    virtual void                  * getConsume() const;
    virtual Function              * getLLVMConsume() const {return Fconsume;}

    virtual void registerFunction(const char *, Function *);
    virtual Function * const getFunction(string funcName) const;

protected:
    virtual size_t                  prepareStateArgument();
    virtual llvm::Value           * getStateLLVMValue();

    virtual void                    prepareInitDeinit();

    virtual void                  * getCompiledFunction(Function * f);

    virtual void                    registerFunctions();
    virtual Function              * prepareConsumeWrapper();

};

#endif /* RAW_GPU_PIPELINE_HPP_ */