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

#ifndef GPU_RAW_CONTEXT_HPP_
#define GPU_RAW_CONTEXT_HPP_

#include "util/raw-context.hpp"
#include "util/raw-pipeline.hpp"

class GpuRawContext: public RawContext {
public:

    GpuRawContext(const string& moduleName, bool gpu_root = false);
    virtual ~GpuRawContext();

    virtual size_t appendParameter(llvm::Type * ptype, bool noalias  = false, bool readonly = false);
    virtual size_t appendStateVar (llvm::Type * ptype, std::string name = "");
    virtual size_t appendStateVar (llvm::Type * ptype, std::function<init_func_t> init, std::function<deinit_func_t> deinit, std::string name = "");


    virtual llvm::Argument * getArgument(size_t id) const;
    virtual llvm::Value    * getStateVar(size_t id) const;
    virtual llvm::Value    * getStateVar()          const;
    virtual llvm::Value    * getSubStateVar()       const;
    virtual std::vector<llvm::Type *> getStateVars()          const;

    void registerOpen (const void * owner, std::function<void (RawPipeline * pip)> open );
    void registerClose(const void * owner, std::function<void (RawPipeline * pip)> close);

    // void pushNewPipeline    (RawPipelineGen *copyStateFrom = NULL);
    // void pushNewCpuPipeline (RawPipelineGen *copyStateFrom = NULL);

private:
    void pushDeviceProvider(RawPipelineGenFactory * factory);
    
    template<typename T>
    void pushDeviceProvider(){
        pushDeviceProvider(&(T::getInstance()));
    }

    void popDeviceProvider();

    friend class DeviceCross;
    friend class CpuToGpu;
    friend class GpuToCpu;
public:
    void pushPipeline(RawPipelineGen * copyStateFrom = NULL);
    void popPipeline();

    RawPipelineGen * removeLatestPipeline();
    RawPipelineGen * getCurrentPipeline()  ;
    void             setChainedPipeline(RawPipelineGen * next);

    virtual Module      * getModule () const {
        return generators.back()->getModule ();
    }
    
    virtual IRBuilder<> * getBuilder() const {
        return generators.back()->getBuilder();
    }

    Function * const getFunction(string funcName) const{
        return generators.back()->getFunction(funcName);
    }

    virtual void setGlobalFunction(bool leaf);
    virtual void setGlobalFunction(Function *F = nullptr, bool leaf = false);
    virtual void prepareFunction(Function *F){}

    virtual llvm::Value * threadId ();
    virtual llvm::Value * threadIdInBlock();
    virtual llvm::Value * blockId  ();
    virtual llvm::Value * blockDim ();
    virtual llvm::Value * gridDim  ();
    virtual llvm::Value * threadNum();
    virtual llvm::Value * laneId   ();
    virtual void          createMembar_gl();

    virtual BasicBlock* getEndingBlock()                            {return generators.back()->getEndingBlock();}
    virtual void        setEndingBlock(BasicBlock* codeEnd)         {generators.back()->setEndingBlock(codeEnd);}
    virtual BasicBlock* getCurrentEntryBlock()                      {return generators.back()->getCurrentEntryBlock();}
    virtual void        setCurrentEntryBlock(BasicBlock* codeEntry) {generators.back()->setCurrentEntryBlock(codeEntry);}

    virtual llvm::Value * allocateStateVar  (llvm::Type  *t);
    virtual void          deallocateStateVar(llvm::Value *v);

    // string emitPTX();

    void compileAndLoad();

    // std::vector<CUfunction> getKernel();
    std::vector<RawPipeline *> getPipelines();
    
    //Provide support for some extern functions
    virtual void registerFunction(const char* funcName, Function* func);

    RawPipelineGen * operator->() const {return generators.back();}

protected:
    virtual void createJITEngine();

public:
    std::unique_ptr<TargetMachine> TheTargetMachine;
    ExecutionEngine * TheExecutionEngine;
    ExecutionEngine * TheCPUExecutionEngine;

    // CUmodule cudaModule;

protected:
    string                      kernelName;
    size_t                      pip_cnt   ;

    std::vector<RawPipelineGenFactory *> pipFactories;

    // Module * TheCPUModule;

    std::vector<RawPipelineGen *> pipelines ;

    std::vector<RawPipelineGen *> generators;

    std::vector<bool            > leafpip   ;

    std::vector<bool            > leafgen   ;
};

#endif /* GPU_RAW_CONTEXT_HPP_ */
