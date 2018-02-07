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

// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include "common/gpu/gpu-common.hpp"
#include <utility>

class RawPipeline;

extern "C"{
    void yield();
}

typedef void (opener_t)(RawPipeline *);
typedef void (closer_t)(RawPipeline *);

typedef llvm::Value * (  init_func_t)(llvm::Value *);
typedef void          (deinit_func_t)(llvm::Value *, llvm::Value *);

// __device__ void devprinti64(uint64_t x);

class RawPipelineGen {
protected:
    //Last (current) basic block. This changes every time a new scan is triggered
    BasicBlock* codeEnd;
    //Current entry basic block. This changes every time a new scan is triggered
    BasicBlock* currentCodeEntry;

    std::vector<std::function<  init_func_t>> open_var;
    Function *                  open__function  ;

    std::vector<std::function<deinit_func_t>> close_var;
    Function *                  close_function  ;

    std::vector<llvm::Type *>   inputs          ;
    std::vector<bool     >      inputs_noalias  ;
    std::vector<bool     >      inputs_readonly ;

    std::vector<llvm::Type *>   state_vars      ;
    std::vector<Argument *>     args            ;

    std::vector<std::pair<const void *, std::function<opener_t>>> openers;
    std::vector<std::pair<const void *, std::function<closer_t>>> closers;

    std::string                 pipName         ;
    RawContext                * context         ;

    std::string                 func_name       ;

    void                      * func            ;

    llvm::Value               * state           ;
    llvm::StructType          * state_type      ;

    IRBuilder<>               * TheBuilder      ;

    RawPipelineGen            * copyStateFrom   ;

    RawPipelineGen            * execute_after_close;

//     //Used to include optimization passes
//     legacy::FunctionPassManager * TheFPM        ;
// #if MODULEPASS
//     ModulePassManager           * TheMPM        ;
// #endif

//     legacy::PassManager         * ThePM         ;

    // ExecutionEngine             * TheExecutionEngine;

    map<string, Function*>    availableFunctions   ;
public:
    Function *                  F               ;

protected:
    RawPipelineGen(RawContext * context, std::string pipName = "pip", RawPipelineGen * copyStateFrom = NULL);
    // ~RawPipelineGen();

public:
    virtual size_t appendParameter(llvm::Type * ptype, bool noalias  = false, bool readonly = false);
    virtual size_t appendStateVar (llvm::Type * ptype);
    virtual size_t appendStateVar (llvm::Type * ptype, std::function<init_func_t> init, std::function<deinit_func_t> deinit);

    virtual llvm::Argument* getArgument(size_t id) const;
    virtual llvm::Value   * getStateVar(size_t id) const;
    virtual llvm::Value   * getStateVar()          const;
    virtual llvm::Value   * getSubStateVar()       const;

    virtual llvm::Value   * allocateStateVar  (llvm::Type  * t);
    virtual void            deallocateStateVar(llvm::Value * v);

    virtual Function              * prepare();
    virtual RawPipeline           * getPipeline(int group_id = 0);
    virtual void                  * getKernel  () const;

    virtual void                    setChainedPipeline(RawPipelineGen * next){
        assert(!execute_after_close && "No support for multiple pipelines after a single one, create a chain");
        execute_after_close = next;
    }

    virtual void                  * getConsume() const{return getKernel();}
    virtual Function              * getLLVMConsume() const {return F;}

    std::string                     getName() const{return pipName;}

    virtual BasicBlock * getEndingBlock()                            {return codeEnd;}
    virtual void         setEndingBlock(BasicBlock* codeEnd)         {this->codeEnd = codeEnd;}
    virtual BasicBlock * getCurrentEntryBlock()                      {return currentCodeEntry;}
    virtual void         setCurrentEntryBlock(BasicBlock* codeEntry) {this->currentCodeEntry = codeEntry;}


    virtual void compileAndLoad() = 0;

    void registerOpen (const void * owner, std::function<void (RawPipeline * pip)> open );
    void registerClose(const void * owner, std::function<void (RawPipeline * pip)> close);

    [[deprecated]] virtual Function              * getFunction() const;

    virtual Module                * getModule () const = 0;//{return TheModule ;}
    virtual IRBuilder<>           * getBuilder() const {return TheBuilder;}

    virtual void registerFunction(const char *, Function *);

    virtual Function * const getFunction(string funcName) const;

    virtual Function    * const createHelperFunction(string funcName, std::vector<llvm::Type *> ins, std::vector<bool> readonly, std::vector<bool> noalias) const;
    virtual llvm::Value *       invokeHelperFunction(Function * f, std::vector<llvm::Value *> args) const;

    std::vector<llvm::Type *> getStateVars() const;

    static void init();

protected:
    virtual void                    registerSubPipeline();
    virtual size_t                  prepareStateArgument();
    virtual llvm::Value           * getStateLLVMValue();
    virtual void                    prepareFunction();
    virtual void                    prepareInitDeinit();
public:
    virtual void                  * getCompiledFunction(Function * f) = 0;
protected:
    void registerFunctions();
};

class RawPipeline{
protected:
    void              * cons      ;
    llvm::StructType  * state_type;
    const int32_t       group_id  ;
    size_t              state_size;
    const DataLayout  & layout    ;

    std::vector<std::pair<const void *, std::function<opener_t>>> openers;
    std::vector<std::pair<const void *, std::function<closer_t>>> closers;

    void              * init_state;
    void              * deinit_state;

    RawPipeline       * execute_after_close;

    RawPipeline(void * cons, size_t state_size, RawPipelineGen * gen, llvm::StructType * state_type,
        const std::vector<std::pair<const void *, std::function<opener_t>>> &openers,
        const std::vector<std::pair<const void *, std::function<closer_t>>> &closers,
        void *init_state,
        void *deinit_state,
        int32_t group_id = 0, //FIXME: group id should be handled to comply with the requirements!
        RawPipeline * execute_after_close = NULL);

    // void copyStateFrom  (RawPipeline * p){
    //     std::cout << p->state_size << std::endl;
    //     memcpy(state, p->state, p->state_size);
    //     std::cout << ((void **) state)[0] << std::endl;
    //     std::cout << ((void **) state)[1] << std::endl;
    //     std::cout << ((void **) state)[2] << std::endl;
    //     std::cout << ((void **) p->state)[0] << std::endl;
    //     std::cout << ((void **) p->state)[1] << std::endl;
    //     std::cout << ((void **) p->state)[2] << std::endl;
    // }

    // void copyStateBackTo(RawPipeline * p){
    //     memcpy(p->state, state, p->state_size);
    // }

    friend class RawPipelineGen;
    friend class RawGpuPipelineGen;
    friend class RawCpuPipelineGen;
public:
    void     * state;

    ~RawPipeline();

    void * getState() const{
        return state;
    }

    size_t getSizeOf(llvm::Type * t) const;

    template<typename T>
    void setStateVar(size_t state_id, const T &value){
        size_t offset = layout.getStructLayout(state_type)->getElementOffset(state_id);

        *((T *) (((char *) state) + offset)) = value;
    }

    template<typename T>
    T getStateVar(size_t state_id){
        size_t offset = layout.getStructLayout(state_type)->getElementOffset(state_id);

        return *((T *) (((char *) state) + offset));
    }

    int32_t getGroup() const;

    virtual execution_conf getExecConfiguration() const{
        return execution_conf{};
    }

    virtual void open();
    
    template<typename... Tin>
    void consume(size_t N, const Tin * ... src){ //FIXME: cleanup + remove synchronization
        // ((void (*)(const Tin * ..., size_t, void *)) cons)(src..., N, state);
        ((void (*)(const Tin * ..., void *)) cons)(src..., state);
    }//;// cnt_t N, vid_t v, cid_t c){

    // template<typename... Tin>
    // void consume(cnt_t N, const Tin * ... src){ //FIXME: cleanup + remove synchronization
    //     consume(N, 0, 0, src...);
    // }


    template<typename... Tin>
    void consume_gpu(size_t N, const Tin * ... src){ //FIXME: cleanup + remove synchronization
        void *KernelParams[] = {(&src)...,
                                &N,
                                state};

        launch_kernel((CUfunction) cons, KernelParams);

        // gpu_run(cudaDeviceSynchronize());
    }//;// cnt_t N, vid_t v, cid_t c){

    virtual void close();
};

// class GpuRawPipeline: public RawPipeline{
// private:
//     GpuRawPipeline(void * cons, size_t state_size, RawPipelineGen * gen, llvm::StructType * state_type,
//         const std::vector<std::function<void (RawPipeline * pip)>> &openers,
//         const std::vector<std::function<void (RawPipeline * pip)>> &closers,
//         int32_t group_id = 0); //FIXME: group id should be handled to comply with the requirements!

//     friend class GpuRawPipelineGen;
// public:
//     ~GpuRawPipeline();
    
    // template<typename... Tin>
    // virtual void consume(size_t N, const Tin * ... src){ //FIXME: cleanup + remove synchronization
    //     void *KernelParams[] = {(&src)...,
    //                             &N,
    //                             state};

    //     launch_kernel((CUfunction) cons, KernelParams);

    //     // gpu_run(cudaDeviceSynchronize());
    // }//;// cnt_t N, vid_t v, cid_t c){
// };


template<typename... Tin>
class RawPipelineOp{
private:
    RawPipeline * const pip;

public:
    RawPipelineOp(RawPipeline * pip): pip(pip){}

    void open(){
        pip->open();
    }
    
    void consume(const Tin * ... src, cnt_t N, vid_t v, cid_t c){ //FIXME: cleanup + remove synchronization
        pip->consume(N, src...);
    }

    void close(){
        pip->close();
    }
};

#endif /* RAW_PIPELINE_HPP_ */