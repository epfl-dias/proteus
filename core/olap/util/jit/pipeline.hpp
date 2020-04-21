/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#ifndef PIPELINE_HPP_
#define PIPELINE_HPP_

#include <vector>

#include "util/context.hpp"

// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include <utility>

#include "common/gpu/gpu-common.hpp"

class Pipeline;

extern "C" {
void yield();
}

typedef void(opener_t)(Pipeline *);
typedef void(closer_t)(Pipeline *);

typedef llvm::Value *(init_func_t)(llvm::Value *);
typedef void(deinit_func_t)(llvm::Value *, llvm::Value *);

// __device__ void devprinti64(uint64_t x);

class PipelineGen {
 protected:
  // Last (current) basic block. This changes every time a new scan is triggered
  llvm::BasicBlock *codeEnd;
  // Current entry basic block. This changes every time a new scan is triggered
  llvm::BasicBlock *currentCodeEntry;

  std::vector<std::pair<std::function<init_func_t>, size_t>> open_var;
  llvm::Function *open__function;

  std::vector<std::pair<std::function<deinit_func_t>, size_t>> close_var;
  llvm::Function *close_function;

  std::vector<llvm::Type *> inputs;
  std::vector<bool> inputs_noalias;
  std::vector<bool> inputs_readonly;

  std::vector<llvm::Type *> state_vars;
  std::vector<llvm::Argument *> args;

  std::vector<std::pair<const void *, std::function<opener_t>>> openers;
  std::vector<std::pair<const void *, std::function<closer_t>>> closers;

  std::string pipName;
  Context *context;

  std::string func_name;

  void *func;

  llvm::Value *state;
  llvm::StructType *state_type;

  llvm::IRBuilder<> *TheBuilder;

  PipelineGen *copyStateFrom;

  PipelineGen *execute_after_close;

  //     //Used to include optimization passes
  //     legacy::FunctionPassManager * TheFPM        ;
  // #if MODULEPASS
  //     ModulePassManager           * TheMPM        ;
  // #endif

  //     legacy::PassManager         * ThePM         ;

  // ExecutionEngine             * TheExecutionEngine;

  map<string, llvm::Function *> availableFunctions;

  unsigned int maxBlockSize;
  unsigned int maxGridSize;

 public:
  llvm::Function *F;

 protected:
  PipelineGen(Context *context, std::string pipName = "pip",
              PipelineGen *copyStateFrom = nullptr);

  virtual ~PipelineGen() {}

 public:
  virtual size_t appendParameter(llvm::Type *ptype, bool noalias = false,
                                 bool readonly = false);
  virtual StateVar appendStateVar(llvm::Type *ptype);
  virtual StateVar appendStateVar(llvm::Type *ptype,
                                  std::function<init_func_t> init,
                                  std::function<deinit_func_t> deinit);

  void callPipRegisteredOpen(size_t indx, Pipeline *pip);
  void callPipRegisteredClose(size_t indx, Pipeline *pip);

  virtual llvm::Argument *getArgument(size_t id) const;
  virtual llvm::Value *getStateVar(StateVar id) const;
  virtual llvm::Value *getStateVar() const;
  virtual llvm::Value *getStateVarPtr() const;
  virtual llvm::Value *getSubStateVar() const;

  virtual llvm::Value *allocateStateVar(llvm::Type *t);
  virtual void deallocateStateVar(llvm::Value *v);

  virtual llvm::Function *prepare();
  virtual std::unique_ptr<Pipeline> getPipeline(int group_id = 0);
  virtual void *getKernel() const;

  virtual std::string convertTypeToFuncSuffix(llvm::Type *type);
  virtual llvm::Function *getFunctionOverload(std::string name,
                                              llvm::Type *type);
  virtual std::string getFunctionNameOverload(std::string name,
                                              llvm::Type *type);

  virtual void setChainedPipeline(PipelineGen *next) {
    assert(
        !execute_after_close &&
        "No support for multiple pipelines after a single one, create a chain");
    execute_after_close = next;
  }

  virtual void *getConsume() const { return getKernel(); }
  virtual llvm::Function *getLLVMConsume() const { return F; }

  std::string getName() const { return pipName; }

  virtual llvm::BasicBlock *getEndingBlock() { return codeEnd; }
  virtual void setEndingBlock(llvm::BasicBlock *codeEnd) {
    this->codeEnd = codeEnd;
  }
  virtual llvm::BasicBlock *getCurrentEntryBlock() { return currentCodeEntry; }
  virtual void setCurrentEntryBlock(llvm::BasicBlock *codeEntry) {
    this->currentCodeEntry = codeEntry;
  }

  virtual void setMaxWorkerSize(unsigned int maxBlock, unsigned int maxGrid) {
    maxBlockSize = std::min(maxBlockSize, maxBlock);
    maxGridSize = std::min(maxGridSize, maxGrid);
  }

  virtual void compileAndLoad() = 0;

  void registerOpen(const void *owner, std::function<void(Pipeline *pip)> open);
  void registerClose(const void *owner,
                     std::function<void(Pipeline *pip)> close);

  [[deprecated]] virtual llvm::Function *getFunction() const;

  virtual llvm::Module *getModule() const = 0;  //{return TheModule ;}
  virtual llvm::IRBuilder<> *getBuilder() const { return TheBuilder; }

  virtual void registerFunction(std::string, llvm::Function *);

  virtual llvm::Function *const getFunction(string funcName) const;

  virtual llvm::Function *const createHelperFunction(
      string funcName, std::vector<llvm::Type *> ins,
      std::vector<bool> readonly, std::vector<bool> noalias);
  virtual llvm::Value *invokeHelperFunction(
      llvm::Function *f, std::vector<llvm::Value *> args) const;

  std::vector<llvm::Type *> getStateVars() const;

  static void init();

  virtual llvm::Value *workerScopedAtomicAdd(llvm::Value *ptr,
                                             llvm::Value *inc);
  virtual llvm::Value *workerScopedAtomicXchg(llvm::Value *ptr,
                                              llvm::Value *val);

  virtual void workerScopedMembar();

 protected:
  virtual void registerSubPipeline();
  virtual size_t prepareStateArgument();
  virtual llvm::Value *getStateLLVMValue();
  virtual void prepareFunction();
  virtual void prepareInitDeinit();

 public:
  virtual void *getCompiledFunction(llvm::Function *f) = 0;

 protected:
  virtual void registerFunctions();
};

class Pipeline {
 protected:
  void *cons;
  llvm::StructType *state_type;
  const int32_t group_id;
  size_t state_size;
  const llvm::DataLayout &layout;

  std::vector<std::pair<const void *, std::function<opener_t>>> openers;
  std::vector<std::pair<const void *, std::function<closer_t>>> closers;

  void *init_state;
  void *deinit_state;

  std::shared_ptr<Pipeline> execute_after_close;

  struct guard {
    explicit guard(int) {}
  };

 public:
  Pipeline(guard, void *cons, size_t state_size, PipelineGen *gen,
           llvm::StructType *state_type,
           const std::vector<std::pair<const void *, std::function<opener_t>>>
               &openers,
           const std::vector<std::pair<const void *, std::function<closer_t>>>
               &closers,
           void *init_state, void *deinit_state,
           int32_t group_id = 0,  // FIXME: group id should be handled to comply
                                  // with the requirements!
           std::shared_ptr<Pipeline> execute_after_close = nullptr);

 protected:
  template <typename... T>
  static auto create(T &&... args) {
    return std::make_unique<Pipeline>(guard{0}, std::forward<T>(args)...);
  }
  // void copyStateFrom  (Pipeline * p){
  //     std::cout << p->state_size << std::endl;
  //     memcpy(state, p->state, p->state_size);
  //     std::cout << ((void **) state)[0] << std::endl;
  //     std::cout << ((void **) state)[1] << std::endl;
  //     std::cout << ((void **) state)[2] << std::endl;
  //     std::cout << ((void **) p->state)[0] << std::endl;
  //     std::cout << ((void **) p->state)[1] << std::endl;
  //     std::cout << ((void **) p->state)[2] << std::endl;
  // }

  // void copyStateBackTo(Pipeline * p){
  //     memcpy(p->state, state, p->state_size);
  // }

  friend class PipelineGen;
  friend class GpuPipelineGen;
  friend class CpuPipelineGen;

 public:
  void *state;

  virtual ~Pipeline();

  void *getState() const { return state; }

  size_t getSizeOf(llvm::Type *t) const;

  template <typename T>
  void setStateVar(StateVar state_id, const T &value) {
    size_t offset = layout.getStructLayout(state_type)
                        ->getElementOffset(state_id.getIndex());

    *((T *)(((char *)state) + offset)) = value;
  }

  template <typename T>
  T getStateVar(StateVar state_id) {
    size_t offset = layout.getStructLayout(state_type)
                        ->getElementOffset(state_id.getIndex());

    return *((T *)(((char *)state) + offset));
  }

  int32_t getGroup() const;

  virtual execution_conf getExecConfiguration() const {
    return execution_conf{};
  }

  virtual void open();

  template <typename... Tin>
  void consume(size_t N,
               const Tin *... src) {  // FIXME: cleanup + remove synchronization
    // ((void (*)(const Tin * ..., size_t, void *)) cons)(src..., N, state);
    ((void (*)(const Tin *..., void *))cons)(src..., state);
  }  //;// cnt_t N, vid_t v, cid_t c){

  virtual void close();
};

class PipelineGenFactory {
 protected:
  PipelineGenFactory() {}

  virtual ~PipelineGenFactory() {}

 public:
  virtual PipelineGen *create(Context *context, std::string pipName = "pip",
                              PipelineGen *copyStateFrom = nullptr) = 0;
};

#endif /* PIPELINE_HPP_ */
