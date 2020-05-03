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

#ifndef PARALLEL_CONTEXT_HPP_
#define PARALLEL_CONTEXT_HPP_

#include "util/context.hpp"

class PipelineGenFactory;
class PipelineGen;
class Pipeline;

class ParallelContext : public Context {
 public:
  ParallelContext(const string &moduleName, bool gpu_root = false);
  virtual ~ParallelContext();

  virtual size_t appendParameter(llvm::Type *ptype, bool noalias = false,
                                 bool readonly = false);
  virtual StateVar appendStateVar(llvm::Type *ptype, std::string name = "");
  virtual StateVar appendStateVar(llvm::Type *ptype,
                                  std::function<init_func_t> init,
                                  std::function<deinit_func_t> deinit,
                                  std::string name = "");

  virtual llvm::Argument *getArgument(size_t id) const;
  virtual llvm::Value *getStateVar(const StateVar &id) const;
  virtual llvm::Value *getStateVar() const;
  virtual llvm::Value *getSubStateVar() const;
  virtual std::vector<llvm::Type *> getStateVars() const;

  void registerOpen(const void *owner, std::function<void(Pipeline *pip)> open);
  void registerClose(const void *owner,
                     std::function<void(Pipeline *pip)> close);

  // void pushNewPipeline    (PipelineGen *copyStateFrom = nullptr);
  // void pushNewCpuPipeline (PipelineGen *copyStateFrom = nullptr);

 private:
  void pushDeviceProvider(PipelineGenFactory *factory);

  template <typename T>
  void pushDeviceProvider() {
    pushDeviceProvider(&(T::getInstance()));
  }

  void popDeviceProvider();

  friend class DeviceCross;
  friend class CpuToGpu;
  friend class GpuToCpu;

 public:
  void pushPipeline(PipelineGen *copyStateFrom = nullptr);
  void popPipeline();

  PipelineGen *removeLatestPipeline();
  PipelineGen *getCurrentPipeline() const;
  void setChainedPipeline(PipelineGen *next);

  virtual llvm::Module *getModule() const;
  virtual llvm::IRBuilder<> *getBuilder() const;
  llvm::Function *getFunction(string funcName) const;

  virtual void setGlobalFunction(bool leaf);
  virtual void setGlobalFunction(llvm::Function *F = nullptr,
                                 bool leaf = false);
  virtual void prepareFunction(llvm::Function *F) {}

  virtual llvm::Value *threadId();
  virtual llvm::Value *threadIdInBlock();
  virtual llvm::Value *blockId();
  virtual llvm::Value *blockDim();
  virtual llvm::Value *gridDim();
  virtual llvm::Value *threadNum();
  virtual llvm::Value *laneId();
  virtual void createMembar_gl();
  virtual void workerScopedMembar();

  virtual void log(llvm::Value *out,
                   decltype(__builtin_FILE()) file = __builtin_FILE(),
                   decltype(__builtin_LINE()) line = __builtin_LINE());

  [[nodiscard]] virtual llvm::Function *getFunctionOverload(std::string name,
                                                            llvm::Type *type);
  [[nodiscard]] virtual std::string getFunctionNameOverload(std::string name,
                                                            llvm::Type *type);

  virtual llvm::BasicBlock *getEndingBlock();
  virtual void setEndingBlock(llvm::BasicBlock *codeEnd);
  virtual llvm::BasicBlock *getCurrentEntryBlock();
  virtual void setCurrentEntryBlock(llvm::BasicBlock *codeEntry);

  virtual llvm::Value *workerScopedAtomicAdd(llvm::Value *ptr,
                                             llvm::Value *inc);
  virtual llvm::Value *workerScopedAtomicXchg(llvm::Value *ptr,
                                              llvm::Value *val);

  virtual llvm::Value *allocateStateVar(llvm::Type *t);
  virtual void deallocateStateVar(llvm::Value *v);

  // string emitPTX();

  void compileAndLoad();

  // std::vector<CUfunction> getKernel();
  std::vector<std::unique_ptr<Pipeline>> getPipelines();

  // Provide support for some extern functions
  virtual void registerFunction(const char *funcName, llvm::Function *func);

  PipelineGen *operator->() const { return getCurrentPipeline(); }

 protected:
  virtual void createJITEngine();

 public:
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine;
  llvm::ExecutionEngine *TheExecutionEngine;
  llvm::ExecutionEngine *TheCPUExecutionEngine;

  // CUmodule cudaModule;

 protected:
  string kernelName;
  size_t pip_cnt;

  std::vector<PipelineGenFactory *> pipFactories;

  // Module * TheCPUModule;

  std::vector<PipelineGen *> pipelines;

  std::vector<PipelineGen *> generators;

  std::vector<bool> leafpip;

  std::vector<bool> leafgen;
};

#endif /* PARALLEL_CONTEXT_HPP_ */
