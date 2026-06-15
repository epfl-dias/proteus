/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#ifndef PROTEUS_PIPELINE_HPP
#define PROTEUS_PIPELINE_HPP

#include <cstdint>
#include <limits>
#include <vector>

// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include <llvm/IR/BasicBlock.h>

#include <codegen/context/context.hpp>
#include <future>
#include <platform/common/gpu/gpu-common.hpp>
#include <stduuid/uuid.hpp>
#include <utility>

class Pipeline;

extern "C" {
void yield();
}

typedef void(opener_t)(Pipeline *);
typedef void(closer_t)(Pipeline *);

typedef llvm::Value *(init_func_t)(llvm::Value *);
typedef void(deinit_func_t)(llvm::Value *, llvm::Value *);

class PipelineGen {
 protected:
  /// Last (current) basic block. This changes every time a new scan is
  /// triggered
  llvm::BasicBlock *codeEnd;
  /// Current entry basic block. This changes every time a new scan is triggered
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
  [[deprecated]] Context *context;

  std::shared_future<void *> compiledFunctionFuture;

  llvm::Value *state;
  llvm::StructType *state_type;
  size_t state_size;  /// in bytes

  llvm::IRBuilder<> *TheBuilder;

  PipelineGen *copyStateFrom;

  PipelineGen *execute_after_close;

  map<string, llvm::Function *> availableFunctions;

  unsigned int maxBlockSize;
  unsigned int maxGridSize;

  StateVar session_parameters_ptr;
  const uuids::uuid id;  /// Unique identifier for this PipelineGen

 public:
  llvm::Function *F;

 protected:
  PipelineGen(Context *context, std::string pipName = "pip",
              PipelineGen *copyStateFrom = nullptr);

  virtual ~PipelineGen() { compiledFunctionFuture.wait(); }

  friend class ParallelContext;

 public:
  uuids::uuid getUUID() const { return id; }
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
  virtual void *getKernel();

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

  virtual void *getConsume() { return getKernel(); }
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
    const auto requestedBlock = std::min(maxBlockSize, maxBlock);

    // Keep roughly the same total thread budget when lowering block size.
    // This avoids severe under-launch (e.g. 1024->128 threads/block with an
    // unchanged grid).
    //
    // If the caller provides a larger maxGrid, treat it as an explicit request
    // for more launch parallelism than the preserved-budget minimum.
    if (requestedBlock < maxBlockSize && requestedBlock > 0) {
      const uint64_t threadBudget =
          static_cast<uint64_t>(maxBlockSize) * maxGridSize;
      const uint64_t requestedBudget =
          static_cast<uint64_t>(requestedBlock) * maxGrid;
      const uint64_t desiredBudget = std::max(threadBudget, requestedBudget);
      uint64_t scaledGrid =
          (desiredBudget + requestedBlock - 1) / requestedBlock;
      maxGridSize = static_cast<unsigned int>(
          std::min<uint64_t>(scaledGrid, std::numeric_limits<unsigned int>::max()));
    }

    maxBlockSize = requestedBlock;
    maxGridSize = std::min(maxGridSize, maxGrid);
  }
  [[nodiscard]] unsigned int getMaxBlockSize() const { return maxBlockSize; }
  [[nodiscard]] unsigned int getMaxGridSize() const { return maxGridSize; }

  virtual void compileAndLoad() = 0;

  void registerOpen(const void *owner, std::function<void(Pipeline *pip)> open);
  void registerClose(const void *owner,
                     std::function<void(Pipeline *pip)> close);

  [[deprecated]] virtual llvm::Function *getFunction() const;

  virtual llvm::Module *getModule() const = 0;
  virtual const llvm::DataLayout &getDataLayout() const {
    return getModule()->getDataLayout();
  }
  virtual llvm::IRBuilder<> *getBuilder() const {
    assert(TheBuilder && "Is the function signature ready?");
    return TheBuilder;
  }

  virtual void registerFunction(std::string, llvm::Function *);

  [[nodiscard]] virtual llvm::Function *getFunction(string funcName) const;

  virtual llvm::Function *const createHelperFunction(
      string funcName, std::vector<llvm::Type *> ins,
      std::vector<bool> readonly, std::vector<bool> noalias);
  virtual llvm::Value *invokeHelperFunction(
      llvm::Function *f, std::vector<llvm::Value *> args) const;

  std::vector<llvm::Type *> getStateVars() const;
  [[nodiscard]] llvm::Value *getSessionParametersPtr() const;

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

  const uuids::uuid id;          /// Unique identifier for the pipeline
  const uuids::uuid pip_gen_id;  /// Unique identifier of the PipelineGen used
                                 /// to create this pipeline

  std::shared_ptr<Pipeline> execute_after_close;

  struct guard {
    explicit guard(int) {}
  };

 public:
  /**
   * @brief Constructor for Pipeline.
   * @param cons Pointer to the function representing the pipeline. @see
   * PipelineGen::getKernel
   * @param state_size Size in bytes of the state required by the pipeline.
   * @param gen Pointer to the PipelineGen that generated this pipeline. @see
   * PipelineGen::getPipeline
   * @param state_type LLVM type of the pipeline's state. @see
   * PipelineGen::prepareStateArgument
   * @param openers Function to be called when opening the pipeline.
   * @param closers Functions to be called when closing the pipeline.
   * @param init_state Pointer to the function for initializing the pipeline's
   * state. @see PipelineGen::prepareInitDeinit
   * @param deinit_state Pointer to the function for deinitializing the
   * pipeline's state.
   * @param group_id Identifier for the group to which the pipeline belongs.
   * @param execute_after_close Pointer to another pipeline to be executed after
   * this one closes.
   */
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
  static auto create(T &&...args) {
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

  const void *session = nullptr;

 public:
  void *state;

  [[nodiscard]] uuids::uuid getUUID() const { return id; }
  [[nodiscard]] uuids::uuid getGeneratorUUID() const { return pip_gen_id; }

  virtual ~Pipeline();

  void *getState() const { return state; }

  /**
   * @brief Get the size in bytes of the given LLVM type.
   */
  size_t getSizeOf(llvm::Type *t) const;

  template <typename T>
  void setStateVar(StateVar state_id, const T &value) {
    size_t offset = layout.getStructLayout(state_type)
                        ->getElementOffset(state_id.getIndex());

    *((T *)(((char *)state) + offset)) = value;
  }

  [[nodiscard]] const void *getSession() const {
    assert(session);
    return session;
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

  /**
   * @brief Opens the pipeline for execution and acquires resources.
   * @param session Pointer to the session associated with the pipeline
   * execution. The session is currently always a pointer to int64_t. The
   * session is shared with any chained pipelines and any pipelines that
   * copyStateFrom this pipeline. Sessions can be useful for passing constants
   * to a pipeline.
   * @see ExpressionGeneratorVisitor::visit(const
   * expressions::PlaceholderExpression *e) for an example
   */
  virtual void open(const void *session);

  /**
   * Invoke the compiled pipeline function with the given arguments.
   *
   * @tparam Tin types of the arguments
   * @param src pointers to the arguments
   */
  template <typename... Tin>
  void consume(const Tin *...src) {
    assert(this);
    assert(cons);
    assert(getSession());
    ((void (*)(const Tin *..., void *))cons)(src..., state);
  }

  /**
   * @brief Closes the pipeline after execution and free resources.
   */
  virtual void close();
};

class PipelineGenFactory {
 protected:
  PipelineGenFactory() {}

  virtual ~PipelineGenFactory() {}

  virtual void registerFunctions(PipelineGen *pipelineGen);

 public:
  virtual PipelineGen *create(Context *context, std::string pipName = "pip",
                              PipelineGen *copyStateFrom = nullptr) = 0;
};

#endif  // PROTEUS_PIPELINE_HPP
