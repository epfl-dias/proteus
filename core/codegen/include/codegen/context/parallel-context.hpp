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
#ifndef PROTEUS_PARALLEL_CONTEXT_HPP
#define PROTEUS_PARALLEL_CONTEXT_HPP

#include <codegen/context/context.hpp>

class PipelineGenFactory;
class PipelineGen;
class Pipeline;

class ParallelContext : public Context {
 public:
  /**
   * @brief Initialize the parallel context with the corresponding pipeline
   * generator factory (CPU or GPU) and create an initial pipeline generator
   * using the factory
   *
   * @param moduleName Name of the LLVM module that the context is generating
   * code in
   * @param gpuRoot Indicates whether the root of the execution is on GPU.
   * @return A pointer to the prepared ParallelContext instance.
   */
  static ParallelContext *prepareParallelContext(const string &moduleName,
                                                 bool gpuRoot);
  ~ParallelContext() override;

  /**
   * @brief Append a parameter to the main pipeline function
   *
   * Accessible inside the pipeline codegen through @see getArgument. The
   * parameter is passed to the pipeline as an additional argument of the @see
   * Pipeline::consume function.
   * @param ptrType The LLVM type of the parameter. The parameter type must be a
   * pointer type
   * @param noalias Indicates if the parameter has no alias.
   * @param readonly Indicates if the parameter is read-only.
   * @return The index of the appended parameter.
   */
  virtual size_t appendParameter(llvm::Type *ptrType, bool noalias = false,
                                 bool readonly = false);

  /**
   * @brief Appends a state variable to the current pipeline.
   * A StateVar maintains state across invocations of pip->consume
   * @param ptrType The LLVM type of the state variable. Must be a pointer type
   * @param name Optional name for the state variable. Currently unused.
   * @return A StateVar object representing the appended state variable. This
   * can be used to fetch the llvm::Value* pointer (the state var itself) using
   * @see getStateVar call
   */
  StateVar appendStateVar(llvm::Type *ptrType, std::string name = "") override;

  /**
   * @brief Appends a state variable with initialization and deinitialization
   * functions.
   * @param ptrType The LLVM type of the state variable. Must be a pointer type
   * @param init A function to initialize the state variable. Initialization
   * occurs at open time
   * @param deinit A function to deinitialize the state variable.
   * Deinitialization occurs at close time
   * @param name Optional name for the state variable. Currently unused.
   * @return A StateVar object representing the appended state variable. This
   * can be used to fetch the llvm::Value* pointer (the state var itself) using
   * @see getStateVar call
   * @note This is useful in situations when you want to initialize memory for
   * the state var inside the generated code.
   */
  StateVar appendStateVar(llvm::Type *ptrType, std::function<init_func_t> init,
                          std::function<deinit_func_t> deinit,
                          std::string name = "") override;

  [[nodiscard]] virtual llvm::Value *getSessionParametersPtr() const;

  /**
   * @brief Retrieves the LLVM argument based on its ID for the current
   * PipelineGen Arguments are the actual data that is passed to the function.
   * @param idx The index of the argument from the appendParameter call.
   * @return A pointer to the LLVM argument.
   */
  [[nodiscard]] virtual llvm::Argument *getArgument(size_t idx) const;
  [[nodiscard]] llvm::Value *getStateVar(const StateVar &idx) const override;
  [[nodiscard]] virtual llvm::Value *getStateVar() const;
  [[nodiscard]] virtual llvm::Value *getSubStateVar() const;
  [[nodiscard]] virtual std::vector<llvm::Type *> getStateVars() const;

  /**
   * @brief Registers an open function for the current PipelineGen.
   * The registered open function will be propagated to the Pipeline and
   * executed during the @see Pipeline::open method with the pointer to the
   * Pipeline passed as an argument
   * @param owner Pointer to the object that the open method belongs to.
   * @param open The open function to be registered. This is will be executed
   * before the first consume of the pipeline is called.
   */
  void registerOpen(const void *owner, std::function<void(Pipeline *pip)> open);

  /**
   * @brief Registers a close function for the current pipeline.
   * The registered close function will be propagated to the Pipeline and
   * executed during the @see Pipeline::close method with the pointer to the
   * Pipeline passed as an argument
   * @param owner Pointer to the object that the close method belongs to.
   * @param close The close function to be registered. This is executed after
   * the pipeline has finished consuming all the data.
   */
  void registerClose(const void *owner,
                     std::function<void(Pipeline *pip)> close);

 public:
  /**
   * @brief Pushes a new pipeline generator created using the most recently
   * pushed PipelineGenFactory (or the initial factory created in the
   * constructor)
   * @param copyStateFrom Optional parameter to copy state from another
   * pipeline. Note: if this parameter is passed, then ParallelContext will
   * store the state of the copyStateFrom as the state variable in the current
   * Pipeline. See ChainedPipelinesWithArguments test as an example of usage
   * @param pip_name_prefix optional prefix for the pipeline name. Can be useful
   * for debugging to more easily find the relevant generated IR when
   * print_generated_code is set
   */
  void pushPipeline(PipelineGen *copyStateFrom = nullptr,
                    std::optional<std::string> pip_name_prefix = std::nullopt);
  void popPipeline();

  /**
   * @brief Removes the latest PipelineGen from the context.
   * @return A pointer to the removed PipelineGen.
   */
  [[nodiscard]] PipelineGen *removeLatestPipeline();
  [[nodiscard]] PipelineGen *getCurrentPipeline() const;
  /**
   * @brief Sets the next pipeline to be executed after the current pipeline
   * closes.
   * @param next The next PipelineGen to be executed.
   */
  void setChainedPipeline(PipelineGen *next);

  [[nodiscard]] llvm::Module *getModule() const override;
  [[nodiscard]] llvm::IRBuilder<> *getBuilder() const override;
  [[nodiscard]] llvm::Function *getFunction(string funcName) const override;

  void setGlobalFunction(bool leaf) override;
  void setGlobalFunction(llvm::Function *F = nullptr,
                         bool leaf = false) override;
  void prepareFunction(llvm::Function *F) override {}

  virtual llvm::Value *threadId();
  virtual llvm::Value *threadIdInBlock();
  virtual llvm::Value *blockId();
  virtual llvm::Value *blockDim();
  virtual llvm::Value *gridDim();
  virtual llvm::Value *threadNum();
  virtual llvm::Value *laneId();
  virtual void createMembar_gl();
  virtual void workerScopedMembar();

  void log(llvm::Value *out, decltype(__builtin_FILE()) file = __builtin_FILE(),
           decltype(__builtin_LINE()) line = __builtin_LINE()) override;

  [[nodiscard]] virtual llvm::Function *getFunctionOverload(std::string name,
                                                            llvm::Type *type);
  [[nodiscard]] virtual std::string getFunctionNameOverload(std::string name,
                                                            llvm::Type *type);

  llvm::BasicBlock *getEndingBlock() override;
  void setEndingBlock(llvm::BasicBlock *codeEnd) override;
  llvm::BasicBlock *getCurrentEntryBlock() override;
  void setCurrentEntryBlock(llvm::BasicBlock *codeEntry) override;

  virtual llvm::Value *workerScopedAtomicAdd(llvm::Value *ptr,
                                             llvm::Value *inc);
  virtual llvm::Value *workerScopedAtomicXchg(llvm::Value *ptr,
                                              llvm::Value *val);

  llvm::Value *allocateStateVar(llvm::Type *t) override;
  void deallocateStateVar(llvm::Value *v) override;

  // string emitPTX();

  void linkExternModule(std::unique_ptr<llvm::Module> module) override;

  /**
   * @brief Compiles and loads the generated code into the current context. This
   * is non-blocking, compilation occurs asynchronously
   */
  void compileAndLoad();

  /**
   * @brief For each leaf PipelineGen, waits for the function to complete
   * compilation and set up the Pipeline
   * @return A vector of unique pointers to the leaf pipelines.
   */
  std::vector<std::unique_ptr<Pipeline>> getPipelines();

  /**
   * @brief Builds leaf Pipeline objects while retaining PipelineGen ownership
   * inside the context.
   *
   * This is used when another object (e.g. PreparedStatement) needs the
   * compiled modules/functions to remain alive for the lifetime of the
   * returned pipelines.
   */
  std::vector<std::unique_ptr<Pipeline>> getPipelinesRetained();

  // Provide support for some extern functions
  void registerFunction(const char *funcName, llvm::Function *func) override;

  PipelineGen *operator->() const { return getCurrentPipeline(); }

 protected:
  explicit ParallelContext(const string &moduleName);

  void pushDeviceProvider(PipelineGenFactory *factory);

  template <typename T>
  void pushDeviceProvider() {
    pushDeviceProvider(&(T::getInstance()));
  }

  void popDeviceProvider();

 public:
  std::unique_ptr<llvm::TargetMachine> TheTargetMachine;

 protected:
  string kernelName;

  std::vector<PipelineGenFactory *> pipFactories;

  std::vector<PipelineGen *> pipelines;

  std::vector<PipelineGen *> generators;

  std::vector<bool> leafpip;

  std::vector<bool> leafgen;
};

#endif  // PROTEUS_PARALLEL_CONTEXT_HPP
