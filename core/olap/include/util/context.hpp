/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

/** Original code of the following functions derived from
 *  the source code of Impala (https://github.com/cloudera/Impala/):
 *  CreateForLoop,
 *  CreateIfElseBlocks,
 *  CastPtrToLlvmPtr,
 *  CodegenMemcpy
 */

#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include <llvm/IR/IRBuilder.h>

#include "common/common.hpp"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "memory/memory-allocator.hpp"
#include "operators/operator-state.hpp"
#include "util/jit/control-flow/if-statement.hpp"
#include "values/expressionTypes.hpp"

#define MODULEPASS 0

//#ifdef DEBUG
#define DEBUGCTX
//#endif

void addOptimizerPipelineDefault(llvm::legacy::FunctionPassManager *TheFPM);

#if MODULEPASS
void __attribute__((unused))
addOptimizerPipelineInlining(ModulePassManager *TheMPM);
#endif

void __attribute__((unused))
addOptimizerPipelineVectorization(llvm::legacy::FunctionPassManager *TheFPM);

extern bool print_generated_code;

class StateVar {
 public:
  size_t index_in_pip;
  const void *pip;

 public:
  constexpr StateVar() : index_in_pip(~size_t{0}), pip(nullptr) {}
  // constexpr StateVar(StateVar &&) = default;
  // constexpr StateVar(const StateVar &) = default;
  // StateVar &operator=(StateVar &&) = default;

  inline bool operator==(const StateVar &o) const {
    return index_in_pip == o.index_in_pip && pip == o.pip;
  }

 private:
  constexpr StateVar(size_t index, const void *pip)
      : index_in_pip(index), pip(pip) {}

 private:
  size_t getIndex() const { return index_in_pip; }
  friend class Context;
  friend class Pipeline;
  friend class PipelineGen;
  friend class GpuPipelineGen;
};

std::string getFunctionName(void *f);

class Context {
 private:
  const std::string moduleName;

 public:
  typedef llvm::Value *(init_func_t)(llvm::Value *);
  typedef void(deinit_func_t)(llvm::Value *, llvm::Value *);

 protected:
  Context(const string &moduleName);

 public:
  virtual ~Context() {
    LOG(WARNING) << "[Context: ] Destructor";
    // XXX Has to be done in an appropriate sequence - segfaults otherwise
    //        delete Builder;
    //            delete TheFPM;
    //            delete TheExecutionEngine;
    //            delete TheFunction;
    //            delete llvmContext;
    //            delete TheFunction;
  }

  std::string getModuleName() const { return moduleName; }

  llvm::LLVMContext &getLLVMContext() { return getModule()->getContext(); }

  virtual void prepareFunction(llvm::Function *F) = 0;

  llvm::ExecutionEngine const *getExecEngine() { return TheExecutionEngine; }

  virtual void setGlobalFunction(bool leaf);
  virtual void setGlobalFunction(llvm::Function *F = nullptr,
                                 bool leaf = false);
  llvm::Function *getGlobalFunction() const { return TheFunction; }
  virtual llvm::Module *getModule() const = 0;
  virtual llvm::IRBuilder<> *getBuilder() const = 0;
  virtual llvm::Function *getFunction(string funcName) const;

  llvm::ConstantInt *createInt8(char val);
  llvm::ConstantInt *createInt32(int val);
  llvm::ConstantInt *createInt64(int val);
  llvm::ConstantInt *createInt64(unsigned int val);
  llvm::ConstantInt *createInt64(size_t val);
  llvm::ConstantInt *createInt64(int64_t val);
  llvm::ConstantInt *createSizeT(size_t val);
  llvm::ConstantInt *createTrue();
  llvm::ConstantInt *createFalse();

  llvm::IntegerType *createSizeType();

  virtual size_t getSizeOf(llvm::Type *type) const;
  virtual size_t getSizeOf(llvm::Value *val) const;

  llvm::Type *CreateCustomType(char *typeName);
  llvm::StructType *CreateJSMNStruct();
  static llvm::StructType *CreateJSMNStruct(llvm::LLVMContext &);
  llvm::StructType *CreateStringStruct();
  static llvm::StructType *CreateStringStruct(llvm::LLVMContext &);
  llvm::PointerType *CreateJSMNStructPtr();
  static llvm::PointerType *CreateJSMNStructPtr(llvm::LLVMContext &);
  llvm::StructType *CreateJSONPosStruct();
  static llvm::StructType *CreateCustomStruct(llvm::LLVMContext &,
                                              vector<llvm::Type *> innerTypes);
  llvm::StructType *CreateCustomStruct(vector<llvm::Type *> innerTypes);
  llvm::StructType *ReproduceCustomStruct(list<typeID> innerTypes);

  template <typename InputIt>
  llvm::Value *constructStruct(InputIt begin, InputIt end) {
    std::vector<llvm::Type *> types;
    types.reserve(end - begin);
    for (auto it = begin; it != end; ++it) {
      types.emplace_back((*it)->getType());
    }
    return constructStruct(begin, end,
                           llvm::StructType::get(getLLVMContext(), types));
  }

  template <typename InputIt>
  llvm::Value *constructStruct(InputIt begin, InputIt end,
                               llvm::StructType *structType) {
    llvm::Value *agg = llvm::UndefValue::get(structType);
    size_t i = 0;
    for (auto it = begin; it != end; ++it) {
      agg = getBuilder()->CreateInsertValue(agg, *it, i++);
    }
    return agg;
  }

  inline llvm::Value *constructStruct(
      std::initializer_list<llvm::Value *> elems) {
    return constructStruct(elems.begin(), elems.end());
  }

  inline llvm::Value *constructStruct(
      std::initializer_list<llvm::Value *> elems, llvm::StructType *type) {
    return constructStruct(elems.begin(), elems.end(), type);
  }

  /**
   * Does not involve AllocaInst, but still is a memory position
   * NOTE: 1st elem of Struct is 0!!
   */
  llvm::Value *getStructElem(llvm::Value *mem_struct, int elemNo);
  llvm::Value *getStructElem(llvm::AllocaInst *mem_struct, int elemNo);
  void updateStructElem(llvm::Value *toStore, llvm::Value *mem_struct,
                        int elemNo);
  llvm::Value *getStructElemMem(llvm::Value *mem_struct, int elemNo);
  llvm::Value *CreateGlobalString(char *str);
  llvm::Value *CreateGlobalString(const char *str);
  llvm::PointerType *getPointerType(llvm::Type *type);

  // Utility functions, similar to ones from Impala
  llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction,
                                           const std::string &VarName,
                                           llvm::Type *varType,
                                           llvm::Value *arraySize = nullptr);

  llvm::AllocaInst *CreateEntryBlockAlloca(const std::string &VarName,
                                           llvm::Type *varType,
                                           llvm::Value *arraySize = nullptr);

  llvm::AllocaInst *createAlloca(llvm::BasicBlock *InsertAtBB,
                                 const string &VarName, llvm::Type *varType);

  void CreateForLoop(const string &cond, const string &body, const string &inc,
                     const string &end, llvm::BasicBlock **cond_block,
                     llvm::BasicBlock **body_block,
                     llvm::BasicBlock **inc_block, llvm::BasicBlock **end_block,
                     llvm::BasicBlock *insert_before = nullptr);

  void CreateIfElseBlocks(llvm::Function *fn, const string &if_name,
                          const string &else_name, llvm::BasicBlock **if_block,
                          llvm::BasicBlock **else_block,
                          llvm::BasicBlock *insert_before = nullptr);
  void CreateIfBlock(llvm::Function *fn, const string &if_name,
                     llvm::BasicBlock **if_block,
                     llvm::BasicBlock *insert_before = nullptr);

  llvm::BasicBlock *CreateIfBlock(llvm::Function *fn, const string &if_label,
                                  llvm::BasicBlock *insert_before = nullptr);

  llvm::Value *CastPtrToLlvmPtr(llvm::PointerType *type, const void *ptr);
  llvm::Value *getArrayElem(llvm::AllocaInst *mem_ptr, llvm::Value *offset);
  llvm::Value *getArrayElem(llvm::Value *val_ptr, llvm::Value *offset);
  /**
   * Does not involve AllocaInst, but still returns a memory position
   */
  llvm::Value *getArrayElemMem(llvm::Value *val_ptr, llvm::Value *offset);

  virtual void log(llvm::Value *out,
                   decltype(__builtin_FILE()) file = __builtin_FILE(),
                   decltype(__builtin_LINE()) line = __builtin_LINE()) {
    google::LogMessage(file, line, google::GLOG_INFO).stream()
        << "Unimplemented";
  }

  inline ProteusValueMemory toMem(llvm::Value *val, llvm::Value *isNull,
                                  const std::string &name) {
    auto mem = CreateEntryBlockAlloca(name, val->getType());
    getBuilder()->CreateStore(val, mem);
    return {mem, isNull};
  }

  inline ProteusValueMemory toMem(
      llvm::Value *val, llvm::Value *isNull,
      /* The weird argument order is to avoid conflicts with the above def */
      decltype(__builtin_LINE()) line = __builtin_LINE(),
      decltype(__builtin_FILE()) file = __builtin_FILE()) {
    return toMem(val, isNull, std::string{file} + std::to_string(line));
  }

  inline ProteusValueMemory toMem(
      const ProteusValue &val,
      /* The weird argument order is to avoid conflicts with the above def */
      decltype(__builtin_LINE()) line = __builtin_LINE(),
      decltype(__builtin_FILE()) file = __builtin_FILE()) {
    return toMem(val.value, val.isNull, line, file);
  }

  inline ProteusValueMemory toMem(const ProteusValue &val,
                                  const std::string &name) {
    return toMem(val.value, val.isNull, name);
  }

  // Not used atm
  void CodegenMemcpy(llvm::Value *dst, llvm::Value *src, int size);
  void CodegenMemcpy(llvm::Value *dst, llvm::Value *src, llvm::Value *size);

  void CodegenMemset(llvm::Value *dst, llvm::Value *byte, int size);
  void CodegenMemset(llvm::Value *dst, llvm::Value *bytes, llvm::Value *size);

  virtual void registerFunction(const char *, llvm::Function *);
  virtual llvm::BasicBlock *getEndingBlock() { return codeEnd; }
  virtual void setEndingBlock(llvm::BasicBlock *codeEnd) {
    this->codeEnd = codeEnd;
  }
  virtual llvm::BasicBlock *getCurrentEntryBlock() {
    assert(currentCodeEntry != nullptr && "No entry block is set!");
    assert(currentCodeEntry->getTerminator() == nullptr &&
           "Current entry block is terminated!");
    return currentCodeEntry;
  }
  virtual void setCurrentEntryBlock(llvm::BasicBlock *codeEntry) {
    this->currentCodeEntry = codeEntry;
  }

  virtual if_branch gen_if(ProteusValue cond);

  virtual if_branch gen_if(const expression_t &expr,
                           const OperatorState &state);

  template <typename T>
  llvm::Type *toLLVM() {
    if constexpr (std::is_pointer_v<T>) {
      return llvm::PointerType::getUnqual(
          toLLVM<std::remove_cv_t<std::remove_pointer_t<T>>>());
    } else if constexpr (std::is_integral_v<T>) {
      return llvm::Type::getIntNTy(getLLVMContext(), sizeof(T) * 8);
    } else if constexpr (std::is_same_v<T, double>) {
      return llvm::Type::getDoubleTy(getLLVMContext());
    } else if constexpr (std::is_same_v<T, float>) {
      return llvm::Type::getFloatTy(getLLVMContext());
    } else {
      LOG(INFO) << "Unknown type " << typeid(T).name()
                << " substituting with sized placeholder";
      return llvm::Type::getIntNTy(getLLVMContext(), sizeof(T) * 8);
    }
  }

  template <typename R, typename... Args>
  llvm::Value *gen_call(R (*func)(Args...),
                        std::initializer_list<llvm::Value *> args) {
    auto fname = getFunctionName((void *)func);
    return gen_call(fname, args, toLLVM<std::remove_cv_t<R>>());
  }

  virtual llvm::Value *gen_call(std::string func,
                                std::initializer_list<llvm::Value *> args,
                                llvm::Type *ret = nullptr);

  virtual llvm::Value *gen_call(llvm::Function *func,
                                std::initializer_list<llvm::Value *> args);

  /**
   * Not sure the HT methods belong here
   */
  // Metadata maintained per bucket.
  // Will probably use an array of such structs per HT
  static llvm::StructType *getHashtableMetadataType(llvm::LLVMContext &ctx) {
    llvm::Type *int64_type = llvm::Type::getInt64Ty(ctx);
    llvm::Type *keyType = int64_type;
    llvm::Type *bucketSizeType = int64_type;
    vector<llvm::Type *> types_htMetadata{keyType, bucketSizeType};
    int htMetadataSize = (keyType->getPrimitiveSizeInBits() / 8);
    htMetadataSize += (bucketSizeType->getPrimitiveSizeInBits() / 8);

    // Result type specified
    return llvm::StructType::get(ctx, types_htMetadata);
  }

  llvm::StructType *getHashtableMetadataType() {
    return getHashtableMetadataType(getLLVMContext());
  }

  llvm::Value *getMemResultCtr() { return mem_resultCtr; }

  const char *getName();

  virtual StateVar appendStateVar(llvm::Type *ptype, std::string name = "");
  virtual StateVar appendStateVar(llvm::Type *ptype,
                                  std::function<init_func_t> init,
                                  std::function<deinit_func_t> deinit,
                                  std::string name = "");

  virtual llvm::Value *allocateStateVar(llvm::Type *t);
  virtual void deallocateStateVar(llvm::Value *v);

  virtual llvm::Value *getStateVar(const StateVar &id) const;

 protected:
  virtual void prepareStateVars();
  virtual void endStateVars();

  llvm::Module *TheModule;
  llvm::IRBuilder<> *TheBuilder = nullptr;

 public:
  llvm::ExecutionEngine *TheExecutionEngine;

 protected:
  // JIT Driver
  llvm::Function *TheFunction;
  map<string, llvm::Function *> availableFunctions;

  // Last (current) basic block. This changes every time a new scan is
  // triggered
  llvm::BasicBlock *codeEnd;
  // Current entry basic block. This changes every time a new scan is
  // triggered
  llvm::BasicBlock *currentCodeEntry;

  /**
   * Basic stats / info to be used during codegen
   */
  // XXX used to keep a counter of final output results
  // and be utilized in actions such as flushing out delimiters
  // NOTE: Must check whether sth similar is necessary for nested collections
  llvm::Value *mem_resultCtr;

  /**
   * Helper function to create the LLVM objects required for JIT execution. */
  virtual void createJITEngine();

 private:
  std::vector<std::tuple<llvm::Type *, std::string, std::function<init_func_t>,
                         std::function<deinit_func_t>>>
      to_prepare_state_vars;
  std::vector<llvm::Value *> state_vars;
};

typedef struct HashtableBucketMetadata {
  size_t hashKey;
  size_t bucketSize;
} HashtableBucketMetadata;

class save_current_blocks_and_restore_at_exit_scope {
  llvm::BasicBlock *current;
  llvm::BasicBlock *entry;
  llvm::BasicBlock *ending;
  Context *context;

 public:
  save_current_blocks_and_restore_at_exit_scope(Context *context)
      : context(context),
        entry(context->getCurrentEntryBlock()),
        ending(context->getEndingBlock()),
        current(context->getBuilder()->GetInsertBlock()) {}

  ~save_current_blocks_and_restore_at_exit_scope() {
    context->setCurrentEntryBlock(entry);
    context->setEndingBlock(ending);
    context->getBuilder()->SetInsertPoint(current);
  }
};

#endif /* CONTEXT_HPP_ */
