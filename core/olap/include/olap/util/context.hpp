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

#include <platform/common/common.hpp>
#include <platform/memory/memory-allocator.hpp>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "olap/operators/operator-state.hpp"
#include "olap/util/jit/control-flow/if-statement.hpp"
#include "olap/values/expressionTypes.hpp"

#define MODULEPASS 0

//#ifdef DEBUG
#define DEBUGCTX
//#endif

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
  [[nodiscard]] size_t getIndex() const { return index_in_pip; }
  friend class Context;
  friend class Pipeline;
  friend class PipelineGen;
  friend class GpuPipelineGen;
};

struct pb {
  void *p1;
  void *p2;
};

std::string getFunctionName(void *f);

class While {
 private:
  std::function<ProteusValue()> cond;
  Context *context;

 private:
  While(std::function<ProteusValue()> cond, Context *context)
      : cond(std::move(cond)), context(context) {}

  friend class Context;

 public:
  template <typename Fbody>
  void operator()(Fbody body) &&;
};

class DoWhile {
 private:
  std::function<void()> body;
  Context *context;

 private:
  DoWhile(std::function<void()> body, Context *context)
      : body(std::move(body)), context(context) {
    assert(context);
  }

  friend class Context;

 public:
  ~DoWhile() { assert(!context && "gen_do without body?"); }

  void gen_while(const std::function<ProteusValue()> &cond) &&;
};

class Context {
 private:
  const std::string moduleName;

 public:
  using init_func_t = llvm::Value *(llvm::Value *);
  using deinit_func_t = void(llvm::Value *, llvm::Value *);

 protected:
  explicit Context(const string &moduleName);

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

  [[nodiscard]] std::string getModuleName() const { return moduleName; }

  [[nodiscard]] llvm::LLVMContext &getLLVMContext() const {
    return getModule()->getContext();
  }

  virtual void prepareFunction(llvm::Function *F) = 0;

  virtual void setGlobalFunction(bool leaf);
  virtual void setGlobalFunction(llvm::Function *F = nullptr,
                                 bool leaf = false);
  [[nodiscard]] llvm::Function *getGlobalFunction() const {
    return TheFunction;
  }
  [[nodiscard]] virtual llvm::Module *getModule() const = 0;
  [[nodiscard]] virtual llvm::IRBuilder<> *getBuilder() const = 0;
  [[nodiscard]] virtual llvm::Function *getFunction(string funcName) const;

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
  void CodegenMemcpy(llvm::Value *dst, llvm::Value *src, size_t size);
  void CodegenMemcpy(llvm::Value *dst, llvm::Value *src, llvm::Value *size);

  void CodegenMemset(llvm::Value *dst, llvm::Value *byte, size_t size);
  void CodegenMemset(llvm::Value *dst, llvm::Value *bytes, llvm::Value *size);

  virtual void registerFunction(const char *, llvm::Function *);
  virtual llvm::BasicBlock *getEndingBlock() { return codeEnd; }
  virtual void setEndingBlock(llvm::BasicBlock *cdEnd) {
    this->codeEnd = cdEnd;
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

  virtual While gen_while(std::function<ProteusValue()> cond);
  [[nodiscard]] virtual DoWhile gen_do(std::function<void()> whileBody);

  virtual if_branch gen_if(ProteusValue cond);

  virtual if_branch gen_if(const expression_t &expr,
                           const OperatorState &state);

  template <typename T>
  llvm::Type *toLLVM() {
    if constexpr (std::is_void_v<T>) {
      return llvm::Type::getVoidTy(getLLVMContext());
    } else if constexpr (std::is_pointer_v<T>) {
      if constexpr (std::is_void_v<
                        std::remove_cv_t<std::remove_pointer_t<T>>>) {
        // No void ptr type in llvm ir
        return llvm::PointerType::getUnqual(toLLVM<char>());
      } else {
        return llvm::PointerType::getUnqual(
            toLLVM<std::remove_cv_t<std::remove_pointer_t<T>>>());
      }
    } else if constexpr (std::is_integral_v<T>) {
      return llvm::Type::getIntNTy(getLLVMContext(), sizeof(T) * 8);
    } else if constexpr (std::is_same_v<T, double>) {
      return llvm::Type::getDoubleTy(getLLVMContext());
    } else if constexpr (std::is_same_v<T, float>) {
      return llvm::Type::getFloatTy(getLLVMContext());
    } else if constexpr (std::is_same_v<T, std::string>) {
      return CreateStringStruct();
    } else if constexpr (std::is_same_v<T, pb>) {
      auto charPtrType = toLLVM<char *>();
      return llvm::StructType::get(getLLVMContext(),
                                   {charPtrType, charPtrType});
    } else {
      {
        static std::set<decltype(typeid(T).name())> unknown_types;
        static std::mutex m;
        std::lock_guard<std::mutex> lock{m};
        LOG_IF(INFO, unknown_types.emplace(typeid(T).name()).second)
            << "Unknown type " << typeid(T).name()
            << " substituting with sized placeholder";
      }
      return llvm::Type::getIntNTy(getLLVMContext(), sizeof(T) * 8);
    }
  }

  template <typename R, typename... Args>
  llvm::Value *gen_call(R (*func)(Args...),
                        std::initializer_list<llvm::Value *> args,
                        llvm::Type *ret) {
    auto fname = getFunctionName((void *)func);
    return gen_call(fname, args, ret);
  }

  template <typename R, typename... Args>
  llvm::Value *gen_call(R (*func)(Args...),
                        std::initializer_list<llvm::Value *> args) {
    return gen_call(func, args, toLLVM<std::remove_cv_t<R>>());
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

    // Result type specified
    return llvm::StructType::get(ctx, {keyType, bucketSizeType});
  }

  [[nodiscard]] llvm::StructType *getHashtableMetadataType() const {
    return getHashtableMetadataType(getLLVMContext());
  }

  const char *getName();

  virtual StateVar appendStateVar(llvm::Type *ptype, std::string name = "");
  virtual StateVar appendStateVar(llvm::Type *ptype,
                                  std::function<init_func_t> init,
                                  std::function<deinit_func_t> deinit,
                                  std::string name = "");

  virtual llvm::Value *allocateStateVar(llvm::Type *t);
  virtual void deallocateStateVar(llvm::Value *v);

  [[nodiscard]] virtual llvm::Value *getStateVar(const StateVar &id) const;

 protected:
  virtual void prepareStateVars();

  llvm::Module *TheModule;
  llvm::IRBuilder<> *TheBuilder = nullptr;

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

 private:
  std::vector<std::tuple<llvm::Type *, std::string, std::function<init_func_t>,
                         std::function<deinit_func_t>>>
      to_prepare_state_vars;
  std::vector<llvm::Value *> state_vars;
};

struct HashtableBucketMetadata {
  size_t hashKey;
  size_t bucketSize;
};

class save_current_blocks_and_restore_at_exit_scope {
  llvm::BasicBlock *current;
  llvm::BasicBlock *entry;
  llvm::BasicBlock *ending;
  Context *context;

 public:
  explicit save_current_blocks_and_restore_at_exit_scope(Context *context)
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

template <typename Fbody>
void While::operator()(Fbody body) && {
  auto &llvmContext = context->getLLVMContext();
  auto F = context->getBuilder()->GetInsertBlock()->getParent();

  auto CondBB = llvm::BasicBlock::Create(llvmContext, "cond", F);
  auto BodyBB = llvm::BasicBlock::Create(llvmContext, "body", F);
  auto AfterBB = llvm::BasicBlock::Create(llvmContext, "after", F);

  auto Builder = context->getBuilder();
  Builder->CreateBr(CondBB);
  Builder->SetInsertPoint(CondBB);

  auto condition = cond();

  auto loop_cond = Builder->CreateCondBr(condition.value, BodyBB, AfterBB);

  Builder->SetInsertPoint(BodyBB);

  body(loop_cond);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(AfterBB);
}

#endif /* CONTEXT_HPP_ */
