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

#include "util/context.hpp"

#include "expressions/expressions-generator.hpp"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "util/jit/control-flow/if-statement.hpp"

bool print_generated_code = true;

using namespace llvm;

void Context::createJITEngine() {
  LLVMLinkInMCJIT();
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();

  // Create the JIT.  This takes ownership of the module.
  string ErrStr;
  TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(TheModule))
                           .setErrorStr(&ErrStr)
                           .create();
  if (TheExecutionEngine == nullptr) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }
}

void addOptimizerPipelineDefault(legacy::FunctionPassManager *TheFPM) {
  // Provide basic AliasAnalysis support for GVN.
  TheFPM->add(createBasicAAWrapperPass());
  // Promote allocas to registers.
  TheFPM->add(createPromoteMemoryToRegisterPass());
  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createNewGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());
  // Aggressive Dead Code Elimination. Make sure work takes place
  TheFPM->add(createAggressiveDCEPass());
}

#if MODULEPASS
void __attribute__((unused))
addOptimizerPipelineInlining(ModulePassManager *TheMPM) {
  /* Inlining: Not sure it works */
  // LSC: FIXME: No add member to a ModulePassManager
  TheMPM->add(createFunctionInliningPass());
  TheMPM->add(createAlwaysInlinerPass());
}
#endif

void __attribute__((unused))
addOptimizerPipelineVectorization(legacy::FunctionPassManager *TheFPM) {
  /* Vectorization */
  // TheFPM->add(createBBVectorizePass());
  TheFPM->add(createLoopVectorizePass());
  TheFPM->add(createSLPVectorizerPass());
}

const char *Context::getName() { return TheModule->getName().str().c_str(); }

Context::Context(const string &moduleName, bool setGlobFunction)
    : moduleName(moduleName) {
  TheModule = new Module(moduleName, getLLVMContext());
  TheBuilder = new IRBuilder<>(getLLVMContext());

  TheExecutionEngine = nullptr;
  TheFunction = nullptr;
  codeEnd = nullptr;

  /* OPTIMIZER PIPELINE, function passes */
  TheFPM = new legacy::FunctionPassManager(getModule());
  addOptimizerPipelineDefault(TheFPM);

  // LSC: Seems to be faster without the vectorization, at least
  // while running the unit-tests, but this might be because the
  // datasets are too small.
  // addOptimizerPipelineVectorization(TheFPM);

#if MODULEPASS
  /* OPTIMIZER PIPELINE, module passes */
  PassManagerBuilder pmb;
  pmb.OptLevel = 3;
  TheMPM = new ModulePassManager();
  pmb.populateModulePassManager(*TheMPM);
  addOptimizerPipelineInlining(TheMPM);
#endif

  TheFPM->doInitialization();

  if (setGlobFunction) {
    llvm::Type *int_type = Type::getInt32Ty(getLLVMContext());
    vector<Type *> Ints(1, int_type);
    FunctionType *FT =
        FunctionType::get(Type::getInt32Ty(getLLVMContext()), Ints, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, moduleName,
                                   getModule());

    setGlobalFunction(F);  // Be careful, calls from constructor non-virtual!
  }
}

void Context::setGlobalFunction(bool leaf) { setGlobalFunction(nullptr, leaf); }

void Context::setGlobalFunction(Function *F, bool leaf) {
  if (TheFunction) {
    assert(F == nullptr &&
           "Should only be called if global function has not be set.");
    return;
  }
  // Setting the 'global' function
  TheFunction = F;
  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getLLVMContext(), "entry", F);
  getBuilder()->SetInsertPoint(BB);
  setCurrentEntryBlock(BB);

  /**
   * Preparing global info to be maintained
   */
  llvm::Type *int64_type = Type::getInt64Ty(getLLVMContext());
  mem_resultCtr = this->CreateEntryBlockAlloca(F, "resultCtr", int64_type);
  getBuilder()->CreateStore(this->createInt64(0), mem_resultCtr);

  prepareStateVars();

  createJITEngine();
}

void Context::prepareFunction(Function *F) {
  endStateVars();

  // FIXME Have a (tmp) return value for now at this point
  getBuilder()->CreateRet(getBuilder()->getInt32(114));

#ifdef DEBUGCTX
  //    getModule()->dump();
#endif
  // Validate the generated code, checking for consistency.
  verifyFunction(*F);

  // Optimize the function.
  TheFPM->run(*F);
#if MODULEPASS
  TheMPM->runOnModule(getModule());
#endif

  // JIT the function, returning a function pointer.
  TheExecutionEngine->finalizeObject();
  void *FPtr = TheExecutionEngine->getPointerToFunction(F);

  int (*FP)(void) = (int (*)(void))FPtr;
  assert(FP != nullptr && "Code generation failed!");

  // TheModule->dump();
  // Run function
  struct timespec t0, t1;
  clock_gettime(CLOCK_REALTIME, &t0);
  int jitFuncResult = FP();
  // LOG(INFO) << "Mock return value of generated function " << FP(11);
  clock_gettime(CLOCK_REALTIME, &t1);
  printf("(Already compiled) Execution took %f seconds\n", diff(t0, t1));
  cout << "Return flag: " << jitFuncResult << endl;

  TheFPM = nullptr;
  // Dump to see final (optimized) form
#ifdef DEBUGCTX
  //    F->dump();
#endif
}

Function *Context::getFunction(string funcName) const {
  map<string, Function *>::const_iterator it;
  it = availableFunctions.find(funcName);
  if (it == availableFunctions.end()) {
    throw runtime_error(string("Unknown function name: ") + funcName);
  }
  return it->second;
}

// bytes
size_t Context::getSizeOf(llvm::Type *type) const {
  return getModule()->getDataLayout().getTypeAllocSize(type);
}

size_t Context::getSizeOf(llvm::Value *val) const {
  return getSizeOf(val->getType());
}

void Context::CodegenMemcpy(Value *dst, Value *src, int size) {
  LLVMContext &ctx = getLLVMContext();

  CodegenMemcpy(dst, src, createSizeT(size));
}

void Context::CodegenMemcpy(Value *dst, Value *src, Value *size) {
  LLVMContext &ctx = getLLVMContext();
  // Cast src/dst to int8_t*.  If they already are, this will get optimized
  // away
  //  DCHECK(PointerType::classof(dst->getType()));
  //  DCHECK(PointerType::classof(src->getType()));
  Value *false_value_ = ConstantInt::get(ctx, APInt(1, false, true));

  PointerType *ptr_type = PointerType::getInt8PtrTy(ctx);

  dst = getBuilder()->CreateBitCast(dst, ptr_type);
  src = getBuilder()->CreateBitCast(src, ptr_type);

  // Get intrinsic function.
  Function *memcpy_fn = getFunction("memcpy");
  if (memcpy_fn == nullptr) {
    throw runtime_error(string("Could not load memcpy intrinsic"));
  }

  // The fourth argument is the alignment.  For non-zero values, the caller
  // must guarantee that the src and dst values are aligned to that byte
  // boundary.
  // TODO: We should try to take advantage of this since our tuples are well
  // aligned.
  Value *args[] = {
      dst, src, size, false_value_  // is_volatile.
  };
  getBuilder()->CreateCall(memcpy_fn, args);
}

void Context::CodegenMemset(Value *dst, Value *byte, int size) {
  LLVMContext &ctx = getLLVMContext();

  CodegenMemset(dst, byte, createSizeT(size));
}

void Context::CodegenMemset(Value *dst, Value *bytes, Value *size) {
  LLVMContext &ctx = getLLVMContext();
  // Cast src/dst to int8_t*.  If they already are, this will get optimized
  // away
  //  DCHECK(PointerType::classof(dst->getType()));
  //  DCHECK(PointerType::classof(src->getType()));
  PointerType *ptr_type = PointerType::getInt8PtrTy(ctx);

  // Value *zero = ConstantInt::get(ctx, APInt(32, 0));

  dst = getBuilder()->CreateBitCast(dst, ptr_type);

  // Get intrinsic function.
  Function *memset_fn = getFunction("memset");
  if (memset_fn == nullptr) {
    throw runtime_error(string("Could not load memset intrinsic"));
  }

  // Type *int32_type = IntegerType::getInt32Ty(ctx);
  Value *byte = getBuilder()->CreateZExtOrTrunc(
      bytes, memset_fn->getFunctionType()->params()[1]);

  // The fourth argument is the alignment.  For non-zero values, the caller
  // must guarantee that the src and dst values are aligned to that byte
  // boundary.
  // TODO: We should try to take advantage of this since our tuples are well
  // aligned.
  std::vector<Value *> args = {dst, byte, size};
  if (memset_fn->getFunctionType()->getNumParams() == 4) {
    // FIXME: for now assume CPU side if 4 attributes, in which case the
    // fourth argume is `isvolatiles`
    args.push_back(createFalse());
  }
  getBuilder()->CreateCall(memset_fn, args);
}

ConstantInt *Context::createInt8(char val) {
  return ConstantInt::get(getLLVMContext(), APInt(8, val));
}

ConstantInt *Context::createInt32(int val) {
  return ConstantInt::get(getLLVMContext(), APInt(32, val));
}

ConstantInt *Context::createInt64(int val) {
  return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt *Context::createInt64(unsigned int val) {
  return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt *Context::createInt64(size_t val) {
  return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt *Context::createInt64(int64_t val) {
  return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt *Context::createSizeT(size_t val) {
  return ConstantInt::get(createSizeType(), val);
}

IntegerType *Context::createSizeType() {
  return Type::getIntNTy(getLLVMContext(), sizeof(size_t) * 8);
}

ConstantInt *Context::createTrue() {
  return ConstantInt::get(getLLVMContext(), APInt(1, 1));
}

ConstantInt *Context::createFalse() {
  return ConstantInt::get(getLLVMContext(), APInt(1, 0));
}

Value *Context::CastPtrToLlvmPtr(PointerType *type, const void *ptr) {
  Constant *const_int = createInt64((uint64_t)ptr);
  Value *llvmPtr = ConstantExpr::getIntToPtr(const_int, type);
  return llvmPtr;
}

Value *Context::getArrayElem(AllocaInst *mem_ptr, Value *offset) {
  Value *val_ptr = getBuilder()->CreateLoad(mem_ptr, "mem_ptr");
  Value *shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
  Value *val_shifted = getBuilder()->CreateLoad(shiftedPtr, "val_shifted");
  return val_shifted;
}

Value *Context::getArrayElem(Value *val_ptr, Value *offset) {
  Value *shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
  Value *val_shifted = getBuilder()->CreateLoad(shiftedPtr, "val_shifted");
  return val_shifted;
}

Value *Context::getArrayElemMem(Value *val_ptr, Value *offset) {
  Value *shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
  return shiftedPtr;
}

Value *Context::getStructElem(Value *mem_struct, int elemNo) {
  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(elemNo));
  // Shift in struct ptr
  Value *mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
  Value *val_struct_shifted = getBuilder()->CreateLoad(mem_struct_shifted);
  return val_struct_shifted;
}

Value *Context::getStructElemMem(Value *mem_struct, int elemNo) {
  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(elemNo));
  // Shift in struct ptr
  Value *mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
  return mem_struct_shifted;
}

Value *Context::getStructElem(AllocaInst *mem_struct, int elemNo) {
  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(elemNo));
  // Shift in struct ptr
  Value *mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
  Value *val_struct_shifted = getBuilder()->CreateLoad(mem_struct_shifted);
  return val_struct_shifted;
}

void Context::updateStructElem(Value *toStore, Value *mem_struct, int elemNo) {
  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(elemNo));
  // Shift in struct ptr
  Value *structPtr = getBuilder()->CreateGEP(mem_struct, idxList);
  getBuilder()->CreateStore(toStore, structPtr);
}

void Context::CreateForLoop(const string &cond, const string &body,
                            const string &inc, const string &end,
                            BasicBlock **cond_block, BasicBlock **body_block,
                            BasicBlock **inc_block, BasicBlock **end_block,
                            BasicBlock *insert_before) {
  Function *fn = TheFunction;
  LLVMContext &ctx = getLLVMContext();
  *cond_block = BasicBlock::Create(ctx, string(cond), fn, insert_before);
  *body_block = BasicBlock::Create(ctx, string(body), fn, insert_before);
  *inc_block = BasicBlock::Create(ctx, string(inc), fn, insert_before);
  *end_block = BasicBlock::Create(ctx, string(end), fn, insert_before);
}

void Context::CreateIfElseBlocks(Function *fn, const string &if_label,
                                 const string &else_label,
                                 BasicBlock **if_block, BasicBlock **else_block,
                                 BasicBlock *insert_before) {
  LLVMContext &ctx = getLLVMContext();
  *if_block = BasicBlock::Create(ctx, if_label, fn, insert_before);
  *else_block = BasicBlock::Create(ctx, else_label, fn, insert_before);
}

BasicBlock *Context::CreateIfBlock(Function *fn, const string &if_label,
                                   BasicBlock *insert_before) {
  return BasicBlock::Create(getLLVMContext(), if_label, fn, insert_before);
}

void Context::CreateIfBlock(Function *fn, const string &if_label,
                            BasicBlock **if_block, BasicBlock *insert_before) {
  *if_block = CreateIfBlock(fn, if_label, insert_before);
}

AllocaInst *Context::CreateEntryBlockAlloca(Function *TheFunction,
                                            const string &VarName,
                                            Type *varType, Value *arraySize) {
  IRBuilder<> TmpBuilder(&TheFunction->getEntryBlock(),
                         TheFunction->getEntryBlock().begin());
  return TmpBuilder.CreateAlloca(varType, arraySize, VarName.c_str());
}

AllocaInst *Context::CreateEntryBlockAlloca(const string &VarName,
                                            Type *varType, Value *arraySize) {
  Function *F = getBuilder()->GetInsertBlock()->getParent();
  return CreateEntryBlockAlloca(F, VarName, varType, arraySize);
}

AllocaInst *Context::createAlloca(BasicBlock *InsertAtBB, const string &VarName,
                                  Type *varType) {
  IRBuilder<> TmpBuilder(InsertAtBB, InsertAtBB->begin());
  return TmpBuilder.CreateAlloca(varType, nullptr, VarName.c_str());
}

Value *Context::CreateGlobalString(char *str) {
  LLVMContext &ctx = getLLVMContext();
  ArrayType *ArrayTy_0 =
      ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

  GlobalVariable *gvar_array__str = new GlobalVariable(
      *getModule(),
      /*Type=*/ArrayTy_0,
      /*isConstant=*/true,
      /*Linkage=*/GlobalValue::PrivateLinkage,
      /*Initializer=*/nullptr,  // has initializer, specified below
      /*Name=*/".str");

  Constant *tmpHTname = ConstantDataArray::getString(ctx, str, true);
  PointerType *charPtrType = PointerType::get(IntegerType::get(ctx, 8), 0);
  AllocaInst *AllocaName =
      CreateEntryBlockAlloca(string("globalStr"), charPtrType);

  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(0));
  Constant *shifted =
      ConstantExpr::getGetElementPtr(ArrayTy_0, gvar_array__str, idxList);
  gvar_array__str->setInitializer(tmpHTname);

  getBuilder()->CreateStore(shifted, AllocaName);
  Value *globalStr = getBuilder()->CreateLoad(AllocaName);
  return globalStr;
}

Value *Context::CreateGlobalString(const char *str) {
  LLVMContext &ctx = getLLVMContext();
  ArrayType *ArrayTy_0 =
      ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

  GlobalVariable *gvar_array__str = new GlobalVariable(
      *getModule(),
      /*Type=*/ArrayTy_0,
      /*isConstant=*/true,
      /*Linkage=*/GlobalValue::PrivateLinkage,
      /*Initializer=*/nullptr,  // has initializer, specified below
      /*Name=*/".str");

  Constant *tmpHTname = ConstantDataArray::getString(ctx, str, true);
  PointerType *charPtrType = PointerType::get(IntegerType::get(ctx, 8), 0);
  AllocaInst *AllocaName =
      CreateEntryBlockAlloca(string("globalStr"), charPtrType);

  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(createInt32(0));
  idxList.push_back(createInt32(0));
  Constant *shifted =
      ConstantExpr::getGetElementPtr(ArrayTy_0, gvar_array__str, idxList);
  gvar_array__str->setInitializer(tmpHTname);

  getBuilder()->CreateStore(shifted, AllocaName);
  Value *globalStr = getBuilder()->CreateLoad(AllocaName);
  return globalStr;
}

PointerType *Context::getPointerType(Type *type) {
  return PointerType::get(type, 0);
}

StructType *Context::CreateCustomStruct(vector<Type *> innerTypes) {
  llvm::StructType *valueType =
      llvm::StructType::get(getLLVMContext(), innerTypes);
  return valueType;
}

StructType *Context::ReproduceCustomStruct(list<typeID> innerTypes) {
  LLVMContext &ctx = getLLVMContext();
  vector<Type *> llvmTypes;
  list<typeID>::iterator it;
  for (it = innerTypes.begin(); it != innerTypes.end(); it++) {
    switch (*it) {
      case INT: {
        Type *int32_type = Type::getInt32Ty(ctx);
        llvmTypes.push_back(int32_type);
        break;
      }
      case BOOL: {
        Type *int1_type = Type::getInt1Ty(ctx);
        llvmTypes.push_back(int1_type);
        break;
      }
      case FLOAT: {
        Type *float_type = Type::getDoubleTy(ctx);
        llvmTypes.push_back(float_type);
        break;
      }
      case INT64: {
        Type *int64_type = Type::getInt64Ty(ctx);
        llvmTypes.push_back(int64_type);
        break;
      }
      case STRING:
      case RECORD:
      case LIST:
      case BAG:
      case SET:
      case COMPOSITE:
      default: {
        string error_msg = "No explicit caching support for this type yet";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
    }
  }
  llvm::StructType *valueType = llvm::StructType::get(ctx, llvmTypes);
  return valueType;
}

StructType *Context::CreateJSONPosStruct() {
  llvm::Type *int64_type = Type::getInt64Ty(getLLVMContext());
  vector<Type *> json_pos_types;
  json_pos_types.push_back(int64_type);
  json_pos_types.push_back(int64_type);
  return CreateCustomStruct(json_pos_types);
}

PointerType *Context::CreateJSMNStructPtr() {
  Type *jsmnStructType = CreateJSMNStruct();
  PointerType *ptr_jsmnStructType = PointerType::get(jsmnStructType, 0);
  return ptr_jsmnStructType;
}

StructType *Context::CreateJSMNStruct() {
  LLVMContext &ctx = getLLVMContext();
  vector<Type *> jsmn_pos_types;
#ifndef JSON_TIGHT
  llvm::Type *int32_type = Type::getInt32Ty(ctx);
  jsmn_pos_types.push_back(int32_type);
  jsmn_pos_types.push_back(int32_type);
  jsmn_pos_types.push_back(int32_type);
  jsmn_pos_types.push_back(int32_type);
#endif
#ifdef JSON_TIGHT
  llvm::Type *int16_type = Type::getInt16Ty(ctx);
  llvm::Type *int8_type = Type::getInt8Ty(ctx);
  jsmn_pos_types.push_back(int8_type);
  jsmn_pos_types.push_back(int16_type);
  jsmn_pos_types.push_back(int16_type);
  jsmn_pos_types.push_back(int8_type);
#endif
  return CreateCustomStruct(jsmn_pos_types);
}

StructType *Context::CreateStringStruct() {
  LLVMContext &ctx = getLLVMContext();
  llvm::Type *int32_type = Type::getInt32Ty(ctx);
  llvm::Type *char_type = Type::getInt8Ty(ctx);
  PointerType *ptr_char_type = PointerType::get(char_type, 0);
  vector<Type *> string_obj_types;
  string_obj_types.push_back(ptr_char_type);
  string_obj_types.push_back(int32_type);

  return CreateCustomStruct(string_obj_types);
}

// Provide support for some extern functions
void Context::registerFunction(const char *funcName, Function *func) {
  availableFunctions[funcName] = func;
}

StateVar Context::appendStateVar(llvm::Type *ptype, std::string name) {
  return appendStateVar(
      ptype, [ptype](llvm::Value *) { return UndefValue::get(ptype); },
      [](llvm::Value *, llvm::Value *) {}, name);
}

llvm::Value *Context::getStateVar(const StateVar &id) const {
  assert(state_vars.size() > id.getIndex() &&
         "Has the function been created? Is it a valid ID?");
  return state_vars[id.getIndex()];
}

StateVar Context::appendStateVar(llvm::Type *ptype,
                                 std::function<init_func_t> init,
                                 std::function<deinit_func_t> deinit,
                                 std::string name) {
  size_t id = to_prepare_state_vars.size();

  if (getGlobalFunction()) {
    // FIXME: deprecated path...  remove
    Value *pip = nullptr;  // FIXME should introduce a pipeline ptr

    // save current block
    BasicBlock *currBlock = getBuilder()->GetInsertBlock();
    // go to entry block
    getBuilder()->SetInsertPoint(getCurrentEntryBlock());

    Value *var = init(pip);
    var->setName(name);

    // save new entry block
    setCurrentEntryBlock(getBuilder()->GetInsertBlock());

    // restore insert point
    getBuilder()->SetInsertPoint(currBlock);

    state_vars.emplace_back(var);
  }

  to_prepare_state_vars.emplace_back(
      ptype, name + "_statevar" + std::to_string(id), init, deinit);

  return {id, nullptr};
}

llvm::Value *Context::allocateStateVar(llvm::Type *t) {
  return CreateEntryBlockAlloca("", t);
}

void Context::deallocateStateVar(llvm::Value *t) {}

void Context::prepareStateVars() {
  assert(state_vars.size() == 0);
  // //save current block
  // BasicBlock *currBlock = getBuilder()->GetInsertBlock();
  // //go to entry block
  // getBuilder()->SetInsertPoint(getCurrentEntryBlock());

  // //save new entry block
  // setCurrentEntryBlock(getBuilder()->GetInsertBlock());
  // //restore insert point
  // getBuilder()->SetInsertPoint(currBlock);
  for (size_t i = 0; i < to_prepare_state_vars.size(); ++i) {
    Value *var = nullptr;  // FIXME should introduce a pipeline ptr
    var = std::get<2>(to_prepare_state_vars[i])(var);
    var->setName(std::get<1>(to_prepare_state_vars[i]));
    state_vars.emplace_back(var);
  }
  // size_t id = state_vars.size();

  // AllocaInst * var     = CreateEntryBlockAlloca(
  //                         getBuilder()->GetInsertBlock()->getParent(),
  //                         name + std::to_string(id),
  //                         ptype
  //                     );
  // state_vars.emplace_back(var);
  // return id;

  // size_t id = appendStateVar(ptype, name);

  // //save current block
  // BasicBlock *currBlock = getBuilder()->GetInsertBlock();
  // //go to entry block
  // getBuilder()->SetInsertPoint(getCurrentEntryBlock());

  // init(getStateVar(id));

  // //save new entry block
  // setCurrentEntryBlock(getBuilder()->GetInsertBlock());
  // //restore insert point
  // getBuilder()->SetInsertPoint(currBlock);
  // return id;
}

void Context::endStateVars() {
  // //save current block
  // BasicBlock *currBlock = getBuilder()->GetInsertBlock();
  // //go to entry block
  // getBuilder()->SetInsertPoint(getCurrentEntryBlock());

  // //save new entry block
  // setCurrentEntryBlock(getBuilder()->GetInsertBlock());
  // //restore insert point
  // getBuilder()->SetInsertPoint(currBlock);
  for (size_t i = 0; i < to_prepare_state_vars.size(); ++i) {
    Value *pip = nullptr;  // FIXME should introduce a pipeline ptr
    std::get<3>(to_prepare_state_vars[i])(pip, state_vars[i]);
  }
  // size_t id = state_vars.size();

  // AllocaInst * var     = CreateEntryBlockAlloca(
  //                         getBuilder()->GetInsertBlock()->getParent(),
  //                         name + std::to_string(id),
  //                         ptype
  //                     );
  // state_vars.emplace_back(var);
  // return id;

  // size_t id = appendStateVar(ptype, name);

  // //save current block
  // BasicBlock *currBlock = getBuilder()->GetInsertBlock();
  // //go to entry block
  // getBuilder()->SetInsertPoint(getCurrentEntryBlock());

  // init(getStateVar(id));

  // //save new entry block
  // setCurrentEntryBlock(getBuilder()->GetInsertBlock());
  // //restore insert point
  // getBuilder()->SetInsertPoint(currBlock);
  // return id;
}

llvm::Value *Context::gen_call(std::string func,
                               std::initializer_list<llvm::Value *> args,
                               llvm::Type *ret) {
  llvm::Function *f;
  try {
    f = getFunction(func);
    assert(ret == f->getReturnType());
  } catch (std::runtime_error &) {
    std::vector<llvm::Type *> v;
    v.reserve(args.size());
    for (const auto &arg : args) v.emplace_back(arg->getType());
    auto FTfunc = llvm::FunctionType::get(ret, v, false);

    f = llvm::Function::Create(FTfunc, llvm::Function::ExternalLinkage, func,
                               getModule());

    registerFunction((new std::string{func})->c_str(), f);
  }

  return getBuilder()->CreateCall(f, args);
}

if_branch Context::gen_if(ProteusValue cond) { return if_branch(cond, this); }

if_branch Context::gen_if(const expression_t &expr,
                          const OperatorState &state) {
  return if_branch(expr, state, this);
}
