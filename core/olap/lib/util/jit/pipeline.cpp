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

#include "pipeline.hpp"

#include <llvm/InitializePasses.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>

#include <olap/util/parallel-context.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/util/timing.hpp>
#include <thread>

using namespace llvm;

size_t PipelineGen::appendParameter(llvm::Type *ptype, bool noalias,
                                    bool readonly) {
  inputs.push_back(ptype);
  inputs_noalias.push_back(noalias);
  inputs_readonly.push_back(readonly);

  return inputs.size() - 1;
}

StateVar PipelineGen::appendStateVar(llvm::Type *ptype) {
  return appendStateVar(
      ptype, [ptype](Value *) { return UndefValue::get(ptype); },
      [](Value *, Value *) {});
}

const int64_t *getSession(Pipeline *p) {
  return static_cast<const int64_t *>(p->getSession());
}

StateVar PipelineGen::appendStateVar(llvm::Type *ptype,
                                     std::function<init_func_t> init,
                                     std::function<deinit_func_t> deinit) {
  state_vars.push_back(ptype);
  size_t var_id = state_vars.size() - 1;
  open_var.emplace_back(init, var_id);
  close_var.emplace_back(deinit, var_id);
  return {var_id, this};
}

void PipelineGen::registerOpen(const void *owner,
                               std::function<void(Pipeline *pip)> open) {
  openers.emplace_back(owner, open);
  size_t indx = openers.size() - 1;
  open_var.emplace_back(
      [=](Value *pip) {
        Function *f = context->getFunction("callPipRegisteredOpen");
        PointerType *charPtrType =
            Type::getInt8PtrTy(context->getLLVMContext());
        Value *this_ptr = context->CastPtrToLlvmPtr(charPtrType, this);
        Value *this_opener = context->createSizeT(indx);
        getBuilder()->CreateCall(
            f, std::vector<Value *>{this_ptr, this_opener, pip});
        return (Value *)nullptr;
      },
      ~((size_t)0));
}

void PipelineGen::registerClose(const void *owner,
                                std::function<void(Pipeline *pip)> close) {
  closers.emplace_back(owner, close);
  size_t indx = closers.size() - 1;
  close_var.emplace_back(
      [=](Value *pip, Value *) {
        Function *f = context->getFunction("callPipRegisteredClose");
        PointerType *charPtrType =
            Type::getInt8PtrTy(context->getLLVMContext());
        Value *this_ptr = context->CastPtrToLlvmPtr(charPtrType, this);
        Value *this_closer = context->createSizeT(indx);
        getBuilder()->CreateCall(
            f, std::vector<Value *>{this_ptr, this_closer, pip});
      },
      ~((size_t)0));
}

void PipelineGen::callPipRegisteredOpen(size_t indx, Pipeline *pip) {
  (openers[indx].second)(pip);
}

void PipelineGen::callPipRegisteredClose(size_t indx, Pipeline *pip) {
  (closers[indx].second)(pip);
}

extern "C" {
void callPipRegisteredOpen(PipelineGen *pipgen, size_t indx, Pipeline *pip) {
  pipgen->callPipRegisteredOpen(indx, pip);
}

void callPipRegisteredClose(PipelineGen *pipgen, size_t indx, Pipeline *pip) {
  pipgen->callPipRegisteredClose(indx, pip);
}
}

std::vector<llvm::Type *> PipelineGen::getStateVars() const {
  return state_vars;
}

extern "C" {
void *allocate_pinned(size_t bytes) {
  return MemoryManager::mallocPinned(bytes);  // FIXME: create releases
}
void *allocate_gpu(size_t bytes) {
  return MemoryManager::mallocGpu(bytes);  // FIXME: create releases
}
void deallocate_pinned(void *x) {
  return MemoryManager::freePinned(x);  // FIXME: create releases
}
void deallocate_gpu(void *x) {
  return MemoryManager::freeGpu(x);  // FIXME: create releases
}
}

llvm::Value *PipelineGen::allocateStateVar(llvm::Type *t) {
  Function *alloc = getFunction("allocate");
  ConstantInt *s = context->createSizeT(context->getSizeOf(t));

  Value *ptr = getBuilder()->CreateCall(alloc, s);
  return getBuilder()->CreateBitCast(ptr, PointerType::getUnqual(t));
}

void PipelineGen::deallocateStateVar(llvm::Value *p) {
  Function *dealloc = getFunction("deallocate");
  Type *charPtrType = Type::getInt8PtrTy(context->getLLVMContext());
  Value *ptr = getBuilder()->CreateBitCast(p, charPtrType);

  getBuilder()->CreateCall(dealloc, ptr);
}

Argument *PipelineGen::getArgument(size_t id) const {
  assert(id < args.size());
  return args[id];
}

Value *PipelineGen::getStateVar() const {
  assert(state);
  Function *Fcurrent = getBuilder()->GetInsertBlock()->getParent();
  if (Fcurrent == close_function) return state;
  if (Fcurrent == open__function) return state;
  if (Fcurrent != F) {
    return context->getBuilder()->CreateLoad(
        (Fcurrent->arg_end() - 1)->getType()->getPointerElementType(),
        Fcurrent->arg_end() - 1);
  }
  return state;  // getArgument(args.size() - 1);
}

Value *PipelineGen::getStateVar(StateVar id) const {
  Value *arg = getStateVar();
  assert(id.pip == this);
  assert(id.getIndex() < state_vars.size());
  return context->getBuilder()->CreateExtractValue(arg, id.getIndex());
}

Value *PipelineGen::getStateVarPtr() const {
  assert(F == getBuilder()->GetInsertBlock()->getParent() &&
         "Only supported in main function");
  return args.back();
}

Value *PipelineGen::getSubStateVar() const {
  assert(copyStateFrom);
  Value *subState = getStateVar(StateVar{0, this});
  subState->setName("subState");
  return subState;
}

Function *const PipelineGen::createHelperFunction(string funcName,
                                                  std::vector<Type *> ins,
                                                  std::vector<bool> readonly,
                                                  std::vector<bool> noalias) {
  assert(readonly.size() == noalias.size());
  assert(readonly.size() == 0 || readonly.size() == ins.size());

  ins.push_back(PointerType::getUnqual(state_type));

  FunctionType *ftype =
      FunctionType::get(Type::getVoidTy(context->getLLVMContext()), ins, false);
  // use f_num to overcome an llvm bu with keeping dots in function names when
  // generating PTX (which is invalid in PTX)
  Function *helper =
      Function::Create(ftype, Function::ExternalLinkage,
                       pipName + "_" + funcName, context->getModule());

  if (readonly.size() == ins.size()) {
    Attribute readOnly = Attribute::get(context->getLLVMContext(),
                                        Attribute::AttrKind::ReadOnly);
    Attribute noAlias =
        Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias);

    readonly.push_back(true);
    noalias.push_back(true);
    std::vector<std::pair<unsigned, Attribute>> attrs;
    for (size_t i = 1; i <= ins.size();
         ++i) {  //+1 because 0 is the return value
      if (readonly[i - 1]) attrs.emplace_back(i, readOnly);
      if (noalias[i - 1]) attrs.emplace_back(i, noAlias);
    }

    helper->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
  }

  BasicBlock *BB =
      BasicBlock::Create(context->getLLVMContext(), "entry", helper);
  getBuilder()->SetInsertPoint(BB);

  return helper;
}

Value *PipelineGen::invokeHelperFunction(Function *f,
                                         std::vector<Value *> args) const {
  args.push_back(getStateVar());
  return getBuilder()->CreateCall(F, args);
}

extern "C" {
void yield() { std::this_thread::yield(); }
}

void PipelineGen::init() {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeCoroutines(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeScalarizeMaskedMemIntrinLegacyPassPass(Registry);
  initializeCodeGenPreparePass(Registry);
  initializeAtomicExpandPass(Registry);
  initializeRewriteSymbolsLegacyPassPass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeWasmEHPreparePass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeSjLjEHPreparePass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeGlobalMergePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeEntryExitInstrumenterPass(Registry);
  initializePostInlineEntryExitInstrumenterPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeExpandReductionsPass(Registry);
}

PipelineGen::PipelineGen(Context *context, std::string pipName,
                         PipelineGen *copyStateFrom)
    : F(nullptr),
      pipName(pipName),
      context(context),
      copyStateFrom(copyStateFrom),
      execute_after_close(nullptr),
      TheBuilder(nullptr) {
  maxBlockSize = 1;
  maxGridSize = 1;

  state = nullptr;

  // ThePM = new legacy::PassManager();
  // {
  //     auto &LTM = static_cast<LLVMTargetMachine &>(*(((ParallelContext *)
  //     context)->TheTargetMachine)); Pass *TPC = (Pass *)
  //     LTM.createPassConfig(*ThePM); ThePM->add(TPC);
  // }
}

void PipelineGen::registerSubPipeline() {
  if (copyStateFrom) {
    FunctionType *f_t = copyStateFrom->getLLVMConsume()->getFunctionType();
    Function *f = Function::Create(f_t, Function::ExternalLinkage,
                                   copyStateFrom->getLLVMConsume()->getName(),
                                   getModule());

    //    sys::DynamicLibrary::AddSymbol(
    //        copyStateFrom->getLLVMConsume()->getName(),
    //        copyStateFrom
    //            ->getConsume());  // FIMXE: this can be a little bit more
    //            elegant...
    //                              // alos it may create name conflicts...
    registerFunction("subpipeline_consume", f);
  }
}

void PipelineGen::registerFunction(std::string funcName, Function *func) {
  availableFunctions[funcName] = func;
}

llvm::Function *PipelineGen::getFunction(std::string funcName) const {
  map<string, Function *>::const_iterator it;
  it = availableFunctions.find(funcName);
  if (it == availableFunctions.end()) {
    auto str = std::string("Unknown function name: ") + funcName + " (" +
               pipName + ")\nAvailable functions: \n";
    for (auto &t : availableFunctions) str += t.first + '\n';
    throw std::runtime_error(str);
  }
  return it->second;
}

size_t PipelineGen::prepareStateArgument() {
  LLVMContext &TheContext = context->getLLVMContext();

  Type *int32Type = Type::getInt32Ty(TheContext);

  if (state_vars.empty())
    appendStateVar(int32Type);  // FIMXE: should not be necessary... there
                                // should be some way to bypass it...

  state_type = StructType::create(state_vars, pipName + "_state_t");
  size_t state_id =
      appendParameter(PointerType::get(state_type, 0), true, true);  // true);
  state_size = getDataLayout().getTypeAllocSize(state_type);

  return state_id;
}

void PipelineGen::prepareInitDeinit() {
  // create open and close functions
  BasicBlock *BB = getBuilder()->GetInsertBlock();

  Type *charPtrType = Type::getInt8PtrTy(context->getLLVMContext());
  std::vector<Type *> ins{charPtrType, PointerType::getUnqual(state_type)};
  FunctionType *ftype =
      FunctionType::get(Type::getVoidTy(context->getLLVMContext()), ins, false);

  Attribute noAlias =
      Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias);

  std::vector<std::pair<unsigned, Attribute>> attrs;
  for (size_t i = 1; i <= 2; ++i) {  //+1 because 0 is the return value
    if (inputs_noalias[i - 1]) attrs.emplace_back(i, noAlias);
  }

  size_t s = 0;
  {
    open__function = Function::Create(ftype, Function::ExternalLinkage,
                                      pipName + "_open", context->getModule());
    open__function->setAttributes(
        AttributeList::get(context->getLLVMContext(), attrs));
    std::vector<Argument *> args;
    for (auto &t : open__function->args()) args.push_back(&t);

    BasicBlock *openBB =
        BasicBlock::Create(context->getLLVMContext(), "entry", open__function);

    getBuilder()->SetInsertPoint(openBB);
    Value *state = UndefValue::get(state_type);

    for (size_t i = 0; i < open_var.size(); ++i) {
      Value *var = (open_var[i].first)(args[0]);
      if (open_var[i].second != ~((size_t)0)) {
        state = getBuilder()->CreateInsertValue(state, var, s++);
      }
    }

    getBuilder()->CreateStore(state, args[1]);
    getBuilder()->CreateRetVoid();
  }
  {
    close_function = Function::Create(ftype, Function::ExternalLinkage,
                                      pipName + "_close", context->getModule());
    close_function->setAttributes(
        AttributeList::get(context->getLLVMContext(), attrs));
    std::vector<Argument *> args;
    for (auto &t : close_function->args()) args.push_back(&t);

    BasicBlock *closeBB =
        BasicBlock::Create(context->getLLVMContext(), "entry", close_function);

    getBuilder()->SetInsertPoint(closeBB);
    Value *tmp = state;
    state = getBuilder()->CreateLoad(
        args[1]->getType()->getPointerElementType(), args[1]);

    for (size_t i = close_var.size(); i > 0; --i) {
      Value *var = nullptr;
      if (close_var[i - 1].second != ~((size_t)0)) {
        var = getBuilder()->CreateExtractValue(state, --s);
      }
      (close_var[i - 1].first)(args[0], var);
    }

    getBuilder()->CreateRetVoid();

    state = tmp;
  }
  assert(s == 0);

  getBuilder()->SetInsertPoint(BB);
}

void PipelineGen::prepareFunction() {
  FunctionType *ftype = FunctionType::get(
      Type::getVoidTy(getModule()->getContext()), inputs, false);
  // use f_num to overcome an llvm bu with keeping dots in function names when
  // generating PTX (which is invalid in PTX)
  F = Function::Create(ftype, Function::ExternalLinkage, pipName, getModule());

  Attribute readOnly =
      Attribute::get(getModule()->getContext(), Attribute::AttrKind::ReadOnly);
  Attribute noAlias =
      Attribute::get(getModule()->getContext(), Attribute::AttrKind::NoAlias);

  std::vector<std::pair<unsigned, Attribute>> attrs;
  for (size_t i = 1; i <= inputs.size();
       ++i) {  //+1 because 0 is the return value
    if (inputs_readonly[i - 1]) attrs.emplace_back(i, readOnly);
    if (inputs_noalias[i - 1]) attrs.emplace_back(i, noAlias);
  }

  F->setAttributes(AttributeList::get(getModule()->getContext(), attrs));
  for (auto &t : F->args()) args.push_back(&t);

  TheBuilder = new IRBuilder<>(getModule()->getContext());
  BasicBlock *BB = BasicBlock::Create(getModule()->getContext(), "entry", F);
  getBuilder()->SetInsertPoint(BB);
}

Value *PipelineGen::getStateLLVMValue() {
  return getBuilder()->CreateLoad(
      getArgument(args.size() - 1)->getType()->getPointerElementType(),
      getArgument(args.size() - 1));
}

llvm::Value *PipelineGen::getSessionParametersPtr() const {
  assert(session_parameters_ptr != StateVar{});

  return getStateVar(session_parameters_ptr);
}

Function *PipelineGen::prepare() {
  assert(!F);
  session_parameters_ptr = appendStateVar(
      IntegerType::getInt64PtrTy(context->getLLVMContext()),
      [=](llvm::Value *pip) { return context->gen_call(getSession, {pip}); },
      [](auto, auto) {});

  size_t state_id = prepareStateArgument();

  prepareFunction();

  // Get the ENTRY BLOCK
  // context->setCurrentEntryBlock(F->getEntryBlock());

  Argument *stateArg = getArgument(state_id);
  stateArg->setName("state_ptr");

  state = getStateLLVMValue();
  state->setName("state");

  prepareInitDeinit();

  return F;
}

Function *PipelineGen::getFunction() const {
  assert(F);
  return F;
}

void *PipelineGen::getKernel() {
  time_block t(TimeRegistry::Key{
      "Compile and Load (CPU, waiting - critical - getKernel)"});
  auto ret = func.get();
  assert(ret != nullptr);
  // assert(!F);
  return (void *)ret;
}

std::unique_ptr<Pipeline> PipelineGen::getPipeline(int group_id) {
  auto ssize = state_size;
  void *func = getKernel();

  std::vector<std::pair<const void *, std::function<opener_t>>>
      openers{};  // this->openers};
  std::vector<std::pair<const void *, std::function<closer_t>>>
      closers{};  // this->closers};

  if (copyStateFrom) {
    std::shared_ptr<Pipeline> copyFrom = copyStateFrom->getPipeline(group_id);

    openers.insert(openers.begin(),
                   std::make_pair(this, [copyFrom, this](Pipeline *pip) {
                     copyFrom->open(pip->getSession());
                     pip->setStateVar({0, this}, copyFrom->state);
                   }));
    // closers.emplace_back([copyFrom](Pipeline *
    // pip){pip->copyStateBackTo(copyFrom);});
    closers.insert(
        closers.begin(),
        std::make_pair(this, [copyFrom](Pipeline *pip) { copyFrom->close(); }));
  } else {
    openers.insert(openers.begin(), std::make_pair(this, [](Pipeline *pip) {}));
    // closers.emplace_back([copyFrom](Pipeline *
    // pip){pip->copyStateBackTo(copyFrom);});
    closers.insert(closers.begin(), std::make_pair(this, [](Pipeline *pip) {}));
  }
  return Pipeline::create(func, ssize, this, state_type, openers, closers,
                          getCompiledFunction(open__function),
                          getCompiledFunction(close_function), group_id,
                          execute_after_close
                              ? execute_after_close->getPipeline(group_id)
                              : nullptr);
}

llvm::Value *PipelineGen::workerScopedAtomicAdd(llvm::Value *ptr,
                                                llvm::Value *inc) {
  IRBuilder<> *Builder = context->getBuilder();
  Value *old =
      Builder->CreateLoad(ptr->getType()->getPointerElementType(), ptr);
  Builder->CreateStore(Builder->CreateAdd(old, inc), ptr);
  return old;
}

llvm::Value *PipelineGen::workerScopedAtomicXchg(llvm::Value *ptr,
                                                 llvm::Value *val) {
  IRBuilder<> *Builder = context->getBuilder();
  Value *old =
      Builder->CreateLoad(ptr->getType()->getPointerElementType(), ptr);
  Builder->CreateStore(val, ptr);
  return old;
}

void PipelineGen::workerScopedMembar() {
  // No-op for single-threaded workers
}

Pipeline::Pipeline(
    guard, void *f, size_t state_size, PipelineGen *gen, StructType *state_type,
    const std::vector<std::pair<const void *, std::function<opener_t>>>
        &openers,
    const std::vector<std::pair<const void *, std::function<closer_t>>>
        &closers,
    void *init_state, void *deinit_state, int32_t group_id,
    std::shared_ptr<Pipeline> execute_after_close)
    : cons(f),
      state_type(state_type),
      openers(openers),
      closers(closers),
      group_id(group_id),
      layout(gen->getDataLayout()),
      state_size(state_size),
      init_state(init_state),
      deinit_state(deinit_state),
      execute_after_close(std::move(execute_after_close)) {
  assert(!openers.empty() && "Openers should be non-empty");
  assert(!closers.empty() && "Closers should be non-empty");
  assert(openers.size() == 1 && "Openers should contain a single element");
  assert(closers.size() == 1 && "Closers should contain a single element");

  state = MemoryManager::mallocPinned(
      state_size);  //(getModule()->getDataLayout().getTypeSizeInBits(state_type)
                    //+ 7) / 8);
  assert(state);
}

// GpuRawPipeline::GpuRawPipeline(void * f, size_t state_size, PipelineGen *
// gen, StructType * state_type,
//         const std::vector<std::function<void (Pipeline * pip)>> &openers,
//         const std::vector<std::function<void (Pipeline * pip)>> &closers,
//         int32_t group_id):
//             Pipeline(f, state_size, gen, state_type, openers, closers,
//             group_id){}

Pipeline::~Pipeline() { MemoryManager::freePinned(state); }

// bytes
size_t Pipeline::getSizeOf(llvm::Type *t) const {
  return layout.getTypeAllocSize(t);
}

int32_t Pipeline::getGroup() const { return group_id; }

void Pipeline::open(const void *s) {
  this->session = s;
  // TODO: for sure it can be done in at least N log N by sorting...
  // for (size_t i = openers.size() ; i > 0 ; --i) {
  //     bool is_last = true;
  //     const void * owner = openers[i - 1].first;
  //     for (size_t j = openers.size() ; j > i ; --j) {
  //         if (openers[j - 1].first == owner){
  //             is_last = false;
  //             break;
  //         }
  //     }
  //     if (is_last) (openers[i - 1].second)(this);
  // }
  assert(!openers.empty());
  (openers[0].second)(this);

  assert(init_state);
  ((void (*)(Pipeline *, void *))init_state)(this, state);

  // for (size_t i = 1 ; i < openers.size() ; ++i) {
  //     bool is_first = true;
  //     const void * owner = openers[i].first;
  //     for (size_t j = 0 ; j < i ; ++j) {
  //         if (openers[j].first == owner){
  //             is_first = false;
  //             break;
  //         }
  //     }
  //     if (is_first){
  //         // std::cout << "o:" << closers[i - 1].first  << std::endl;
  //         (openers[i].second)(this);
  //     }
  // }
  // for (const auto &opener: openers) opener(this);
}

void Pipeline::close() {
  // TODO: for sure it can be done in at least N log N by sorting...
  // for (size_t i = 0 ; i < closers.size() ; ++i) {
  //     bool is_first = true;
  //     const void * owner = closers[i].first;
  //     for (size_t j = 0 ; j < i ; ++j) {
  //         if (closers[j].first == owner){
  //             is_first = false;
  //             break;
  //         }
  //     }
  //     if (is_first) (closers[i].second)(this);
  // }
  // for (size_t i = closers.size() ; i > 1 ; --i) {
  //     bool is_last = true;
  //     const void * owner = closers[i - 1].first;
  //     for (size_t j = closers.size() ; j > i ; --j) {
  //         if (closers[j - 1].first == owner){
  //             is_last = false;
  //             break;
  //         }
  //     }
  //     assert(closers[i - 1].second && "Null closer!");
  //     if (is_last) {
  //         // std::cout << "c:" << closers[i - 1].first  << std::endl;
  //         (closers[i - 1].second)(this);
  //     }
  // }

  assert(deinit_state);
  ((void (*)(Pipeline *, void *))deinit_state)(this, state);

  assert(!closers.empty());
  (closers[0].second)(this);
  // for (size_t i = closers.size() ; i > 0 ; --i) closers[i - 1](this);

  if (execute_after_close) {
    execute_after_close->open(getSession());
    execute_after_close->consume(0);
    execute_after_close->close();
  }
}

std::string PipelineGen::convertTypeToFuncSuffix(llvm::Type *type) {
  if (type->isIntegerTy()) {
    return "i" + std::to_string(type->getIntegerBitWidth());
  }
  if (type->isFloatTy()) return "float";
  if (type->isDoubleTy()) return "double";
  if (type->isPointerTy()) {
    return convertTypeToFuncSuffix(type->getPointerElementType()) + "ptr";
  }
  if (type->isStructTy() && type->getStructNumElements() == 2) {
    auto ti0 = type->getStructElementType(0);
    auto ti1 = type->getStructElementType(1);
    if (ti0->isPointerTy() && ti0->getPointerElementType()->isIntegerTy(8) &&
        ti1->isIntegerTy(32)) {
      return "str";
    }
  }
  {
    auto msg = std::string{"Unexpected type: "};
    {
      llvm::raw_string_ostream ostr{msg};
      type->print(ostr);
    }
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}

std::string PipelineGen::getFunctionNameOverload(std::string name,
                                                 llvm::Type *type) {
  return name + convertTypeToFuncSuffix(type);
}

llvm::Function *PipelineGen::getFunctionOverload(std::string name,
                                                 llvm::Type *type) {
  return getFunction(getFunctionNameOverload(name, type));
}

/*
 * For now, copied  from util/raw-functions.cpp and transformed for Pipeline
 *
 * TODO: deduplicate code...
 */
void PipelineGen::registerFunctions() {
  LLVMContext &ctx = getModule()->getContext();
  Module *TheModule = getModule();
  assert(TheModule != nullptr);

  Type *int1_bool_type = Type::getInt1Ty(ctx);
  Type *int8_type = Type::getInt8Ty(ctx);
  Type *int16_type = Type::getInt16Ty(ctx);
  Type *int32_type = Type::getInt32Ty(ctx);
  Type *int64_type = Type::getInt64Ty(ctx);
  Type *void_type = Type::getVoidTy(ctx);
  Type *double_type = Type::getDoubleTy(ctx);
  StructType *strObjType = Context::CreateStringStruct(ctx);
  PointerType *void_ptr_type = PointerType::get(int8_type, 0);
  PointerType *char_ptr_type = PointerType::get(int8_type, 0);
  PointerType *int32_ptr_type = PointerType::get(int32_type, 0);

  vector<Type *> Ints8Ptr(1, Type::getInt8PtrTy(ctx));
  vector<Type *> Ints8(1, int8_type);
  vector<Type *> Ints1(1, int1_bool_type);
  vector<Type *> Ints(1, int32_type);
  vector<Type *> Ints64(1, int64_type);
  vector<Type *> Floats(1, double_type);
  vector<Type *> Shorts(1, int16_type);

  vector<Type *> ArgsCmpTokens;
  ArgsCmpTokens.insert(ArgsCmpTokens.begin(), char_ptr_type);
  ArgsCmpTokens.insert(ArgsCmpTokens.begin(), int32_type);
  ArgsCmpTokens.insert(ArgsCmpTokens.begin(), int32_type);
  ArgsCmpTokens.insert(ArgsCmpTokens.begin(), char_ptr_type);

  vector<Type *> ArgsCmpTokens64;
  ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);
  ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
  ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
  ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);

  vector<Type *> ArgsConvBoolean;
  ArgsConvBoolean.insert(ArgsConvBoolean.begin(), int32_type);
  ArgsConvBoolean.insert(ArgsConvBoolean.begin(), int32_type);
  ArgsConvBoolean.insert(ArgsConvBoolean.begin(), char_ptr_type);

  vector<Type *> ArgsConvBoolean64;
  ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(), int64_type);
  ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(), int64_type);
  ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(), char_ptr_type);

  vector<Type *> ArgsAtois;
  ArgsAtois.insert(ArgsAtois.begin(), int32_type);
  ArgsAtois.insert(ArgsAtois.begin(), char_ptr_type);

  vector<Type *> ArgsStringObjCmp;
  ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(), strObjType);
  ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(), strObjType);

  vector<Type *> ArgsStringCmp;
  ArgsStringCmp.insert(ArgsStringCmp.begin(), char_ptr_type);
  ArgsStringCmp.insert(ArgsStringCmp.begin(), char_ptr_type);

  /**
   * Args of functions computing hash
   */
  vector<Type *> ArgsHashInt;
  ArgsHashInt.insert(ArgsHashInt.begin(), int32_type);

  vector<Type *> ArgsHashInt64;
  ArgsHashInt64.insert(ArgsHashInt64.begin(), int64_type);

  vector<Type *> ArgsHashDouble;
  ArgsHashDouble.insert(ArgsHashDouble.begin(), double_type);

  vector<Type *> ArgsHashStringC;
  ArgsHashStringC.insert(ArgsHashStringC.begin(), int64_type);
  ArgsHashStringC.insert(ArgsHashStringC.begin(), int64_type);
  ArgsHashStringC.insert(ArgsHashStringC.begin(), char_ptr_type);

  vector<Type *> ArgsHashStringObj;
  ArgsHashStringObj.insert(ArgsHashStringObj.begin(), strObjType);

  vector<Type *> ArgsHashBoolean;
  ArgsHashBoolean.insert(ArgsHashBoolean.begin(), int1_bool_type);

  vector<Type *> ArgsHashCombine;
  ArgsHashCombine.insert(ArgsHashCombine.begin(), int64_type);
  ArgsHashCombine.insert(ArgsHashCombine.begin(), int64_type);

  /**
   * Args of functions computing flush
   */
  vector<Type *> ArgsFlushInt;
  ArgsFlushInt.insert(ArgsFlushInt.begin(), char_ptr_type);
  ArgsFlushInt.insert(ArgsFlushInt.begin(), int32_type);

  vector<Type *> ArgsFlushDString;
  ArgsFlushDString.insert(ArgsFlushDString.begin(), char_ptr_type);
  ArgsFlushDString.insert(ArgsFlushDString.begin(), char_ptr_type);
  ArgsFlushDString.insert(ArgsFlushDString.begin(), int32_type);

  vector<Type *> ArgsFlushInt64;
  ArgsFlushInt64.insert(ArgsFlushInt64.begin(), char_ptr_type);
  ArgsFlushInt64.insert(ArgsFlushInt64.begin(), int64_type);

  vector<Type *> ArgsFlushDate;
  ArgsFlushDate.insert(ArgsFlushDate.begin(), char_ptr_type);
  ArgsFlushDate.insert(ArgsFlushDate.begin(), int64_type);

  vector<Type *> ArgsFlushDouble;
  ArgsFlushDouble.insert(ArgsFlushDouble.begin(), char_ptr_type);
  ArgsFlushDouble.insert(ArgsFlushDouble.begin(), double_type);

  vector<Type *> ArgsFlushStringC;
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), char_ptr_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), int64_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), int64_type);
  ArgsFlushStringC.insert(ArgsFlushStringC.begin(), char_ptr_type);

  vector<Type *> ArgsFlushStringCv2;
  ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(), char_ptr_type);
  ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(), char_ptr_type);

  vector<Type *> ArgsFlushStringObj;
  ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(), char_ptr_type);
  ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(), strObjType);

  vector<Type *> ArgsFlushBoolean;
  ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(), char_ptr_type);
  ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(), int1_bool_type);

  vector<Type *> ArgsFlushStartEnd;
  ArgsFlushStartEnd.insert(ArgsFlushStartEnd.begin(), char_ptr_type);

  vector<Type *> ArgsFlushChar;
  ArgsFlushChar.insert(ArgsFlushChar.begin(), char_ptr_type);
  ArgsFlushChar.insert(ArgsFlushChar.begin(), int8_type);

  vector<Type *> ArgsFlushDelim;
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), char_ptr_type);
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), int8_type);
  ArgsFlushDelim.insert(ArgsFlushDelim.begin(), int64_type);

  vector<Type *> ArgsFlushOutput;
  ArgsFlushOutput.insert(ArgsFlushOutput.begin(), char_ptr_type);

  vector<Type *> ArgsMemoryChunk;
  ArgsMemoryChunk.insert(ArgsMemoryChunk.begin(), int64_type);
  vector<Type *> ArgsIncrMemoryChunk;
  ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(), int64_type);
  ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(), void_ptr_type);
  vector<Type *> ArgsRelMemoryChunk;
  ArgsRelMemoryChunk.insert(ArgsRelMemoryChunk.begin(), void_ptr_type);

  /**
   * Args of timing functions
   */
  // Empty on purpose
  vector<Type *> ArgsTiming;

  FunctionType *FTint = FunctionType::get(Type::getInt32Ty(ctx), Ints, false);
  FunctionType *FTint64 =
      FunctionType::get(Type::getInt32Ty(ctx), Ints64, false);
  FunctionType *FTcharPtr =
      FunctionType::get(Type::getInt32Ty(ctx), Ints8Ptr, false);
  FunctionType *FTatois = FunctionType::get(int32_type, ArgsAtois, false);
  FunctionType *FTatof = FunctionType::get(double_type, Ints8Ptr, false);
  FunctionType *FTprintFloat_ = FunctionType::get(int32_type, Floats, false);
  FunctionType *FTprintShort_ = FunctionType::get(int16_type, Shorts, false);
  FunctionType *FTcompareTokenString_ =
      FunctionType::get(int32_type, ArgsCmpTokens, false);
  FunctionType *FTcompareTokenString64_ =
      FunctionType::get(int32_type, ArgsCmpTokens64, false);
  FunctionType *FTconvertBoolean_ =
      FunctionType::get(int1_bool_type, ArgsConvBoolean, false);
  FunctionType *FTconvertBoolean64_ =
      FunctionType::get(int1_bool_type, ArgsConvBoolean64, false);
  FunctionType *FTprintBoolean_ = FunctionType::get(void_type, Ints1, false);
  FunctionType *FTcompareStringObjs =
      FunctionType::get(int1_bool_type, ArgsStringObjCmp, false);
  FunctionType *FTcompareString =
      FunctionType::get(int1_bool_type, ArgsStringCmp, false);
  FunctionType *FThashInt = FunctionType::get(int64_type, ArgsHashInt, false);
  FunctionType *FThashInt64 =
      FunctionType::get(int64_type, ArgsHashInt64, false);
  FunctionType *FThashDouble =
      FunctionType::get(int64_type, ArgsHashDouble, false);
  FunctionType *FThashStringC =
      FunctionType::get(int64_type, ArgsHashStringC, false);
  FunctionType *FThashStringObj =
      FunctionType::get(int64_type, ArgsHashStringObj, false);
  FunctionType *FThashBoolean =
      FunctionType::get(int64_type, ArgsHashBoolean, false);
  FunctionType *FThashCombine =
      FunctionType::get(int64_type, ArgsHashCombine, false);
  FunctionType *FTflushInt = FunctionType::get(void_type, ArgsFlushInt, false);
  FunctionType *FTflushDString =
      FunctionType::get(void_type, ArgsFlushDString, false);
  FunctionType *FTflushInt64 =
      FunctionType::get(void_type, ArgsFlushInt64, false);
  FunctionType *FTflushDate =
      FunctionType::get(void_type, ArgsFlushDate, false);
  FunctionType *FTflushDouble =
      FunctionType::get(void_type, ArgsFlushDouble, false);
  FunctionType *FTflushStringC =
      FunctionType::get(void_type, ArgsFlushStringC, false);
  FunctionType *FTflushStringCv2 =
      FunctionType::get(void_type, ArgsFlushStringCv2, false);
  FunctionType *FTflushStringObj =
      FunctionType::get(void_type, ArgsFlushStringObj, false);
  FunctionType *FTflushBoolean =
      FunctionType::get(void_type, ArgsFlushBoolean, false);
  FunctionType *FTflushStartEnd =
      FunctionType::get(void_type, ArgsFlushStartEnd, false);
  FunctionType *FTflushChar =
      FunctionType::get(void_type, ArgsFlushChar, false);
  FunctionType *FTflushDelim =
      FunctionType::get(void_type, ArgsFlushDelim, false);
  FunctionType *FTflushOutput =
      FunctionType::get(void_type, ArgsFlushOutput, false);

  FunctionType *FTmemoryChunk =
      FunctionType::get(void_ptr_type, ArgsMemoryChunk, false);
  FunctionType *FTincrMemoryChunk =
      FunctionType::get(void_ptr_type, ArgsIncrMemoryChunk, false);
  FunctionType *FTreleaseMemoryChunk =
      FunctionType::get(void_type, ArgsRelMemoryChunk, false);

  Function *printi_ =
      Function::Create(FTint, Function::ExternalLinkage, "printi", TheModule);
  Function *printi64_ = Function::Create(FTint64, Function::ExternalLinkage,
                                         "printi64", TheModule);
  Function *printc_ = Function::Create(FTcharPtr, Function::ExternalLinkage,
                                       "printc", TheModule);
  Function *printFloat_ = Function::Create(
      FTprintFloat_, Function::ExternalLinkage, "printFloat", TheModule);
  Function *printShort_ = Function::Create(
      FTprintShort_, Function::ExternalLinkage, "printShort", TheModule);
  Function *printBoolean_ = Function::Create(
      FTprintBoolean_, Function::ExternalLinkage, "printBoolean", TheModule);

  Function *atoi_ =
      Function::Create(FTcharPtr, Function::ExternalLinkage, "atoi", TheModule);
  Function *atois_ =
      Function::Create(FTatois, Function::ExternalLinkage, "atois", TheModule);
  atois_->addFnAttr(llvm::Attribute::AlwaysInline);
  Function *atof_ =
      Function::Create(FTatof, Function::ExternalLinkage, "atof", TheModule);

  Function *compareTokenString_ =
      Function::Create(FTcompareTokenString_, Function::ExternalLinkage,
                       "compareTokenString", TheModule);
  compareTokenString_->addFnAttr(llvm::Attribute::AlwaysInline);
  Function *compareTokenString64_ =
      Function::Create(FTcompareTokenString64_, Function::ExternalLinkage,
                       "compareTokenString64", TheModule);
  Function *stringObjEquality =
      Function::Create(FTcompareStringObjs, Function::ExternalLinkage,
                       "equalStringObjs", TheModule);
  stringObjEquality->addFnAttr(llvm::Attribute::AlwaysInline);
  Function *stringEquality = Function::Create(
      FTcompareString, Function::ExternalLinkage, "equalStrings", TheModule);
  stringEquality->addFnAttr(llvm::Attribute::AlwaysInline);

  Function *convertBoolean_ =
      Function::Create(FTconvertBoolean_, Function::ExternalLinkage,
                       "convertBoolean", TheModule);
  convertBoolean_->addFnAttr(llvm::Attribute::AlwaysInline);
  Function *convertBoolean64_ =
      Function::Create(FTconvertBoolean64_, Function::ExternalLinkage,
                       "convertBoolean64", TheModule);
  convertBoolean64_->addFnAttr(llvm::Attribute::AlwaysInline);

  /**
   * Hashing
   */
  Function *hashInt_ = Function::Create(FThashInt, Function::ExternalLinkage,
                                        "hashInt", TheModule);
  Function *hashInt64_ = Function::Create(
      FThashInt64, Function::ExternalLinkage, "hashInt64", TheModule);
  Function *hashDouble_ = Function::Create(
      FThashDouble, Function::ExternalLinkage, "hashDouble", TheModule);
  Function *hashStringC_ = Function::Create(
      FThashStringC, Function::ExternalLinkage, "hashStringC", TheModule);
  Function *hashStringObj_ =
      Function::Create(FThashStringObj, Function::ExternalLinkage,
                       "hashStringObject", TheModule);
  Function *hashBoolean_ = Function::Create(
      FThashBoolean, Function::ExternalLinkage, "hashBoolean", TheModule);
  Function *hashCombine_ = Function::Create(
      FThashCombine, Function::ExternalLinkage, "combineHashes", TheModule);
  Function *hashCombineNoOrder_ =
      Function::Create(FThashCombine, Function::ExternalLinkage,
                       "combineHashesNoOrder", TheModule);

#if 0
    /**
     * Debug (TMP)
     */
    vector<Type*> ArgsDebug;
    ArgsDebug.insert(ArgsDebug.begin(),void_ptr_type);
    FunctionType *FTdebug = FunctionType::get(void_type, ArgsDebug, false);
    Function *debug_ = Function::Create(FTdebug, Function::ExternalLinkage,
                "debug", TheModule);
#endif

  /**
   * Flushing
   */
  Function *flushInt_ = Function::Create(FTflushInt, Function::ExternalLinkage,
                                         "flushInt", TheModule);
  Function *flushDString_ = Function::Create(
      FTflushDString, Function::ExternalLinkage, "flushDString", TheModule);
  Function *flushInt64_ = Function::Create(
      FTflushInt64, Function::ExternalLinkage, "flushInt64", TheModule);
  Function *flushDate_ = Function::Create(
      FTflushDate, Function::ExternalLinkage, "flushDate", TheModule);
  Function *flushDouble_ = Function::Create(
      FTflushDouble, Function::ExternalLinkage, "flushDouble", TheModule);
  Function *flushStringC_ = Function::Create(
      FTflushStringC, Function::ExternalLinkage, "flushStringC", TheModule);
  Function *flushStringCv2_ =
      Function::Create(FTflushStringCv2, Function::ExternalLinkage,
                       "flushStringReady", TheModule);
  Function *flushStringObj_ =
      Function::Create(FTflushStringObj, Function::ExternalLinkage,
                       "flushStringObject", TheModule);
  Function *flushBoolean_ = Function::Create(
      FTflushBoolean, Function::ExternalLinkage, "flushBoolean", TheModule);
  Function *flushObjectStart_ =
      Function::Create(FTflushStartEnd, Function::ExternalLinkage,
                       "flushObjectStart", TheModule);
  Function *flushArrayStart_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushArrayStart", TheModule);
  Function *flushObjectEnd_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushObjectEnd", TheModule);
  Function *flushArrayEnd_ = Function::Create(
      FTflushStartEnd, Function::ExternalLinkage, "flushArrayEnd", TheModule);
  Function *flushChar_ = Function::Create(
      FTflushChar, Function::ExternalLinkage, "flushChar", TheModule);
  Function *flushDelim_ = Function::Create(
      FTflushDelim, Function::ExternalLinkage, "flushDelim", TheModule);
  Function *flushOutput_ = Function::Create(
      FTflushOutput, Function::ExternalLinkage, "flushOutput", TheModule);

  /* Memory Management */
  Function *getMemoryChunk_ = Function::Create(
      FTmemoryChunk, Function::ExternalLinkage, "getMemoryChunk", TheModule);
  Function *increaseMemoryChunk_ =
      Function::Create(FTincrMemoryChunk, Function::ExternalLinkage,
                       "increaseMemoryChunk", TheModule);
  Function *releaseMemoryChunk_ =
      Function::Create(FTreleaseMemoryChunk, Function::ExternalLinkage,
                       "releaseMemoryChunk", TheModule);

  // Memcpy - not used (yet)
  Type *types[] = {void_ptr_type, void_ptr_type, Type::getInt32Ty(ctx)};
  Function *memcpy_ =
      Intrinsic::getDeclaration(TheModule, Intrinsic::memcpy, types);
  if (memcpy_ == nullptr) {
    throw runtime_error(string("Could not find memcpy intrinsic"));
  }

  /**
   * HASHTABLES FOR JOINS / AGGREGATIONS
   */
  // Last type is needed to capture file size. Tentative
  Type *ht_int_types[] = {int32_type, int32_type, void_ptr_type, int32_type};
  FunctionType *FTintHT = FunctionType::get(void_type, ht_int_types, false);
  Function *insertIntKeyToHT_ = Function::Create(
      FTintHT, Function::ExternalLinkage, "insertIntKeyToHT", TheModule);

  Type *ht_types[] = {char_ptr_type, int64_type, void_ptr_type, int32_type};
  FunctionType *FT_HT = FunctionType::get(void_type, ht_types, false);
  Function *insertToHT_ = Function::Create(FT_HT, Function::ExternalLinkage,
                                           "insertToHT", TheModule);

  Type *ht_int_probe_types[] = {int32_type, int32_type, int32_type};
  PointerType *void_ptr_ptr_type = context->getPointerType(void_ptr_type);
  FunctionType *FTint_probeHT =
      FunctionType::get(void_ptr_ptr_type, ht_int_probe_types, false);
  Function *probeIntHT_ = Function::Create(
      FTint_probeHT, Function::ExternalLinkage, "probeIntHT", TheModule);
  probeIntHT_->addFnAttr(llvm::Attribute::AlwaysInline);

  Type *ht_probe_types[] = {char_ptr_type, int64_type};
  FunctionType *FT_probeHT =
      FunctionType::get(void_ptr_ptr_type, ht_probe_types, false);
  Function *probeHT_ = Function::Create(FT_probeHT, Function::ExternalLinkage,
                                        "probeHT", TheModule);
  probeHT_->addFnAttr(llvm::Attribute::AlwaysInline);

  Type *ht_get_metadata_types[] = {char_ptr_type};
  StructType *metadataType = Context::getHashtableMetadataType(ctx);
  PointerType *metadataArrayType = PointerType::get(metadataType, 0);
  FunctionType *FTget_metadata_HT =
      FunctionType::get(metadataArrayType, ht_get_metadata_types, false);
  Function *getMetadataHT_ = Function::Create(
      FTget_metadata_HT, Function::ExternalLinkage, "getMetadataHT", TheModule);

  /**
   * Radix
   */
  /* What the type of HT buckets is */
  vector<Type *> htBucketMembers;
  // int *bucket;
  htBucketMembers.push_back(int32_ptr_type);
  // int *next;
  htBucketMembers.push_back(int32_ptr_type);
  // uint32_t mask;
  htBucketMembers.push_back(int32_type);
  // int count;
  htBucketMembers.push_back(int32_type);
  StructType *htBucketType = StructType::get(ctx, htBucketMembers);
  PointerType *htBucketPtrType = PointerType::get(htBucketType, 0);

  /* JOIN!!! */
  /* What the type of HT entries is */
  /* (int32, void*) */
  vector<Type *> htEntryMembers;
  htEntryMembers.push_back(int32_type);
  htEntryMembers.push_back(int64_type);
  StructType *htEntryType = StructType::get(ctx, htEntryMembers);
  PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

  Type *radix_partition_types[] = {int64_type, htEntryPtrType};
  FunctionType *FTradix_partition =
      FunctionType::get(int32_ptr_type, radix_partition_types, false);
  Function *radix_partition =
      Function::Create(FTradix_partition, Function::ExternalLinkage,
                       "partitionHTLLVM", TheModule);

  Type *bucket_chaining_join_prepare_types[] = {htEntryPtrType, int32_type,
                                                htBucketPtrType};
  FunctionType *FTbucket_chaining_join_prepare =
      FunctionType::get(void_type, bucket_chaining_join_prepare_types, false);
  Function *bucket_chaining_join_prepare = Function::Create(
      FTbucket_chaining_join_prepare, Function::ExternalLinkage,
      "bucket_chaining_join_prepareLLVM", TheModule);

  /* AGGR! */
  /* What the type of HT entries is */
  /* (int64, void*) */
  vector<Type *> htAggEntryMembers;
  htAggEntryMembers.push_back(int64_type);
  htAggEntryMembers.push_back(int64_type);
  StructType *htAggEntryType = StructType::get(ctx, htAggEntryMembers);
  PointerType *htAggEntryPtrType = PointerType::get(htAggEntryType, 0);
  Type *radix_partition_agg_types[] = {int64_type, htAggEntryPtrType};
  FunctionType *FTradix_partition_agg =
      FunctionType::get(int32_ptr_type, radix_partition_agg_types, false);
  Function *radix_partition_agg =
      Function::Create(FTradix_partition_agg, Function::ExternalLinkage,
                       "partitionAggHTLLVM", TheModule);

  Type *bucket_chaining_agg_prepare_types[] = {htAggEntryPtrType, int32_type,
                                               htBucketPtrType};
  FunctionType *FTbucket_chaining_agg_prepare =
      FunctionType::get(void_type, bucket_chaining_agg_prepare_types, false);
  Function *bucket_chaining_agg_prepare =
      Function::Create(FTbucket_chaining_agg_prepare, Function::ExternalLinkage,
                       "bucket_chaining_agg_prepareLLVM", TheModule);
  /**
   * End of Radix
   */

  /**
   * Parsing
   */
  Type *newline_types[] = {char_ptr_type, int64_type};
  FunctionType *FT_newline =
      FunctionType::get(int64_type, newline_types, false);
  Function *newline = Function::Create(FT_newline, Function::ExternalLinkage,
                                       "newlineAVX", TheModule);
  /* Does not make a difference... */
  newline->addFnAttr(llvm::Attribute::AlwaysInline);

  //  vector<Type*> tokenMembers;
  //  tokenMembers.push_back(int32_type);
  //  tokenMembers.push_back(int32_type);
  //  tokenMembers.push_back(int32_type);
  //  tokenMembers.push_back(int32_type);
  //  StructType *tokenType = StructType::get(ctx,tokenMembers);
  StructType *tokenType = Context::CreateJSMNStruct(ctx);

  PointerType *tokenPtrType = PointerType::get(tokenType, 0);
  PointerType *token2DPtrType = PointerType::get(tokenPtrType, 0);
  Type *parse_line_json_types[] = {char_ptr_type, int64_type, int64_type,
                                   token2DPtrType, int64_type};
  FunctionType *FT_parse_line_json =
      FunctionType::get(void_type, parse_line_json_types, false);
  Function *parse_line_json =
      Function::Create(FT_parse_line_json, Function::ExternalLinkage,
                       "parseLineJSON", TheModule);

  Type *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  for (auto t :
       std::vector<llvm::Type *>{int1_bool_type, int8_type, int16_type,
                                 int32_type, int64_type, char_ptr_type}) {
    auto fName = "log" + convertTypeToFuncSuffix(t);

    auto fType =
        FunctionType::get(void_type, {t, char_ptr_type, int32_type}, false);

    auto f =
        Function::Create(fType, Function::ExternalLinkage, fName, TheModule);

    registerFunction(fName, f);
  }

  FunctionType *intrcallPipRegistered = FunctionType::get(
      void_type, std::vector<Type *>{char_ptr_type, size_type, char_ptr_type},
      false);
  Function *intr_pcallPipRegisteredOpen =
      Function::Create(intrcallPipRegistered, Function::ExternalLinkage,
                       "callPipRegisteredOpen", getModule());
  Function *intr_pcallPipRegisteredClose =
      Function::Create(intrcallPipRegistered, Function::ExternalLinkage,
                       "callPipRegisteredClose", getModule());
  registerFunction("callPipRegisteredOpen", intr_pcallPipRegisteredOpen);
  registerFunction("callPipRegisteredClose", intr_pcallPipRegisteredClose);

  FunctionType *intrget_dev_buffer =
      FunctionType::get(char_ptr_type, std::vector<Type *>{}, false);
  Function *intr_pget_dev_buffer =
      Function::Create(intrget_dev_buffer, Function::ExternalLinkage,
                       "get_dev_buffer", getModule());
  registerFunction("get_dev_buffer", intr_pget_dev_buffer);

  FunctionType *intrprintptr =
      FunctionType::get(void_type, std::vector<Type *>{char_ptr_type}, false);
  Function *intr_pprintptr = Function::Create(
      intrprintptr, Function::ExternalLinkage, "printptr", getModule());
  registerFunction("printptr", intr_pprintptr);

  registerFunction("printi", printi_);
  registerFunction("printi64", printi64_);
  registerFunction("printFloat", printFloat_);
  registerFunction("printShort", printShort_);
  registerFunction("printBoolean", printBoolean_);
  registerFunction("printc", printc_);

  registerFunction("atoi", atoi_);
  registerFunction("atois", atois_);
  registerFunction("atof", atof_);

  registerFunction("insertInt", insertIntKeyToHT_);
  registerFunction("probeInt", probeIntHT_);
  registerFunction("insertHT", insertToHT_);
  registerFunction("probeHT", probeHT_);
  registerFunction("getMetadataHT", getMetadataHT_);

  registerFunction("compareTokenString", compareTokenString_);
  registerFunction("compareTokenString64", compareTokenString64_);
  registerFunction("convertBoolean", convertBoolean_);
  registerFunction("convertBoolean64", convertBoolean64_);
  registerFunction("equalStringObjs", stringObjEquality);
  registerFunction("equalStrings", stringEquality);

  registerFunction("hashInt", hashInt_);
  registerFunction("hashInt64", hashInt64_);
  registerFunction("hashDouble", hashDouble_);
  registerFunction("hashStringC", hashStringC_);
  registerFunction("hashStringObject", hashStringObj_);
  registerFunction("hashBoolean", hashBoolean_);
  registerFunction("combineHashes", hashCombine_);
  registerFunction("combineHashesNoOrder", hashCombineNoOrder_);

  registerFunction("flushInt", flushInt_);
  registerFunction("flushDString", flushDString_);
  registerFunction("flushInt64", flushInt64_);
  registerFunction("flushDate", flushDate_);
  registerFunction("flushDouble", flushDouble_);
  registerFunction("flushStringC", flushStringC_);
  registerFunction("flushStringCv2", flushStringCv2_);
  registerFunction("flushStringObj", flushStringObj_);
  registerFunction("flushBoolean", flushBoolean_);
  registerFunction("flushChar", flushChar_);
  registerFunction("flushDelim", flushDelim_);
  registerFunction("flushOutput", flushOutput_);

  registerFunction("flushObjectStart", flushObjectStart_);
  registerFunction("flushArrayStart", flushArrayStart_);
  registerFunction("flushObjectEnd", flushObjectEnd_);
  registerFunction("flushArrayEnd", flushArrayEnd_);
  registerFunction("flushArrayEnd", flushArrayEnd_);

  registerFunction("getMemoryChunk", getMemoryChunk_);
  registerFunction("increaseMemoryChunk", increaseMemoryChunk_);
  registerFunction("releaseMemoryChunk", releaseMemoryChunk_);
  registerFunction("memcpy", memcpy_);

  registerFunction("partitionHT", radix_partition);
  registerFunction("bucketChainingPrepare", bucket_chaining_join_prepare);
  registerFunction("partitionAggHT", radix_partition_agg);
  registerFunction("bucketChainingAggPrepare", bucket_chaining_agg_prepare);

  registerFunction("newline", newline);
  registerFunction("parseLineJSON", parse_line_json);
}
