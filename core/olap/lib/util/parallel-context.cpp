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

#include <llvm/MC/TargetRegistry.h>

#include <olap/util/parallel-context.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/util/timing.hpp>

#include "lib/util/jit/cpu-pipeline.hpp"
#include "lib/util/jit/gpu-pipeline.hpp"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

void ParallelContext::createJITEngine() {
  //     LLVMLinkInMCJIT();
  //     LLVMInitializeNativeTarget();
  //     LLVMInitializeNativeAsmPrinter();
  //     LLVMInitializeNativeAsmParser();

  //     // Create the JIT.  This takes ownership of the module.
  //     std::string ErrStr;
  //     TheCPUExecutionEngine =
  //         EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr).create();
  //     if (TheCPUExecutionEngine == nullptr) {
  //         fprintf(stderr, "Could not create ExecutionEngine: %s\n",
  //                 ErrStr.c_str());
  //         exit(1);
  //     }

  //     // LLVMLinkInMCJIT();
  //     LLVMInitializeNVPTXTarget();
  //     LLVMInitializeNVPTXTargetInfo();
  //     LLVMInitializeNVPTXTargetMC();
  //     LLVMInitializeNVPTXAsmPrinter();
  //     // LLVMInitializeNVPTXAsmParser();

  //     Triple TheTriple("nvptx64-nvidia-cuda");

  //     std::string error_msg;
  //     const Target *target =
  //     TargetRegistry::lookupTarget(TheTriple.getTriple(),
  //                                                         error_msg);
  //     if (!target) {
  //         std::cout << error_msg << std::endl;
  //         throw runtime_error(error_msg);
  //     }

  //     std::string FeaturesStr = getFeaturesStr();

  //     CodeGenOpt::Level OLvl = CodeGenOpt::Aggressive;

  //     TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  //     Options.DisableIntegratedAS             = 1;
  //     Options.MCOptions.ShowMCEncoding        = 1;
  //     Options.MCOptions.MCUseDwarfDirectory   = 1;
  //     // Options.MCOptions.AsmVerbose            = 1;
  //     Options.MCOptions.PreserveAsmComments   = 1;

  //     TheTargetMachine.reset(target->createTargetMachine(
  //                                     TheTriple.getTriple(),
  //                                     "sm_61",
  //                                     FeaturesStr,
  //                                     Options,
  //                                     getRelocModel(),
  //                                     CMModel,
  //                                     OLvl));

  //     assert(TheTargetMachine && "Could not allocate target machine!");

  //     TheFPM->add(new TargetLibraryInfoWrapperPass(TheTriple));
  // //LinkLibdeviceIfNecessary(module, compute_capability, libdevice_dir_path)

  //     // Create the JIT.  This takes ownership of the module.
  //     // std::string ErrStr;
  //     // const auto &eng_bld =
  //     EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr);

  //     // std::string FeaturesStr = getFeaturesStr();

  //     // TargetMachine * target_machine = eng_bld.selectTarget(
  //     // TheTriple.getTriple(),
  //     //                                                     "sm_61",
  //     //                                                     FeaturesStr,
  //     //                                                     vector<
  //     std::string >{}
  //     //                                                 );

  //     TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(TheModule))
  //                                 .setErrorStr(&ErrStr)
  //                                 .create(TheTargetMachine.get());

  //     if (!TheExecutionEngine) {
  //         std::cout << ErrStr << std::endl;
  //         // fprintf(stderr, "Could not create ExecutionEngine: %s\n",
  //         ErrStr.c_str()); throw runtime_error(error_msg);
  //     }
}

size_t ParallelContext::appendParameter(llvm::Type *ptype, bool noalias,
                                        bool readonly) {
  return getCurrentPipeline()->appendParameter(ptype, noalias, readonly);
}

StateVar ParallelContext::appendStateVar(llvm::Type *ptype, std::string name) {
  return generators.back()->appendStateVar(ptype);
}

StateVar ParallelContext::appendStateVar(llvm::Type *ptype,
                                         std::function<init_func_t> init,
                                         std::function<deinit_func_t> deinit,
                                         std::string name) {
  return generators.back()->appendStateVar(ptype, init, deinit);
}

[[nodiscard]] llvm::Value *ParallelContext::getSessionParametersPtr() const {
  return generators.back()->getSessionParametersPtr();
}

llvm::Argument *ParallelContext::getArgument(size_t id) const {
  return getCurrentPipeline()->getArgument(id);
}

llvm::Value *ParallelContext::getStateVar(const StateVar &id) const {
  return generators.back()->getStateVar(id);
}

llvm::Value *ParallelContext::getStateVar() const {
  return getCurrentPipeline()->getStateVar();
}

std::vector<llvm::Type *> ParallelContext::getStateVars() const {
  return getCurrentPipeline()->getStateVars();
}

llvm::Value *ParallelContext::getSubStateVar() const {
  return getCurrentPipeline()->getSubStateVar();
}

// static void __attribute__((unused))
// addOptimizerPipelineDefault(legacy::FunctionPassManager * TheFPM) {
//     //Provide basic AliasAnalysis support for GVN.
//     TheFPM->add(createBasicAAWrapperPass());
//     // Promote allocas to registers.
//     TheFPM->add(createPromoteMemoryToRegisterPass());
//     //Do simple "peephole" optimizations and bit-twiddling optzns.
//     TheFPM->add(createInstructionCombiningPass());
//     // Reassociate expressions.
//     TheFPM->add(createReassociatePass());
//     // Eliminate Common SubExpressions.
//     TheFPM->add(createGVNPass());
//     // Simplify the control flow graph (deleting unreachable blocks, etc).
//     TheFPM->add(createCFGSimplificationPass());
//     // Aggressive Dead Code Elimination. Make sure work takes place
//     TheFPM->add(createAggressiveDCEPass());
// }

// #if MODULEPASS
// static void __attribute__((unused))
// addOptimizerPipelineInlining(ModulePassManager * TheMPM) {
//     /* Inlining: Not sure it works */
//     // LSC: FIXME: No add member to a ModulePassManager
//     TheMPM->add(createFunctionInliningPass());
//     TheMPM->add(createAlwaysInlinerPass());
// }
// #endif

// static void __attribute__((unused))
// addOptimizerPipelineVectorization(legacy::FunctionPassManager * TheFPM) {
//     /* Vectorization */
//     TheFPM->add(createBBVectorizePass());
//     TheFPM->add(createLoopVectorizePass());
//     TheFPM->add(createSLPVectorizerPass());
// }

ParallelContext::ParallelContext(const std::string &moduleName, bool gpu_root)
    : Context(moduleName), kernelName(moduleName), pip_cnt(0) {
  createJITEngine();
  if (gpu_root)
    pushDeviceProvider(&(GpuPipelineGenFactory::getInstance()));
  else
    pushDeviceProvider(&(CpuPipelineGenFactory::getInstance()));

  pushPipeline();
}

ParallelContext::~ParallelContext() {
  popDeviceProvider();
  assert(pipFactories.empty() && "someone forgot to pop a device provider");
  LOG(WARNING) << "[ParallelContext: ] Destructor";
  // XXX Has to be done in an appropriate sequence - segfaults otherwise
  //      delete Builder;
  //          delete TheFPM;
  //          delete TheExecutionEngine;
  //          delete TheFunction;
  //          delete llvmContext;
  //          delete TheFunction;

  // gpu_run(cuModuleUnload(cudaModule));

  // FIMXE: free pipelines
}

void ParallelContext::setGlobalFunction(bool leaf) {
  setGlobalFunction(nullptr, leaf);
}

void ParallelContext::setGlobalFunction(llvm::Function *F, bool leaf) {
  if (F) {
    std::string error_msg(
        "[ParallelContext: ] Should not set global function for GPU context!");
    std::cout << error_msg << std::endl;
    throw runtime_error(error_msg);
  }

  TheFunction = getCurrentPipeline()->prepare();
  leafgen.push_back(leaf);

  // Context::setGlobalFunction(getCurrentPipeline()->prepare());
}

// void ParallelContext::pushNewPipeline   (PipelineGen * copyStateFrom){
//     time_block t("TregpipsG: ");
//     TheFunction = nullptr;
//     generators.emplace_back(new GpuPipelineGen(this, kernelName + "_pip" +
//     std::to_std::string(pip_cnt++), copyStateFrom));
// }

// void ParallelContext::pushNewCpuPipeline(PipelineGen * copyStateFrom){
//     time_block t("TregpipsC: ");
//     TheFunction = nullptr;
//     generators.emplace_back(new CpuPipelineGen(this, kernelName + "_pip" +
//     std::to_std::string(pip_cnt++), copyStateFrom));
// }

void ParallelContext::pushDeviceProvider(PipelineGenFactory *factory) {
  pipFactories.emplace_back(factory);
}

void ParallelContext::popDeviceProvider() { pipFactories.pop_back(); }

void ParallelContext::pushPipeline(PipelineGen *copyStateFrom) {
  static size_t pip_cnt = 0;
  TheFunction = nullptr;
  generators.emplace_back(pipFactories.back()->create(
      this, kernelName + "_pip" + std::to_string(pip_cnt++), copyStateFrom));

  TheBuilder = new llvm::IRBuilder<>(getLLVMContext());
}

void ParallelContext::popPipeline() {
  getBuilder()->CreateRetVoid();

  pipelines.push_back(generators.back());
  leafpip.push_back(leafgen.back());
  pipelines.back()->compileAndLoad();

  generators.pop_back();

  TheFunction = (generators.size() != 0) ? getCurrentPipeline()->F : nullptr;
}

PipelineGen *ParallelContext::removeLatestPipeline() {
  assert(!pipelines.empty());
  PipelineGen *p = pipelines.back();
  pipelines.pop_back();
  leafpip.pop_back();
  return p;
}

PipelineGen *ParallelContext::getCurrentPipeline() const {
  assert(!generators.empty());
  return generators.back();
}

void ParallelContext::setChainedPipeline(PipelineGen *next) {
  getCurrentPipeline()->setChainedPipeline(next);
}

void ParallelContext::compileAndLoad() {
  popPipeline();
  assert(generators.size() == 0 && "Leftover pipelines!");
  // prepare for next code generation
  pushPipeline();
}

std::vector<std::unique_ptr<Pipeline>> ParallelContext::getPipelines() {
  std::vector<std::unique_ptr<Pipeline>> pips;

  assert(pipelines.size() == leafpip.size());
  for (size_t i = 0; i < pipelines.size(); ++i) {
    if (!leafpip[i]) continue;
    pips.emplace_back(pipelines[i]->getPipeline());
  }
  pipelines.clear();
  leafpip.clear();
  return pips;
}

void ParallelContext::registerOpen(const void *owner,
                                   std::function<void(Pipeline *pip)> open) {
  getCurrentPipeline()->registerOpen(owner, open);
}

void ParallelContext::registerClose(const void *owner,
                                    std::function<void(Pipeline *pip)> close) {
  getCurrentPipeline()->registerClose(owner, close);
}

llvm::Value *ParallelContext::threadIdInBlock() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    llvm::Function *fx = getFunction("llvm.nvvm.read.ptx.sreg.tid.x");

    std::vector<llvm::Value *> v{};

    llvm::Value *threadID_x = getBuilder()->CreateCall(fx, v, "threadID_x");

    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    llvm::Value *tid_x =
        getBuilder()->CreateZExt(threadID_x, int64_type, "thread_id_in_block");

    return tid_x;
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

llvm::Value *ParallelContext::blockId() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    // llvm::Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
    // llvm::Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x" );
    llvm::Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    std::vector<llvm::Value *> v{};

    // llvm::Value * threadID_x = getBuilder()->CreateCall(fx , v,
    // "threadID_x"); llvm::Value * blockDim_x = getBuilder()->CreateCall(fnx,
    // v, "blockDim_x");
    llvm::Value *blockID_x = getBuilder()->CreateCall(fbx, v, "blockID_x");

    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    // llvm::Value * tid_x      = getBuilder()->CreateZExt(threadID_x,
    // int64_type); llvm::Value * bd_x       =
    // getBuilder()->CreateZExt(blockDim_x, int64_type);
    llvm::Value *bid_x =
        getBuilder()->CreateZExt(blockID_x, int64_type, "block_id");
    return bid_x;
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

llvm::Value *ParallelContext::blockDim() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    // llvm::Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
    llvm::Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    // llvm::Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    std::vector<llvm::Value *> v{};

    // llvm::Value * threadID_x = getBuilder()->CreateCall(fx , v,
    // "threadID_x");
    llvm::Value *blockDim_x = getBuilder()->CreateCall(fnx, v, "blockDim_x");
    llvm::Value *bd_x = getBuilder()->CreateZExt(blockDim_x, int64_type);
    return bd_x;
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

llvm::Value *ParallelContext::gridDim() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    // llvm::Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
    llvm::Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.nctaid.x");
    // llvm::Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    std::vector<llvm::Value *> v{};

    // llvm::Value * threadID_x = getBuilder()->CreateCall(fx , v,
    // "threadID_x");
    llvm::Value *gridDim_x = getBuilder()->CreateCall(fnx, v, "gridDim_x");
    llvm::Value *bd_x = getBuilder()->CreateZExt(gridDim_x, int64_type);
    return bd_x;
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

llvm::Value *ParallelContext::threadId() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    // llvm::Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
    llvm::Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    // llvm::Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    std::vector<llvm::Value *> v{};

    // llvm::Value * threadID_x = getBuilder()->CreateCall(fx , v,
    // "threadID_x");
    llvm::Value *blockDim_x = getBuilder()->CreateCall(fnx, v, "blockDim_x");
    // llvm::Value * blockID_x  = getBuilder()->CreateCall(fbx, v, "blockID_x"
    // );

    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    llvm::Value *tid_x = threadIdInBlock();
    llvm::Value *bd_x = getBuilder()->CreateZExt(blockDim_x, int64_type);
    llvm::Value *bid_x = blockId();

    llvm::Value *rowid = getBuilder()->CreateMul(bid_x, bd_x, "rowid");
    return getBuilder()->CreateAdd(tid_x, rowid, "thread_id");
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

llvm::Value *ParallelContext::threadNum() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    llvm::Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    llvm::Function *fnbx = getFunction("llvm.nvvm.read.ptx.sreg.nctaid.x");

    std::vector<llvm::Value *> v{};

    llvm::Value *blockDim_x = getBuilder()->CreateCall(fnx, v, "blockDim_x");
    llvm::Value *gridDim_x = getBuilder()->CreateCall(fnbx, v, "gridDim_x");

    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    llvm::Value *bd_x = getBuilder()->CreateZExt(blockDim_x, int64_type);
    llvm::Value *gd_x = getBuilder()->CreateZExt(gridDim_x, int64_type);

    return getBuilder()->CreateMul(bd_x, gd_x);
  } else {
    return llvm::ConstantInt::get(int64_type, 1);
  }
}

llvm::Value *ParallelContext::laneId() {
  auto int64_type = llvm::Type::getInt64Ty(getLLVMContext());

  if (dynamic_cast<GpuPipelineGen *>(generators.back())) {
    llvm::Function *laneid_fun = getFunction("llvm.nvvm.read.ptx.sreg.laneid");
    return getBuilder()->CreateCall(laneid_fun, std::vector<llvm::Value *>{},
                                    "laneid");
  } else {
    return llvm::ConstantInt::get(int64_type, 0);
  }
}

[[deprecated]] void ParallelContext::createMembar_gl() {
  assert(dynamic_cast<GpuPipelineGen *>(generators.back()) &&
         "Unimplemented for CPU");
  llvm::Function *membar_fun = getFunction("llvm.nvvm.membar.gl");
  getBuilder()->CreateCall(membar_fun, std::vector<llvm::Value *>{});
}

void ParallelContext::workerScopedMembar() {
  getCurrentPipeline()->workerScopedMembar();
}

// Provide support for some extern functions
void ParallelContext::registerFunction(const char *funcName,
                                       llvm::Function *func) {
  getCurrentPipeline()->registerFunction(funcName, func);
}

llvm::Value *ParallelContext::allocateStateVar(llvm::Type *t) {
  return getCurrentPipeline()->allocateStateVar(t);
}

void ParallelContext::deallocateStateVar(llvm::Value *v) {
  return getCurrentPipeline()->deallocateStateVar(v);
}

llvm::Value *ParallelContext::workerScopedAtomicAdd(llvm::Value *ptr,
                                                    llvm::Value *inc) {
  return getCurrentPipeline()->workerScopedAtomicAdd(ptr, inc);
}

llvm::Value *ParallelContext::workerScopedAtomicXchg(llvm::Value *ptr,
                                                     llvm::Value *val) {
  return getCurrentPipeline()->workerScopedAtomicXchg(ptr, val);
}

void ParallelContext::log(llvm::Value *out, decltype(__builtin_FILE()) file,
                          decltype(__builtin_LINE()) line) {
  auto f = getFunctionOverload("log", out->getType());
  getBuilder()->CreateCall(f,
                           {out, CreateGlobalString(file), createInt32(line)});
}

llvm::BasicBlock *ParallelContext::getEndingBlock() {
  return getCurrentPipeline()->getEndingBlock();
}

void ParallelContext::setEndingBlock(llvm::BasicBlock *codeEnd) {
  getCurrentPipeline()->setEndingBlock(codeEnd);
}

llvm::BasicBlock *ParallelContext::getCurrentEntryBlock() {
  return getCurrentPipeline()->getCurrentEntryBlock();
}

void ParallelContext::setCurrentEntryBlock(llvm::BasicBlock *codeEntry) {
  getCurrentPipeline()->setCurrentEntryBlock(codeEntry);
}

llvm::Module *ParallelContext::getModule() const {
  return getCurrentPipeline()->getModule();
}

llvm::IRBuilder<> *ParallelContext::getBuilder() const {
  return getCurrentPipeline()->getBuilder();
}

llvm::Function *ParallelContext::getFunction(string funcName) const {
  return getCurrentPipeline()->getFunction(funcName);
}

llvm::Function *ParallelContext::getFunctionOverload(std::string name,
                                                     llvm::Type *type) {
  return getCurrentPipeline()->getFunctionOverload(name, type);
}

std::string ParallelContext::getFunctionNameOverload(std::string name,
                                                     llvm::Type *type) {
  return getCurrentPipeline()->getFunctionNameOverload(name, type);
}
