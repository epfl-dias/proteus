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

#include "util/gpu/gpu-raw-context.hpp"
#include "common/gpu/gpu-common.hpp"

// #include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
// #include "llvm/Target/TargetSubtargetInfo.h"
#include "util/jit/raw-gpu-pipeline.hpp"
#include "util/jit/raw-cpu-pipeline.hpp"

#define DEBUGCTX

void GpuRawContext::createJITEngine() {
//     LLVMLinkInMCJIT();
//     LLVMInitializeNativeTarget();
//     LLVMInitializeNativeAsmPrinter();
//     LLVMInitializeNativeAsmParser();

//     // Create the JIT.  This takes ownership of the module.
//     string ErrStr;
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
//     const Target *target = TargetRegistry::lookupTarget(TheTriple.getTriple(),
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
//     // string ErrStr;
//     // const auto &eng_bld = EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr);

//     // std::string FeaturesStr = getFeaturesStr();

//     // TargetMachine * target_machine = eng_bld.selectTarget(
//     //                                                     TheTriple.getTriple(),
//     //                                                     "sm_61",
//     //                                                     FeaturesStr,
//     //                                                     vector< std::string >{}
//     //                                                 );

//     TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(TheModule))
//                                 .setErrorStr(&ErrStr)
//                                 .create(TheTargetMachine.get());

//     if (!TheExecutionEngine) {
//         std::cout << ErrStr << std::endl;
//         // fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
//         throw runtime_error(error_msg);
//     }
}

size_t GpuRawContext::appendParameter(llvm::Type * ptype, bool noalias, bool readonly){
    return generators.back()->appendParameter(ptype, noalias, readonly);
}

size_t GpuRawContext::appendStateVar(llvm::Type * ptype, std::string name){
    return generators.back()->appendStateVar(ptype);
}

size_t GpuRawContext::appendStateVar(llvm::Type * ptype, std::function<init_func_t> init, std::function<deinit_func_t> deinit, std::string name){
    return generators.back()->appendStateVar(ptype, init, deinit);
}

Argument * GpuRawContext::getArgument(size_t id) const{
    return generators.back()->getArgument(id);
}

Value * GpuRawContext::getStateVar(size_t id) const{
    return generators.back()->getStateVar(id);
}

Value * GpuRawContext::getStateVar() const{
    return generators.back()->getStateVar();
}

std::vector<llvm::Type *> GpuRawContext::getStateVars() const{
    return generators.back()->getStateVars();
}

Value * GpuRawContext::getSubStateVar() const{
    return generators.back()->getSubStateVar();
}

// static void __attribute__((unused)) addOptimizerPipelineDefault(legacy::FunctionPassManager * TheFPM) {
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
// static void __attribute__((unused)) addOptimizerPipelineInlining(ModulePassManager * TheMPM) {
//     /* Inlining: Not sure it works */
//     // LSC: FIXME: No add member to a ModulePassManager
//     TheMPM->add(createFunctionInliningPass());
//     TheMPM->add(createAlwaysInlinerPass());
// }
// #endif

// static void __attribute__((unused)) addOptimizerPipelineVectorization(legacy::FunctionPassManager * TheFPM) {
//     /* Vectorization */
//     TheFPM->add(createBBVectorizePass());
//     TheFPM->add(createLoopVectorizePass());
//     TheFPM->add(createSLPVectorizerPass());
// }

GpuRawContext::GpuRawContext(const string& moduleName, bool gpu_root): 
            RawContext(moduleName, false), kernelName(moduleName), pip_cnt(0){
    createJITEngine();
    if (gpu_root) pushDeviceProvider(&(RawGpuPipelineGenFactory::getInstance()));
    else          pushDeviceProvider(&(RawCpuPipelineGenFactory::getInstance()));

    pushPipeline();
}


GpuRawContext::~GpuRawContext() {
    popDeviceProvider();
    assert(pipFactories.empty() && "someone forgot to pop a device provider");
    LOG(WARNING)<< "[GpuRawContext: ] Destructor";
    //XXX Has to be done in an appropriate sequence - segfaults otherwise
//      delete Builder;
//          delete TheFPM;
//          delete TheExecutionEngine;
//          delete TheFunction;
//          delete llvmContext;
//          delete TheFunction;

    // gpu_run(cuModuleUnload(cudaModule));

    //FIMXE: free pipelines
}

void GpuRawContext::setGlobalFunction(bool leaf){
    setGlobalFunction(NULL, leaf);
}

void GpuRawContext::setGlobalFunction(Function *F, bool leaf){
    if (F){
        string error_msg("[GpuRawContext: ] Should not set global function for GPU context!");
        std::cout << error_msg << std::endl;
        throw runtime_error(error_msg);
    }

    TheFunction = generators.back()->prepare();
    leafgen.push_back(leaf);

    // RawContext::setGlobalFunction(generators.back()->prepare());
}

// void GpuRawContext::pushNewPipeline   (RawPipelineGen * copyStateFrom){
//     time_block t("TregpipsG: ");
//     TheFunction = nullptr;
//     generators.emplace_back(new RawGpuPipelineGen(this, kernelName + "_pip" + std::to_string(pip_cnt++), copyStateFrom));
// }

// void GpuRawContext::pushNewCpuPipeline(RawPipelineGen * copyStateFrom){
//     time_block t("TregpipsC: ");
//     TheFunction = nullptr;
//     generators.emplace_back(new RawCpuPipelineGen(this, kernelName + "_pip" + std::to_string(pip_cnt++), copyStateFrom));
// }

void GpuRawContext::pushDeviceProvider(RawPipelineGenFactory * factory){
    pipFactories.emplace_back(factory);
}

void GpuRawContext::popDeviceProvider(){
    pipFactories.pop_back();
}

void GpuRawContext::pushPipeline(RawPipelineGen * copyStateFrom){
    time_block t("Tregpips: ");
    TheFunction = nullptr;
    generators.emplace_back(pipFactories.back()->create(this, kernelName + "_pip" + std::to_string(pip_cnt++), copyStateFrom));
}

void GpuRawContext::popPipeline(){
    getBuilder()->CreateRetVoid();
    
    pipelines.push_back(generators.back());
    leafpip.push_back(leafgen.back());
    pipelines.back()->compileAndLoad();
    
    generators.pop_back();
    
    TheFunction = (generators.size() != 0) ? generators.back()->F : nullptr;
}

RawPipelineGen * GpuRawContext::removeLatestPipeline(){
    assert(!pipelines.empty());
    RawPipelineGen * p = pipelines.back();
    pipelines.pop_back();
    leafpip.pop_back();
    return p;
}

RawPipelineGen * GpuRawContext::getCurrentPipeline(){
    return generators.back();
}

void GpuRawContext::setChainedPipeline(RawPipelineGen * next){
    generators.back()->setChainedPipeline(next);
}

void GpuRawContext::compileAndLoad(){
    popPipeline();
    assert(generators.size() == 0 && "Leftover pipelines!");
}

std::vector<RawPipeline *> GpuRawContext::getPipelines(){
    std::vector<RawPipeline *> pips;

    assert(pipelines.size() == leafpip.size());
    for (size_t i = 0 ; i < pipelines.size() ; ++i) {
        if (!leafpip[i]) continue;
        pips.emplace_back(pipelines[i]->getPipeline());
    }

    return pips;
}

void GpuRawContext::registerOpen (const void * owner, std::function<void (RawPipeline * pip)> open ){
    generators.back()->registerOpen (owner, open );
}

void GpuRawContext::registerClose(const void * owner, std::function<void (RawPipeline * pip)> close){
    generators.back()->registerClose(owner, close);
}

Value * GpuRawContext::threadIdInBlock(){
    IntegerType * int64_type = Type::getInt64Ty(getLLVMContext());

    if (dynamic_cast<RawGpuPipelineGen *>(generators.back())){
        Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );

        std::vector<Value *> v{};

        Value * threadID_x = getBuilder()->CreateCall(fx , v, "threadID_x");

        // llvm does not provide i32 x i32 => i64, so we cast them to i64
        Value * tid_x      = getBuilder()->CreateZExt(threadID_x, int64_type, "thread_id_in_block");

        return tid_x;
    } else {
        return ConstantInt::get(int64_type, 0);
    }
}

Value * GpuRawContext::blockId(){
    IntegerType * int64_type = Type::getInt64Ty(getLLVMContext());

    if (dynamic_cast<RawGpuPipelineGen *>(generators.back())){
        // Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
        // Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x" );
        Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

        std::vector<Value *> v{};

        // Value * threadID_x = getBuilder()->CreateCall(fx , v, "threadID_x");
        // Value * blockDim_x = getBuilder()->CreateCall(fnx, v, "blockDim_x");
        Value * blockID_x  = getBuilder()->CreateCall(fbx, v, "blockID_x" );


        // llvm does not provide i32 x i32 => i64, so we cast them to i64
        // Value * tid_x      = getBuilder()->CreateZExt(threadID_x, int64_type);
        // Value * bd_x       = getBuilder()->CreateZExt(blockDim_x, int64_type);
        Value * bid_x      = getBuilder()->CreateZExt(blockID_x , int64_type, "block_id");
        return bid_x;
    } else {
        return ConstantInt::get(int64_type, 0);
    }
}

Value * GpuRawContext::threadId(){
    IntegerType * int64_type = Type::getInt64Ty(getLLVMContext());

    if (dynamic_cast<RawGpuPipelineGen *>(generators.back())){
        // Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
        Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x" );
        // Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

        std::vector<Value *> v{};

        // Value * threadID_x = getBuilder()->CreateCall(fx , v, "threadID_x");
        Value * blockDim_x = getBuilder()->CreateCall(fnx, v, "blockDim_x");
        // Value * blockID_x  = getBuilder()->CreateCall(fbx, v, "blockID_x" );


        // llvm does not provide i32 x i32 => i64, so we cast them to i64
        Value * tid_x      = threadIdInBlock();
        Value * bd_x       = getBuilder()->CreateZExt(blockDim_x, int64_type);
        Value * bid_x      = blockId() ;

        Value * rowid      = getBuilder()->CreateMul(bid_x, bd_x, "rowid");
        return getBuilder()->CreateAdd(tid_x, rowid, "thread_id");
    } else {
        return ConstantInt::get(int64_type, 0);
    }
}

Value * GpuRawContext::threadNum(){
    IntegerType * int64_type = Type::getInt64Ty(getLLVMContext());

    if (dynamic_cast<RawGpuPipelineGen *>(generators.back())){
        Function *fnx  = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
        Function *fnbx = getFunction("llvm.nvvm.read.ptx.sreg.nctaid.x");

        std::vector<Value *> v{};

        Value * blockDim_x = getBuilder()->CreateCall(fnx , v, "blockDim_x");
        Value * gridDim_x  = getBuilder()->CreateCall(fnbx, v, "gridDim_x" );

        // llvm does not provide i32 x i32 => i64, so we cast them to i64
        Value * bd_x       = getBuilder()->CreateZExt(blockDim_x, int64_type);
        Value * gd_x       = getBuilder()->CreateZExt(gridDim_x , int64_type);

        return getBuilder()->CreateMul(bd_x, gd_x);
    } else {
        return ConstantInt::get(int64_type, 1);
    }
}

Value * GpuRawContext::laneId(){
    IntegerType * int64_type = Type::getInt64Ty(getLLVMContext());

    if (dynamic_cast<RawGpuPipelineGen *>(generators.back())){
        Function * laneid_fun = getFunction("llvm.nvvm.read.ptx.sreg.laneid");
        return getBuilder()->CreateCall(laneid_fun, std::vector<Value *>{}, "laneid");
    } else {
        return ConstantInt::get(int64_type, 0);
    }
}


void GpuRawContext::createMembar_gl(){
    assert(dynamic_cast<RawGpuPipelineGen *>(generators.back()));
    Function * membar_fun = getFunction("llvm.nvvm.membar.gl");
    getBuilder()->CreateCall(membar_fun, std::vector<Value *>{});
}

//Provide support for some extern functions
void GpuRawContext::registerFunction(const char* funcName, Function* func) {
    generators.back()->registerFunction(funcName, func);
}


llvm::Value * GpuRawContext::allocateStateVar  (llvm::Type *t){
    return generators.back()->allocateStateVar(t);
}

void          GpuRawContext::deallocateStateVar(llvm::Value *v){
    return generators.back()->deallocateStateVar(v);
}
