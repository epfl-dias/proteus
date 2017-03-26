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

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

void GpuRawContext::createJITEngine() {
    LLVMLinkInMCJIT();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    // LLVMInitializeNVPTXAsmParser();

    Triple TheTriple("nvptx64-nvidia-cuda");

    std::string error_msg;
    const Target *target = TargetRegistry::lookupTarget(TheTriple.getTriple(),
                                                        error_msg);
    if (!target) {
        std::cout << error_msg << std::endl;
        throw runtime_error(error_msg);
    }

    std::string FeaturesStr = getFeaturesStr();

    CodeGenOpt::Level OLvl = CodeGenOpt::Aggressive;

    TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
    Options.DisableIntegratedAS             = 1;
    Options.MCOptions.ShowMCEncoding        = 1;
    Options.MCOptions.MCUseDwarfDirectory   = 1;
    // Options.MCOptions.AsmVerbose            = 1;
    Options.MCOptions.PreserveAsmComments   = 1;
    
    TheTargetMachine.reset(target->createTargetMachine(
                                    TheTriple.getTriple(), 
                                    "sm_61", 
                                    FeaturesStr,
                                    Options, 
                                    getRelocModel(), 
                                    CMModel, 
                                    OLvl));

    assert(TheTargetMachine && "Could not allocate target machine!");

    TheFPM->add(new TargetLibraryInfoWrapperPass(TheTriple));
//LinkLibdeviceIfNecessary(module, compute_capability, libdevice_dir_path)




    // Create the JIT.  This takes ownership of the module.
    string ErrStr;
    // const auto &eng_bld = EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr);

    // std::string FeaturesStr = getFeaturesStr();

    // TargetMachine * target_machine = eng_bld.selectTarget(
    //                                                     TheTriple.getTriple(),
    //                                                     "sm_61",
    //                                                     FeaturesStr,
    //                                                     vector< std::string >{}
    //                                                 );

    TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(TheModule))
                                .setErrorStr(&ErrStr)
                                .create(TheTargetMachine.get());

    if (!TheExecutionEngine) {
        std::cout << ErrStr << std::endl;
        // fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
        throw runtime_error(error_msg);
    }
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

GpuRawContext::GpuRawContext(const string& moduleName): 
            RawContext(moduleName, false) {
    Module * mod = getModule();

    if (sizeof(void*) == 8) {
        mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                           "v64:64:64-v128:128:128-n16:32:64");
        mod->setTargetTriple("nvptx64-nvidia-cuda");
    } else {
        mod->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                           "v64:64:64-v128:128:128-n16:32:64");
        mod->setTargetTriple("nvptx-nvidia-cuda");
    }

    Type * int32_type = Type::getInt32Ty(getLLVMContext());

    std::vector<Type *> inputs3{3, int32_type};

    FunctionType *intr = FunctionType::get(int32_type, inputs3, false);
    
    Function *intr_p = Function::Create(intr, Function::ExternalLinkage, "llvm.nvvm.shfl.bfly.i32", mod);
    registerFunction("llvm.nvvm.shfl.bfly.i32", intr_p);

    FunctionType *intr2 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p2 = Function::Create(intr2, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ntid.x", mod);
    registerFunction("llvm.nvvm.read.ptx.sreg.ntid.x", intr_p2);

    FunctionType *intr3 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p3 = Function::Create(intr3, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.tid.x", mod);
    registerFunction("llvm.nvvm.read.ptx.sreg.tid.x", intr_p3);

    FunctionType *intr2b = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p2b = Function::Create(intr2b, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.nctaid.x", mod);
    registerFunction("llvm.nvvm.read.ptx.sreg.nctaid.x", intr_p2b);

    FunctionType *intr3b = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p3b = Function::Create(intr3b, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ctaid.x", mod);
    registerFunction("llvm.nvvm.read.ptx.sreg.ctaid.x", intr_p3b);

    FunctionType *intr4 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p4 = Function::Create(intr4, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.laneid", mod);
    registerFunction("llvm.nvvm.read.ptx.sreg.laneid", intr_p4);
}

GpuRawContext::~GpuRawContext() {
    LOG(WARNING)<< "[GpuRawContext: ] Destructor";
    //XXX Has to be done in an appropriate sequence - segfaults otherwise
//      delete Builder;
//          delete TheFPM;
//          delete TheExecutionEngine;
//          delete TheFunction;
//          delete llvmContext;
//          delete TheFunction;

    gpu_run(cuModuleUnload(cudaModule));
}

void GpuRawContext::setGlobalFunction(Function *F){
    RawContext::setGlobalFunction(F);

    Type *int32Type           = Type::getInt32Ty(TheContext); 
    
    std::vector<llvm::Metadata *> Vals;

    NamedMDNode * annot = TheModule->getOrInsertNamedMetadata("nvvm.annotations");
    MDString    * str   = MDString::get(TheContext, "kernel");
    Value       * one   = ConstantInt::get(int32Type, 1);

    Vals.push_back(ValueAsMetadata::get(F));
    Vals.push_back(str);
    Vals.push_back(ValueAsMetadata::getConstant(one));
    
    MDNode * mdNode = MDNode::get(TheContext, Vals);

    annot->addOperand(mdNode); 
}


void GpuRawContext::compileAndLoad(){
    string ptx = emitPTX();

    gpu_run(cuModuleLoadDataEx(&cudaModule, ptx.c_str(), 0, 0, 0));

}

CUfunction GpuRawContext::getKernel(std::string kernelName){
    CUfunction function;
    gpu_run(cuModuleGetFunction(&function, cudaModule, kernelName.c_str()));
    return function;
}

string GpuRawContext::emitPTX(){
// Based on : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc
// And another forgotten source...
    string ptx;
    {
        raw_string_ostream stream(ptx);
        buffer_ostream ostream(stream);
        
        legacy::PassManager PM;

        // Ask the target to add backend passes as necessary.
        TheTargetMachine->addPassesToEmitFile(PM, ostream, llvm::TargetMachine::CGFT_AssemblyFile, false);

        PM.run(*getModule());
    } // flushes stream and ostream
#ifdef DEBUGCTX
    {
        std::ofstream optx("generated_ptx.ptx");
        optx << ptx;
    }
#endif

    return ptx;
// // std::string Error;
//     // const Target *TheTarget = TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", Error);
//     // if (!TheTarget) {
//     //     std::cout << Error << std::endl;
//     //     EXPECT_TRUE(false);
//     // }

//     // std::string FeaturesStr = getFeaturesStr();

//     // CodeGenOpt::Level OLvl = CodeGenOpt::Aggressive;

//     // TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
//     // // Options.DisableIntegratedAS = llvm::NoIntegratedAssembler;
//     // // Options.MCOptions.ShowMCEncoding = llvm::ShowMCEncoding;
//     // // Options.MCOptions.MCUseDwarfDirectory = llvm::EnableDwarfDirectory;
//     // // Options.MCOptions.AsmVerbose = llvm::AsmVerbose;
//     // // Options.MCOptions.PreserveAsmComments = llvm::PreserveComments;
    
//     // Triple TheTriple("nvptx64-nvidia-cuda");
    
//     // std::unique_ptr<TargetMachine> Target(
//     //   TheTarget->createTargetMachine(TheTriple.getTriple(), "sm_61", FeaturesStr,
//     //                                  Options, getRelocModel(), CMModel, OLvl));

//     // assert(Target && "Could not allocate target machine!");

//     // Build up all of the passes that we want to do to the module.
//     legacy::PassManager PM;

//     // // Add an appropriate TargetLibraryInfo pass for the module's triple.
//     // TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

//     // // The -disable-simplify-libcalls flag actually disables all builtin optzns.
//     // if (DisableSimplifyLibCalls)
//     //   TLII.disableAllFunctions();
//     // PM.add(new TargetLibraryInfoWrapperPass(TLII));

//     // Add the target data from the target machine, if it exists, or the module.
//     // M->setDataLayout(Target->createDataLayout());

//     // Override function attributes based on CPUStr, FeaturesStr, and command line
//     // flags.
//     // setFunctionAttributes(CPUStr, FeaturesStr, *M);

//     SmallString<128> strptx;
//     raw_svector_ostream OS(strptx);

//     // Ask the target to add backend passes as necessary.
//     // if (Target->addPassesToEmitFile(PM, OS, llvm::TargetMachine::CGFT_AssemblyFile, false)) EXPECT_TRUE(false);

//     PM.run(*mod);

//     std::cout << strptx.str().str() << std::endl;
}

void GpuRawContext::prepareFunction(Function *F) {
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    // std::cout << " Here "  << std::endl;
#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code.ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif
    // Validate the generated code, checking for consistency.
    verifyFunction(*F);

    // Optimize the function.
    TheFPM->run(*F);
#if MODULEPASS
    TheMPM->runOnModule(getModule());
#endif

    // JIT the function, returning a function pointer.
    // TheExecutionEngine->finalizeObject();
    // void *FPtr = TheExecutionEngine->getPointerToFunction(F);

    // int (*FP)(void) = (int (*)(void))FPtr;
    // assert(FP != nullptr && "Code generation failed!");


    // //TheModule->dump();
    // //Run function
    // struct timespec t0, t1;
    // clock_gettime(CLOCK_REALTIME, &t0);
    // int jitFuncResult = FP();
    // //LOG(INFO) << "Mock return value of generated function " << FP(11);
    // clock_gettime(CLOCK_REALTIME, &t1);
    // printf("(Already compiled) Execution took %f seconds\n",diff(t0, t1));
    // cout << "Return flag: " << jitFuncResult << endl;

    TheFPM = 0;
    //Dump to see final (optimized) form
#ifdef DEBUGCTX
    // getModule()->dump();
    
    {
        std::error_code EC;
        raw_fd_ostream out("generated_code_opt.ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif
    // std::cout << " Here "  << std::endl;
}

Value * GpuRawContext::threadId(){
    // Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x" );
    // Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    // Function *fy  = getFunction("llvm.nvvm.read.ptx.sreg.tid.y" );

    // std::vector<Value *> v{};

    // Value * threadID_x = TheBuilder->CreateCall(fx , v, "threadID_x");
    // Value * blockDim_x = TheBuilder->CreateCall(fnx, v, "blockDim_x");
    // Value * threadID_y = TheBuilder->CreateCall(fy , v, "threadID_y");

    // Value * rowid      = TheBuilder->CreateMul(threadID_y, blockDim_x, "rowid");
    // return TheBuilder->CreateAdd(threadID_x, rowid, "thread_id");
    Type * int64_type = Type::getInt64Ty(getLLVMContext());

    Function *fx  = getFunction("llvm.nvvm.read.ptx.sreg.tid.x"  );
    Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x" );
    Function *fbx = getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");

    std::vector<Value *> v{};

    Value * threadID_x = TheBuilder->CreateCall(fx , v, "threadID_x");
    Value * blockDim_x = TheBuilder->CreateCall(fnx, v, "blockDim_x");
    Value * blockID_x  = TheBuilder->CreateCall(fbx, v, "blockID_x" );


    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    Value * tid_x      = TheBuilder->CreateZExt(threadID_x, int64_type);
    Value * bd_x       = TheBuilder->CreateZExt(blockDim_x, int64_type);
    Value * bid_x      = TheBuilder->CreateZExt(blockID_x , int64_type);

    Value * rowid      = TheBuilder->CreateMul(bid_x, bd_x, "rowid");
    return TheBuilder->CreateAdd(tid_x, rowid, "thread_id");
}

Value * GpuRawContext::threadNum(){
    // Function *fnx = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    // Function *fny = getFunction("llvm.nvvm.read.ptx.sreg.ntid.y");

    // std::vector<Value *> v{};

    // Value * blockDim_x = TheBuilder->CreateCall(fnx, v, "blockDim_x");
    // Value * blockDim_y = TheBuilder->CreateCall(fny, v, "blockDim_y");

    // return TheBuilder->CreateMul(blockDim_x, blockDim_y);
    Type * int64_type = Type::getInt64Ty(getLLVMContext());

    Function *fnx  = getFunction("llvm.nvvm.read.ptx.sreg.ntid.x");
    Function *fnbx = getFunction("llvm.nvvm.read.ptx.sreg.nctaid.x");

    std::vector<Value *> v{};

    Value * blockDim_x = TheBuilder->CreateCall(fnx , v, "blockDim_x");
    Value * gridDim_x  = TheBuilder->CreateCall(fnbx, v, "gridDim_x" );

    // llvm does not provide i32 x i32 => i64, so we cast them to i64
    Value * bd_x       = TheBuilder->CreateZExt(blockDim_x, int64_type);
    Value * gd_x       = TheBuilder->CreateZExt(gridDim_x , int64_type);

    return TheBuilder->CreateMul(bd_x, gd_x);
}

Value * GpuRawContext::laneId(){
    Function * laneid_fun = getFunction("llvm.nvvm.read.ptx.sreg.laneid");
    return TheBuilder->CreateCall(laneid_fun, std::vector<Value *>{}, "laneid");
}

