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

#include "raw-cpu-pipeline.hpp"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/CodeGen/TargetPassConfig.h"


LLVMTargetMachine                             * RawCpuPipelineGen::TheTargetMachine = nullptr;
legacy::PassManager                             RawCpuPipelineGen::Passes;
PassManagerBuilder                              RawCpuPipelineGen::Builder;
std::unique_ptr<legacy::FunctionPassManager>    RawCpuPipelineGen::FPasses;

RawCpuPipelineGen::RawCpuPipelineGen(   RawContext        * context         , 
                                        std::string         pipName         , 
                                        RawPipelineGen    * copyStateFrom   ): 
            RawPipelineGen(context, pipName, copyStateFrom){
//     /* OPTIMIZER PIPELINE, function passes */
//     TheFPM = new legacy::FunctionPassManager(getModule());
//     addOptimizerPipelineDefault(TheFPM);
//     // TheFPM->add(createLoadCombinePass());


//     //LSC: Seems to be faster without the vectorization, at least
//     //while running the unit-tests, but this might be because the
//     //datasets are too small.
//     addOptimizerPipelineVectorization(TheFPM);
        
// #if MODULEPASS
//     /* OPTIMIZER PIPELINE, module passes */
//     PassManagerBuilder pmb;
//     pmb.OptLevel=3;
//     TheMPM = new ModulePassManager();
//     pmb.populateModulePassManager(*TheMPM);
//     addOptimizerPipelineInlining(TheMPM);
// #endif

    // TheFPM->doInitialization();
    Type * bool_type    = Type::getInt1Ty   (context->getLLVMContext());
    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * int64_type   = Type::getInt64Ty  (context->getLLVMContext());
    Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    FunctionType * FTlaunch_kernel        = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0)}, 
                                                    false
                                                );

    Function * launch_kernel_             = Function::Create(
                                                    FTlaunch_kernel,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel",launch_kernel_);

    FunctionType * FTlaunch_kernel_strm        = FunctionType::get(
                                                    void_type, 
                                                    std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0), charPtrType}, 
                                                    false
                                                );

    Function * launch_kernel_strm_             = Function::Create(
                                                    FTlaunch_kernel_strm,
                                                    Function::ExternalLinkage, 
                                                    "launch_kernel_strm", 
                                                    getModule()
                                                );

    registerFunction("launch_kernel_strm",launch_kernel_strm_);

    Type * pair_type = StructType::get(context->getLLVMContext(), std::vector<Type *>{charPtrType, charPtrType});
    FunctionType *make_mem_move_device = FunctionType::get(pair_type, std::vector<Type *>{charPtrType, size_type, int32_type, charPtrType}, false);
    Function *fmake_mem_move_device = Function::Create(make_mem_move_device, Function::ExternalLinkage, "make_mem_move_device", getModule());
    registerFunction("make_mem_move_device", fmake_mem_move_device);

    FunctionType *make_mem_move_local_to = FunctionType::get(pair_type, std::vector<Type *>{charPtrType, size_type, int32_type, charPtrType}, false);
    Function *fmake_mem_move_local_to = Function::Create(make_mem_move_local_to, Function::ExternalLinkage, "make_mem_move_local_to", getModule());
    registerFunction("make_mem_move_local_to", fmake_mem_move_local_to);

    FunctionType *acquireBuffer = FunctionType::get(charPtrType, std::vector<Type *>{int32_type, charPtrType}, false);
    Function *facquireBuffer = Function::Create(acquireBuffer, Function::ExternalLinkage, "acquireBuffer", getModule());
    {
        std::vector<std::pair<unsigned, Attribute>> attrs;
        Attribute def  = Attribute::getWithDereferenceableBytes(context->getLLVMContext(), h_vector_size*sizeof(int32_t)); //FIXME: at some point this should change...
        attrs.emplace_back(0, def);
        facquireBuffer->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    }
    registerFunction("acquireBuffer", facquireBuffer);

    FunctionType *try_acquireBuffer = FunctionType::get(charPtrType, std::vector<Type *>{int32_type, charPtrType}, false);
    Function *ftry_acquireBuffer = Function::Create(try_acquireBuffer, Function::ExternalLinkage, "try_acquireBuffer", getModule());
    {
        std::vector<std::pair<unsigned, Attribute>> attrs;
        Attribute def  = Attribute::getWithDereferenceableOrNullBytes(context->getLLVMContext(), h_vector_size*sizeof(int32_t)); //FIXME: at some point this should change...
        attrs.emplace_back(0, def);
        ftry_acquireBuffer->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    }
    registerFunction("try_acquireBuffer", ftry_acquireBuffer);

    FunctionType *allocate = FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
    Function *fallocate = Function::Create(allocate, Function::ExternalLinkage, "allocate_pinned", getModule());
    std::vector<std::pair<unsigned, Attribute>> attrs;
    Attribute noAlias  = Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias);
    attrs.emplace_back(0, noAlias);
    fallocate->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    registerFunction("allocate", fallocate);

    FunctionType *deallocate = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *fdeallocate = Function::Create(deallocate, Function::ExternalLinkage, "deallocate_pinned", getModule());
    registerFunction("deallocate", fdeallocate);

    FunctionType *releaseBuffer = FunctionType::get(void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType}, false);
    Function *freleaseBuffer = Function::Create(releaseBuffer, Function::ExternalLinkage, "releaseBuffer", getModule());
    registerFunction("releaseBuffer", freleaseBuffer);

    FunctionType *freeBuffer = FunctionType::get(void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType}, false);
    Function *ffreeBuffer = Function::Create(freeBuffer, Function::ExternalLinkage, "freeBuffer", getModule());
    registerFunction("freeBuffer", ffreeBuffer);

    FunctionType *crand = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *fcrand = Function::Create(crand, Function::ExternalLinkage, "rand", getModule());
    registerFunction("rand", fcrand);

    FunctionType *cprintTime = FunctionType::get(void_type, std::vector<Type *>{}, false);
    Function *fcprintTime = Function::Create(cprintTime, Function::ExternalLinkage, "printTime", getModule());
    registerFunction("printTime", fcprintTime);

    FunctionType *get_buffer = FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
    Function *fget_buffer = Function::Create(get_buffer, Function::ExternalLinkage, "get_buffer", getModule());
    registerFunction("get_buffer", fget_buffer);

    FunctionType *release_buffer = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *frelease_buffer = Function::Create(release_buffer, Function::ExternalLinkage, "release_buffer", getModule());
    registerFunction("release_buffer" , frelease_buffer);
    registerFunction("release_buffers", frelease_buffer);

    FunctionType *yield = FunctionType::get(void_type, std::vector<Type *>{}, false);
    Function *fyield = Function::Create(yield, Function::ExternalLinkage, "yield", getModule());
    registerFunction("yield", fyield);

    FunctionType *get_ptr_device = FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
    Function *fget_ptr_device = Function::Create(get_ptr_device, Function::ExternalLinkage, "get_ptr_device", getModule());
    registerFunction("get_ptr_device", fget_ptr_device);

    FunctionType *get_ptr_device_or_rand_for_host = FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
    Function *fget_ptr_device_or_rand_for_host = Function::Create(get_ptr_device_or_rand_for_host, Function::ExternalLinkage, "get_ptr_device_or_rand_for_host", getModule());
    registerFunction("get_ptr_device_or_rand_for_host", fget_ptr_device_or_rand_for_host);
    
    FunctionType *get_rand_core_local_to_ptr = FunctionType::get(int32_type, std::vector<Type *>{charPtrType}, false);
    Function *fget_rand_core_local_to_ptr = Function::Create(get_rand_core_local_to_ptr, Function::ExternalLinkage, "get_rand_core_local_to_ptr", getModule());
    registerFunction("get_rand_core_local_to_ptr", fget_rand_core_local_to_ptr);

    FunctionType *acquireWorkUnit = FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
    Function *facquireWorkUnit = Function::Create(acquireWorkUnit, Function::ExternalLinkage, "acquireWorkUnit", getModule());
    registerFunction("acquireWorkUnit", facquireWorkUnit);

    FunctionType *propagateWorkUnit = FunctionType::get(void_type, std::vector<Type *>{charPtrType, charPtrType, bool_type}, false);
    Function *fpropagateWorkUnit = Function::Create(propagateWorkUnit, Function::ExternalLinkage, "propagateWorkUnit", getModule());
    registerFunction("propagateWorkUnit", fpropagateWorkUnit);

    FunctionType *acquirePendingWorkUnit = FunctionType::get(bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
    Function *facquirePendingWorkUnit = Function::Create(acquirePendingWorkUnit, Function::ExternalLinkage, "acquirePendingWorkUnit", getModule());
    registerFunction("acquirePendingWorkUnit", facquirePendingWorkUnit);

    FunctionType *releaseWorkUnit = FunctionType::get(void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
    Function *freleaseWorkUnit = Function::Create(releaseWorkUnit, Function::ExternalLinkage, "releaseWorkUnit", getModule());
    registerFunction("releaseWorkUnit", freleaseWorkUnit);

    FunctionType *mem_move_local_to_acquireWorkUnit = FunctionType::get(charPtrType, std::vector<Type *>{charPtrType}, false);
    Function *fmem_move_local_to_acquireWorkUnit = Function::Create(mem_move_local_to_acquireWorkUnit, Function::ExternalLinkage, "mem_move_local_to_acquireWorkUnit", getModule());
    registerFunction("mem_move_local_to_acquireWorkUnit", fmem_move_local_to_acquireWorkUnit);

    FunctionType *mem_move_local_to_propagateWorkUnit = FunctionType::get(void_type, std::vector<Type *>{charPtrType, charPtrType, bool_type}, false);
    Function *fmem_move_local_to_propagateWorkUnit = Function::Create(mem_move_local_to_propagateWorkUnit, Function::ExternalLinkage, "mem_move_local_to_propagateWorkUnit", getModule());
    registerFunction("mem_move_local_to_propagateWorkUnit", fmem_move_local_to_propagateWorkUnit);

    FunctionType *mem_move_local_to_acquirePendingWorkUnit = FunctionType::get(bool_type, std::vector<Type *>{charPtrType, charPtrType}, false);
    Function *fmem_move_local_to_acquirePendingWorkUnit = Function::Create(mem_move_local_to_acquirePendingWorkUnit, Function::ExternalLinkage, "mem_move_local_to_acquirePendingWorkUnit", getModule());
    registerFunction("mem_move_local_to_acquirePendingWorkUnit", fmem_move_local_to_acquirePendingWorkUnit);

    FunctionType *mem_move_local_to_releaseWorkUnit = FunctionType::get(void_type, std::vector<Type *>{charPtrType, charPtrType}, false);
    Function *fmem_move_local_to_releaseWorkUnit = Function::Create(mem_move_local_to_releaseWorkUnit, Function::ExternalLinkage, "mem_move_local_to_releaseWorkUnit", getModule());
    registerFunction("mem_move_local_to_releaseWorkUnit", fmem_move_local_to_releaseWorkUnit);

    registerFunctions(); //FIXME: do we have to register them every time ?

    if (TheTargetMachine == nullptr) init();

    //Inform the module about the current configuration
    getModule()->setDataLayout  (TheTargetMachine->createDataLayout()           );
    getModule()->setTargetTriple(TheTargetMachine->getTargetTriple().getTriple());

    string ErrStr;
    TheExecutionEngine =
        EngineBuilder(std::unique_ptr<Module>(getModule())).setErrorStr(&ErrStr).create();
    if (TheExecutionEngine == nullptr) {
        fprintf(stderr, "Could not create ExecutionEngine: %s\n",
                ErrStr.c_str());
        exit(1);
    }

    JITEventListener* vtuneProfiler = JITEventListener::createIntelJITEventListener();
    if (vtuneProfiler == nullptr) {
        fprintf(stderr, "Could not create VTune listener\n");
    } else {
        TheExecutionEngine->RegisterJITEventListener(vtuneProfiler);
    }
}

void RawCpuPipelineGen::init(){
    //Get the triplet for current CPU
    auto TargetTriple = sys::getDefaultTargetTriple();

    string ErrStr;
    auto Target = TargetRegistry::lookupTarget(TargetTriple, ErrStr);
    
    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!Target) {
        fprintf(stderr, "Could not create TargetTriple: %s\n",
                ErrStr.c_str());
        exit(1);
    }

    auto CPU      = "generic";//sys::getHostCPUName(); //FIXME: for now it produces faster code... LLVM 6.0.0 improves the scheduler for our system
    // auto CPU      = sys::getHostCPUName(); //FIXME: for now it produces faster code... LLVM 6.0.0 improves the scheduler for our system

    SubtargetFeatures Features;
    StringMap<bool> HostFeatures;
    if (sys::getHostCPUFeatures(HostFeatures)){
      for (auto &F : HostFeatures) Features.AddFeature(F.first(), F.second);
    }

    // std::cout << CPU.str()            << std::endl;
    // std::cout << Features.getString() << std::endl;

    TargetOptions opt;
    auto RM = Optional<Reloc::Model>();
    TheTargetMachine = (LLVMTargetMachine *) Target->createTargetMachine(TargetTriple, CPU, 
                                                    "",//Features.getString(), //FIXME: for now it produces faster code... LLVM 6.0.0 improves the scheduler for our system
                                                    opt, RM, 
                                                    CodeModel::Model::Default, 
                                                    CodeGenOpt::Aggressive);

                                  // // Override function attributes based on CPUStr, FeaturesStr, and command line
                                  // // flags.
                                  // setFunctionAttributes(CPUStr, FeaturesStr, *M);

    Triple ModuleTriple(TargetTriple);
    TargetLibraryInfoImpl TLII(ModuleTriple);
    
    Passes.add(new TargetLibraryInfoWrapperPass(TLII));

    // Add internal analysis passes from the target machine.
    Passes.add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    // FPasses.reset(new legacy::FunctionPassManager(getModule()));
    // FPasses->add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    Pass *TPC = TheTargetMachine->createPassConfig(Passes);
    Passes.add(TPC);

    // if (!NoVerify || VerifyEach)
    //   FPM.add(createVerifierPass()); // Verify that input is correct

    Builder.OptLevel  = 3;
    Builder.SizeLevel = 0;

    Builder.Inliner = createFunctionInliningPass(3, 0, false);

    Builder.DisableUnrollLoops = false;
    Builder.LoopVectorize = true;

    // When #pragma vectorize is on for SLP, do the same as above
    Builder.SLPVectorize = true;

    TheTargetMachine->adjustPassManager(Builder);

  // if (Coroutines)
  //   addCoroutinePassesToExtensionPoints(Builder);

    // Builder.populateFunctionPassManager(*FPasses);
    Builder.populateModulePassManager(Passes);



    // PassBuilder             PB  (TargetMachine);
    // LoopAnalysisManager     LAM (false);
    // FunctionAnalysisManager FAM (false);
    // CGSCCAnalysisManager    CGAM(false);
    // ModuleAnalysisManager   MAM (false);

    // // Register the AA manager first so that our version is the one used.
    // // FAM.registerPass([&] { return std::move(AA); });

    // // Register all the basic analyses with the managers.
    // PB.registerModuleAnalyses   (MAM);
    // PB.registerCGSCCAnalyses    (CGAM);
    // PB.registerFunctionAnalyses (FAM);
    // PB.registerLoopAnalyses     (LAM);
    // PB.crossRegisterProxies     (LAM, FAM, CGAM, MAM);

    // ModulePassManager MPM(false);


    // MPM.run(*(getModule()), MAM);

};

void RawCpuPipelineGen::optimizeModule(Module * M){
    time_block t("Optimization time: ");
    FPasses.reset(new legacy::FunctionPassManager(M));
    FPasses->add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    Builder.populateFunctionPassManager(*FPasses);
    FPasses->add(createLoopDataPrefetchPass());

    FPasses->doInitialization();
    for (Function &F : *M) FPasses->run(F);
    FPasses->doFinalization();
    
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
}

void RawCpuPipelineGen::compileAndLoad(){
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    time_block t(pipName + " C: ");
    // std::cout << pipName << " C" << std::endl;

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code/" + pipName + ".ll", EC, (llvm::sys::fs::OpenFlags) 0); // FIXME: llvm::sys::fs::OpenFlags::F_NONE is the correct one but it gives a compilation error

        getModule()->print(out, nullptr, false, true);
    }
#endif
    
    optimizeModule(getModule());

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code/" + pipName + "_opt.ll", EC, (llvm::sys::fs::OpenFlags) 0); // FIXME: llvm::sys::fs::OpenFlags::F_NONE is the correct one but it gives a compilation error

        getModule()->print(out, nullptr, false, true);
    }
#endif


#ifdef DEBUGCTX
    {
        string assembly;
        {
            raw_string_ostream stream(assembly);
            buffer_ostream ostream(stream);
            
            legacy::PassManager PM;

            // Ask the target to add backend passes as necessary.
            TheTargetMachine->addPassesToEmitFile(PM, ostream, llvm::TargetMachine::CGFT_AssemblyFile, false);

            PM.run(*(getModule()));
        }
        std::ofstream oassembly("generated_code/" + pipName + ".s");
        oassembly << assembly;
    }
#endif
    // JIT the function, returning a function pointer.
    TheExecutionEngine->finalizeObject();
    func = TheExecutionEngine->getPointerToFunction(F);
    assert(func);

    // F->eraseFromParent();
    // F = NULL;
}

void * RawCpuPipelineGen::getCompiledFunction(Function * f){
    return TheExecutionEngine->getPointerToFunction(f);
}