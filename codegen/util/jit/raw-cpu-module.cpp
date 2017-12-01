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
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/CodeGen/TargetPassConfig.h"

#include "util/jit/raw-cpu-module.hpp"

LLVMTargetMachine * RawCpuModule::TheTargetMachine = nullptr;
legacy::PassManager RawCpuModule::Passes                    ;
PassManagerBuilder  RawCpuModule::Builder                   ;

RawCpuModule::RawCpuModule(RawContext * context, std::string pipName):
    RawModule(context, pipName){

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

    // JITEventListener* gdbDebugger = JITEventListener::createGDBRegistrationListener();
    // if (gdbDebugger == nullptr) {
    //     fprintf(stderr, "Could not create GDB listener\n");
    // } else {
    //     TheExecutionEngine->RegisterJITEventListener(gdbDebugger);
    // }
}

void RawCpuModule::init(){
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
}

void RawCpuModule::optimizeModule(Module * M){
    time_block t("Optimization time: ");

    llvm::legacy::FunctionPassManager FPasses{M};
    FPasses.add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    Builder.populateFunctionPassManager(FPasses);
    FPasses.add(createLoopDataPrefetchPass());

    FPasses.doInitialization();
    for (Function &F : *M) FPasses.run(F);
    FPasses.doFinalization();
    
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
}

void RawCpuModule::compileAndLoad(){
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
    // func = TheExecutionEngine->getPointerToFunction(F);
    // assert(func);

    // F->eraseFromParent();
    // F = NULL;
}

void * RawCpuModule::getCompiledFunction(Function * f) const{
    return TheExecutionEngine->getPointerToFunction(f);
}