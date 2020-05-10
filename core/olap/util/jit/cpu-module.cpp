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
#include "util/jit/cpu-module.hpp"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
#pragma push_macro("NDEBUG")
#define NDEBUG
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include "llvm/Analysis/TargetTransformInfo.h"
#pragma pop_macro("NDEBUG")
#include "util/timing.hpp"

using namespace llvm;

// LLVMTargetMachine *CpuModule::TheTargetMachine = nullptr;
// legacy::PassManager CpuModule::Passes;
// PassManagerBuilder CpuModule::Builder;

CpuModule::CpuModule(Context *context, std::string pipName)
    : JITModule(context, pipName) {
  string ErrStr;
  TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(getModule()))
                           .setErrorStr(&ErrStr)
                           .create();
  if (TheExecutionEngine == nullptr) {
    fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
    exit(1);
  }

  JITEventListener *vtuneProfiler =
      JITEventListener::createIntelJITEventListener();
  if (vtuneProfiler == nullptr) {
    fprintf(stderr, "Could not create VTune listener\n");
  } else {
    TheExecutionEngine->RegisterJITEventListener(vtuneProfiler);
  }

  // Inform the module about the current configuration
  getModule()->setDataLayout(TheExecutionEngine->getDataLayout());
  getModule()->setTargetTriple(
      TheExecutionEngine->getTargetMachine()->getTargetTriple().getTriple());

  // JITEventListener* gdbDebugger =
  // JITEventListener::createGDBRegistrationListener(); if (gdbDebugger ==
  // nullptr) {
  //     fprintf(stderr, "Could not create GDB listener\n");
  // } else {
  //     TheExecutionEngine->RegisterJITEventListener(gdbDebugger);
  // }
}

void CpuModule::optimizeModule(Module *M) {
  time_block t("Optimization time: ");
  legacy::PassManager Passes;
  llvm::legacy::FunctionPassManager FPasses{M};

  Pass *TPC =
      dynamic_cast<LLVMTargetMachine *>(TheExecutionEngine->getTargetMachine())
          ->createPassConfig(Passes);
  Passes.add(TPC);

  Triple ModuleTriple(
      TheExecutionEngine->getTargetMachine()->getTargetTriple());
  TargetLibraryInfoImpl TLII(ModuleTriple);

  Passes.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(
      TheExecutionEngine->getTargetMachine()->getTargetIRAnalysis()));

  {
    PassManagerBuilder Builder;

    Builder.OptLevel = 3;
    Builder.SizeLevel = 0;

    Builder.Inliner = createFunctionInliningPass(3, 0, false);

    Builder.DisableUnrollLoops = false;
    Builder.LoopVectorize = true;

    // When #pragma vectorize is on for SLP, do the same as above
    Builder.SLPVectorize = true;

    TheExecutionEngine->getTargetMachine()->adjustPassManager(Builder);

    // if (Coroutines)
    //   addCoroutinePassesToExtensionPoints(Builder);

    Builder.populateFunctionPassManager(FPasses);
    Builder.populateModulePassManager(Passes);
  }

  FPasses.add(createTargetTransformInfoWrapperPass(
      TheExecutionEngine->getTargetMachine()->getTargetIRAnalysis()));
  //
  //  Builder.populateFunctionPassManager(FPasses);
  FPasses.add(createLoopDataPrefetchPass());

  FPasses.doInitialization();
  for (Function &F : *M) FPasses.run(F);
  FPasses.doFinalization();

  // Now that we have all of the passes ready, run them.
  Passes.run(*M);
}

class T {
 public:
  std::chrono::milliseconds total;
  T() : total(0) {}

  ~T() { LOG(INFO) << "Total optimization time: " << total.count(); }

} tmp;

void CpuModule::compileAndLoad() {
  time_block t_CnL([&](std::chrono::milliseconds d) { tmp.total += d; });
  time_block t(pipName + " C: ");
  // std::cout << pipName << " C" << std::endl;

  //  auto x = llvm::orc::LLJITBuilder().create();

#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    raw_fd_ostream out("generated_code/" + pipName + ".ll", EC,
                       llvm::sys::fs::OpenFlags::F_None);

    getModule()->print(out, nullptr, false, true);
  }
#endif

  optimizeModule(getModule());

#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    raw_fd_ostream out("generated_code/" + pipName + "_opt.ll", EC,
                       llvm::sys::fs::OpenFlags::F_None);

    getModule()->print(out, nullptr, false, true);
  }
#endif

#ifdef DEBUGCTX
  if (print_generated_code) {
    string assembly;
    {
      raw_string_ostream stream(assembly);
      buffer_ostream ostream(stream);

      legacy::PassManager PM;

      // Ask the target to add backend passes as necessary.
      TheExecutionEngine->getTargetMachine()->addPassesToEmitFile(
          PM, ostream,
#if LLVM_VERSION_MAJOR >= 7
          nullptr,
#endif
          llvm::CGFT_AssemblyFile, false);

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
  // F = nullptr;
}

void *CpuModule::getCompiledFunction(Function *f) const {
  return TheExecutionEngine->getPointerToFunction(f);
}
