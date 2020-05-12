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

#pragma push_macro("NDEBUG")
#define NDEBUG
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/Utils/Cloning.h>
#pragma pop_macro("NDEBUG")

#include "util/timing.hpp"

using namespace llvm;

CpuModule::CpuModule(Context *context, std::string pipName)
    : JITModule(context, pipName) {
  string ErrStr;
  TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(getModule()))
                           .setErrorStr(&ErrStr)
                           .setMCPU(llvm::sys::getHostCPUName())
                           .setOptLevel(CodeGenOpt::Aggressive)
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
  //  getModule()->setDataLayout(TheExecutionEngine->getDataLayout());
  //  getModule()->setTargetTriple(
  //      TheExecutionEngine->getTargetMachine()->getTargetTriple().getTriple());

  // JITEventListener* gdbDebugger =
  // JITEventListener::createGDBRegistrationListener(); if (gdbDebugger ==
  // nullptr) {
  //     fprintf(stderr, "Could not create GDB listener\n");
  // } else {
  //     TheExecutionEngine->RegisterJITEventListener(gdbDebugger);
  // }
}

class JITer_impl;

class JITer {
 public:
  std::unique_ptr<JITer_impl> p_impl;

 public:
  JITer();
  ~JITer();

  LLVMContext &getContext();
};

JITer::JITer() : p_impl(std::make_unique<JITer_impl>()) {}

JITer::~JITer() = default;

class JITer_impl {
 public:
  DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  llvm::orc::ThreadSafeContext Ctx;

  llvm::orc::ExecutionSession ES;
  llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
  llvm::orc::IRCompileLayer CompileLayer;
  //  llvm::orc::IRTransformLayer TransformLayer;

  llvm::orc::JITDylib &MainJD;

 public:
  explicit JITer_impl(
      llvm::orc::JITTargetMachineBuilder JTMB =
          llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost())
              .setCodeGenOptLevel(CodeGenOpt::Aggressive))
      : DL(llvm::cantFail(JTMB.getDefaultDataLayoutForTarget())),
        Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(
            ES, ObjectLayer,
            std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
        MainJD(ES.createJITDylib("main")) {
    CompileLayer.setCloneToNewContextOnEmit(true);
    ObjectLayer.setNotifyEmitted(
        [](llvm::orc::VModuleKey k, std::unique_ptr<MemoryBuffer> mb) {
          LOG(INFO) << "Emitted " << k << " "
                    << mb.get();  //->getBufferIdentifier().str();
        });
    MainJD.addGenerator(llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            this->DL.getGlobalPrefix())));
  }

  const DataLayout &getDataLayout() const { return DL; }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  void addModule(std::unique_ptr<Module> M) {
    addModule(llvm::orc::ThreadSafeModule(std::move(M), Ctx));
  }

  void addModule(llvm::orc::ThreadSafeModule M) {
    llvm::cantFail(CompileLayer.add(MainJD, std::move(M)));
  }

  JITEvaluatedSymbol lookup(StringRef Name) {
    return llvm::cantFail(ES.lookup({&MainJD}, Mangle(Name.str())));
  }
};

LLVMContext &JITer::getContext() { return p_impl->getContext(); }

auto &getJiter() {
  static JITer jiter;
  return jiter;
}

void CpuModule::optimizeModule(Module *M) {
  time_block t("Optimization time: ");

  auto JTM =
      dynamic_cast<LLVMTargetMachine *>(TheExecutionEngine->getTargetMachine());
  legacy::PassManager Passes;
  llvm::legacy::FunctionPassManager FPasses{M};

  Pass *TPC = JTM->createPassConfig(Passes);
  Passes.add(TPC);

  Triple ModuleTriple(JTM->getTargetTriple());
  TargetLibraryInfoImpl TLII(ModuleTriple);

  Passes.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(JTM->getTargetIRAnalysis()));

  {
    PassManagerBuilder Builder;

    Builder.OptLevel = 3;
    Builder.SizeLevel = 0;

    Builder.Inliner = createFunctionInliningPass(3, 0, false);

    Builder.DisableUnrollLoops = false;
    Builder.LoopVectorize = true;

    // When #pragma vectorize is on for SLP, do the same as above
    Builder.SLPVectorize = true;

    JTM->adjustPassManager(Builder);

    // if (Coroutines)
    //   addCoroutinePassesToExtensionPoints(Builder);

    Builder.populateFunctionPassManager(FPasses);
    Builder.populateModulePassManager(Passes);
  }

  FPasses.add(createTargetTransformInfoWrapperPass(JTM->getTargetIRAnalysis()));
  //
  //  Builder.populateFunctionPassManager(FPasses);
  FPasses.add(createLoopDataPrefetchPass());

  FPasses.doInitialization();
  for (Function &F : *M) FPasses.run(F);
  FPasses.doFinalization();

  // Now that we have all of the passes ready, run them.
  Passes.run(*M);
}

void CpuModule::compileAndLoad() {
  time_block t(pipName + " C: ", TimeRegistry::Key{"Compile and Load (CPU)"});
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
  auto tsmw = llvm::orc::ThreadSafeModule(llvm::CloneModule(*getModule()),
                                          getJiter().p_impl->Ctx);
  auto ptr = llvm::orc::cloneToNewContext(tsmw);

  // llvm::CloneModule(*getModule());
  ptr.withModuleDo([pipName = this->pipName](Module &nm) {
    nm.setSourceFileName("this_" + pipName);
  });
  //  ptr->dump();
  getJiter().p_impl->addModule(std::move(ptr));

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
  //  TheExecutionEngine->finalizeObject();
  // func = TheExecutionEngine->getPointerToFunction(F);
  // assert(func);

  // F->eraseFromParent();
  // F = nullptr;

  //  for (Function &ftmp : *getModule()) {
  //    if (!ftmp.isDeclaration()) getCompiledFunction(&ftmp);
  //  }
}

void *CpuModule::getCompiledFunction(std::string str) const {
  return (void *)getJiter().p_impl->lookup(str).getAddress();
}

const llvm::DataLayout &CpuModule::getDL() {
  return getJiter().p_impl->getDataLayout();
}

void *CpuModule::getCompiledFunction(Function *f) const {
  //  time_block t_CnL([&](std::chrono::milliseconds d) { tmp.total += d; });
  //    assert(f);
  //    static std::mutex m;
  //    std::scoped_lock<std::mutex> lock{m};
  //    return (void *)getJiter().p_impl->lookup(f->getName()).getAddress();
  assert(f);
  return getCompiledFunction(f->getName().str());
  //  return TheExecutionEngine->getPointerToFunction(f);
}
