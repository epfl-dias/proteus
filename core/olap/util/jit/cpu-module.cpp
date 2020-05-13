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

#include <threadpool/threadpool.hpp>

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

class PassConfiguration {
 public:
  std::unique_ptr<TargetMachine> JTM;
  Triple ModuleTriple;
  TargetLibraryInfoImpl TLII;
  legacy::PassManager Passes;
  PassManagerBuilder Builder;
  ImmutablePass *TTIPass;
  FunctionPass *PrefetchPass;
  std::mutex m;

 public:
  explicit PassConfiguration(std::unique_ptr<TargetMachine> TM)
      : JTM(std::move(TM)),
        ModuleTriple(JTM->getTargetTriple()),
        TLII(ModuleTriple),
        TTIPass(
            createTargetTransformInfoWrapperPass(JTM->getTargetIRAnalysis())),
        PrefetchPass(createLoopDataPrefetchPass()) {
    time_block trun(TimeRegistry::Key{"Optimization avoidable phase"});

    Pass *TPC =
        dynamic_cast<LLVMTargetMachine *>(JTM.get())->createPassConfig(Passes);
    Passes.add(TPC);

    Passes.add(new TargetLibraryInfoWrapperPass(TLII));

    // Add internal analysis passes from the target machine.
    Passes.add(
        createTargetTransformInfoWrapperPass(JTM->getTargetIRAnalysis()));

    Builder.OptLevel = 3;
    Builder.SizeLevel = 0;

    Builder.Inliner = createFunctionInliningPass(3, 0, false);

    Builder.DisableUnrollLoops = false;
    Builder.LoopVectorize = true;

    Builder.SLPVectorize = true;

    JTM->adjustPassManager(Builder);

    Builder.populateModulePassManager(Passes);
  }
};

llvm::orc::ThreadSafeModule optimizeModule(llvm::orc::ThreadSafeModule TSM,
                                           std::unique_ptr<TargetMachine> TM) {
  TSM.withModuleDo([&TM](Module &M) {
    time_block t("Optimization time: ",
                 TimeRegistry::Key{"Optimization phase"});
    // if (Coroutines)
    //   addCoroutinePassesToExtensionPoints(Builder);

    PassConfiguration pc{std::move(TM)};

    llvm::legacy::FunctionPassManager FPasses{&M};

    pc.Builder.populateFunctionPassManager(FPasses);

    FPasses.add(pc.TTIPass);
    FPasses.add(pc.PrefetchPass);

    {
      time_block trun(TimeRegistry::Key{"Optimization run phase"});

      FPasses.doInitialization();
      for (Function &F : M) FPasses.run(F);
      FPasses.doFinalization();

      // Now that we have all of the passes ready, run them.
      pc.Passes.run(M);
    }
  });
  return TSM;
}

class JITer_impl {
 public:
  ::ThreadPool pool;

  DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  llvm::orc::ThreadSafeContext Ctx;

  PassConfiguration PassConf;

  llvm::orc::ExecutionSession ES;
  llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
  llvm::orc::IRCompileLayer CompileLayer;
  llvm::orc::IRTransformLayer TransformLayer;

  llvm::orc::JITDylib &MainJD;

 public:
  explicit JITer_impl(
      llvm::orc::JITTargetMachineBuilder JTMB =
          llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost())
              .setCodeGenOptLevel(CodeGenOpt::Aggressive))
      : pool(false, std::thread::hardware_concurrency()),
        DL(llvm::cantFail(JTMB.getDefaultDataLayoutForTarget())),
        Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        PassConf(llvm::cantFail(JTMB.createTargetMachine())),
        ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer,
                     std::make_unique<llvm::orc::ConcurrentIRCompiler>(JTMB)),
        TransformLayer(
            ES, CompileLayer,
            [JTMB = std::move(JTMB)](
                llvm::orc::ThreadSafeModule TSM,
                const llvm::orc::MaterializationResponsibility &R) mutable
            -> Expected<llvm::orc::ThreadSafeModule> {
              auto TM = JTMB.createTargetMachine();
              if (!TM) return TM.takeError();
              return optimizeModule(std::move(TSM), std::move(TM.get()));
            }),
        MainJD(ES.createJITDylib("main")) {
    ObjectLayer.setNotifyEmitted(
        [](llvm::orc::VModuleKey k, std::unique_ptr<MemoryBuffer> mb) {
          LOG(INFO) << "Emitted " << k << " "
                    << mb.get();  //->getBufferIdentifier().str();
        });

    ES.setDispatchMaterialization(
        [&p = pool](llvm::orc::JITDylib &JD,
                    std::unique_ptr<llvm::orc::MaterializationUnit> MU) {
          p.enqueue([&JD, MU = std::move(MU)]() { MU->doMaterialize(JD); });
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
    llvm::cantFail(TransformLayer.add(MainJD, std::move(M)));
  }

  JITEvaluatedSymbol lookup(StringRef Name) {
    return llvm::cantFail(ES.lookup({&MainJD}, Mangle(Name.str())));
  }
};

LLVMContext &JITer::getContext() { return p_impl->getContext(); }
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
auto &getJiter() {
  static JITer jiter;
  return jiter;
}

// void CpuModule::optimizeModule(Module *M) {
//  auto JTM =
//      dynamic_cast<LLVMTargetMachine
//      *>(TheExecutionEngine->getTargetMachine());
//  llvm::orc::ThreadSafeModule TSM(std::unique_ptr<Module>(M),
//                                  getJiter().p_impl->Ctx);
//  ::optimizeModule(std::move(TSM), *JTM);
//}

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

  //  optimizeModule(getModule());
  llvm::orc::ThreadSafeModule ptr;
  {
    time_block tCopy{"Tcopy: ", TimeRegistry::Key{"Module Copying"}};
    // FIXME: we have to copy the module into a different context, as
    //  asynchronous consumers still have copies all over the place.
    //  as soon as we start preventing these issues, we should remove
    //  this slow copying process.
    Module &M = *getModule();
    SmallVector<char, 1> ClonedModuleBuffer;

    {
      BitcodeWriter BCWriter(ClonedModuleBuffer);

      BCWriter.writeModule(M);
      BCWriter.writeSymtab();
      BCWriter.writeStrtab();
    }

    MemoryBufferRef ClonedModuleBufferRef(
        StringRef(ClonedModuleBuffer.data(), ClonedModuleBuffer.size()),
        "cloned module buffer");
    llvm::orc::ThreadSafeContext NewTSCtx(std::make_unique<LLVMContext>());

    auto ClonedModule = cantFail(
        parseBitcodeFile(ClonedModuleBufferRef, *NewTSCtx.getContext()));
    ClonedModule->setModuleIdentifier(M.getName());
    ptr = llvm::orc::ThreadSafeModule(std::move(ClonedModule),
                                      std::move(NewTSCtx));
  }

  // Change the source file name, otherwise the main function conflicts with the
  // source file name and it does not get emitted
  ptr.withModuleDo([pipName = this->pipName](Module &nm) {
    nm.setSourceFileName("this_" + pipName);
  });
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
  time_block t(TimeRegistry::Key{"Compile and Load (CPU, waiting)"});
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
