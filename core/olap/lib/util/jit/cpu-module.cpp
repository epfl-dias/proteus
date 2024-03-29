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
#include "cpu-module.hpp"

#pragma push_macro("NDEBUG")
#define NDEBUG
#if LLVM_VERSION_MAJOR >= 13
#define dumpDispatchInfo(x) (5)
#else
#define dumpDispatchInfo(x, y) (5)
#endif
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
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
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>
#pragma pop_macro("NDEBUG")

#include <platform/threadpool/threadpool.hpp>
#include <platform/util/timing.hpp>

using namespace llvm;

CpuModule::CpuModule(Context *context, std::string pipName)
    : JITModule(context, pipName) {
  string ErrStr;
  TheExecutionEngine = EngineBuilder(std::unique_ptr<Module>(getModule()))
                           .setErrorStr(&ErrStr)
                           .setMCPU(llvm::sys::getHostCPUName())
                           .setOptLevel(CodeGenOpt::Aggressive)
                           .setCodeModel(llvm::CodeModel::Model::Medium)
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
    time_block t(TimeRegistry::Key{"Optimization phase"});
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

Expected<llvm::orc::ThreadSafeModule> printIR(orc::ThreadSafeModule module,
                                              const std::string &suffix = "") {
  module.withModuleDo([&suffix](Module &m) {
    std::error_code EC;
    raw_fd_ostream out("generated_code/" + m.getName().str() + suffix + ".ll",
                       EC, llvm::sys::fs::OpenFlags::OF_None);
    m.print(out, nullptr, false, true);
  });
  return std::move(module);
}

extern "C" void __register_frame(void *);
extern "C" void __deregister_frame(void *);

namespace proteus {

static const char *processFDE(const char *Entry, bool isDeregister) {
  const char *P = Entry;
  uint32_t Length = *((const uint32_t *)P);
  P += 4;
  uint32_t Offset = *((const uint32_t *)P);
  if (Offset != 0) {
    if (isDeregister)
      __deregister_frame(const_cast<char *>(Entry));
    else
      __register_frame(const_cast<char *>(Entry));
  }
  return P + Length;
}

// This implementation handles frame registration for local targets.
// Memory managers for remote targets should re-implement this function
// and use the LoadAddr parameter.
void registerEHFramesInProcess(uint8_t *Addr, size_t Size) {
  // On OS X OS X __register_frame takes a single FDE as an argument.
  // See http://lists.llvm.org/pipermail/llvm-dev/2013-April/061737.html
  // and projects/libunwind/src/UnwindLevel1-gcc-ext.c.
  const char *P = (const char *)Addr;
  const char *End = P + Size;
  do {
    P = processFDE(P, false);
  } while (P != End);
}

void deregisterEHFramesInProcess(uint8_t *Addr, size_t Size) {
  const char *P = (const char *)Addr;
  const char *End = P + Size;
  do {
    P = processFDE(P, true);
  } while (P != End);
}

/**
 * Patches the default SectionMemoryManager to silence "FDE is really a CIE".
 *
 * LLVM bug link: https://bugs.llvm.org/show_bug.cgi?id=44074
 *
 * Details:
 *  LLVM detects which libunwind library is used based on the __APPLE__ flag.
 *  If __APPLE__ is set, it assumes the LLVM's lib, otherwise it assumes
 *  the GCC lib. We use LLVM's lib, even on Linux, for which case, LLVM
 *  peaks the wrong path and causes the JITer to populate the log with
 *  "libunwind: __unw_add_dynamic_fde: bad fde: FDE is really a CIE"
 *  messages.
 *
 *  This patch overwrites the default path, silencing the issue.
 *  Please keep an eye for when the default path is fixed and/or for any
 *  updates on the overwritten functions.
 */
class SectionMemoryManager : public llvm::SectionMemoryManager {
  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {
    proteus::registerEHFramesInProcess(Addr, Size);
    EHFrames.push_back({Addr, Size});
  }

  void deregisterEHFrames() override {
    for (auto &Frame : EHFrames)
      proteus::deregisterEHFramesInProcess(Frame.Addr, Frame.Size);
    EHFrames.clear();
  }
};
}  // namespace proteus

class JITer_impl {
 public:
  ::ThreadPool pool;

  DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  llvm::orc::ThreadSafeContext Ctx;

  PassConfiguration PassConf;

#if LLVM_VERSION_MAJOR >= 13
  llvm::orc::ExecutionSession ES{
      llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create())};
#else
  llvm::orc::ExecutionSession ES;
#endif
  llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
  llvm::orc::IRCompileLayer CompileLayer;
  llvm::orc::IRTransformLayer PrintOptimizedIRLayer;
  llvm::orc::IRTransformLayer TransformLayer;
  llvm::orc::IRTransformLayer PrintGeneratedIRLayer;

  llvm::orc::JITDylib &MainJD;

  std::mutex vtuneLock;
  JITEventListener *vtuneProfiler;

 public:
  ~JITer_impl() {
    if (auto Err = ES.endSession()) ES.reportError(std::move(Err));
  }

  explicit JITer_impl(
      llvm::orc::JITTargetMachineBuilder JTMB =
          llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost())
              .setCodeGenOptLevel(CodeGenOpt::Aggressive)
              .setCodeModel(llvm::CodeModel::Model::Large))
      : pool(false, std::thread::hardware_concurrency()),
        DL(llvm::cantFail(JTMB.getDefaultDataLayoutForTarget())),
        Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        PassConf(llvm::cantFail(JTMB.createTargetMachine())),
        ObjectLayer(
            ES,
            []() { return std::make_unique<proteus::SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer,
                     std::make_unique<llvm::orc::ConcurrentIRCompiler>(JTMB)),
        PrintOptimizedIRLayer(
            ES, CompileLayer,
            [](llvm::orc::ThreadSafeModule TSM,
               const llvm::orc::MaterializationResponsibility &R)
                -> Expected<llvm::orc::ThreadSafeModule> {
              if (print_generated_code) return printIR(std::move(TSM), "_opt");
              return std::move(TSM);
            }),
        TransformLayer(
            ES, PrintOptimizedIRLayer,
            [JTMB = std::move(JTMB)](
                llvm::orc::ThreadSafeModule TSM,
                const llvm::orc::MaterializationResponsibility &R) mutable
            -> Expected<llvm::orc::ThreadSafeModule> {
              auto TM = JTMB.createTargetMachine();
              if (!TM) return TM.takeError();
              return optimizeModule(std::move(TSM), std::move(TM.get()));
            }),
        PrintGeneratedIRLayer(
            ES, TransformLayer,
            [](llvm::orc::ThreadSafeModule TSM,
               const llvm::orc::MaterializationResponsibility &R)
                -> Expected<llvm::orc::ThreadSafeModule> {
              if (print_generated_code) return printIR(std::move(TSM));
              return std::move(TSM);
            }),
        MainJD(llvm::cantFail(ES.createJITDylib("main"))),
        vtuneProfiler(JITEventListener::createIntelJITEventListener()) {
    if (vtuneProfiler == nullptr) {
      LOG(WARNING) << "Could not create VTune listener";
    } else {
      ObjectLayer.setNotifyLoaded(
          [this](llvm::orc::MaterializationResponsibility &R,
                 const object::ObjectFile &Obj,
                 const RuntimeDyld::LoadedObjectInfo &loi) {
            std::scoped_lock<std::mutex> lock{vtuneLock};
            cantFail(R.withResourceKeyDo([&](llvm::orc::ResourceKey k) {
              vtuneProfiler->notifyObjectLoaded(k, Obj, loi);
            }));
          });
    }

    //    ObjectLayer.setNotifyEmitted(
    //        [](llvm::orc::VModuleKey k, std::unique_ptr<MemoryBuffer> mb) {
    //          LOG(INFO) << "Emitted " << k << " " << mb.get();
    //        });

#if LLVM_VERSION_MAJOR >= 13
    ES.setDispatchTask([&p = pool](std::unique_ptr<llvm::orc::Task> T) {
      p.enqueue([UnownedT = std::move(T)]() mutable { UnownedT->run(); });
    });
#else
    ES.setDispatchMaterialization(
        [&p = pool](
            std::unique_ptr<llvm::orc::MaterializationUnit> MU,
            std::unique_ptr<llvm::orc::MaterializationResponsibility> MR) {
          auto SharedMU =
              std::shared_ptr<llvm::orc::MaterializationUnit>(std::move(MU));
          p.enqueue(
              [SharedMU](
                  std::unique_ptr<llvm::orc::MaterializationResponsibility>
                      MRinner) { SharedMU->materialize(std::move(MRinner)); },
              std::move(MR));
        });
#endif

    MainJD.addGenerator(llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            this->DL.getGlobalPrefix())));
  }

  const DataLayout &getDataLayout() const { return DL; }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  void addModule(llvm::orc::ThreadSafeModule M) {
    llvm::cantFail(PrintGeneratedIRLayer.add(MainJD, std::move(M)));
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

void CpuModule::compileAndLoad() {
  time_block t(TimeRegistry::Key{"Compile and Load (CPU)"});

  llvm::orc::ThreadSafeModule ptr;
  {
    time_block tCopy{TimeRegistry::Key{"Module Copying"}};
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

  ::ThreadPool::getInstance().enqueue(
      [name = getModule()->getName()]() { getJiter().p_impl->lookup(name); });
  //#ifdef DEBUGCTX
  //  if (print_generated_code) {
  //    string assembly;
  //    {
  //      raw_string_ostream stream(assembly);
  //      buffer_ostream ostream(stream);
  //
  //      legacy::PassManager PM;
  //
  //      // Ask the target to add backend passes as necessary.
  //      TheExecutionEngine->getTargetMachine()->addPassesToEmitFile(
  //          PM, ostream,
  //#if LLVM_VERSION_MAJOR >= 7
  //          nullptr,
  //#endif
  //          llvm::CGFT_AssemblyFile, false);
  //
  //      PM.run(*(getModule()));
  //    }
  //    std::ofstream oassembly("generated_code/" + pipName + ".s");
  //    oassembly << assembly;
  //  }
  //#endif
  //  for (Function &f : *getModule()) {
  //    if (!f.isDeclaration()) {
  //      auto addr = getJiter().p_impl->lookup(f.getName());
  //      LOG(INFO) << f.getName().str() << " " << (void *)addr.getAddress();
  //    }
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
