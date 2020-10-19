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
#include "gpu-module.hpp"

#include <dlfcn.h>
#pragma push_macro("NDEBUG")
#define NDEBUG
#define dumpDispatchInfo(x, y) (5)
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
#include <llvm/ExecutionEngine/Orc/OrcABISupport.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/Cloning.h>
#pragma pop_macro("NDEBUG")

#include "topology/affinity_manager.hpp"
#include "util/timing.hpp"

void initializeModule(CUmodule &cudaModule);

using namespace llvm;

#ifndef NVPTX_MAX_REGS
#define NVPTX_MAX_REGS (32)
#endif

constexpr uint64_t nvptx_max_regs = NVPTX_MAX_REGS;

LLVMTargetMachine *GpuModule::TheTargetMachine = nullptr;
legacy::PassManager GpuModule::Passes;
PassManagerBuilder GpuModule::Builder;

GpuModule::GpuModule(Context *context, std::string pipName)
    : JITModule(context, pipName) {
  uint32_t gpu_cnt = topology::getInstance().getGpuCount();
  cudaModule = (CUmodule *)malloc(gpu_cnt * sizeof(CUmodule));

  if (TheTargetMachine == nullptr) init();

  // Inform the module about the current configuration
  getModule()->setDataLayout(TheTargetMachine->createDataLayout());
  getModule()->setTargetTriple(TheTargetMachine->getTargetTriple().getTriple());

  // string ErrStr;
  // TheExecutionEngine =
  //     EngineBuilder(std::unique_ptr<Module>(getModule())).setErrorStr(&ErrStr).create();
  // if (TheExecutionEngine == nullptr) {
  //     fprintf(stderr, "Could not create ExecutionEngine: %s\n",
  //             ErrStr.c_str());
  //     exit(1);
  // }

  // // JITEventListener* vtuneProfiler =
  // JITEventListener::createIntelJITEventListener();
  // // if (vtuneProfiler == nullptr) {
  // //     fprintf(stderr, "Could not create VTune listener\n");
  // // } else {
  // //     TheExecutionEngine->RegisterJITEventListener(vtuneProfiler);
  // // }

  // JITEventListener* gdbDebugger =
  // JITEventListener::createGDBRegistrationListener(); if (gdbDebugger ==
  // nullptr) {
  //     fprintf(stderr, "Could not create GDB listener\n");
  // } else {
  //     TheExecutionEngine->RegisterJITEventListener(gdbDebugger);
  // }
}

auto getGPU() {
  assert(topology::getInstance().getGpuCount() > 0);
  int dev = 0;
  gpu_run(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp;
  gpu_run(cudaGetDeviceProperties(&deviceProp, dev));
  return "sm_" + std::to_string(deviceProp.major * 10 + deviceProp.minor);
}

void GpuModule::init() {
  // Get the triplet for GPU
  std::string TargetTriple("nvptx64-nvidia-cuda");

  string ErrStr;
  auto Target = TargetRegistry::lookupTarget(TargetTriple, ErrStr);

  // Print an error and exit if we couldn't find the requested target.
  // This generally occurs if we've forgotten to initialise the
  // TargetRegistry or we have a bogus target triple.
  if (!Target) {
    fprintf(stderr, "Could not create TargetTriple: %s\n", ErrStr.c_str());
    exit(1);
  }

  auto GPU = getGPU();

  llvm::TargetOptions opt;
  opt.DisableIntegratedAS = 1;
  opt.MCOptions.ShowMCEncoding = 1;
  opt.MCOptions.MCUseDwarfDirectory = 1;
  // opt.MCOptions.AsmVerbose            = 1;
  opt.MCOptions.PreserveAsmComments = 1;

  auto RM = llvm::Optional<llvm::Reloc::Model>();
  TheTargetMachine = (llvm::LLVMTargetMachine *)Target->createTargetMachine(
      TargetTriple, GPU,
      "+ptx64",  // PTX 6.4 + Scoped Atomics
      //"+ptx60,+satom", //for V100
      opt, RM,
      llvm::Optional<llvm::CodeModel::Model>{},  // CodeModel::Model::Default,
      llvm::CodeGenOpt::Aggressive);

  // // Override function attributes based on CPUStr, FeaturesStr, and command
  // line
  // // flags.
  // setFunctionAttributes(CPUStr, FeaturesStr, *M);

  // TheTargetMachine->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
  //                        "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
  //                        "v64:64:64-v128:128:128-n16:32:64");
  llvm::Triple ModuleTriple(TargetTriple);
  llvm::TargetLibraryInfoImpl TLII(ModuleTriple);

  Passes.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(
      TheTargetMachine->getTargetIRAnalysis()));

  // FPasses.reset(new llvm::legacy::FunctionPassManager(getModule()));
  // FPasses->add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

  llvm::Pass *TPC = TheTargetMachine->createPassConfig(Passes);
  Passes.add(TPC);

  // if (!NoVerify || VerifyEach)
  //   FPM.add(createVerifierPass()); // Verify that input is correct

  Builder.OptLevel = 3;
  Builder.SizeLevel = 0;

  Builder.Inliner = llvm::createFunctionInliningPass(3, 0, false);

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

class GpuPassConfiguration {
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
  explicit GpuPassConfiguration(std::unique_ptr<TargetMachine> TM)
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

void GpuModule::optimizeModule(llvm::Module *M) {
  time_block t("Optimization time: ");

  llvm::legacy::FunctionPassManager FPasses{M};
  FPasses.add(createTargetTransformInfoWrapperPass(
      TheTargetMachine->getTargetIRAnalysis()));

  Builder.populateFunctionPassManager(FPasses);

  FPasses.doInitialization();
  for (llvm::Function &F : *M) FPasses.run(F);
  FPasses.doFinalization();

  // Now that we have all of the passes ready, run them.
  Passes.run(*M);
}

constexpr size_t BUFFER_SIZE = 8192;
static char error_log[BUFFER_SIZE];
static char info_log[BUFFER_SIZE];

std::pair<char *, char *> discover_bc() {
  // FIXME: should use a loop instead of nested ifs... also, we should cache
  // the result per sm
  // Compute symbol name for one of the GPU architecture, and use that to
  // retrieve the corresponding binary blob.
  // FIXME: We assume compute and arch are equal!
  // FIXME: Add error handling, we are generating a symbol name and
  //        assuming it is going to be available...
  int dev = 0;
  gpu_run(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp;
  gpu_run(cudaGetDeviceProperties(&deviceProp, dev));
  auto sm_code = std::to_string(deviceProp.major * 10 + deviceProp.minor);

  auto sim_prefix =
      "_binary_buffer_manager_cuda_nvptx64_nvidia_cuda_sm_" + sm_code + "_bc_";
  auto sim_start = sim_prefix + "start";
  auto sim_end = sim_prefix + "end";

  void *handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
  assert(handle);

  auto _binary_buffer_manager_cubin_start =
      (char *)dlsym(handle, sim_start.c_str());
  auto _binary_buffer_manager_cubin_end =
      (char *)dlsym(handle, sim_end.c_str());

  assert(_binary_buffer_manager_cubin_start && "cubin start symbol not found!");
  assert(_binary_buffer_manager_cubin_end && "cubin end symbol not found!");

  LOG(INFO) << "[Load CUBIN blob: ] sim_start: " << sim_start
            << ", sim_end: " << sim_end;
  LOG(INFO) << "[Load CUBIN blob: ] start: "
            << (void *)_binary_buffer_manager_cubin_start
            << ", end: " << (void *)_binary_buffer_manager_cubin_end;

  return {_binary_buffer_manager_cubin_start, _binary_buffer_manager_cubin_end};
}

auto createModule(LLVMContext &llvmContext) {
  auto binary_bc = discover_bc();
  auto start = (const char *)binary_bc.first;
  auto end = (const char *)binary_bc.second;
  size_t size = end - start;
  auto mb2 =
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef{start, size}, "", false);
  assert(mb2);
  llvm::SMDiagnostic error;

  auto mod = llvm::parseIR(mb2->getMemBufferRef(), error, llvmContext);
  assert(mod);

  return mod;
}

auto getModuleRef(LLVMContext &llvmContext) {
  static auto mod{createModule(llvmContext)};
  if (&llvmContext != &mod->getContext()) mod = createModule(llvmContext);
  return llvm::CloneModule(*mod);
}

#include <threadpool/threadpool.hpp>

Expected<llvm::orc::JITTargetMachineBuilder> detectGPU() {
  llvm::orc::JITTargetMachineBuilder TMBuilder(Triple{"nvptx64-nvidia-cuda"});

  //  llvm::StringMap<bool> FeatureMap;
  //  //  llvm::sys::getHostCPUFeatures(FeatureMap);
  //  FeatureMap["+ptx64"] = true;
  //  for (auto &Feature : FeatureMap)
  TMBuilder.getFeatures().AddFeature("+ptx64", true);

  TMBuilder.setCPU(getGPU());

  return {TMBuilder};
}

class GPUJITer_impl;

class GPUJITer {
 public:
  std::unique_ptr<GPUJITer_impl> p_impl;

 public:
  GPUJITer();
  ~GPUJITer();

  LLVMContext &getContext();
};

GPUJITer::GPUJITer() : p_impl(std::make_unique<GPUJITer_impl>()) {}

GPUJITer::~GPUJITer() = default;

class NVPTXLinkLayer final : public llvm::orc::ObjectLayer {
 public:
  static auto gpuMangle(StringRef Name, decltype(topology::gpunode::id) id) {
    return ("_gpu" + std::to_string(id) + "_" + Name).str();
  }

  auto gpuMangle(StringRef Name) {
    return gpuMangle(Name, topology::getInstance().getActiveGpu().id);
  }

  explicit NVPTXLinkLayer(llvm::orc::ExecutionSession &ES) : ObjectLayer(ES) {}
  //  using ObjectLayer::ObjectLayer;

  template <typename F>
  void setNotifyEmitted(F &&f) {}

  void emit(llvm::orc::MaterializationResponsibility R,
            std::unique_ptr<MemoryBuffer> Obj) override {
    //    llvm::orc::LocalJITCompileCallbackManager<llvm::orc::OrcGenericABI>::Create();

    CUmodule cudaModule[topology::getInstance().getGpuCount()];

    {
      time_block t("TloadModule: ", TimeRegistry::Key{"Load GPU module"});
      // FIXME: Currently requires all GPUs to be of the same compute
      //  capability, or to be more precise, all of them to be compatible with
      //  the CC of the current device

      constexpr size_t opt_size = 8;
      CUjit_option options[opt_size];
      void *values[opt_size];

      options[0] = CU_JIT_TARGET_FROM_CUCONTEXT;
      values[0] = nullptr;
      options[1] = CU_JIT_ERROR_LOG_BUFFER;
      values[1] = (void *)error_log;
      options[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      values[2] = (void *)BUFFER_SIZE;
      options[3] = CU_JIT_MAX_REGISTERS;
      values[3] = (void *)nvptx_max_regs;
      options[4] = CU_JIT_INFO_LOG_BUFFER;
      values[4] = (void *)info_log;
      options[5] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      values[5] = (void *)BUFFER_SIZE;
      options[6] = CU_JIT_LOG_VERBOSE;
      values[6] = (void *)1;
      options[7] = CU_JIT_OPTIMIZATION_LEVEL;
      values[7] = (void *)4;

      {
        for (const auto &gpu : topology::getInstance().getGpus()) {
          set_device_on_scope d(gpu);

          auto x =
              (cuModuleLoadDataEx(&cudaModule[gpu.id], Obj->getBuffer().data(),
                                  opt_size, options, values));

          if (info_log[0] != '\0') LOG(INFO) << info_log;
          if (x != CUDA_SUCCESS) {
            LOG(INFO) << error_log;
            gpu_run(x);
          }
          initializeModule(cudaModule[gpu.id]);
        }
      }
    }

    std::vector<std::pair<std::string, JITSymbolFlags>> reqKernels;

    for (auto &r : R.getSymbols()) {
      reqKernels.emplace_back(*r.getFirst(), r.getSecond());

      CUfunction func = nullptr;
      gpu_run(cuModuleGetFunction(
          &func, cudaModule[topology::getInstance().getGpus()[0].id],
          (*r.getFirst()).str().c_str()));

      JITEvaluatedSymbol sym{pointerToJITTargetAddress(func), r.getSecond()};
      cantFail(R.notifyResolved({{r.getFirst(), sym}}));
    }

    // merging with above invalidates the loop iterator
    for (const auto &s : reqKernels) {
      for (const auto &gpu : topology::getInstance().getGpus()) {
        auto kernel_name = gpuMangle(s.first, gpu.id);
        auto intern = getExecutionSession().intern(kernel_name);

        cantFail(R.defineMaterializing({{intern, s.second}}));

        CUfunction func = nullptr;
        gpu_run(
            cuModuleGetFunction(&func, cudaModule[gpu.id], s.first.c_str()));
        JITEvaluatedSymbol sym{pointerToJITTargetAddress(func), s.second};
        cantFail(R.notifyResolved({{intern, sym}}));
      }
    }

    cantFail(R.notifyEmitted());
  }
};

class ConcurrentNVPTXCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  explicit ConcurrentNVPTXCompiler(llvm::orc::JITTargetMachineBuilder JTMB,
                                   ObjectCache *ObjCache = nullptr)
      : IRCompiler(
            llvm::orc::irManglingOptionsFromTargetOptions(JTMB.getOptions())),
        JTMB(std::move(JTMB)),
        ObjCache(ObjCache) {}

  void setObjectCache(ObjectCache *ObjCache) { this->ObjCache = ObjCache; }

 protected:
  llvm::orc::JITTargetMachineBuilder JTMB;
  ObjectCache *ObjCache = nullptr;

  Expected<std::unique_ptr<MemoryBuffer>> operator()(Module &M) override {
    //    CompileResult CachedObject = tryToLoadFromObjectCache(M);
    //    if (CachedObject) return std::move(CachedObject);

    SmallVector<char, 0> ObjBufferSV;
    auto TM = cantFail(JTMB.createTargetMachine());

    auto ptx = new string{};

    {
      llvm::raw_string_ostream stream(*ptx);
      llvm::buffer_ostream ostream(stream);

      llvm::legacy::PassManager PM;

      // Ask the target to add backend passes as necessary.
#ifndef NDEBUG
      bool x =
#endif
          TM->addPassesToEmitFile(PM, ostream, nullptr, llvm::CGFT_AssemblyFile,
                                  false);
      assert(!x);
      PM.run(M);
    }  // flushes stream and ostream

    return MemoryBuffer::getMemBuffer(*ptx);
  }
};

llvm::orc::ThreadSafeModule optimizeGpuModule(
    llvm::orc::ThreadSafeModule TSM, std::unique_ptr<TargetMachine> TM) {
  TSM.withModuleDo([&TM](Module &M) {
    time_block t(TimeRegistry::Key{"Optimization phase (GPU)"});
    // if (Coroutines)
    //   addCoroutinePassesToExtensionPoints(Builder);

    GpuPassConfiguration pc{std::move(TM)};

    llvm::legacy::FunctionPassManager FPasses{&M};

    pc.Builder.populateFunctionPassManager(FPasses);

    FPasses.add(pc.TTIPass);
    FPasses.add(pc.PrefetchPass);

    {
      time_block trun(TimeRegistry::Key{"Optimization run phase (GPU)"});

      FPasses.doInitialization();
      for (Function &F : M) FPasses.run(F);
      FPasses.doFinalization();

      // Now that we have all of the passes ready, run them.
      pc.Passes.run(M);
    }
  });
  return TSM;
}

class GPUJITer_impl {
 public:
  ::ThreadPool pool;

  DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  llvm::orc::ThreadSafeContext Ctx;

  //  PassConfiguration PassConf;

  llvm::orc::ExecutionSession ES;
  NVPTXLinkLayer ObjectLayer;
  llvm::orc::IRCompileLayer CompileLayer;
  llvm::orc::IRTransformLayer TransformLayer;

  llvm::orc::JITDylib &MainJD;

 public:
  explicit GPUJITer_impl(llvm::orc::JITTargetMachineBuilder JTMB =
                             llvm::cantFail(detectGPU())
                                 .setCodeGenOptLevel(CodeGenOpt::Aggressive))
      : pool(false, 4 * std::thread::hardware_concurrency()),
        DL(llvm::cantFail(JTMB.getDefaultDataLayoutForTarget())),
        Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        //        PassConf(llvm::cantFail(JTMB.createTargetMachine())),
        ObjectLayer(ES),
        CompileLayer(ES, ObjectLayer,
                     std::make_unique<ConcurrentNVPTXCompiler>(JTMB)),
        TransformLayer(
            ES, CompileLayer,
            [JTMB = std::move(JTMB)](
                llvm::orc::ThreadSafeModule TSM,
                const llvm::orc::MaterializationResponsibility &R) mutable
            -> Expected<llvm::orc::ThreadSafeModule> {
              auto TM = JTMB.createTargetMachine();
              if (!TM) return TM.takeError();
              return optimizeGpuModule(std::move(TSM), std::move(TM.get()));
            }),
        MainJD(llvm::cantFail(ES.createJITDylib("main"))) {
    ObjectLayer.setNotifyEmitted(
        [](llvm::orc::VModuleKey k, std::unique_ptr<MemoryBuffer> mb) {
          LOG(INFO) << "GPU Emitted " << k << " "
                    << mb.get();  //->getBufferIdentifier().str();
        });

    ES.setDispatchMaterialization(
        [&p = pool](std::unique_ptr<llvm::orc::MaterializationUnit> MU,
                    llvm::orc::MaterializationResponsibility MR) {
          auto SharedMU =
              std::shared_ptr<llvm::orc::MaterializationUnit>(std::move(MU));
          auto SharedMR =
              std::make_shared<llvm::orc::MaterializationResponsibility>(
                  std::move(MR));
          p.enqueue([SharedMU, SharedMR]() {
            SharedMU->materialize(std::move(*SharedMR));
          });
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

 private:
  auto lookup_internal(StringRef Name) {
    return ES.lookup({&MainJD}, Mangle(Name.str()));
  }

  auto gpuMangle(StringRef Name) { return ObjectLayer.gpuMangle(Name); }

 public:
  JITEvaluatedSymbol lookup(StringRef Name) {
    auto ret = lookup_internal(gpuMangle(Name));
    if (ret) return ret.get();
    llvm::consumeError(ret.takeError());
    // symbol is not generated due to mangling (there is probably  nicer way to
    // do that). Retry without mangling to fire the generation and then
    // look it up again as gpu-mangled
    llvm::cantFail(lookup_internal(Name));
    return llvm::cantFail(lookup_internal(gpuMangle(Name)));
  }
};

LLVMContext &GPUJITer::getContext() { return p_impl->getContext(); }
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
auto &getGPUJiter() {
  static GPUJITer jiter;
  return jiter;
}

void GpuModule::compileAndLoad() {
#ifndef NCUDA
  time_block t(pipName + " G: ", TimeRegistry::Key{"Compile and Load (GPU)"});

#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    llvm::raw_fd_ostream out("generated_code/" + pipName + ".ll", EC,
                             llvm::sys::fs::OpenFlags::F_None);

    getModule()->print(out, nullptr, false, true);
  }
#endif
  auto mod = getModuleRef(getModule()->getContext());

  {
    time_block tlink{"Link and internalize (GPU)"};
    llvm::Linker::linkModules(*getModule(), std::move(mod));

    {
      llvm::ModulePassManager mpm;
      llvm::ModuleAnalysisManager mam;

      PassBuilder b;
      b.registerModuleAnalyses(mam);

      std::set<std::string> gs{
          //        "pool", "npool", "deviceId", "buff_start", "buff_end",
      };

      mpm.addPass(llvm::InternalizePass([&](const GlobalValue &g) {
        if (gs.count(g.getName().str())) return true;
        if (g.getName().startswith(getModule()->getName())) return true;
        return preserveFromInternalization.count(g.getName().str()) > 0;
      }));
      mpm.run(*getModule(), mam);
    }
  }

  //  optimizeModule(getModule());
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

  // Change the source file name, otherwise the main function conflicts with
  // the source file name and it does not get emitted
  ptr.withModuleDo([pipName = this->pipName](Module &nm) {
    nm.setSourceFileName("this_gpu_" + pipName);
  });
  getGPUJiter().p_impl->addModule(std::move(ptr));

  // Dump to see final (optimized) form
#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    llvm::raw_fd_ostream out("generated_code/" + pipName + "_opt.ll", EC,
                             llvm::sys::fs::OpenFlags::F_None);

    getModule()->print(out, nullptr, false, true);
  }
#endif

  //  string ptx;
  //  {
  //    llvm::raw_string_ostream stream(ptx);
  //    llvm::buffer_ostream ostream(stream);
  //
  //    llvm::legacy::PassManager PM;
  //
  //    // Ask the target to add backend passes as necessary.
  //    TheTargetMachine->addPassesToEmitFile(PM, ostream,
  //#if LLVM_VERSION_MAJOR >= 7
  //                                          nullptr,
  //#endif
  //                                          llvm::CGFT_AssemblyFile, false);
  //
  //    PM.run(*(getModule()));
  //  }  // flushes stream and ostream
  //#ifdef DEBUGCTX
  //  if (print_generated_code) {
  //    std::ofstream optx("generated_code/" + pipName + ".ptx");
  //    optx << ptx;
  //  }
  //#endif

#else
  assert(false);
#endif
  //  auto addr = getGPUJiter().p_impl->lookup(getModule()->getName());
  //  LOG(INFO) << getModule()->getName().str() << " " << (void
  //  *)addr.getAddress();
}

void *GpuModule::getCompiledFunction(Function *f) const {
  time_block t(TimeRegistry::Key{"Compile and Load (GPU, waiting)"});
  return (void *)getGPUJiter().p_impl->lookup(f->getName().str()).getAddress();
}

void GpuModule::markToAvoidInteralizeFunction(std::string func) {
  preserveFromInternalization.emplace(std::move(func));
}
