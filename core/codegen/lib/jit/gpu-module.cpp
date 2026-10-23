/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#include <dlfcn.h>

#include <codegen/jit/gpu-module.hpp>
#pragma push_macro("NDEBUG")
#define NDEBUG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-macros"
#if LLVM_VERSION_MAJOR >= 13
#define dumpDispatchInfo(x) (5)
#else
#define dumpDispatchInfo(x, y) (5)
#endif
#pragma clang diagnostic pop
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
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
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/Cloning.h>
#pragma pop_macro("NDEBUG")

#include <platform/threadpool/threadpool.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/timing.hpp>
#include <cerrno>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <utility>

void initializeModule(CUmodule &cudaModule);

using namespace llvm;

#ifndef NVPTX_MAX_REGS
#define NVPTX_MAX_REGS (64)
#endif

constexpr uint64_t nvptx_max_regs = NVPTX_MAX_REGS;

static uint64_t getNvptxMaxRegs() {
  const char *env = std::getenv("PROTEUS_NVPTX_MAX_REGS");
  if (!env || *env == '\0') return nvptx_max_regs;

  errno = 0;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0') {
    LOG(WARNING) << "Ignoring invalid PROTEUS_NVPTX_MAX_REGS='" << env
                 << "', using default " << nvptx_max_regs;
    return nvptx_max_regs;
  }

  if (parsed == 0 || parsed > 255) {
    LOG(WARNING) << "PROTEUS_NVPTX_MAX_REGS out of range (" << parsed
                 << "), expected 1..255; using default " << nvptx_max_regs;
    return nvptx_max_regs;
  }

  return static_cast<uint64_t>(parsed);
}

LLVMTargetMachine *GpuModule::TheTargetMachine = nullptr;

GpuModule::GpuModule(Context *context, std::string pipName)
    : JITModule(context, std::move(pipName)) {
  if (TheTargetMachine == nullptr) init();

  // Inform the module about the current configuration
  getModule()->setDataLayout(TheTargetMachine->createDataLayout());
  getModule()->setTargetTriple(TheTargetMachine->getTargetTriple().getTriple());
}

GpuModule::~GpuModule() {
  if (resourceTracker && !resourceTracker->isDefunct()) {
    if (auto Err = resourceTracker->remove()) {
      LOG(WARNING) << "Failed to remove GPU JIT resources for " << pipName
                   << ": " << toString(std::move(Err));
    }
  }
  delete TheModule;
  TheModule = nullptr;
}

auto getGPU() {
  assert(topology::getInstance().getGpuCount() > 0);
  int dev = 0;
  gpu_run(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp{};
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
  opt.MCOptions.ShowMCEncoding = true;
  opt.MCOptions.MCUseDwarfDirectory = true;
  // opt.MCOptions.AsmVerbose            = 1;
  opt.MCOptions.PreserveAsmComments = true;

  auto RM = llvm::Optional<llvm::Reloc::Model>();
  TheTargetMachine = (llvm::LLVMTargetMachine *)Target->createTargetMachine(
      TargetTriple, GPU,
      "+ptx64",  // PTX 6.4 + Scoped Atomics
      //"+ptx60,+satom", //for V100
      opt, RM,
      llvm::Optional<llvm::CodeModel::Model>{},  // CodeModel::Model::Default,
      llvm::CodeGenOpt::Aggressive);
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
    Passes.add(createGlobalDCEPass());
    //    Passes.add(createDeadCodeEliminationPass());
    //    Passes.add(createWarnMissedTransformationsPass());

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

constexpr size_t BUFFER_SIZE = 8192;
static char error_log[BUFFER_SIZE];
static char info_log[BUFFER_SIZE];

static llvm::StringRef discover_bc() {
  // FIXME: should use a loop instead of nested ifs... also, we should cache
  // the result per sm
  // Compute symbol name for one of the GPU architecture, and use that to
  // retrieve the corresponding binary blob.
  // FIXME: We assume compute and arch are equal!
  // FIXME: Add error handling, we are generating a symbol name and
  //        assuming it is going to be available...
  int dev = 0;
  gpu_run(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp{};
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

  auto start = (const char *)_binary_buffer_manager_cubin_start;
  auto end = (const char *)_binary_buffer_manager_cubin_end;
  size_t size = end - start;
  return {start, size};
}

auto getCachedBC() {
  static auto bounds{discover_bc()};
  return bounds;
}

auto getModuleRef(LLVMContext &llvmContext) {
  return cantFail(llvm::getOwningLazyBitcodeModule(
      llvm::MemoryBuffer::getMemBuffer(getCachedBC(), "", false), llvmContext,
      false, false));
}

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

  LLVMContext &getContext() const;
};

GPUJITer::GPUJITer() : p_impl(std::make_unique<GPUJITer_impl>()) {}

GPUJITer::~GPUJITer() = default;

class NVPTXLinkLayer final : public llvm::orc::ObjectLayer {
 public:
  using RegisterLoadedModulesFn =
      std::function<void(llvm::orc::ResourceKey,
                         const std::vector<CUmodule> &)>;

  static auto gpuMangle(StringRef Name, decltype(topology::gpunode::id) id) {
    return ("_gpu" + std::to_string(id) + "_" + Name).str();
  }

  auto gpuMangle(StringRef Name) {
    return gpuMangle(Name, topology::getInstance().getActiveGpu().id);
  }

  explicit NVPTXLinkLayer(llvm::orc::ExecutionSession &ES,
                          RegisterLoadedModulesFn register_loaded_modules = {})
      : ObjectLayer(ES),
        registerLoadedModules(std::move(register_loaded_modules)) {}
  //  using ObjectLayer::ObjectLayer;

  template <typename F>
  void setNotifyEmitted(F &&f) {}

  void emit(std::unique_ptr<llvm::orc::MaterializationResponsibility> R,
            std::unique_ptr<MemoryBuffer> Obj) override {
    //    llvm::orc::LocalJITCompileCallbackManager<llvm::orc::OrcGenericABI>::Create();

    std::vector<CUmodule> cudaModules(topology::getInstance().getGpuCount(),
                                      nullptr);

    {
      time_block t(TimeRegistry::Key{"Load GPU module"});
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
      const auto max_regs = getNvptxMaxRegs();
      values[3] =
          reinterpret_cast<void *>(static_cast<uintptr_t>(max_regs));
      options[4] = CU_JIT_INFO_LOG_BUFFER;
      values[4] = (void *)info_log;
      options[5] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      values[5] = (void *)BUFFER_SIZE;
      options[6] = CU_JIT_LOG_VERBOSE;
      values[6] = (void *)1;
      options[7] = CU_JIT_OPTIMIZATION_LEVEL;
      values[7] = (void *)4;

      {
        time_block tInit{TimeRegistry::Key{"NVML compile module"}};
        LOG(INFO) << "NVPTX CU_JIT_MAX_REGISTERS: " << max_regs;
        for (const auto &gpu : topology::getInstance().getGpus()) {
          set_device_on_scope d(gpu);

          auto x =
              (cuModuleLoadDataEx(&cudaModules[gpu.id], Obj->getBuffer().data(),
                                  opt_size, options, values));

          if (info_log[0] != '\0') LOG(INFO) << info_log;
          if (x != CUDA_SUCCESS) {
            LOG(ERROR) << error_log;
            R->failMaterialization();
          }
        }
      }
      {
        time_block tInit{TimeRegistry::Key{"Init GPU module"}};
        for (const auto &gpu : topology::getInstance().getGpus()) {
          set_device_on_scope d(gpu);

          initializeModule(cudaModules[gpu.id]);
        }
      }

      if (registerLoadedModules) {
        cantFail(R->withResourceKeyDo([&](const auto &key) {
          registerLoadedModules(key, cudaModules);
        }));
      }
    }

    std::vector<std::pair<std::string, JITSymbolFlags>> reqKernels;

    for (auto &r : R->getSymbols()) {
      reqKernels.emplace_back(*r.getFirst(), r.getSecond());
      const auto kernel_name = (*r.getFirst()).str();

      CUfunction func = nullptr;
      gpu_run(cuModuleGetFunction(
          &func, cudaModules[topology::getInstance().getGpus()[0].id],
          kernel_name.c_str()));
      register_gpu_kernel_symbol(func, kernel_name.c_str());

      JITEvaluatedSymbol sym{pointerToJITTargetAddress(func), r.getSecond()};
      cantFail(R->notifyResolved({{r.getFirst(), sym}}));
    }

    // merging with above invalidates the loop iterator
    for (const auto &s : reqKernels) {
      for (const auto &gpu : topology::getInstance().getGpus()) {
        auto kernel_name = gpuMangle(s.first, gpu.id);
        auto intern = getExecutionSession().intern(kernel_name);

        cantFail(R->defineMaterializing({{intern, s.second}}));

        CUfunction func = nullptr;
        gpu_run(
            cuModuleGetFunction(&func, cudaModules[gpu.id], s.first.c_str()));
        register_gpu_kernel_symbol(func, s.first.c_str());
        JITEvaluatedSymbol sym{pointerToJITTargetAddress(func), s.second};
        cantFail(R->notifyResolved({{intern, sym}}));
      }
    }

    cantFail(R->notifyEmitted());
  }

 private:
  RegisterLoadedModulesFn registerLoadedModules;
};

class ConcurrentNVPTXCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  explicit ConcurrentNVPTXCompiler(llvm::orc::JITTargetMachineBuilder JTMB,
                                   ObjectCache *ObjCache = nullptr)
      : IRCompiler(
            llvm::orc::irManglingOptionsFromTargetOptions(JTMB.getOptions())),
        JTMB(std::move(JTMB)),
        ObjCache(ObjCache) {}

  void setObjectCache(ObjectCache *objCache) { this->ObjCache = objCache; }

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

    if (print_generated_code) {
      std::error_code EC;
      llvm::raw_fd_ostream out(
          ("generated_code/" + M.getName() + "_opt.ll").str(), EC,
          llvm::sys::fs::OpenFlags::OF_None);

      M.print(out, nullptr, false, true);
    }
  });
  return TSM;
}

// defined in cpu-module.cpp
Expected<llvm::orc::ThreadSafeModule> printIR(orc::ThreadSafeModule module,
                                              const std::string &suffix = "",
                                              bool add_debug_info = false);

class GPUJITer_impl : public llvm::orc::ResourceManager {
 public:
  ::ThreadPool pool;

  DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  llvm::orc::ThreadSafeContext Ctx;

  //  PassConfiguration PassConf;

#if LLVM_VERSION_MAJOR >= 13
  llvm::orc::ExecutionSession ES{
      llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create())};
#else
  llvm::orc::ExecutionSession ES;
#endif
  NVPTXLinkLayer ObjectLayer;
  llvm::orc::IRCompileLayer CompileLayer;
  llvm::orc::IRTransformLayer PrintOptimizedIRLayer;
  llvm::orc::IRTransformLayer TransformLayer;
  llvm::orc::IRTransformLayer InternalizeLayer;
  llvm::orc::IRTransformLayer PrintGeneratedIRLayer;

  llvm::orc::JITDylib &MainJD;

  std::mutex preserve_m;
  std::map<llvm::orc::ResourceKey, std::set<std::string>> preserve;
  std::mutex loaded_modules_m;
  std::unordered_map<llvm::orc::ResourceKey, std::vector<CUmodule>>
      loaded_modules;

  Error handleRemoveResources(llvm::orc::ResourceKey K) override {
    {
      std::unique_lock<std::mutex> lock{preserve_m};
      preserve.erase(K);
    }

    std::vector<CUmodule> modules_to_unload;
    {
      std::unique_lock<std::mutex> lock{loaded_modules_m};
      auto it = loaded_modules.find(K);
      if (it != loaded_modules.end()) {
        modules_to_unload = std::move(it->second);
        loaded_modules.erase(it);
      }
    }

    for (const auto &gpu : topology::getInstance().getGpus()) {
      if (gpu.id >= modules_to_unload.size()) continue;
      auto mod = modules_to_unload[gpu.id];
      if (!mod) continue;

      set_device_on_scope d(gpu);
      auto res = cuModuleUnload(mod);
      if (res != CUDA_SUCCESS) {
        LOG(WARNING) << "Failed to unload GPU module for resource key " << K
                     << " on gpu " << gpu.id << " (CUresult=" << res << ")";
      }
    }

    return Error::success();
  }

  void handleTransferResources(llvm::orc::ResourceKey DstK,
                               llvm::orc::ResourceKey SrcK) override {
    {
      std::unique_lock<std::mutex> lock{preserve_m};
      if (preserve.contains(SrcK)) {
        auto &src = preserve[SrcK];
        auto &dst = preserve[DstK];
        std::move(src.begin(), src.end(), std::inserter(dst, dst.begin()));

        preserve.erase(SrcK);
      }
    }

    std::unique_lock<std::mutex> lock{loaded_modules_m};
    if (loaded_modules.contains(SrcK)) {
      auto &src = loaded_modules[SrcK];
      auto &dst = loaded_modules[DstK];
      if (dst.size() < src.size()) dst.resize(src.size(), nullptr);
      for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] != nullptr) dst[i] = src[i];
      }
      loaded_modules.erase(SrcK);
    }
  }

 public:
  explicit GPUJITer_impl(llvm::orc::JITTargetMachineBuilder JTMB =
                             llvm::cantFail(detectGPU())
                                 .setCodeGenOptLevel(CodeGenOpt::Aggressive)
                                 .setCodeModel(llvm::CodeModel::Model::Large))
      : pool(false, std::thread::hardware_concurrency()),
        DL(llvm::cantFail(JTMB.getDefaultDataLayoutForTarget())),
        Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        //        PassConf(llvm::cantFail(JTMB.createTargetMachine())),
        ObjectLayer(ES, [this](llvm::orc::ResourceKey key,
                               const std::vector<CUmodule> &modules) {
          std::unique_lock<std::mutex> lock{loaded_modules_m};
          loaded_modules[key] = modules;
        }),
        CompileLayer(ES, ObjectLayer,
                     std::make_unique<ConcurrentNVPTXCompiler>(JTMB)),
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
              return optimizeGpuModule(std::move(TSM), std::move(TM.get()));
            }),
        InternalizeLayer(
            ES, TransformLayer,
            [this](llvm::orc::ThreadSafeModule TSM,
                   const llvm::orc::MaterializationResponsibility &R)
                -> Expected<llvm::orc::ThreadSafeModule> {
              auto preserveFromInternalization = [&]() {
                decltype(preserve)::value_type::second_type tmp;

                cantFail(R.withResourceKeyDo([this, &tmp](const auto &key) {
                  std::unique_lock<std::mutex> lock{preserve_m};
                  auto it = preserve.find(key);
                  assert(it != preserve.end());
                  tmp = it->second;
                }));

                return tmp;
              }();
              TSM.withModuleDo([&preserveFromInternalization](Module &mod) {
                time_block tlink{
                    TimeRegistry::Key{"Link and internalize (GPU)"}};
                llvm::Linker::linkModules(mod, getModuleRef(mod.getContext()),
                                          llvm::Linker::LinkOnlyNeeded);

                {
                  llvm::ModulePassManager mpm;
                  llvm::ModuleAnalysisManager mam;

                  PassBuilder b;
                  b.registerModuleAnalyses(mam);

                  std::set<std::string> gs{
                      // "pool", "npool", "deviceId", "buff_start", "buff_end",
                  };

                  mpm.addPass(llvm::InternalizePass([&](const GlobalValue &g) {
                    if (gs.count(g.getName().str())) return true;
                    if (g.getName().startswith(mod.getName())) return true;
                    return preserveFromInternalization.count(
                               g.getName().str()) > 0;
                  }));
                  mpm.addPass(llvm::GlobalDCEPass{});
                  mpm.run(mod, mam);
                }
              });
              return std::move(TSM);
            }),
        PrintGeneratedIRLayer(
            ES, InternalizeLayer,
            [](llvm::orc::ThreadSafeModule TSM,
               const llvm::orc::MaterializationResponsibility &R)
                -> Expected<llvm::orc::ThreadSafeModule> {
              if (print_generated_code) return printIR(std::move(TSM));
              return std::move(TSM);
            }),
        MainJD(llvm::cantFail(ES.createJITDylib("main"))) {
    ObjectLayer.setNotifyEmitted([](llvm::orc::MaterializationResponsibility &R,
                                    std::unique_ptr<MemoryBuffer> mb) {
      cantFail(R.withResourceKeyDo([&](const auto &k) {
        LOG(INFO) << "GPU Emitted " << k << " " << mb.get();
      }));
    });

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

    ES.registerResourceManager(*this);

    MainJD.addGenerator(llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            this->DL.getGlobalPrefix())));
  }

  ~GPUJITer_impl() override {
    if (auto Err = ES.endSession()) ES.reportError(std::move(Err));
  }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  llvm::orc::ResourceTrackerSP addModule(
      llvm::orc::ThreadSafeModule M,
      std::set<std::string> preserveFromInternalization) {
    class PreserveMaterializationUnit : public orc::IRMaterializationUnit {
     private:
      std::function<void(std::unique_ptr<orc::MaterializationResponsibility>,
                         llvm::orc::ThreadSafeModule M)>
          Materialize;

     public:
      PreserveMaterializationUnit(
          llvm::orc::ExecutionSession &ES,
          const llvm::orc::IRSymbolMapper::ManglingOptions &MO,
          decltype(Materialize) Materialize, llvm::orc::ThreadSafeModule M)
          : IRMaterializationUnit(ES, MO, std::move(M)),
            Materialize(std::move(Materialize)) {}

      void materialize(
          std::unique_ptr<orc::MaterializationResponsibility> R) override {
        Materialize(std::move(R), std::move(TSM));
      }
    };

    auto MU = std::make_unique<PreserveMaterializationUnit>(
        ES, *PrintGeneratedIRLayer.getManglingOptions(),
        [this, preserveFromInternalization](
            std::unique_ptr<llvm::orc::MaterializationResponsibility> R,
            llvm::orc::ThreadSafeModule M2) {
          cantFail(R->withResourceKeyDo([&](const auto &key) {
            std::unique_lock<std::mutex> lock{preserve_m};
            preserve.emplace(key, std::move(preserveFromInternalization));
          }));
          PrintGeneratedIRLayer.emit(std::move(R), std::move(M2));
        },
        std::move(M));

    auto RT = MainJD.createResourceTracker();
    llvm::cantFail(RT->getJITDylib().define(std::move(MU), std::move(RT)));
    return RT;
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

LLVMContext &GPUJITer::getContext() const { return p_impl->getContext(); }

auto &getGPUJiter() {
  static GPUJITer jiter;
  return jiter;
}

void GpuModule::compileAndLoad() {
  time_block t(TimeRegistry::Key{"Compile and Load (GPU)"});

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
  resourceTracker = getGPUJiter().p_impl->addModule(
      std::move(ptr), std::move(preserveFromInternalization));

  ::ThreadPool::getInstance().enqueue([name = getModule()->getName()]() {
    getGPUJiter().p_impl->lookup(name);
  });
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
