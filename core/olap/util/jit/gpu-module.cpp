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
#include "util/jit/gpu-module.hpp"

#include <dlfcn.h>

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
#pragma push_macro("NDEBUG")
#define NDEBUG
#include "llvm/Analysis/TargetTransformInfo.h"
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

  int dev;
  gpu_run(cudaGetDevice(&dev));
  cudaDeviceProp deviceProp;
  gpu_run(cudaGetDeviceProperties(&deviceProp, dev));
  auto GPU = "sm_" + std::to_string(deviceProp.major * 10 + deviceProp.minor);

  llvm::TargetOptions opt;
  opt.DisableIntegratedAS = 1;
  opt.MCOptions.ShowMCEncoding = 1;
  opt.MCOptions.MCUseDwarfDirectory = 1;
  // opt.MCOptions.AsmVerbose            = 1;
  opt.MCOptions.PreserveAsmComments = 1;

  std::cout << GPU << std::endl;

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

// extern char _binary_device_funcs_cubin_end  [];
// extern char _binary_device_funcs_cubin_size   ; //size = (size_t)
// &_binary_device_funcs_cubin_size extern char
// _binary_device_funcs_cubin_start[];

constexpr size_t BUFFER_SIZE = 8192;
char error_log[BUFFER_SIZE];
char info_log[BUFFER_SIZE];

void GpuModule::compileAndLoad() {
#ifndef NCUDA
  LOG(INFO) << "[Prepare Function: ] Exit";  // and dump code so far";
  time_block t(pipName + " G: ");

#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    llvm::raw_fd_ostream out(
        "generated_code/" + pipName + ".ll", EC,
        (llvm::sys::fs::OpenFlags)0);  // FIXME:
                                       // llvm::sys::fs::OpenFlags::F_NONE is
                                       // the correct one but it gives a
                                       // compilation error

    getModule()->print(out, nullptr, false, true);
  }
#endif

  optimizeModule(getModule());

  // Dump to see final (optimized) form
#ifdef DEBUGCTX
  // getModule()->dump();

  if (print_generated_code) {
    std::error_code EC;
    llvm::raw_fd_ostream out(
        "generated_code/" + pipName + "_opt.ll", EC,
        (llvm::sys::fs::OpenFlags)0);  // FIXME:
                                       // llvm::sys::fs::OpenFlags::F_NONE is
                                       // the correct one but it gives a
                                       // compilation error

    getModule()->print(out, nullptr, false, true);
  }
#endif

  string ptx;
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream ostream(stream);

    llvm::legacy::PassManager PM;

    // Ask the target to add backend passes as necessary.
    TheTargetMachine->addPassesToEmitFile(
        PM, ostream,
#if LLVM_VERSION_MAJOR >= 7
        nullptr,
#endif
        llvm::TargetMachine::CGFT_AssemblyFile, false);

    PM.run(*(getModule()));
  }  // flushes stream and ostream
#ifdef DEBUGCTX
  if (print_generated_code) {
    std::ofstream optx("generated_code/" + pipName + ".ptx");
    optx << ptx;
  }
#endif

  // {
  //     time_block t("Tcuda_comp: ");
  //     CUlinkState linkState;

  //     gpu_run(cuLinkCreate  (0, nullptr, nullptr, &linkState));
  //     gpu_run(cuLinkAddData (linkState, CU_JIT_INPUT_PTX, (void *)
  //     ptx.c_str(), ptx.length() + 1, 0, 0, 0, 0)); gpu_run(cuLinkAddFile
  //     (linkState, CU_JIT_INPUT_LIBRARY,
  //     "/usr/local/cuda/lib64/libcudadevrt.a", 0, nullptr, nullptr));
  //     gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX,
  //     "/home/chrysoge/Documents/pelago/src/raw-jit-executor/codegen/device_funcs.ptx",
  //     0, nullptr, nullptr)); gpu_run(cuLinkComplete(linkState, &cubin,
  //     &cubinSize)); gpu_run(cuLinkDestroy (linkState));
  // }
  {
    time_block t("TcuCompile: ");  // FIXME: Currently requires all GPUs to be
                                   // of the same compute capability, or to be
                                   // more precise, all of them to be compatible
                                   // with the CC of the current device
    void *cubin;
    size_t cubinSize;

    CUlinkState linkState = nullptr;

    constexpr size_t opt_size = 6;
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

    char *_binary_buffer_manager_cubin_start;
    char *_binary_buffer_manager_cubin_end;
    // FIXME: should use a loop instead of nested ifs... also, we should cache
    // the result per sm
    {  // Compute symbol name for one of the GPU architecture, and use that to
      // retrieve the corresponding binary blob.
      // FIXME: We assume compute and arch are equal!
      // FIXME: Add error handling, we are generating a symbol name and
      //        assuming it is going to be available...
      int dev;
      gpu_run(cudaGetDevice(&dev));
      cudaDeviceProp deviceProp;
      gpu_run(cudaGetDeviceProperties(&deviceProp, dev));
      auto sm_code = std::to_string(deviceProp.major * 10 + deviceProp.minor);

      auto sim_prefix = "_binary_buffer_manager_compute_" + sm_code + "_sm_" +
                        sm_code + "_cubin_";
      auto sim_start = sim_prefix + "start";
      auto sim_end = sim_prefix + "end";

      void *handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
      assert(handle);

      _binary_buffer_manager_cubin_start =
          (char *)dlsym(handle, sim_start.c_str());
      _binary_buffer_manager_cubin_end = (char *)dlsym(handle, sim_end.c_str());

      if (!_binary_buffer_manager_cubin_start) {
        assert(!_binary_buffer_manager_cubin_end &&
               "Only one of the symbols found!");
        // CUDA 8.0 in RHEL does not include the compute_XX part of the
        // string
        auto sim_prefix = "_binary_buffer_manager_sm_" + sm_code + "_cubin_";
        sim_start = sim_prefix + "start";
        sim_end = sim_prefix + "end";

        _binary_buffer_manager_cubin_start =
            (char *)dlsym(handle, sim_start.c_str());
        _binary_buffer_manager_cubin_end =
            (char *)dlsym(handle, sim_end.c_str());

        if (!_binary_buffer_manager_cubin_start) {
          assert(!_binary_buffer_manager_cubin_end &&
                 "Only one of the symbols found!");
          // CUDA 8.0 in RHEL does not include the compute_XX part of the
          // string
          auto sim_prefix =
              "_binary_buffer_manager_compute_" + sm_code + "_cubin_";
          sim_start = sim_prefix + "start";
          sim_end = sim_prefix + "end";

          _binary_buffer_manager_cubin_start =
              (char *)dlsym(handle, sim_start.c_str());
          _binary_buffer_manager_cubin_end =
              (char *)dlsym(handle, sim_end.c_str());

          if (!_binary_buffer_manager_cubin_start) {
            assert(!_binary_buffer_manager_cubin_end &&
                   "Only one of the symbols found!");
            // CUDA 8.0 in RHEL does not include the compute_XX part of the
            // string
            std::string sim_prefix = "_binary_buffer_manager_cubin_";
            sim_start = sim_prefix + "start";
            sim_end = sim_prefix + "end";

            _binary_buffer_manager_cubin_start =
                (char *)dlsym(handle, sim_start.c_str());
            _binary_buffer_manager_cubin_end =
                (char *)dlsym(handle, sim_end.c_str());
          }
        }
      }
      assert(_binary_buffer_manager_cubin_start &&
             "cubin start symbol not found!");
      assert(_binary_buffer_manager_cubin_end && "cubin end symbol not found!");

      LOG(INFO) << "[Load CUBIN blob: ] sim_start: " << sim_start
                << ", sim_end: " << sim_end;
      LOG(INFO) << "[Load CUBIN blob: ] start: "
                << (void *)_binary_buffer_manager_cubin_start
                << ", end: " << (void *)_binary_buffer_manager_cubin_end;
    }

    // size_t size = _binary_device_funcs_cubin_end -
    // _binary_device_funcs_cubin_start;
    size_t size =
        _binary_buffer_manager_cubin_end - _binary_buffer_manager_cubin_start;

    gpu_run(cuLinkCreate(opt_size, options, values, &linkState));
    // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_LIBRARY,
    // "/usr/local/cuda/lib64/libcudadevrt.a", 0, nullptr, nullptr));
    // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_CUBIN,
    // "/home/chrysoge/Documents/pelago/opt/res/device_funcs.cubin", 0, nullptr,
    // nullptr)); auto x = (cuLinkAddData (linkState, CU_JIT_INPUT_CUBIN,
    // _binary_device_funcs_cubin_start, size, nullptr, 0, nullptr, nullptr));
    auto x = (cuLinkAddData(linkState, CU_JIT_INPUT_CUBIN,
                            _binary_buffer_manager_cubin_start, size, nullptr,
                            0, nullptr, nullptr));

    // the strange file name comes from FindCUDA... hopefully there is way to
    // change it...
    // auto x = (cuLinkAddFile (linkState, CU_JIT_INPUT_CUBIN,
    // "/home/chrysoge/Documents/pelago/build/raw-jit-executor/codegen/multigpu/CMakeFiles/multigpu.dir/multigpu_generated_buffer_manager.cu.o.cubin.txt",
    // 0, nullptr, nullptr)); auto x = (cuLinkAddFile (linkState,
    // CU_JIT_INPUT_CUBIN,
    // "/home/chrysoge/Documents/pelago/opt/res/buffer-manager.cubin", 0,
    // nullptr, nullptr)); libmultigpu.a", 0, nullptr, nullptr));
    if (x != CUDA_SUCCESS) {
      // If you get an error message similar to "no kernel image is available
      // for execution on the device" it usually means that the target sm_xy in
      // root CMakeLists.txt is not set to the current GPU's CC.
      printf("[CUcompile: ] %s\n", info_log);
      printf("[CUcompile: ] %s\n", error_log);
      gpu_run(x);
    }
    // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX,
    // "/home/chrysoge/Documents/pelago/src/raw-jit-executor/codegen/device_funcs.ptx",
    // 0, nullptr, nullptr)); gpu_run(cuLinkAddFile (linkState,
    // CU_JIT_INPUT_PTX,
    // ("generated_code/" + pipName + ".ptx").c_str(), 0, nullptr, nullptr));
    x = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void *)ptx.c_str(),
                      ptx.length() + 1, nullptr, 0, nullptr, nullptr);
    if (x != CUDA_SUCCESS) {
      printf("[CUcompile: ] %s\n", info_log);
      printf("[CUcompile: ] %s\n", error_log);
      gpu_run(x);
    }
    x = cuLinkComplete(linkState, &cubin, &cubinSize);
    if (x != CUDA_SUCCESS) {
      printf("[CUcompile: ] %s\n", info_log);
      printf("[CUcompile: ] %s\n", error_log);
      gpu_run(x);
    }

    for (const auto &gpu : topology::getInstance().getGpus()) {
      time_block t("TloadModule: ");
      set_device_on_scope d(gpu);

      // gpu_run(cuModuleLoadDataEx(&cudaModule[i], ptx.c_str(), 0, 0, 0));
      gpu_run(cuModuleLoadFatBinary(&cudaModule[gpu.id], cubin));
      {
        time_block t("TinitModule: ");
        initializeModule(cudaModule[gpu.id]);
      }
    }

    gpu_run(cuLinkDestroy(linkState));
  }
  // func_name = F->getName().str();
#else
  assert(false);
#endif
}

void *GpuModule::getCompiledFunction(Function *f) const {
#ifndef NCUDA
  CUfunction func = nullptr;
  gpu_run(cuModuleGetFunction(
      &func, cudaModule[topology::getInstance().getActiveGpu().id],
      f->getName().str().c_str()));

  return (void *)func;
#else
  assert(false);
  return nullptr;
#endif
}
