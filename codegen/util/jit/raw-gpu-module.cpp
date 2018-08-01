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
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/PassManager.h"
#include "topology/affinity_manager.hpp"

#include "multigpu/buffer_manager.cuh" //initializeModule

#include "util/jit/raw-gpu-module.hpp"

LLVMTargetMachine * RawGpuModule::TheTargetMachine = nullptr;
legacy::PassManager RawGpuModule::Passes                    ;
PassManagerBuilder  RawGpuModule::Builder                   ;

RawGpuModule::RawGpuModule(RawContext * context, std::string pipName):
    RawModule(context, pipName){
    uint32_t gpu_cnt = topology::getInstance().getGpuCount();
    cudaModule = (CUmodule *) malloc(gpu_cnt * sizeof(CUmodule));

    if (TheTargetMachine == nullptr) init();

    //Inform the module about the current configuration
    getModule()->setDataLayout  (TheTargetMachine->createDataLayout()           );
    getModule()->setTargetTriple(TheTargetMachine->getTargetTriple().getTriple());

    // string ErrStr;
    // TheExecutionEngine =
    //     EngineBuilder(std::unique_ptr<Module>(getModule())).setErrorStr(&ErrStr).create();
    // if (TheExecutionEngine == nullptr) {
    //     fprintf(stderr, "Could not create ExecutionEngine: %s\n",
    //             ErrStr.c_str());
    //     exit(1);
    // }

    // // JITEventListener* vtuneProfiler = JITEventListener::createIntelJITEventListener();
    // // if (vtuneProfiler == nullptr) {
    // //     fprintf(stderr, "Could not create VTune listener\n");
    // // } else {
    // //     TheExecutionEngine->RegisterJITEventListener(vtuneProfiler);
    // // }

    // JITEventListener* gdbDebugger = JITEventListener::createGDBRegistrationListener();
    // if (gdbDebugger == nullptr) {
    //     fprintf(stderr, "Could not create GDB listener\n");
    // } else {
    //     TheExecutionEngine->RegisterJITEventListener(gdbDebugger);
    // }
}

void RawGpuModule::init(){
    //Get the triplet for GPU
    std::string TargetTriple("nvptx64-nvidia-cuda");

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

    int dev;
    gpu_run(cudaGetDevice(&dev));
    cudaDeviceProp deviceProp;
    gpu_run(cudaGetDeviceProperties(&deviceProp, dev));
    auto GPU      = "sm_" + std::to_string(deviceProp.major * 10 + deviceProp.minor);

    // SubtargetFeatures Features;
    // StringMap<bool> HostFeatures;
    // if (sys::getHostCPUFeatures(HostFeatures)){
    //   for (auto &F : HostFeatures) Features.AddFeature(F.first(), F.second);
    // }

    // std::cout << GPU.str()            << std::endl;
    // std::cout << Features.getString() << std::endl;

    TargetOptions opt;
    opt.DisableIntegratedAS             = 1;
    opt.MCOptions.ShowMCEncoding        = 1;
    opt.MCOptions.MCUseDwarfDirectory   = 1;
    // opt.MCOptions.AsmVerbose            = 1;
    opt.MCOptions.PreserveAsmComments   = 1;

    std::cout << GPU << std::endl;

    auto RM = Optional<Reloc::Model>();
    TheTargetMachine = (LLVMTargetMachine *) Target->createTargetMachine(TargetTriple, GPU, 
                                                    "+ptx50,+ptx60,+satom", //PTX 5.0 + Scoped Atomics
                                                    opt, RM, 
                                                    Optional<CodeModel::Model>{},//CodeModel::Model::Default, 
                                                    CodeGenOpt::Aggressive);

                                  // // Override function attributes based on CPUStr, FeaturesStr, and command line
                                  // // flags.
                                  // setFunctionAttributes(CPUStr, FeaturesStr, *M);


    // TheTargetMachine->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
    //                        "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
    //                        "v64:64:64-v128:128:128-n16:32:64");
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

void RawGpuModule::optimizeModule(Module * M){
    time_block t("Optimization time: ");

    llvm::legacy::FunctionPassManager FPasses{M};
    FPasses.add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    Builder.populateFunctionPassManager(FPasses);

    FPasses.doInitialization();
    for (Function &F : *M) FPasses.run(F);
    FPasses.doFinalization();
    
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
}

// extern char _binary_device_funcs_cubin_end  [];
// extern char _binary_device_funcs_cubin_size   ; //size = (size_t) &_binary_device_funcs_cubin_size
// extern char _binary_device_funcs_cubin_start[];

extern char _binary_buffer_manager_cubin_end  [];
extern char _binary_buffer_manager_cubin_size   ; //size = (size_t) &_binary_buffer_manager_cubin_size
extern char _binary_buffer_manager_cubin_start[];

constexpr size_t BUFFER_SIZE = 8192;
char error_log[BUFFER_SIZE];
char info_log [BUFFER_SIZE];

void RawGpuModule::compileAndLoad(){
#ifndef NCUDA
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    time_block t(pipName + " G: ");

#ifdef DEBUGCTX
    // getModule()->dump();

    if (print_generated_code){
        std::error_code EC;
        raw_fd_ostream out("generated_code/" + pipName + ".ll", EC, (llvm::sys::fs::OpenFlags) 0); // FIXME: llvm::sys::fs::OpenFlags::F_NONE is the correct one but it gives a compilation error

        getModule()->print(out, nullptr, false, true);
    }
#endif

    optimizeModule(getModule());

    //Dump to see final (optimized) form
#ifdef DEBUGCTX
    // getModule()->dump();
    
    if (print_generated_code){
        std::error_code EC;
        raw_fd_ostream out("generated_code/" + pipName + "_opt.ll", EC, (llvm::sys::fs::OpenFlags) 0); // FIXME: llvm::sys::fs::OpenFlags::F_NONE is the correct one but it gives a compilation error

        getModule()->print(out, nullptr, false, true);
    }
#endif

    string ptx;
    {
        raw_string_ostream stream(ptx);
        buffer_ostream ostream(stream);
        
        legacy::PassManager PM;

        // Ask the target to add backend passes as necessary.
        TheTargetMachine->addPassesToEmitFile(PM, ostream, llvm::TargetMachine::CGFT_AssemblyFile, false); //NULL for LLVM7.0

        PM.run(*(getModule()));
    } // flushes stream and ostream
#ifdef DEBUGCTX
    if (print_generated_code){
        std::ofstream optx("generated_code/" + pipName + ".ptx");
        optx << ptx;
    }
#endif
    
    // {
    //     time_block t("Tcuda_comp: ");
    //     CUlinkState linkState;

    //     gpu_run(cuLinkCreate  (0, NULL, NULL, &linkState));
    //     gpu_run(cuLinkAddData (linkState, CU_JIT_INPUT_PTX, (void *) ptx.c_str(), ptx.length() + 1, 0, 0, 0, 0));
    //     gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_LIBRARY, "/usr/local/cuda/lib64/libcudadevrt.a", 0, NULL, NULL));
    //     gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX, "/home/chrysoge/Documents/pelago/src/raw-jit-executor/codegen/device_funcs.ptx", 0, NULL, NULL));
    //     gpu_run(cuLinkComplete(linkState, &cubin, &cubinSize));
    //     gpu_run(cuLinkDestroy (linkState));
    // }
    {
        time_block t("TcuCompile: "); //FIXME: Currently requires all GPUs to be of the same compute capability, or to be more precise, all of them to be compatible with the CC of the current device
        void * cubin;
        size_t cubinSize;

        CUlinkState linkState;
        
        constexpr size_t opt_size = 6;
        CUjit_option options[opt_size];
        void       * values [opt_size];

        options [0] = CU_JIT_TARGET_FROM_CUCONTEXT;
        values  [0] = 0;
        options [1] = CU_JIT_ERROR_LOG_BUFFER;
        values  [1] = (void *) error_log;
        options [2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        values  [2] = (void *) BUFFER_SIZE;
        options [3] = CU_JIT_MAX_REGISTERS;
        values  [3] = (void *) ((uint64_t) 32);
        options [4] = CU_JIT_INFO_LOG_BUFFER;
        values  [4] = (void *) info_log;
        options [5] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        values  [5] = (void *) BUFFER_SIZE;

        // size_t size = _binary_device_funcs_cubin_end - _binary_device_funcs_cubin_start;
        size_t size = _binary_buffer_manager_cubin_end - _binary_buffer_manager_cubin_start;

        gpu_run(cuLinkCreate  (opt_size, options, values, &linkState));
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_LIBRARY, "/usr/local/cuda/lib64/libcudadevrt.a", 0, NULL, NULL));
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_CUBIN, "/home/chrysoge/Documents/pelago/opt/res/device_funcs.cubin", 0, NULL, NULL));
        // auto x = (cuLinkAddData (linkState, CU_JIT_INPUT_CUBIN, _binary_device_funcs_cubin_start, size, NULL, 0, NULL, NULL));
        auto x = (cuLinkAddData (linkState, CU_JIT_INPUT_CUBIN, _binary_buffer_manager_cubin_start, size, NULL, 0, NULL, NULL));

        //the strange file name comes from FindCUDA... hopefully there is way to change it...
        // auto x = (cuLinkAddFile (linkState, CU_JIT_INPUT_CUBIN, "/home/chrysoge/Documents/pelago/build/raw-jit-executor/codegen/multigpu/CMakeFiles/multigpu.dir/multigpu_generated_buffer_manager.cu.o.cubin.txt", 0, NULL, NULL));
        // auto x = (cuLinkAddFile (linkState, CU_JIT_INPUT_CUBIN, "/home/chrysoge/Documents/pelago/opt/res/buffer_manager.cubin", 0, NULL, NULL));
            // libmultigpu.a", 0, NULL, NULL));
        if (x != CUDA_SUCCESS) {
            //If you get an error message similar to "no kernel image is available for execution on the device"
            //it usually means that the target sm_xy in root CMakeLists.txt is not set to the current GPU's CC.
            printf("[CUcompile: ] %s\n", info_log );
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX, "/home/chrysoge/Documents/pelago/src/raw-jit-executor/codegen/device_funcs.ptx", 0, NULL, NULL));
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX, ("generated_code/" + pipName + ".ptx").c_str(), 0, NULL, NULL));
        x = cuLinkAddData (linkState, CU_JIT_INPUT_PTX, (void *) ptx.c_str(), ptx.length() + 1, NULL, 0, NULL, NULL);
        if (x != CUDA_SUCCESS) {
            printf("[CUcompile: ] %s\n", info_log );
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }
        x = cuLinkComplete(linkState, &cubin, &cubinSize);
        if (x != CUDA_SUCCESS) {
            printf("[CUcompile: ] %s\n", info_log );
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }

        for (const auto &gpu: topology::getInstance().getGpus()){
            time_block t("TloadModule: ");
            set_device_on_scope d(gpu);

            // gpu_run(cuModuleLoadDataEx(&cudaModule[i], ptx.c_str(), 0, 0, 0));
            gpu_run(cuModuleLoadFatBinary(&cudaModule[gpu.id], cubin));
            {
                time_block t("TinitModule: ");
                initializeModule(cudaModule[gpu.id]);
            }
        }
        
        gpu_run(cuLinkDestroy (linkState));
    }
    // func_name = F->getName().str();
#else
    assert(false);
#endif
}

void * RawGpuModule::getCompiledFunction(Function * f) const{
#ifndef NCUDA
    CUfunction func;
    gpu_run(cuModuleGetFunction(&func, cudaModule[topology::getInstance().getActiveGpu().id], f->getName().str().c_str()));
    
    return (void *) func;
#else
    assert(false);
    return NULL;
#endif
}