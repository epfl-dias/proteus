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

#include "raw-gpu-pipeline.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "multigpu/buffer_manager.cuh"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/CodeGen/TargetPassConfig.h"

LLVMTargetMachine                             * RawGpuPipelineGen::TheTargetMachine = nullptr;
legacy::PassManager                             RawGpuPipelineGen::Passes;
PassManagerBuilder                              RawGpuPipelineGen::Builder;
std::unique_ptr<legacy::FunctionPassManager>    RawGpuPipelineGen::FPasses;

RawGpuPipelineGen::RawGpuPipelineGen(RawContext * context, std::string pipName, RawPipelineGen * copyStateFrom): 
            RawPipelineGen(context, pipName, copyStateFrom){
    // getModule()->setDataLayout(((GpuRawContext *) context)->TheTargetMachine->createDataLayout());
    cudaModule = (CUmodule *) malloc(get_num_of_gpus() * sizeof(CUmodule));

    //TheFPM = new legacy::FunctionPassManager(getModule());
    //                    addOptimizerPipelineDefault(TheFPM);

    // ThePM = new legacy::PassManager();

    //MapD uses:
    // ThePM->add(llvm::createAlwaysInlinerPass());
    // ThePM->add(llvm::createPromoteMemoryToRegisterPass());
    // ThePM->add(llvm::createInstructionSimplifierPass());
    // ThePM->add(llvm::createInstructionCombiningPass());
    // ThePM->add(llvm::createGlobalOptimizerPass());
    // ThePM->add(llvm::createLICMPass());
    // ThePM->add(llvm::createLoopStrengthReducePass());

    //LSC: Seems to be faster without the vectorization, at least
    //while running the unit-tests, but this might be because the
    //datasets are too small.
    //                    addOptimizerPipelineVectorization(TheFPM);
    
//#if MODULEPASS
//    /* OPTIMIZER PIPELINE, module passes */
//    PassManagerBuilder pmb;
//    pmb.OptLevel=3;
//    TheMPM = new ModulePassManager();
//    pmb.populateModulePassManager(*TheMPM);
//    addOptimizerPipelineInlining(TheMPM);
//#endif
//    ThePM = new legacy::PassManager();

    // ThePM->add(llvm::createNVPTXAssignValidGlobalNamesPass());

    // TargetPassConfig *TPC = ((LLVMTargetMachine *) ((GpuRawContext *) context)->TheTargetMachine.get())->createPassConfig(*ThePM);
    // ThePM->add(TPC);

    //PassManagerBuilder Builder;
    //Builder.OptLevel = 3;
    //((GpuRawContext *) context)->TheTargetMachine->adjustPassManager(Builder);
    //Builder.populateFunctionPassManager(*TheFPM);
    //Builder.populateModulePassManager(*ThePM);
    // Builder.populateLTOPassManager (*ThePM);

    // TheFPM->doInitialization();

    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * int64_type   = Type::getInt64Ty  (context->getLLVMContext());
    Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    std::vector<Type *> inputs3{3, int32_type};

    FunctionType *intr = FunctionType::get(int32_type, inputs3, false);
    
    registerFunction("llvm.nvvm.shfl.bfly.i32"              , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_bfly_i32)             );

    registerFunction("llvm.nvvm.shfl.idx.i32"               , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_shfl_idx_i32)              );

    registerFunction("llvm.nvvm.read.ptx.sreg.ntid.x"       , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_ntid_x)      );

    registerFunction("llvm.nvvm.read.ptx.sreg.tid.x"        , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_tid_x)       );

    registerFunction("llvm.nvvm.read.ptx.sreg.lanemask.lt"  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_lanemask_lt) );

    registerFunction("llvm.nvvm.read.ptx.sreg.lanemask.eq"  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_lanemask_eq) );
    
    registerFunction("llvm.nvvm.read.ptx.sreg.nctaid.x"     , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_nctaid_x)    );

    registerFunction("llvm.nvvm.read.ptx.sreg.ctaid.x"      , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_ctaid_x)     );

    registerFunction("llvm.nvvm.read.ptx.sreg.laneid"       , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_read_ptx_sreg_laneid)      );

    registerFunction("llvm.nvvm.membar.cta"                 , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_cta)                );
    registerFunction("threadfence_block"                    , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_cta)                );

    registerFunction("llvm.nvvm.membar.gl"                  , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_gl)                 );
    registerFunction("threadfence"                          , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_gl)                 );

    registerFunction("llvm.nvvm.membar.sys"                 , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_membar_sys)                );

    registerFunction("llvm.nvvm.barrier0"                   , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_barrier0)                  );
    registerFunction("syncthreads"                          , Intrinsic::getDeclaration(getModule(), Intrinsic::nvvm_barrier0)                  );

    registerFunction("llvm.ctpop"                           , Intrinsic::getDeclaration(getModule(), Intrinsic::ctpop         , int32_type     ));

    FunctionType *intrprinti64 = FunctionType::get(void_type, std::vector<Type *>{int64_type}, false);
    Function *intr_pprinti64 = Function::Create(intrprinti64, Function::ExternalLinkage, "dprinti64", getModule());
    registerFunction("printi64", intr_pprinti64);
    
    FunctionType *intrget_buffers = FunctionType::get(charPtrType, std::vector<Type *>{}, false);
    Function *intr_pget_buffers = Function::Create(intrget_buffers, Function::ExternalLinkage, "get_buffers", getModule());
    registerFunction("get_buffers", intr_pget_buffers);

    FunctionType *intrrelease_buffers = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *intr_prelease_buffers = Function::Create(intrrelease_buffers, Function::ExternalLinkage, "release_buffers", getModule());
    registerFunction("release_buffers", intr_prelease_buffers);

    FunctionType *allocate = FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
    Function *fallocate = Function::Create(allocate, Function::ExternalLinkage, "allocate_gpu", getModule());
    std::vector<std::pair<unsigned, Attribute>> attrs;
    Attribute noAlias  = Attribute::get(context->getLLVMContext(), Attribute::AttrKind::NoAlias);
    attrs.emplace_back(0, noAlias);
    fallocate->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    registerFunction("allocate", fallocate);

    FunctionType *deallocate = FunctionType::get(void_type, std::vector<Type *>{charPtrType}, false);
    Function *fdeallocate = Function::Create(deallocate, Function::ExternalLinkage, "deallocate_gpu", getModule());
    registerFunction("deallocate", fdeallocate);

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
};

void RawGpuPipelineGen::init(){
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

    auto GPU      = "sm_61";//sys::getHostCPUName(); //FIXME: for now it produces faster code... LLVM 6.0.0 improves the scheduler for our system

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

    auto RM = Optional<Reloc::Model>();
    TheTargetMachine = (LLVMTargetMachine *) Target->createTargetMachine(TargetTriple, GPU, 
                                                    "+ptx61",
                                                    opt, RM, 
                                                    CodeModel::Model::Default, 
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

void RawGpuPipelineGen::optimizeModule(Module * M){
    time_block t("Optimization time: ");
    FPasses.reset(new legacy::FunctionPassManager(M));
    FPasses->add(createTargetTransformInfoWrapperPass(TheTargetMachine->getTargetIRAnalysis()));

    Builder.populateFunctionPassManager(*FPasses);

    FPasses->doInitialization();
    for (Function &F : *M) FPasses->run(F);
    FPasses->doFinalization();
    
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
}

size_t RawGpuPipelineGen::prepareStateArgument(){
    LLVMContext &TheContext     = context->getLLVMContext();

    Type *int32Type             = Type::getInt32Ty(TheContext);
    
    if (state_vars.empty()) appendStateVar(int32Type); //FIMXE: should not be necessary... there should be some way to bypass it...

    state_type                  = StructType::create(state_vars, pipName + "_state_t");
    size_t state_id             = appendParameter(state_type, false, false);//true);

    return state_id;
}

Value * RawGpuPipelineGen::getStateLLVMValue(){
    return getArgument(args.size() - 1);
}

Function * RawGpuPipelineGen::prepare(){
    assert(!F);
    RawPipelineGen::prepare();

    LLVMContext &TheContext = context->getLLVMContext();

    Type *int32Type           = Type::getInt32Ty(TheContext);
    
    std::vector<llvm::Metadata *> Vals;

    NamedMDNode * annot = getModule()->getOrInsertNamedMetadata("nvvm.annotations");
    MDString    * str   = MDString::get(TheContext, "kernel");
    Value       * one   = ConstantInt::get(int32Type, 1);

    Vals.push_back(ValueAsMetadata::get(F));
    Vals.push_back(str);
    Vals.push_back(ValueAsMetadata::getConstant(one));
    
    MDNode * mdNode = MDNode::get(TheContext, Vals);

    annot->addOperand(mdNode);

    return F;
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


void RawGpuPipelineGen::compileAndLoad(){
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    time_block t(pipName + " G: ");

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code/" + pipName + ".ll", EC, (llvm::sys::fs::OpenFlags) 0); // FIXME: llvm::sys::fs::OpenFlags::F_NONE is the correct one but it gives a compilation error

        getModule()->print(out, nullptr, false, true);
    }
#endif

    optimizeModule(getModule());

    // ThePM->run(*getModule());

    // JIT the function, returning a function pointer.
    // TheExecutionEngine->finalizeObject();
    // void *FPtr = TheExecutionEngine->getPointerToFunction(F);

    // int (*FP)(void) = (int (*)(void))FPtr;
    // assert(FP != nullptr && "Code generation failed!");


    // //TheModule->dump();
    // //Run function
    // struct timespec t0, t1;
    // clock_gettime(CLOCK_REALTIME, &t0);
    // int jitFuncResult = FP();
    // //LOG(INFO) << "Mock return value of generated function " << FP(11);
    // clock_gettime(CLOCK_REALTIME, &t1);
    // printf("(Already compiled) Execution took %f seconds\n",diff(t0, t1));
    // cout << "Return flag: " << jitFuncResult << endl;

    // TheFPM = 0;
    //Dump to see final (optimized) form
#ifdef DEBUGCTX
    // getModule()->dump();
    
    {
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
        TheTargetMachine->addPassesToEmitFile(PM, ostream, llvm::TargetMachine::CGFT_AssemblyFile, false);

        PM.run(*(getModule()));
    } // flushes stream and ostream
#ifdef DEBUGCTX
    {
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
        
        constexpr size_t opt_size = 3;
        CUjit_option options[opt_size];
        void       * values [opt_size];

        options [0] = CU_JIT_TARGET_FROM_CUCONTEXT;
        values  [0] = 0;
        options [1] = CU_JIT_ERROR_LOG_BUFFER;
        values  [1] = (void *) error_log;
        options [2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        values  [2] = (void *) BUFFER_SIZE;
        // options [3] = CU_JIT_INFO_LOG_BUFFER;
        // values  [3] = (void *) info_log;
        // options [4] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        // values  [4] = (void *) BUFFER_SIZE;

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
            printf("[CUcompile: ] %s\n", info_log );
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX, "/home/chrysoge/Documents/pelago/src/raw-jit-executor/codegen/device_funcs.ptx", 0, NULL, NULL));
        // gpu_run(cuLinkAddFile (linkState, CU_JIT_INPUT_PTX, ("generated_code/" + pipName + ".ptx").c_str(), 0, NULL, NULL));
        x = cuLinkAddData (linkState, CU_JIT_INPUT_PTX, (void *) ptx.c_str(), ptx.length() + 1, NULL, 0, NULL, NULL);
        if (x != CUDA_SUCCESS) {
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }
        x = cuLinkComplete(linkState, &cubin, &cubinSize);
        if (x != CUDA_SUCCESS) {
            printf("[CUcompile: ] %s\n", error_log);
            gpu_run(x);
        }

        int devices = get_num_of_gpus();
        for (int i = 0 ; i < devices ; ++i){
            time_block t("TloadModule: ");
            set_device_on_scope d(i);

            // gpu_run(cuModuleLoadDataEx(&cudaModule[i], ptx.c_str(), 0, 0, 0));
            gpu_run(cuModuleLoadFatBinary(&cudaModule[i], cubin));
            {
                time_block t("TinitModule: ");
                initializeModule(cudaModule[i]);
            }
        }
        
        gpu_run(cuLinkDestroy (linkState));
    }
    func_name = F->getName().str();

    // F->eraseFromParent();
    // F = NULL;
}

void * RawGpuPipelineGen::getKernel() const{
    assert(func_name != "");
    // assert(!F);

    CUfunction func;
    gpu_run(cuModuleGetFunction(&func, cudaModule[get_current_gpu()], func_name.c_str()));
    
    return (void *) func;
}

RawPipeline * RawGpuPipelineGen::getPipeline(int group_id){
    // assert(false);
    void       * func       = getKernel();
    // return NULL;
    return new RawPipeline(func, (getModule()->getDataLayout().getTypeSizeInBits(state_type) + 7) / 8, this, state_type, openers, closers, TheExecutionEngine->getPointerToFunction(open__function), TheExecutionEngine->getPointerToFunction(close_function), group_id);
}

void * RawGpuPipelineGen::getCompiledFunction(Function * f){
    return TheExecutionEngine->getPointerToFunction(f);
}