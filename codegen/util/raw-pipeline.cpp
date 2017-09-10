#include "util/raw-pipeline.hpp"
#include "common/gpu/gpu-common.hpp"
#include "util/gpu/gpu-raw-context.hpp"

#include <thread>

size_t RawPipelineGen::appendParameter(llvm::Type * ptype, bool noalias, bool readonly){
    inputs.push_back(ptype);
    inputs_noalias.push_back(noalias);
    inputs_readonly.push_back(readonly);

    return inputs.size() - 1;
}

size_t RawPipelineGen::appendStateVar(llvm::Type * ptype){
    state_vars.push_back(ptype);

    return state_vars.size() - 1;
}

std::vector<llvm::Type *> RawPipelineGen::getStateVars() const{
    return state_vars;
}

Argument * RawPipelineGen::getArgument(size_t id) const{
    assert(id < args.size());
    return args[id];
}

Value * RawPipelineGen::getStateVar() const{
    assert(state);
    return state; //getArgument(args.size() - 1);
}

Value * RawPipelineGen::getStateVar(size_t id) const{
    Value * arg = getStateVar();
    assert(id < state_vars.size());
    return context->getBuilder()->CreateExtractValue(arg, id);
}

Value * RawPipelineGen::getSubStateVar() const{
    assert(copyStateFrom);
    Value * subState = getStateVar(0);
    subState->setName("subState");
    return subState;
}

extern "C"{
    void yield(){
        std::this_thread::yield();
    }
}

RawPipelineGen::RawPipelineGen(RawContext * context, std::string pipName, RawPipelineGen * copyStateFrom, bool initEngine): 
            F(nullptr), pipName(pipName), context(context), copyStateFrom(copyStateFrom){
    TheModule  = new Module(pipName, context->getLLVMContext());
    TheBuilder = new IRBuilder<>(context->getLLVMContext());
    
    state      = NULL;

    if (copyStateFrom){
        Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());
        appendStateVar(charPtrType);
    }


    TheExecutionEngine = nullptr;

    if (initEngine){
        /* OPTIMIZER PIPELINE, function passes */
        TheFPM = new legacy::FunctionPassManager(getModule());
        addOptimizerPipelineDefault(TheFPM);
        TheFPM->add(createLoadCombinePass());


        //LSC: Seems to be faster without the vectorization, at least
        //while running the unit-tests, but this might be because the
        //datasets are too small.
        //addOptimizerPipelineVectorization(TheFPM);
        
#if MODULEPASS
        /* OPTIMIZER PIPELINE, module passes */
        PassManagerBuilder pmb;
        pmb.OptLevel=3;
        TheMPM = new ModulePassManager();
        pmb.populateModulePassManager(*TheMPM);
        addOptimizerPipelineInlining(TheMPM);
#endif

        TheFPM->doInitialization();

        Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
        Type * int64_type   = Type::getInt64Ty  (context->getLLVMContext());
        Type * void_type    = Type::getVoidTy   (context->getLLVMContext());
        Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

        Type * size_type;
        if      (sizeof(size_t) == 4) size_type = int32_type;
        else if (sizeof(size_t) == 8) size_type = int64_type;
        else                          assert(false);

        FunctionType * FTlaunch_kernel        = FunctionType::get(
                                                        void_type, 
                                                        std::vector<Type *>{charPtrType, PointerType::get(charPtrType, 0)}, 
                                                        false
                                                    );

        Function * launch_kernel_             = Function::Create(
                                                        FTlaunch_kernel,
                                                        Function::ExternalLinkage, 
                                                        "launch_kernel", 
                                                        getModule()
                                                    );

        registerFunction("launch_kernel",launch_kernel_);


        FunctionType *make_mem_move_device = FunctionType::get(charPtrType, std::vector<Type *>{charPtrType, size_type, int32_type, charPtrType}, false);
        Function *fmake_mem_move_device = Function::Create(make_mem_move_device, Function::ExternalLinkage, "make_mem_move_device", getModule());
        registerFunction("make_mem_move_device", fmake_mem_move_device);


        FunctionType *acquireBuffer = FunctionType::get(charPtrType, std::vector<Type *>{int32_type, charPtrType}, false);
        Function *facquireBuffer = Function::Create(acquireBuffer, Function::ExternalLinkage, "acquireBuffer", getModule());
        registerFunction("acquireBuffer", facquireBuffer);


        FunctionType *releaseBuffer = FunctionType::get(void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType}, false);
        Function *freleaseBuffer = Function::Create(releaseBuffer, Function::ExternalLinkage, "releaseBuffer", getModule());
        registerFunction("releaseBuffer", freleaseBuffer);

        FunctionType *freeBuffer = FunctionType::get(void_type, std::vector<Type *>{int32_type, charPtrType, charPtrType}, false);
        Function *ffreeBuffer = Function::Create(freeBuffer, Function::ExternalLinkage, "freeBuffer", getModule());
        registerFunction("freeBuffer", ffreeBuffer);


        FunctionType *crand = FunctionType::get(int32_type, std::vector<Type *>{}, false);
        Function *fcrand = Function::Create(crand, Function::ExternalLinkage, "rand", getModule());
        registerFunction("rand", fcrand);

        FunctionType *get_buffer = FunctionType::get(charPtrType, std::vector<Type *>{size_type}, false);
        Function *fget_buffer = Function::Create(get_buffer, Function::ExternalLinkage, "get_buffer", getModule());
        registerFunction("get_buffer", fget_buffer);

        FunctionType *yield = FunctionType::get(void_type, std::vector<Type *>{}, false);
        Function *fyield = Function::Create(yield, Function::ExternalLinkage, "yield", getModule());
        registerFunction("yield", fyield);

        registerFunctions(); //FIXME: do we have to register them every time ?

        string ErrStr;
        TheExecutionEngine =
            EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr).create();
        if (TheExecutionEngine == nullptr) {
            fprintf(stderr, "Could not create ExecutionEngine: %s\n",
                    ErrStr.c_str());
            exit(1);
        }
    }
};

GpuRawPipelineGen::GpuRawPipelineGen(RawContext * context, std::string pipName, RawPipelineGen * copyStateFrom): 
            RawPipelineGen(context, pipName, copyStateFrom, false){
    // getModule()->setDataLayout(((GpuRawContext *) context)->TheTargetMachine->createDataLayout());
    cudaModule = (CUmodule *) malloc(get_num_of_gpus() * sizeof(CUmodule));

    TheFPM = new legacy::FunctionPassManager(getModule());
    addOptimizerPipelineDefault(TheFPM);

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
    addOptimizerPipelineVectorization(TheFPM);
    
#if MODULEPASS
    /* OPTIMIZER PIPELINE, module passes */
    PassManagerBuilder pmb;
    pmb.OptLevel=3;
    TheMPM = new ModulePassManager();
    pmb.populateModulePassManager(*TheMPM);
    addOptimizerPipelineInlining(TheMPM);
#endif

    TheFPM->doInitialization();

    if (sizeof(void*) == 8) {
        getModule()->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                           "v64:64:64-v128:128:128-n16:32:64");
        getModule()->setTargetTriple("nvptx64-nvidia-cuda");
    } else {
        getModule()->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                           "v64:64:64-v128:128:128-n16:32:64");
        getModule()->setTargetTriple("nvptx-nvidia-cuda");
    }


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
    
    Function *intr_p = Function::Create(intr, Function::ExternalLinkage, "llvm.nvvm.shfl.bfly.i32", getModule());
    registerFunction("llvm.nvvm.shfl.bfly.i32", intr_p);

    FunctionType *intr2 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p2 = Function::Create(intr2, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ntid.x", getModule());
    registerFunction("llvm.nvvm.read.ptx.sreg.ntid.x", intr_p2);

    FunctionType *intr3 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p3 = Function::Create(intr3, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.tid.x", getModule());
    registerFunction("llvm.nvvm.read.ptx.sreg.tid.x", intr_p3);

    FunctionType *intr2b = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p2b = Function::Create(intr2b, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.nctaid.x", getModule());
    registerFunction("llvm.nvvm.read.ptx.sreg.nctaid.x", intr_p2b);

    FunctionType *intr3b = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p3b = Function::Create(intr3b, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.ctaid.x", getModule());
    registerFunction("llvm.nvvm.read.ptx.sreg.ctaid.x", intr_p3b);

    FunctionType *intr4 = FunctionType::get(int32_type, std::vector<Type *>{}, false);
    Function *intr_p4 = Function::Create(intr4, Function::ExternalLinkage, "llvm.nvvm.read.ptx.sreg.laneid", getModule());
    registerFunction("llvm.nvvm.read.ptx.sreg.laneid", intr_p4);

    FunctionType *intrmembargl = FunctionType::get(void_type, std::vector<Type *>{}, false);
    Function *intr_pmembargl = Function::Create(intrmembargl, Function::ExternalLinkage, "llvm.nvvm.membar.gl", getModule());
    registerFunction("llvm.nvvm.membar.gl", intr_pmembargl);

    FunctionType *intrmembarsys = FunctionType::get(void_type, std::vector<Type *>{}, false);
    Function *intr_pmembarsys = Function::Create(intrmembarsys, Function::ExternalLinkage, "llvm.nvvm.membar.sys", getModule());
    registerFunction("llvm.nvvm.membar.sys", intr_pmembargl);


    string ErrStr;
    TheExecutionEngine =
        EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr).create();
    if (TheExecutionEngine == nullptr) {
        fprintf(stderr, "Could not create ExecutionEngine: %s\n",
                ErrStr.c_str());
        exit(1);
    }
};

void RawPipelineGen::registerFunction(const char* funcName, Function* func) {
    availableFunctions[funcName] = func;
}

Function * const RawPipelineGen::getFunction(string funcName) const {
    map<string, Function*>::const_iterator it;
    it = availableFunctions.find(funcName);
    if (it == availableFunctions.end()) {
            throw runtime_error(string("Unknown function name: ") + funcName + " (" + pipName + ")");
    }
    return it->second;
}


size_t RawPipelineGen::prepareStateArgument(){
    LLVMContext &TheContext     = context->getLLVMContext();

    Type *int32Type             = Type::getInt32Ty(TheContext);
    
    if (state_vars.empty()) appendStateVar(int32Type); //FIMXE: should not be necessary... there should be some way to bypass it...

    state_type                  = StructType::create(state_vars, pipName + "_state_t");
    size_t state_id             = appendParameter(PointerType::get(state_type, 0), true, true);//true);

    return state_id;
}

size_t GpuRawPipelineGen::prepareStateArgument(){
    LLVMContext &TheContext     = context->getLLVMContext();

    Type *int32Type             = Type::getInt32Ty(TheContext);
    
    if (state_vars.empty()) appendStateVar(int32Type); //FIMXE: should not be necessary... there should be some way to bypass it...

    state_type                  = StructType::create(state_vars, pipName + "_state_t");
    size_t state_id             = appendParameter(state_type, false, false);//true);

    return state_id;
}


void RawPipelineGen::prepareFunction(){
    FunctionType *ftype = FunctionType::get(Type::getVoidTy(context->getLLVMContext()), inputs, false);
    //use f_num to overcome an llvm bu with keeping dots in function names when generating PTX (which is invalid in PTX)
    F = Function::Create(ftype, Function::ExternalLinkage, pipName, context->getModule());

    for (size_t i = 1 ; i <= inputs.size() ; ++i){ //+1 because 0 is the return value
        if (inputs_readonly[i - 1]) F->setOnlyReadsMemory(i);
        if (inputs_noalias [i - 1]) F->setDoesNotAlias(   i);
    }

    for (auto &t: F->args()) args.push_back(&t);

    BasicBlock *BB = BasicBlock::Create(context->getLLVMContext(), "entry", F);
    getBuilder()->SetInsertPoint(BB);
}

Value * RawPipelineGen::getStateLLVMValue(){
    return getBuilder()->CreateLoad(getArgument(args.size() - 1));
}

Value * GpuRawPipelineGen::getStateLLVMValue(){
    return getArgument(args.size() - 1);
}

Function * RawPipelineGen::prepare(){
    assert(!F);
    std::cout << pipName << " prepare" << std::endl;

    size_t state_id = prepareStateArgument();
    
    prepareFunction();

    //Get the ENTRY BLOCK
    // context->setCurrentEntryBlock(F->getEntryBlock());

    Argument * stateArg     = getArgument(state_id);
    stateArg->setName("state_ptr");

    state                   = getStateLLVMValue();
    state->setName("state");
    
    return F;
}

Function * GpuRawPipelineGen::prepare(){
    assert(!F);
    RawPipelineGen::prepare();

    LLVMContext &TheContext = context->getLLVMContext();

    Type *int32Type           = Type::getInt32Ty(TheContext);
    
    std::vector<llvm::Metadata *> Vals;

    NamedMDNode * annot = context->getModule()->getOrInsertNamedMetadata("nvvm.annotations");
    MDString    * str   = MDString::get(TheContext, "kernel");
    Value       * one   = ConstantInt::get(int32Type, 1);

    Vals.push_back(ValueAsMetadata::get(F));
    Vals.push_back(str);
    Vals.push_back(ValueAsMetadata::getConstant(one));
    
    MDNode * mdNode = MDNode::get(TheContext, Vals);

    annot->addOperand(mdNode);

    return F;
}

void RawPipelineGen::compileAndLoad(){
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    std::cout << pipName << " C" << std::endl;

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code_" + pipName + ".ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif

    // Validate the generated code, checking for consistency.
    verifyFunction(*F);

    // Optimize the function.
    TheFPM->run(*F);
#if MODULEPASS
    TheMPM->runOnModule(getModule());
#endif

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code_" + pipName + "_opt.ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif
    // JIT the function, returning a function pointer.
    TheExecutionEngine->finalizeObject();
    func = TheExecutionEngine->getPointerToFunction(F);
    assert(func);

    // F->eraseFromParent();
    F = NULL;
}


void GpuRawPipelineGen::compileAndLoad(){
    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
    std::cout << pipName << " G" << std::endl;

#ifdef DEBUGCTX
    // getModule()->dump();

    {
        std::error_code EC;
        raw_fd_ostream out("generated_code_" + pipName + ".ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif

    // Validate the generated code, checking for consistency.
    verifyFunction(*F);

//     // Optimize the function.
    TheFPM->run(*F);
#if MODULEPASS
    TheMPM->runOnModule(getModule());
#endif

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
        raw_fd_ostream out("generated_code_" + pipName + "_opt.ll", EC, sys::fs::F_None);

        getModule()->print(out, nullptr, false, true);
    }
#endif

    string ptx;
    {
        raw_string_ostream stream(ptx);
        buffer_ostream ostream(stream);
        
        legacy::PassManager PM;

        // Ask the target to add backend passes as necessary.
        ((GpuRawContext *) context)->TheTargetMachine->addPassesToEmitFile(PM, ostream, llvm::TargetMachine::CGFT_AssemblyFile, false);

        PM.run(*(getModule()));
    } // flushes stream and ostream
#ifdef DEBUGCTX
    {
        std::ofstream optx("generated_ptx_" + pipName + ".ptx");
        optx << ptx;
    }
#endif


    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i){
        time_block t("TloadModule: ");
        set_device_on_scope d(i);

        gpu_run(cuModuleLoadDataEx(&cudaModule[i], ptx.c_str(), 0, 0, 0));
    }
    func_name = F->getName().str();

    // F->eraseFromParent();
    F = NULL;
}

Function * RawPipelineGen::getFunction() const{
    assert(F);
    return F;
}


void * RawPipelineGen::getKernel() const{
    assert(func != nullptr);
    assert(!F);
    return (void *) func;
}

void * GpuRawPipelineGen::getKernel() const{
    assert(func_name != "");
    assert(!F);

    CUfunction func;
    gpu_run(cuModuleGetFunction(&func, cudaModule[get_current_gpu()], func_name.c_str()));
    
    return (void *) func;
}


void RawPipelineGen::registerOpen (std::function<void (RawPipeline * pip)> open ){
    openers.push_back(open );
}

void RawPipelineGen::registerClose(std::function<void (RawPipeline * pip)> close){
    closers.push_back(close);
}


RawPipeline * RawPipelineGen::getPipeline(int group_id){
    void       * func       = getKernel();

    std::vector<std::function<void (RawPipeline * pip)>> openers{this->openers};
    std::vector<std::function<void (RawPipeline * pip)>> closers{this->closers};

    if (copyStateFrom){
        RawPipeline * copyFrom = copyStateFrom->getPipeline(group_id);

        openers.push_back([copyFrom](RawPipeline * pip){copyFrom->open ();});
        openers.push_back([copyFrom](RawPipeline * pip){pip->setStateVar(0, copyFrom->state);});

        // openers.push_back([copyFrom](RawPipeline * pip){std::cout << " ASDASd " << std::endl; pip->copyStateFrom  (copyFrom);});
        // closers.push_back([copyFrom](RawPipeline * pip){pip->copyStateBackTo(copyFrom);});
        closers.push_back([copyFrom](RawPipeline * pip){copyFrom->close();});
    }

    return new RawPipeline(func, (getModule()->getDataLayout().getTypeSizeInBits(state_type) + 7) / 8, this, state_type, openers, closers, group_id);
}

RawPipeline * GpuRawPipelineGen::getPipeline(int group_id){
    // assert(false);
    void       * func       = getKernel();
    // return NULL;
    return new RawPipeline(func, (getModule()->getDataLayout().getTypeSizeInBits(state_type) + 7) / 8, this, state_type, openers, closers, group_id);
}

RawPipeline::RawPipeline(void * f, size_t state_size, RawPipelineGen * gen, StructType * state_type,
        const std::vector<std::function<void (RawPipeline * pip)>> &openers,
        const std::vector<std::function<void (RawPipeline * pip)>> &closers,
        int32_t group_id): 
            cons(f), state_type(state_type), openers(openers), closers(closers), group_id(group_id), layout(gen->getModule()->getDataLayout()), state_size(state_size){
    state = malloc(state_size); //(getModule()->getDataLayout().getTypeSizeInBits(state_type) + 7) / 8);
    assert(state);
}

// GpuRawPipeline::GpuRawPipeline(void * f, size_t state_size, RawPipelineGen * gen, StructType * state_type,
//         const std::vector<std::function<void (RawPipeline * pip)>> &openers,
//         const std::vector<std::function<void (RawPipeline * pip)>> &closers,
//         int32_t group_id): 
//             RawPipeline(f, state_size, gen, state_type, openers, closers, group_id){}

RawPipeline::~RawPipeline(){
    free(state);
}

size_t RawPipeline::getSizeOf(llvm::Type * t) const{
    return layout.getTypeSizeInBits(t)/8;
}

int32_t RawPipeline::getGroup() const{
    return group_id;
}

void RawPipeline::open(){
    for (const auto &opener: openers) opener(this);
}

void RawPipeline::close(){
    for (size_t i = closers.size() ; i > 0 ; --i) closers[i - 1](this);
}



/*
 * For now, copied  from util/raw-functions.cpp and transformed for RawPipeline
 * 
 * TODO: deduplicate code...
 */
void RawPipelineGen::registerFunctions()    {
    LLVMContext& ctx = context->getLLVMContext();
    Module* TheModule = getModule();
    assert(TheModule != nullptr);

    Type* int1_bool_type = Type::getInt1Ty(ctx);
    Type* int8_type = Type::getInt8Ty(ctx);
    Type* int16_type = Type::getInt16Ty(ctx);
    Type* int32_type = Type::getInt32Ty(ctx);
    Type* int64_type = Type::getInt64Ty(ctx);
    Type* void_type = Type::getVoidTy(ctx);
    Type* double_type = Type::getDoubleTy(ctx);
    StructType* strObjType = context->CreateStringStruct();
    PointerType* void_ptr_type = PointerType::get(int8_type, 0);
    PointerType* char_ptr_type = PointerType::get(int8_type, 0);
    PointerType* int32_ptr_type = PointerType::get(int32_type, 0);

    vector<Type*> Ints8Ptr(1,Type::getInt8PtrTy(ctx));
    vector<Type*> Ints8(1,int8_type);
    vector<Type*> Ints1(1,int1_bool_type);
    vector<Type*> Ints(1,int32_type);
    vector<Type*> Ints64(1,int64_type);
    vector<Type*> Floats(1,double_type);
    vector<Type*> Shorts(1,int16_type);

    vector<Type*> ArgsCmpTokens;
    ArgsCmpTokens.insert(ArgsCmpTokens.begin(),char_ptr_type);
    ArgsCmpTokens.insert(ArgsCmpTokens.begin(),int32_type);
    ArgsCmpTokens.insert(ArgsCmpTokens.begin(),int32_type);
    ArgsCmpTokens.insert(ArgsCmpTokens.begin(),char_ptr_type);

    vector<Type*> ArgsCmpTokens64;
    ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);
    ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
    ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), int64_type);
    ArgsCmpTokens64.insert(ArgsCmpTokens64.begin(), char_ptr_type);

    vector<Type*> ArgsConvBoolean;
    ArgsConvBoolean.insert(ArgsConvBoolean.begin(),int32_type);
    ArgsConvBoolean.insert(ArgsConvBoolean.begin(),int32_type);
    ArgsConvBoolean.insert(ArgsConvBoolean.begin(),char_ptr_type);

    vector<Type*> ArgsConvBoolean64;
    ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),int64_type);
    ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),int64_type);
    ArgsConvBoolean64.insert(ArgsConvBoolean64.begin(),char_ptr_type);

    vector<Type*> ArgsAtois;
    ArgsAtois.insert(ArgsAtois.begin(),int32_type);
    ArgsAtois.insert(ArgsAtois.begin(),char_ptr_type);

    vector<Type*> ArgsStringObjCmp;
    ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(),strObjType);
    ArgsStringObjCmp.insert(ArgsStringObjCmp.begin(),strObjType);

    vector<Type*> ArgsStringCmp;
    ArgsStringCmp.insert(ArgsStringCmp.begin(), char_ptr_type);
    ArgsStringCmp.insert(ArgsStringCmp.begin(),char_ptr_type);

    /**
     * Args of functions computing hash
     */
    vector<Type*> ArgsHashInt;
    ArgsHashInt.insert(ArgsHashInt.begin(),int32_type);

    vector<Type*> ArgsHashDouble;
    ArgsHashDouble.insert(ArgsHashDouble.begin(),double_type);

    vector<Type*> ArgsHashStringC;
    ArgsHashStringC.insert(ArgsHashStringC.begin(),int64_type);
    ArgsHashStringC.insert(ArgsHashStringC.begin(),int64_type);
    ArgsHashStringC.insert(ArgsHashStringC.begin(),char_ptr_type);

    vector<Type*> ArgsHashStringObj;
    ArgsHashStringObj.insert(ArgsHashStringObj.begin(),strObjType);

    vector<Type*> ArgsHashBoolean;
    ArgsHashBoolean.insert(ArgsHashBoolean.begin(),int1_bool_type);

    vector<Type*> ArgsHashCombine;
    ArgsHashCombine.insert(ArgsHashCombine.begin(),int64_type);
    ArgsHashCombine.insert(ArgsHashCombine.begin(),int64_type);

    /**
     * Args of functions computing flush
     */
    vector<Type*> ArgsFlushInt;
    ArgsFlushInt.insert(ArgsFlushInt.begin(),char_ptr_type);
    ArgsFlushInt.insert(ArgsFlushInt.begin(),int32_type);

    vector<Type*> ArgsFlushInt64;
    ArgsFlushInt64.insert(ArgsFlushInt64.begin(),char_ptr_type);
    ArgsFlushInt64.insert(ArgsFlushInt64.begin(),int64_type);

    vector<Type*> ArgsFlushDouble;
    ArgsFlushDouble.insert(ArgsFlushDouble.begin(),char_ptr_type);
    ArgsFlushDouble.insert(ArgsFlushDouble.begin(),double_type);

    vector<Type*> ArgsFlushStringC;
    ArgsFlushStringC.insert(ArgsFlushStringC.begin(),char_ptr_type);
    ArgsFlushStringC.insert(ArgsFlushStringC.begin(),int64_type);
    ArgsFlushStringC.insert(ArgsFlushStringC.begin(),int64_type);
    ArgsFlushStringC.insert(ArgsFlushStringC.begin(),char_ptr_type);

    vector<Type*> ArgsFlushStringCv2;
    ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(),char_ptr_type);
    ArgsFlushStringCv2.insert(ArgsFlushStringCv2.begin(),char_ptr_type);

    vector<Type*> ArgsFlushStringObj;
    ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(),char_ptr_type);
    ArgsFlushStringObj.insert(ArgsFlushStringObj.begin(),strObjType);

    vector<Type*> ArgsFlushBoolean;
    ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(),int1_bool_type);
    ArgsFlushBoolean.insert(ArgsFlushBoolean.begin(),char_ptr_type);

    vector<Type*> ArgsFlushStartEnd;
    ArgsFlushStartEnd.insert(ArgsFlushStartEnd.begin(),char_ptr_type);

    vector<Type*> ArgsFlushChar;
    ArgsFlushChar.insert(ArgsFlushChar.begin(),char_ptr_type);
    ArgsFlushChar.insert(ArgsFlushChar.begin(),int8_type);

    vector<Type*> ArgsFlushDelim;
    ArgsFlushDelim.insert(ArgsFlushDelim.begin(),char_ptr_type);
    ArgsFlushDelim.insert(ArgsFlushDelim.begin(),int8_type);
    ArgsFlushDelim.insert(ArgsFlushDelim.begin(),int64_type);

    vector<Type*> ArgsFlushOutput;
    ArgsFlushOutput.insert(ArgsFlushOutput.begin(),char_ptr_type);

    vector<Type*> ArgsMemoryChunk;
    ArgsMemoryChunk.insert(ArgsMemoryChunk.begin(),int64_type);
    vector<Type*> ArgsIncrMemoryChunk;
    ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(),int64_type);
    ArgsIncrMemoryChunk.insert(ArgsIncrMemoryChunk.begin(),void_ptr_type);
    vector<Type*> ArgsRelMemoryChunk;
    ArgsRelMemoryChunk.insert(ArgsRelMemoryChunk.begin(),void_ptr_type);

    /**
     * Args of timing functions
     */
    //Empty on purpose
    vector<Type*> ArgsTiming;


    FunctionType *FTint                   = FunctionType::get(Type::getInt32Ty(ctx), Ints, false);
    FunctionType *FTint64                 = FunctionType::get(Type::getInt32Ty(ctx), Ints64, false);
    FunctionType *FTcharPtr               = FunctionType::get(Type::getInt32Ty(ctx), Ints8Ptr, false);
    FunctionType *FTatois                 = FunctionType::get(int32_type, ArgsAtois, false);
    FunctionType *FTatof                  = FunctionType::get(double_type, Ints8Ptr, false);
    FunctionType *FTprintFloat_           = FunctionType::get(int32_type, Floats, false);
    FunctionType *FTprintShort_           = FunctionType::get(int16_type, Shorts, false);
    FunctionType *FTcompareTokenString_   = FunctionType::get(int32_type, ArgsCmpTokens, false);
    FunctionType *FTcompareTokenString64_ = FunctionType::get(int32_type, ArgsCmpTokens64, false);
    FunctionType *FTconvertBoolean_       = FunctionType::get(int1_bool_type, ArgsConvBoolean, false);
    FunctionType *FTconvertBoolean64_     = FunctionType::get(int1_bool_type, ArgsConvBoolean64, false);
    FunctionType *FTprintBoolean_         = FunctionType::get(void_type, Ints1, false);
    FunctionType *FTcompareStringObjs     = FunctionType::get(int1_bool_type, ArgsStringObjCmp, false);
    FunctionType *FTcompareString         = FunctionType::get(int1_bool_type, ArgsStringCmp, false);
    FunctionType *FThashInt               = FunctionType::get(int64_type, ArgsHashInt, false);
    FunctionType *FThashDouble            = FunctionType::get(int64_type, ArgsHashDouble, false);
    FunctionType *FThashStringC           = FunctionType::get(int64_type, ArgsHashStringC, false);
    FunctionType *FThashStringObj         = FunctionType::get(int64_type, ArgsHashStringObj, false);
    FunctionType *FThashBoolean           = FunctionType::get(int64_type, ArgsHashBoolean, false);
    FunctionType *FThashCombine           = FunctionType::get(int64_type, ArgsHashCombine, false);
    FunctionType *FTflushInt              = FunctionType::get(void_type, ArgsFlushInt, false);
    FunctionType *FTflushInt64            = FunctionType::get(void_type, ArgsFlushInt64, false);
    FunctionType *FTflushDouble           = FunctionType::get(void_type, ArgsFlushDouble, false);
    FunctionType *FTflushStringC          = FunctionType::get(void_type, ArgsFlushStringC, false);
    FunctionType *FTflushStringCv2        = FunctionType::get(void_type, ArgsFlushStringCv2, false);
    FunctionType *FTflushStringObj        = FunctionType::get(void_type, ArgsFlushStringObj, false);
    FunctionType *FTflushBoolean          = FunctionType::get(void_type, ArgsFlushBoolean, false);
    FunctionType *FTflushStartEnd         = FunctionType::get(void_type, ArgsFlushStartEnd, false);
    FunctionType *FTflushChar             = FunctionType::get(void_type, ArgsFlushChar, false);
    FunctionType *FTflushDelim            = FunctionType::get(void_type, ArgsFlushDelim, false);
    FunctionType *FTflushOutput           = FunctionType::get(void_type, ArgsFlushOutput, false);

    FunctionType *FTmemoryChunk           = FunctionType::get(void_ptr_type, ArgsMemoryChunk, false);
    FunctionType *FTincrMemoryChunk       = FunctionType::get(void_ptr_type, ArgsIncrMemoryChunk, false);
    FunctionType *FTreleaseMemoryChunk    = FunctionType::get(void_type, ArgsRelMemoryChunk, false);

    FunctionType *FTtiming                = FunctionType::get(void_type, ArgsTiming, false);


    Function *printi_       = Function::Create(FTint, Function::ExternalLinkage,"printi", TheModule);
    Function *printi64_     = Function::Create(FTint64, Function::ExternalLinkage,"printi64", TheModule);
    Function *printc_       = Function::Create(FTcharPtr, Function::ExternalLinkage,"printc", TheModule);
    Function *printFloat_   = Function::Create(FTprintFloat_, Function::ExternalLinkage, "printFloat", TheModule);
    Function *printShort_   = Function::Create(FTprintShort_, Function::ExternalLinkage, "printShort", TheModule);
    Function *printBoolean_ = Function::Create(FTprintBoolean_, Function::ExternalLinkage, "printBoolean", TheModule);

    Function *atoi_     = Function::Create(FTcharPtr, Function::ExternalLinkage,"atoi", TheModule);
    Function *atois_    = Function::Create(FTatois, Function::ExternalLinkage,"atois", TheModule);
    atois_->addFnAttr(llvm::Attribute::AlwaysInline);
    Function *atof_     = Function::Create(FTatof, Function::ExternalLinkage,"atof", TheModule);

    Function *compareTokenString_   = Function::Create(FTcompareTokenString_,
            Function::ExternalLinkage, "compareTokenString", TheModule);
    compareTokenString_->addFnAttr(llvm::Attribute::AlwaysInline);
    Function *compareTokenString64_ = Function::Create(FTcompareTokenString64_,
                Function::ExternalLinkage, "compareTokenString64", TheModule);
    Function *stringObjEquality         = Function::Create(FTcompareStringObjs,
            Function::ExternalLinkage, "equalStringObjs", TheModule);
    stringObjEquality->addFnAttr(llvm::Attribute::AlwaysInline);
    Function *stringEquality = Function::Create(FTcompareString,
            Function::ExternalLinkage, "equalStrings", TheModule);
    stringEquality->addFnAttr(llvm::Attribute::AlwaysInline);

    Function *convertBoolean_   = Function::Create(FTconvertBoolean_,
                Function::ExternalLinkage, "convertBoolean", TheModule);
    convertBoolean_->addFnAttr(llvm::Attribute::AlwaysInline);
    Function *convertBoolean64_ = Function::Create(FTconvertBoolean64_,
                    Function::ExternalLinkage, "convertBoolean64", TheModule);
    convertBoolean64_->addFnAttr(llvm::Attribute::AlwaysInline);

    /**
     * Hashing
     */
    Function *hashInt_ = Function::Create(FThashInt, Function::ExternalLinkage,
            "hashInt", TheModule);
    Function *hashDouble_ = Function::Create(FThashDouble,
            Function::ExternalLinkage, "hashDouble", TheModule);
    Function *hashStringC_ = Function::Create(FThashStringC,
            Function::ExternalLinkage, "hashStringC", TheModule);
    Function *hashStringObj_ = Function::Create(FThashStringObj,
            Function::ExternalLinkage, "hashStringObject", TheModule);
    Function *hashBoolean_ = Function::Create(FThashBoolean,
            Function::ExternalLinkage, "hashBoolean", TheModule);
    Function *hashCombine_ = Function::Create(FThashCombine,
            Function::ExternalLinkage, "combineHashes", TheModule);
    Function *hashCombineNoOrder_ = Function::Create(FThashCombine,
            Function::ExternalLinkage, "combineHashesNoOrder", TheModule);

#if 0
    /**
     * Debug (TMP)
     */
    vector<Type*> ArgsDebug;
    ArgsDebug.insert(ArgsDebug.begin(),void_ptr_type);
    FunctionType *FTdebug = FunctionType::get(void_type, ArgsDebug, false);
    Function *debug_ = Function::Create(FTdebug, Function::ExternalLinkage,
                "debug", TheModule);
#endif

    /**
    * Flushing
    */
    Function *flushInt_ = Function::Create(FTflushInt,
            Function::ExternalLinkage, "flushInt", TheModule);
    Function *flushInt64_ = Function::Create(FTflushInt64,
                Function::ExternalLinkage, "flushInt64", TheModule);
    Function *flushDouble_ = Function::Create(FTflushDouble,
            Function::ExternalLinkage, "flushDouble", TheModule);
    Function *flushStringC_ = Function::Create(FTflushStringC,
            Function::ExternalLinkage, "flushStringC", TheModule);
    Function *flushStringCv2_ = Function::Create(FTflushStringCv2,
                Function::ExternalLinkage, "flushStringReady", TheModule);
    Function *flushStringObj_ = Function::Create(FTflushStringObj,
                    Function::ExternalLinkage, "flushStringObject", TheModule);
    Function *flushBoolean_ = Function::Create(FTflushBoolean,
            Function::ExternalLinkage, "flushBoolean", TheModule);
    Function *flushObjectStart_ = Function::Create(FTflushStartEnd,
                Function::ExternalLinkage, "flushObjectStart", TheModule);
    Function *flushArrayStart_ = Function::Create(FTflushStartEnd,
                Function::ExternalLinkage, "flushArrayStart", TheModule);
    Function *flushObjectEnd_ = Function::Create(FTflushStartEnd,
                Function::ExternalLinkage, "flushObjectEnd", TheModule);
    Function *flushArrayEnd_ = Function::Create(FTflushStartEnd,
                Function::ExternalLinkage, "flushArrayEnd", TheModule);
    Function *flushChar_ = Function::Create(FTflushChar,
                    Function::ExternalLinkage, "flushChar", TheModule);
    Function *flushDelim_ = Function::Create(FTflushDelim,
                        Function::ExternalLinkage, "flushDelim", TheModule);
    Function *flushOutput_ = Function::Create(FTflushOutput,
                        Function::ExternalLinkage, "flushOutput", TheModule);

    /* Memory Management */
    Function *getMemoryChunk_ = Function::Create(FTmemoryChunk,
                Function::ExternalLinkage, "getMemoryChunk", TheModule);
    Function *increaseMemoryChunk_ = Function::Create(FTincrMemoryChunk,
                    Function::ExternalLinkage, "increaseMemoryChunk", TheModule);
    Function *releaseMemoryChunk_ = Function::Create(FTreleaseMemoryChunk,
                        Function::ExternalLinkage, "releaseMemoryChunk", TheModule);

    /* Timing */
    Function *resetTime_ = Function::Create(FTtiming, Function::ExternalLinkage,
            "resetTime", TheModule);
    Function *calculateTime_ = Function::Create(FTtiming,
            Function::ExternalLinkage, "calculateTime", TheModule);

    //Memcpy - not used (yet)
    Type* types[] = { void_ptr_type, void_ptr_type, Type::getInt32Ty(ctx) };
    Function* memcpy_ = Intrinsic::getDeclaration(TheModule, Intrinsic::memcpy, types);
    if (memcpy_ == NULL) {
        throw runtime_error(string("Could not find memcpy intrinsic"));
    }

    /**
     * HASHTABLES FOR JOINS / AGGREGATIONS
     */
    //Last type is needed to capture file size. Tentative
    Type* ht_int_types[] = { int32_type, int32_type, void_ptr_type, int32_type };
    FunctionType *FTintHT = FunctionType::get(void_type, ht_int_types, false);
    Function* insertIntKeyToHT_ = Function::Create(FTintHT, Function::ExternalLinkage, "insertIntKeyToHT", TheModule);

    Type* ht_types[] = { char_ptr_type, int64_type, void_ptr_type, int32_type };
    FunctionType *FT_HT = FunctionType::get(void_type, ht_types, false);
    Function* insertToHT_ = Function::Create(FT_HT, Function::ExternalLinkage, "insertToHT", TheModule);

    Type* ht_int_probe_types[] = { int32_type, int32_type, int32_type };
    PointerType* void_ptr_ptr_type = context->getPointerType(void_ptr_type);
    FunctionType *FTint_probeHT = FunctionType::get(void_ptr_ptr_type, ht_int_probe_types, false);
    Function* probeIntHT_ = Function::Create(FTint_probeHT, Function::ExternalLinkage, "probeIntHT", TheModule);
    probeIntHT_->addFnAttr(llvm::Attribute::AlwaysInline);

    Type* ht_probe_types[] = { char_ptr_type, int64_type };
    FunctionType *FT_probeHT = FunctionType::get(void_ptr_ptr_type, ht_probe_types, false);
    Function* probeHT_ = Function::Create(FT_probeHT,   Function::ExternalLinkage, "probeHT", TheModule);
    probeHT_->addFnAttr(llvm::Attribute::AlwaysInline);

    Type* ht_get_metadata_types[] = { char_ptr_type };
    StructType *metadataType = context->getHashtableMetadataType();
    PointerType *metadataArrayType = PointerType::get(metadataType,0);
    FunctionType *FTget_metadata_HT = FunctionType::get(metadataArrayType,
            ht_get_metadata_types, false);
    Function* getMetadataHT_ = Function::Create(FTget_metadata_HT,
            Function::ExternalLinkage, "getMetadataHT", TheModule);

    /**
     * Radix
     */
    /* What the type of HT buckets is */
    vector<Type*> htBucketMembers;
    //int *bucket;
    htBucketMembers.push_back(int32_ptr_type);
    //int *next;
    htBucketMembers.push_back(int32_ptr_type);
    //uint32_t mask;
    htBucketMembers.push_back(int32_type);
    //int count;
    htBucketMembers.push_back(int32_type);
    StructType *htBucketType = StructType::get(ctx, htBucketMembers);
    PointerType *htBucketPtrType = PointerType::get(htBucketType, 0);

    /* JOIN!!! */
    /* What the type of HT entries is */
    /* (int32, void*) */
    vector<Type*> htEntryMembers;
    htEntryMembers.push_back(int32_type);
    htEntryMembers.push_back(int64_type);
    StructType *htEntryType = StructType::get(ctx,htEntryMembers);
    PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

    Type* radix_partition_types[] = { int64_type, htEntryPtrType };
    FunctionType *FTradix_partition = FunctionType::get(int32_ptr_type, radix_partition_types, false);
    Function *radix_partition = Function::Create(FTradix_partition,
                            Function::ExternalLinkage, "partitionHTLLVM", TheModule);

    Type* bucket_chaining_join_prepare_types[] = { htEntryPtrType, int32_type,
            htBucketPtrType };
    FunctionType *FTbucket_chaining_join_prepare = FunctionType::get(void_type,
            bucket_chaining_join_prepare_types, false);
    Function *bucket_chaining_join_prepare = Function::Create(
            FTbucket_chaining_join_prepare, Function::ExternalLinkage,
            "bucket_chaining_join_prepareLLVM", TheModule);

    /* AGGR! */
    /* What the type of HT entries is */
    /* (int64, void*) */
    vector<Type*> htAggEntryMembers;
    htAggEntryMembers.push_back(int64_type);
    htAggEntryMembers.push_back(int64_type);
    StructType *htAggEntryType = StructType::get(ctx,htAggEntryMembers);
        PointerType *htAggEntryPtrType = PointerType::get(htAggEntryType, 0);
    Type* radix_partition_agg_types[] = { int64_type, htAggEntryPtrType };
    FunctionType *FTradix_partition_agg = FunctionType::get(int32_ptr_type,
            radix_partition_agg_types, false);
    Function *radix_partition_agg = Function::Create(FTradix_partition_agg,
            Function::ExternalLinkage, "partitionAggHTLLVM", TheModule);

    Type* bucket_chaining_agg_prepare_types[] = { htAggEntryPtrType, int32_type,
            htBucketPtrType };
    FunctionType *FTbucket_chaining_agg_prepare = FunctionType::get(void_type,
            bucket_chaining_agg_prepare_types, false);
    Function *bucket_chaining_agg_prepare = Function::Create(
            FTbucket_chaining_agg_prepare, Function::ExternalLinkage,
            "bucket_chaining_agg_prepareLLVM", TheModule);
    /**
     * End of Radix
     */


    /**
     * Parsing
     */
    Type* newline_types[] = { char_ptr_type , int64_type };
    FunctionType *FT_newline = FunctionType::get(int64_type, newline_types, false);
    Function *newline = Function::Create(FT_newline, Function::ExternalLinkage,
            "newlineAVX", TheModule);
    /* Does not make a difference... */
    newline->addFnAttr(llvm::Attribute::AlwaysInline);

//  vector<Type*> tokenMembers;
//  tokenMembers.push_back(int32_type);
//  tokenMembers.push_back(int32_type);
//  tokenMembers.push_back(int32_type);
//  tokenMembers.push_back(int32_type);
//  StructType *tokenType = StructType::get(ctx,tokenMembers);
    StructType *tokenType = context->CreateJSMNStruct();


    PointerType *tokenPtrType = PointerType::get(tokenType, 0);
    PointerType *token2DPtrType = PointerType::get(tokenPtrType, 0);
    Type* parse_line_json_types[] = { char_ptr_type, int64_type, int64_type,
            token2DPtrType, int64_type };
    FunctionType *FT_parse_line_json =
            FunctionType::get(void_type, parse_line_json_types, false);
    Function *parse_line_json = Function::Create(FT_parse_line_json,
            Function::ExternalLinkage, "parseLineJSON", TheModule);


    registerFunction("printi", printi_);
    registerFunction("printi64", printi64_);
    registerFunction("printFloat", printFloat_);
    registerFunction("printShort", printShort_);
    registerFunction("printBoolean", printBoolean_);
    registerFunction("printc", printc_);

    registerFunction("atoi", atoi_);
    registerFunction("atois", atois_);
    registerFunction("atof", atof_);

    registerFunction("insertInt", insertIntKeyToHT_);
    registerFunction("probeInt", probeIntHT_);
    registerFunction("insertHT", insertToHT_);
    registerFunction("probeHT", probeHT_);
    registerFunction("getMetadataHT", getMetadataHT_);

    registerFunction("compareTokenString", compareTokenString_);
    registerFunction("compareTokenString64", compareTokenString64_);
    registerFunction("convertBoolean", convertBoolean_);
    registerFunction("convertBoolean64", convertBoolean64_);
    registerFunction("equalStringObjs", stringObjEquality);
    registerFunction("equalStrings", stringEquality);

    registerFunction("hashInt", hashInt_);
    registerFunction("hashDouble", hashDouble_);
    registerFunction("hashStringC", hashStringC_);
    registerFunction("hashStringObject", hashStringObj_);
    registerFunction("hashBoolean", hashBoolean_);
    registerFunction("combineHashes", hashCombine_);
    registerFunction("combineHashesNoOrder", hashCombineNoOrder_);

    registerFunction("flushInt", flushInt_);
    registerFunction("flushInt64", flushInt64_);
    registerFunction("flushDouble", flushDouble_);
    registerFunction("flushStringC", flushStringC_);
    registerFunction("flushStringCv2", flushStringCv2_);
    registerFunction("flushStringObj", flushStringObj_);
    registerFunction("flushBoolean", flushBoolean_);
    registerFunction("flushChar", flushChar_);
    registerFunction("flushDelim", flushDelim_);
    registerFunction("flushOutput", flushOutput_);

    registerFunction("flushObjectStart", flushObjectStart_);
    registerFunction("flushArrayStart", flushArrayStart_);
    registerFunction("flushObjectEnd", flushObjectEnd_);
    registerFunction("flushArrayEnd", flushArrayEnd_);
    registerFunction("flushArrayEnd", flushArrayEnd_);

    registerFunction("getMemoryChunk", getMemoryChunk_);
    registerFunction("increaseMemoryChunk", increaseMemoryChunk_);
    registerFunction("releaseMemoryChunk", releaseMemoryChunk_);
    registerFunction("memcpy", memcpy_);

    registerFunction("resetTime", resetTime_);
    registerFunction("calculateTime", calculateTime_);

    registerFunction("partitionHT",radix_partition);
    registerFunction("bucketChainingPrepare",bucket_chaining_join_prepare);
    registerFunction("partitionAggHT",radix_partition_agg);
    registerFunction("bucketChainingAggPrepare",bucket_chaining_agg_prepare);

    registerFunction("newline",newline);
    registerFunction("parseLineJSON",parse_line_json);
}
