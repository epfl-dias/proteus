#include "util/raw-pipeline.hpp"
#include "common/gpu/gpu-common.hpp"


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

Argument * RawPipelineGen::getArgument(size_t id) const{
    assert(id < args.size());
    return args[id];
}

Value * RawPipelineGen::getStateVar(size_t id) const{
    Argument * arg = getArgument(args.size() - 1);
    return context->getBuilder()->CreateExtractValue(arg, id);
}

// RawPipeline * RawPipelineGen::generate(){
//     // return new RawPipeline();
// }


Function * RawPipelineGen::prepare(){
    assert(!F);

    LLVMContext &TheContext = context->getLLVMContext();
    StructType * stateType = StructType::create(state_vars, pipName + "_state_t");

    size_t state_id        = appendParameter(stateType, false, false);//true);
    
    FunctionType *ftype = FunctionType::get(Type::getVoidTy(context->getLLVMContext()), inputs, false);
    //use f_num to overcome an llvm bu with keeping dots in function names when generating PTX (which is invalid in PTX)
    F = Function::Create(ftype, Function::ExternalLinkage, pipName, context->getModule());

    for (size_t i = 1 ; i <= inputs.size() ; ++i){ //+1 because 0 is the return value
        if (inputs_readonly[i - 1]) F->setOnlyReadsMemory(i);
        if (inputs_noalias [i - 1]) F->setDoesNotAlias(   i);
    }

    for (auto &t: F->args()) args.push_back(&t);

    Argument * stateArg    = getArgument(state_id);
    stateArg->setName("state");
    
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

Function * RawPipelineGen::getFunction() const{
    assert(F);
    return F;
}


void RawPipelineGen::registerOpen (std::function<void (RawPipeline * pip)> open ){
    openers.push_back(open );
}

void RawPipelineGen::registerClose(std::function<void (RawPipeline * pip)> close){
    closers.push_back(close);
}


RawPipeline * RawPipelineGen::getPipeline(CUmodule cudaModule) const{
    assert(F);
    StructType * state_type = (StructType *) inputs.back();

    CUfunction func;

    gpu_run(cuModuleGetFunction(&func, cudaModule, F->getName().str().c_str()));
    return new RawPipeline(func, context, state_type, openers, closers);
}

RawPipeline::RawPipeline(CUfunction f, RawContext * context, StructType * state_type,
        const std::vector<std::function<void (RawPipeline * pip)>> &openers,
        const std::vector<std::function<void (RawPipeline * pip)>> &closers,
        int32_t group_id): 
            cons(f), state_type(state_type), openers(openers), closers(closers), group_id(group_id){
    state = malloc((context->getModule()->getDataLayout().getTypeStoreSize(state_type) + 7) / 8);
    assert(state);
}

RawPipeline::~RawPipeline(){
    free(state);
}

int32_t RawPipeline::getGroup() const{
    return group_id;
}

void RawPipeline::open(){
    for (const auto &opener: openers) opener(this);
}

void RawPipeline::close(){
    for (const auto &closer: closers) closer(this);
}
