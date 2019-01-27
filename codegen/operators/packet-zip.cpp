#include "packet-zip.hpp"
#include "expressions/expressions-generator.hpp"

#define CACHE_SIZE 1024*1024

ZipInitiate::ZipInitiate (
                        RecordAttribute *                           ptrAttr,
                        RecordAttribute *                           splitter,
                        RecordAttribute *                           targetAttr,
                        RawOperator * const                         child,
                        GpuRawContext * const                       context,
                        int                                         numOfBuckets,
                        ZipState&                                   state1,
                        ZipState&                                   state2,
                        string                                      opLabel) :
                            ptrAttr(ptrAttr),
                            splitter(splitter),
                            targetAttr(targetAttr),
                            UnaryRawOperator(child),
                            context(context),
                            numOfBuckets(numOfBuckets),
                            opLabel(opLabel),
                            state1(state1),
                            state2(state2),
                            calls(0)
{

}

void ZipInitiate::produce () {
    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type* charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

    partition_fwd = context->appendStateVar(PointerType::get(int32_type, 0));
    left_blocks_id = context->appendStateVar(PointerType::get(charPtrType, 0));
    right_blocks_id = context->appendStateVar(PointerType::get(charPtrType, 0));

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open_fwd(pip);});
    generate_send ();
    context->popPipeline();

    launch.push_back(((GpuRawContext *) context)->removeLatestPipeline());

    context->pushPipeline();

    if (++calls != 2) {
        context->setGlobalFunction();
        return;
    }

    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->ctrl(pip);});
    context->setGlobalFunction();
    context->popPipeline();

    auto next_pip  = ((GpuRawContext *) context)->removeLatestPipeline();

    context->pushPipeline();
    context->setChainedPipeline(next_pip);

    partition_alloc_cache = context->appendStateVar(PointerType::get(int32_type, 0));
    partition_cnt_cache = context->appendStateVar(PointerType::get(int32_type, 0));
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open_cache(pip);});
    getChild()->produce();
}

void ZipInitiate::consume(RawContext* const context, const OperatorState& childState) {
    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();
    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();

    Value* mem_cnt = context->getStateVar(partition_cnt_cache);
    Value* mem_alloc = context->getStateVar(partition_alloc_cache);

    Value* offset = Builder->CreateLoad(mem_cnt);
    RawValueMemory mem_valWrapper = (bindings.find(*targetAttr))->second;
    Value* target = Builder->CreateLoad(mem_valWrapper.mem);

    Builder->CreateStore(target, Builder->CreateInBoundsGEP(mem_alloc, offset));
    Builder->CreateStore(Builder->CreateAdd(offset, context->createInt32(1)), mem_cnt);
}

void ZipInitiate::generate_send() {
    context->setGlobalFunction();

    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();
    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

    BasicBlock *SendCondBB   = BasicBlock::Create(llvmContext, "SendCond", TheFunction);
    BasicBlock *SendBodyBB   = BasicBlock::Create(llvmContext, "SendBody", TheFunction);
    BasicBlock *SendMergeBB    = BasicBlock::Create(llvmContext, "SendMerge", TheFunction);

    Value* mem_blocks1 = ((const GpuRawContext *) context)->getStateVar(left_blocks_id);
    Value* mem_blocks2 = ((const GpuRawContext *) context)->getStateVar(right_blocks_id);
    Value* mem_target = ((const GpuRawContext *) context)->getStateVar(partition_fwd);
    
    Value* target = Builder->CreateLoad(mem_target);
    Value* mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", int32_type);
    Builder->CreateStore(context->createInt32(0), mem_current);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    Builder->SetInsertPoint(SendCondBB);    
    context->setEndingBlock(SendMergeBB);

    Value * current = Builder->CreateLoad(mem_current);
    Value * send_cond = Builder->CreateICmpSLT(current, context->createInt32(1));
    Builder->CreateCondBr(send_cond, SendBodyBB, SendMergeBB);

    Builder->SetInsertPoint(SendBodyBB);

    map<RecordAttribute, RawValueMemory>* bindings = new map<RecordAttribute, RawValueMemory>();

    Plugin* pg = RawCatalog::getInstance().getPlugin(targetAttr->getRelationName());

    Value* ptr = Builder->CreateSelect(Builder->CreateICmpNE(current, context->createInt32(0)), Builder->CreateLoad(mem_blocks2), Builder->CreateLoad(mem_blocks1));
    ptr = Builder->CreatePointerCast(ptr, Type::getInt32PtrTy(context->getLLVMContext()));

    RecordAttribute ptr_attr = *ptrAttr;
    AllocaInst * mem_fwd_ptr = context->CreateEntryBlockAlloca(TheFunction, "mem_ptr_N", Type::getInt32PtrTy(context->getLLVMContext()));
    Builder->CreateStore(ptr, mem_fwd_ptr);
    RawValueMemory mem_valWrapper_ptr;
    mem_valWrapper_ptr.mem    = mem_fwd_ptr;
    mem_valWrapper_ptr.isNull = context->createFalse();
    (*bindings)[ptr_attr] = mem_valWrapper_ptr;

    RecordAttribute target_attr = *targetAttr;
    AllocaInst * mem_fwd_target = context->CreateEntryBlockAlloca(TheFunction, "mem_target_N", target->getType());
    Builder->CreateStore(target, mem_fwd_target);
    RawValueMemory mem_valWrapper_target;
    mem_valWrapper_target.mem    = mem_fwd_target;
    mem_valWrapper_target.isNull = context->createFalse();
    (*bindings)[target_attr] = mem_valWrapper_target;

    RecordAttribute splitter_attr = *splitter;
    AllocaInst * mem_fwd_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current_N", current->getType());
    Builder->CreateStore(current, mem_fwd_current);
    RawValueMemory mem_valWrapper_current;
    mem_valWrapper_current.mem    = mem_fwd_current;
    mem_valWrapper_current.isNull = context->createFalse();
    (*bindings)[splitter_attr] = mem_valWrapper_current;

    RecordAttribute tupleOID = RecordAttribute(targetAttr->getRelationName(), activeLoop, pg->getOIDType());
    AllocaInst* mem_arg3 = context->CreateEntryBlockAlloca(TheFunction, "mem_oid_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg3);
    RawValueMemory mem_valWrapper3;
    mem_valWrapper3.mem    = mem_arg3;
    mem_valWrapper3.isNull = context->createFalse();
    (*bindings)[tupleOID] = mem_valWrapper3;

    RecordAttribute tupleCnt = RecordAttribute(ptrAttr->getRelationName(), "activeCnt", pg->getOIDType());
    AllocaInst* mem_arg4 = context->CreateEntryBlockAlloca(TheFunction, "mem_cnt_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg4);
    RawValueMemory mem_valWrapper4;
    mem_valWrapper4.mem    = mem_arg4;
    mem_valWrapper4.isNull = context->createFalse();
    (*bindings)[tupleCnt] = mem_valWrapper4;

    OperatorState* newState = new OperatorState(*this, *bindings);
    getParent()->consume(context, *newState);

    Builder->CreateStore(Builder->CreateAdd(current, context->createInt32(1)), mem_current);
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(context->getEndingBlock());
}

void ZipInitiate::ctrl (RawPipeline * pip) {
    std::cout << "close" << launch.size() << std::endl;

RawPipeline * lpip = launch[0]->getPipeline(pip->getGroup());
    RawPipeline * rpip = launch[1]->getPipeline(pip->getGroup());

    int targs = *(partition_ptr[pip->getGroup()]);

    RawPipeline * jpip = join_pip->getPipeline(pip->getGroup());

    jpip->open();

    for (int i = 0; i < targs; i++) {
        *(partition_ptr[pip->getGroup()]) = partitions[pip->getGroup()][i];

        std::cout << "TIME TO CONSUME " << *(partition_ptr[pip->getGroup()]) << std::endl;

        lpip->open();
        lpip->consume(0);
        lpip->close();

        rpip->open();
        rpip->consume(0);
        rpip->close();

        jpip->consume(0);
    }

    jpip->close();
}

void ZipInitiate::open_fwd (RawPipeline * pip) {
    pip->setStateVar<int*>(partition_fwd, partition_ptr[pip->getGroup()]);
    pip->setStateVar<void**>(left_blocks_id, state1.blocks[0]);
    pip->setStateVar<void**>(right_blocks_id, state2.blocks[0]);
}

void ZipInitiate::open_cache (RawPipeline * pip) {
    partition_ptr[pip->getGroup()] = (int*) malloc(sizeof(int));
    partitions[pip->getGroup()] = (int*) malloc(numOfBuckets*sizeof(int));

    *(partition_ptr[pip->getGroup()]) = 0;

    pip->setStateVar<int*>(partition_cnt_cache, partition_ptr[pip->getGroup()]);
    pip->setStateVar<int*>(partition_alloc_cache, partitions[pip->getGroup()]);
}

ZipCollect::ZipCollect (RecordAttribute*                            ptrAttr,
                        RecordAttribute*                            splitter,
                        RecordAttribute*                            targetAttr,
                        RecordAttribute*                            inputLeft,
                        RecordAttribute*                            inputRight,
						RawOperator * const             			leftChild,
						RawOperator * const             			rightChild,
                    	GpuRawContext * const           			context,
                    	int                             			numOfBuckets,
                    	RecordAttribute*                            hash_key_left,
                        const vector<expression_t>&                 wantedFieldsLeft,
                        RecordAttribute*                            hash_key_right,
                    	const vector<expression_t>&                 wantedFieldsRight,
                    	string                          			opLabel) :
                            ptrAttr(ptrAttr),
                            splitter(splitter),
							BinaryRawOperator (leftChild, rightChild),
                            targetAttr(targetAttr),
                            inputLeft(inputLeft),
                            inputRight(inputRight),
							context(context),
							opLabel(opLabel),
							wantedFieldsRight(wantedFieldsRight),
							wantedFieldsLeft(wantedFieldsLeft),
							numOfBuckets(numOfBuckets),
                            hash_key_left(hash_key_left),
                            hash_key_right(hash_key_right)

{

} 

void ZipCollect::produce () {
    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

    pipeFormat();
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open_pipe(pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close_pipe(pip);});
    generate_send();
    context->popPipeline();

    auto next_pip  = ((GpuRawContext *) context)->removeLatestPipeline();

    context->pushPipeline();
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open_cache_left(pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close_cache_left(pip);});
    cacheFormatLeft();
    getLeftChild()->produce();
    //context->getModule()->dump();
    context->popPipeline(); 

    context->pushPipeline();
    context->setChainedPipeline(next_pip); 
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open_cache_right(pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close_cache_right(pip);});    
    cacheFormatRight();
    getRightChild()->produce();
}

void ZipCollect::consume (RawContext* const context, const OperatorState& childState) {
	const RawOperator& caller = childState.getProducer();

    if(caller == *(getLeftChild())){
        generate_cache_left(context, childState);
    } else {
    	generate_cache_right(context, childState);
    }
}

void ZipCollect::cacheFormatLeft () {
    LLVMContext    &llvmContext = context->getLLVMContext();
	ZipParam p;

	Plugin* pg = RawCatalog::getInstance().getPlugin(inputLeft->getRelationName());

	Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int64_type = Type::getInt64Ty(context->getLLVMContext());
	Type * char_type  = Type::getInt8Ty(context->getLLVMContext());

	Type * t_cnt = PointerType::get(int32_type, 0);
    Type * t2_cnt = PointerType::get(int64_type, 0);
	Type * c_cnt = PointerType::get(PointerType::get(char_type, 0), 0);
	Type * o_cnt = PointerType::get(pg->getOIDType()->getLLVMType(llvmContext), 0);

	p.heads_id = context->appendStateVar(t_cnt);
	p.sizes_id = context->appendStateVar(t2_cnt);
	p.oids_id  = context->appendStateVar(t2_cnt);
	p.blocks_id =  context->appendStateVar(c_cnt);
	p.chains_id = context->appendStateVar(t_cnt);
    p.offset_id = context->appendStateVar(t_cnt);

	cache_left_p = p;
}

void ZipCollect::cacheFormatRight () {
    LLVMContext    &llvmContext = context->getLLVMContext();
	ZipParam p;

	Plugin* pg = RawCatalog::getInstance().getPlugin(inputRight->getRelationName());

	Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int64_type = Type::getInt64Ty(context->getLLVMContext());
    Type * char_type  = Type::getInt8Ty(context->getLLVMContext());

    Type * t_cnt = PointerType::get(int32_type, 0);
    Type * t2_cnt = PointerType::get(int64_type, 0);
    Type * c_cnt = PointerType::get(PointerType::get(char_type, 0), 0);
    Type * o_cnt = PointerType::get(pg->getOIDType()->getLLVMType(llvmContext), 0);

	p.heads_id = context->appendStateVar(t_cnt);
	p.sizes_id = context->appendStateVar(t2_cnt);
	p.oids_id  = context->appendStateVar(t2_cnt);
	p.blocks_id =  context->appendStateVar(c_cnt);
	p.chains_id = context->appendStateVar(t_cnt);
    p.offset_id = context->appendStateVar(t_cnt);

	cache_right_p = p;
}

void ZipCollect::pipeFormat () {
    LLVMContext    &llvmContext = context->getLLVMContext();
    ZipParam p1, p2;

    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFieldsLeft[0].getRegisteredRelName());

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int64_type = Type::getInt64Ty(context->getLLVMContext());
    Type * char_type  = Type::getInt8Ty(context->getLLVMContext());

    Type * t_cnt = PointerType::get(int32_type, 0);
    Type * t2_cnt = PointerType::get(int64_type, 0);
    Type * c_cnt = PointerType::get(PointerType::get(char_type, 0), 0);
    Type * o_cnt = PointerType::get(pg->getOIDType()->getLLVMType(llvmContext), 0);

    p1.heads_id = context->appendStateVar(t_cnt);
    p1.sizes_id = context->appendStateVar(t2_cnt);
    p1.oids_id  = context->appendStateVar(t2_cnt);
    p1.blocks_id =  context->appendStateVar(c_cnt);
    p1.chains_id = context->appendStateVar(t_cnt);

    pipe_left_p = p1;

    p2.heads_id = context->appendStateVar(t_cnt);
    p2.sizes_id = context->appendStateVar(t2_cnt);
    p2.oids_id  = context->appendStateVar(t2_cnt);
    p2.blocks_id =  context->appendStateVar(c_cnt);
    p2.chains_id = context->appendStateVar(t_cnt);

    pipe_right_p = p2;

    partition_id = context->appendStateVar(t_cnt);
}


void ZipCollect::generate_cache_left(RawContext* const context, const OperatorState& childState) {
	IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    const map<RecordAttribute, RawValueMemory>& old_bindings = childState.getBindings();

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    Value* mem_heads = ((const GpuRawContext *) context)->getStateVar(cache_left_p.heads_id);
    Value* mem_sizes = ((const GpuRawContext *) context)->getStateVar(cache_left_p.sizes_id);
    Value* mem_oids = ((const GpuRawContext *) context)->getStateVar(cache_left_p.oids_id);
    Value* mem_blocks = ((const GpuRawContext *) context)->getStateVar(cache_left_p.blocks_id);
    Value* mem_chains = ((const GpuRawContext *) context)->getStateVar(cache_left_p.chains_id);
    Value* mem_offset = ((const GpuRawContext *) context)->getStateVar(cache_left_p.offset_id);

    //AllocaInst * mem_offset = context->CreateEntryBlockAlloca(TheFunction, "mem_offset", int32_type);
    BasicBlock* ins = Builder->GetInsertBlock();
    //Builder->SetInsertPoint(context->getCurrentEntryBlock());
    //Builder->CreateStore(context->createInt32(0), mem_offset);
    Builder->SetInsertPoint(ins);

    Value* step = context->createInt32(wantedFieldsLeft.size());
    Value* offset = Builder->CreateLoad(mem_offset);
    Value* offset_blk = Builder->CreateMul(offset, step);

    Plugin* pg = RawCatalog::getInstance().getPlugin(inputLeft->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(inputLeft->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?
    auto it = old_bindings.find(tupleCnt);
    
    RawValueMemory mem_cntWrapper = it->second;
    Value * N               = Builder->CreateLoad(mem_cntWrapper.mem);
    Builder->CreateStore(N, Builder->CreateInBoundsGEP(mem_sizes, offset));

    RecordAttribute tupleIdentifier = RecordAttribute(inputLeft->getRelationName(),  activeLoop, pg->getOIDType()); 
    it = old_bindings.find(tupleIdentifier);
    
    RawValueMemory mem_oidWrapper = it->second;
    Value * oid             = Builder->CreateLoad(mem_oidWrapper.mem);
    Builder->CreateStore(oid, Builder->CreateInBoundsGEP(mem_oids, offset));

    for (int i = 0; i < wantedFieldsLeft.size(); i++) {
    	ExpressionGeneratorVisitor exprGen{context, childState};
        RawValue currVal = wantedFieldsLeft[i].accept(exprGen);
        Value* valToMaterialize = currVal.value;

        std::cout << wantedFieldsLeft[i].getExpressionType()->getType() << std::endl;
        std::cout << "type === " << wantedFieldsLeft[i].getExpressionType()->getLLVMType(llvmContext)->getTypeID() << std::endl;
        std::cout << "type === " << valToMaterialize->getType()->getTypeID() << std::endl;

        //Value * blk_ptr            = Builder->CreatePointerCast(valToMaterialize, charPtrType);
        //Builder->CreateStore(blk_ptr, Builder->CreateInBoundsGEP(mem_blocks, offset_blk));
        Value * blk_ptr            = Builder->CreatePointerCast(Builder->CreateInBoundsGEP(mem_blocks, offset_blk), PointerType::get(wantedFieldsLeft[i].getExpressionType()->getLLVMType(llvmContext), 0));
        std::cout << "type === " << blk_ptr->getType()->getTypeID() << std::endl;

        Builder->CreateStore(valToMaterialize, blk_ptr);
        offset_blk = Builder->CreateAdd(offset_blk, context->createInt32(1));

    }

    std::cout <<"bindings" << old_bindings.size() << std::endl;

    Value * target = Builder->CreateLoad((old_bindings.find(*hash_key_left)->second).mem);
    target = Builder->CreateURem(target, context->createInt32(numOfBuckets));
    
    Value* prev = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_heads, target));
    Builder->CreateStore(prev, Builder->CreateInBoundsGEP(mem_chains, offset));
    Builder->CreateStore(offset, Builder->CreateInBoundsGEP(mem_heads, target));

    Value* next_offset = Builder->CreateAdd(offset, context->createInt32(1));
    Builder->CreateStore(next_offset, mem_offset);

    /*vector<Value*> ArgsV1; 
    ArgsV1.push_back(prev);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV1);

    vector<Value*> ArgsV2; 
    ArgsV2.push_back(offset);
    debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV2);*/
}

void ZipCollect::generate_cache_right(RawContext* const context, const OperatorState& childState) {
	IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    const map<RecordAttribute, RawValueMemory>& old_bindings = childState.getBindings();

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    Value* mem_heads = ((const GpuRawContext *) context)->getStateVar(cache_right_p.heads_id);
    Value* mem_sizes = ((const GpuRawContext *) context)->getStateVar(cache_right_p.sizes_id);
    Value* mem_oids = ((const GpuRawContext *) context)->getStateVar(cache_right_p.oids_id);
    Value* mem_blocks = ((const GpuRawContext *) context)->getStateVar(cache_right_p.blocks_id);
    Value* mem_chains = ((const GpuRawContext *) context)->getStateVar(cache_right_p.chains_id);
    Value* mem_offset = ((const GpuRawContext *) context)->getStateVar(cache_right_p.offset_id);

    //AllocaInst * mem_offset = context->CreateEntryBlockAlloca(TheFunction, "mem_offset", int32_type);
    BasicBlock* ins = Builder->GetInsertBlock();
    //Builder->SetInsertPoint(context->getCurrentEntryBlock());
    //Builder->CreateStore(context->createInt32(0), mem_offset);
    Builder->SetInsertPoint(ins);

    Value* step = context->createInt32(wantedFieldsRight.size());
    Value* offset = Builder->CreateLoad(mem_offset);
    Value* offset_blk = Builder->CreateMul(offset, step);

    Plugin* pg = RawCatalog::getInstance().getPlugin(inputRight->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(inputRight->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?
    auto it = old_bindings.find(tupleCnt);
    
    RawValueMemory mem_cntWrapper = it->second;
    Value * N               = Builder->CreateLoad(mem_cntWrapper.mem);
    Builder->CreateStore(N, Builder->CreateInBoundsGEP(mem_sizes, offset));

    RecordAttribute tupleIdentifier = RecordAttribute(inputRight->getRelationName(),  activeLoop, pg->getOIDType()); 
    it = old_bindings.find(tupleIdentifier);
    
    RawValueMemory mem_oidWrapper = it->second;
    Value * oid             = Builder->CreateLoad(mem_oidWrapper.mem);
    Builder->CreateStore(oid, Builder->CreateInBoundsGEP(mem_oids, offset));


    for (int i = 0; i < wantedFieldsRight.size(); i++) {
    	ExpressionGeneratorVisitor exprGen{context, childState};
        RawValue currVal = wantedFieldsRight[i].accept(exprGen);
        Value* valToMaterialize = currVal.value;

        std::cout << "type === " << valToMaterialize->getType()->getTypeID() << std::endl;

        //Value * blk_ptr            = Builder->CreatePointerCast(valToMaterialize, charPtrType);
        //Builder->CreateStore(blk_ptr, Builder->CreateInBoundsGEP(mem_blocks, offset_blk));
        Value * blk_ptr = Builder->CreatePointerCast(Builder->CreateInBoundsGEP(mem_blocks, offset_blk), PointerType::get(wantedFieldsRight[i].getExpressionType()->getLLVMType(llvmContext), 0));
        
        std::cout << "type === " << blk_ptr->getType()->getTypeID() << std::endl;

        Builder->CreateStore(valToMaterialize, blk_ptr);
        offset_blk = Builder->CreateAdd(offset_blk, context->createInt32(1));
    }

    Value * target = Builder->CreateLoad((old_bindings.find(*hash_key_right)->second).mem);
    target = Builder->CreateURem(target, context->createInt32(numOfBuckets));

    Value* prev = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_heads, target));
	Builder->CreateStore(prev, Builder->CreateInBoundsGEP(mem_chains, offset));
	Builder->CreateStore(offset, Builder->CreateInBoundsGEP(mem_heads, target));

	Value* next_offset = Builder->CreateAdd(offset, context->createInt32(1));
	Builder->CreateStore(next_offset, mem_offset);
}

void ZipCollect::generate_send() {
    context->setGlobalFunction();

    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();
    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

    BasicBlock *SendCondBB   = BasicBlock::Create(llvmContext, "SendCond", TheFunction);
    BasicBlock *SendBodyBB   = BasicBlock::Create(llvmContext, "SendBody", TheFunction);
    BasicBlock *SendMergeBB    = BasicBlock::Create(llvmContext, "SendMerge", TheFunction);

    Value* mem_heads1 = ((const GpuRawContext *) context)->getStateVar(pipe_left_p.heads_id);
    Value* mem_sizes1 = ((const GpuRawContext *) context)->getStateVar(pipe_left_p.sizes_id);
    Value* mem_oids1 = ((const GpuRawContext *) context)->getStateVar(pipe_left_p.oids_id);
    Value* mem_blocks1 = ((const GpuRawContext *) context)->getStateVar(pipe_left_p.blocks_id);
    Value* mem_chains1 = ((const GpuRawContext *) context)->getStateVar(pipe_left_p.chains_id);

    Value* mem_heads2 = ((const GpuRawContext *) context)->getStateVar(pipe_right_p.heads_id);
    Value* mem_sizes2 = ((const GpuRawContext *) context)->getStateVar(pipe_right_p.sizes_id);
    Value* mem_oids2 = ((const GpuRawContext *) context)->getStateVar(pipe_right_p.oids_id);
    Value* mem_blocks2 = ((const GpuRawContext *) context)->getStateVar(pipe_right_p.blocks_id);
    Value* mem_chains2 = ((const GpuRawContext *) context)->getStateVar(pipe_right_p.chains_id);

    Value* mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", int32_type);
    Builder->CreateStore(context->createInt32(0), mem_current);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    Builder->SetInsertPoint(SendCondBB);    
    context->setEndingBlock(SendMergeBB);

    Value * current = Builder->CreateLoad(mem_current);
    Value * send_cond = Builder->CreateICmpSLT(current, context->createInt32(numOfBuckets));
    Builder->CreateCondBr(send_cond, SendBodyBB, SendMergeBB);

    Builder->SetInsertPoint(SendBodyBB);

    /*vector<Value*> ArgsV; 
    ArgsV.push_back(current);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);*/

    map<RecordAttribute, RawValueMemory>* bindings = new map<RecordAttribute, RawValueMemory>();

    Plugin* pg = RawCatalog::getInstance().getPlugin(targetAttr->getRelationName());

    Value* ptr = Builder->CreateLoad(mem_blocks2);
    ptr = Builder->CreatePointerCast(ptr, Type::getInt32PtrTy(context->getLLVMContext()));

    RecordAttribute ptr_attr = *ptrAttr;
    AllocaInst * mem_fwd_ptr = context->CreateEntryBlockAlloca(TheFunction, "mem_ptr_N", Type::getInt32PtrTy(context->getLLVMContext()));
    Builder->CreateStore(ptr, mem_fwd_ptr);
    RawValueMemory mem_valWrapper_ptr;
    mem_valWrapper_ptr.mem    = mem_fwd_ptr;
    mem_valWrapper_ptr.isNull = context->createFalse();
    (*bindings)[ptr_attr] = mem_valWrapper_ptr;

    RecordAttribute target_attr = *targetAttr;
    AllocaInst * mem_fwd_target = context->CreateEntryBlockAlloca(TheFunction, "mem_target_N", current->getType());
    Builder->CreateStore(current, mem_fwd_target);
    RawValueMemory mem_valWrapper_target;
    mem_valWrapper_target.mem    = mem_fwd_target;
    mem_valWrapper_target.isNull = context->createFalse();
    (*bindings)[target_attr] = mem_valWrapper_target;

    std::cout << splitter << std::endl;
    RecordAttribute splitter_attr = *splitter;
    AllocaInst * mem_fwd_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current_N", current->getType());
    Builder->CreateStore(current, mem_fwd_current);
    RawValueMemory mem_valWrapper_current;
    mem_valWrapper_current.mem    = mem_fwd_current;
    mem_valWrapper_current.isNull = context->createFalse();
    (*bindings)[splitter_attr] = mem_valWrapper_current;

    RecordAttribute tupleOID = RecordAttribute(targetAttr->getRelationName(), activeLoop, pg->getOIDType());
    AllocaInst* mem_arg3 = context->CreateEntryBlockAlloca(TheFunction, "mem_oid_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg3);
    RawValueMemory mem_valWrapper3;
    mem_valWrapper3.mem    = mem_arg3;
    mem_valWrapper3.isNull = context->createFalse();
    (*bindings)[tupleOID] = mem_valWrapper3;

    RecordAttribute tupleCnt = RecordAttribute(ptrAttr->getRelationName(), "activeCnt", pg->getOIDType());
    AllocaInst* mem_arg4 = context->CreateEntryBlockAlloca(TheFunction, "mem_cnt_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg4);
    RawValueMemory mem_valWrapper4;
    mem_valWrapper4.mem    = mem_arg4;
    mem_valWrapper4.isNull = context->createFalse();
    (*bindings)[tupleCnt] = mem_valWrapper4;


    ///////////////////////////////////////////////////


    OperatorState* newState = new OperatorState(*this, *bindings);
    getParent()->consume(context, *newState);

    Builder->CreateStore(Builder->CreateAdd(current, context->createInt32(1)), mem_current);
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(context->getEndingBlock());
}





void ZipCollect::open_cache_left (RawPipeline * pip) {
    std::cout << "OPening cache left" << pip->getGroup() << std::endl;

    offset_left[0] = (int*) malloc(sizeof(int));

    offset_left[0][0] = 0;

    pip->setStateVar<int*>(cache_left_p.offset_id, offset_left[0]);

	state_left.cnts[0] =(int64_t*) malloc(sizeof(int64_t) * CACHE_SIZE);

	pip->setStateVar<int64_t*>(cache_left_p.sizes_id, state_left.cnts[0]);

	state_left.oids[0] =(int64_t*) malloc(sizeof(int64_t) * CACHE_SIZE);

	pip->setStateVar<int64_t*>(cache_left_p.oids_id, state_left.oids[0]);

	state_left.blocks[0] = (void**) malloc(sizeof(void*) * CACHE_SIZE * wantedFieldsLeft.size());

	pip->setStateVar<char**>(cache_left_p.blocks_id, (char**) state_left.blocks[0]);

	state_left.blocks_chain[0] = (int*) malloc(sizeof(int) * CACHE_SIZE);

	pip->setStateVar<int*>(cache_left_p.chains_id, state_left.blocks_chain[0]);

	state_left.blocks_head[0] = (int*) malloc(sizeof(int) * numOfBuckets);

	for (int i = 0; i < numOfBuckets; i++)
		state_left.blocks_head[0][i] = -1;

	pip->setStateVar<int*>(cache_left_p.heads_id, state_left.blocks_head[0]);

    std::cout << "OPened cache left" << std::endl;
}

void ZipCollect::close_cache_left (RawPipeline * pip) {
    std::cout << "close cache left" << std::endl;

    int sum = offset_left[0][0];

    std::cout << "elements partitioned" << sum << std::endl;

    int max = 0;
    for (int i = 0; i < sum; i++) {
        if (state_left.cnts[0][i] > max)
            max = state_left.cnts[0][i];
        for (int j = 0; j < wantedFieldsLeft.size(); j++)
            if (state_left.blocks[0] == NULL)
                printf ("i=%d empty\n", i);
    }

    std::cout << "max partitioned" << max << std::endl;
}

void ZipCollect::open_cache_right (RawPipeline * pip) {
    std::cout << "OPening cache right" <<  pip->getGroup() << std::endl;

    offset_right[0] = (int*) malloc(sizeof(int));

    offset_right[0][0] = 0;

    pip->setStateVar<int*>(cache_right_p.offset_id, offset_right[0]);

	state_right.cnts[0] =(int64_t*) malloc(sizeof(int64_t) * CACHE_SIZE);

	pip->setStateVar<int64_t*>(cache_right_p.sizes_id, state_right.cnts[0]);

	state_right.oids[0] =(int64_t*) malloc(sizeof(int64_t) * CACHE_SIZE);

	pip->setStateVar<int64_t*>(cache_right_p.oids_id, state_right.oids[0]);

	state_right.blocks[0] = (void**) malloc(sizeof(void*) * CACHE_SIZE * wantedFieldsRight.size());

	pip->setStateVar<char**>(cache_right_p.blocks_id, (char**) state_right.blocks[0]);

	state_right.blocks_chain[0] = (int*) malloc(sizeof(int) * CACHE_SIZE);

	pip->setStateVar<int32_t*>(cache_right_p.chains_id, state_right.blocks_chain[0]);

	state_right.blocks_head[0] = (int*) malloc(sizeof(int) * numOfBuckets);

	for (int i = 0; i < numOfBuckets; i++)
		state_right.blocks_head[0][i] = -1;

	pip->setStateVar<int32_t*>(cache_right_p.heads_id, state_right.blocks_head[0]);

    std::cout << "OPened cache right" << std::endl;
}

void ZipCollect::close_cache_right (RawPipeline * pip) {
    std::cout << "close cache right" << std::endl;

    int sum = offset_right[0][0];

    std::cout << "elements partitioned" << sum << std::endl;

    int max = 0;
    for (int i = 0; i < sum; i++) {
        if (state_right.cnts[0][i] > max)
            max = state_right.cnts[0][i];
        for (int j = 0; j < wantedFieldsRight.size(); j++)
            if (state_right.blocks[0] == NULL)
                printf ("i=%d empty\n", i);
    }

    std::cout << "max partitioned" << max << std::endl;
}

void ZipCollect::open_pipe (RawPipeline * pip) {
    std::cout << "OPening pipe" << std::endl;

    pip->setStateVar<int*>(partition_id, partition_ptr[0]);

    

	pip->setStateVar<int64_t*>(pipe_left_p.sizes_id, state_left.cnts[0]);

	pip->setStateVar<int64_t*>(pipe_left_p.oids_id, state_left.oids[0]);

	pip->setStateVar<char**>(pipe_left_p.blocks_id, (char**) state_left.blocks[0]);

	pip->setStateVar<int*>(pipe_left_p.chains_id, state_left.blocks_chain[0]);

	pip->setStateVar<int*>(pipe_left_p.heads_id, state_left.blocks_head[0]);

    /////////////////////    PREPARE CONNECTION TO FORWARDING OPERATOR

    pip->setStateVar<int64_t*>(pipe_right_p.sizes_id, state_right.cnts[0]);

    pip->setStateVar<int64_t*>(pipe_right_p.oids_id, state_right.oids[0]);

    pip->setStateVar<char**>(pipe_right_p.blocks_id, (char**) state_right.blocks[0]);

    pip->setStateVar<int*>(pipe_right_p.chains_id, state_right.blocks_chain[0]);

    pip->setStateVar<int*>(pipe_right_p.heads_id, state_right.blocks_head[0]);
}

void ZipCollect::close_pipe (RawPipeline * pip) {
    std::cout << "close pipe left" << std::endl;
}



ZipForward::ZipForward (
                RecordAttribute*                            splitter,
                RecordAttribute*                             targetAttr,
                RecordAttribute*                             inputAttr,
                RawOperator * const                           child,
                GpuRawContext * const                        context,
                int                                          numOfBuckets,
                const vector<expression_t>&      wantedFields,
                string                                       opLabel,
                ZipState&                                    state) :
                                splitter(splitter),
                                targetAttr(targetAttr),
                                inputAttr(inputAttr),
                                UnaryRawOperator(child),
                                context(context),
                                numOfBuckets(numOfBuckets),
                                wantedFields(wantedFields),
                                opLabel(opLabel),
                                state(state) {}



void ZipForward::produce () {
    cacheFormat();
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open(pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});

    getChild()->setParent(this);
    getChild()->produce();
}


void ZipForward::cacheFormat () {
    LLVMContext    &llvmContext = context->getLLVMContext();

    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int64_type = Type::getInt64Ty(context->getLLVMContext());
    Type * char_type  = Type::getInt8Ty(context->getLLVMContext());

    Type * t_cnt = PointerType::get(int32_type, 0);
    Type * t2_cnt = PointerType::get(int64_type, 0);
    Type * c_cnt = PointerType::get(PointerType::get(char_type, 0), 0);
    Type * o_cnt = PointerType::get(pg->getOIDType()->getLLVMType(llvmContext), 0);

    p.heads_id = context->appendStateVar(t_cnt);
    p.sizes_id = context->appendStateVar(t2_cnt);
    p.oids_id  = context->appendStateVar(t2_cnt);
    p.blocks_id =  context->appendStateVar(c_cnt);
    p.chains_id = context->appendStateVar(t_cnt);
}




void ZipForward::consume (RawContext* const context, const OperatorState& childState) {
    //context->setGlobalFunction();

    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());


    BasicBlock *SendCondBB   = BasicBlock::Create(llvmContext, "SendCond", TheFunction);
    BasicBlock *SendBodyBB   = BasicBlock::Create(llvmContext, "SendBody", TheFunction);
    BasicBlock *SendMergeBB    = BasicBlock::Create(llvmContext, "SendMerge", TheFunction);

    Value* mem_heads = ((const GpuRawContext *) context)->getStateVar(p.heads_id);
    Value* mem_sizes = ((const GpuRawContext *) context)->getStateVar(p.sizes_id);
    Value* mem_oids = ((const GpuRawContext *) context)->getStateVar(p.oids_id);
    Value* mem_blocks = ((const GpuRawContext *) context)->getStateVar(p.blocks_id);
    Value* mem_chains = ((const GpuRawContext *) context)->getStateVar(p.chains_id);

    const map<RecordAttribute, RawValueMemory>& child_bindings = childState.getBindings();

    RecordAttribute target_attr = *targetAttr;
    Value* mem_partition = ((child_bindings.find(target_attr))->second).mem;

    /*RecordAttribute splitter_attr = *splitter;
    Value* mem_hash = ((child_bindings.find(splitter_attr))->second).mem;;*/

    Value* target = Builder->CreateLoad(mem_partition);

    Value* step = context->createInt32(wantedFields.size());
    Value* mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", int32_type);
    Value* start = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_heads, target));
    Builder->CreateStore(start, mem_current);
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(SendCondBB);
    Value* current = Builder->CreateLoad(mem_current);
    Value* cond = Builder->CreateICmpNE(current, context->createInt32(-1));
    Builder->CreateCondBr(cond, SendBodyBB, SendMergeBB);

    Builder->SetInsertPoint(SendBodyBB);

    map<RecordAttribute, RawValueMemory>* bindings = new map<RecordAttribute, RawValueMemory>();

    Value * N = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_sizes, current));
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0].getRegisteredRelName(), "activeCnt", pg->getOIDType());
    AllocaInst * mem_arg_N = context->CreateEntryBlockAlloca(TheFunction, "mem_" + wantedFields[0].getRegisteredRelName() + "_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(Builder->CreateTrunc(N, int32_type), mem_arg_N);
    RawValueMemory mem_valWrapper1;
    mem_valWrapper1.mem    = mem_arg_N;
    mem_valWrapper1.isNull = context->createFalse();
    (*bindings)[tupleCnt] = mem_valWrapper1;

    /*vector<Value*> ArgsV1; 
    ArgsV1.push_back(N);
    Function* debugInt = context->getFunction("printi64");
    Builder->CreateCall(debugInt, ArgsV1);*/

    /*vector<Value*> ArgsV2; 
    ArgsV2.push_back(current);
    Function* debugInt2 = context->getFunction("printi");
    Builder->CreateCall(debugInt2, ArgsV2);*/

    Value * oid = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_oids, current));    
    RecordAttribute tupleIdentifier = RecordAttribute(wantedFields[0].getRegisteredRelName(),  activeLoop, pg->getOIDType());
    AllocaInst * mem_arg_oid = context->CreateEntryBlockAlloca(TheFunction, "mem_" + wantedFields[0].getRegisteredRelName() + "_oid", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(Builder->CreateTrunc(oid, int32_type), mem_arg_oid);
    RawValueMemory mem_valWrapper2;
    mem_valWrapper2.mem    = mem_arg_oid;
    mem_valWrapper2.isNull = context->createFalse();
    (*bindings)[tupleIdentifier] = mem_valWrapper2;

    Value* offset_blk = Builder->CreateMul(current, step);

    for (int i = 0; i < wantedFields.size(); i++) {
        RecordAttribute block_attr  ((wantedFields[i].getRegisteredAs()), true);
        std::cout << wantedFields[i].getExpressionType()->getType() << std::endl;
        Value * blk_ptr = Builder->CreatePointerCast(Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_blocks, offset_blk)),  PointerType::get(wantedFields[i].getExpressionType()->getLLVMType(llvmContext),0));
        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction, "mem_" + wantedFields[i].getRegisteredAttrName(), PointerType::get(wantedFields[i].getExpressionType()->getLLVMType(llvmContext),0));
        Builder->CreateStore(blk_ptr, mem_arg);
        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();
        (*bindings)[block_attr] = mem_valWrapper;

        offset_blk = Builder->CreateAdd(offset_blk, context->createInt32(1));
    }

    OperatorState* newState = new OperatorState(*this, *bindings);
    getParent()->consume(context, *newState);

    Value* next = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_chains, current));
    Builder->CreateStore(next, mem_current);
    Builder->CreateBr(SendCondBB);

    Builder->SetInsertPoint(SendMergeBB);

    /*vector<Value*> ArgsV1; 
    ArgsV1.push_back(target);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV1);*/
}




void ZipForward::open (RawPipeline * pip) {
    std::cout << "OPening pipe " << pip->getGroup() << std::endl;

    pip->setStateVar<int64_t*>(p.sizes_id, state.cnts[0]);

    pip->setStateVar<int64_t*>(p.oids_id, state.oids[0]);

    pip->setStateVar<char**>(p.blocks_id, (char**) state.blocks[0]);

    pip->setStateVar<int*>(p.chains_id, state.blocks_chain[0]);

    pip->setStateVar<int*>(p.heads_id, state.blocks_head[0]);
}

void ZipForward::close (RawPipeline * pip) {
    std::cout << "close pipe " << pip->getGroup() << std::endl;
}


