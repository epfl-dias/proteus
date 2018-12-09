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

#include "operators/hash-rearrange.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-hasher.hpp"

extern "C"{
    void * get_buffer(size_t bytes);
}

void HashRearrange::produce() {
    LLVMContext & llvmContext   = context->getLLVMContext();

    Plugin * pg         = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
    Type   * oid_type   = pg->getOIDType()->getLLVMType(llvmContext);
    Type   * cnt_type   = PointerType::getUnqual(ArrayType::get(oid_type, numOfBuckets));
    cntVar_id           = context->appendStateVar(cnt_type);
    
    oidVar_id           = context->appendStateVar(PointerType::getUnqual(oid_type));

    std::vector<Type *> block_types;
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        block_types.emplace_back(RecordAttribute(wantedFields[i]->getRegisteredAs(), true).getLLVMType(llvmContext));
        wfSizes.emplace_back(context->getSizeOf(wantedFields[i]->getExpressionType()->getLLVMType(llvmContext)));
    }

    Type * block_stuct = StructType::get(llvmContext, block_types);

    blkVar_id           = context->appendStateVar(PointerType::getUnqual(ArrayType::get(block_stuct, numOfBuckets)));

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
    
    getChild()->produce();
}

Value * HashRearrange::hash(const std::vector<expressions::Expression *> &exprs, RawContext* const context, const OperatorState& childState){
    if (exprs.size() == 1){
        ExpressionHasherVisitor hasher{context, childState};
        return exprs[0]->accept(hasher).value;
    } else {
        std::list<expressions::AttributeConstruction> a;
        size_t i = 0;
        for (const auto &e: exprs) a.emplace_back("k" + std::to_string(i++), e);

        ExpressionHasherVisitor hasher{context, childState};
        return expressions::RecordConstruction{a}.accept(hasher).value;
    }
}

void HashRearrange::consume(RawContext* const context, const OperatorState& childState) {
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    map<RecordAttribute, RawValueMemory> bindings{childState.getBindings()};


    Plugin * pg       = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
    IntegerType * oid_type     = (IntegerType *) pg->getOIDType()->getLLVMType(llvmContext);

    IntegerType * int32_type   = Type::getInt32Ty  (llvmContext);
    IntegerType * int64_type   = Type::getInt64Ty  (llvmContext);

    IntegerType * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    size_t max_width = 0;
    for (const auto &e: wantedFields){
        std::cout << e->getExpressionType()->getType() << std::endl;
        max_width = std::max(max_width, context->getSizeOf(e->getExpressionType()->getLLVMType(llvmContext)));
    }

    cap                   = blockSize / max_width;
    Value * capacity      = ConstantInt::get(oid_type, cap);
    Value * last_index    = ConstantInt::get(oid_type, cap - 1);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    AllocaInst * blockN_ptr = context->CreateEntryBlockAlloca(F, "blockN", oid_type);
    Builder->CreateStore(capacity, blockN_ptr);
    AllocaInst * ready_cnt  = context->CreateEntryBlockAlloca(F, "readyN", int32_type);
    Builder->CreateStore(ConstantInt::get(int32_type, 0), ready_cnt);

    Builder->SetInsertPoint(insBB);


    map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();
    //Generate target
    ExpressionGeneratorVisitor exprGenerator{context, childState};
    Value * target            = HashRearrange::hash(std::vector<expressions::Expression *>{hashExpr}, context, childState);
    target = Builder->CreateTruncOrBitCast(target, int32_type);
    IntegerType * target_type = (IntegerType *) target->getType();
    // Value * target            = hashExpr->accept(exprGenerator).value;
    if (hashProject){
        //Save hash in bindings
        AllocaInst * hash_ptr = context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);
        
        Builder->CreateStore(target, hash_ptr);

        RawValueMemory mem_hashWrapper;
        mem_hashWrapper.mem      = hash_ptr;
        mem_hashWrapper.isNull   = context->createFalse();
        (*variableBindings)[*hashProject] = mem_hashWrapper;
    }

    Value * numOfBucketsV = ConstantInt::get(target_type, numOfBuckets);

    target = Builder->CreateURem(target, numOfBucketsV);
    target->setName("target");

    vector<Type *> members;
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
        members.push_back(tblock.getLLVMType(llvmContext));
    }
    members.push_back(target->getType());

    StructType * partition = StructType::get(llvmContext, members);
    Value * ready = context->CreateEntryBlockAlloca(F, "complete_partitions", ArrayType::get(partition, 1024));


    // Value * indexes = Builder->CreateLoad(((GpuRawContext *) context)->getStateVar(cntVar_id), "indexes");

    // indexes->dump();
    // indexes->getType()->dump();
    // ((GpuRawContext *) context)->getStateVar(cntVar_id)->getType()->dump();
    Value * indx_addr = Builder->CreateInBoundsGEP(((GpuRawContext *) context)->getStateVar(cntVar_id), std::vector<Value *>{context->createInt32(0), target});
    Value * indx = Builder->CreateLoad(indx_addr);
    // Value * indx      = Builder->Load(indx_addr);

    Value * blocks = ((GpuRawContext *) context)->getStateVar(blkVar_id);
    Value * curblk = Builder->CreateInBoundsGEP(blocks, std::vector<Value *>{context->createInt32(0), target});

    std::vector<Value *> els;
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        Value * block     = Builder->CreateLoad(Builder->CreateInBoundsGEP(curblk, std::vector<Value *>{context->createInt32(0), context->createInt32(i)}));

        Value * el_ptr    = Builder->CreateInBoundsGEP(block, indx);

        ExpressionGeneratorVisitor exprGenerator(context, childState);
        RawValue valWrapper = wantedFields[i]->accept(exprGenerator);
        Value * el          = valWrapper.value;

        Builder->CreateStore(el, el_ptr);
        els.push_back(block);
    }

    // if indx  >= vectorSize - 1
    BasicBlock *fullBB  = BasicBlock::Create(llvmContext, "propagate", F);
    BasicBlock *elseBB  = BasicBlock::Create(llvmContext, "else", F);
    BasicBlock *mergeBB = BasicBlock::Create(llvmContext, "merge", F);

    Value * cond = Builder->CreateICmpUGE(indx, last_index);

    Builder->CreateCondBr(cond, fullBB, elseBB);

    Builder->SetInsertPoint(fullBB);

    RecordAttribute tupCnt  = RecordAttribute(wantedFields[0]->getRegisteredRelName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    (*variableBindings)[tupCnt] = mem_cntWrapper;


    Value * new_oid = Builder->CreateLoad(((GpuRawContext *) context)->getStateVar(oidVar_id), "oid");
    Builder->CreateStore(Builder->CreateAdd(new_oid, capacity), ((GpuRawContext *) context)->getStateVar(oidVar_id));
    
    AllocaInst * new_oid_ptr = context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
    Builder->CreateStore(new_oid, new_oid_ptr);

    RecordAttribute tupleIdentifier = RecordAttribute(wantedFields[0]->getRegisteredRelName(),  activeLoop, pg->getOIDType());
    
    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem      = new_oid_ptr;
    mem_oidWrapper.isNull   = context->createFalse();
    (*variableBindings)[tupleIdentifier] = mem_oidWrapper;

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};

        AllocaInst * tblock_ptr = context->CreateEntryBlockAlloca(F, wantedFields[i]->getRegisteredAttrName() + "_ptr", tblock.getLLVMType(llvmContext));

        Builder->CreateStore(els[i], tblock_ptr);

        RawValueMemory memWrapper;
        memWrapper.mem      = tblock_ptr;
        memWrapper.isNull   = context->createFalse();
        (*variableBindings)[tblock] = memWrapper;
    }

    OperatorState state{*this, *variableBindings};
    getParent()->consume(context, state);

    Function * get_buffer = context->getFunction("get_buffer");

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
        Value * size     = context->createSizeT(cap * context->getSizeOf(wantedFields[i]->getExpressionType()->getLLVMType(llvmContext)));

        Value * new_buff = Builder->CreateCall(get_buffer, std::vector<Value *>{size});

        new_buff = Builder->CreateBitCast(new_buff, tblock.getLLVMType(llvmContext));

        Builder->CreateStore(new_buff, Builder->CreateInBoundsGEP(curblk, std::vector<Value *>{context->createInt32(0), context->createInt32(i)}));
    }

    Builder->CreateStore(ConstantInt::get(oid_type, 0), indx_addr);

    Builder->CreateBr(mergeBB);

    // else
    Builder->SetInsertPoint(elseBB);

    Builder->CreateStore(Builder->CreateAdd(indx, ConstantInt::get(oid_type, 1)), indx_addr);

    Builder->CreateBr(mergeBB);

    // merge
    Builder->SetInsertPoint(mergeBB);

    // flush remaining elements
    consume_flush();
}

void HashRearrange::consume_flush(){
    save_current_blocks_and_restore_at_exit_scope blks{context};
    LLVMContext &llvmContext    = context->getLLVMContext();

    flushingFunc = (*context)->createHelperFunction("flush", std::vector<Type *>{}, std::vector<bool>{}, std::vector<bool>{});
    closingPip   = (context->operator->());
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();
    //Get the ENTRY BLOCK
    context->setCurrentEntryBlock(Builder->GetInsertBlock());





    // std::vector<Type *> args{context->getStateVars()};

    // context->pushNewCpuPipeline();

    // for (Type * t: args) context->appendStateVar(t);

    // LLVMContext & llvmContext   = context->getLLVMContext();

    Plugin * pg       = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
    Type   * oid_type = pg->getOIDType()->getLLVMType(llvmContext);

    IntegerType * int32_type   = Type::getInt32Ty  (llvmContext);
    IntegerType * int64_type   = Type::getInt64Ty  (llvmContext);

    IntegerType * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    // context->setGlobalFunction();

    // IRBuilder<> * Builder       = context->getBuilder    ();
    // BasicBlock  * insBB         = Builder->GetInsertBlock();
    // Function    * F             = insBB->getParent();
    //Get the ENTRY BLOCK
    // context->setCurrentEntryBlock(Builder->GetInsertBlock());

    BasicBlock *CondBB = BasicBlock::Create(llvmContext, "flushCond", F);

    // // Start insertion in CondBB.
    // Builder->SetInsertPoint(CondBB);

    // Make the new basic block for the loop header (BODY), inserting after current block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "flushBody", F);

    // Make the new basic block for the increment, inserting after current block.
    BasicBlock *IncBB = BasicBlock::Create(llvmContext, "flushInc", F);

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "flushEnd", F);
    context->setEndingBlock(AfterBB);


    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    AllocaInst * blockN_ptr = context->CreateEntryBlockAlloca(F, "blockN", pg->getOIDType()->getLLVMType(llvmContext));


    IntegerType * target_type = size_type;
    if (hashProject) target_type = (IntegerType *) hashProject->getLLVMType(llvmContext);

    AllocaInst *mem_bucket = context->CreateEntryBlockAlloca(F, "target_ptr", target_type);
    Builder->CreateStore(ConstantInt::get(target_type, 0), mem_bucket);


    Builder->SetInsertPoint(CondBB);

    Value * numOfBuckets = ConstantInt::get(target_type, this->numOfBuckets);
    numOfBuckets->setName("numOfBuckets");


    Value * target = Builder->CreateLoad(mem_bucket, "target");


    Value * cond = Builder->CreateICmpSLT(target, numOfBuckets);
    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    map<RecordAttribute, RawValueMemory> variableBindings;

    if (hashProject){
        //Save hash in bindings
        AllocaInst * hash_ptr = context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);
        
        Builder->CreateStore(target, hash_ptr);

        RawValueMemory mem_hashWrapper;
        mem_hashWrapper.mem      = hash_ptr;
        mem_hashWrapper.isNull   = context->createFalse();
        variableBindings[*hashProject] = mem_hashWrapper;
    }

    Value * indx_addr = Builder->CreateInBoundsGEP(((GpuRawContext *) context)->getStateVar(cntVar_id), std::vector<Value *>{context->createInt32(0), target});
    Value * indx = Builder->CreateLoad(indx_addr);

    Builder->CreateStore(indx, blockN_ptr);

    Value * blocks = ((GpuRawContext *) context)->getStateVar(blkVar_id);
    Value * curblk = Builder->CreateInBoundsGEP(blocks, std::vector<Value *>{context->createInt32(0), target});

    std::vector<Value *> block_ptr_addrs;
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        Value * block     = Builder->CreateLoad(Builder->CreateInBoundsGEP(curblk, std::vector<Value *>{context->createInt32(0), context->createInt32(i)}));
        block_ptr_addrs.push_back(block);
    }
    
    RecordAttribute tupCnt{wantedFields[0]->getRegisteredRelName(), "activeCnt", pg->getOIDType()}; //FIXME: OID type for blocks ?

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    variableBindings[tupCnt] = mem_cntWrapper;

    Value * new_oid = Builder->CreateLoad(context->getStateVar(oidVar_id), "oid");
    Builder->CreateStore(Builder->CreateAdd(new_oid, indx), context->getStateVar(oidVar_id));
    
    AllocaInst * new_oid_ptr = context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
    Builder->CreateStore(new_oid, new_oid_ptr);

    RecordAttribute tupleIdentifier = RecordAttribute(wantedFields[0]->getRegisteredRelName(),  activeLoop, pg->getOIDType());
    
    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem      = new_oid_ptr;
    mem_oidWrapper.isNull   = context->createFalse();
    variableBindings[tupleIdentifier] = mem_oidWrapper;


    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};

        AllocaInst * tblock_ptr = context->CreateEntryBlockAlloca(F, wantedFields[i]->getRegisteredAttrName() + "_ptr", tblock.getLLVMType(llvmContext));

        Builder->CreateStore(block_ptr_addrs[i], tblock_ptr);

        RawValueMemory memWrapper;
        memWrapper.mem      = tblock_ptr;
        memWrapper.isNull   = context->createFalse();
        variableBindings[tblock] = memWrapper;
    }

    // Function * f = context->getFunction("printi");
    // Builder->CreateCall(f, indx);

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);


    Builder->SetInsertPoint(IncBB);
    
    Value * next = Builder->CreateAdd(target, ConstantInt::get(target_type, 1));
    Builder->CreateStore(next, mem_bucket);
    
    Builder->CreateBr(CondBB);



    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(context->getEndingBlock());
    Builder->CreateRetVoid();

    // Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // // Insert an explicit fall through from the current (entry) block to the CondBB.
    // Builder->CreateBr(CondBB);


    // Builder->SetInsertPoint(context->getEndingBlock());
    // // Builder->CreateRetVoid();


    // context->popNewPipeline();

    // RawPipelineGen * closingPip = context->removeLatestPipeline();
    // flushFunc                   = closingPip->getKernel();
}

void HashRearrange::open (RawPipeline * pip){
    size_t * cnts = (size_t *) malloc(sizeof(size_t) * (numOfBuckets + 1)); //FIXME: is it always size_t the correct type ?

    for (int i = 0 ; i < numOfBuckets + 1; ++i) cnts[i] = 0;

    pip->setStateVar<size_t *>(cntVar_id, cnts);

    void ** blocks = (void **) malloc(sizeof(void *) * numOfBuckets * wantedFields.size());

    for (int i = 0 ; i < numOfBuckets ; ++i){
        for (size_t j = 0 ; j < wantedFields.size() ; ++j){
            blocks[i * wantedFields.size() + j] = get_buffer(wfSizes[j] * cap);
        }
    }

    pip->setStateVar<void   *>(blkVar_id, (void *) blocks);

    pip->setStateVar<size_t *>(oidVar_id, cnts + numOfBuckets);
}

void HashRearrange::close(RawPipeline * pip){
    // ((void (*)(void *)) this->flushFunc)(pip->getState());
    ((void (*)(void *)) closingPip->getCompiledFunction(flushingFunc))(pip->getState());

    free(pip->getStateVar<size_t *>(cntVar_id));
    free(pip->getStateVar<void   *>(blkVar_id)); //FIXME: release buffers before freeing memory!
    // oidVar is part of cntVar, so they are freed together
}
