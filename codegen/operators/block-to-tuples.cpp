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

#include "operators/block-to-tuples.hpp"
#include "multigpu/buffer_manager.cuh"
#include "util/raw-memory-manager.hpp"

void BlockToTuples::produce()    {

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        old_buffs.push_back(
                                context->appendStateVar(
                                    PointerType::getUnqual(
                                        RecordAttribute{wantedFields[i]->getRegisteredAs(), true}.
                                            getLLVMType(context->getLLVMContext())
                                    )
                                )
                            );
    }


    getChild()->produce();
}

void BlockToTuples::nextEntry()   {

    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);
    IRBuilder<>* Builder = context->getBuilder();

    //Increment and store back

    Value* val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);
    
    Value * inc;
    if (gpu && granularity == gran_t::GRID){
        inc = Builder->CreateIntCast(context->threadNum(), val_curr_itemCtr->getType(), false);
    } else {
        inc = ConstantInt::get((IntegerType *) val_curr_itemCtr->getType(), 1);
    }

    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, inc);
    Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

void BlockToTuples::consume(RawContext* const context, const OperatorState& childState) {
    GpuRawContext * ctx = dynamic_cast<GpuRawContext *>(context);
    if (!ctx){
        string error_msg = "[BlockToTuples: ] Operator only supports code generation using the GpuRawContext";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }
    consume(ctx, childState);
}

void BlockToTuples::consume(GpuRawContext* const context, const OperatorState& childState) {
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();
    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type = Type::getInt64Ty(llvmContext);

    //Container for the variable bindings
    map<RecordAttribute, RawValueMemory> oldBindings{childState.getBindings()};
    map<RecordAttribute, RawValueMemory> variableBindings;

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *releaseBB = BasicBlock::Create(llvmContext, "releaseIf", F);
    BasicBlock *rlAfterBB = BasicBlock::Create(llvmContext, "releaseEnd" , F);

    Value * tId       = context->threadId();
    Value * is_leader = Builder->CreateICmpEQ(tId, ConstantInt::get(tId->getType(), 0));
    Builder->CreateCondBr(is_leader, releaseBB, rlAfterBB); //FIXME: assumes thread 0 gets to execute block2tuples

    Builder->SetInsertPoint(releaseBB);

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr{wantedFields[i]->getRegisteredAs(), true};
        Value        * arg = Builder->CreateLoad(oldBindings[attr].mem);
        Value        * old = Builder->CreateLoad(context->getStateVar(old_buffs[i]));
        old                = Builder->CreateBitCast(old, charPtrType);

        Function * f = context->getFunction("release_buffers"); //FIXME: Assumes grid launch + Assumes 1 block per kernel!
        Builder->CreateCall(f, std::vector<Value *>{old});

        Builder->CreateStore(arg, context->getStateVar(old_buffs[i]));
    }

    Builder->CreateBr(rlAfterBB);

    Builder->SetInsertPoint(rlAfterBB);

    //Get the ENTRY BLOCK
    // context->setCurrentEntryBlock(Builder->GetInsertBlock());

    BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanBlkCond", F);

    // // Start insertion in CondBB.
    // Builder->SetInsertPoint(CondBB);

    // Make the new basic block for the loop header (BODY), inserting after current block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBlkBody", F);

    // Make the new basic block for the increment, inserting after current block.
    BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanBlkInc", F);

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanBlkEnd", F);
    // context->setEndingBlock(AfterBB);

    // Builder->CreateBr      (CondBB);

    // Builder->SetInsertPoint(context->getCurrentEntryBlock());
    
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
    

    RecordAttribute tupleCnt{wantedFields[0]->getRegisteredRelName(), "activeCnt", pg->getOIDType()}; //FIXME: OID type for blocks ?
    Value * cnt = Builder->CreateLoad(oldBindings[tupleCnt].mem, "cnt");

    mem_itemCtr = context->CreateEntryBlockAlloca(F, "i_ptr", cnt->getType());
    Builder->CreateStore(
                Builder->CreateIntCast(context->threadId(),
                                        cnt->getType(),
                                        false),
                mem_itemCtr);

    // Function * f = context->getFunction("devprinti64");
    // Builder->CreateCall(f, std::vector<Value *>{cnt});

    Builder->CreateBr      (CondBB);
    Builder->SetInsertPoint(CondBB);
    

    /**
     * Equivalent:
     * while(itemCtr < size)
     */
    Value    *lhs = Builder->CreateLoad(mem_itemCtr, "i");
    
    Value   *cond = Builder->CreateICmpSLT(lhs, cnt);

    // Insert the conditional branch into the end of CondBB.
    BranchInst * loop_cond = Builder->CreateCondBr(cond, LoopBB, AfterBB);


    // NamedMDNode * annot = context->getModule()->getOrInsertNamedMetadata("nvvm.annotations");
    // MDString    * str   = MDString::get(TheContext, "kernel");
    // Value       * one   = ConstantInt::get(int32Type, 1);

    MDNode * LoopID;

    {
        // MDString       * vec_st   = MDString::get(llvmContext, "llvm.loop.vectorize.enable");
        // Type           * int1Type = Type::getInt1Ty(llvmContext);
        // Metadata       * one      = ConstantAsMetadata::get(ConstantInt::get(int1Type, 1));
        // llvm::Metadata * vec_en[] = {vec_st, one};
        // MDNode * vectorize_enable = MDNode::get(llvmContext, vec_en);

        // MDString       * itr_st   = MDString::get(llvmContext, "llvm.loop.interleave.count");
        // Type           * int32Type= Type::getInt32Ty(llvmContext);
        // Metadata       * count    = ConstantAsMetadata::get(ConstantInt::get(int32Type, 4));
        // llvm::Metadata * itr_en[] = {itr_st, count};
        // MDNode * interleave_count = MDNode::get(llvmContext, itr_en);

        llvm::Metadata * Args[] = {NULL};//, vectorize_enable, interleave_count};
        LoopID = MDNode::get(llvmContext, Args);
        LoopID->replaceOperandWith(0, LoopID);

        loop_cond->setMetadata("llvm.loop", LoopID);
    }
    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    //Get the 'oid' of each record and pass it along.
    //More general/lazy plugins will only perform this action,
    //instead of eagerly 'converting' fields
    //FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier{wantedFields[0]->getRegisteredRelName(), activeLoop, pg->getOIDType()};

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    variableBindings[tupleIdentifier] = mem_posWrapper;

    //Actual Work (Loop through attributes etc.)
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr{wantedFields[i]->getRegisteredAs(), true};

        // Argument * arg = context->getArgument(wantedFieldsArg_id[i]);
        // arg->setName(attr.getAttrName() + "_ptr");

        // size_t offset = 0;

        /* Read wanted field.
         * Reminder: Primitive, scalar types have the (raw) buffer
         * already cast to appr. type*/
        // if (!attr.getOriginalType()->isPrimitive()){
        //     LOG(ERROR)<< "[BINARY COL PLUGIN: ] Only primitives are currently supported";
        //     throw runtime_error(string("[BINARY COL PLUGIN: ] Only primitives are currently supported"));
        // }

        // const PrimitiveType * t = dynamic_cast<const PrimitiveType *>(attr.getOriginalType());

        // AllocaInst * attr_alloca = context->CreateEntryBlockAlloca(F, attr.getAttrName(), t->getLLVMType(llvmContext));

        // variableBindings[tupleIdentifier] = mem_posWrapper;
        //Move to next position
        // Value* val_offset = context->createInt64(offset);
        // skipLLVM(attr, val_offset);

        Value        * arg       = Builder->CreateLoad(oldBindings[attr].mem);

        // string posVarStr = string(posVar);
        // string currPosVar = posVarStr + "." + attr.getAttrName();
        string bufVarStr  = wantedFields[0]->getRegisteredRelName();
        string currBufVar = bufVarStr + "." + attr.getAttrName();

        // Value *parsed = Builder->CreateLoad(bufShiftedPtr); //attr_alloca
        Value       * ptr   = Builder->CreateGEP(arg, lhs);

        // Function    * pfetch = Intrinsic::getDeclaration(Builder->GetInsertBlock()->getParent()->getParent(), Intrinsic::prefetch);

        // Instruction * ins = Builder->CreateCall(pfetch, std::vector<Value*>{
        //                     Builder->CreateBitCast(ptr, charPtrType),
        //                     context->createInt32(0),
        //                     context->createInt32(3),
        //                     context->createInt32(1)}
        //                     );
        // {
        //     ins->setMetadata("llvm.mem.parallel_loop_access", LoopID);
        // }

        Instruction *parsed = Builder->CreateLoad(ptr); //TODO : use CreateAllignedLoad 
        {
            parsed->setMetadata("llvm.mem.parallel_loop_access", LoopID);
        }

        AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(F, currBufVar, parsed->getType());
        Builder->CreateStore(parsed, mem_currResult);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_currResult;
        mem_valWrapper.isNull = context->createFalse();
        variableBindings[wantedFields[i]->getRegisteredAs()] = mem_valWrapper;
    }

    // Start insertion in IncBB.
    Builder->SetInsertPoint(IncBB);
    nextEntry();
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(LoopBB);

    //Triggering parent
    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);

    // Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // // Insert an explicit fall through from the current (entry) block to the CondBB.
    // Builder->CreateBr(CondBB);

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(AfterBB);

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
}



void BlockToTuples::open (RawPipeline * pip){
    int device = get_device();

    execution_conf ec = pip->getExecConfiguration();

    size_t grid_size  = ec.gridSize();

    void ** buffs;

    if (gpu) {
        buffs = (void **) RawMemoryManager::mallocGpu(sizeof(void  *) * wantedFields.size());
        cudaStream_t strm;
        gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
        gpu_run(cudaMemsetAsync(buffs, 0, sizeof(void  *) * wantedFields.size(), strm));
        gpu_run(cudaStreamSynchronize(strm));
        gpu_run(cudaStreamDestroy(strm));
    } else {
        buffs = (void **) RawMemoryManager::mallocPinned(sizeof(void  *) * wantedFields.size());
        memset(buffs, 0, sizeof(void  *) * wantedFields.size());
    }


    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        pip->setStateVar<void *>(old_buffs[i], buffs + i);
    }
}

void BlockToTuples::close(RawPipeline * pip){
    void ** h_buffs;
    void ** buffs   = pip->getStateVar<void **>(old_buffs[0]);

    if (gpu){
        h_buffs = (void **) malloc(sizeof(void  *) * wantedFields.size());
        cudaStream_t strm;
        gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
        gpu_run(cudaMemcpyAsync(h_buffs, buffs, sizeof(void  *) * wantedFields.size(), cudaMemcpyDefault, strm));
        gpu_run(cudaStreamSynchronize(strm));
        gpu_run(cudaStreamDestroy(strm));
    } else {
        h_buffs = buffs;
    }

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        buffer_manager<int32_t>::release_buffer((int32_t *) h_buffs[i]);
    }

    if (gpu) RawMemoryManager::freeGpu(buffs);
    else     RawMemoryManager::freePinned(buffs);

    if (gpu) free(h_buffs);
}
