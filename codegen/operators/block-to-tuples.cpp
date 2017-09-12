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

void BlockToTuples::produce()    {
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
    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr,
        Builder->CreateIntCast(context->threadNum(), val_curr_itemCtr->getType(), false));
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

    //Get the ENTRY BLOCK
    // context->setCurrentEntryBlock(Builder->GetInsertBlock());

    BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

    // // Start insertion in CondBB.
    // Builder->SetInsertPoint(CondBB);

    // Make the new basic block for the loop header (BODY), inserting after current block.
    BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);

    // Make the new basic block for the increment, inserting after current block.
    BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

    // Create the "AFTER LOOP" block and insert it.
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
    // context->setEndingBlock(AfterBB);

    Builder->CreateBr      (CondBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    
    mem_itemCtr = context->CreateEntryBlockAlloca(F, "i_ptr", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(
                Builder->CreateIntCast(context->threadId(),
                                        pg->getOIDType()->getLLVMType(llvmContext),
                                        false),
                mem_itemCtr);

    RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()}; //FIXME: OID type for blocks ?
    Value * cnt = Builder->CreateLoad(oldBindings[tupleCnt].mem, "cnt");

    // Function * f = context->getFunction("devprinti64");
    // Builder->CreateCall(f, std::vector<Value *>{cnt});

    Builder->SetInsertPoint(CondBB);
    

    /**
     * Equivalent:
     * while(itemCtr < size)
     */
    Value    *lhs = Builder->CreateLoad(mem_itemCtr, "i");
    
    Value   *cond = Builder->CreateICmpSLT(lhs, cnt);

    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    //Get the 'oid' of each record and pass it along.
    //More general/lazy plugins will only perform this action,
    //instead of eagerly 'converting' fields
    //FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType()};

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    variableBindings[tupleIdentifier] = mem_posWrapper;

    //Actual Work (Loop through attributes etc.)
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr{*(wantedFields[i]), true};

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
        string bufVarStr  = wantedFields[0]->getRelationName();
        string currBufVar = bufVarStr + "." + attr.getAttrName();

        // Value *parsed = Builder->CreateLoad(bufShiftedPtr); //attr_alloca
        Value *parsed = Builder->CreateLoad(Builder->CreateGEP(arg, lhs)); //TODO : use CreateAllignedLoad 

        AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(F, currBufVar, parsed->getType());
        Builder->CreateStore(parsed, mem_currResult);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_currResult;
        mem_valWrapper.isNull = context->createFalse();
        variableBindings[*(wantedFields[i])] = mem_valWrapper;
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
}



