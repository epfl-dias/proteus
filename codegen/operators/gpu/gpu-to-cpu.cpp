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

#include "operators/gpu/gpu-to-cpu.hpp"
#include "numa_utils.cuh"

void GpuToCpu::produce() {
    LLVMContext & llvmContext   = context->getLLVMContext();
    Type        * int32_type    = Type::getInt32Ty  (llvmContext);
    Type        * charPtrType   = Type::getInt8PtrTy(llvmContext);

    Plugin * pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    vector<Type *> child_params;
    for (const auto t: wantedFields){
        child_params.emplace_back(t->getOriginalType()->getLLVMType(llvmContext));
    }
    child_params.emplace_back(pg->getOIDType()->getLLVMType(llvmContext));
    child_params.emplace_back(pg->getOIDType()->getLLVMType(llvmContext));

    params_type = StructType::get(llvmContext, child_params);

    flagsVar_id_catch = context->appendStateVar(PointerType::get(int32_type , 0));
    storeVar_id_catch = context->appendStateVar(PointerType::get(params_type, 0));
    eofVar_id_catch   = context->appendStateVar(PointerType::get(int32_type , 0));

    generate_catch();

    context->popNewPipeline();

    cpu_pip = context->removeLatestPipeline();

    context->pushNewPipeline();

    lockVar_id      = context->appendStateVar(PointerType::get(int32_type , 0));
    lastVar_id      = context->appendStateVar(PointerType::get(int32_type , 0));
    flagsVar_id     = context->appendStateVar(PointerType::get(int32_type , 0));
    storeVar_id     = context->appendStateVar(PointerType::get(params_type, 0));
    threadVar_id    = context->appendStateVar(charPtrType);
    eofVar_id       = context->appendStateVar(PointerType::get(int32_type , 0));

    getChild()->produce();
}

void GpuToCpu::consume(RawContext* const context, const OperatorState& childState) {
    GpuRawContext * ctx = dynamic_cast<GpuRawContext *>(context);
    if (!ctx){
        string error_msg = "[GpuToCpu: ] Operator only supports code generation using the GpuRawContext";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }
    consume(ctx, childState);
}

void GpuToCpu::consume(GpuRawContext * const context, const OperatorState& childState) {
    //Prepare
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    // Builder->SetInsertPoint(context->getCurrentEntryBlock());

    Type  * charPtrType         = Type::getInt8PtrTy(llvmContext);
    Type  * int64Type           = Type::getInt64Ty  (llvmContext);
    Type  * int32_type          = Type::getInt32Ty  (llvmContext);
    Type  * ptr_t               = PointerType::get(charPtrType, 0);

    const map<RecordAttribute, RawValueMemory>& activeVars = childState.getBindings();

    Value * kernel_params      = UndefValue::get(params_type);
    // Value * kernel_params_addr = context->CreateEntryBlockAlloca(F, "cpu_params", params_type);

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        auto it = activeVars.find(*(wantedFields[i]));
        assert(it != activeVars.end());
        Value * mem_val = Builder->CreateLoad(it->second.mem);

        kernel_params = Builder->CreateInsertValue( kernel_params, 
                                                    mem_val,
                                                    i);
    }
    
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = activeVars.find(tupleCnt);
    assert(it != activeVars.end());

    RawValueMemory mem_cntWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateLoad(mem_cntWrapper.mem), wantedFields.size() + 1);


    RecordAttribute tupleOID = RecordAttribute(wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType()); //FIXME: OID type for blocks ?

    it = activeVars.find(tupleOID);
    assert(it != activeVars.end());

    RawValueMemory mem_oidWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateLoad(mem_oidWrapper.mem), wantedFields.size()    );


    // Value * subState   = ((GpuRawContext *) context)->getSubStateVar();

    // kernel_params = Builder->CreateInsertValue(kernel_params, subState, wantedFields.size() + 1);

    // Builder->CreateStore(kernel_params, kernel_params_addr);

    llvm::AtomicOrdering order = llvm::AtomicOrdering::Monotonic;

    Value * zero        = ConstantInt::get((IntegerType *) int32_type, 0);
    Value * one         = ConstantInt::get((IntegerType *) int32_type, 1);

    Value * lock_ptr    = context->getStateVar(lockVar_id );
    Value * last_ptr    = context->getStateVar(lastVar_id );
    Value * flags_ptr   = context->getStateVar(flagsVar_id);
    Value * store_ptr   = context->getStateVar(storeVar_id);


    BasicBlock * acquireLockBB = BasicBlock::Create(llvmContext, "acquireLock", F);
    BasicBlock * readLastBB    = BasicBlock::Create(llvmContext, "readLast"   , F);
    BasicBlock * waitSlotBB    = BasicBlock::Create(llvmContext, "waitSlot"   , F);
    BasicBlock * saveItemBB    = BasicBlock::Create(llvmContext, "saveItem"   , F);
    BasicBlock * afterBB       = BasicBlock::Create(llvmContext, "afterBB"    , F);


    Value * tid  = context->threadId();
    Value * cond = Builder->CreateICmpEQ(tid, ConstantInt::get((IntegerType *) tid->getType(), 0));


    Builder->CreateCondBr(cond, acquireLockBB, afterBB);
    // Builder->CreateBr(acquireLockBB);

    Builder->SetInsertPoint(acquireLockBB);

    // while (atomicCAS((int *) &lock, 0, 1));

    Value * locked   = Builder->CreateAtomicCmpXchg(lock_ptr,
                                                    zero,
                                                    one,
                                                    order,
                                                    order);

    Value * got_lock = Builder->CreateExtractValue(locked, 1);

    Builder->CreateCondBr(got_lock, readLastBB, acquireLockBB);


    Builder->SetInsertPoint(readLastBB);

    Value * end      = Builder->CreateLoad(last_ptr, true);

    Value * cflg_ptr = Builder->CreateInBoundsGEP(flags_ptr, end);
    Value * cstr_ptr = Builder->CreateInBoundsGEP(store_ptr, end);

    Builder->CreateBr(waitSlotBB);


    // while (*(flags+end) != 0);
    Builder->SetInsertPoint(waitSlotBB);

    Value * flag     = Builder->CreateLoad(cflg_ptr, true);

    Value * got_slot = Builder->CreateICmpEQ(flag, zero);

    Builder->CreateCondBr(got_slot, saveItemBB, waitSlotBB);



    Builder->SetInsertPoint(saveItemBB);

    Function * threadfence_system = context->getFunction("llvm.nvvm.membar.sys");

    // store[end] = current_packet;
    Builder->CreateStore(kernel_params, cstr_ptr, true);

    // __threadfence_system();
    Builder->CreateCall(threadfence_system);

    // flags[end] = 1;
    Builder->CreateStore(one          , cflg_ptr, true);

    // __threadfence_system();
    Builder->CreateCall(threadfence_system);

    Value * new_end  = Builder->CreateAdd (end    , ConstantInt::get(end->getType(),    1));
    new_end          = Builder->CreateURem(new_end, ConstantInt::get(end->getType(), size));

    Builder->CreateStore(new_end, last_ptr, true);

    // lock = 0;
    Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, lock_ptr, zero, order);

    Builder->CreateBr(afterBB);

    Builder->SetInsertPoint(afterBB);


    context->registerOpen ([this](RawPipeline * pip){this->open (pip);});
    context->registerClose([this](RawPipeline * pip){this->close(pip);});
}


void GpuToCpu::generate_catch(){
    context->setGlobalFunction();

    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    IntegerType * int32_type    = Type::getInt32Ty  (llvmContext);

    Value * zero                = ConstantInt::get(int32_type, 0);
    Value * one                 = ConstantInt::get(int32_type, 1);

    BasicBlock * infLoopBB = BasicBlock::Create(llvmContext, "infloop", F);
    BasicBlock * consumeBB = BasicBlock::Create(llvmContext, "consume", F);
    BasicBlock * isEOFBB   = BasicBlock::Create(llvmContext, "is_eof" , F);
    BasicBlock * waitBB    = BasicBlock::Create(llvmContext, "wait"   , F);
    BasicBlock * endBB     = BasicBlock::Create(llvmContext, "end"    , F);

    //Get the ENTRY BLOCK
    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    context->setEndingBlock(endBB);

    Value * flags_ptr   = context->getStateVar(flagsVar_id_catch);
    Value * store_ptr   = context->getStateVar(storeVar_id_catch);
    Value * eof_ptr     = context->getStateVar(eofVar_id_catch);

    // int front = 0;
    Value * front_ptr = context->CreateEntryBlockAlloca(F, "front_ptr", int32_type);
    Builder->CreateStore(zero, front_ptr);



    // while (true){
    //     while (flags[front] != 1) {
    //         if (*eof == 1){
    //             // parent->close();
    //             return;
    //         }
    //         this_thread::yield();
    //     }
    
    Builder->SetInsertPoint(infLoopBB);

    Value * front    = Builder->CreateLoad(front_ptr);
    Value * cflg_ptr = Builder->CreateInBoundsGEP(flags_ptr, front);
    Value * cstr_ptr = Builder->CreateInBoundsGEP(store_ptr, front);

    Value * rdy_flg  = Builder->CreateLoad(cflg_ptr, true);

    Value * is_ready = Builder->CreateICmpEQ(rdy_flg, one);

    Builder->CreateCondBr(is_ready, consumeBB, isEOFBB);

    Builder->SetInsertPoint(isEOFBB);

    Value * is_eof   = Builder->CreateICmpEQ(Builder->CreateLoad(eof_ptr, true),
                                                one
                                            );

    Builder->CreateCondBr(is_eof, endBB, waitBB);

    Builder->SetInsertPoint(waitBB);

    Function * yield = context->getFunction("yield");
    Builder->CreateCall(yield);

    Builder->CreateBr(infLoopBB);



    Builder->SetInsertPoint(consumeBB);

    //      packet tmp = store[front];  //volatile
    Value * packet = Builder->CreateLoad(cstr_ptr, true);

    // flags[front] = 0;           //volatile
    Builder->CreateStore(zero, cflg_ptr, true);

    //      front = (front + 1) % size;
    Value * new_front   = Builder->CreateAdd(front, 
                                    ConstantInt::get(front->getType(),    1));
    new_front           = Builder->CreateURem(new_front, 
                                    ConstantInt::get(front->getType(), size));
    Builder->CreateStore(new_front, front_ptr);

    map<RecordAttribute, RawValueMemory> variableBindings;

    Plugin * pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        Value * val    = Builder->CreateExtractValue(packet, i);
        val->setName(wantedFields[i]->getAttrName());
        AllocaInst * valPtr = context->CreateEntryBlockAlloca(F, wantedFields[i]->getAttrName() + "_ptr", val->getType());

        Builder->CreateStore(val, valPtr);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem      = valPtr;
        mem_valWrapper.isNull   = context->createFalse();

        variableBindings[*(wantedFields[i])] = mem_valWrapper;
    }

    {
        Value * val    = Builder->CreateExtractValue(packet, wantedFields.size() + 1);
        val->setName("cnt");
        AllocaInst * valPtr = context->CreateEntryBlockAlloca(F, "activeCnt_ptr", val->getType());
        Builder->CreateStore(val, valPtr);
        
        RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()};

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem      = valPtr;
        mem_valWrapper.isNull   = context->createFalse();

        variableBindings[tupleCnt] = mem_valWrapper;


        Function * printi = context->getFunction("printi64");
        Builder->CreateCall(printi, std::vector<Value *>{val});
    }

    {
        Value * val    = Builder->CreateExtractValue(packet, wantedFields.size());
        val->setName("oid");
        AllocaInst * valPtr = context->CreateEntryBlockAlloca(F, "activeLoop_ptr", val->getType());
        Builder->CreateStore(val, valPtr);
        
        RecordAttribute tupleIOD{wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType()};

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem      = valPtr;
        mem_valWrapper.isNull   = context->createFalse();

        variableBindings[tupleIOD] = mem_valWrapper;
    }

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    // }
    Builder->CreateBr(infLoopBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(infLoopBB);

    // Builder->SetInsertPoint(context->getEndingBlock());
    // Builder->CreateRetVoid();

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(endBB);
}

void GpuToCpu::open(RawPipeline * pip){
    volatile int64_t * store;
    volatile int32_t * flags;
    volatile int32_t * eof  ;
    int32_t          * lock ;
    int32_t          * last ;

    int device;
    gpu_run(cudaGetDevice(&device));

    store = (volatile int64_t *) malloc_host_local_to_gpu(
                                        pip->getSizeOf(params_type) * size,
                                        device
                                    );

    flags = (volatile int32_t *) malloc_host_local_to_gpu(
                                                sizeof(int32_t) * (size + 1), 
                                                device
                                            );

    for (size_t i = 0 ; i <= size ; ++i) flags[i] = 0;
    eof = flags + size;

    gpu_run(cudaMalloc((void **) &lock, sizeof(int32_t) * 2));
    gpu_run(cudaMemset(        lock, 0, sizeof(int32_t) * 2));
    last = lock + 1;

    pip->setStateVar<int32_t *>(lockVar_id , (int32_t *) lock );
    pip->setStateVar<int32_t *>(lastVar_id , (int32_t *) last );
    pip->setStateVar<int32_t *>(flagsVar_id, (int32_t *) flags);
    pip->setStateVar<void    *>(storeVar_id, (void    *) store);
    pip->setStateVar<int32_t *>(eofVar_id  , (int32_t *) eof  );

    RawPipeline * cpip = cpu_pip->getPipeline(pip->getGroup());

    cpip->setStateVar<int32_t *>(flagsVar_id_catch, (int32_t *) flags);
    cpip->setStateVar<void    *>(storeVar_id_catch, (void    *) store);
    cpip->setStateVar<int32_t *>(eofVar_id_catch  , (int32_t *) eof  );

    std::thread * t = new std::thread([cpip, device](){
        set_affinity_local_to_gpu(device);
        cpip->open ();
        cpip->consume(0);
        cpip->close();
    });

    pip->setStateVar<void    *>(threadVar_id, t);
}

__global__ void write_eof(volatile int32_t * eof){
    *eof = 1;
    __threadfence_system();
}

void GpuToCpu::close(RawPipeline * pip){
    volatile int32_t * eof = pip->getStateVar<volatile int32_t *>(eofVar_id);
    // *eof = 1;

    // A kernel needs to be launched to be able to guarantee that the CPU will 
    // see the EOF __AFTER__ every flag write
    // Otherwise, there is not guarantee on the order the CPU will see the 
    // remote write compared to the (local) volatile write
    write_eof<<<1, 1>>>(eof);
    gpu_run(cudaDeviceSynchronize());


    std::thread * t = pip->getStateVar<std::thread *>(threadVar_id);
    
    t->join();

    // int32_t * fg = pip->getStateVar<int32_t *>(flagsVar_id_catch);
    // int64_t * st = pip->getStateVar<int64_t *>(storeVar_id_catch);

    // for (size_t i = 0 ; i < size ; ++i) std::cout << "=====> " << i << " " << fg[i] << " " << st[i * 6 + 5] << std::endl;

}