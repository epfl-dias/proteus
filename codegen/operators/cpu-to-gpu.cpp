/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2014
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

#include "operators/cpu-to-gpu.hpp"

void CpuToGpu::produce() {
    generateGpuSide();

    context->popNewPipeline();

    gpu_pip                     = context->removeLatestPipeline();
    // entry_point                 = (CUfunction) gpu_pip->getKernel();

    context->pushNewCpuPipeline(gpu_pip);

    LLVMContext & llvmContext   = context->getLLVMContext();
    Type  * charPtrType         = Type::getInt8PtrTy(llvmContext);

    childVar_id = context->appendStateVar(charPtrType);

    getChild()->produce();
}

void CpuToGpu::generateGpuSide(){
    LLVMContext & llvmContext   = context->getLLVMContext();

    std::vector<size_t> wantedFieldsArg_id;
    for (const auto &tin: wantedFields){
        Type * t = tin->getOriginalType()->getLLVMType(llvmContext);

        wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));
    }

    Type * size_type;
    if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty(llvmContext);
    else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty(llvmContext);
    else                          assert(false);

    size_t tupleOIDArg_id = context->appendParameter(size_type, false, false);
    size_t tupleCntArg_id = context->appendParameter(size_type, false, false);

    context->setGlobalFunction();

    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();


    BasicBlock * AfterBB = BasicBlock::Create(llvmContext, "end" , F);
    BasicBlock * MainBB  = BasicBlock::Create(llvmContext, "main", F);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    context->setEndingBlock(AfterBB);

    map<RecordAttribute, RawValueMemory> variableBindings;

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        Value      * arg = context->getArgument(wantedFieldsArg_id[i]);
        arg->setName(wantedFields[i]->getAttrName());
        AllocaInst * mem = context->CreateEntryBlockAlloca(F, wantedFields[i]->getAttrName() + "_ptr", arg->getType());
        Builder->CreateStore(arg, mem);

        RawValueMemory tmp;
        tmp.mem     = mem;
        tmp.isNull  = context->createFalse();

        variableBindings[*(wantedFields[i])] = tmp;
    }
    
    Plugin     * pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());

    {
        RecordAttribute tupleOID = RecordAttribute(wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType()); //FIXME: OID type for blocks ?

        Value      * oid = context->getArgument(tupleOIDArg_id);
        oid->setName("oid");
        AllocaInst * mem = context->CreateEntryBlockAlloca(F, "activeLoop_ptr", oid->getType());
        Builder->CreateStore(oid, mem);

        RawValueMemory tmp;
        tmp.mem     = mem;
        tmp.isNull  = context->createFalse();

        variableBindings[tupleOID] = tmp;
    }

    {
        RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

        Value      * N   = context->getArgument(tupleCntArg_id);
        N->setName("cnt");
        AllocaInst * mem = context->CreateEntryBlockAlloca(F, "activeCnt_ptr", N->getType());
        Builder->CreateStore(N, mem);

        // Function * printi = context->getFunction("printi64");
        // Builder->CreateCall(printi, std::vector<Value *>{N});

        RawValueMemory tmp;
        tmp.mem     = mem;
        tmp.isNull  = context->createFalse();

        variableBindings[tupleCnt] = tmp;
    }

    Builder->SetInsertPoint(MainBB);
    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to AfterBB.
    Builder->CreateBr(AfterBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(MainBB);

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(context->getEndingBlock());
}

void CpuToGpu::consume(RawContext* const context, const OperatorState& childState) {
    //Prepare
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    Builder->SetInsertPoint(context->getCurrentEntryBlock());

    Type  * charPtrType         = Type::getInt8PtrTy(llvmContext);
    Type  * int64Type           = Type::getInt64Ty  (llvmContext);
    Type  * ptr_t               = PointerType::get(charPtrType, 0);

    const map<RecordAttribute, RawValueMemory>& activeVars = childState.getBindings();

    Type * kernel_params_type = ArrayType::get(charPtrType, wantedFields.size() + 3); //input + N + oid + state

    Value * kernel_params      = UndefValue::get(kernel_params_type);
    Value * kernel_params_addr = context->CreateEntryBlockAlloca(F, "gpu_params", kernel_params_type);

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        auto it = activeVars.find(*(wantedFields[i]));
        assert(it != activeVars.end());
        RawValueMemory mem_valWrapper = it->second;

        kernel_params = Builder->CreateInsertValue( kernel_params, 
                                                    Builder->CreateBitCast(
                                                        mem_valWrapper.mem, 
                                                        charPtrType
                                                    ), 
                                                    i);
    }
    
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    
    RecordAttribute tupleOID = RecordAttribute(wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = activeVars.find(tupleOID);
    assert(it != activeVars.end());

    RawValueMemory mem_oidWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateBitCast(mem_oidWrapper.mem, charPtrType), wantedFields.size()    );

    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    it = activeVars.find(tupleCnt);
    assert(it != activeVars.end());

    RawValueMemory mem_cntWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateBitCast(mem_cntWrapper.mem, charPtrType), wantedFields.size() + 1);

    Value * subState   = ((GpuRawContext *) context)->getSubStateVar();

    kernel_params = Builder->CreateInsertValue(kernel_params, subState, wantedFields.size() + 2);

    Builder->CreateStore(kernel_params, kernel_params_addr);


    Value * entry   = ((GpuRawContext *) context)->getStateVar(childVar_id);

    // Value * entryPtr = ConstantInt::get(llvmContext, APInt(64, ((uint64_t) entry_point)));
    // Value * entry    = Builder->CreateIntToPtr(entryPtr, charPtrType);

    vector<Value *> kernel_args{entry, Builder->CreateBitCast(kernel_params_addr, ptr_t)};
    
    Function * launchk = context->getFunction("launch_kernel");

    Builder->SetInsertPoint(insBB);

    // Launch GPU kernel
    Builder->CreateCall(launchk, kernel_args);

    ((GpuRawContext *) context)->registerOpen([this](RawPipeline * pip){
        pip->setStateVar<void *>(this->childVar_id, gpu_pip->getKernel());
    });

    ((GpuRawContext *) context)->registerClose([this](RawPipeline * pip){
        gpu_run(cudaDeviceSynchronize());
    });
}
