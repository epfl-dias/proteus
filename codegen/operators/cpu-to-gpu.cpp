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
    map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();

    OperatorState* state = new OperatorState(*this, *variableBindings);
    getParent()->consume(context, *state);

    context->popNewPipeline();

    gpu_pip                     = context->removeLatestPipeline();
    // entry_point                 = (CUfunction) gpu_pip->getKernel();

    context->pushNewCpuPipeline(gpu_pip);

    LLVMContext & llvmContext   = context->getLLVMContext();
    Type  * charPtrType         = Type::getInt8PtrTy(llvmContext);

    childVar_id = context->appendStateVar(charPtrType);

    getChild()->produce();
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

    Type * kernel_params_type = ArrayType::get(charPtrType, wantedFields.size() + 2); //input + N + state

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
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = activeVars.find(tupleCnt);
    assert(it != activeVars.end());

    RawValueMemory mem_cntWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(kernel_params, Builder->CreateBitCast(mem_cntWrapper.mem, charPtrType), wantedFields.size()    );

    Value * subState   = ((GpuRawContext *) context)->getSubStateVar();

    kernel_params = Builder->CreateInsertValue(kernel_params, subState, wantedFields.size() + 1);

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
