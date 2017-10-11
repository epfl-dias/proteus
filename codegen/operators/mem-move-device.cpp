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

#include "operators/mem-move-device.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
// #include "buffer_manager.cuh"

// void * make_mem_move_device(char * src, size_t bytes, int target_device, cudaStream_t strm){
//     int dev = get_device(src);

//     if (dev == target_device) return (void *) src; // block already in correct device

//     set_device_on_scope d(dev);

//     if (dev >= 0) set_affinity_local_to_gpu(dev);

//     assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to blocks of arbitary size
//     char * buff = (char *) buffer_manager<int32_t>::h_get_buffer(target_device);

//     buffer_manager<int32_t>::overwrite(buff, src, bytes, strm, true);

//     return buff;
// }

void MemMoveDevice::produce() {
    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());

    device_id_var       = context->appendStateVar(int32_type );
    cu_stream_var       = context->appendStateVar(charPtrType);

    getChild()->produce();
}

void MemMoveDevice::consume(RawContext* const context, const OperatorState& childState) {
    //Prepare
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    Type * charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

    map<RecordAttribute, RawValueMemory> new_bindings{childState.getBindings()};
    
    // Find block size
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = new_bindings.find(tupleCnt);
    assert(it != new_bindings.end());

    RawValueMemory mem_cntWrapper = it->second;

    Function * make_mem_move = context->getFunction("make_mem_move_device");
    
    Builder->SetInsertPoint(context->getCurrentEntryBlock());

    Value * device_id       = ((GpuRawContext *) context)->getStateVar(device_id_var);
    Value * cu_stream       = ((GpuRawContext *) context)->getStateVar(cu_stream_var);

    Builder->SetInsertPoint(insBB);
    Value * N               = Builder->CreateLoad(mem_cntWrapper.mem);
    
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute attr        (*(wantedFields[i]));
        RecordAttribute block_attr  (attr, true);

        auto it = new_bindings.find(block_attr);
        assert(it != new_bindings.end());
        RawValueMemory mem_valWrapper = it->second;
                
        Value * mv              = Builder->CreateBitCast(
                                                            Builder->CreateLoad(mem_valWrapper.mem), 
                                                            charPtrType
                                                        );

        Type  * mv_block_type   = mem_valWrapper.mem->getType()->getPointerElementType()->getPointerElementType();

        Value * size            = ConstantInt::get(llvmContext, APInt(64, context->getSizeOf(mv_block_type)));
        size                    = Builder->CreateMul(size, N);

        vector<Value *> mv_args{mv, size, device_id, cu_stream};

        // Do actual mem move
        Value * new_ptr         = Builder->CreateCall(make_mem_move, mv_args);
        //FIMXE: someone should release the buffer... But who ?
        //This operator should not release it, siblings may use it
        //Maybe the buffer manager should keep counts ?
        
        AllocaInst * new_ptr_addr    = context->CreateEntryBlockAlloca(F, attr.getAttrName(), mem_valWrapper.mem->getType()->getPointerElementType());

        mem_valWrapper.mem      = new_ptr_addr;
        it->second              = mem_valWrapper;
        
        Builder->CreateStore(Builder->CreateBitCast(new_ptr, mem_valWrapper.mem->getType()->getPointerElementType()), new_ptr_addr);
    }

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});

    OperatorState newState{*this, new_bindings};
    //Triggering parent
    getParent()->consume(context, newState);
}

void MemMoveDevice::open (RawPipeline * pip){
    int device = get_device();
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    pip->setStateVar<int         >(device_id_var, device);

    pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
}

void MemMoveDevice::close(RawPipeline * pip){
    int device = get_device();
    cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);

    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy    (strm));
}