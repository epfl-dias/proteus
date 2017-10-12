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

#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "util/gpu/gpu-intrinsics.hpp"
#include "expressions/expressions-generator.hpp"
#include <algorithm>
#include "multigpu/buffer_manager.cuh"

void GpuHashRearrange::produce() {
    LLVMContext & llvmContext   = context->getLLVMContext();

    Type   * idx_type           = Type::getInt32Ty(llvmContext);

    for (const auto &e: matExpr){
        PointerType * t_ptr = PointerType::get(e->getExpressionType()->getLLVMType(llvmContext), /* address space */ 0);
        buffVar_id.push_back(context->appendStateVar(PointerType::getUnqual(ArrayType::get(t_ptr, numOfBuckets))));
    }

    Plugin * pg       = RawCatalog::getInstance().getPlugin(matExpr[0]->getRegisteredRelName());
    Type   * oid_type = pg->getOIDType()->getLLVMType(llvmContext);

    cntVar_id  = context->appendStateVar(PointerType::getUnqual(ArrayType::get(idx_type , numOfBuckets)));
    wcntVar_id = context->appendStateVar(PointerType::getUnqual(ArrayType::get(idx_type , numOfBuckets)));
    oidVar_id  = context->appendStateVar(PointerType::getUnqual(oid_type));

    getChild()->produce();
}

void GpuHashRearrange::consume(RawContext* const context, const OperatorState& childState) {
    GpuRawContext * ctx = dynamic_cast<GpuRawContext *>(context);
    if (!ctx){
        string error_msg = "[GpuToCpu: ] Operator only supports code generation using the GpuRawContext";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
    }
    consume(ctx, childState);
}

void GpuHashRearrange::consume(GpuRawContext* const context, const OperatorState& childState) {
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    map<RecordAttribute, RawValueMemory> bindings{childState.getBindings()};


    Plugin * pg       = RawCatalog::getInstance().getPlugin(matExpr[0]->getRegisteredRelName());
    Type   * oid_type = pg->getOIDType()->getLLVMType(llvmContext);

    IntegerType * int32_type        = Type::getInt32Ty  (llvmContext);
    IntegerType * int64_type        = Type::getInt64Ty  (llvmContext);
    Type        * bool_type         = context->createTrue()->getType();

    IntegerType * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    Type   * idx_type = int32_type;

    IntegerType * target_type = size_type;
    if (hashProject) target_type = (IntegerType *) hashProject->getLLVMType(llvmContext);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());

    // std::sort(matExpr.begin(), matExpr.end(), [](const GpuMatExpr& a, const GpuMatExpr& b){
    //     return a.packet < b.packet || a.bitoffset < b.bitoffset;
    // });

    // size_t i = 0;

    std::vector<Value *> buffs;
    // for (size_t p = 0 ; p < packet_widths.size() ; ++p){
    //     // Type * t     = PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(), packet_widths[p]), /* address space */ 1);

    //     size_t bindex  = 0;
    //     size_t packind = 0;

    //     std::vector<Type *> body;
    //     while (i < matExpr.size() && matExpr[i].packet == p){
    //         if (matExpr[i].bitoffset != bindex){
    //             //insert space
    //             assert(matExpr[i].bitoffset > bindex);
    //             body.push_back(Type::getIntNTy(context->getLLVMContext(), (matExpr[i].bitoffset - bindex)));
    //             ++packind;
    //         }

    //         const ExpressionType * out_type = matExpr[i].expr->getExpressionType();

    //         Type * llvm_type = out_type->getLLVMType(context->getLLVMContext());

    //         body.push_back(llvm_type);
    //         bindex = matExpr[i].bitoffset + context->getSizeOf(llvm_type);
    //         matExpr[i].packind = packind++;
    //         ++i;
    //     }
    //     assert(packet_widths[p] >= bindex);

    //     if (packet_widths[p] > bindex) {
    //         body.push_back(Type::getIntNTy(context->getLLVMContext(), (packet_widths[p] - bindex)));
    //     }

    //     Type * t     = StructType::create(body, "hash_rearrange_struct_" + std::to_string(p), true);
    //     Type * t_ptr = PointerType::get(t, /* address space */ 1);

    //     Value * buff = new GlobalVariable( 
    //                                     ArrayType::get(t_ptr, numOfBuckets),
    //                                     false,
    //                                     GlobalVariable::LinkageTypes::InternalLinkage,
    //                                     nullptr,
    //                                     "buff",
    //                                     GlobalVariable::ThreadLocalMode::NotThreadLocal,
    //                                     3,
    //                                     false
    //                                 );

    //     buffs.push_back(buff);
    // }

    size_t max_width = 0;

    for (const auto &e: matExpr){
        PointerType * t_ptr = PointerType::get(e->getExpressionType()->getLLVMType(llvmContext), /* address space */ 0);

        Value * buff = new GlobalVariable(
                                        *(context->getModule()),
                                        ArrayType::get(t_ptr, numOfBuckets),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(t_ptr, numOfBuckets)
                                        ),
                                        "buff",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                    );

        buffs.push_back(buff);

        max_width = std::max(max_width, context->getSizeOf(buff));
    }

    Value * cnt  = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(idx_type , numOfBuckets),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(idx_type, numOfBuckets)
                                        ),
                                        "cnt",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );

    Value * wcnt = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(idx_type , numOfBuckets),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(idx_type, numOfBuckets)
                                        ),
                                        "wcnt",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );


    assert(numOfBuckets < 32);

    Value * numOfBucketsV = ConstantInt::get(target_type, numOfBuckets);
    Value * zero_oid      = ConstantInt::get(idx_type   , 0);

    int64_t cap           = blockSize / max_width;
    Value * capacity      = ConstantInt::get(idx_type   , cap);
    Value * last_index    = ConstantInt::get(idx_type   , cap - 1);

    BasicBlock *initBB  = BasicBlock::Create(llvmContext, "init", F);
    BasicBlock *endIBB  = BasicBlock::Create(llvmContext, "endI", F);

    // if (threadIdx.x < parts){
    //     buff[threadIdx.x] = (uint32_t *) get_buffers();

    //     cnt [threadIdx.x] = 0;
    //     wcnt[threadIdx.x] = 0;
    // }
    // __syncthreads();

    Value * thrd = context->threadIdInBlock();
    Value * numOfBucketsT = ConstantInt::get(thrd->getType(), numOfBuckets);
    Value * cond = Builder->CreateICmpULT(thrd, numOfBucketsT);

    Builder->CreateCondBr(cond, initBB, endIBB);

    Builder->SetInsertPoint(initBB);
    
    Function * get_buffers = context->getFunction("get_buffers");
    Function * syncthreads = context->getFunction("syncthreads");

    std::vector<Value *> idx      {context->createInt32(0), thrd};
    std::vector<Value *> idxStored{context->blockId()     , thrd};
    for (size_t i = 0 ; i < buffs.size() ; ++i){
        Value * buff_thrd        = Builder->CreateInBoundsGEP(buffs[i], idx);
        Value * stored_buff_thrd = Builder->CreateInBoundsGEP(context->getStateVar(buffVar_id[i]), idxStored);

        // Value * new_buff  = Builder->CreateCall(get_buffers);

        // new_buff          = Builder->CreateBitCast(new_buff, buff_thrd->getType()->getPointerElementType());
        Value * stored_buff      = Builder->CreateLoad (stored_buff_thrd);
        Builder->CreateStore(stored_buff, buff_thrd, true);
    }

    Value * cnt__thrd       = Builder->CreateInBoundsGEP(cnt , idx);
    Value * stored_cnt_ptr  = Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
    Value * stored_cnt      = Builder->CreateLoad(stored_cnt_ptr);
    Builder->CreateStore(stored_cnt, cnt__thrd, true);

    Value * wcnt_thrd = Builder->CreateInBoundsGEP(wcnt, idx);
    Value * stored_wcnt_ptr = Builder->CreateInBoundsGEP(context->getStateVar(wcntVar_id), idxStored);
    Value * stored_wcnt     = Builder->CreateLoad(stored_wcnt_ptr);
    Builder->CreateStore(stored_wcnt, wcnt_thrd, true);

    Builder->CreateBr(endIBB);

    Builder->SetInsertPoint(endIBB);

    Builder->CreateCall(syncthreads);

    context->setCurrentEntryBlock(endIBB);


    Builder->SetInsertPoint(context->getEndingBlock());


    BasicBlock *save_initBB  = BasicBlock::Create(llvmContext, "save_init", F);
    BasicBlock *save_endIBB  = BasicBlock::Create(llvmContext, "save_endI", F);

    // __syncthreads();
    // if (threadIdx.x < parts){
    //     buff[threadIdx.x] = (uint32_t *) get_buffers();

    //     cnt [threadIdx.x] = 0;
    //     wcnt[threadIdx.x] = 0;
    // }

    // Value * thrd = context->threadIdInBlock();
    // Value * numOfBucketsT = ConstantInt::get(thrd->getType(), numOfBuckets);
    // Value * cond = Builder->CreateICmpULT(thrd, numOfBucketsT);

    Builder->CreateCall(syncthreads);
    Builder->CreateCondBr(cond, save_initBB, save_endIBB);

    Builder->SetInsertPoint(save_initBB);

    for (size_t i = 0 ; i < buffs.size() ; ++i){
        Value * buff_thrd        = Builder->CreateInBoundsGEP(buffs[i], idx);
        Value * stored_buff_thrd = Builder->CreateInBoundsGEP(context->getStateVar(buffVar_id[i]), idxStored);

        // Value * new_buff  = Builder->CreateCall(get_buffers);

        // new_buff          = Builder->CreateBitCast(new_buff, buff_thrd->getType()->getPointerElementType());
        Value * stored_buff      = Builder->CreateLoad (buff_thrd, true);
        Builder->CreateStore(stored_buff, stored_buff_thrd);
    }

    cnt__thrd       = Builder->CreateInBoundsGEP(cnt , idx);
    stored_cnt_ptr  = Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
    stored_cnt      = Builder->CreateLoad(cnt__thrd, true);
    Builder->CreateStore(stored_cnt, stored_cnt_ptr);

    wcnt_thrd       = Builder->CreateInBoundsGEP(wcnt, idx);
    stored_wcnt_ptr = Builder->CreateInBoundsGEP(context->getStateVar(wcntVar_id), idxStored);
    stored_wcnt     = Builder->CreateLoad(wcnt_thrd, true);
    Builder->CreateStore(stored_wcnt, stored_wcnt_ptr);

    Builder->CreateBr(save_endIBB);

    context->setEndingBlock(save_endIBB);




    Builder->SetInsertPoint(insBB);

    //Generate target
    ExpressionGeneratorVisitor exprGenerator{context, childState};
    Value * target            = hashExpr->accept(exprGenerator).value;

    target = Builder->CreateURem(target, numOfBucketsV);
    target->setName("target");

    map<RecordAttribute, RawValueMemory> variableBindings;
    AllocaInst * not_done_ptr = context->CreateEntryBlockAlloca(F, "not_done_ptr", bool_type);
    Builder->CreateStore(context->createTrue(), not_done_ptr);

    if (hashProject){
        //Save hash in bindings
        AllocaInst * hash_ptr = context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);
        Builder->CreateStore(target, hash_ptr);

        RawValueMemory mem_hashWrapper;
        mem_hashWrapper.mem      = hash_ptr;
        mem_hashWrapper.isNull   = context->createFalse();
        variableBindings[*hashProject] = mem_hashWrapper;
    }

    BasicBlock * startInsBB = BasicBlock::Create(llvmContext, "startIns", F);
    BasicBlock * endIndBB   = BasicBlock::Create(llvmContext, "endInd"  , F);

    Builder->CreateBr(startInsBB);
    Builder->SetInsertPoint(startInsBB);

    AllocaInst * mask_ptr = context->CreateEntryBlockAlloca(F, "mask_ptr", int32_type);
    Builder->CreateStore(UndefValue::get(int32_type), mask_ptr);

    Value * laneid = context->laneId();
    


    // uint32_t mask;
    // #pragma unroll
    // for (uint32_t j = 0 ; j < parts ; ++j){
    //     uint32_t tmp = __ballot((!done) && (h == j));
    //     if (j == h) mask = tmp;
    // }
    for (int i = 0 ; i < numOfBuckets ; ++i){
        Value * comp = Builder->CreateICmpEQ(target, ConstantInt::get(target_type, i));
        Value * vote = gpu_intrinsic::ballot(context, Builder->CreateAnd(comp, Builder->CreateLoad(not_done_ptr)));

        BasicBlock * stBB    = BasicBlock::Create(llvmContext, "st", F);
        BasicBlock * mgBB    = BasicBlock::Create(llvmContext, "mg", F);

        Builder->CreateCondBr(comp, stBB, mgBB);

        Builder->SetInsertPoint(stBB);
        Builder->CreateStore(vote, mask_ptr);
        Builder->CreateBr(mgBB);

        Builder->SetInsertPoint(mgBB);
    }

    BasicBlock * incIdxBB    = BasicBlock::Create(llvmContext, "incIdx", F);
    BasicBlock * mgiIdxBB    = BasicBlock::Create(llvmContext, "mgiIdx", F);

    AllocaInst * indx_ptr    = context->CreateEntryBlockAlloca(F, "indx_ptr", idx_type); //consider lowering it into 32bit
    Builder->CreateStore(UndefValue::get(idx_type), indx_ptr);


    // uint32_t idx;

    // uint32_t h_leader    = mask & -mask;
    // uint32_t lanemask_eq = 1 << get_laneid();
    Value * mask     = Builder->CreateLoad(mask_ptr);
    Value * h_leader = Builder->CreateAnd(mask, Builder->CreateNeg(mask));

    Function * f_lanemask_eq = context->getFunction("llvm.nvvm.read.ptx.sreg.lanemask.eq");
    Value * lanemask_eq = Builder->CreateCall(f_lanemask_eq);

    // if (lanemask_eq == h_leader) idx = atomicAdd_block(cnt + h, __popc(mask));
    Value * inc_cond = Builder->CreateICmpEQ(lanemask_eq, h_leader);
    Builder->CreateCondBr(inc_cond, incIdxBB, mgiIdxBB);


    // SmallVectorImpl< StringRef > SSNs{100};
    // llvmContext.getSyncScopeNames(SSNs);
    // for (const auto &t: SSNs) std::cout << t << std::endl;


    Function * popc = context->getFunction("llvm.ctpop.i32");

    Builder->SetInsertPoint(incIdxBB);

    idx[1] = target;
    Value * cnt_ptr         = Builder->CreateInBoundsGEP(cnt, idx);
    Value * pop             = Builder->CreateCall(popc, mask);
    //TODO: make it block atomic! but it is on shared memory, so it does not matter :P
    Value * new_idx         = Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Add, cnt_ptr, Builder->CreateZExt(pop, idx_type), AtomicOrdering::Monotonic);
    Builder->CreateStore(new_idx, indx_ptr);
    Builder->CreateBr(mgiIdxBB);

    Builder->SetInsertPoint(mgiIdxBB);


    // idx  = __shfl(idx, __popc(h_leader - 1)) + __popc(mask & ((1 << get_laneid()) - 1));

    Value * h_leader_lt    = Builder->CreateSub (h_leader, ConstantInt::get((IntegerType *) (h_leader->getType()), 1)); 
    Value * leader_id      = Builder->CreateCall(popc, h_leader_lt);

    Function * shfl        = context->getFunction("llvm.nvvm.shfl.idx.i32");
    Function * lanemask_lt = context->getFunction("llvm.nvvm.read.ptx.sreg.lanemask.lt");

    std::vector<Value *> args{  Builder->CreateLoad(indx_ptr), 
                                leader_id,
                                context->createInt32(warp_size - 1)
                            };
    Value * idx_g  = Builder->CreateAdd(
                        Builder->CreateCall(shfl, args),
                        Builder->CreateCall(popc, std::vector<Value *>{
                            Builder->CreateAnd(
                                mask,
                                Builder->CreateCall(lanemask_lt)
                            )
                        })
                    );

    BasicBlock * storeBB = BasicBlock::Create(llvmContext, "store", F);
    BasicBlock * mergeBB = BasicBlock::Create(llvmContext, "merge", F);


    Value * storeCond = Builder->CreateAnd( 
                                        Builder->CreateLoad(not_done_ptr), 
                                        Builder->CreateICmpULT(idx_g, capacity)
                                    );
    //if (!done){
    //    if (idx < h_vector_size/4){
    Builder->CreateCondBr(storeCond, storeBB, mergeBB);

    Builder->SetInsertPoint(storeBB);


    //        uint32_t * b_old = buff[h];
    //        reinterpret_cast<int4 *>(b_old)[idx] = tmp.vec;
    //        __threadfence_block(); //even safer and with the same performance!!!! : __threadfence();
    std::vector<Value *> buff_ptrs;
    // std::vector<Value *> out_ptrs;
    // std::vector<Value *> out_vals;

    idx[1] = target;
    for (size_t i = 0 ; i < buffs.size() ; ++i){
        // out_vals.push_back(UndefValue::get(b->getType()->getPointerElementType()->getPointerElementType()));
        buff_ptrs.push_back(Builder->CreateLoad(Builder->CreateInBoundsGEP(buffs[i], idx), true));
        Value * out_ptr = Builder->CreateInBoundsGEP(buff_ptrs.back(), idx_g);

        ExpressionGeneratorVisitor exprGenerator(context, childState);
        RawValue valWrapper = matExpr[i]->accept(exprGenerator);
        Builder->CreateStore(valWrapper.value, out_ptr);
    }

    // for (const GpuMatExpr &mexpr: matExpr){
    //     ExpressionGeneratorVisitor exprGenerator(context, childState);
    //     RawValue valWrapper = mexpr.expr->accept(exprGenerator);
        
    //     out_vals[mexpr.packet] = Builder->CreateInsertValue(out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
    // }

    // for (size_t i = 0 ; i < out_ptrs.size() ; ++i){
    //     // Builder->CreateStore(out_vals[i], out_ptrs[i]);
    // }

    Function * threadfence = context->getFunction("threadfence");
    Builder->CreateCall(threadfence);

    Value * w = Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Add, 
                                        Builder->CreateInBoundsGEP(wcnt, idx), 
                                        ConstantInt::get(idx_type, 1),
                                        AtomicOrdering::Monotonic);

    BasicBlock * replaceBB = BasicBlock::Create(llvmContext, "replace", F);
    BasicBlock * remergeBB = BasicBlock::Create(llvmContext, "remerge", F);

    Builder->CreateCondBr(  
                        Builder->CreateICmpEQ(w, last_index),
                        replaceBB,
                        remergeBB);

    Builder->SetInsertPoint(replaceBB);

    for (size_t i = 0 ; i < buffs.size() ; ++i){
        Value * buff_thrd = Builder->CreateInBoundsGEP(buffs[i], idx);
        Value * new_buff  = Builder->CreateCall(get_buffers);
        new_buff          = Builder->CreateBitCast(new_buff, buff_thrd->getType()->getPointerElementType());
        Builder->CreateStore(new_buff, buff_thrd, true);
    }


    Function * threadfence_block = context->getFunction("threadfence_block");
    Builder->CreateCall(threadfence_block);


    Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Xchg, 
                                        Builder->CreateInBoundsGEP(wcnt, idx), 
                                        ConstantInt::get(idx_type, 0),
                                        AtomicOrdering::Monotonic);

    Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Xchg, 
                                        Builder->CreateInBoundsGEP( cnt, idx), 
                                        ConstantInt::get(idx_type, 0),
                                        AtomicOrdering::Monotonic);


    // call parent
    RecordAttribute tupCnt  = RecordAttribute(matExpr[0]->getRegisteredRelName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    AllocaInst * blockN_ptr      = context->CreateEntryBlockAlloca(F, "blockN_ptr", oid_type);
    Builder->CreateStore(ConstantInt::get(oid_type, cap), blockN_ptr);

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    variableBindings[tupCnt] = mem_cntWrapper;



    Value * new_oid = Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Add,
                            ((GpuRawContext *) context)->getStateVar(oidVar_id),
                            ConstantInt::get(oid_type, cap),
                            AtomicOrdering::Monotonic);
    new_oid->setName("oid");

    AllocaInst * new_oid_ptr = context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
    Builder->CreateStore(new_oid, new_oid_ptr);

    RecordAttribute tupleIdentifier = RecordAttribute(matExpr[0]->getRegisteredRelName(),  activeLoop, pg->getOIDType());
    
    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem      = new_oid_ptr;
    mem_oidWrapper.isNull   = context->createFalse();
    variableBindings[tupleIdentifier] = mem_oidWrapper;

    for (size_t i = 0 ; i < matExpr.size() ; ++i){

        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(F,
                                "mem_" + matExpr[i]->getRegisteredAttrName(),
                                buff_ptrs[i]->getType());

        Builder->CreateStore(buff_ptrs[i], mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();

        RecordAttribute battr(  RecordAttribute{
                                    matExpr[i]->getRegisteredRelName() , 
                                    matExpr[i]->getRegisteredAttrName(), 
                                    matExpr[i]->getExpressionType()    
                                },
                                true
                            );

        variableBindings[battr] = mem_valWrapper;
    }

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    Builder->CreateBr(remergeBB);
    Builder->SetInsertPoint(remergeBB);

    Builder->CreateStore(context->createFalse(), not_done_ptr);

    Builder->CreateBr(mergeBB);
    Builder->SetInsertPoint(mergeBB);

    Value * any = gpu_intrinsic::any(context, Builder->CreateLoad(not_done_ptr));

    Builder->CreateCondBr(any, startInsBB, endIndBB);

    Builder->SetInsertPoint(endIndBB);







    // Function * get_buffer = context->getFunction("get_buffer");

    // for (size_t i = 0 ; i < wantedFields.size() ; ++i){
    //     RecordAttribute tblock{*(wantedFields[i]), true};
    //     Value * size     = context->createSizeT(blockSize * context->getSizeOf(wantedFields[i]->getLLVMType(llvmContext)));

    //     Value * new_buff = Builder->CreateCall(get_buffer, std::vector<Value *>{size});

    //     new_buff = Builder->CreateBitCast(new_buff, tblock.getLLVMType(llvmContext));

    //     Builder->CreateStore(new_buff, Builder->CreateInBoundsGEP(curblk, std::vector<Value *>{context->createInt32(0), context->createInt32(i)}));
    // }

    // Builder->CreateStore(ConstantInt::get(oid_type, 0), indx_addr);

    // Builder->CreateBr(mergeBB);

    // // else
    // Builder->SetInsertPoint(elseBB);

    // Builder->CreateStore(Builder->CreateAdd(indx, ConstantInt::get(oid_type, 1)), indx_addr);

    // Builder->CreateBr(mergeBB);

    // // merge
    // Builder->SetInsertPoint(mergeBB);

    // flush remaining elements
    consume_flush();

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
}

void GpuHashRearrange::consume_flush(){
    std::vector<Type *> args{context->getStateVars()};

    context->pushNewPipeline();

    for (Type * t: args) context->appendStateVar(t);

    LLVMContext & llvmContext   = context->getLLVMContext();

    Plugin * pg       = RawCatalog::getInstance().getPlugin(matExpr[0]->getRegisteredRelName());
    Type   * oid_type = pg->getOIDType()->getLLVMType(llvmContext);

    IntegerType * int32_type   = Type::getInt32Ty  (llvmContext);
    IntegerType * int64_type   = Type::getInt64Ty  (llvmContext);

    IntegerType * size_type;
    if      (sizeof(size_t) == 4) size_type = int32_type;
    else if (sizeof(size_t) == 8) size_type = int64_type;
    else                          assert(false);

    IntegerType * target_type = size_type;
    if (hashProject) target_type = (IntegerType *) hashProject->getLLVMType(llvmContext);

    context->setGlobalFunction();

    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();
    //Get the ENTRY BLOCK
    context->setCurrentEntryBlock(Builder->GetInsertBlock());

    BasicBlock *mainBB  = BasicBlock::Create(llvmContext, "main", F);

    Builder->SetInsertPoint(mainBB);

    BasicBlock *initBB  = BasicBlock::Create(llvmContext, "init", F);
    BasicBlock *endIBB  = BasicBlock::Create(llvmContext, "endI", F);

    context->setEndingBlock(endIBB);

    // if (threadIdx.x < parts){
    //     buff[threadIdx.x] = (uint32_t *) get_buffers();

    //     cnt [threadIdx.x] = 0;
    //     wcnt[threadIdx.x] = 0;
    // }
    // __syncthreads();

    Value * thrd          = context->threadIdInBlock();
    Value * target        = Builder->CreateZExtOrTrunc(thrd, target_type);
    Value * numOfBucketsT = ConstantInt::get(target_type, numOfBuckets);
    Value * cond          = Builder->CreateICmpULT(target, numOfBucketsT);

    Builder->CreateCondBr(cond, initBB, endIBB);

    Builder->SetInsertPoint(initBB);
    
    Function * get_buffers = context->getFunction("get_buffers");
    Function * syncthreads = context->getFunction("syncthreads");

    std::vector<Value *> idx      {context->createInt32(0), thrd};
    std::vector<Value *> idxStored{context->blockId()     , thrd};

    std::vector<Value *> buff_ptrs;
    for (size_t i = 0 ; i < matExpr.size() ; ++i){
        Value * stored_buff_thrd = Builder->CreateInBoundsGEP(context->getStateVar(buffVar_id[i]), idxStored);

        buff_ptrs.push_back(Builder->CreateLoad (stored_buff_thrd));
    }

    // Value * cnt__thrd       = Builder->CreateInBoundsGEP(cnt , idx);
    Value * stored_cnt_ptr  = Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
    Value * cnt             = Builder->CreateLoad(stored_cnt_ptr);

    // call parent
    map<RecordAttribute, RawValueMemory> variableBindings;

    RecordAttribute tupCnt  = RecordAttribute(matExpr[0]->getRegisteredRelName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    AllocaInst * blockN_ptr      = context->CreateEntryBlockAlloca(F, "blockN_ptr", oid_type);

    if (hashProject){
        //Save hash in bindings
        AllocaInst * hash_ptr = context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);
        Builder->CreateStore(target, hash_ptr);

        RawValueMemory mem_hashWrapper;
        mem_hashWrapper.mem      = hash_ptr;
        mem_hashWrapper.isNull   = context->createFalse();
        variableBindings[*hashProject] = mem_hashWrapper;
    }

    cnt = Builder->CreateZExt(cnt, oid_type);

    Builder->CreateStore(cnt, blockN_ptr);

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem      = blockN_ptr;
    mem_cntWrapper.isNull   = context->createFalse();
    variableBindings[tupCnt] = mem_cntWrapper;

    Value * new_oid = Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Add,
                            ((GpuRawContext *) context)->getStateVar(oidVar_id),
                            cnt,
                            AtomicOrdering::Monotonic);
    new_oid->setName("oid");

    AllocaInst * new_oid_ptr = context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
    Builder->CreateStore(new_oid, new_oid_ptr);

    RecordAttribute tupleIdentifier = RecordAttribute(matExpr[0]->getRegisteredRelName(),  activeLoop, pg->getOIDType());
    
    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem      = new_oid_ptr;
    mem_oidWrapper.isNull   = context->createFalse();
    variableBindings[tupleIdentifier] = mem_oidWrapper;

    for (size_t i = 0 ; i < matExpr.size() ; ++i){

        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(F,
                                "mem_" + matExpr[i]->getRegisteredAttrName(),
                                buff_ptrs[i]->getType());

        Builder->CreateStore(buff_ptrs[i], mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();

        RecordAttribute battr(  RecordAttribute{
                                    matExpr[i]->getRegisteredRelName() , 
                                    matExpr[i]->getRegisteredAttrName(), 
                                    matExpr[i]->getExpressionType()    
                                },
                                true
                            );

        variableBindings[battr] = mem_valWrapper;
    }

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    Builder->CreateBr(endIBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(mainBB);

    Builder->SetInsertPoint(context->getEndingBlock());
    // Builder->CreateRetVoid();

    context->popNewPipeline();

    closingPip = context->removeLatestPipeline();
}

void GpuHashRearrange::open (RawPipeline * pip){
    int device = get_device();

    execution_conf ec = pip->getExecConfiguration();

    size_t grid_size  = ec.gridSize();
    std::cout << grid_size << std::endl;

    void   ** buffs;
    int32_t * cnts ;
    size_t  * oid  ;
    gpu_run(cudaMalloc(&buffs, sizeof(void  *) * numOfBuckets * buffVar_id.size() * grid_size));
    gpu_run(cudaMalloc(&cnts , sizeof(int32_t) * numOfBuckets * 2                 * grid_size));
    gpu_run(cudaMalloc(&oid  , sizeof(size_t )                   ));

    gpu_run(cudaMemset(cnts, 0, numOfBuckets * 2 * sizeof(int32_t) * grid_size));
    gpu_run(cudaMemset(oid , 0,                    sizeof(size_t )            ));

    // for (int i = 0 ; i < numOfBuckets * 2; ++i) cnts[i] = 0;
    // *oid = 0;

    pip->setStateVar<int32_t *>(cntVar_id , cnts                           );
    pip->setStateVar<int32_t *>(wcntVar_id, cnts + numOfBuckets * grid_size);
    pip->setStateVar<size_t  *>(oidVar_id , oid                            );

    void ** h_buffs = (void **) malloc(sizeof(void  *) * numOfBuckets * buffVar_id.size() * grid_size);
    
    for (int i = 0 ; i < numOfBuckets * buffVar_id.size() * grid_size; ++i){
        h_buffs[i] = buffer_manager<int32_t>::h_get_buffer(device); // FIMXE: assumes that all buffers need at most a h_host_vector * sizeof(int32_t) size

    }

    gpu_run(cudaMemcpy(buffs, h_buffs, sizeof(void  *) * numOfBuckets * buffVar_id.size() * grid_size, cudaMemcpyDefault));

    free(h_buffs);

    for (int i = 0 ; i < buffVar_id.size() ; ++i){
        pip->setStateVar<void **>(buffVar_id[i], buffs + numOfBuckets * i * grid_size);
    }
}

void GpuHashRearrange::close(RawPipeline * pip){
    // ((void (*)(void *)) this->flushFunc)(pip->getState());
    void *KernelParams[] = {pip->getState()};
    launch_kernel((CUfunction) closingPip->getKernel(), KernelParams);
    gpu_run(cudaDeviceSynchronize());

    gpu_run(cudaFree(pip->getStateVar<size_t *>(cntVar_id    )));
    gpu_run(cudaFree(pip->getStateVar<size_t *>(oidVar_id    )));
    gpu_run(cudaFree(pip->getStateVar<void  **>(buffVar_id[0]))); //FIXME: release buffers before freeing memory!
    // wcntVar_id is part of cntVar, so they are freed together
    // rest of mem for buffers is part of buffVar_id
}
