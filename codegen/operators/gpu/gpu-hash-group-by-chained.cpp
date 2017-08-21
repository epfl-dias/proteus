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

#include "operators/gpu/gpu-hash-group-by-chained.hpp"
#include "operators/gpu/gmonoids.hpp"
#include "expressions/expressions-generator.hpp"

GpuHashGroupByChained::GpuHashGroupByChained(
            const std::vector<GpuAggrMatExpr>              &agg_exprs, 
            const std::vector<size_t>                      &packet_widths,
            const std::vector<expressions::Expression *>    key_expr,
            RawOperator * const                             child,

            int                                             hash_bits,

            GpuRawContext *                                 context,
            string                                          opLabel): 
                agg_exprs(agg_exprs),
                packet_widths(packet_widths),
                key_expr(key_expr),
                hash_bits(hash_bits),
                UnaryRawOperator(child), 
                context(context),
                opLabel(opLabel){
}

void GpuHashGroupByChained::produce() {
    // context->pushNewPipeline(); //FIXME: find a better way to do this
    buildHashTableFormat();

    getChild()->produce();

    // context->compileAndLoad(); //FIXME: Remove!!!! causes an extra compilation! this compile will be done again later!
    // Get kernel function
    // probe_kernel = context->getKernel();
    // context->popNewPipeline(); //FIXME: find a better way to do this
    // generate_scan();
}

void GpuHashGroupByChained::consume(RawContext* const context, const OperatorState& childState) {
    generate_build(context, childState);
}

void GpuHashGroupByChained::buildHashTableFormat(){
    agg_exprs.emplace_back(new expressions::IntConstant(0), 0,  0);
    
    size_t bitoffset = 32;
    for (const auto &key: key_expr){
        agg_exprs.emplace_back(key                , 0, bitoffset);

        const ExpressionType * out_type = key->getExpressionType();

        if (!out_type->isPrimitive()){
            string error_msg("[GpuHashGroupByChained: ] Currently only supports keys of primitive type");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }

        Type * llvm_type = ((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext());

        bitoffset += llvm_type->getPrimitiveSizeInBits();
    }

    std::sort(agg_exprs.begin(), agg_exprs.end(), [](const GpuAggrMatExpr& a, const GpuAggrMatExpr& b){
        return a.packet < b.packet || a.bitoffset < b.bitoffset;
    });

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *t_head_ptr = PointerType::get(int32_type, /* address space */ 1);
    head_param_id = context->appendStateVar(t_head_ptr);

    size_t i = 0;

    for (size_t p = 0 ; p < packet_widths.size() ; ++p){
        // Type * t     = PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(), packet_widths[p]), /* address space */ 1);

        size_t bindex  = 0;
        size_t packind = 0;

        std::vector<Type *> body;
        while (i < agg_exprs.size() && agg_exprs[i].packet == p){
            if (agg_exprs[i].bitoffset != bindex){
                //insert space
                assert(agg_exprs[i].bitoffset > bindex);
                body.push_back(Type::getIntNTy(context->getLLVMContext(), (agg_exprs[i].bitoffset - bindex)));
                ++packind;
            }

            const ExpressionType * out_type = agg_exprs[i].expr->getExpressionType();

            if (!out_type->isPrimitive()){
                string error_msg("[GpuExprMaterializer: ] Currently only supports materialization of primitive types");
                LOG(ERROR)<< error_msg;
                throw runtime_error(error_msg);
            }

            Type * llvm_type = ((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext());

            body.push_back(llvm_type);
            bindex = agg_exprs[i].bitoffset + llvm_type->getPrimitiveSizeInBits();
            agg_exprs[i].packind = packind++;
            ++i;
        }
        assert(packet_widths[p] >= bindex);

        if (packet_widths[p] > bindex) {
            body.push_back(Type::getIntNTy(context->getLLVMContext(), (packet_widths[p] - bindex)));
        }

        Type * t     = StructType::create(body, opLabel + "_struct_" + std::to_string(p), true);
        Type * t_ptr = PointerType::get(t, /* address space */ 1);

        out_param_ids.push_back(context->appendStateVar(t_ptr));//, true, false));
    }
    assert(i == agg_exprs.size());

    agg_exprs.erase(agg_exprs.begin()); //erase dummy entry for next

    // Type * t     = PointerType::get(((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext()), /* address space */ 1);
    // out_param_id = context->appendStateVar(t    );//, true, false);

    Type * t_cnt = PointerType::get(int32_type, /* address space */ 1);
    cnt_param_id = context->appendStateVar(t_cnt);//, true, false);
}

//NOTE: no MOD hashtable_size here!
Value * GpuHashGroupByChained::hash(Value * key){
    IRBuilder<>    *Builder     = context->getBuilder();

    Value * hash = key;

    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));
    hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0x85ebca6b));
    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 13));
    hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0xc2b2ae35));
    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));

    return hash;
}

//boost::hash_combine
// seed ^= hash_value(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

//NOTE: no MOD hashtable_size here!
Value * GpuHashGroupByChained::hash(Value * old_seed, Value * key){
    IRBuilder<>    *Builder     = context->getBuilder();

    Value * hv = hash(key);
    
    hv = Builder->CreateAdd(hv, ConstantInt::get(hv->getType(), 0x9e3779b9));
    hv = Builder->CreateAdd(hv, Builder->CreateShl (old_seed,  6));
    hv = Builder->CreateAdd(hv, Builder->CreateLShr(old_seed,  2));
    hv = Builder->CreateXor(hv, old_seed);

    return hv;
}

Value * GpuHashGroupByChained::hash(const std::vector<expressions::Expression *> &exprs, RawContext* const context, const OperatorState& childState){
    IRBuilder<>    *Builder     = context->getBuilder();
    ExpressionGeneratorVisitor exprGenerator(context, childState);
    RawValue keyWrapper = exprs[0]->accept(exprGenerator); //FIXME hash composite key!
    Value * hash = GpuHashGroupByChained::hash(keyWrapper.value);

    for (size_t i = 1 ; i < exprs.size() ; ++i){
        RawValue keyWrapper = exprs[i]->accept(exprGenerator); //FIXME hash composite key!
        hash = GpuHashGroupByChained::hash(hash, keyWrapper.value);
    }

    hash = Builder->CreateAnd(hash, ConstantInt::get(hash->getType(), ((size_t(1)) << hash_bits) - 1));
    return hash;
}

void GpuHashGroupByChained::generate_build(RawContext* const context, const OperatorState& childState) {
    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    // PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
    // Type *int8_type = Type::getInt8Ty(llvmContext);
    // PointerType *void_ptr_type = PointerType::get(int8_type, 0);
    // Type *int64_type = Type::getInt64Ty(llvmContext);
    // Type *int32_type = Type::getInt32Ty(llvmContext);
    Value * v_true  = ConstantInt::getTrue (llvmContext);
    Value * v_false = ConstantInt::getFalse(llvmContext);

    Value * out_cnt = ((const GpuRawContext *) context)->getStateVar(cnt_param_id);
    out_cnt ->setName(opLabel + "_cnt_ptr");

    Value * head_ptr = ((const GpuRawContext *) context)->getStateVar(head_param_id);
    head_ptr->setName(opLabel + "_head_ptr");

    Value * hash = GpuHashGroupByChained::hash(key_expr, context, childState);

    //current = head[hash(key)]
    Value * current = Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash), context->getSizeOf(head_ptr->getType()->getPointerElementType()));
    current->setName("current");

    AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", current->getType());
    Builder->CreateStore(current, mem_current);

    AllocaInst *mem_idx     = context->CreateEntryBlockAlloca(TheFunction, "mem_idx", out_cnt->getType()->getPointerElementType());
    Builder->CreateStore(UndefValue::get(out_cnt->getType()->getPointerElementType()), mem_idx);

    AllocaInst *mem_written = context->CreateEntryBlockAlloca(TheFunction, "mem_written", v_false->getType());
    Builder->CreateStore(v_false, mem_written);

    Value * eochain   = ConstantInt::get(current->getType(), ~((size_t) 0));

    // std::vector<Value *> out_ptrs;
    std::vector<Value *> out_vals;

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);
        if (out_param_ids.size() != 1){
            out_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
        } else {
            out_ptr->setName(opLabel + "_data_ptr");
        }
        // out_ptrs.push_back(out_ptr);

        // out_ptr->addAttr(Attribute::getWithAlignment(llvmContext, context->getSizeOf(out_ptr)));

        // out_ptrs.push_back(Builder->CreateInBoundsGEP(out_ptr, old_cnt));
        out_vals.push_back(UndefValue::get(out_ptr->getType()->getPointerElementType()));
    }

    out_vals[0] = Builder->CreateInsertValue(out_vals[0], eochain, 0);

    for (const GpuAggrMatExpr &mexpr: agg_exprs){
        ExpressionGeneratorVisitor exprGenerator(context, childState);
        RawValue valWrapper = mexpr.expr->accept(exprGenerator);
        
        out_vals[mexpr.packet] = Builder->CreateInsertValue(out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
    }

    // BasicBlock *InitCondBB  = BasicBlock::Create(llvmContext, "setHeadCond", TheFunction);
    BasicBlock *InitThenBB  = BasicBlock::Create(llvmContext, "setHead", TheFunction);
    BasicBlock *InitMergeBB = BasicBlock::Create(llvmContext, "cont"   , TheFunction);

    Value * init_cond = Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

    // if (current == ((uint32_t) -1)){
    Builder->CreateCondBr(init_cond, InitThenBB, InitMergeBB);

    Builder->SetInsertPoint(InitThenBB);

    //index
    Value * old_cnt  = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
                                                out_cnt,
                                                ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
                                                llvm::AtomicOrdering::Monotonic);
    old_cnt->setName("index");
    Builder->CreateStore(old_cnt, mem_idx);

            // next[idx].sum  = val;
            // next[idx].key  = key;
            // next[idx].next =  -1;

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);

        Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
        Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i]/8);
    }

            // written = true;
    Builder->CreateStore(v_true, mem_written);

    ((GpuRawContext *) context)->createMembar_gl();

            // current = atomicCAS(&(first[bucket]), -1, idx);
    Value * old_current = Builder->CreateAtomicCmpXchg(Builder->CreateInBoundsGEP(head_ptr, hash),
                                                eochain,
                                                old_cnt,
                                                llvm::AtomicOrdering::Monotonic,
                                                llvm::AtomicOrdering::Monotonic);
    Builder->CreateStore(Builder->CreateExtractValue(old_current, 0), mem_current);

    // }
    Builder->CreateBr(InitMergeBB);
    Builder->SetInsertPoint(InitMergeBB);


    // if (current != ((uint32_t) -1)){
    //     while (true) {

    BasicBlock *ThenBB  = BasicBlock::Create(llvmContext, "chainFollow"    , TheFunction);
    BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont"           , TheFunction);

    Value * chain_cond = Builder->CreateICmpNE(Builder->CreateLoad(mem_current), eochain);

    Builder->CreateCondBr(chain_cond, ThenBB, MergeBB);

    Builder->SetInsertPoint(ThenBB);
    current = Builder->CreateLoad(mem_current);

    Value * next_bucket_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), Builder->CreateLoad(mem_current));
    Value * next_bucket     = Builder->CreateAlignedLoad(next_bucket_ptr, packet_widths[0]/8, "next_bucket");

                // int32_t   next_bucket = next[current].next;
    Value * next            = Builder->CreateExtractValue(next_bucket, 0);
    // Value * key             = Builder->CreateExtractValue(next_bucket, 1);

    BasicBlock *BucketFoundBB = BasicBlock::Create(llvmContext, "BucketFound", TheFunction);
    BasicBlock *ContFollowBB  = BasicBlock::Create(llvmContext, "ContFollow" , TheFunction);


    Value * bucket_cond = v_true;
    for (size_t i = 0 ; i < key_expr.size() ; ++i){
        ExpressionGeneratorVisitor exprGenerator(context, childState);
        RawValue keyWrapper = key_expr[i]->accept(exprGenerator);

        Value  * key        = Builder->CreateExtractValue(next_bucket, i + 1);
        Value  * key_comp   = Builder->CreateICmpEQ(key, keyWrapper.value);

        bucket_cond         = Builder->CreateAnd(bucket_cond, key_comp);
    }
                // if (next[current].key == key) {
    Builder->CreateCondBr(bucket_cond, BucketFoundBB, ContFollowBB);

    Builder->SetInsertPoint(BucketFoundBB);

    BasicBlock *InvalidateEntryBB = BasicBlock::Create(llvmContext, "InvalidateEntry", TheFunction);

                    // atomicAdd(&(next[current].sum), val);
    for (size_t i = 0 ; i < agg_exprs.size() ; ++i){
        if (agg_exprs[i].is_aggregation()){
            gpu::Monoid * gm = gpu::Monoid::get(agg_exprs[i].m);
            std::vector<Value *> tmp{current, context->createInt32(agg_exprs[i].packind)};

            Value * aggr     = Builder->CreateExtractValue(out_vals[agg_exprs[i].packet], agg_exprs[i].packind);

            Value * gl_accum = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[agg_exprs[i].packet]), tmp);

            gm->createAtomicUpdate(context, gl_accum, aggr, llvm::AtomicOrdering::Monotonic);
        }
    }

    Builder->CreateCondBr(Builder->CreateLoad(mem_written), InvalidateEntryBB, MergeBB);
                    // if (written) next[idx].next = idx;
                    // break;
    Builder->SetInsertPoint(InvalidateEntryBB);

    Value * str = UndefValue::get(((const GpuRawContext *) context)->getStateVar(out_param_ids[0])->getType()->getPointerElementType());
    str = Builder->CreateInsertValue(str, Builder->CreateLoad(mem_idx), 0);
    Builder->CreateAlignedStore(str, Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), Builder->CreateLoad(mem_idx)), packet_widths[0]/8);


    Builder->CreateBr(MergeBB);


    Builder->SetInsertPoint(ContFollowBB);
        // current = next_bucket;
    Builder->CreateStore(next, mem_current);

    Value * chain_end_cond = Builder->CreateICmpEQ(next, eochain);

    BasicBlock *EndFoundBB = BasicBlock::Create(llvmContext, "BucketFound", TheFunction);

    Builder->CreateCondBr(chain_end_cond, EndFoundBB, ThenBB);

    Builder->SetInsertPoint(EndFoundBB);


    BasicBlock *CreateBucketBB = BasicBlock::Create(llvmContext, "CreateBucket", TheFunction);
    BasicBlock *ContLinkingBB  = BasicBlock::Create(llvmContext, "ContLinking" , TheFunction);

                    // if (!written){
    Builder->CreateCondBr(Builder->CreateLoad(mem_written), ContLinkingBB, CreateBucketBB);

    Builder->SetInsertPoint(CreateBucketBB);

        //index
    old_cnt  = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
                                                out_cnt,
                                                ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
                                                llvm::AtomicOrdering::Monotonic);
    Builder->CreateStore(old_cnt, mem_idx);

                        // next[idx].sum  = val;
                        // next[idx].key  = key;
                        // next[idx].next =  -1;

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);

        Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
        Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i]/8);
    }

                        // written = true;
    Builder->CreateStore(v_true, mem_written);

                        // __threadfence();
                    // }
    ((GpuRawContext *) context)->createMembar_gl();
    Builder->CreateBr(ContLinkingBB);

    Builder->SetInsertPoint(ContLinkingBB); 
                    // new_next = atomicCAS(&(next[current].next), -1, idx);
    std::vector<Value *> tmp{current, context->createInt32(0)};
    Value * new_next = Builder->CreateAtomicCmpXchg(Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), tmp),
                                                eochain,
                                                Builder->CreateLoad(mem_idx),
                                                llvm::AtomicOrdering::Monotonic,
                                                llvm::AtomicOrdering::Monotonic);
                    // current = new_next;
    Builder->CreateStore(Builder->CreateExtractValue(new_next, 0), mem_current);
                    // if (new_next == ((uint32_t) -1)) 
    // Value * valid_insert = Builder->CreateICmpEQ(new_next, eochain);
    Builder->CreateCondBr(Builder->CreateExtractValue(new_next, 1), MergeBB, ThenBB);

    Builder->SetInsertPoint(MergeBB);

    ((GpuRawContext *) context)->registerOpen ([this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose([this](RawPipeline * pip){this->close(pip);});
}

// void GpuHashGroupByChained::generate_probe(RawContext* const context, const OperatorState& childState) {
//     IRBuilder<>    *Builder     = context->getBuilder();
//     LLVMContext    &llvmContext = context->getLLVMContext();
//     Function       *TheFunction = Builder->GetInsertBlock()->getParent();

//     // PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
//     // Type *int8_type = Type::getInt8Ty(llvmContext);
//     // PointerType *void_ptr_type = PointerType::get(int8_type, 0);
//     // Type *int64_type = Type::getInt64Ty(llvmContext);
//     // Type *int32_type = Type::getInt32Ty(llvmContext);

//     Argument * head_ptr = ((const GpuRawContext *) context)->getStateVar(probe_head_param_id);
//     head_ptr->setName(opLabel + "_head_ptr");

//     ExpressionGeneratorVisitor exprGenerator(context, childState);
//     RawValue keyWrapper = probe_keyexpr->accept(exprGenerator);
//     Value * hash = GpuHashGroupByChained::hash(keyWrapper.value);

//     //current = head[hash(key)]
//     Value * current = Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash), context->getSizeOf(head_ptr->getType()->getPointerElementType()));
//     current->setName("current");

//     AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", current->getType());

//     Builder->CreateStore(current, mem_current);

//     //while (current != eoc){

//     BasicBlock *CondBB  = BasicBlock::Create(llvmContext, "chainFollowCond", TheFunction);
//     BasicBlock *ThenBB  = BasicBlock::Create(llvmContext, "chainFollow"    , TheFunction);
//     BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont"           , TheFunction);

//     Builder->CreateBr(CondBB);

//     Builder->SetInsertPoint(CondBB);

//     //check end of chain

//     Value * condition = Builder->CreateICmpNE(Builder->CreateLoad(mem_current), ConstantInt::get(current->getType(), ~((size_t) 0)));

//     Builder->CreateCondBr(condition, ThenBB, MergeBB);
//     Builder->SetInsertPoint(ThenBB);

//     //check match

//     std::vector<Value *> in_ptrs;
//     std::vector<Value *> in_vals;
//     for (size_t i = 0 ; i < in_param_ids.size() ; ++i) {
//         Argument * in_ptr = ((const GpuRawContext *) context)->getStateVar(in_param_ids[i]);
//         if (in_param_ids.size() != 1){
//             in_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
//         } else {
//             in_ptr->setName(opLabel + "_data_ptr");
//         }
//         // in_ptrs.push_back(in_ptr);
        
//         in_ptrs.push_back(Builder->CreateInBoundsGEP(in_ptr, Builder->CreateLoad(mem_current)));
//         in_vals.push_back(Builder->CreateAlignedLoad(in_ptrs.back(), context->getSizeOf(in_ptrs.back()->getType()->getPointerElementType())));
//     }

//     Value * next      = Builder->CreateExtractValue(in_vals[0], 0);
//     Value * build_key = Builder->CreateExtractValue(in_vals[0], 1);

//     Builder->CreateStore(next, mem_current);

//     Value * match_condition = Builder->CreateICmpEQ(keyWrapper.value, build_key); //FIXME replace with EQ expression to support multiple types!

//     BasicBlock *MatchThenBB  = BasicBlock::Create(llvmContext, "matchChainFollow"    , TheFunction);

//     Builder->CreateCondBr(match_condition, MatchThenBB, CondBB);

//     Builder->SetInsertPoint(MatchThenBB);

//     // Triggering parent
//     OperatorState* newState = new OperatorState(*this,childState.getBindings());
//     getParent()->consume(context, *newState);

//     Builder->CreateBr(CondBB);

//     // TheFunction->getBasicBlockList().push_back(MergeBB);
//     Builder->SetInsertPoint(MergeBB);

// }

// void GpuHashGroupByChained::generate_scan() {

// }

void GpuHashGroupByChained::open(RawPipeline * pip){
    int32_t * cnt  ;
    int32_t * first;
    std::vector<void *> next;

    gpu_run(cudaMalloc((void **) &cnt  , sizeof(int32_t  )                   ));
    gpu_run(cudaMalloc((void **) &first, sizeof(int32_t  ) * (1 << hash_bits)));

    for (const auto &w: packet_widths){
        next.emplace_back();
        gpu_run(cudaMalloc((void **) &(next.back()), (w/8) * 1024 * 1024 * 256)); //FIXME constant ==> max input size
    }

    gpu_run(cudaMemset(  cnt,  0,                    sizeof(int32_t)));
    gpu_run(cudaMemset(first, -1, (1 << hash_bits) * sizeof(int32_t)));

    pip->setStateVar<int32_t  *>(context, cnt_param_id , cnt);
    pip->setStateVar<int32_t  *>(context, head_param_id, first);

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i){
        pip->setStateVar<void *>(context, out_param_ids[i], next[i]);
    }
}

void GpuHashGroupByChained::close(RawPipeline * pip){
    gpu_run(cudaFree(pip->getStateVar<int32_t  *>(context, cnt_param_id)));
    gpu_run(cudaFree(pip->getStateVar<int32_t  *>(context, head_param_id)));

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i){
        gpu_run(cudaFree(pip->getStateVar<void *>(context, out_param_ids[i])));
    }
}