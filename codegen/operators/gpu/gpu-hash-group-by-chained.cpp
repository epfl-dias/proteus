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
#include "expressions/expressions-hasher.hpp"
#include "util/raw-memory-manager.hpp"

GpuHashGroupByChained::GpuHashGroupByChained(
            const std::vector<GpuAggrMatExpr>              &agg_exprs, 
            // const std::vector<size_t>                      &packet_widths,
            const std::vector<expressions::Expression *>    key_expr,
            RawOperator * const                             child,

            int                                             hash_bits,

            GpuRawContext *                                 context,
            size_t                                          maxInputSize,
            string                                          opLabel): 
                agg_exprs(agg_exprs),
                // packet_widths(packet_widths),
                key_expr(key_expr),
                hash_bits(hash_bits),
                UnaryRawOperator(child), 
                context(context),
                maxInputSize(maxInputSize),
                opLabel(opLabel){
}

void GpuHashGroupByChained::produce() {
    context->pushNewPipeline();

    buildHashTableFormat();
    
    getChild()->produce();

    context->popNewPipeline();

    probe_gen = context->getCurrentPipeline();
    generate_scan();
}

void GpuHashGroupByChained::consume(RawContext* const context, const OperatorState& childState) {
    generate_build(context, childState);
}

void GpuHashGroupByChained::buildHashTableFormat(){
    agg_exprs.emplace_back(new expressions::IntConstant(0), ~((size_t) 0),  0);
    
    size_t bitoffset = 0;
    for (const auto &key: key_expr){
        agg_exprs.emplace_back(key                , 0, bitoffset);

        const ExpressionType * out_type = key->getExpressionType();

        Type * llvm_type = out_type->getLLVMType(context->getLLVMContext());

        bitoffset += llvm_type->getPrimitiveSizeInBits();
    }

    std::sort(agg_exprs.begin(), agg_exprs.end(), [](const GpuAggrMatExpr& a, const GpuAggrMatExpr& b){
        return a.packet+1 < b.packet+1 || (a.packet == b.packet && a.bitoffset < b.bitoffset);
    });

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *t_head_ptr = PointerType::get(int32_type, /* address space */ 0);
    head_param_id = context->appendStateVar(t_head_ptr);

    size_t i = 0;
    size_t p = 0;

    while (i < agg_exprs.size()){
        // Type * t     = PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(), packet_widths[p]), /* address space */ 0);

        size_t bindex  = 0;
        size_t packind = 0;

        std::vector<Type *> body;
        while (i < agg_exprs.size() && agg_exprs[i].packet+1 == p){
            agg_exprs[i].packet++;
            if (agg_exprs[i].bitoffset != bindex){
                //insert space
                assert(agg_exprs[i].bitoffset > bindex);
                body.push_back(Type::getIntNTy(context->getLLVMContext(), (agg_exprs[i].bitoffset - bindex)));
                ++packind;
            }

            const ExpressionType * out_type = agg_exprs[i].expr->getExpressionType();

            Type * llvm_type = out_type->getLLVMType(context->getLLVMContext());

            body.push_back(llvm_type);
            bindex = agg_exprs[i].bitoffset + llvm_type->getPrimitiveSizeInBits();
            agg_exprs[i].packind = packind++;
            ++i;
        }
        // assert(packet_widths[p] >= bindex);

        if (bindex & (bindex - 1)) {
            size_t v = bindex - 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v++;
            body.push_back(Type::getIntNTy(context->getLLVMContext(), (v - bindex)));
        // if (packet_widths[p] > bindex) {
        //     body.push_back(Type::getIntNTy(context->getLLVMContext(), (packet_widths[p] - bindex)));
        // }
            bindex = v;
        }
        packet_widths.push_back(bindex);
        Type * t     = StructType::create(body, opLabel + "_struct_" + std::to_string(p), true);
        Type * t_ptr = PointerType::get(t, /* address space */ 0);
        ptr_types.push_back(t_ptr);

        out_param_ids.push_back(context->appendStateVar(t_ptr));//, true, false));

        ++p;
    }
    assert(i == agg_exprs.size());

    agg_exprs.erase(agg_exprs.begin()); //erase dummy entry for next

    // Type * t     = PointerType::get(((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext()), /* address space */ 0);
    // out_param_id = context->appendStateVar(t    );//, true, false);

    Type * t_cnt = PointerType::get(int32_type, /* address space */ 0);
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

    hash = Builder->CreateURem(hash, ConstantInt::get(hash->getType(), (size_t(1) << hash_bits)));
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


    // expressions::Expression * kexpr;
    // if (key_expr.size() == 1) {
    //     kexpr = key_expr[0];
    // } else {
    //     auto attrs = new list<expressions::AttributeConstruction>;
    //     for (const auto &k: key_expr){
    //         attrs->emplace_back(k->getRegisteredAttrName(), k);
    //     }
    //     kexpr = new expressions::RecordConstruction(*attrs);
    // }

    // ExpressionHasherVisitor exphasher{context, childState};
    // Value * hash = kexpr->accept(exphasher).value;
    Value * hash = GpuHashGroupByChained::hash(key_expr, context, childState);

    Value * eochain   = ConstantInt::get((IntegerType *) head_ptr->getType()->getPointerElementType(), ~((size_t) 0));
    //current = head[hash(key)]
    // Value * current = Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash), context->getSizeOf(head_ptr->getType()->getPointerElementType()));
    Value * head_w_hash_ptr = Builder->CreateInBoundsGEP(head_ptr, hash);
    Value * current = Builder->CreateExtractValue(Builder->CreateAtomicCmpXchg(head_w_hash_ptr,
                                                eochain,
                                                eochain,
                                                llvm::AtomicOrdering::Monotonic,
                                                llvm::AtomicOrdering::Monotonic), 0);
    
    current->setName("current");

    AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", current->getType());
    Builder->CreateStore(current, mem_current);

    AllocaInst *mem_idx     = context->CreateEntryBlockAlloca(TheFunction, "mem_idx", out_cnt->getType()->getPointerElementType());
    Builder->CreateStore(UndefValue::get(out_cnt->getType()->getPointerElementType()), mem_idx);

    AllocaInst *mem_written = context->CreateEntryBlockAlloca(TheFunction, "mem_written", v_false->getType());
    Builder->CreateStore(v_false, mem_written);

    // std::vector<Value *> out_ptrs;
    std::vector<Value *> out_vals;

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);
        out_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
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

    BasicBlock *ThenBB  = BasicBlock::Create(llvmContext, "chainFollow"    , TheFunction);
    BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont"           , TheFunction);

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
    Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg,
        Builder->CreateBitCast(Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]), old_cnt), PointerType::getInt32PtrTy(llvmContext)),
        eochain,
        llvm::AtomicOrdering::Monotonic);

    for (size_t i = 1 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);

        Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
        Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i]/8, true);
    }

            // written = true;
    Builder->CreateStore(v_true, mem_written);

    ((GpuRawContext *) context)->createMembar_gl();

            // current = atomicCAS(&(first[bucket]), -1, idx);
    Value * old_current = Builder->CreateAtomicCmpXchg(head_w_hash_ptr,
                                                eochain,
                                                old_cnt,
                                                llvm::AtomicOrdering::Monotonic,
                                                llvm::AtomicOrdering::Monotonic);

    Builder->CreateStore(Builder->CreateExtractValue(old_current, 0), mem_current);
    Value * suc = Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

    Builder->CreateCondBr(suc, MergeBB, InitMergeBB);
    // }
    // Builder->CreateBr(InitMergeBB);
    Builder->SetInsertPoint(InitMergeBB);


    // if (current != ((uint32_t) -1)){
    //     while (true) {

    Value * chain_cond = Builder->CreateICmpNE(Builder->CreateLoad(mem_current), eochain);

    Builder->CreateCondBr(chain_cond, ThenBB, MergeBB);

    Builder->SetInsertPoint(ThenBB);
    current = Builder->CreateLoad(mem_current);

    ((GpuRawContext *) context)->createMembar_gl();

    Value * next_bucket_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[1]), current);
    Value * next_bucket     = Builder->CreateAlignedLoad(next_bucket_ptr, packet_widths[1]/8, true, "next_bucket");

    ((GpuRawContext *) context)->createMembar_gl();

    Value * next = Builder->CreateExtractValue(Builder->CreateAtomicCmpXchg(Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), std::vector<Value *>{current, context->createInt32(0)}),
                                                eochain,
                                                eochain,
                                                llvm::AtomicOrdering::Monotonic,
                                                llvm::AtomicOrdering::Monotonic), 0);

    // next_bucket_next->setName("next_bucket_next");
    //             // int32_t   next_bucket = next[current].next;
    // Value * next            = Builder->CreateExtractValue(next_bucket, 0);
    // Value * key             = Builder->CreateExtractValue(next_bucket, 1);

    BasicBlock *BucketFoundBB = BasicBlock::Create(llvmContext, "BucketFound", TheFunction);
    BasicBlock *ContFollowBB  = BasicBlock::Create(llvmContext, "ContFollow" , TheFunction);


    Value * bucket_cond = v_true;
    for (size_t i = 0 ; i < key_expr.size() ; ++i){
        ExpressionGeneratorVisitor exprGenerator(context, childState);
        RawValue keyWrapper = key_expr[i]->accept(exprGenerator);

        Value  * key        = Builder->CreateExtractValue(next_bucket, i);
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

    // Value * str = UndefValue::get(((const GpuRawContext *) context)->getStateVar(out_param_ids[0])->getType()->getPointerElementType());
    // str = Builder->CreateInsertValue(str, Builder->CreateLoad(mem_idx), 0);
    Value * inv_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), std::vector<Value *>{Builder->CreateLoad(mem_idx), context->createInt32(0)});

    Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, 
                                                inv_ptr,
                                                Builder->CreateLoad(mem_idx),
                                                llvm::AtomicOrdering::Monotonic);
    // Builder->CreateAlignedStore(str, , packet_widths[0]/8);


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
    Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg,
        Builder->CreateBitCast(Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]), old_cnt), PointerType::getInt32PtrTy(llvmContext)),
        eochain,
        llvm::AtomicOrdering::Monotonic);

    for (size_t i = 1 ; i < out_param_ids.size() ; ++i) {
        Value * out_ptr = context->getStateVar(out_param_ids[i]);

        Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
        Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i]/8, true);
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
    ((GpuRawContext *) context)->createMembar_gl();
                    // current = new_next;
    Builder->CreateStore(Builder->CreateExtractValue(new_next, 0), mem_current);
                    // if (new_next == ((uint32_t) -1)) 
    // Value * valid_insert = Builder->CreateICmpEQ(new_next, eochain);
    Builder->CreateCondBr(Builder->CreateExtractValue(new_next, 1), MergeBB, ThenBB);

    Builder->SetInsertPoint(MergeBB);

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
}

                                // void GpuHashGroupByChained::generate_build(RawContext* const context, const OperatorState& childState) {
                                //     IRBuilder<>    *Builder     = context->getBuilder();
                                //     LLVMContext    &llvmContext = context->getLLVMContext();
                                //     Function       *TheFunction = Builder->GetInsertBlock()->getParent();

                                //     // PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
                                //     // Type *int8_type = Type::getInt8Ty(llvmContext);
                                //     // PointerType *void_ptr_type = PointerType::get(int8_type, 0);
                                //     // Type *int64_type = Type::getInt64Ty(llvmContext);
                                //     // Type *int32_type = Type::getInt32Ty(llvmContext);
                                //     Value * v_true  = ConstantInt::getTrue (llvmContext);
                                //     Value * v_false = ConstantInt::getFalse(llvmContext);

                                //     Value * out_cnt = ((const GpuRawContext *) context)->getStateVar(cnt_param_id);
                                //     out_cnt ->setName(opLabel + "_cnt_ptr");

                                //     Value * head_ptr = ((const GpuRawContext *) context)->getStateVar(head_param_id);
                                //     head_ptr->setName(opLabel + "_head_ptr");


                                //     // expressions::Expression * kexpr;
                                //     // if (key_expr.size() == 1) {
                                //     //     kexpr = key_expr[0];
                                //     // } else {
                                //     //     auto attrs = new list<expressions::AttributeConstruction>;
                                //     //     for (const auto &k: key_expr){
                                //     //         attrs->emplace_back(k->getRegisteredAttrName(), k);
                                //     //     }
                                //     //     kexpr = new expressions::RecordConstruction(*attrs);
                                //     // }

                                //     // ExpressionHasherVisitor exphasher{context, childState};
                                //     // Value * hash = kexpr->accept(exphasher).value;
                                //     Value * hash = GpuHashGroupByChained::hash(key_expr, context, childState);

                                //     Value * eochain   = ConstantInt::get((IntegerType *) head_ptr->getType()->getPointerElementType(), ~((size_t) 0));
                                //     //current = head[hash(key)]
                                //     // Value * current = Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash), context->getSizeOf(head_ptr->getType()->getPointerElementType()));
                                //     Value * head_w_hash_ptr = Builder->CreateInBoundsGEP(head_ptr, hash);
                                //     // Value * current = Builder->CreateExtractValue(Builder->CreateAtomicCmpXchg(head_w_hash_ptr,
                                //     //                                             eochain,
                                //     //                                             eochain,
                                //     //                                             llvm::AtomicOrdering::Monotonic,
                                //     //                                             llvm::AtomicOrdering::Monotonic), 0);
                                    
                                //     // current->setName("current");

                                //     // AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", current->getType());
                                //     // Builder->CreateStore(current, mem_current);

                                //     BasicBlock * saveBB = Builder->GetInsertBlock();

                                //     Builder->SetInsertPoint(context->getCurrentEntryBlock());
                                //     AllocaInst *mem_idx     = context->CreateEntryBlockAlloca(TheFunction, "mem_idx", eochain->getType());
                                //     Builder->CreateStore(eochain, mem_idx);

                                //     Builder->SetInsertPoint(saveBB);

                                //     // AllocaInst *mem_written = context->CreateEntryBlockAlloca(TheFunction, "mem_written", v_false->getType());
                                //     // Builder->CreateStore(v_false, mem_written);

                                //     // std::vector<Value *> out_ptrs;
                                //     std::vector<Value *> out_vals;

                                //     for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
                                //         Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);
                                //         out_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
                                //         // out_ptrs.push_back(out_ptr);

                                //         // out_ptr->addAttr(Attribute::getWithAlignment(llvmContext, context->getSizeOf(out_ptr)));

                                //         // out_ptrs.push_back(Builder->CreateInBoundsGEP(out_ptr, old_cnt));
                                //         out_vals.push_back(UndefValue::get(out_ptr->getType()->getPointerElementType()));
                                //     }

                                //     out_vals[0] = Builder->CreateInsertValue(out_vals[0], eochain, 0);

                                //     for (const GpuAggrMatExpr &mexpr: agg_exprs){
                                //         ExpressionGeneratorVisitor exprGenerator(context, childState);
                                //         RawValue valWrapper = mexpr.expr->accept(exprGenerator);
                                        
                                //         out_vals[mexpr.packet] = Builder->CreateInsertValue(out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
                                //     }


                                //     //if mem_idx == -1:
                                //     Value * alloc = Builder->CreateICmpEQ(Builder->CreateLoad(mem_idx), eochain);

                                //     BasicBlock *allocBB      = BasicBlock::Create(llvmContext, "alloc"     , TheFunction);
                                //     BasicBlock *allocAfterBB = BasicBlock::Create(llvmContext, "allocAfter", TheFunction);
                                //     Builder->CreateCondBr(alloc, allocBB, allocAfterBB);


                                //     BasicBlock *deallocBB      = BasicBlock::Create(llvmContext, "dealloc"     , TheFunction);
                                //     BasicBlock *deallocAfterBB = BasicBlock::Create(llvmContext, "deallocAfter", TheFunction);
                                //     Builder->SetInsertPoint(context->getEndingBlock());
                                //     Value * dealloc = Builder->CreateICmpNE(Builder->CreateLoad(mem_idx), eochain);
                                //     Builder->CreateCondBr(dealloc, deallocBB, deallocAfterBB);

                                //     Builder->SetInsertPoint(deallocBB);

                                //     Value * inv_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), std::vector<Value *>{Builder->CreateLoad(mem_idx), context->createInt32(0)});

                                //     Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, 
                                //                                                 inv_ptr,
                                //                                                 Builder->CreateLoad(mem_idx),
                                //                                                 llvm::AtomicOrdering::Monotonic);

                                //     Builder->CreateBr(deallocAfterBB);
                                //     context->setEndingBlock(deallocAfterBB);


                                //     //index
                                //     Builder->SetInsertPoint(allocBB);
                                //     Value * alloced_idx  = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
                                //                                                 out_cnt,
                                //                                                 ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
                                //                                                 llvm::AtomicOrdering::Monotonic);
                                //     Builder->CreateStore(alloced_idx, mem_idx);

                                //             // next[idx].sum  = val;
                                //             // next[idx].key  = key;
                                //             // next[idx].next =  -1;
                                //     // Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg,
                                //     //     Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]), std::vector<Value *>{alloced_idx, context->createInt32(0)}),
                                //     //     eochain,
                                //     //     llvm::AtomicOrdering::Monotonic);

                                //     for (size_t i = 1 ; i < out_param_ids.size() ; ++i) {
                                //         Value * out_ptr = ((const GpuRawContext *) context)->getStateVar(out_param_ids[i]);

                                //         Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, alloced_idx);
                                //         Builder->CreateStore(out_vals[i], out_ptr_i, true); //Aligned: , packet_widths[i]/8
                                //     }

                                //             // written = true;
                                //     // Builder->CreateStore(v_true, mem_written);

                                //     ((GpuRawContext *) context)->createMembar_gl();

                                //     Builder->CreateBr(allocAfterBB);
                                //     Builder->SetInsertPoint(allocAfterBB);
                                    
                                //     Value * old_cnt = Builder->CreateLoad(mem_idx);
                                //     old_cnt->setName("index");



                                //     // BasicBlock *InitCondBB  = BasicBlock::Create(llvmContext, "setHeadCond", TheFunction);
                                //     BasicBlock *InitThenBB  = BasicBlock::Create(llvmContext, "setHead", TheFunction);
                                //     BasicBlock *InitMergeBB = BasicBlock::Create(llvmContext, "cont"   , TheFunction);

                                //     BasicBlock *ThenBB  = BasicBlock::Create(llvmContext, "chainFollow"    , TheFunction);
                                //     BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont"           , TheFunction);
                                //     BasicBlock *AfterMergeBB = BasicBlock::Create(llvmContext, "groupbyend"     , TheFunction);

                                //     // Value * init_cond = Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

                                //     // if (current == ((uint32_t) -1)){
                                //     // Builder->CreateCondBr(init_cond, InitThenBB, InitMergeBB);

                                //     // Builder->SetInsertPoint(InitThenBB);


                                //             // current = atomicCAS(&(first[bucket]), -1, idx);
                                //     Value * old_current = Builder->CreateAtomicCmpXchg(head_w_hash_ptr,
                                //                                                 eochain,
                                //                                                 old_cnt,
                                //                                                 llvm::AtomicOrdering::Monotonic,
                                //                                                 llvm::AtomicOrdering::Monotonic);

                                //     AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current", Builder->CreateExtractValue(old_current, 0)->getType());
                                //     Builder->CreateStore(Builder->CreateExtractValue(old_current, 0), mem_current);

                                //     Value * suc = Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

                                //     Builder->CreateCondBr(suc, MergeBB, InitMergeBB);
                                //     // }
                                //     // Builder->CreateBr(InitMergeBB);
                                //     Builder->SetInsertPoint(InitMergeBB);

                                //     BasicBlock *BucketFoundBB = BasicBlock::Create(llvmContext, "BucketFound", TheFunction);
                                //     BasicBlock *ContFollowBB  = BasicBlock::Create(llvmContext, "ContFollow" , TheFunction);

                                //     Value * next_bucket_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[1]), Builder->CreateLoad(mem_current));
                                //     Value * next_bucket     = Builder->CreateLoad(next_bucket_ptr, true, "next_bucket"); //Aligned: , packet_widths[1]/8

                                //     Value * bucket_cond = v_true;
                                //     for (size_t i = 0 ; i < key_expr.size() ; ++i){
                                //         ExpressionGeneratorVisitor exprGenerator(context, childState);
                                //         RawValue keyWrapper = key_expr[i]->accept(exprGenerator);

                                //         Value  * key        = Builder->CreateExtractValue(next_bucket, i);
                                //         Value  * key_comp   = Builder->CreateICmpEQ(key, keyWrapper.value);

                                //         bucket_cond         = Builder->CreateAnd(bucket_cond, key_comp);
                                //     }
                                //                 // if (next[current].key == key) {
                                //     Builder->CreateCondBr(bucket_cond, BucketFoundBB, ContFollowBB);

                                //     Builder->SetInsertPoint(BucketFoundBB);

                                //                     // atomicAdd(&(next[current].sum), val);
                                //                                                                                                             // for (size_t i = 0 ; i < agg_exprs.size() ; ++i){
                                //                                                                                                             //     if (agg_exprs[i].is_aggregation()){
                                //                                                                                                             //         gpu::Monoid * gm = gpu::Monoid::get(agg_exprs[i].m);
                                //                                                                                                             //         std::vector<Value *> tmp{Builder->CreateLoad(mem_current), context->createInt32(agg_exprs[i].packind)};

                                //                                                                                                             //         Value * aggr     = Builder->CreateExtractValue(out_vals[agg_exprs[i].packet], agg_exprs[i].packind);

                                //                                                                                                             //         Value * gl_accum = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[agg_exprs[i].packet]), tmp);

                                //                                                                                                             //         gm->createAtomicUpdate(context, gl_accum, aggr, llvm::AtomicOrdering::Monotonic);
                                //                                                                                                             //     }
                                //                                                                                                             // }


                                //     // Value * str = UndefValue::get(((const GpuRawContext *) context)->getStateVar(out_param_ids[0])->getType()->getPointerElementType());
                                //     // str = Builder->CreateInsertValue(str, Builder->CreateLoad(mem_idx), 0);
                                //     // Value * inv_ptr = Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), std::vector<Value *>{old_cnt, context->createInt32(0)});

                                //     // Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, 
                                //     //                                             inv_ptr,
                                //     //                                             old_cnt,
                                //     //                                             llvm::AtomicOrdering::Monotonic);
                                //     // Builder->CreateAlignedStore(str, , packet_widths[0]/8);


                                //     Builder->CreateBr(AfterMergeBB);

                                //     Builder->SetInsertPoint(ContFollowBB);
                                //     Value * next = Builder->CreateAtomicCmpXchg(Builder->CreateInBoundsGEP(((const GpuRawContext *) context)->getStateVar(out_param_ids[0]), std::vector<Value *>{Builder->CreateLoad(mem_current), context->createInt32(0)}),
                                //                                                 eochain,
                                //                                                 old_cnt,
                                //                                                 llvm::AtomicOrdering::Monotonic,
                                //                                                 llvm::AtomicOrdering::Monotonic);

                                //     Builder->CreateStore(Builder->CreateExtractValue(next, 0), mem_current);

                                //     // if (current != ((uint32_t) -1)){
                                //     //     while (true) {

                                //     Value * chain_cond = Builder->CreateICmpNE(Builder->CreateExtractValue(next, 0), eochain);
                                //     // Value * chain_cond2 = Builder->CreateICmpNE(Builder->CreateLoad(mem_current), old_cnt);

                                //     Builder->CreateCondBr(chain_cond, InitMergeBB, MergeBB);

                                //     Builder->SetInsertPoint(MergeBB);

                                //     Builder->CreateStore(eochain, mem_idx);

                                //     Builder->CreateBr(AfterMergeBB);
                                //     Builder->SetInsertPoint(AfterMergeBB);

                                //     ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
                                //     ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
                                // }

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

void GpuHashGroupByChained::generate_scan() {
    //Prepare
    LLVMContext& llvmContext = context->getLLVMContext();

    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
    Type* int64Type   = Type::getInt64Ty(llvmContext);
    Type* int32_type  = Type::getInt32Ty(llvmContext);

    //Container for the variable bindings
    map<RecordAttribute, RawValueMemory> variableBindings;

    Type * t_cnt = PointerType::get(int32_type, /* address space */ 0);
    size_t cnt_ptr_param = context->appendParameter(t_cnt, true, true);

    vector<size_t> out_param_ids_scan;
    for (const auto &p: ptr_types) {
        out_param_ids_scan.push_back(context->appendParameter(p, true, true));
    }

    context->setGlobalFunction();

    IRBuilder<>* Builder = context->getBuilder();
    Function *F = Builder->GetInsertBlock()->getParent();

    // // Create the "AFTER LOOP" block and insert it.
    // BasicBlock *releaseBB = BasicBlock::Create(llvmContext, "releaseIf", F);
    // BasicBlock *rlAfterBB = BasicBlock::Create(llvmContext, "releaseEnd" , F);

    Value * tId       = context->threadId();
    Value * is_leader = Builder->CreateICmpEQ(tId, ConstantInt::get(tId->getType(), 0));

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
    context->setEndingBlock(AfterBB);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    // Builder->CreateBr      (CondBB);

    
    std::string relName = agg_exprs[0].expr->getRegisteredRelName();
    Plugin* pg = RawCatalog::getInstance().getPlugin(relName);
    
    AllocaInst * mem_itemCtr = context->CreateEntryBlockAlloca(F, "i_ptr", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(
                Builder->CreateIntCast(context->threadId(),
                                        pg->getOIDType()->getLLVMType(llvmContext),
                                        false),
                mem_itemCtr);

    RecordAttribute tupleCnt{relName, "activeCnt", pg->getOIDType()}; //FIXME: OID type for blocks ?
    Value * cnt = Builder->CreateLoad(context->getArgument(cnt_ptr_param), "cnt");
    AllocaInst * mem_cnt = context->CreateEntryBlockAlloca(F, "cnt_mem", cnt->getType());
    Builder->CreateStore(cnt, mem_cnt);

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem = mem_cnt;
    mem_cntWrapper.isNull = context->createFalse();
    variableBindings[tupleCnt] = mem_cntWrapper;

    // Function * f = context->getFunction("devprinti64");
    // Builder->CreateCall(f, std::vector<Value *>{cnt});

    // Builder->CreateBr      (CondBB);
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
    RecordAttribute tupleIdentifier{relName, activeLoop, pg->getOIDType()};

    RawValueMemory mem_posWrapper;
    mem_posWrapper.mem = mem_itemCtr;
    mem_posWrapper.isNull = context->createFalse();
    variableBindings[tupleIdentifier] = mem_posWrapper;

    //Actual Work (Loop through attributes etc.)

    vector<Value *> in_vals;
    for (size_t i = 0 ; i < out_param_ids_scan.size() ; ++i) {
        Value * out_ptr = context->getArgument(out_param_ids_scan[i]);

        Value * out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, lhs);
        Value * val = Builder->CreateAlignedLoad(out_ptr_i, packet_widths[i]/8);

        in_vals.push_back(val);
    }

    Value * next = Builder->CreateExtractValue(in_vals[0], 0);

    BasicBlock *groupProcessBB = BasicBlock::Create(llvmContext, "releaseIf", F);
    
    Value * isGroup = Builder->CreateICmpNE(next, lhs);
    Builder->CreateCondBr(isGroup, groupProcessBB, IncBB);

    Builder->SetInsertPoint(groupProcessBB);

    for (const GpuAggrMatExpr &mexpr: agg_exprs){

        Value * v = Builder->CreateExtractValue(in_vals[mexpr.packet], mexpr.packind);
        AllocaInst * v_mem = context->CreateEntryBlockAlloca(mexpr.expr->getRegisteredAttrName(), v->getType());
        Builder->CreateStore(v, v_mem);

        RawValueMemory val_mem;
        val_mem.mem = v_mem;
        val_mem.isNull = context->createFalse();
        
        variableBindings[mexpr.expr->getRegisteredAs()] = val_mem;
    }

    //Triggering parent
    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    // Insert an explicit fall through from the current (body) block to IncBB.
    Builder->CreateBr(IncBB);

    // Start insertion in IncBB.
    Builder->SetInsertPoint(IncBB);

    //Increment and store back
    Value* val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);
    
    Value * inc = Builder->CreateIntCast(context->threadNum(), val_curr_itemCtr->getType(), false);

    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, inc);
    Builder->CreateStore(val_new_itemCtr, mem_itemCtr);

    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(CondBB);

    //  Finish up with end (the AfterLoop)
    //  Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(AfterBB);

    // ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    // ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
}

void GpuHashGroupByChained::open(RawPipeline * pip){
    int32_t * cnt   = (int32_t *) RawMemoryManager::mallocGpu(sizeof(int32_t  )                   );
    int32_t * first = (int32_t *) RawMemoryManager::mallocGpu(sizeof(int32_t  ) * (1 << hash_bits));
    std::vector<void *> next;

    for (const auto &w: packet_widths){
        next.emplace_back(RawMemoryManager::mallocGpu((w/8) * maxInputSize));
    }

    gpu_run(cudaMemset(  cnt,  0,                    sizeof(int32_t)));
    gpu_run(cudaMemset(first, -1, (1 << hash_bits) * sizeof(int32_t)));
    // gpu_run(cudaMemset(next[0], -1, (packet_widths[0]/8) * maxInputSize));

    pip->setStateVar<int32_t  *>(cnt_param_id , cnt);
    pip->setStateVar<int32_t  *>(head_param_id, first);

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i){
        pip->setStateVar<void *>(out_param_ids[i], next[i]);
    }

    // std::cout << cnt << " " << get_device(cnt) << std::endl;
    // std::cout << first << " " << get_device(first) << std::endl;
    // std::cout << next[0] << " " << get_device(next[0]) << std::endl;
}

struct entry{
    int32_t index;
    // int32_t key0;
    // int32_t key1;
    // int32_t gb;
};

void GpuHashGroupByChained::close(RawPipeline * pip){
    int32_t * cnt_ptr = pip->getStateVar<int32_t  *>(cnt_param_id);
    // entry * h_next;
    // int32_t * h_first;
    int32_t cnt;
    // std::cout << packet_widths[0]/8 << " " << sizeof(entry) << std::endl;
    // assert(packet_widths[0]/8 == sizeof(entry));
    // size_t size = (packet_widths[0]/8) * maxInputSize;
    // gpu_run(cudaMallocHost((void **) &h_next , size));
    // gpu_run(cudaMallocHost((void **) &h_first, sizeof(int32_t  ) * (1 << hash_bits)));
    gpu_run(cudaMemcpy(&cnt  , cnt_ptr, sizeof(int32_t), cudaMemcpyDefault));
    std::cout << "---------------------------> " << cnt << " " << maxInputSize << std::endl;
    // gpu_run(cudaMemcpy(h_next, pip->getStateVar<void *>(out_param_ids[0]), cnt * (packet_widths[0]/8), cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(h_first, pip->getStateVar<void *>(head_param_id), sizeof(int32_t  ) * (1 << hash_bits), cudaMemcpyDefault));
    // for (int32_t i = 0 ; i < cnt ; ++i){
    //     if (h_next[i].index != i){
    //         std::cout << i << " " << h_next[i].index << std::endl;//" " << h_next[i].key0 << " " << h_next[i].key1 << std::endl;
    //     }
    // }
    // std::cout << "---" << std::endl;
    // for (int32_t i = 0 ; i < (1 << hash_bits) ; ++i){
    //     if (h_first[i] != -1){
    //         std::cout << i << " " << h_first[i] << std::endl;
    //     }
    // }
    // std::cout << "---+++" << std::endl;
    // for (int32_t i = 0 ; i < (1 << hash_bits) ; ++i){
    //     if (h_first[i] != -1){
    //         std::cout << i << " " << h_next[h_first[i]].index << std::endl;
    //     }
    // }
    // std::cout << "---------------------------> " << cnt << std::endl;


    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    execution_conf ec = pip->getExecConfiguration();
    size_t grid_size  = ec.gridSize();
    
    // void   ** buffs = pip->getStateVar<void   **>(buffVar_id[0]);
    // int32_t * cnts  = pip->getStateVar<int32_t *>(cntVar_id    );

    RawPipeline * probe_pip = probe_gen->getPipeline(pip->getGroup());
    probe_pip->open();

    std::vector<void *> args;
    args.push_back(pip->getStateVar<int32_t  *>(cnt_param_id));
    for (const auto &params: out_param_ids){
        args.push_back(pip->getStateVar<void *>(params));
    }

    std::vector<void **> kp;
    for (size_t i = 0 ; i < args.size() ; ++i) {
        kp.push_back(args.data() + i);
    }
    kp.push_back((void **) probe_pip->getState());
    
    launch_kernel((CUfunction) probe_gen->getKernel(), (void **) kp.data(), strm);
    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy    (strm));


    probe_pip->close();






    RawMemoryManager::freeGpu(pip->getStateVar<int32_t  *>(cnt_param_id ));
    RawMemoryManager::freeGpu(pip->getStateVar<int32_t  *>(head_param_id));

    for (size_t i = 0 ; i < out_param_ids.size() ; ++i){
         RawMemoryManager::freeGpu(pip->getStateVar<void *>(out_param_ids[i]));
    }
}