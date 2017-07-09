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

#include "operators/gpu/gmonoids.hpp"
#include "llvm/IR/InlineAsm.h"
#include "util/gpu/gpu-intrinsics.hpp"

namespace gpu {

void Monoid::createUpdate(RawContext* const context, 
                                    AllocaInst * val_accumulating,
                                    Value * val_in){

    context->getBuilder()->CreateStore(
                                    create(context, val_accumulating, val_in),
                                    val_accumulating
                                );
}

Value * MaxMonoid::create(RawContext* const context, 
                            Value * val_accumulating,
                            Value * val_in){

    IRBuilder<>* Builder = context->getBuilder();

    Value* maxCondition;
    
    if        (val_accumulating->getType()->isIntegerTy()       ){
        maxCondition = Builder->CreateICmpSGT(val_in, val_accumulating);
    } else if (val_accumulating->getType()->isFloatingPointTy() ){
        maxCondition = Builder->CreateFCmpOGT(val_in, val_accumulating);
    } else {
        string error_msg = string("[MaxMonoid: ] Max operates on numerics");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    return Builder->CreateSelect(maxCondition, val_in, val_accumulating);
}

void MaxMonoid::createUpdate(RawContext* const context,
                                AllocaInst * mem_accumulating,
                                Value * val_in) {

    IRBuilder<>* Builder = context->getBuilder();

    BasicBlock *curBlock = Builder->GetInsertBlock();

    Function *TheFunction = curBlock->getParent();
    BasicBlock *endBlock = BasicBlock::Create(context->getLLVMContext(), "maxEnd", TheFunction);

    if (curBlock == context->getEndingBlock()) context->setEndingBlock(endBlock);

    /**
     * if(curr > max) max = curr;
     */
    BasicBlock* ifGtMaxBlock;
    context->CreateIfBlock(context->getGlobalFunction(), "maxCond", &ifGtMaxBlock);
    Value* val_accumulating = Builder->CreateLoad(mem_accumulating);
    
    Value* maxCondition;
    
    if        (val_accumulating->getType()->isIntegerTy()       ){
        maxCondition = Builder->CreateICmpSGT(val_in, val_accumulating);
    } else if (val_accumulating->getType()->isFloatingPointTy() ){
        maxCondition = Builder->CreateFCmpOGT(val_in, val_accumulating);
    } else {
        string error_msg = string("[MaxMonoid: ] Max accumulator operates on numerics");
        LOG(ERROR)<< error_msg;
        throw runtime_error(error_msg);
    }

    Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

    Builder->SetInsertPoint(ifGtMaxBlock);
    Builder->CreateStore(val_in, mem_accumulating);
    Builder->CreateBr(endBlock);

    //Back to 'normal' flow
    Builder->SetInsertPoint(endBlock);
}

void MaxMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){

    context->getBuilder()->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Max, 
                                            accumulator_ptr,
                                            val_in,
                                            order);

}

Value * SumMonoid::create(RawContext* const context, 
                            Value * val_accumulating, 
                            Value * val_in) {
    return context->getBuilder()->CreateAdd(val_in, val_accumulating);
}

void SumMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){

    context->getBuilder()->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
                                            accumulator_ptr,
                                            val_in,
                                            order);

}

Value * LogOrMonoid::create(RawContext* const context, 
                            Value * val_accumulating, 
                            Value * val_in) {
    return context->getBuilder()->CreateOr(val_in, val_accumulating);
}

void LogOrMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){
    //no atomics for i1
    //FIXME: check if there is a better way to do this + whether this is correct
    IRBuilder<>* Builder = context->getBuilder();

    BasicBlock *curBlock = Builder->GetInsertBlock();

    Function *TheFunction = curBlock->getParent();
    BasicBlock *endBlock = BasicBlock::Create(context->getLLVMContext(),
                                                "atomOrEnd",
                                                TheFunction);

    if (curBlock == context->getEndingBlock())context->setEndingBlock(endBlock);

    /**
     * if(val_in) *accumulator_ptr = true;
     */
    BasicBlock* ifBlock;
    context->CreateIfBlock(context->getGlobalFunction(),
                            "atomOrCnd",
                            &ifBlock);
    
    Builder->CreateCondBr(val_in, ifBlock, endBlock);

    Value * true_const = ConstantInt::getTrue(val_in->getType());

    Builder->SetInsertPoint(ifBlock);
    Builder->CreateStore(true_const, accumulator_ptr);
    Builder->CreateBr(endBlock);

    //Back to 'normal' flow
    Builder->SetInsertPoint(endBlock);
}

Value * LogAndMonoid::create(RawContext* const context, 
                            Value * val_accumulating, 
                            Value * val_in) {
    return context->getBuilder()->CreateAnd(val_in, val_accumulating);
}

void LogAndMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){
    //no atomics for i1
    //FIXME: check if there is a better way to do this + whether this is correct
    IRBuilder<>* Builder = context->getBuilder();

    BasicBlock *curBlock = Builder->GetInsertBlock();

    Function *TheFunction = curBlock->getParent();
    BasicBlock *endBlock = BasicBlock::Create(context->getLLVMContext(),
                                                "atomLAndEnd",
                                                TheFunction);

    if (curBlock == context->getEndingBlock())context->setEndingBlock(endBlock);

    /**
     * if(!val_in) *accumulator_ptr = false;
     */
    BasicBlock* ifBlock;
    context->CreateIfBlock(context->getGlobalFunction(),
                            "atomlAndCond",
                            &ifBlock);
    
    Builder->CreateCondBr(val_in, endBlock, ifBlock);

    Value * false_const = ConstantInt::getFalse(val_in->getType());

    Builder->SetInsertPoint(ifBlock);
    Builder->CreateStore(false_const, accumulator_ptr);
    Builder->CreateBr(endBlock);

    //Back to 'normal' flow
    Builder->SetInsertPoint(endBlock);
}

Value * BitOrMonoid::create(RawContext* const context, 
                            Value * val_accumulating, 
                            Value * val_in) {
    return context->getBuilder()->CreateOr(val_in, val_accumulating);
}

Value * BitAndMonoid::create(RawContext* const context, 
                            Value * val_accumulating, 
                            Value * val_in) {
    return context->getBuilder()->CreateAnd(val_in, val_accumulating);
}

void BitOrMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){

    context->getBuilder()->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Or, 
                                            accumulator_ptr,
                                            val_in,
                                            order);

}


void BitAndMonoid::createAtomicUpdate(RawContext* const context,
                                    Value * accumulator_ptr,
                                    Value * val_in,
                                    llvm::AtomicOrdering order){

    context->getBuilder()->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::And, 
                                            accumulator_ptr,
                                            val_in,
                                            order);

}

Monoid * Monoid::get(::Monoid m){
    switch (m){
        case MAX:
            return new MaxMonoid;
        case SUM:
            return new SumMonoid;
        case OR:
            return new LogOrMonoid;
        case AND:
            return new LogAndMonoid;
        default:
            string error_msg = string("[gpu::Monoids: ] Unimplemented monoid for gpu");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
    }
}

Value * Monoid::createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in){
    for (int i = 16 ; i > 0 ; i >>= 1){
        //NOTE: only whole (32 threads) warps are supported!
        Value * shfl_res = gpu_intrinsic::shfl_bfly((GpuRawContext * const) context, val_in, i);
        shfl_res->setName("shfl_res_" + std::to_string(i));
        
        val_in      = create(context, val_in, shfl_res);
    }

    return val_in;
}

Value * LogOrMonoid::createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in){
    return gpu_intrinsic::any((GpuRawContext * const) context, val_in);
}

Value * LogAndMonoid::createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in){
    return gpu_intrinsic::all((GpuRawContext * const) context, val_in);
}

} //namemspace gpu

namespace std{
    string to_string(const gpu::Monoid &m){
        return m.to_string();
    }
}