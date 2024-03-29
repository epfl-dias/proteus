/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#include "gmonoids.hpp"

#include "lib/util/gpu/gpu-intrinsics.hpp"
#include "llvm/IR/InlineAsm.h"
#include "olap/util/jit/control-flow/if-statement.hpp"

namespace gpu {

void Monoid::createUpdate(Context *const context, llvm::Value *val_accumulating,
                          llvm::Value *val_in) {
  context->getBuilder()->CreateStore(
      create(context,
             context->getBuilder()->CreateLoad(
                 val_accumulating->getType()->getPointerElementType(),
                 val_accumulating),
             val_in),
      val_accumulating);
}

void Monoid::createUpdateFromBareValue(Context *const context,
                                       llvm::Value *val_accumulating,
                                       llvm::Value *val_in) {
  createUpdate(context, val_accumulating, createUnary(context, val_in));
}

const ExpressionType *Monoid::getOutputType(const ExpressionType *eType) {
  return eType;
}

llvm::Type *Monoid::getStorageType(Context *context, llvm::Type *updateType) {
  return updateType;
}

llvm::Value *MaxMonoid::create(Context *const context,
                               llvm::Value *val_accumulating,
                               llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  llvm::Value *maxCondition;

  if (val_accumulating->getType()->isIntegerTy()) {
    maxCondition = Builder->CreateICmpSGT(val_in, val_accumulating);
  } else if (val_accumulating->getType()->isFloatingPointTy()) {
    maxCondition = Builder->CreateFCmpOGT(val_in, val_accumulating);
  } else {
    string error_msg = string("[MaxMonoid: ] Max operates on numerics");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  return Builder->CreateSelect(maxCondition, val_in, val_accumulating);
}

void MaxMonoid::createUpdate(Context *const context,
                             llvm::Value *mem_accumulating,
                             llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  llvm::BasicBlock *curBlock = Builder->GetInsertBlock();

  llvm::Function *TheFunction = curBlock->getParent();
  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      context->getLLVMContext(), "maxEnd", TheFunction);

  if (curBlock == context->getEndingBlock()) context->setEndingBlock(endBlock);

  /**
   * if(curr > max) max = curr;
   */
  llvm::BasicBlock *ifGtMaxBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "maxCond",
                         &ifGtMaxBlock);
  llvm::Value *val_accumulating = Builder->CreateLoad(
      mem_accumulating->getType()->getPointerElementType(), mem_accumulating);

  llvm::Value *maxCondition;

  if (val_accumulating->getType()->isIntegerTy()) {
    maxCondition = Builder->CreateICmpSGT(val_in, val_accumulating);
  } else if (val_accumulating->getType()->isFloatingPointTy()) {
    maxCondition = Builder->CreateFCmpOGT(val_in, val_accumulating);
  } else {
    string error_msg =
        string("[MaxMonoid: ] Max accumulator operates on numerics");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

  Builder->SetInsertPoint(ifGtMaxBlock);
  Builder->CreateStore(val_in, mem_accumulating);
  Builder->CreateBr(endBlock);

  // Back to 'normal' flow
  Builder->SetInsertPoint(endBlock);
}

void MaxMonoid::createAtomicUpdate(Context *const context,
                                   llvm::Value *accumulator_ptr,
                                   llvm::Value *val_in,
                                   llvm::AtomicOrdering order) {
  auto type = llvm::Type::getIntNTy(context->getLLVMContext(),
                                    context->getSizeOf(val_in) * 8);
  val_in = context->getBuilder()->CreateBitCast(val_in, type);
  context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::Max,
      context->getBuilder()->CreateBitCast(accumulator_ptr,
                                           llvm::PointerType::getUnqual(type)),
#if LLVM_VERSION_MAJOR >= 13
      val_in, llvm::Align(context->getSizeOf(val_in)), order);
#else
      val_in, order);
#endif
}

llvm::Value *MinMonoid::evalCondition(Context *const context,
                                      llvm::Value *val_accumulating,
                                      llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  if (val_accumulating->getType()->isIntegerTy()) {
    return Builder->CreateICmpSLT(val_in, val_accumulating);
  } else if (val_accumulating->getType()->isFloatingPointTy()) {
    return Builder->CreateFCmpOLT(val_in, val_accumulating);
  } else {
    string error_msg =
        string("[MinMonoid: ] Min accumulator operates on numerics");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
}

llvm::Value *MinMonoid::create(Context *const context,
                               llvm::Value *val_accumulating,
                               llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  auto minCondition = evalCondition(context, val_accumulating, val_in);
  return Builder->CreateSelect(minCondition, val_in, val_accumulating);
}

void MinMonoid::createUpdate(Context *const context,
                             llvm::Value *mem_accumulating,
                             llvm::Value *val_in) {
  auto Builder = context->getBuilder();

  bool setEnding = (Builder->GetInsertBlock() == context->getEndingBlock());

  auto curr = Builder->CreateLoad(
      mem_accumulating->getType()->getPointerElementType(), mem_accumulating);

  auto minCondition = evalCondition(context, curr, val_in);

  /**
   * if(curr > min) min = curr;
   */
  context->gen_if({minCondition, context->createFalse()})(
      [&]() { Builder->CreateStore(val_in, mem_accumulating); });

  if (setEnding) context->setEndingBlock(Builder->GetInsertBlock());
}  // namespace gpu

void MinMonoid::createAtomicUpdate(Context *const context,
                                   llvm::Value *accumulator_ptr,
                                   llvm::Value *val_in,
                                   llvm::AtomicOrdering order) {
  context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::Min, accumulator_ptr, val_in,
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(context->getSizeOf(val_in)),
#endif
      order);
}

llvm::Value *SumMonoid::create(Context *const context,
                               llvm::Value *val_accumulating,
                               llvm::Value *val_in) {
  if (val_in->getType()->isIntegerTy()) {
    return context->getBuilder()->CreateAdd(val_in, val_accumulating);
  } else {
    return context->getBuilder()->CreateFAdd(val_in, val_accumulating);
  }
}

void SumMonoid::createAtomicUpdate(Context *const context,
                                   llvm::Value *accumulator_ptr,
                                   llvm::Value *val_in,
                                   llvm::AtomicOrdering order) {
  if (val_in->getType()
          ->isIntegerTy()) {  // FIXME : llvm does not like non integer atomics
    context->getBuilder()->CreateAtomicRMW(
        llvm::AtomicRMWInst::BinOp::Add, accumulator_ptr, val_in,
#if LLVM_VERSION_MAJOR >= 13
        llvm::Align(context->getSizeOf(val_in)),
#endif
        order);
#if LLVM_VERSION_MAJOR <= 8
  } else if (val_in->getType()->isDoubleTy()) {
    llvm::Function *f = context->getFunction("atomicAdd_double");
    context->getBuilder()->CreateCall(
        f, std::vector<llvm::Value *>{accumulator_ptr, val_in});
  } else if (val_in->getType()->isFloatTy()) {
    llvm::Function *f = context->getFunction("atomicAdd_float");
    context->getBuilder()->CreateCall(
        f, std::vector<llvm::Value *>{accumulator_ptr, val_in});
#else
  } else if (val_in->getType()->isFloatingPointTy()) {
    context->getBuilder()->CreateAtomicRMW(
        llvm::AtomicRMWInst::BinOp::FAdd, accumulator_ptr, val_in,
#if LLVM_VERSION_MAJOR >= 13
        llvm::Align(context->getSizeOf(val_in)),
#endif
        order);
#endif
  } else {
    string error_msg = string("[gpu::SumMonoid: ] Unimplemented atomic update");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
}

llvm::Value *LogOrMonoid::create(Context *const context,
                                 llvm::Value *val_accumulating,
                                 llvm::Value *val_in) {
  return context->getBuilder()->CreateOr(val_in, val_accumulating);
}

void LogOrMonoid::createAtomicUpdate(Context *const context,
                                     llvm::Value *accumulator_ptr,
                                     llvm::Value *val_in,
                                     llvm::AtomicOrdering order) {
  // no atomics for i1
  // FIXME: check if there is a better way to do this + whether this is correct
  auto Builder = context->getBuilder();

  llvm::BasicBlock *curBlock = Builder->GetInsertBlock();

  llvm::Function *TheFunction = curBlock->getParent();
  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      context->getLLVMContext(), "atomOrEnd", TheFunction);

  if (curBlock == context->getEndingBlock()) context->setEndingBlock(endBlock);

  /**
   * if(val_in) *accumulator_ptr = true;
   */
  llvm::BasicBlock *ifBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "atomOrCnd", &ifBlock);

  Builder->CreateCondBr(val_in, ifBlock, endBlock);

  llvm::Value *true_const = llvm::ConstantInt::getTrue(val_in->getType());

  Builder->SetInsertPoint(ifBlock);
  Builder->CreateStore(true_const, accumulator_ptr);
  Builder->CreateBr(endBlock);

  // Back to 'normal' flow
  Builder->SetInsertPoint(endBlock);
}

llvm::Value *LogAndMonoid::create(Context *const context,
                                  llvm::Value *val_accumulating,
                                  llvm::Value *val_in) {
  return context->getBuilder()->CreateAnd(val_in, val_accumulating);
}

void LogAndMonoid::createAtomicUpdate(Context *const context,
                                      llvm::Value *accumulator_ptr,
                                      llvm::Value *val_in,
                                      llvm::AtomicOrdering order) {
  // no atomics for i1
  // FIXME: check if there is a better way to do this + whether this is correct
  auto Builder = context->getBuilder();

  llvm::BasicBlock *curBlock = Builder->GetInsertBlock();

  llvm::Function *TheFunction = curBlock->getParent();
  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      context->getLLVMContext(), "atomLAndEnd", TheFunction);

  if (curBlock == context->getEndingBlock()) context->setEndingBlock(endBlock);

  /**
   * if(!val_in) *accumulator_ptr = false;
   */
  llvm::BasicBlock *ifBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "atomlAndCond",
                         &ifBlock);

  Builder->CreateCondBr(val_in, endBlock, ifBlock);

  llvm::Value *false_const = llvm::ConstantInt::getFalse(val_in->getType());

  Builder->SetInsertPoint(ifBlock);
  Builder->CreateStore(false_const, accumulator_ptr);
  Builder->CreateBr(endBlock);

  // Back to 'normal' flow
  Builder->SetInsertPoint(endBlock);
}

llvm::Value *BitOrMonoid::create(Context *const context,
                                 llvm::Value *val_accumulating,
                                 llvm::Value *val_in) {
  return context->getBuilder()->CreateOr(val_in, val_accumulating);
}

llvm::Value *BitAndMonoid::create(Context *const context,
                                  llvm::Value *val_accumulating,
                                  llvm::Value *val_in) {
  return context->getBuilder()->CreateAnd(val_in, val_accumulating);
}

void BitOrMonoid::createAtomicUpdate(Context *const context,
                                     llvm::Value *accumulator_ptr,
                                     llvm::Value *val_in,
                                     llvm::AtomicOrdering order) {
  context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::Or, accumulator_ptr, val_in,
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(context->getSizeOf(val_in)),
#endif
      order);
}

void BitAndMonoid::createAtomicUpdate(Context *const context,
                                      llvm::Value *accumulator_ptr,
                                      llvm::Value *val_in,
                                      llvm::AtomicOrdering order) {
  context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::And, accumulator_ptr, val_in,
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(context->getSizeOf(val_in)),
#endif
      order);
}

Monoid *Monoid::get(::Monoid m) {
  switch (m) {
    case MAX:
      return new MaxMonoid;
    case MIN:
      return new MinMonoid;
    case SUM:
      return new SumMonoid;
    case OR:
      return new LogOrMonoid;
    case AND:
      return new LogAndMonoid;
    case BAGUNION:
      return new CollectMonoid;
    default:
      string error_msg =
          string("[gpu::Monoids: ] Unimplemented monoid for gpu");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
  }
}

llvm::Value *Monoid::createWarpAggregateToAll(Context *const context,
                                              llvm::Value *val_in) {
  for (int i = 16; i > 0; i >>= 1) {
    // NOTE: only whole (32 threads) warps are supported!
    llvm::Value *shfl_res =
        gpu_intrinsic::shfl_bfly((ParallelContext *const)context, val_in, i);
    shfl_res->setName("shfl_res_" + std::to_string(i));

    val_in = create(context, val_in, shfl_res);
  }

  return val_in;
}

llvm::Value *LogOrMonoid::createWarpAggregateToAll(Context *const context,
                                                   llvm::Value *val_in) {
  return gpu_intrinsic::any((ParallelContext *const)context, val_in);
}

llvm::Value *LogAndMonoid::createWarpAggregateToAll(Context *const context,
                                                    llvm::Value *val_in) {
  return gpu_intrinsic::all((ParallelContext *const)context, val_in);
}

llvm::Value *CollectMonoid::create(Context *const context,
                                   llvm::Value *val_accumulating,
                                   llvm::Value *val_in) {
  throw proteus::unsupported_operation(
      "unimplemented out-of-place update for collect as it's too expensive");
}

std::deque<int32_t> *create_proteus_int_collection(int32_t v) {
  auto x = new std::deque<int32_t>;
  x->emplace_back(v);
  return x;
}

void insert_proteus_int_collection(std::deque<int32_t> *col,
                                   std::deque<int32_t> *v) {
  assert(v->size() == 1);
  col->emplace_back(v->front());
  delete v;
}

void CollectMonoid::createUpdate(Context *const context,
                                 llvm::Value *val_accumulating,
                                 llvm::Value *val_in) {
  context->gen_call(insert_proteus_int_collection,
                    {context->getBuilder()->CreateLoad(
                         val_accumulating->getType()->getPointerElementType(),
                         val_accumulating),
                     val_in});
}

llvm::Value *CollectMonoid::createUnary(Context *context, llvm::Value *val_in) {
  return context->gen_call(create_proteus_int_collection, {val_in});
}

void CollectMonoid::createAtomicUpdate(Context *const context,
                                       llvm::Value *accumulator_ptr,
                                       llvm::Value *val_in,
                                       llvm::AtomicOrdering order) {
  throw proteus::unsupported_operation("atomic collect is unimplemented");
}

const ExpressionType *CollectMonoid::getOutputType(
    const ExpressionType *eType) {
  return new BagType(*eType);
}

llvm::Type *CollectMonoid::getStorageType(Context *context,
                                          llvm::Type *updateType) {
  return context->toLLVM<std::invoke_result_t<
      decltype(create_proteus_int_collection), int32_t> >();
}

}  // namespace gpu

namespace std {
string to_string(const gpu::Monoid &m) { return m.to_string(); }
}  // namespace std
