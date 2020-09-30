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

#include "gpu-hash-rearrange.hpp"

#include <algorithm>

#include "common/gpu/gpu-common.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/expressions-hasher.hpp"
#include "lib/util/gpu/gpu-intrinsics.hpp"
#include "lib/util/jit/pipeline.hpp"
#include "memory/buffer-manager.cuh"
#include "memory/memory-manager.hpp"
#include "olap/util/jit/control-flow/if-statement.hpp"
#include "topology/topology.hpp"

using namespace llvm;

void GpuHashRearrange::produce_(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  Type *idx_type = Type::getInt32Ty(llvmContext);

  for (const auto &e : matExpr) {
    PointerType *t_ptr = PointerType::get(
        e.getExpressionType()->getLLVMType(llvmContext), /* address space */ 0);
    buffVar_id.push_back(context->appendStateVar(
        PointerType::getUnqual(ArrayType::get(t_ptr, numOfBuckets))));
  }

  Plugin *pg =
      Catalog::getInstance().getPlugin(matExpr[0].getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);

  cntVar_id = context->appendStateVar(
      PointerType::getUnqual(ArrayType::get(idx_type, numOfBuckets)));
  wcntVar_id = context->appendStateVar(
      PointerType::getUnqual(ArrayType::get(idx_type, numOfBuckets)));
  oidVar_id = context->appendStateVar(PointerType::getUnqual(oid_type));

  ((ParallelContext *)context)->registerOpen(this, [this](Pipeline *pip) {
    this->open(pip);
  });
  ((ParallelContext *)context)->registerClose(this, [this](Pipeline *pip) {
    this->close(pip);
  });

  getChild()->produce(context);
}

Value *GpuHashRearrange::hash(const std::vector<expression_t> &exprs,
                              ParallelContext *context,
                              const OperatorState &childState) {
  if (exprs.size() == 1) {
    ExpressionHasherVisitor hasher{context, childState};
    return exprs[0].accept(hasher).value;
  } else {
    std::list<expressions::AttributeConstruction> a;
    size_t i = 0;
    for (const auto &e : exprs) a.emplace_back("k" + std::to_string(i++), e);

    ExpressionHasherVisitor hasher{context, childState};
    return expressions::RecordConstruction{a}.accept(hasher).value;
  }
}

void GpuHashRearrange::consume(ParallelContext *context,
                               const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  Plugin *pg =
      Catalog::getInstance().getPlugin(matExpr[0].getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);
  Type *bool_type = context->createTrue()->getType();

  Type *idx_type = int32_type;

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  // std::sort(matExpr.begin(), matExpr.end(), [](const GpuMatExpr& a, const
  // GpuMatExpr& b){
  //     return a.packet < b.packet || a.bitoffset < b.bitoffset;
  // });

  // size_t i = 0;

  std::vector<Value *> buffs;
  // for (size_t p = 0 ; p < packet_widths.size() ; ++p){
  //     // Type * t     =
  //     PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(),
  //     packet_widths[p]), /* address space */ 1);

  //     size_t bindex  = 0;
  //     size_t packind = 0;

  //     std::vector<Type *> body;
  //     while (i < matExpr.size() && matExpr[i].packet == p){
  //         if (matExpr[i].bitoffset != bindex){
  //             //insert space
  //             assert(matExpr[i].bitoffset > bindex);
  //             body.push_back(Type::getIntNTy(context->getLLVMContext(),
  //             (matExpr[i].bitoffset - bindex)));
  //             ++packind;
  //         }

  //         const ExpressionType * out_type =
  //         matExpr[i].expr->getExpressionType();

  //         Type * llvm_type =
  //         out_type->getLLVMType(context->getLLVMContext());

  //         body.push_back(llvm_type);
  //         bindex = matExpr[i].bitoffset + context->getSizeOf(llvm_type);
  //         matExpr[i].packind = packind++;
  //         ++i;
  //     }
  //     assert(packet_widths[p] >= bindex);

  //     if (packet_widths[p] > bindex) {
  //         body.push_back(Type::getIntNTy(context->getLLVMContext(),
  //         (packet_widths[p] - bindex)));
  //     }

  //     Type * t     = StructType::create(body, "hash_rearrange_struct_" +
  //     std::to_string(p), true); Type * t_ptr = PointerType::get(t, /* address
  //     space */ 1);

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
  for (const auto &e : matExpr) {
    PointerType *t_ptr = PointerType::get(
        e.getExpressionType()->getLLVMType(llvmContext), /* address space */ 0);
    Value *buff = new GlobalVariable(
        *(context->getModule()), ArrayType::get(t_ptr, numOfBuckets), false,
        GlobalVariable::LinkageTypes::InternalLinkage,
        ConstantAggregateZero::get(ArrayType::get(t_ptr, numOfBuckets)), "buff",
        nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal, 3, false);

    buffs.push_back(buff);

    max_width = std::max(
        max_width,
        context->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)));
  }

  Value *cnt = new GlobalVariable(
      *(context->getModule()), ArrayType::get(idx_type, numOfBuckets), false,
      GlobalVariable::LinkageTypes::InternalLinkage,
      ConstantAggregateZero::get(ArrayType::get(idx_type, numOfBuckets)), "cnt",
      nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal, 3, false);

  Value *wcnt = new GlobalVariable(
      *(context->getModule()), ArrayType::get(idx_type, numOfBuckets), false,
      GlobalVariable::LinkageTypes::InternalLinkage,
      ConstantAggregateZero::get(ArrayType::get(idx_type, numOfBuckets)),
      "wcnt", nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal, 3,
      false);

  assert(numOfBuckets < 32);

  cap = blockSize / max_width;

  Value *capacity = ConstantInt::get(idx_type, cap);
  Value *last_index = ConstantInt::get(idx_type, cap - 1);

  // if (threadIdx.x < parts){
  //     buff[threadIdx.x] = (uint32_t *) get_buffers();

  //     cnt [threadIdx.x] = 0;
  //     wcnt[threadIdx.x] = 0;
  // }
  // __syncthreads();

  Value *thrd = context->threadIdInBlock();
  Value *numOfBucketsT = ConstantInt::get(thrd->getType(), numOfBuckets);
  Value *cond = Builder->CreateICmpULT(thrd, numOfBucketsT);

  std::vector<Value *> idx{context->createInt32(0), thrd};
  std::vector<Value *> idxStored{context->blockId(), thrd};

  Function *get_buffers = context->getFunction("get_buffers");
  Function *syncthreads = context->getFunction("syncthreads");

  expressions::ProteusValueExpression condExpr{new BoolType(),
                                               {cond, context->createFalse()}};

  auto activemask = gpu_intrinsic::activemask(context);
  context->gen_if(condExpr, childState)([&] {
    for (size_t i = 0; i < buffs.size(); ++i) {
      Value *buff_thrd = Builder->CreateInBoundsGEP(buffs[i], idx);
      Value *stored_buff_thrd = Builder->CreateInBoundsGEP(
          context->getStateVar(buffVar_id[i]), idxStored);

      // Value * new_buff  = Builder->CreateCall(get_buffers);

      // new_buff          = Builder->CreateBitCast(new_buff,
      // buff_thrd->getType()->getPointerElementType());
      Value *stored_buff = Builder->CreateLoad(stored_buff_thrd);
      Builder->CreateStore(stored_buff, buff_thrd, true);
    }

    Value *cnt_thrd = Builder->CreateInBoundsGEP(cnt, idx);
    Value *stored_cnt_ptr =
        Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
    Value *stored_cnt = Builder->CreateLoad(stored_cnt_ptr);
    Builder->CreateStore(stored_cnt, cnt_thrd, true);

    Value *wcnt_thrd = Builder->CreateInBoundsGEP(wcnt, idx);
    Value *stored_wcnt_ptr =
        Builder->CreateInBoundsGEP(context->getStateVar(wcntVar_id), idxStored);
    Value *stored_wcnt = Builder->CreateLoad(stored_wcnt_ptr);
    Builder->CreateStore(stored_wcnt, wcnt_thrd, true);
  });
  auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  Builder->CreateCall(warpsync, {activemask});

  Builder->CreateCall(syncthreads);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  Builder->SetInsertPoint(context->getEndingBlock());

  // __syncthreads();
  // if (threadIdx.x < parts){
  //     buff[threadIdx.x] = (uint32_t *) get_buffers();

  //     cnt [threadIdx.x] = 0;
  //     wcnt[threadIdx.x] = 0;
  // }

  Builder->CreateCall(syncthreads);

  activemask = gpu_intrinsic::activemask(context);
  context->gen_if(condExpr, childState)([&] {
    for (size_t i = 0; i < buffs.size(); ++i) {
      Value *buff_thrd = Builder->CreateInBoundsGEP(buffs[i], idx);
      Value *stored_buff_thrd = Builder->CreateInBoundsGEP(
          context->getStateVar(buffVar_id[i]), idxStored);

      // Value * new_buff  = Builder->CreateCall(get_buffers);

      // new_buff          = Builder->CreateBitCast(new_buff,
      // buff_thrd->getType()->getPointerElementType());
      Value *stored_buff = Builder->CreateLoad(buff_thrd, true);
      Builder->CreateStore(stored_buff, stored_buff_thrd);
    }

    auto cnt_thrd = Builder->CreateInBoundsGEP(cnt, idx);
    auto stored_cnt_ptr =
        Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
    auto stored_cnt = Builder->CreateLoad(cnt_thrd, true);
    Builder->CreateStore(stored_cnt, stored_cnt_ptr);

    auto wcnt_thrd = Builder->CreateInBoundsGEP(wcnt, idx);
    auto stored_wcnt_ptr =
        Builder->CreateInBoundsGEP(context->getStateVar(wcntVar_id), idxStored);
    auto stored_wcnt = Builder->CreateLoad(wcnt_thrd, true);
    Builder->CreateStore(stored_wcnt, stored_wcnt_ptr);
  });
  Builder->CreateCall(warpsync, {activemask});

  context->setEndingBlock(Builder->GetInsertBlock());

  Builder->SetInsertPoint(insBB);

  // Generate target
  // ExpressionHasherVisitor exprGenerator{context, childState};
  Value *target = GpuHashRearrange::hash({hashExpr}, context, childState);
  target = Builder->CreateTruncOrBitCast(target, int32_type);

  auto *target_type = (IntegerType *)target->getType();
  if (hashProject) {
    target_type = (IntegerType *)hashProject->getLLVMType(llvmContext);
    target = Builder->CreateZExtOrTrunc(target, target_type);
  }

  Value *numOfBucketsV = ConstantInt::get(target_type, numOfBuckets);
  target = Builder->CreateURem(target, numOfBucketsV);
  target->setName("target");

  map<RecordAttribute, ProteusValueMemory> variableBindings;
  AllocaInst *not_done_ptr =
      context->CreateEntryBlockAlloca(F, "not_done_ptr", bool_type);
  Builder->CreateStore(context->createTrue(), not_done_ptr);

  if (hashProject) {
    // Save hash in bindings
    variableBindings[*hashProject] =
        context->toMem(target, context->createFalse(), "hash_ptr");
  }

  BasicBlock *startInsBB = BasicBlock::Create(llvmContext, "startIns", F);
  BasicBlock *endIndBB = BasicBlock::Create(llvmContext, "endInd", F);

  auto activemask_any = gpu_intrinsic::activemask(context);
  Builder->CreateBr(startInsBB);
  Builder->SetInsertPoint(startInsBB);

  AllocaInst *mask_ptr =
      context->CreateEntryBlockAlloca(F, "mask_ptr", int32_type);
  Builder->CreateStore(UndefValue::get(int32_type), mask_ptr);

  activemask = gpu_intrinsic::activemask(context);
  Value *mask = UndefValue::get(int32_type);

  // uint32_t mask;
  // #pragma unroll
  // for (uint32_t j = 0 ; j < parts ; ++j){
  //     uint32_t tmp = __ballot((!done) && (h == j));
  //     if (j == h) mask = tmp;
  // }
  for (int i = 0; i < numOfBuckets; ++i) {
    Value *comp =
        Builder->CreateICmpEQ(target, ConstantInt::get(target_type, i));
    Value *vote = gpu_intrinsic::ballot(
        context, Builder->CreateAnd(comp, Builder->CreateLoad(not_done_ptr)));
    mask = Builder->CreateSelect(comp, vote, mask);
  }

  Builder->CreateCall(warpsync, {activemask});
  AllocaInst *indx_ptr = context->CreateEntryBlockAlloca(
      F, "indx_ptr", idx_type);  // consider lowering it into 32bit
  Builder->CreateStore(UndefValue::get(idx_type), indx_ptr);

  // uint32_t idx;

  // uint32_t h_leader    = mask & -mask;
  // uint32_t lanemask_eq = 1 << get_laneid();
  Value *h_leader = Builder->CreateAnd(mask, Builder->CreateNeg(mask));

  Function *f_lanemask_eq =
      context->getFunction("llvm.nvvm.read.ptx.sreg.lanemask.eq");
  Value *lanemask_eq = Builder->CreateCall(f_lanemask_eq);

  // if (lanemask_eq == h_leader) idx = atomicAdd_block(cnt + h, __popc(mask));
  Value *inc_cond = Builder->CreateICmpEQ(lanemask_eq, h_leader);

  Function *popc = context->getFunction("llvm.ctpop");

  expressions::ProteusValueExpression incCondExpr{
      new BoolType(), {inc_cond, context->createFalse()}};

  activemask = gpu_intrinsic::activemask(context);
  context->gen_if(incCondExpr, childState)([&]() {
    idx[1] = target;
    Value *cnt_ptr = Builder->CreateInBoundsGEP(cnt, idx);
    Value *pop = Builder->CreateCall(popc, mask);
    // TODO: make it block atomic! but it is on shared memory, so it does not
    // matter :P
    Value *new_idx = Builder->CreateAtomicRMW(
        AtomicRMWInst::BinOp::Add, cnt_ptr, Builder->CreateZExt(pop, idx_type),
        AtomicOrdering::Monotonic);
    Builder->CreateStore(new_idx, indx_ptr);
  });
  Builder->CreateCall(warpsync, {activemask});

  // idx  = __shfl(idx, __popc(h_leader - 1)) + __popc(mask & ((1 <<
  // get_laneid()) - 1));

  Value *h_leader_lt = Builder->CreateSub(
      h_leader, ConstantInt::get((IntegerType *)(h_leader->getType()), 1));
  Value *leader_id = Builder->CreateCall(popc, h_leader_lt);

  Function *shfl = context->getFunction("llvm.nvvm.shfl.sync.idx.i32");
  Function *lanemask_lt =
      context->getFunction("llvm.nvvm.read.ptx.sreg.lanemask.lt");

  std::vector<Value *> args{activemask, Builder->CreateLoad(indx_ptr),
                            leader_id, context->createInt32(warp_size - 1)};
  Value *idx_g = Builder->CreateAdd(
      Builder->CreateCall(shfl, args),
      Builder->CreateCall(popc, std::vector<Value *>{Builder->CreateAnd(
                                    mask, Builder->CreateCall(lanemask_lt))}));

  Value *storeCond =
      Builder->CreateICmpULT(idx_g, capacity);  // context->createTrue();
  // Builder->CreateAnd(Builder->CreateLoad(not_done_ptr),
  //                    Builder->CreateICmpULT(idx_g, capacity));
  expressions::ProteusValueExpression storeCondExpr{
      new BoolType(), {storeCond, context->createFalse()}};

  // if (!done){
  //    if (idx < h_vector_size/4){
  activemask = gpu_intrinsic::activemask(context);
  context->gen_if(storeCondExpr, childState)([&]() {
    //        uint32_t * b_old = buff[h];
    //        reinterpret_cast<int4 *>(b_old)[idx] = tmp.vec;
    //        __threadfence_block(); //even safer and with the same
    //        performance!!!! : __threadfence();
    std::vector<Value *> buff_ptrs;
    // std::vector<Value *> out_ptrs;
    // std::vector<Value *> out_vals;

    idx[1] = target;
    for (size_t i = 0; i < buffs.size(); ++i) {
      // out_vals.push_back(UndefValue::get(b->getType()->getPointerElementType()->getPointerElementType()));
      buff_ptrs.push_back(
          Builder->CreateLoad(Builder->CreateInBoundsGEP(buffs[i], idx), true));
      Value *out_ptr = Builder->CreateInBoundsGEP(buff_ptrs.back(), idx_g);

      ExpressionGeneratorVisitor exprGenerator(context, childState);
      ProteusValue valWrapper = matExpr[i].accept(exprGenerator);
      Builder->CreateStore(valWrapper.value, out_ptr);
    }

    Function *threadfence = context->getFunction("threadfence");
    Builder->CreateCall(threadfence);

    Value *w = Builder->CreateAtomicRMW(
        AtomicRMWInst::BinOp::Add, Builder->CreateInBoundsGEP(wcnt, idx),
        ConstantInt::get(idx_type, 1), AtomicOrdering::Monotonic);

    expressions::ProteusValueExpression cExpr{
        new BoolType(),
        {Builder->CreateICmpEQ(w, last_index), context->createFalse()}};

    auto activemask = gpu_intrinsic::activemask(context);
    context->gen_if(cExpr, childState)([&] {
      auto activemask = gpu_intrinsic::activemask(context);
      for (const auto &buff : buffs) {
        Value *buff_thrd = Builder->CreateInBoundsGEP(buff, idx);
        Value *new_buff = Builder->CreateCall(get_buffers);
        new_buff = Builder->CreateBitCast(
            new_buff, buff_thrd->getType()->getPointerElementType());
        Builder->CreateStore(new_buff, buff_thrd, true);
      }

      Builder->CreateCall(warpsync, {activemask});
      Function *threadfence_block = context->getFunction("threadfence_block");
      Builder->CreateCall(threadfence_block);

      Builder->CreateAtomicRMW(
          AtomicRMWInst::BinOp::Xchg, Builder->CreateInBoundsGEP(wcnt, idx),
          ConstantInt::get(idx_type, 0), AtomicOrdering::Monotonic);
      Builder->CreateCall(threadfence_block);

      Builder->CreateAtomicRMW(
          AtomicRMWInst::BinOp::Xchg, Builder->CreateInBoundsGEP(cnt, idx),
          ConstantInt::get(idx_type, 0), AtomicOrdering::Monotonic);
      Builder->CreateCall(threadfence_block);

      Builder->CreateCall(warpsync, {activemask});
      // call parent
      RecordAttribute tupCnt{matExpr[0].getRegisteredRelName(), "activeCnt",
                             pg->getOIDType()};  // FIXME: OID type for blocks ?

      variableBindings[tupCnt] =
          context->toMem(ConstantInt::get(oid_type, cap),
                         context->createFalse(), "blockN_ptr");

      Value *new_oid = Builder->CreateAtomicRMW(
          AtomicRMWInst::BinOp::Add,
          ((ParallelContext *)context)->getStateVar(oidVar_id),
          ConstantInt::get(oid_type, cap), AtomicOrdering::Monotonic);
      new_oid->setName("oid");

      RecordAttribute tupleIdentifier{matExpr[0].getRegisteredRelName(),
                                      activeLoop, pg->getOIDType()};
      variableBindings[tupleIdentifier] =
          context->toMem(new_oid, context->createFalse(), "new_oid_ptr");

      for (size_t i = 0; i < matExpr.size(); ++i) {
        RecordAttribute battr(matExpr[i].getRegisteredAs(), true);

        variableBindings[battr] =
            context->toMem(buff_ptrs[i], context->createFalse(),
                           "mem_" + matExpr[i].getRegisteredAttrName());
      }

      OperatorState state{*this, variableBindings};
      getParent()->consume(context, state);

      Builder->CreateCall(warpsync, {activemask});
    });
    Builder->CreateCall(warpsync, {activemask});

    Builder->CreateStore(context->createFalse(), not_done_ptr);
  });
  Builder->CreateCall(warpsync, {activemask});

  // Value *any = gpu_intrinsic::any(context,
  // Builder->CreateLoad(not_done_ptr));
  Value *any = Builder->CreateLoad(not_done_ptr);

  Builder->CreateCondBr(any, startInsBB, endIndBB);

  Builder->SetInsertPoint(endIndBB);
  Builder->CreateCall(warpsync, {activemask_any});

  // Function * get_buffer = context->getFunction("get_buffer");

  // for (size_t i = 0 ; i < wantedFields.size() ; ++i){
  //     RecordAttribute tblock{*(wantedFields[i]), true};
  //     Value * size     = context->createSizeT(blockSize *
  //     context->getSizeOf(wantedFields[i]->getLLVMType(llvmContext)));

  //     Value * new_buff = Builder->CreateCall(get_buffer, std::vector<Value
  //     *>{size});

  //     new_buff = Builder->CreateBitCast(new_buff,
  //     tblock.getLLVMType(llvmContext));

  //     Builder->CreateStore(new_buff, Builder->CreateInBoundsGEP(curblk,
  //     std::vector<Value *>{context->createInt32(0),
  //     context->createInt32(i)}));
  // }

  // Builder->CreateStore(ConstantInt::get(oid_type, 0), indx_addr);

  // Builder->CreateBr(mergeBB);

  // // else
  // Builder->SetInsertPoint(elseBB);

  // Builder->CreateStore(Builder->CreateAdd(indx, ConstantInt::get(oid_type,
  // 1)), indx_addr);

  // Builder->CreateBr(mergeBB);

  // // merge
  // Builder->SetInsertPoint(mergeBB);

  // flush remaining elements
  consume_flush(context, target_type);
}

void GpuHashRearrange::consume_flush(ParallelContext *context,
                                     llvm::IntegerType *target_type) {
  save_current_blocks_and_restore_at_exit_scope blks{context};
  LLVMContext &llvmContext = context->getLLVMContext();

  flushingFunc = (*context)->createHelperFunction(
      "flush", std::vector<Type *>{}, std::vector<bool>{}, std::vector<bool>{});
  closingPip = (context->operator->());
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();
  // Get the ENTRY BLOCK
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  BasicBlock *mainBB = BasicBlock::Create(llvmContext, "main", F);

  Builder->SetInsertPoint(mainBB);

  BasicBlock *initBB = BasicBlock::Create(llvmContext, "init", F);
  BasicBlock *endIBB = BasicBlock::Create(llvmContext, "endI", F);

  context->setEndingBlock(endIBB);

  Plugin *pg =
      Catalog::getInstance().getPlugin(matExpr[0].getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);

  Type *charPtrType = Type::getInt8PtrTy(llvmContext);

  // if (threadIdx.x < parts){
  //     buff[threadIdx.x] = (uint32_t *) get_buffers();

  //     cnt [threadIdx.x] = 0;
  //     wcnt[threadIdx.x] = 0;
  // }
  // __syncthreads();

  Value *thrd = context->threadIdInBlock();
  Value *target = Builder->CreateZExtOrTrunc(thrd, target_type);
  Value *numOfBucketsT = ConstantInt::get(target_type, numOfBuckets);
  Value *cond = Builder->CreateICmpULT(target, numOfBucketsT);

  BasicBlock *endJBB = BasicBlock::Create(llvmContext, "endJ", F);
  auto activemask = gpu_intrinsic::activemask(context);
  Builder->CreateCondBr(cond, initBB, endJBB);

  Builder->SetInsertPoint(initBB);

  std::vector<Value *> idx{context->createInt32(0), thrd};
  std::vector<Value *> idxStored{context->blockId(), thrd};

  std::vector<Value *> buff_ptrs;
  for (size_t i = 0; i < matExpr.size(); ++i) {
    Value *stored_buff_thrd = Builder->CreateInBoundsGEP(
        context->getStateVar(buffVar_id[i]), idxStored);

    buff_ptrs.push_back(Builder->CreateLoad(stored_buff_thrd));
  }

  // Value * cnt__thrd       = Builder->CreateInBoundsGEP(cnt , idx);
  Value *stored_cnt_ptr =
      Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id), idxStored);
  Value *cnt = Builder->CreateLoad(stored_cnt_ptr);

  BasicBlock *emptyBB = BasicBlock::Create(llvmContext, "empty", F);
  BasicBlock *non_emptyBB = BasicBlock::Create(llvmContext, "non_empty", F);

  Value *empty_cond = Builder->CreateICmpEQ(
      cnt, ConstantInt::get((IntegerType *)cnt->getType(), 0));
  Builder->CreateCondBr(empty_cond, emptyBB, non_emptyBB);

  Builder->SetInsertPoint(emptyBB);

  Function *f =
      context->getFunction("release_buffers");  // FIXME: Assumes grid launch +
                                                // Assumes 1 block per kernel!

  for (size_t i = 0; i < matExpr.size(); ++i) {
    Builder->CreateCall(f, std::vector<Value *>{Builder->CreateBitCast(
                               buff_ptrs[i], charPtrType)});
  }

  Builder->CreateBr(endJBB);

  Builder->SetInsertPoint(non_emptyBB);

  // call parent
  map<RecordAttribute, ProteusValueMemory> variableBindings;

  RecordAttribute tupCnt{matExpr[0].getRegisteredRelName(), "activeCnt",
                         pg->getOIDType()};  // FIXME: OID type for blocks ?

  AllocaInst *blockN_ptr =
      context->CreateEntryBlockAlloca(F, "blockN_ptr", oid_type);

  if (hashProject) {
    // Save hash in bindings
    variableBindings[*hashProject] =
        context->toMem(target, context->createFalse(), "hash_ptr");
  }

  cnt = Builder->CreateZExt(cnt, oid_type);

  Builder->CreateStore(cnt, blockN_ptr);

  ProteusValueMemory mem_cntWrapper{blockN_ptr, context->createFalse()};
  variableBindings[tupCnt] = mem_cntWrapper;

  Value *new_oid = Builder->CreateAtomicRMW(
      AtomicRMWInst::BinOp::Add,
      ((ParallelContext *)context)->getStateVar(oidVar_id), cnt,
      AtomicOrdering::Monotonic);
  new_oid->setName("oid");

  RecordAttribute tupleIdentifier{matExpr[0].getRegisteredRelName(), activeLoop,
                                  pg->getOIDType()};
  variableBindings[tupleIdentifier] =
      context->toMem(new_oid, context->createFalse(), "new_oid_ptr");

  for (size_t i = 0; i < matExpr.size(); ++i) {
    RecordAttribute battr(matExpr[i].getRegisteredAs(), true);

    variableBindings[battr] =
        context->toMem(buff_ptrs[i], context->createFalse(),
                       "mem_" + matExpr[i].getRegisteredAttrName());
  }

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  Builder->CreateBr(endJBB);
  Builder->SetInsertPoint(endJBB);
  auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  Builder->CreateCall(warpsync, {activemask});
  Builder->CreateBr(endIBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(mainBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateRetVoid();
}

void call_GpuHashRearrange_acq_buffs(size_t cnt, cudaStream_t strm,
                                     void **buffs);

void GpuHashRearrange::open(Pipeline *pip) {
  execution_conf ec = pip->getExecConfiguration();

  size_t grid_size = ec.gridSize();

  size_t buffs_bytes =
      ((sizeof(void *) * numOfBuckets * buffVar_id.size() * grid_size) + 16 -
       1) &
      ~((size_t)0xF);
  size_t cnts_bytes =
      ((sizeof(int32_t) * numOfBuckets * 2 * grid_size) + 16 - 1) &
      ~((size_t)0xF);
  size_t oid_bytes = ((sizeof(size_t)) + 16 - 1) & ~((size_t)0xF);

  void **buffs =
      (void **)MemoryManager::mallocGpu(buffs_bytes + cnts_bytes + oid_bytes);
  auto *cnts = (int32_t *)(((char *)buffs) + buffs_bytes);
  auto *oid = (size_t *)(((char *)cnts) + cnts_bytes);

  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaMemsetAsync(cnts, 0, cnts_bytes + oid_bytes, strm));
  // for (int i = 0 ; i < numOfBuckets * 2; ++i) cnts[i] = 0;
  // *oid = 0;

  pip->setStateVar<int32_t *>(cntVar_id, cnts);
  pip->setStateVar<int32_t *>(wcntVar_id, cnts + numOfBuckets * grid_size);
  pip->setStateVar<size_t *>(oidVar_id, oid);

  // void ** h_buffs = (void **) malloc(buffs_bytes);

  // for (size_t i = 0 ; i < numOfBuckets * buffVar_id.size() * grid_size; ++i){
  //     h_buffs[i] = curr_buff;
  //     curr_buff += h_vector_size * sizeof(int32_t); // FIMXE: assumes that
  //     all buffers need at most a h_vector_size * sizeof(int32_t) size
  // }

  // for (size_t i = 0 ; i < numOfBuckets * buffVar_id.size() * grid_size; ++i){
  //     h_buffs[i] = buffer-manager<int32_t>::h_get_buffer(device); // FIMXE:
  //     assumes that all buffers need at most a h_vector_size * sizeof(int32_t)
  //     size
  // }

  call_GpuHashRearrange_acq_buffs(numOfBuckets * buffVar_id.size() * grid_size,
                                  strm, buffs);

  // gpu_run(cudaMemcpy(buffs, h_buffs, buffs_bytes, cudaMemcpyDefault));

  // free(h_buffs);

  for (size_t i = 0; i < buffVar_id.size(); ++i) {
    pip->setStateVar<void **>(buffVar_id[i],
                              buffs + numOfBuckets * i * grid_size);
  }
  syncAndDestroyStream(strm);
}

#include <numeric>

struct mv_description {
  const char *__restrict__ from;
  size_t bytes;
  char *__restrict__ to;  //[16];
};

//#ifndef NCUDA
//__device__ void GpuHashRearrange_copy(int4 *__restrict__ to,
//                                      const int4 *__restrict__ from) {
//  *to = *from;
//}
//
//__global__ void GpuHashRearrange_pack(mv_description *desc) {
//  mv_description d = desc[blockIdx.x];
//
//  const int4 *from = (const int4 *)d.from;
//  int4 *to = (int4 *)(((((uint64_t)d.to) + 16 - 1) / 16) * 16);
//  size_t offset = ((char *)to) - d.to;
//
//  size_t packs = (d.bytes - offset) / 16;
//  size_t rem = (d.bytes - offset) % 16;
//
//#pragma unroll 2
//  for (size_t i = threadIdx.x; i < packs; i += blockDim.x) {
//    GpuHashRearrange_copy(to + i, from + i);
//  }
//
//  if (threadIdx.x < offset) {
//    d.to[threadIdx.x] = d.from[packs * 16 + threadIdx.x];
//  }
//
//  if (threadIdx.x < rem) {
//    d.to[d.bytes - rem + threadIdx.x] =
//        d.from[packs * 16 + offset + threadIdx.x];
//  }
//
//  // if (threadIdx.x == 0) release_buffer(from);
//}
//#endif

void GpuHashRearrange::close(Pipeline *pip) {
  // ((void (*)(void *)) this->flushFunc)(pip->getState());
  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

  execution_conf ec = pip->getExecConfiguration();
  size_t grid_size = ec.gridSize();

  void **buffs = pip->getStateVar<void **>(buffVar_id[0]);
  // int32_t *cnts = pip->getStateVar<int32_t *>(cntVar_id);

  void **h_buffs;
  int32_t *h_cnts;
  mv_description *mv_descs;

  size_t h_buffs_bytes =
      ((sizeof(void *) * numOfBuckets * buffVar_id.size() * grid_size) + 16 -
       1) &
      ~((size_t)0xF);
  size_t h_cnts_bytes =
      ((sizeof(int32_t) * numOfBuckets * 2 * grid_size) + 16 - 1) &
      ~((size_t)0xF);
  size_t mv_descs_bytes =
      ((sizeof(mv_description) * numOfBuckets * grid_size * buffVar_id.size()) +
       16 - 1) &
      ~((size_t)0xF);

  // gpu_run(cudaMallocHost ((void **) &h_buffs, h_buffs_bytes + h_cnts_bytes +
  // mv_descs_bytes));
  h_buffs = (void **)malloc(h_buffs_bytes + h_cnts_bytes);
  mv_descs = (mv_description *)MemoryManager::mallocPinned(mv_descs_bytes);
  gpu_run(cudaMemcpyAsync(h_buffs, buffs, h_buffs_bytes + h_cnts_bytes,
                          cudaMemcpyDefault, strm));
  gpu_run(cudaStreamSynchronize(strm));

  h_cnts = (int32_t *)(((char *)h_buffs) + h_buffs_bytes);
  // mv_descs = (mv_description *) (((char *) h_cnts ) + h_cnts_bytes );

  // for (size_t part = 0 ; part < numOfBuckets ; ++part){
  //     int bucks = numOfBuckets;
  //     std::vector<int32_t> idx(grid_size);
  //     for (size_t i = 0 ; i < grid_size ; ++i) idx[i] = i * bucks + part;
  //     std::sort(idx.begin(), idx.end(), [&h_cnts](int32_t i, int32_t j)
  //     {return h_cnts[i] > h_cnts[j];});

  //     int32_t i = 0            ;
  //     int32_t j = grid_size - 1;

  //     size_t attr_size[buffVar_id.size()];
  //     for (size_t attr_i = 0; attr_i < buffVar_id.size() ; ++attr_i){
  //         attr_size[attr_i] =
  //         pip->getSizeOf(matExpr[attr_i]->getExpressionType()->getLLVMType(context->getLLVMContext()));
  //     }

  //     while (i < j){
  //         if (h_cnts[idx[i]] + h_cnts[idx[j]] > cap/16) {
  //             ++i;
  //             continue;
  //         }
  //         for (size_t attr_i = 0; attr_i < buffVar_id.size() ; ++attr_i){
  //             gpu_run(cudaMemcpyAsync(((char *) h_buffs[idx[i]]) +
  //             h_cnts[idx[i]] * attr_size[attr_i], h_buffs[idx[j]],
  //             h_cnts[idx[j]] * attr_size[attr_i], cudaMemcpyDefault, strm));
  //         }

  //         h_cnts[idx[i]] += h_cnts[idx[j]];
  //         h_cnts[idx[j]]  = 0;

  //         --j;
  //     }
  // }

  nvtxRangePushA("waiting_to_release");

  // mv_description * mv_descs = new mv_description[part * grid_size];
  size_t mv_descs_i = 0;
  for (int part = 0; part < numOfBuckets; ++part) {
    std::vector<int32_t> idx(grid_size);
    for (size_t i = 0; i < grid_size; ++i) {
      idx[i] = i * numOfBuckets + part;
      // std::cout << h_buffs[idx[i]] << std::endl;
    }
    std::sort(idx.begin(), idx.end(), [&h_cnts](int32_t i, int32_t j) {
      return h_cnts[i] > h_cnts[j];
    });

    int32_t i = 0;
    int32_t j = grid_size - 1;

    while (j >= 0 && h_cnts[idx[j]] <= 0) --j;

    while (i < j) {
      if (h_cnts[idx[i]] <= 0) {
        ++i;
        continue;
      }
      if (h_cnts[idx[i]] + h_cnts[idx[j]] > cap / 8) {
        ++i;
        continue;
      }
      assert(buffVar_id.size() <
             16);  // limited by the capacity of `to` in mv_descs
      for (size_t attr_i = 0; attr_i < buffVar_id.size();
           ++attr_i) {  // FIXME: generalize for attr_size.size() > 1
        mv_descs[mv_descs_i].from =
            (char *)h_buffs[idx[j] + attr_i * numOfBuckets * grid_size];
        mv_descs[mv_descs_i].to =
            ((char *)h_buffs[idx[i] + attr_i * numOfBuckets * grid_size]) +
            h_cnts[idx[i]] * attr_size[attr_i];
        mv_descs[mv_descs_i].bytes = h_cnts[idx[j]] * attr_size[attr_i];
        // std::cout << (void *) mv_descs[mv_descs_i].from << " " << (void *)
        // mv_descs[mv_descs_i].to << " " << mv_descs[mv_descs_i].bytes <<
        // std::endl;
        mv_descs_i++;
      }

      h_cnts[idx[i]] += h_cnts[idx[j]];
      h_cnts[idx[j]] = 0;

      --j;
    }
  }
  if (mv_descs_i > 0) {
    assert(mv_descs_i < numOfBuckets * grid_size * buffVar_id.size());
    // gpu_run(cudaMemcpyAsync(buffs, h_buffs, h_buffs_bytes + h_cnts_bytes,
    // cudaMemcpyDefault, strm)); gpu_run(cudaMemcpyAsync(buffs, h_buffs,
    // sizeof(void  *) * numOfBuckets * buffVar_id.size() * grid_size,
    // cudaMemcpyDefault, strm)); gpu_run(cudaMemcpyAsync(cnts , h_cnts ,
    // sizeof(int32_t) * numOfBuckets                     * grid_size,
    // cudaMemcpyDefault, strm));

    // GpuHashRearrange_pack<<<mv_descs_i, 1024, 0, strm>>>(mv_descs);
  }

  nvtxRangePop();

  void *KernelParams[] = {pip->getState()};
  launch_kernel((CUfunction)closingPip->getCompiledFunction(flushingFunc),
                KernelParams, strm);
  syncAndDestroyStream(strm);

  // gpu_run(cudaFreeHost(h_buffs));
  free(h_buffs);
  MemoryManager::freePinned(mv_descs);

  // MemoryManager::freeGpu(pip->getStateVar<size_t *> (cntVar_id    ));
  // MemoryManager::freeGpu(pip->getStateVar<size_t *> (oidVar_id    ));
  MemoryManager::freeGpu(pip->getStateVar<void **>(buffVar_id[0]));
  // wcntVar_id is part of cntVar, so they are freed together
  // rest of mem for buffers is part of buffVar_id
}
