/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "gpu-hash-group-by-chained.hpp"

#include <algorithm>
#include <cmath>
#include <codegen/util/gpu/gpu-intrinsics.hpp>
#include <olap/util/jit/control-flow/if-statement.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/topology.hpp>

#include "gmonoids.hpp"
#include "lib/expressions/expressions-generator.hpp"

using namespace llvm;

GpuHashGroupByChained::GpuHashGroupByChained(
    std::vector<GpuAggrMatExpr> agg_exprs, std::vector<expression_t> key_expr,
    Operator *child, int hash_bits, size_t maxInputSize_pre)
    : HashGroupByChained(std::move(agg_exprs), std::move(key_expr), child,
                         hash_bits,
                         std::max(maxInputSize_pre, size_t{128} * 1024)) {
  VLOG_IF(1, maxInputSize_pre != maxInputSize)
      << "GroupBy's hashtable too small for GPU algo, auto-resizing";
}

void GpuHashGroupByChained::produce_(OlapParallelContext *context) {
  context->pushPipeline();

  buildHashTableFormat(context);

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);

  context->popPipeline();

  probe_gen = context->getCurrentPipeline();
  generate_scan(context);
}

void GpuHashGroupByChained::buildHashTableFormat(OlapParallelContext *context) {
  prepareDescription(context);
  for (const auto &t : ptr_types) {
    out_param_ids.push_back(context->appendStateVar(t));  //, true, false));
  }

  Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

  Type *t_cnt = PointerType::get(int32_type, /* address space */ 0);
  cnt_param_id = context->appendStateVar(t_cnt);  //, true, false);

  Type *t_head_ptr = PointerType::get(int32_type, /* address space */ 0);
  head_param_id = context->appendStateVar(t_head_ptr);
}

void GpuHashGroupByChained::generate_build(OlapParallelContext *context,
                                           const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Value *v_true = ConstantInt::getTrue(llvmContext);
  Value *v_false = ConstantInt::getFalse(llvmContext);

  Value *out_cnt = context->getStateVar(cnt_param_id);
  out_cnt->setName(opLabel + "_cnt_ptr");

  Value *head_ptr = context->getStateVar(head_param_id);
  head_ptr->setName(opLabel + "_head_ptr");

  // ExpressionHasherVisitor exphasher{context, childState};
  // Value * hash = kexpr.accept(exphasher).value;
  Value *hash = HashGroupByChained::hash(key_expr, context, childState);

  Value *eochain = ConstantInt::get(
      (IntegerType *)head_ptr->getType()->getPointerElementType(),
      ~((size_t)0));
  // current = head[hash(key)]
  // Value * current =
  // Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash),
  // context->getSizeOf(head_ptr->getType()->getPointerElementType()));
  Value *head_w_hash_ptr = Builder->CreateInBoundsGEP(
      head_ptr->getType()->getNonOpaquePointerElementType(), head_ptr, hash);
  // Atomic READ of the chain head. This was a CAS(head, eochain, eochain)
  // used as an atomic load; the CAS is a read-modify-write that serializes
  // in L2 against every other head access on the same (hot) line -- for
  // low-cardinality group-bys this is one same-line RMW per row. A relaxed
  // atomic load has identical semantics for a pure read and none of the
  // serialization.
  llvm::LoadInst *head_ld = Builder->CreateAlignedLoad(
      eochain->getType(), head_w_hash_ptr,
      llvm::Align(context->getSizeOf(eochain)), "gb_head_read");
  head_ld->setAtomic(llvm::AtomicOrdering::Monotonic);
  Value *current = head_ld;

  current->setName("current");

  AllocaInst *mem_current = context->CreateEntryBlockAlloca(
      TheFunction, "mem_current", current->getType());
  Builder->CreateStore(current, mem_current);

  AllocaInst *mem_idx = context->CreateEntryBlockAlloca(
      TheFunction, "mem_idx", out_cnt->getType()->getPointerElementType());
  Builder->CreateStore(
      UndefValue::get(out_cnt->getType()->getPointerElementType()), mem_idx);

  AllocaInst *mem_written = context->CreateEntryBlockAlloca(
      TheFunction, "mem_written", v_false->getType());
  Builder->CreateStore(v_false, mem_written);

  // std::vector<Value *> out_ptrs;
  std::vector<Value *> out_vals;

  for (size_t i = 0; i < out_param_ids.size(); ++i) {
    Value *out_ptr = context->getStateVar(out_param_ids[i]);
    out_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");

    out_vals.push_back(
        UndefValue::get(out_ptr->getType()->getPointerElementType()));
  }

  out_vals[0] = Builder->CreateInsertValue(out_vals[0], eochain, 0);

  for (const GpuAggrMatExpr &mexpr : agg_exprs) {
    ExpressionGeneratorVisitor exprGenerator(context, childState);
    ProteusValue valWrapper = mexpr.expr.accept(exprGenerator);

    out_vals[mexpr.packet] = Builder->CreateInsertValue(
        out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
  }

  BasicBlock *ThenBB =
      BasicBlock::Create(llvmContext, "chainFollow", TheFunction);
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont", TheFunction);

  Value *init_cond = Builder->CreateICmpEQ(
      Builder->CreateLoad(mem_current->getType()->getPointerElementType(),
                          mem_current),
      eochain);

  // if (current == ((uint32_t) -1)){

  expressions::ProteusValueExpression initCondExpr{
      new BoolType(), ProteusValue{init_cond, context->createFalse()}};

  auto activemask = gpu_intrinsic::activemask(context);
  gen_if(initCondExpr, childState, context)([&]() {
    // index
    Value *old_cnt = context->workerScopedAtomicAdd(
        out_cnt,
        ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1));
    old_cnt->setName("index");
    Builder->CreateStore(old_cnt, mem_idx);

    // next[idx].sum  = val;
    // next[idx].key  = key;
    // next[idx].next =  -1;
    context->workerScopedAtomicXchg(
        Builder->CreateBitCast(
            Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0])
                                           ->getType()
                                           ->getNonOpaquePointerElementType(),
                                       context->getStateVar(out_param_ids[0]),
                                       old_cnt),
            PointerType::getInt32PtrTy(llvmContext)),
        eochain);

    for (size_t i = 1; i < out_param_ids.size(); ++i) {
      Value *out_ptr = context->getStateVar(out_param_ids[i]);

      Value *out_ptr_i = Builder->CreateInBoundsGEP(
          out_ptr->getType()->getNonOpaquePointerElementType(), out_ptr,
          old_cnt);
      Builder->CreateAlignedStore(out_vals[i], out_ptr_i,
                                  llvm::Align(packet_widths[i] / 8), true);
    }

    // written = true;
    Builder->CreateStore(v_true, mem_written);

    context->workerScopedMembar();

    // current = atomicCAS(&(first[bucket]), -1, idx);
    Value *old_current = Builder->CreateAtomicCmpXchg(
        head_w_hash_ptr, eochain, old_cnt,
#if LLVM_VERSION_MAJOR >= 13
        llvm::Align(context->getSizeOf(eochain)),
#endif
        llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic);

    Builder->CreateStore(Builder->CreateExtractValue(old_current, 0),
                         mem_current);
  });
  auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  Builder->CreateCall(warpsync, {activemask});

  // if (current != ((uint32_t) -1)){
  //     while (true) {

  Value *chain_cond = Builder->CreateICmpNE(
      Builder->CreateLoad(mem_current->getType()->getPointerElementType(),
                          mem_current),
      eochain);

  Builder->CreateCondBr(chain_cond, ThenBB, MergeBB);

  Builder->SetInsertPoint(MergeBB);
  // Do not reuse a pre-loop warp mask here. Chain-follow loops are
  // data-dependent; using a stale mask can deadlock when lanes reconverge with
  // a different active set.
  Builder->CreateCall(warpsync, {gpu_intrinsic::activemask(context)});

  Builder->SetInsertPoint(ThenBB);
  current = Builder->CreateLoad(mem_current->getType()->getPointerElementType(),
                                mem_current);

  context->workerScopedMembar();

  // Keys
  Value *next_bucket_ptr = Builder->CreateInBoundsGEP(
      context->getStateVar(out_param_ids[1])
          ->getType()
          ->getNonOpaquePointerElementType(),
      context->getStateVar(out_param_ids[1]), current);
  Value *next_bucket = Builder->CreateAlignedLoad(
      next_bucket_ptr->getType()->getPointerElementType(), next_bucket_ptr,
      llvm::Align(packet_widths[1] / 8), true, "next_bucket");

  context->workerScopedMembar();

  // Atomic READ of next[current].next (was CAS-as-load; see head read above).
  llvm::LoadInst *next_ld = Builder->CreateAlignedLoad(
      eochain->getType(),
      Builder->CreateInBoundsGEP(
          context->getStateVar(out_param_ids[0])
              ->getType()
              ->getNonOpaquePointerElementType(),
          context->getStateVar(out_param_ids[0]),
          std::vector<Value *>{current, context->createInt32(0)}),
      llvm::Align(context->getSizeOf(eochain)), "gb_next_read");
  next_ld->setAtomic(llvm::AtomicOrdering::Monotonic);
  Value *next = next_ld;

  // next_bucket_next->setName("next_bucket_next");
  //             // int32_t   next_bucket = next[current].next;
  // Value * next            = Builder->CreateExtractValue(next_bucket, 0);
  // Value * key             = Builder->CreateExtractValue(next_bucket, 1);

  BasicBlock *BucketFoundBB =
      BasicBlock::Create(llvmContext, "BucketFound", TheFunction);
  BasicBlock *ContFollowBB =
      BasicBlock::Create(llvmContext, "ContFollow", TheFunction);

  Value *bucket_cond = v_true;
  for (size_t i = 0; i < key_expr.size(); ++i) {
    ExpressionGeneratorVisitor exprGenerator(context, childState);
    ProteusValue keyWrapper = key_expr[i].accept(exprGenerator);

    Value *key = Builder->CreateExtractValue(next_bucket, i);

    expressions::ProteusValueExpression kexpr{key_expr[i].getExpressionType(),
                                              keyWrapper};
    expressions::ProteusValueExpression kbuck{
        key_expr[i].getExpressionType(),
        ProteusValue{key, context->createFalse()}};

    ProteusValue eq_v = eq(kexpr, kbuck).accept(exprGenerator);
    bucket_cond = Builder->CreateAnd(bucket_cond, eq_v.value);
  }
  // if (next[current].key == key) {
  Builder->CreateCondBr(bucket_cond, BucketFoundBB, ContFollowBB);

  Builder->SetInsertPoint(BucketFoundBB);

  // atomicAdd(&(next[current].sum), val);
  //
  // Low-cardinality group-bys (e.g. TPC-H Q1: ~6 groups over 591M rows) make
  // every lane hammer the same few payload cache lines with global atomics,
  // which serialize in L2 and dominate the pipeline (same pathology as the
  // GpuReduce per-lane epilogue, doc 17). Fast path: when the warp is fully
  // converged AND every lane matched the SAME bucket (`current` is
  // warp-uniform -- the common case for skewed/clustered keys), shfl-reduce
  // each aggregate across the warp and let lane 0 issue ONE atomic per
  // aggregate. Any divergence or mixed-bucket warp falls back to the original
  // per-lane atomics, which stay correct for every monoid.
  std::vector<gpu::Monoid *> agg_gms;
  std::vector<Value *> agg_vals;
  std::vector<Value *> agg_ptrs;
  for (size_t i = 0; i < agg_exprs.size(); ++i) {
    if (agg_exprs[i].is_aggregation()) {
      agg_gms.push_back(gpu::Monoid::get(agg_exprs[i].m));
      std::vector<Value *> tmp{current,
                               context->createInt32(agg_exprs[i].packind)};

      agg_vals.push_back(Builder->CreateExtractValue(
          out_vals[agg_exprs[i].packet], agg_exprs[i].packind));

      agg_ptrs.push_back(Builder->CreateInBoundsGEP(
          context->getStateVar(out_param_ids[agg_exprs[i].packet])
              ->getType()
              ->getNonOpaquePointerElementType(),
          context->getStateVar(out_param_ids[agg_exprs[i].packet]), tmp));
    }
  }

  {
    auto *int32_type = Type::getInt32Ty(llvmContext);
    Value *upd_mask = gpu_intrinsic::activemask(context);
    Value *full_warp = Builder->CreateICmpEQ(
        upd_mask, ConstantInt::get(int32_type, ~0u), "gb_full_warp");

    BasicBlock *checkUniBB =
        BasicBlock::Create(llvmContext, "gbCheckUniform", TheFunction);
    BasicBlock *warpAggBB =
        BasicBlock::Create(llvmContext, "gbWarpAgg", TheFunction);
    BasicBlock *leaderBB =
        BasicBlock::Create(llvmContext, "gbWarpLeader", TheFunction);
    BasicBlock *laneBB =
        BasicBlock::Create(llvmContext, "gbLaneAtomics", TheFunction);
    BasicBlock *updDoneBB =
        BasicBlock::Create(llvmContext, "gbUpdateDone", TheFunction);

    Builder->CreateCondBr(full_warp, checkUniBB, laneBB);

    // Full warp: is `current` (the matched bucket) identical on all lanes?
    // Butterfly equality over all 5 strides covers every edge of the 5-cube,
    // which is connected, so vote.all(eq) <=> all 32 lanes equal.
    Builder->SetInsertPoint(checkUniBB);
    Value *eq_all = ConstantInt::getTrue(llvmContext);
    for (int stride = 16; stride > 0; stride >>= 1) {
      Value *peer_cur = gpu_intrinsic::shfl_bfly(
          context, current, ConstantInt::get(int32_type, stride),
          ConstantInt::get(int32_type, 31));
      eq_all = Builder->CreateAnd(eq_all,
                                  Builder->CreateICmpEQ(current, peer_cur));
    }
    Value *uniform = gpu_intrinsic::all(context, eq_all);

    // Mixed-bucket full warp: when every aggregate is an integer SUM (the
    // common analytic case), loop over the distinct buckets present in the
    // warp (find the min remaining bucket via a butterfly-min, identity-pad
    // non-members, butterfly-sum, one atomic per bucket per aggregate from
    // the member leader). k distinct buckets cost ~k*(6+#aggs*5) shuffles and
    // #aggs*k atomics instead of #aggs*32 atomics -- for low-cardinality
    // group-bys this removes the L2 same-line atomic serialization that
    // otherwise dominates (TPC-H Q1: 3.4M atomic requests per launch, 38%
    // long-scoreboard stalls). Non-SUM or non-integer aggregates keep the
    // per-lane path.
    bool all_int_sum = true;
    for (const auto &ae : agg_exprs) {
      if (!ae.is_aggregation()) continue;
      if (ae.m != SUM) all_int_sum = false;
    }
    for (auto *v : agg_vals) {
      if (!v->getType()->isIntegerTy()) all_int_sum = false;
    }

    if (!all_int_sum) {
      Builder->CreateCondBr(uniform, warpAggBB, laneBB);
    } else {
      BasicBlock *mbEntryBB =
          BasicBlock::Create(llvmContext, "gbMultiBucketEntry", TheFunction);
      BasicBlock *mbLoopBB =
          BasicBlock::Create(llvmContext, "gbMultiBucketLoop", TheFunction);
      BasicBlock *mbLeaderBB =
          BasicBlock::Create(llvmContext, "gbMultiBucketLeader", TheFunction);
      BasicBlock *mbLatchBB =
          BasicBlock::Create(llvmContext, "gbMultiBucketLatch", TheFunction);

      Builder->CreateCondBr(uniform, warpAggBB, mbEntryBB);

      Builder->SetInsertPoint(mbEntryBB);
      Value *mb_laneid = Builder->CreateTrunc(context->laneId(), int32_type);
      Value *lane_bit = Builder->CreateShl(
          ConstantInt::get(int32_type, 1), mb_laneid, "gb_lane_bit");
      Builder->CreateBr(mbLoopBB);

      Builder->SetInsertPoint(mbLoopBB);
      llvm::PHINode *remaining =
          Builder->CreatePHI(int32_type, 2, "gb_remaining");
      remaining->addIncoming(ConstantInt::get(int32_type, ~0u), mbEntryBB);

      Value *in_rem = Builder->CreateICmpNE(
          Builder->CreateAnd(remaining, lane_bit),
          ConstantInt::get(int32_type, 0), "gb_in_remaining");
      // min bucket among the remaining lanes (INT_MAX-padded butterfly-min);
      // full-warp shfl.sync keeps the warp converged across iterations
      gpu::MinMonoid min_m;
      Value *cur_or_max = Builder->CreateSelect(
          in_rem, current, ConstantInt::get(int32_type, 0x7fffffff));
      Value *lead_bucket =
          min_m.createWarpAggregateToAll(context, cur_or_max);
      Value *member = Builder->CreateAnd(
          in_rem, Builder->CreateICmpEQ(current, lead_bucket), "gb_member");
      Value *member_ballot = gpu_intrinsic::ballot(context, member);

      std::vector<Value *> mb_sums;
      mb_sums.reserve(agg_vals.size());
      for (size_t i = 0; i < agg_vals.size(); ++i) {
        Value *contrib = Builder->CreateSelect(
            member, agg_vals[i],
            llvm::Constant::getNullValue(agg_vals[i]->getType()));
        mb_sums.push_back(
            agg_gms[i]->createWarpAggregateToAll(context, contrib));
      }

      Value *leader_lane = Builder->CreateIntrinsic(
          llvm::Intrinsic::cttz, {int32_type},
          {member_ballot, ConstantInt::getTrue(llvmContext)});
      Value *mb_is_leader = Builder->CreateICmpEQ(mb_laneid, leader_lane,
                                                  "gb_mb_leader");
      Builder->CreateCondBr(mb_is_leader, mbLeaderBB, mbLatchBB);

      Builder->SetInsertPoint(mbLeaderBB);
      for (size_t i = 0; i < agg_vals.size(); ++i) {
        agg_gms[i]->createAtomicUpdate(context, agg_ptrs[i], mb_sums[i],
                                       llvm::AtomicOrdering::Monotonic);
      }
      Builder->CreateBr(mbLatchBB);

      Builder->SetInsertPoint(mbLatchBB);
      Value *new_remaining = Builder->CreateAnd(
          remaining, Builder->CreateNot(member_ballot), "gb_new_remaining");
      remaining->addIncoming(new_remaining, mbLatchBB);
      Builder->CreateCondBr(
          Builder->CreateICmpEQ(new_remaining,
                                ConstantInt::get(int32_type, 0)),
          updDoneBB, mbLoopBB);
    }

    // Uniform full warp: one shfl-reduction + one atomic per aggregate.
    Builder->SetInsertPoint(warpAggBB);
    std::vector<Value *> warp_aggs;
    warp_aggs.reserve(agg_vals.size());
    for (size_t i = 0; i < agg_vals.size(); ++i) {
      warp_aggs.push_back(
          agg_gms[i]->createWarpAggregateToAll(context, agg_vals[i]));
    }
    Value *laneid = Builder->CreateTrunc(context->laneId(), int32_type);
    Value *is_leader = Builder->CreateICmpEQ(
        laneid, ConstantInt::get(int32_type, 0), "gb_warp_leader");
    Builder->CreateCondBr(is_leader, leaderBB, updDoneBB);

    Builder->SetInsertPoint(leaderBB);
    for (size_t i = 0; i < agg_vals.size(); ++i) {
      agg_gms[i]->createAtomicUpdate(context, agg_ptrs[i], warp_aggs[i],
                                     llvm::AtomicOrdering::Monotonic);
    }
    Builder->CreateBr(updDoneBB);

    // Divergent warp or mixed buckets: original per-lane atomics.
    Builder->SetInsertPoint(laneBB);
    for (size_t i = 0; i < agg_vals.size(); ++i) {
      agg_gms[i]->createAtomicUpdate(context, agg_ptrs[i], agg_vals[i],
                                     llvm::AtomicOrdering::Monotonic);
    }
    Builder->CreateBr(updDoneBB);

    Builder->SetInsertPoint(updDoneBB);
  }

  activemask = gpu_intrinsic::activemask(context);
  // If this thread inserted a provisional entry but later matched an existing
  // bucket, mark the provisional entry as invalid using self-index.
  // Scan path relies on this sentinel (`next[idx] == idx`) to skip invalid
  // rows.
  expressions::ProteusValueExpression writtenExpr{
      new BoolType(),
      ProteusValue{
          Builder->CreateLoad(mem_written->getType()->getPointerElementType(),
                              mem_written),
          context->createFalse()}};
  gen_if(writtenExpr, childState, context)([&] {
    Value *inv_ptr = Builder->CreateInBoundsGEP(
        context->getStateVar(out_param_ids[0])
            ->getType()
            ->getNonOpaquePointerElementType(),
        context->getStateVar(out_param_ids[0]),
        std::vector<Value *>{
            Builder->CreateLoad(mem_idx->getType()->getPointerElementType(),
                                mem_idx),
            context->createInt32(0)});

    context->workerScopedAtomicXchg(
        inv_ptr, Builder->CreateLoad(
                     mem_idx->getType()->getPointerElementType(), mem_idx));
    // Builder->CreateAlignedStore(str, , packet_widths[0]/8);
  });
  Builder->CreateCall(warpsync, {activemask});
  // break;

  Builder->CreateBr(MergeBB);

  Builder->SetInsertPoint(ContFollowBB);
  // current = next_bucket;
  Builder->CreateStore(next, mem_current);

  Value *chain_end_cond = Builder->CreateICmpEQ(next, eochain);

  BasicBlock *EndFoundBB =
      BasicBlock::Create(llvmContext, "BucketFound", TheFunction);

  Builder->CreateCondBr(chain_end_cond, EndFoundBB, ThenBB);

  Builder->SetInsertPoint(EndFoundBB);

  // BasicBlock *CreateBucketBB =
  //     BasicBlock::Create(llvmContext, "CreateBucket", TheFunction);
  // BasicBlock *ContLinkingBB =
  //     BasicBlock::Create(llvmContext, "ContLinking", TheFunction);

  expressions::ProteusValueExpression condExpr{
      new BoolType(),
      {Builder->CreateNot(Builder->CreateLoad(
           mem_written->getType()->getPointerElementType(), mem_written)),
       context->createFalse()}};

  activemask = gpu_intrinsic::activemask(context);
  // if (!written){
  gen_if(condExpr, childState, context)([&]() {
    // index
    Value *old_cnt = context->workerScopedAtomicAdd(
        out_cnt,
        ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1));
    Builder->CreateStore(old_cnt, mem_idx);

    // next[idx].sum  = val;
    // next[idx].key  = key;
    // next[idx].next =  -1;
    context->workerScopedAtomicXchg(
        Builder->CreateBitCast(
            Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0])
                                           ->getType()
                                           ->getNonOpaquePointerElementType(),
                                       context->getStateVar(out_param_ids[0]),
                                       old_cnt),
            PointerType::getInt32PtrTy(llvmContext)),
        eochain);

    for (size_t i = 1; i < out_param_ids.size(); ++i) {
      Value *out_ptr = context->getStateVar(out_param_ids[i]);

      Value *out_ptr_i = Builder->CreateInBoundsGEP(
          out_ptr->getType()->getNonOpaquePointerElementType(), out_ptr,
          old_cnt);
      Builder->CreateAlignedStore(out_vals[i], out_ptr_i,
                                  llvm::Align(packet_widths[i] / 8), true);
    }

    // written = true;
    Builder->CreateStore(v_true, mem_written);

    // __threadfence();
    // }
    context->workerScopedMembar();
  });
  Builder->CreateCall(warpsync, {activemask});

  // new_next = atomicCAS(&(next[current].next), -1, idx);
  std::vector<Value *> tmp{current, context->createInt32(0)};
  auto activemask3 = gpu_intrinsic::activemask(context);
  Value *new_next = Builder->CreateAtomicCmpXchg(
      Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0])
                                     ->getType()
                                     ->getNonOpaquePointerElementType(),
                                 context->getStateVar(out_param_ids[0]), tmp),
      eochain,
      Builder->CreateLoad(mem_idx->getType()->getPointerElementType(), mem_idx),
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(context->getSizeOf(eochain)),
#endif
      llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic);
  Builder->CreateCall(warpsync, {activemask3});
  context->workerScopedMembar();
  // current = new_next;
  Builder->CreateStore(Builder->CreateExtractValue(new_next, 0), mem_current);
  // if (new_next == ((uint32_t) -1)) break;
  // Value * valid_insert = Builder->CreateICmpEQ(new_next, eochain);
  Builder->CreateCondBr(Builder->CreateExtractValue(new_next, 1), MergeBB,
                        ThenBB);

  Builder->SetInsertPoint(MergeBB);
  // Same rationale as above: use the current active mask at the merge point.
  Builder->CreateCall(warpsync, {gpu_intrinsic::activemask(context)});
}

void GpuHashGroupByChained::open(Pipeline *pip) {
  size_t cnt_size = 2 * sizeof(int32_t);
  int32_t *cnt = (int32_t *)MemoryManager::mallocGpu(cnt_size);
  int32_t *first =
      (int32_t *)MemoryManager::mallocGpu(sizeof(int32_t) * (1 << hash_bits));
  std::vector<void *> next;

  for (const auto &w : packet_widths) {
    next.emplace_back(MemoryManager::mallocGpu((w / 8) * maxInputSize));
  }

  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaMemsetAsync(cnt, 0, cnt_size, strm));
  gpu_run(cudaMemsetAsync(first, -1, (1 << hash_bits) * sizeof(int32_t), strm));
  // gpu_run(cudaMemset(next[0], -1, (packet_widths[0]/8) * maxInputSize));

  pip->setStateVar<int32_t *>(cnt_param_id, cnt);
  pip->setStateVar<int32_t *>(head_param_id, first);

  for (size_t i = 0; i < out_param_ids.size(); ++i) {
    pip->setStateVar<void *>(out_param_ids[i], next[i]);
  }

  syncAndDestroyStream(strm);

  // std::cout << cnt << " " << get_device(cnt) << std::endl;
  // std::cout << first << " " << get_device(first) << std::endl;
  // std::cout << next[0] << " " << get_device(next[0]) << std::endl;
}

struct entry {
  int32_t index;
  // int32_t key0;
  // int32_t key1;
  // int32_t gb;
};

// struct keys{
//     // int32_t index;
//     int32_t key0;
//     int32_t key1;
//     // int32_t gb;
// };

void GpuHashGroupByChained::close(Pipeline *pip) {
  int32_t h_cnt = -1;
  gpu_run(cudaMemcpy(&h_cnt, pip->getStateVar<int32_t *>(cnt_param_id),
                     sizeof(int32_t), cudaMemcpyDefault));
  LOG_IF(INFO, h_cnt < 0.5 * maxInputSize || h_cnt >= maxInputSize)
      << "Actual build input size: " << h_cnt << " (capacity: " << maxInputSize
      << ")";
  assert(((size_t)h_cnt) <= maxInputSize &&
         "Build input sized exceeded given parameter");
  // for (int32_t i = 0 ; i < cnt ; ++i){
  //     if (h_next[i].index != i){
  //         std::cout << i << " " << h_next[i].index << std::endl;//" " <<
  //         h_next[i].key0 << " " << h_next[i].key1 << std::endl;
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

  // keys * h_keys;
  // gpu_run(cudaMallocHost((void **) &h_keys, sizeof(keys) * cnt));
  // gpu_run(cudaMemcpy(h_keys, pip->getStateVar<void *>(out_param_ids[1]), cnt
  // * (packet_widths[1]/8), cudaMemcpyDefault));

  // std::cout << "---+=+" << std::endl;
  // for (int32_t i = 0 ; i < cnt ; ++i){
  //     std::cout << i << " " << h_keys[i].key0 << " " << h_keys[i].key1 <<
  //     std::endl;
  // }

  // for (const auto &w: packet_widths) std::cout << "w" << w << " "; std::cout
  // << std::endl; std::cout << "---------------------------> " << cnt <<
  // std::endl;

  cudaStream_t strm = createNonBlockingStream();

  // void   ** buffs = pip->getStateVar<void   **>(buffVar_id[0]);
  // int32_t * cnts  = pip->getStateVar<int32_t *>(cntVar_id    );

  auto probe_pip = probe_gen->getPipeline(pip->getGroup());
  probe_pip->open(pip->getSession());

  std::vector<void *> args;
  args.push_back(pip->getStateVar<int32_t *>(cnt_param_id));
  for (const auto &params : out_param_ids) {
    args.push_back(pip->getStateVar<void *>(params));
  }

  std::vector<void **> kp;
  for (size_t i = 0; i < args.size(); ++i) {
    kp.push_back(args.data() + i);
  }
  kp.push_back((void **)probe_pip->getState());

  launch_kernel((CUfunction)probe_gen->getKernel(), (void **)kp.data(), strm);
  syncAndDestroyStream(strm);

  probe_pip->close();

  MemoryManager::freeGpu(pip->getStateVar<int32_t *>(cnt_param_id));
  MemoryManager::freeGpu(pip->getStateVar<int32_t *>(head_param_id));

  for (size_t i = 0; i < out_param_ids.size(); ++i) {
    MemoryManager::freeGpu(pip->getStateVar<void *>(out_param_ids[i]));
  }
}
