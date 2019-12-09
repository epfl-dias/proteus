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

#include "operators/gpu/gpu-hash-group-by-chained.hpp"

#include "expressions/expressions-generator.hpp"
#include "memory/memory-manager.hpp"
#include "operators/gpu/gmonoids.hpp"
#include "topology/topology.hpp"
#include "util/gpu/gpu-intrinsics.hpp"

using namespace llvm;

void GpuHashGroupByChained::produce() {
  context->pushPipeline();

  buildHashTableFormat();

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce();

  context->popPipeline();

  probe_gen = context->getCurrentPipeline();
  generate_scan();
}

void GpuHashGroupByChained::buildHashTableFormat() {
  prepareDescription();
  for (const auto &t : ptr_types) {
    out_param_ids.push_back(context->appendStateVar(t));  //, true, false));
  }

  Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

  Type *t_cnt = PointerType::get(int32_type, /* address space */ 0);
  cnt_param_id = context->appendStateVar(t_cnt);  //, true, false);

  Type *t_head_ptr = PointerType::get(int32_type, /* address space */ 0);
  head_param_id = context->appendStateVar(t_head_ptr);
}

void GpuHashGroupByChained::generate_build(ParallelContext *const context,
                                           const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  auto activemask_all = gpu_intrinsic::activemask(context);
  // PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
  // Type *int8_type = Type::getInt8Ty(llvmContext);
  // PointerType *void_ptr_type = PointerType::get(int8_type, 0);
  // Type *int64_type = Type::getInt64Ty(llvmContext);
  // Type *int32_type = Type::getInt32Ty(llvmContext);
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
  Value *head_w_hash_ptr = Builder->CreateInBoundsGEP(head_ptr, hash);
  Value *current = Builder->CreateExtractValue(
      Builder->CreateAtomicCmpXchg(head_w_hash_ptr, eochain, eochain,
                                   llvm::AtomicOrdering::Monotonic,
                                   llvm::AtomicOrdering::Monotonic),
      0);

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
    // out_ptrs.push_back(out_ptr);

    // out_ptr->addAttr(Attribute::getWithAlignment(llvmContext,
    // context->getSizeOf(out_ptr)));

    // out_ptrs.push_back(Builder->CreateInBoundsGEP(out_ptr, old_cnt));
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

  // BasicBlock *InitCondBB  = BasicBlock::Create(llvmContext, "setHeadCond",
  // TheFunction);
  // BasicBlock *InitThenBB =
  //     BasicBlock::Create(llvmContext, "setHead", TheFunction);
  // BasicBlock *InitMergeBB =
  //     BasicBlock::Create(llvmContext, "cont", TheFunction);

  BasicBlock *ThenBB =
      BasicBlock::Create(llvmContext, "chainFollow", TheFunction);
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont", TheFunction);

  Value *init_cond =
      Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

  // if (current == ((uint32_t) -1)){

  expressions::ProteusValueExpression initCondExpr{
      new BoolType(), ProteusValue{init_cond, context->createFalse()}};

  auto activemask = gpu_intrinsic::activemask(context);
  context->gen_if(initCondExpr, childState)([&]() {
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
            Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]),
                                       old_cnt),
            PointerType::getInt32PtrTy(llvmContext)),
        eochain);

    for (size_t i = 1; i < out_param_ids.size(); ++i) {
      Value *out_ptr = context->getStateVar(out_param_ids[i]);

      Value *out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
      Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i] / 8,
                                  true);
    }

    // written = true;
    Builder->CreateStore(v_true, mem_written);

    context->workerScopedMembar();

    // current = atomicCAS(&(first[bucket]), -1, idx);
    Value *old_current = Builder->CreateAtomicCmpXchg(
        head_w_hash_ptr, eochain, old_cnt, llvm::AtomicOrdering::Monotonic,
        llvm::AtomicOrdering::Monotonic);

    Builder->CreateStore(Builder->CreateExtractValue(old_current, 0),
                         mem_current);
  });
  auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  Builder->CreateCall(warpsync, {activemask});

  Value *gactivemask = activemask;

  // if (current != ((uint32_t) -1)){
  //     while (true) {

  Value *chain_cond =
      Builder->CreateICmpNE(Builder->CreateLoad(mem_current), eochain);

  Builder->CreateCondBr(chain_cond, ThenBB, MergeBB);

  Builder->SetInsertPoint(MergeBB);
  Builder->CreateCall(warpsync, {gactivemask});

  Builder->SetInsertPoint(ThenBB);
  current = Builder->CreateLoad(mem_current);

  context->workerScopedMembar();

  // Keys
  Value *next_bucket_ptr = Builder->CreateInBoundsGEP(
      context->getStateVar(out_param_ids[1]), current);
  Value *next_bucket = Builder->CreateAlignedLoad(
      next_bucket_ptr, packet_widths[1] / 8, true, "next_bucket");

  context->workerScopedMembar();

  Value *next = Builder->CreateExtractValue(
      Builder->CreateAtomicCmpXchg(
          Builder->CreateInBoundsGEP(
              context->getStateVar(out_param_ids[0]),
              std::vector<Value *>{current, context->createInt32(0)}),
          eochain, eochain, llvm::AtomicOrdering::Monotonic,
          llvm::AtomicOrdering::Monotonic),
      0);

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
  for (size_t i = 0; i < agg_exprs.size(); ++i) {
    if (agg_exprs[i].is_aggregation()) {
      gpu::Monoid *gm = gpu::Monoid::get(agg_exprs[i].m);
      std::vector<Value *> tmp{current,
                               context->createInt32(agg_exprs[i].packind)};

      Value *aggr = Builder->CreateExtractValue(out_vals[agg_exprs[i].packet],
                                                agg_exprs[i].packind);

      Value *gl_accum = Builder->CreateInBoundsGEP(
          context->getStateVar(out_param_ids[agg_exprs[i].packet]), tmp);

      gm->createAtomicUpdate(context, gl_accum, aggr,
                             llvm::AtomicOrdering::Monotonic);
    }
  }

  activemask = gpu_intrinsic::activemask(context);
  // if (written) next[idx].next = idx;
  expressions::ProteusValueExpression writtenExpr{
      new BoolType(),
      ProteusValue{Builder->CreateLoad(mem_written), context->createFalse()}};
  context->gen_if(writtenExpr, childState)([&] {
    // Value * str = UndefValue::get(((const ParallelContext *)
    // context)->getStateVar(out_param_ids[0])->getType()->getPointerElementType());
    // str = Builder->CreateInsertValue(str, Builder->CreateLoad(mem_idx), 0);
    Value *inv_ptr = Builder->CreateInBoundsGEP(
        context->getStateVar(out_param_ids[0]),
        std::vector<Value *>{Builder->CreateLoad(mem_idx),
                             context->createInt32(0)});

    context->workerScopedAtomicXchg(inv_ptr, Builder->CreateLoad(mem_idx));
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
      {Builder->CreateNot(Builder->CreateLoad(mem_written)),
       context->createFalse()}};

  activemask = gpu_intrinsic::activemask(context);
  // if (!written){
  context->gen_if(condExpr, childState)([&]() {
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
            Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]),
                                       old_cnt),
            PointerType::getInt32PtrTy(llvmContext)),
        eochain);

    for (size_t i = 1; i < out_param_ids.size(); ++i) {
      Value *out_ptr = context->getStateVar(out_param_ids[i]);

      Value *out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, old_cnt);
      Builder->CreateAlignedStore(out_vals[i], out_ptr_i, packet_widths[i] / 8,
                                  true);
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
      Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]), tmp),
      eochain, Builder->CreateLoad(mem_idx), llvm::AtomicOrdering::Monotonic,
      llvm::AtomicOrdering::Monotonic);
  Builder->CreateCall(warpsync, {activemask3});
  context->workerScopedMembar();
  // current = new_next;
  Builder->CreateStore(Builder->CreateExtractValue(new_next, 0), mem_current);
  // if (new_next == ((uint32_t) -1)) break;
  // Value * valid_insert = Builder->CreateICmpEQ(new_next, eochain);
  Builder->CreateCondBr(Builder->CreateExtractValue(new_next, 1), MergeBB,
                        ThenBB);

  Builder->SetInsertPoint(MergeBB);
  Builder->CreateCall(warpsync, {activemask_all});
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
  // int32_t * cnt_ptr = pip->getStateVar<int32_t  *>(cnt_param_id);
  // entry * h_next;
  // int32_t * h_first;
  // int32_t cnt;
  // std::cout << packet_widths[0]/8 << " " << sizeof(entry) << std::endl;
  // assert(packet_widths[0]/8 == sizeof(entry));
  // size_t size = (packet_widths[0]/8) * maxInputSize;
  // gpu_run(cudaMallocHost((void **) &h_next , size));
  // gpu_run(cudaMallocHost((void **) &h_first, sizeof(int32_t  ) * (1 <<
  // hash_bits))); gpu_run(cudaMemcpy(&cnt  , cnt_ptr, sizeof(int32_t),
  // cudaMemcpyDefault)); std::cout << "---------------------------> " << cnt <<
  // " " << maxInputSize << std::endl; gpu_run(cudaMemcpy(h_next,
  // pip->getStateVar<void *>(out_param_ids[0]), cnt * (packet_widths[0]/8),
  // cudaMemcpyDefault)); gpu_run(cudaMemcpy(h_first, pip->getStateVar<void
  // *>(head_param_id), sizeof(int32_t  ) * (1 << hash_bits),
  // cudaMemcpyDefault));

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

  execution_conf ec = pip->getExecConfiguration();

  // void   ** buffs = pip->getStateVar<void   **>(buffVar_id[0]);
  // int32_t * cnts  = pip->getStateVar<int32_t *>(cntVar_id    );

  Pipeline *probe_pip = probe_gen->getPipeline(pip->getGroup());
  probe_pip->open();

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
