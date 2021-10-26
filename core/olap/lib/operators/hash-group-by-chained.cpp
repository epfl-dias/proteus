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

#include "hash-group-by-chained.hpp"

#include <platform/memory/memory-manager.hpp>
#include <platform/util/bitwise-ops.hpp>

#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/expressions-hasher.hpp"
#include "lib/operators/gpu/gmonoids.hpp"

using namespace llvm;

HashGroupByChained::HashGroupByChained(std::vector<GpuAggrMatExpr> agg_exprs,
                                       std::vector<expression_t> key_expr,
                                       Operator *child,

                                       int hash_bits,

                                       size_t maxInputSize, std::string opLabel)
    : UnaryOperator(child),
      agg_exprs(std::move(agg_exprs)),
      key_expr(std::move(key_expr)),
      hash_bits(hash_bits),
      maxInputSize(maxInputSize),
      opLabel(std::move(opLabel)) {}

void HashGroupByChained::produce_(ParallelContext *context) {
  prepareDescription(context);
  generate_scan(context);

  context->popPipeline();

  context->pushPipeline(context->removeLatestPipeline());

  buildHashTableFormat(context);

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

void HashGroupByChained::consume(ParallelContext *context,
                                 const OperatorState &childState) {
  generate_build(context, childState);
}

void HashGroupByChained::prepareDescription(ParallelContext *context) {
  agg_exprs.emplace_back(new expressions::IntConstant(0), ~((size_t)0), 0);

  size_t bitoffset = 0;
  for (const auto &key : key_expr) {
    agg_exprs.emplace_back(key, 0, bitoffset);

    const ExpressionType *out_type = key.getExpressionType();

    Type *llvm_type = out_type->getLLVMType(context->getLLVMContext());

    bitoffset += context->getSizeOf(llvm_type) * 8;
  }

  std::sort(agg_exprs.begin(), agg_exprs.end(),
            [](const GpuAggrMatExpr &a, const GpuAggrMatExpr &b) {
              return a.packet + 1 < b.packet + 1 ||
                     (a.packet == b.packet && a.bitoffset < b.bitoffset);
            });

  size_t i = 0;
  size_t p = 0;

  while (i < agg_exprs.size()) {
    size_t bindex = 0;
    size_t packind = 0;

    std::vector<Type *> body;
    while (i < agg_exprs.size() && agg_exprs[i].packet + 1 == p) {
      agg_exprs[i].packet++;
      if (agg_exprs[i].bitoffset != bindex) {
        // insert space
        assert(agg_exprs[i].bitoffset > bindex);
        body.push_back(Type::getIntNTy(context->getLLVMContext(),
                                       (agg_exprs[i].bitoffset - bindex)));
        ++packind;
      }

      const ExpressionType *out_type = agg_exprs[i].expr.getExpressionType();
      auto llvm_type = out_type->getLLVMType(context->getLLVMContext());

      if (agg_exprs[i].is_aggregation()) {
        gpu::Monoid *gm = gpu::Monoid::get(agg_exprs[i].m);
        llvm_type = gm->getStorageType(context, llvm_type);
      }

      body.push_back(llvm_type);
      bindex = agg_exprs[i].bitoffset + context->getSizeOf(llvm_type) * 8;
      agg_exprs[i].packind = packind++;
      ++i;
    }
    // assert(packet_widths[p] >= bindex);

    if (bindex & (bindex - 1)) {
      size_t v = next_power_of_2(bindex);
      body.push_back(Type::getIntNTy(context->getLLVMContext(), (v - bindex)));
      // if (packet_widths[p] > bindex) {
      //     body.push_back(Type::getIntNTy(context->getLLVMContext(),
      //     (packet_widths[p] - bindex)));
      // }
      bindex = v;
    }

    packet_widths.push_back(bindex);
    Type *t = StructType::create(body, opLabel + "_struct_" + std::to_string(p),
                                 true);
    Type *t_ptr = PointerType::get(t, /* address space */ 0);
    ptr_types.push_back(t_ptr);
    ++p;
  }
  assert(i == agg_exprs.size());

  agg_exprs.erase(agg_exprs.begin());  // erase dummy entry for next
}

void HashGroupByChained::buildHashTableFormat(ParallelContext *context) {
  for (const auto &t : ptr_types) {
    out_param_ids.push_back(context->appendStateVar(t));  //, true, false));
  }

  Type *int32_type = Type::getInt32Ty(context->getLLVMContext());

  Type *t_cnt = PointerType::get(int32_type, /* address space */ 0);
  cnt_param_id = context->appendStateVar(t_cnt);  //, true, false);

  auto t_head = ArrayType::get(int32_type, size_t{1} << hash_bits);
  auto t_head_ptr = PointerType::get(t_head, /* address space */ 0);
  head_param_id = context->appendStateVar(
      t_head_ptr,

      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();

        Value *mem_acc = context->allocateStateVar(t_head);

        context->CodegenMemset(mem_acc, context->createInt8(-1),
                               t_head->getArrayNumElements() * sizeof(int32_t));

        return mem_acc;
      },

      [=, this](llvm::Value *, llvm::Value *s) {
        std::vector<Value *> args;
        args.emplace_back(context->getStateVar(cnt_param_id));
        for (const auto &params : out_param_ids) {
          args.emplace_back(context->getStateVar(params));
        }

        IRBuilder<> *Builder = context->getBuilder();

        Function *f = context->getFunction("subpipeline_consume");
        FunctionType *f_t = f->getFunctionType();

        Type *substate_t = f_t->getParamType(f_t->getNumParams() - 1);

        Value *substate = Builder->CreateBitCast(
            ((ParallelContext *)context)->getSubStateVar(), substate_t);
        args.emplace_back(substate);

        Builder->CreateCall(f, args);
      });
}

Value *HashGroupByChained::hash(const std::vector<expression_t> &exprs,
                                ParallelContext *context,
                                const OperatorState &childState) {
  Value *hash;
  if (exprs.size() == 1) {
    ExpressionHasherVisitor hasher{context, childState};
    hash = exprs[0].accept(hasher).value;
  } else {
    std::list<expressions::AttributeConstruction> a;
    size_t i = 0;
    for (const auto &e : exprs) a.emplace_back("k" + std::to_string(i++), e);

    ExpressionHasherVisitor hasher{context, childState};
    hash = expressions::RecordConstruction{a}.accept(hasher).value;
  }
  auto size = ConstantInt::get(hash->getType(), (size_t(1) << hash_bits));
  return context->getBuilder()->CreateURem(hash, size);
}

std::vector<llvm::Value *> HashGroupByChained::prepareHashTableEntry(
    ParallelContext *context, const OperatorState &childState) const {
  Value *head_ptr = context->getStateVar(head_param_id);
  Value *eochain = ConstantInt::get((IntegerType *)head_ptr->getType()
                                        ->getPointerElementType()
                                        ->getArrayElementType(),
                                    ~((size_t)0));

  IRBuilder<> *Builder = context->getBuilder();

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

    if (mexpr.is_aggregation()) {
      gpu::Monoid *gm = gpu::Monoid::get(mexpr.m);
      valWrapper.value = gm->createUnary(context, valWrapper.value);
    }

    out_vals[mexpr.packet] = Builder->CreateInsertValue(
        out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
  }

  return out_vals;
}

void HashGroupByChained::destroyHashTableEntry(
    ParallelContext *context, const OperatorState &childState,
    std::vector<llvm::Value *> out_vals) const {
  IRBuilder<> *Builder = context->getBuilder();
  for (const GpuAggrMatExpr &mexpr : agg_exprs) {
    if (!mexpr.is_aggregation()) continue;
    auto v = Builder->CreateExtractValue(out_vals[mexpr.packet], mexpr.packind);
    gpu::Monoid *gm = gpu::Monoid::get(mexpr.m);
    gm->destroyUnary(context, v);
  }
}

void HashGroupByChained::generate_build(ParallelContext *context,
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

  Value *eochain = ConstantInt::get((IntegerType *)head_ptr->getType()
                                        ->getPointerElementType()
                                        ->getArrayElementType(),
                                    ~((size_t)0));
  // current = head[hash(key)]
  // Value * current =
  // Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash),
  // context->getSizeOf(head_ptr->getType()->getPointerElementType()));
  Value *head_w_hash_ptr =
      Builder->CreateInBoundsGEP(head_ptr, {context->createInt64(0), hash});
  // head_w_hash_ptr->dump();
  // head_w_hash_ptr->getType()->dump();
  // Value * current =
  // Builder->CreateExtractValue(Builder->CreateAtomicCmpXchg(head_w_hash_ptr,
  //                                             eochain,
  //                                             eochain,
  //                                             llvm::AtomicOrdering::Monotonic,
  //                                             llvm::AtomicOrdering::Monotonic),
  //                                             0);
  Value *current = Builder->CreateLoad(head_w_hash_ptr);

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

  // BasicBlock *InitCondBB  = BasicBlock::Create(llvmContext, "setHeadCond",
  // TheFunction);
  BasicBlock *InitThenBB =
      BasicBlock::Create(llvmContext, "setHead", TheFunction);
  BasicBlock *InitMergeBB =
      BasicBlock::Create(llvmContext, "cont", TheFunction);

  BasicBlock *ThenBB =
      BasicBlock::Create(llvmContext, "chainFollow", TheFunction);
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont", TheFunction);

  Value *init_cond =
      Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

  // if (current == ((uint32_t) -1)){
  Builder->CreateCondBr(init_cond, InitThenBB, InitMergeBB);

  Builder->SetInsertPoint(InitThenBB);

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
  {
    auto out_vals = prepareHashTableEntry(context, childState);

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
    Value *suc =
        Builder->CreateICmpEQ(Builder->CreateLoad(mem_current), eochain);

    Builder->CreateCondBr(suc, MergeBB, InitMergeBB);
    //   if (!suc) goto MergeBB
  }
  // }
  // Builder->CreateBr(InitMergeBB);
  Builder->SetInsertPoint(InitMergeBB);

  // if (current != ((uint32_t) -1)){
  //     while (true) {

  Value *chain_cond =
      Builder->CreateICmpNE(Builder->CreateLoad(mem_current), eochain);

  Builder->CreateCondBr(chain_cond, ThenBB, MergeBB);

  Builder->SetInsertPoint(ThenBB);
  current = Builder->CreateLoad(mem_current);

  context->workerScopedMembar();

  // Keys
  Value *next_bucket_ptr = Builder->CreateInBoundsGEP(
      context->getStateVar(out_param_ids[1]), current);
  Value *next_bucket = Builder->CreateAlignedLoad(
      next_bucket_ptr, packet_widths[1] / 8, true, "next_bucket");

  context->workerScopedMembar();

  // Value * next =
  // Builder->CreateExtractValue(Builder->CreateAtomicCmpXchg(Builder->CreateInBoundsGEP(((const
  // ParallelContext *) context)->getStateVar(out_param_ids[0]),
  // std::vector<Value
  // *>{current, context->createInt32(0)}),
  //                                             eochain,
  //                                             eochain,
  //                                             llvm::AtomicOrdering::Monotonic,
  //                                             llvm::AtomicOrdering::Monotonic),
  //                                             0);
  Value *next = Builder->CreateLoad(
      Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]),
                                 {current, context->createInt32(0)}));

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
  {
    for (auto &agg_expr : agg_exprs) {
      if (agg_expr.is_aggregation()) {
        gpu::Monoid *gm = gpu::Monoid::get(agg_expr.m);
        std::vector<Value *> tmp{current,
                                 context->createInt32(agg_expr.packind)};

        Value *gl_accum = Builder->CreateInBoundsGEP(
            context->getStateVar(out_param_ids[agg_expr.packet]), tmp);

        ExpressionGeneratorVisitor exprGenerator(context, childState);
        ProteusValue valWrapper = agg_expr.expr.accept(exprGenerator);
        gm->createUpdateFromBareValue(context, gl_accum, valWrapper.value);
      }
    }
  }

  // if (written) next[idx].next = idx;
  context->gen_if({Builder->CreateLoad(mem_written)})([&]() {
    // Value * str = UndefValue::get(((const ParallelContext *)
    // context)->getStateVar(out_param_ids[0])->getType()->getPointerElementType());
    // str = Builder->CreateInsertValue(str, Builder->CreateLoad(mem_idx), 0);
    Value *inv_ptr = Builder->CreateInBoundsGEP(
        context->getStateVar(out_param_ids[0]),
        {Builder->CreateLoad(mem_idx), context->createInt32(0)});

    context->workerScopedAtomicXchg(inv_ptr, Builder->CreateLoad(mem_idx));
    // Builder->CreateAlignedStore(str, , packet_widths[0]/8);

    {
      std::vector<llvm::Value *> out_vals;
      out_vals.reserve(out_param_ids.size());
      out_vals.emplace_back(nullptr);
      for (size_t i = 1; i < out_param_ids.size(); ++i) {
        Value *out_ptr = context->getStateVar(out_param_ids[i]);

        Value *out_ptr_i =
            Builder->CreateInBoundsGEP(out_ptr, Builder->CreateLoad(mem_idx));
        out_vals.emplace_back(
            Builder->CreateAlignedLoad(out_ptr_i, packet_widths[i] / 8));
      }
      destroyHashTableEntry(context, childState, std::move(out_vals));
    }
  });

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

  // if (!written){
  context->gen_if(
      {Builder->CreateNot(Builder->CreateLoad(mem_written))})([&]() {
    // index
    old_cnt = context->workerScopedAtomicAdd(
        out_cnt,
        ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1));
    Builder->CreateStore(old_cnt, mem_idx);

    // next[idx].sum  = val;
    // next[idx].key  = key;
    // next[idx].next =  -1;
    // Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg,
    Builder->CreateStore(
        eochain, Builder->CreateBitCast(
                     Builder->CreateInBoundsGEP(
                         context->getStateVar(out_param_ids[0]), old_cnt),
                     PointerType::getInt32PtrTy(
                         llvmContext)));  //,
                                          // eochain,
                                          // llvm::AtomicOrdering::Monotonic);

    auto out_vals = prepareHashTableEntry(context, childState);
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

  // new_next = atomicCAS(&(next[current].next), -1, idx);
  std::vector<Value *> tmp{current, context->createInt32(0)};
  Value *n_ptr =
      Builder->CreateInBoundsGEP(context->getStateVar(out_param_ids[0]), tmp);
  Value *new_next = Builder->CreateLoad(n_ptr);
  // Value * new_next = Builder->CreateAtomicCmpXchg(n_ptr,
  //                                             eochain,
  //                                             Builder->CreateLoad(mem_idx),
  //                                             llvm::AtomicOrdering::Monotonic,
  //                                             llvm::AtomicOrdering::Monotonic);

  context->gen_if({Builder->CreateICmpEQ(new_next, eochain)})(
      [&]() { Builder->CreateStore(Builder->CreateLoad(mem_idx), n_ptr); });

  context->workerScopedMembar();
  // current = new_next;
  Builder->CreateStore(new_next, mem_current);
  // if (new_next == ((uint32_t) -1))
  // Value * valid_insert = Builder->CreateICmpEQ(new_next, eochain);
  Builder->CreateCondBr(Builder->CreateICmpEQ(new_next, eochain), MergeBB,
                        ThenBB);

  Builder->SetInsertPoint(MergeBB);
}

static std::mutex garbage_m;
static std::map<std::pair<void *, int32_t>, std::vector<void *>> garbage;

void HashGroupByChained::generate_scan(ParallelContext *context) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();

  Type *int32_type = Type::getInt32Ty(llvmContext);

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> variableBindings;

  Type *t_cnt = PointerType::get(int32_type, /* address space */ 0);
  size_t cnt_ptr_param = context->appendParameter(t_cnt, true, true);

  vector<size_t> out_param_ids_scan;
  for (const auto &p : ptr_types) {
    out_param_ids_scan.push_back(context->appendParameter(p, true, true));
  }

  context->registerClose(this, [this](Pipeline *pip) {
    std::lock_guard<std::mutex> lock{garbage_m};
    for (const auto &ptr : garbage[std::make_pair(this, pip->getGroup())]) {
      MemoryManager::freePinned(ptr);
    }
  });

  context->setGlobalFunction();

  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();

  // // Create the "AFTER LOOP" block and insert it.
  // BasicBlock *releaseBB = BasicBlock::Create(llvmContext, "releaseIf", F);
  // BasicBlock *rlAfterBB = BasicBlock::Create(llvmContext, "releaseEnd" , F);
  //  Value *is_leader =
  //      Builder->CreateICmpEQ(tId, ConstantInt::get(tId->getType(), 0));

  // Get the ENTRY BLOCK
  // context->setCurrentEntryBlock(Builder->GetInsertBlock());

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanBlkCond", F);

  // // Start insertion in CondBB.
  // Builder->SetInsertPoint(CondBB);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBlkBody", F);

  // Make the new basic block for the increment, inserting after current block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanBlkInc", F);

  // Create the "AFTER LOOP" block and insert it.
  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanBlkEnd", F);
  context->setEndingBlock(AfterBB);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  // Builder->CreateBr      (CondBB);

  std::string relName = agg_exprs[0].expr.getRegisteredRelName();
  Plugin *pg = Catalog::getInstance().getPlugin(relName);

  AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(
      F, "i_ptr", pg->getOIDType()->getLLVMType(llvmContext));
  Builder->CreateStore(
      Builder->CreateIntCast(context->threadId(),
                             pg->getOIDType()->getLLVMType(llvmContext), false),
      mem_itemCtr);

  RecordAttribute tupleCnt{relName, "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?
  Value *cnt = Builder->CreateLoad(context->getArgument(cnt_ptr_param), "cnt");
  if (pg->getOIDType()->getLLVMType(llvmContext)->isIntegerTy()) {
    cnt = Builder->CreateZExt(cnt, pg->getOIDType()->getLLVMType(llvmContext));
  }

  ProteusValueMemory mem_cntWrapper =
      context->toMem(cnt, context->createFalse());
  variableBindings[tupleCnt] = mem_cntWrapper;

  // Function * f = context->getFunction("devprinti64");
  // Builder->CreateCall(f, std::vector<Value *>{cnt});

  // Builder->CreateBr      (CondBB);
  Builder->SetInsertPoint(CondBB);

  /**
   * Equivalent:
   * while(itemCtr < size)
   */
  Value *lhs = Builder->CreateLoad(mem_itemCtr, "i");
  Value *cond =
      Builder->CreateICmpSLT(lhs, Builder->CreateZExt(cnt, lhs->getType()));

  // Insert the conditional branch into the end of CondBB.
  BranchInst *loop_cond = Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // NamedMDNode * annot =
  // context->getModule()->getOrInsertNamedMetadata("nvvm.annotations");
  // MDString    * str   = MDString::get(TheContext, "kernel");
  // Value       * one   = ConstantInt::get(int32Type, 1);

  MDNode *LoopID;

  {
    // MDString       * vec_st   = MDString::get(llvmContext,
    // "llvm.loop.vectorize.enable"); Type           * int1Type =
    // Type::getInt1Ty(llvmContext); Metadata       * one      =
    // ConstantAsMetadata::get(ConstantInt::get(int1Type, 1)); llvm::Metadata *
    // vec_en[] = {vec_st, one}; MDNode * vectorize_enable =
    // MDNode::get(llvmContext, vec_en);

    // MDString       * itr_st   = MDString::get(llvmContext,
    // "llvm.loop.interleave.count"); Type           * int32Type=
    // Type::getInt32Ty(llvmContext); Metadata       * count    =
    // ConstantAsMetadata::get(ConstantInt::get(int32Type, 4)); llvm::Metadata *
    // itr_en[] = {itr_st, count}; MDNode * interleave_count =
    // MDNode::get(llvmContext, itr_en);

    llvm::Metadata *Args[] = {
        nullptr};  //, vectorize_enable, interleave_count};
    LoopID = MDNode::get(llvmContext, Args);
    LoopID->replaceOperandWith(0, LoopID);

    loop_cond->setMetadata(LLVMContext::MD_loop, LoopID);
  }
  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Get the 'oid' of each record and pass it along.
  // More general/lazy plugins will only perform this action,
  // instead of eagerly 'converting' fields
  // FIXME This action corresponds to materializing the oid. Do we want this?
  RecordAttribute tupleIdentifier{relName, activeLoop, pg->getOIDType()};

  ProteusValueMemory mem_posWrapper;
  mem_posWrapper.mem = mem_itemCtr;
  mem_posWrapper.isNull = context->createFalse();
  variableBindings[tupleIdentifier] = mem_posWrapper;

  // Actual Work (Loop through attributes etc.)

  vector<Value *> in_vals;
  for (size_t i = 0; i < out_param_ids_scan.size(); ++i) {
    Value *out_ptr = context->getArgument(out_param_ids_scan[i]);

    Value *out_ptr_i = Builder->CreateInBoundsGEP(out_ptr, lhs);
    Value *val = Builder->CreateAlignedLoad(out_ptr_i, packet_widths[i] / 8);

    in_vals.push_back(val);
  }

  Value *next = Builder->CreateExtractValue(in_vals[0], 0);

  BasicBlock *groupProcessBB = BasicBlock::Create(llvmContext, "releaseIf", F);

  Value *isGroup =
      Builder->CreateICmpNE(Builder->CreateZExt(next, lhs->getType()), lhs);
  Builder->CreateCondBr(isGroup, groupProcessBB, IncBB);

  Builder->SetInsertPoint(groupProcessBB);

  for (const GpuAggrMatExpr &mexpr : agg_exprs) {
    Value *v =
        Builder->CreateExtractValue(in_vals[mexpr.packet], mexpr.packind);
    AllocaInst *v_mem = context->CreateEntryBlockAlloca(
        mexpr.expr.getRegisteredAttrName(), v->getType());
    Builder->CreateStore(v, v_mem);

    ProteusValueMemory val_mem;
    val_mem.mem = v_mem;
    val_mem.isNull = context->createFalse();

    variableBindings[mexpr.expr.getRegisteredAs()] = val_mem;
  }

  // Triggering parent
  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);

  // Start insertion in IncBB.
  Builder->SetInsertPoint(IncBB);

  // Increment and store back
  Value *val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);

  Value *inc = Builder->CreateIntCast(context->threadNum(),
                                      val_curr_itemCtr->getType(), false);

  Value *val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, inc);
  Builder->CreateStore(val_new_itemCtr, mem_itemCtr);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(CondBB);

  //  Finish up with end (the AfterLoop)
  //  Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());

  // ((ParallelContext *) context)->registerOpen (this, [this](Pipeline *
  // pip){this->open (pip);});
  // ((ParallelContext *) context)->registerClose(this, [this](Pipeline *
  // pip){this->close(pip);});
}

void HashGroupByChained::open(Pipeline *pip) {
  size_t cnt_size = sizeof(int64_t);
  auto *cnt = (int32_t *)MemoryManager::mallocPinned(cnt_size);
  // int32_t * first = (int32_t *) MemoryManager::mallocPinned(sizeof(int32_t
  // ) * (1 << hash_bits));
  std::vector<void *> next;

  for (const auto &w : packet_widths) {
    next.emplace_back(MemoryManager::mallocPinned((w / 8) * maxInputSize * 8));
  }
  // gpu_run(cudaMemset(next[0], -1, (packet_widths[0]/8) * maxInputSize));

  memset(cnt, 0, cnt_size);
  // memset(first, -1, (1 << hash_bits) * sizeof(int32_t));

  pip->setStateVar<int32_t *>(cnt_param_id, cnt);
  // pip->setStateVar<int32_t  *>(head_param_id, first);

  for (size_t i = 0; i < out_param_ids.size(); ++i) {
    pip->setStateVar<void *>(out_param_ids[i], next[i]);
  }

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

void HashGroupByChained::close(Pipeline *pip) {
  auto *cnt_ptr = pip->getStateVar<int32_t *>(cnt_param_id);
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
  int32_t h_cnt = *cnt_ptr;

  LOG_IF(INFO, h_cnt < 0.5 * maxInputSize || h_cnt > maxInputSize)
      << "Actual build input size: " << h_cnt << " (capacity: " << maxInputSize
      << ")";

  assert(h_cnt <= maxInputSize);

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

  // cudaStream_t strm = createNonBlockingStream();

  // execution_conf ec = pip->getExecConfiguration();
  // size_t grid_size = ec.gridSize();

  // void   ** buffs = pip->getStateVar<void   **>(buffVar_id[0]);
  // int32_t * cnts  = pip->getStateVar<int32_t *>(cntVar_id    );

  // Pipeline *probe_pip = probe_gen->getPipeline(pip->getGroup());
  // probe_pip->open();

  std::vector<void *> args;
  args.push_back(pip->getStateVar<int32_t *>(cnt_param_id));
  for (const auto &params : out_param_ids) {
    args.push_back(pip->getStateVar<void *>(params));
  }

  // std::vector<void **> kp;
  // for (size_t i = 0; i < args.size(); ++i) {
  //   kp.push_back(args.data() + i);
  // }
  // kp.push_back((void **)probe_pip->getState());

  // launch_kernel((CUfunction)probe_gen->getKernel(), (void **)kp.data(),
  // strm); syncAndDestroyStream(strm);

  // probe_pip->close();

  // FIXME: should get them from the parameters of the scan kernel
  std::lock_guard<std::mutex> lock{garbage_m};
  auto &v = garbage[std::make_pair(this, pip->getGroup())];
  v.clear();

  v.emplace_back(pip->getStateVar<int32_t *>(cnt_param_id));
  v.emplace_back(pip->getStateVar<int32_t *>(head_param_id));

  assert(out_param_ids.size() == packet_widths.size());
  for (auto &out_param_id : out_param_ids) {
    v.emplace_back(pip->getStateVar<void *>(out_param_id));
  }
}
