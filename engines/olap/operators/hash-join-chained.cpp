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

#include "operators/hash-join-chained.hpp"

#include "codegen/memory/memory-manager.hpp"
#include "expressions/expressions-hasher.hpp"
#include "operators/gpu/gmonoids.hpp"

using namespace llvm;

HashJoinChained::HashJoinChained(const std::vector<GpuMatExpr> &build_mat_exprs,
                                 const std::vector<size_t> &build_packet_widths,
                                 expression_t build_keyexpr,
                                 Operator *const build_child,

                                 const std::vector<GpuMatExpr> &probe_mat_exprs,
                                 const std::vector<size_t> &probe_packet_widths,
                                 expression_t probe_keyexpr,
                                 Operator *const probe_child,

                                 int hash_bits,

                                 ParallelContext *context,
                                 size_t maxBuildInputSize, string opLabel)
    : BinaryOperator(build_child, probe_child),
      build_mat_exprs(build_mat_exprs),
      probe_mat_exprs(probe_mat_exprs),
      build_packet_widths(build_packet_widths),
      build_keyexpr(std::move(build_keyexpr)),
      probe_keyexpr(std::move(probe_keyexpr)),
      hash_bits(hash_bits),
      maxBuildInputSize(maxBuildInputSize),
      context(context),
      opLabel(opLabel) {}

void HashJoinChained::produce() {
  context->pushPipeline();  // FIXME: find a better way to do this
  buildHashTableFormat();

  context->registerOpen(this, [this](Pipeline *pip) { this->open_build(pip); });
  context->registerClose(this,
                         [this](Pipeline *pip) { this->close_build(pip); });
  getLeftChild()->produce();

  // context->compileAndLoad(); //FIXME: Remove!!!! causes an extra compilation!
  // this compile will be done again later! Get kernel function probe_kernel =
  // context->getKernel();
  context->popPipeline();  // FIXME: find a better way to do this

  probeHashTableFormat();

  context->registerOpen(this, [this](Pipeline *pip) { this->open_probe(pip); });
  context->registerClose(this,
                         [this](Pipeline *pip) { this->close_probe(pip); });
  getRightChild()->produce();
}

void HashJoinChained::consume(Context *const context,
                              const OperatorState &childState) {
  ParallelContext *const ctx = dynamic_cast<ParallelContext *const>(context);
  assert(ctx);
  consume(ctx, childState);
}

void HashJoinChained::consume(ParallelContext *const context,
                              const OperatorState &childState) {
  const Operator &caller = childState.getProducer();

  if (caller == *(getLeftChild())) {
    generate_build(context, childState);
  } else {
    generate_probe(context, childState);
  }
}

void HashJoinChained::probeHashTableFormat() {
  // assumes than build has already run

  Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
  Type *t_head_ptr =
      PointerType::getUnqual(int32_type);  //, /* address space */ 1);
  probe_head_param_id = context->appendStateVar(t_head_ptr);  //, true, true);

  size_t i = 0;

  for (size_t p = 0; p < build_packet_widths.size(); ++p) {
    // Type * t     =
    // PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(),
    // build_packet_widths[p]), /* address space */ 1);

    size_t bindex = 0;
    size_t packind = 0;

    std::vector<Type *> body;
    while (i < build_mat_exprs.size() && build_mat_exprs[i].packet == p) {
      if (build_mat_exprs[i].bitoffset != bindex) {
        // insert space
        assert(build_mat_exprs[i].bitoffset > bindex);
        body.push_back(
            Type::getIntNTy(context->getLLVMContext(),
                            (build_mat_exprs[i].bitoffset - bindex)));
        ++packind;
      }

      auto out_type = build_mat_exprs[i].expr.getExpressionType();

      Type *llvm_type = out_type->getLLVMType(context->getLLVMContext());

      body.push_back(llvm_type);
      bindex = build_mat_exprs[i].bitoffset + context->getSizeOf(llvm_type) * 8;
      build_mat_exprs[i].packind = packind++;
      ++i;
    }
    assert(build_packet_widths[p] >= bindex);

    if (build_packet_widths[p] > bindex) {
      body.push_back(Type::getIntNTy(context->getLLVMContext(),
                                     (build_packet_widths[p] - bindex)));
    }

    Type *t = StructType::create(body, opLabel + "_struct_" + std::to_string(p),
                                 true);
    Type *t_ptr = PointerType::getUnqual(t);  //, /* address space */ 1);

    in_param_ids.push_back(context->appendStateVar(t_ptr));  //, true, true));
  }
  assert(i == build_mat_exprs.size());

  // build_mat_exprs.erase(build_mat_exprs.begin()); //erase dummy entry for
  // next

  // Type * t     = PointerType::get(((const PrimitiveType *)
  // out_type)->getLLVMType(context->getLLVMContext()), /* address space */ 1);
  // out_param_id = context->appendParameter(t    , true, false);

  // Type * t_cnt = PointerType::get(int32_type, /* address space */ 1);
  // cnt_param_id = context->appendParameter(t_cnt, true, false);
}

void HashJoinChained::buildHashTableFormat() {
  build_mat_exprs.emplace_back(new expressions::IntConstant(0), 0, 0);
  build_mat_exprs.emplace_back(build_keyexpr, 0, 32);

  std::sort(build_mat_exprs.begin(), build_mat_exprs.end(),
            [](const GpuMatExpr &a, const GpuMatExpr &b) {
              if (a.packet == b.packet) return a.bitoffset < b.bitoffset;
              return a.packet < b.packet;
            });

  Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
  Type *t_head_ptr =
      PointerType::getUnqual(int32_type);  //, /* address space */ 1);
  head_param_id = context->appendStateVar(t_head_ptr);  //, true, false);

  size_t i = 0;

  for (size_t p = 0; p < build_packet_widths.size(); ++p) {
    // Type * t     =
    // PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(),
    // build_packet_widths[p]), /* address space */ 1);

    size_t bindex = 0;
    size_t packind = 0;

    std::vector<Type *> body;
    while (i < build_mat_exprs.size() && build_mat_exprs[i].packet == p) {
      if (build_mat_exprs[i].bitoffset != bindex) {
        // insert space
        assert(build_mat_exprs[i].bitoffset > bindex);
        body.push_back(
            Type::getIntNTy(context->getLLVMContext(),
                            (build_mat_exprs[i].bitoffset - bindex)));
        ++packind;
      }

      const ExpressionType *out_type =
          build_mat_exprs[i].expr.getExpressionType();

      Type *llvm_type = out_type->getLLVMType(context->getLLVMContext());

      body.push_back(llvm_type);
      bindex = build_mat_exprs[i].bitoffset + context->getSizeOf(llvm_type) * 8;
      build_mat_exprs[i].packind = packind++;
      ++i;
    }
    assert(build_packet_widths[p] >= bindex);

    if (build_packet_widths[p] > bindex) {
      body.push_back(Type::getIntNTy(context->getLLVMContext(),
                                     (build_packet_widths[p] - bindex)));
    }

    Type *t = StructType::create(body, opLabel + "_struct_" + std::to_string(p),
                                 true);
    Type *t_ptr = PointerType::getUnqual(t);  //, /* address space */ 1);

    out_param_ids.push_back(context->appendStateVar(t_ptr));  //, true, false));
  }
  assert(i == build_mat_exprs.size());

  build_mat_exprs.erase(build_mat_exprs.begin());  // erase dummy entry for next

  // Type * t     = PointerType::get(((const PrimitiveType *)
  // out_type)->getLLVMType(context->getLLVMContext()), /* address space */ 1);
  // out_param_id = context->appendParameter(t    , true, false);

  Type *t_cnt = PointerType::getUnqual(int32_type);  //, /* address space */ 1);
  cnt_param_id = context->appendStateVar(t_cnt);     //, true, false);
}

Value *HashJoinChained::hash(expression_t exprs, Context *const context,
                             const OperatorState &childState) {
  ExpressionHasherVisitor hasher{context, childState};
  Value *hash = exprs.accept(hasher).value;
  auto size = ConstantInt::get(hash->getType(), (size_t(1) << hash_bits));
  return context->getBuilder()->CreateURem(hash, size);
}

void HashJoinChained::generate_build(ParallelContext *context,
                                     const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();

  Value *out_cnt = context->getStateVar(cnt_param_id);
  out_cnt->setName(opLabel + "_cnt_ptr");

  Value *head_ptr = context->getStateVar(head_param_id);
  head_ptr->setName(opLabel + "_head_ptr");

  Value *hash = HashJoinChained::hash(build_keyexpr, context, childState);

  // TODO: consider using just the object id as the index, instead of the atomic
  // index
  Value *old_cnt = context->workerScopedAtomicAdd(
      out_cnt,
      ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1));
  old_cnt->setName("index");

  // old_head = head[index]
  Value *old_head = context->workerScopedAtomicXchg(
      Builder->CreateInBoundsGEP(head_ptr, hash), old_cnt);
  old_head->setName("old_head");

  std::vector<Value *> out_ptrs;
  std::vector<Value *> out_vals;

  for (size_t i = 0; i < out_param_ids.size(); ++i) {
    Value *out_ptr = context->getStateVar(out_param_ids[i]);
    if (out_param_ids.size() != 1) {
      out_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
    } else {
      out_ptr->setName(opLabel + "_data_ptr");
    }
    // out_ptrs.push_back(out_ptr);

    // out_ptr->addAttr(Attribute::getWithAlignment(llvmContext,
    // context->getSizeOf(out_ptr)));

    out_ptrs.push_back(Builder->CreateInBoundsGEP(out_ptr, old_cnt));
    out_vals.push_back(
        UndefValue::get(out_ptr->getType()->getPointerElementType()));
  }

  out_vals[0] = Builder->CreateInsertValue(out_vals[0], old_head, 0);

  for (const GpuMatExpr &mexpr : build_mat_exprs) {
    ExpressionGeneratorVisitor exprGenerator(context, childState);
    ProteusValue valWrapper = mexpr.expr.accept(exprGenerator);

    out_vals[mexpr.packet] = Builder->CreateInsertValue(
        out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
  }

  for (size_t i = 0; i < out_ptrs.size(); ++i) {
    Builder->CreateStore(out_vals[i], out_ptrs[i]);

    // Builder->CreateAlignedStore(out_vals[i], out_ptrs[i],
    //                             build_packet_widths[i] / 8);
  }
}

void HashJoinChained::generate_probe(ParallelContext *context,
                                     const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
  // Type *int8_type = Type::getInt8Ty(llvmContext);
  // PointerType *void_ptr_type = PointerType::get(int8_type, 0);
  // Type *int64_type = Type::getInt64Ty(llvmContext);
  // Type *int32_type = Type::getInt32Ty(llvmContext);

  BasicBlock *insBB = Builder->GetInsertBlock();

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Value *head_ptr = context->getStateVar(probe_head_param_id);
  head_ptr->setName(opLabel + "_head_ptr");

  Builder->SetInsertPoint(insBB);

  ExpressionGeneratorVisitor exprGenerator(context, childState);
  ProteusValue keyWrapper = probe_keyexpr.accept(exprGenerator);
  Value *hash = HashJoinChained::hash(probe_keyexpr, context, childState);

  // current = head[hash(key)]
  size_t s = context->getSizeOf(head_ptr->getType()->getPointerElementType());
  // Value * current =
  // Builder->CreateAlignedLoad(Builder->CreateInBoundsGEP(head_ptr, hash), s &
  // -s);
  Value *current =
      Builder->CreateLoad(Builder->CreateInBoundsGEP(head_ptr, hash));
  current->setName("current");

  AllocaInst *mem_current = context->CreateEntryBlockAlloca(
      TheFunction, "mem_current", current->getType());

  Builder->CreateStore(current, mem_current);

  // while (current != eoc){

  BasicBlock *CondBB =
      BasicBlock::Create(llvmContext, "chainFollowCond", TheFunction);
  BasicBlock *ThenBB =
      BasicBlock::Create(llvmContext, "chainFollow", TheFunction);
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "cont", TheFunction);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(CondBB);

  // check end of chain

  Value *condition =
      Builder->CreateICmpNE(Builder->CreateLoad(mem_current),
                            ConstantInt::get(current->getType(), ~((size_t)0)));

  Builder->CreateCondBr(condition, ThenBB, MergeBB);
  Builder->SetInsertPoint(ThenBB);

  // check match

  std::vector<Value *> in_ptrs;
  std::vector<Value *> in_vals;
  for (size_t i = 0; i < in_param_ids.size(); ++i) {
    Value *in_ptr = context->getStateVar(in_param_ids[i]);
    if (in_param_ids.size() != 1) {
      in_ptr->setName(opLabel + "_data" + std::to_string(i) + "_ptr");
    } else {
      in_ptr->setName(opLabel + "_data_ptr");
    }
    // in_ptrs.push_back(in_ptr);

    in_ptrs.push_back(
        Builder->CreateInBoundsGEP(in_ptr, Builder->CreateLoad(mem_current)));
    size_t s =
        context->getSizeOf(in_ptrs.back()->getType()->getPointerElementType());
    // in_vals.push_back(Builder->CreateAlignedLoad(in_ptrs.back(), s & -s));
    in_vals.push_back(Builder->CreateLoad(in_ptrs.back()));
  }

  Value *next = Builder->CreateExtractValue(in_vals[0], 0);
  Value *build_key = Builder->CreateExtractValue(in_vals[0], 1);

  Builder->CreateStore(next, mem_current);

  ExpressionGeneratorVisitor eqGenerator{context, childState};
  expressions::ProteusValueExpression build_expr{
      probe_keyexpr.getExpressionType(),
      ProteusValue{build_key, context->createFalse()}};
  Value *match_condition =
      eq(probe_keyexpr, build_expr).accept(eqGenerator).value;

  BasicBlock *MatchThenBB =
      BasicBlock::Create(llvmContext, "matchChainFollow", TheFunction);

  Builder->CreateCondBr(match_condition, MatchThenBB, CondBB);

  Builder->SetInsertPoint(MatchThenBB);

  // Reconstruct tuples
  map<RecordAttribute, ProteusValueMemory> *allJoinBindings =
      new map<RecordAttribute, ProteusValueMemory>();

  if (probe_keyexpr.isRegistered()) {
    AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
        TheFunction, "mem_" + probe_keyexpr.getRegisteredAttrName(),
        keyWrapper.value->getType());

    Builder->CreateStore(keyWrapper.value, mem_arg);

    ProteusValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_arg;
    mem_valWrapper.isNull = context->createFalse();  // FIMXE: is this correct ?
    (*allJoinBindings)[probe_keyexpr.getRegisteredAs()] = mem_valWrapper;
  }

  if (probe_keyexpr.getExpressionType()->getTypeID() == RECORD) {
    auto rc = dynamic_cast<const expressions::RecordConstruction *>(
        probe_keyexpr.getUnderlyingExpression());

    size_t i = 0;
    for (const auto &a : rc->getAtts()) {
      auto e = a.getExpression();
      if (e.isRegistered()) {
        Value *d = Builder->CreateExtractValue(keyWrapper.value, i);

        AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
            TheFunction, "mem_" + e.getRegisteredAttrName(), d->getType());

        Builder->CreateStore(d, mem_arg);

        ProteusValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_arg;
        mem_valWrapper.isNull =
            context->createFalse();  // FIMXE: is this correct ?
        (*allJoinBindings)[e.getRegisteredAs()] = mem_valWrapper;
      }
      ++i;
    }
  }

  if (build_keyexpr.isRegistered()) {
    AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
        TheFunction, "mem_" + build_keyexpr.getRegisteredAttrName(),
        build_key->getType());

    Builder->CreateStore(build_key, mem_arg);

    ProteusValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_arg;
    mem_valWrapper.isNull = context->createFalse();  // FIMXE: is this correct ?
    (*allJoinBindings)[build_keyexpr.getRegisteredAs()] = mem_valWrapper;
  }

  if (build_keyexpr.getExpressionType()->getTypeID() == RECORD) {
    auto rc = dynamic_cast<const expressions::RecordConstruction *>(
        build_keyexpr.getUnderlyingExpression());

    size_t i = 0;
    for (const auto &a : rc->getAtts()) {
      auto e = a.getExpression();
      if (e.isRegistered()) {
        Value *d = Builder->CreateExtractValue(build_key, i);

        AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
            TheFunction, "mem_" + e.getRegisteredAttrName(), d->getType());

        Builder->CreateStore(d, mem_arg);

        ProteusValueMemory mem_valWrapper;
        mem_valWrapper.mem = mem_arg;
        mem_valWrapper.isNull =
            context->createFalse();  // FIMXE: is this correct ?
        (*allJoinBindings)[e.getRegisteredAs()] = mem_valWrapper;
      }
      ++i;
    }
  }

  // //from probe side
  // for (const auto &binding: childState.getBindings()){ //FIXME: deprecated...
  //     // std::cout << binding.first.getRelationName() << "--" <<
  //     binding.first.getAttrName() << std::endl;
  //     allJoinBindings->emplace(binding.first, binding.second);
  // }

  // from probe side
  for (const GpuMatExpr &mexpr : probe_mat_exprs) {
    if (mexpr.packet == 0 && mexpr.packind == 0) continue;

    // set activeLoop for build rel if not set (may be multiple ones!)
    {  // NOTE: Is there a better way ?
      Catalog &catalog = Catalog::getInstance();
      string probeRel = mexpr.expr.getRegisteredRelName();
      Plugin *pg = catalog.getPlugin(probeRel);
      assert(pg);
      RecordAttribute *probe_oid =
          new RecordAttribute(probeRel, activeLoop, pg->getOIDType());

      PrimitiveType *pr_oid_type =
          dynamic_cast<PrimitiveType *>(pg->getOIDType());
      if (!pr_oid_type) {
        string error_msg(
            "[HashJoinChained: ] Only primitive OIDs are supported.");
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }

      llvm::Type *llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

      AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
          TheFunction, "mem_" + probeRel + "_" + activeLoop, llvm_oid_type);

      Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

      ProteusValueMemory mem_valWrapper;
      mem_valWrapper.mem = mem_arg;
      mem_valWrapper.isNull = context->createFalse();

      if (allJoinBindings->count(*probe_oid) == 0) {
        (*allJoinBindings)[*probe_oid] = mem_valWrapper;
      }
    }

    ExpressionGeneratorVisitor exprGenerator(context, childState);

    ProteusValue val = mexpr.expr.accept(exprGenerator);

    AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
        TheFunction, "mem_" + mexpr.expr.getRegisteredAttrName(),
        val.value->getType());

    Builder->CreateStore(val.value, mem_arg);

    ProteusValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_arg;
    mem_valWrapper.isNull = context->createFalse();

    (*allJoinBindings)[mexpr.expr.getRegisteredAs()] = mem_valWrapper;
  }

  // from build side
  for (const GpuMatExpr &mexpr : build_mat_exprs) {
    if (mexpr.packet == 0 && mexpr.packind == 0) continue;

    // set activeLoop for build rel if not set (may be multiple ones!)
    {  // NOTE: Is there a better way ?
      Catalog &catalog = Catalog::getInstance();
      string buildRel = mexpr.expr.getRegisteredRelName();
      Plugin *pg = catalog.getPlugin(buildRel);
      assert(pg);
      RecordAttribute *build_oid =
          new RecordAttribute(buildRel, activeLoop, pg->getOIDType());

      PrimitiveType *pr_oid_type =
          dynamic_cast<PrimitiveType *>(pg->getOIDType());
      if (!pr_oid_type) {
        string error_msg(
            "[HashJoinChained: ] Only primitive OIDs are supported.");
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }

      llvm::Type *llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

      AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
          TheFunction, "mem_" + buildRel + "_" + activeLoop, llvm_oid_type);

      Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

      ProteusValueMemory mem_valWrapper;
      mem_valWrapper.mem = mem_arg;
      mem_valWrapper.isNull = context->createFalse();

      if (allJoinBindings->count(*build_oid) == 0) {
        (*allJoinBindings)[*build_oid] = mem_valWrapper;
      }
    }

    // ExpressionGeneratorVisitor exprGenerator(context, childState);

    Value *val =
        Builder->CreateExtractValue(in_vals[mexpr.packet], mexpr.packind);

    AllocaInst *mem_arg = context->CreateEntryBlockAlloca(
        TheFunction, "mem_" + mexpr.expr.getRegisteredAttrName(),
        val->getType());

    Builder->CreateStore(val, mem_arg);

    ProteusValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_arg;
    mem_valWrapper.isNull = context->createFalse();

    (*allJoinBindings)[mexpr.expr.getRegisteredAs()] = mem_valWrapper;
  }

  // Triggering parent
  OperatorState *newState = new OperatorState(*this, *allJoinBindings);
  getParent()->consume(context, *newState);

  Builder->CreateBr(CondBB);

  // TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder->SetInsertPoint(MergeBB);
}

void HashJoinChained::open_build(Pipeline *pip) {
  std::vector<void *> next_w_values;

  uint32_t *head =
      (uint32_t *)malloc(sizeof(uint32_t) * (1 << hash_bits) + sizeof(int32_t));
  int32_t *cnt = (int32_t *)(head + (1 << hash_bits));

  // cudaStream_t strm;
  // gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
  memset(head, -1, sizeof(uint32_t) * (1 << hash_bits));
  memset(cnt, 0, sizeof(int32_t));

  for (const auto &w : build_packet_widths) {
    next_w_values.emplace_back(malloc((w / 8) * maxBuildInputSize));
  }

  pip->setStateVar(head_param_id, head);
  pip->setStateVar(cnt_param_id, cnt);

  for (size_t i = 0; i < build_packet_widths.size(); ++i) {
    pip->setStateVar(out_param_ids[i], next_w_values[i]);
  }

  next_w_values.emplace_back(head);
  confs[pip->getGroup()] = next_w_values;

  // gpu_run(cudaStreamSynchronize(strm));
  // gpu_run(cudaStreamDestroy(strm));
}

void HashJoinChained::open_probe(Pipeline *pip) {
  std::vector<void *> next_w_values = confs[pip->getGroup()];
  uint32_t *head = (uint32_t *)next_w_values.back();

  // next_w_values.pop_back();

  pip->setStateVar(probe_head_param_id, head);

  for (size_t i = 0; i < build_packet_widths.size(); ++i) {
    pip->setStateVar(in_param_ids[i], next_w_values[i]);
  }
}

void HashJoinChained::close_build(Pipeline *pip) {
  int32_t h_cnt;
  memcpy(&h_cnt, pip->getStateVar<int32_t *>(cnt_param_id), sizeof(int32_t));
  assert(((size_t)h_cnt) <= maxBuildInputSize &&
         "Build input sized exceeded given parameter");
}

void HashJoinChained::close_probe(Pipeline *pip) {
  for (const auto &p : confs[pip->getGroup()]) free(p);
}
