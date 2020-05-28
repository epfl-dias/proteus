/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "bloom-filter-repack.hpp"

#include <llvm/IR/IntrinsicsX86.h>

#include <expressions/expressions-generator.hpp>
#include <memory/block-manager.hpp>
#include <memory/buffer-manager.cuh>
#include <memory/memory-manager.hpp>
#include <util/jit/pipeline.hpp>
#include <util/logging.hpp>

void BloomFilterRepack::produce_(ParallelContext *context) {
  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  for (const auto &wantedField : wantedFields) {
    old_buffs.push_back(context->appendStateVar(llvm::PointerType::getUnqual(
        RecordAttribute{wantedField.getRegisteredAs(), true}.getLLVMType(
            context->getLLVMContext()))));

    auto type = llvm::ArrayType::get(wantedField.getRegisteredAs()
                                         .getLLVMType(context->getLLVMContext())
                                         ->getPointerElementType(),
                                     BlockManager::block_size / 4);
    type->dump();
    out_buffs.push_back(context->appendStateVar(
        llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(type)),
        [=](llvm::Value *pip) {
          auto tmp = context->gen_call(
              &get_buffer, {context->createSizeT(BlockManager::block_size)});
          tmp->dump();
          auto v2 = context->getBuilder()->CreateBitCast(
              tmp, llvm::PointerType::getUnqual(type));
          v2->dump();
          auto v = context->allocateStateVar(v2->getType());
          v->dump();
          context->getBuilder()->CreateStore(v2, v);
          return v;
        },
        [context](llvm::Value *pip, llvm::Value *v) {
          //          context->gen_call(
          //              &release_buffer,
          //              {context->getBuilder()->CreateBitCast(
          //                  context->getBuilder()->CreateLoad(v),
          //                  llvm::Type::getInt8PtrTy(context->getLLVMContext()))});
          context->deallocateStateVar(v);
        }));
  }

  auto t = getFilterType(context);
  filter_ptr = context->appendStateVar(
      t,
      [=](llvm::Value *pip) -> llvm::Value * {
        return context->gen_call("getBloomFilter",
                                 {pip, context->createInt64(bloomId)}, t);
      },
      [=](llvm::Value *, llvm::Value *s) {
        //        context->deallocateStateVar(s);
      });

  cntVar_id = context->appendStateVar(
      llvm::Type::getInt32PtrTy(context->getLLVMContext()),
      [=](llvm::Value *pip) -> llvm::Value * {
        auto mem = context->allocateStateVar(
            llvm::Type::getInt32Ty(context->getLLVMContext()));
        context->CodegenMemset(mem, context->createInt8(0), 32 / 4);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); });

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());
  auto oidType = pg->getOIDType()->getLLVMType(context->getLLVMContext());
  oidVar_id = context->appendStateVar(
      llvm::PointerType::getUnqual(oidType),
      [=](llvm::Value *pip) -> llvm::Value * {
        auto mem = context->allocateStateVar(oidType);
        context->CodegenMemset(mem, context->createInt8(0),
                               context->getSizeOf(oidType));
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); });

  getChild()->produce(context);
}

void BloomFilterRepack::consumeVector(ParallelContext *context,
                                      const OperatorState &childState,
                                      llvm::Value *filter, size_t vsize,
                                      llvm::Value *offset, llvm::Value *N) {
  auto Builder = context->getBuilder();

  std::map<RecordAttribute, ProteusValueMemory> variableBindings;

  auto BB = Builder->GetInsertBlock();
  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  std::vector<ProteusValueMemory> outputBuffs;
  {
    size_t i = 0;
    for (const auto &attr : wantedFields) {
      outputBuffs.emplace_back(context->toMem(
          Builder->CreateLoad(context->getStateVar(out_buffs[i++])),
          context->createFalse()));
      variableBindings[attr.getRegisteredAs()] = outputBuffs.back();
    }
  }
  auto outCnt =
      context->toMem(Builder->CreateLoad(context->getStateVar(cntVar_id)),
                     context->createFalse());

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateStore(Builder->CreateLoad(outCnt.mem),
                       context->getStateVar(cntVar_id));

  Builder->SetInsertPoint(BB);

  // Read a vector of hashed predicates
  llvm::Value *vse2 = Builder->CreateInBoundsGEP(
      Builder->CreateLoad(childState[e.getRegisteredAs()].mem), offset);
  auto te = llvm::PointerType::getUnqual(
      llvm::VectorType::get(vse2->getType()->getPointerElementType(), vsize));
  auto vse = Builder->CreateAlignedLoad(Builder->CreateBitCast(vse2, te),
                                        llvm::MaybeAlign{512});

  assert(!(filterSize & (filterSize - 1)));
  // byte offset
  auto vec = Builder->CreateAnd(
      vse,
      Builder->CreateVectorSplat(vsize, context->createInt32(filterSize - 1)));
  auto woffset = Builder->CreateUDiv(
      vec,
      Builder->CreateVectorSplat(
          vsize,
          context->createInt32(
              sizeof(int32_t) /
              context->getSizeOf(filter->getType()->getPointerElementType()))));
  auto filter_orig = filter;
  filter = Builder->CreateBitCast(
      filter, llvm::PointerType::getInt32PtrTy(context->getLLVMContext()));

  // Probe using the vector
  //  auto hits2 = Builder->CreateMaskedGather(
  //      Builder->CreateInBoundsGEP(Builder->CreateVectorSplat(vsize, filter),
  //                                 woffset),
  //      1);
  auto mod = Builder->CreateMul(
      Builder->CreateURem(
          vec, Builder->CreateVectorSplat(vsize, context->createInt32(4))),
      Builder->CreateVectorSplat(vsize, context->createInt32(8)));
  auto div = Builder->CreateUDiv(
      vec, Builder->CreateVectorSplat(vsize, context->createInt32(4)));

  auto f = Builder->CreateBitCast(
      filter_orig, llvm::Type::getIntNPtrTy(context->getLLVMContext(), 32));

  auto gather = llvm::Intrinsic::getDeclaration(
      context->getModule(), llvm::Intrinsic::x86_avx512_mask_gather_dpi_512);

  llvm::Metadata *Args2[] = {nullptr};
  llvm::MDNode *n2 = llvm::MDNode::get(context->getLLVMContext(), Args2);
  n2->replaceOperandWith(0, n2);

  llvm::Metadata *Args[] = {nullptr, n2};
  llvm::MDNode *n = llvm::MDNode::get(context->getLLVMContext(), Args);
  n->replaceOperandWith(0, n);

  llvm::Value *hits2 = llvm::UndefValue::get(llvm::VectorType::get(
      llvm::Type::getInt32Ty(context->getLLVMContext()), vsize));
  std::vector<uint32_t> mask;
  for (size_t s = 0; s < vsize; s += 16) {
    std::vector<uint32_t> mind;
    for (size_t j = 0; j < 16; ++j) {
      mind.emplace_back(s + j);
    }
    LOG(INFO) << 1;
    auto ind = Builder->CreateShuffleVector(
        div, llvm::UndefValue::get(div->getType()), mind);

    ind->dump();

    auto tmp = Builder->CreateCall(
        gather, {llvm::UndefValue::get(llvm::VectorType::get(
                     llvm::Type::getInt32Ty(context->getLLVMContext()), 16)),
                 Builder->CreateBitCast(
                     f, llvm::Type::getInt8PtrTy(context->getLLVMContext())),
                 ind, llvm::ConstantVector::getSplat(16, context->createTrue()),
                 context->createInt32(4)});

    for (size_t j = 0; j < 16; ++j) {
      hits2 = Builder->CreateInsertElement(
          hits2, Builder->CreateExtractElement(tmp, j), s + j);
    }

    {
      tmp->setMetadata(llvm::LLVMContext::MD_alias_scope, n);
      tmp->setMetadata(llvm::LLVMContext::MD_noalias, n);
    }

    {  // Loaded value will be the same in all the places it will be loaded
      //! invariant.load !{i32 1}
      llvm::Metadata *Args[] = {
          llvm::ValueAsMetadata::get(context->createInt32(1))};
      llvm::MDNode *n = llvm::MDNode::get(context->getLLVMContext(), Args);
      tmp->setMetadata(llvm::LLVMContext::MD_invariant_load, n);
    }
  }
  //
  //  hits2 = Builder->CreateMaskedGather(
  //      Builder->CreateInBoundsGEP(Builder->CreateVectorSplat(vsize, f),
  //                                 div),
  //      1);

  //  auto indMask = [&] {
  //    std::vector<llvm::Constant *> vs;
  //    for (size_t i = 0; i < vsize; ++i) {
  //      vs.emplace_back(llvm::ConstantInt::get(offset->getType(), i));
  //    }
  //    auto inds = llvm::ConstantVector::get(vs);
  //    return Builder->CreateICmpUGT(
  //        Builder->CreateVectorSplat(vsize, Builder->CreateSub(N, offset)),
  //        inds);
  //  }();
  auto indMask = llvm::ConstantVector::getSplat(vsize, context->createTrue());

  llvm::Value *hits = Builder->CreateICmpNE(
      Builder->CreateAnd(
          hits2, Builder->CreateShl(
                     Builder->CreateZExt(indMask, mod->getType()), mod)),
      Builder->CreateVectorSplat(vsize, context->createInt32(0)));

  {
    vse->setMetadata(llvm::LLVMContext::MD_alias_scope, n);
    vse->setMetadata(llvm::LLVMContext::MD_noalias, n);
  }

  {  // Loaded value will be the same in all the places it will be loaded
    //! invariant.load !{i32 1}
    llvm::Metadata *Args[] = {
        llvm::ValueAsMetadata::get(context->createInt32(1))};
    llvm::MDNode *n = llvm::MDNode::get(context->getLLVMContext(), Args);
    vse->setMetadata(llvm::LLVMContext::MD_invariant_load, n);
    llvm::Metadata *Args2[] = {
        llvm::ValueAsMetadata::get(context->createTrue())};
    llvm::MDNode *n2 = llvm::MDNode::get(context->getLLVMContext(), Args2);
    vse->setMetadata(llvm::LLVMContext::MD_nontemporal, n2);
  }
  //
  //  auto hits3 = Builder->CreateLShr(
  //      hits2,
  //      Builder->CreateMul(
  //          Builder->CreateURem(
  //              vec,
  //              Builder->CreateVectorSplat(
  //                  vsize, context->createInt32(
  //                             sizeof(int32_t) /
  //                             context->getSizeOf(
  //                                 vec->getType()->getVectorElementType())))),
  //          Builder->CreateVectorSplat(vsize, context->createInt32(8))));

  //  auto hits = Builder->CreateICmpNE(
  //      //            hits3,
  //      Builder->CreateAnd(hits3,
  //                         Builder->CreateVectorSplat(
  //                             vsize, context->createInt32(255 + (255u <<
  //                             24)))),
  //      Builder->CreateVectorSplat(vsize, context->createInt32(0)));

  //  hits = Builder->CreateAnd(hits, indMask);

  //  std::map<RecordAttribute, expressions::RefExpression> buffs;

  auto intvec = llvm::Type::getIntNTy(context->getLLVMContext(), vsize);

  auto popCnt = Builder->CreateCall(
      llvm::Intrinsic::getDeclaration(context->getModule(),
                                      llvm::Intrinsic::ctpop, {intvec}),
      Builder->CreateBitCast(hits, intvec));
  auto cnt = Builder->CreateLoad(
      outCnt.mem);  // popCnt; // FIXME: Update next cnt to offset + popCnt

  context->gen_if({Builder->CreateICmpNE(
                       popCnt, llvm::ConstantInt::get(popCnt->getType(), 0)),
                   context->createFalse()})([&] {
    ExpressionGeneratorVisitor vis{context, childState};

    std::vector<std::pair<llvm::Instruction *, llvm::Value *>> loads;

    size_t i = 0;
    for (const auto &attr : wantedFields) {
      auto tmp = Builder->CreateLoad(outputBuffs[i++].mem);
      tmp->getType()->dump();
      auto b_ptr =
          Builder->CreateInBoundsGEP(tmp, {context->createInt32(0), cnt});
      b_ptr->getType()->dump();
      llvm::Value *vs_tmp = Builder->CreateInBoundsGEP(
          Builder->CreateLoad(childState[attr.getRegisteredAs()].mem), offset);
      auto t = llvm::PointerType::getUnqual(llvm::VectorType::get(
          vs_tmp->getType()->getPointerElementType(), vsize));
      auto ld = Builder->CreateAlignedLoad(Builder->CreateBitCast(vs_tmp, t),
                                           512 / 8);
      loads.emplace_back(std::make_pair(ld, b_ptr));

      //    buffs[attr].assign(buffs[attr] +
      //    expressions::ProteusValueExpression{})
    }

    i = 0;
    for (const auto &attr : wantedFields) {
      auto ld = loads[i].first;
      auto b_ptr = loads[i].second;
      auto F = llvm::Intrinsic::getDeclaration(
          context->getModule(), llvm::Intrinsic::masked_compressstore,
          {ld->getType()});

      auto s = Builder->CreateCall(F, {ld, b_ptr, hits});
      {
        ld->setMetadata(llvm::LLVMContext::MD_alias_scope, n);
        ld->setMetadata(llvm::LLVMContext::MD_noalias, n);
        s->setMetadata(llvm::LLVMContext::MD_noalias, n);
        s->setMetadata(llvm::LLVMContext::MD_alias_scope, n);
      }

      {  // Loaded value will be the same in all the places it will be loaded
        //! invariant.load !{i32 1}
        llvm::Metadata *Args[] = {
            llvm::ValueAsMetadata::get(context->createInt32(1))};
        llvm::MDNode *n = llvm::MDNode::get(context->getLLVMContext(), Args);
        ld->setMetadata(llvm::LLVMContext::MD_invariant_load, n);
        llvm::Metadata *Args2[] = {
            llvm::ValueAsMetadata::get(context->createTrue())};
        llvm::MDNode *n2 = llvm::MDNode::get(context->getLLVMContext(), Args2);
        ld->setMetadata(llvm::LLVMContext::MD_nontemporal, n2);
        s->setMetadata(llvm::LLVMContext::MD_nontemporal, n2);
      }
      i++;

      //    buffs[attr].assign(buffs[attr] +
      //    expressions::ProteusValueExpression{})
    }
    auto nextCnt = Builder->CreateAdd(
        cnt, Builder->CreateZExtOrTrunc(popCnt, cnt->getType()));
    Builder->CreateStore(nextCnt, outCnt.mem);

    size_t capacity = BlockManager::block_size /
                      4;  // FIXME: take into consideration possible
                          //  different types and not only int32_t !
    auto almostFull =
        Builder->CreateICmpUGT(nextCnt, context->createInt32(capacity - vsize));
    context->gen_if({almostFull, context->createFalse()})([&] {
      Plugin *pg = Catalog::getInstance().getPlugin(
          wantedFields[0].getRegisteredRelName());

      auto new_oid =
          Builder->CreateLoad(context->getStateVar(oidVar_id), "oid");
      Builder->CreateStore(
          Builder->CreateAdd(new_oid,
                             Builder->CreateZExt(nextCnt, new_oid->getType())),
          context->getStateVar(oidVar_id));

      RecordAttribute tupleIdentifier{wantedFields[0].getRegisteredRelName(),
                                      activeLoop, pg->getOIDType()};

      variableBindings[tupleIdentifier] =
          context->toMem(new_oid, context->createFalse());

      RecordAttribute tupCnt{wantedFields[0].getRegisteredRelName(),
                             "activeCnt",
                             pg->getOIDType()};  // FIXME: OID type for blocks ?

      variableBindings[tupCnt] =
          context->toMem(Builder->CreateZExt(nextCnt, new_oid->getType()),
                         context->createFalse());
      //    context->log(nextCnt);

      // Triggering parent
      OperatorState state{*this, variableBindings};
      getParent()->consume(context, state);

      Builder->CreateStore(context->createInt32(0), outCnt.mem);

      for (size_t i = 0; i < wantedFields.size(); ++i) {
        auto b = context->gen_call(
            &get_buffer,
            {context->createSizeT(BlockManager::block_size) /* FIXME */});
        b->getType()->dump();
        context->getStateVar(out_buffs[i])->getType()->dump();
        b = Builder->CreateBitCast(
            b, outputBuffs[i].mem->getType()->getPointerElementType());
        Builder->CreateStore(b, context->getStateVar(out_buffs[i]));
        Builder->CreateStore(b, outputBuffs[i].mem);
      }
    });
  });
}

void BloomFilterRepack::consume(ParallelContext *context,
                                const OperatorState &childState) {
  std::vector<RecordAttribute> attributes;

  size_t vsize = 16;  // FIXME: how to tune?
  auto Builder = context->getBuilder();
  auto filter = context->getStateVar(filter_ptr);
  filter = Builder->CreateInBoundsGEP(
      filter, {context->createInt64(0), context->createInt64(0)});

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());

  RecordAttribute tupleCnt{wantedFields[0].getRegisteredRelName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?
  llvm::Value *cnt = Builder->CreateLoad(childState[tupleCnt].mem, "cnt");

  auto mem_itemCtr = context->CreateEntryBlockAlloca("i_ptr", cnt->getType());
  Builder->CreateStore(
      Builder->CreateIntCast(context->threadId(), cnt->getType(), false),
      mem_itemCtr);

  llvm::Value *lhs;

  /**
   * Equivalent:
   * while(itemCtr < size)
   */
  context->gen_while([&]() {
    lhs = Builder->CreateLoad(mem_itemCtr, "i");

    return ProteusValue{
        Builder->CreateICmpULT(
            lhs,
            Builder->CreateAdd(cnt, llvm::ConstantInt::get(cnt->getType(), 0))),
        context->createFalse()};
  })([&](llvm::BranchInst *loop_cond) {
    auto &llvmContext = context->getLLVMContext();

    llvm::MDNode *LoopID;
    {
      llvm::MDString *vec_st =
          llvm::MDString::get(llvmContext, "llvm.loop.vectorize.enable");
      llvm::Type *int1Type = llvm::Type::getInt1Ty(llvmContext);
      llvm::Metadata *one =
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int1Type, 1));
      llvm::Metadata *vec_en[] = {vec_st, one};
      llvm::MDNode *vectorize_enable = llvm::MDNode::get(llvmContext, vec_en);

      llvm::MDString *itr_st =
          llvm::MDString::get(llvmContext, "llvm.loop.interleave.count");
      llvm::Type *int32Type = llvm::Type::getInt32Ty(llvmContext);
      llvm::Metadata *count =
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(int32Type, 4));
      llvm::Metadata *itr_en[] = {itr_st, count};
      llvm::MDNode *interleave_count = llvm::MDNode::get(llvmContext, itr_en);

      llvm::Metadata *Args[] = {
          nullptr};  //, vectorize_enable, interleave_count};
      LoopID = llvm::MDNode::get(llvmContext, Args);
      LoopID->replaceOperandWith(0, LoopID);

      loop_cond->setMetadata(llvm::LLVMContext::MD_loop, LoopID);
    }

    auto offset = Builder->CreateLoad(mem_itemCtr);
    consumeVector(context, childState, filter, vsize, offset, cnt);

    Builder->CreateStore(
        Builder->CreateAdd(offset, context->createInt64(vsize)), mem_itemCtr);
  });

  // FIXME: serious error! we are loosing record here!!! we need to process the
  //  last (up to) vsize-1 items!

  consume_flush(context);
}

void BloomFilterRepack::consume_flush(ParallelContext *context) {
  save_current_blocks_and_restore_at_exit_scope blks{context};
  llvm::LLVMContext &llvmContext = context->getLLVMContext();

  flushingFunc = (*context)->createHelperFunction(
      "flush", std::vector<llvm::Type *>{}, std::vector<bool>{},
      std::vector<bool>{});
  closingPip = (context->operator->());
  auto *Builder = context->getBuilder();
  auto *insBB = Builder->GetInsertBlock();
  auto *F = insBB->getParent();
  // Get the ENTRY BLOCK
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  auto *CondBB = llvm::BasicBlock::Create(llvmContext, "flushCond", F);

  // Create the "AFTER LOOP" block and insert it.
  auto *AfterBB = llvm::BasicBlock::Create(llvmContext, "flushEnd", F);
  context->setEndingBlock(AfterBB);

  Builder->SetInsertPoint(CondBB);

  std::map<RecordAttribute, ProteusValueMemory> variableBindings;
  size_t i = 0;
  for (const auto &attr : wantedFields) {
    variableBindings[attr.getRegisteredAs()] = context->toMem(
        Builder->CreateLoad(context->getStateVar(out_buffs[i++])),
        context->createFalse());
  }

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());

  auto new_oid = Builder->CreateLoad(context->getStateVar(oidVar_id), "oid");

  RecordAttribute tupleIdentifier{wantedFields[0].getRegisteredRelName(),
                                  activeLoop, pg->getOIDType()};

  variableBindings[tupleIdentifier] =
      context->toMem(new_oid, context->createFalse());

  RecordAttribute tupCnt{wantedFields[0].getRegisteredRelName(), "activeCnt",
                         pg->getOIDType()};  // FIXME: OID type for blocks ?

  variableBindings[tupCnt] = context->toMem(
      Builder->CreateZExt(Builder->CreateLoad(context->getStateVar(cntVar_id)),
                          new_oid->getType()),
      context->createFalse());

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  Builder->CreateBr(AfterBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateRetVoid();
}

void BloomFilterRepack::open(Pipeline *pip) {}

void BloomFilterRepack::close(Pipeline *pip) {
  ((void (*)(void *))closingPip->getCompiledFunction(flushingFunc))(
      pip->getState());
}
