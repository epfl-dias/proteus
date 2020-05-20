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

#include "operators/hash-rearrange.hpp"

#include <memory/memory-manager.hpp>
#include <util/bitwise-ops.hpp>
//#include <values/indexed-seq.hpp>

#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-hasher.hpp"
#include "util/jit/pipeline.hpp"

using namespace llvm;

void HashRearrange::produce_(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);
  Type *cnt_type =
      //      (numOfBuckets > 1)
      PointerType::getUnqual(ArrayType::get(oid_type, numOfBuckets));
  //          : PointerType::getUnqual(oid_type);
  cntVar_id = context->appendStateVar(cnt_type);

  oidVar_id = context->appendStateVar(PointerType::getUnqual(oid_type));

  std::vector<Type *> block_types;
  for (const auto &wantedField : wantedFields) {
    block_types.emplace_back(
        RecordAttribute(wantedField.getRegisteredAs(), true)
            .getLLVMType(llvmContext));
    wfSizes.emplace_back(context->getSizeOf(
        wantedField.getExpressionType()->getLLVMType(llvmContext)));
  }

  Type *block_stuct = StructType::get(llvmContext, block_types);

  blkVar_id = context->appendStateVar(
      PointerType::getUnqual(ArrayType::get(block_stuct, numOfBuckets)));

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

llvm::Value *HashRearrange::getIndexPtr(ParallelContext *context,
                                        llvm::Value *target) const {
  //  if (numOfBuckets > 1) {
  IRBuilder<> *Builder = context->getBuilder();
  return Builder->CreateInBoundsGEP(context->getStateVar(cntVar_id),
                                    {context->createInt32(0), target});
  //  } else {
  //    return context->getStateVar(cntVar_id);
  //  }
}

llvm::Value *HashRearrange::getIndex(ParallelContext *context,
                                     llvm::Value *target) const {
  IRBuilder<> *Builder = context->getBuilder();
  return Builder->CreateLoad(getIndexPtr(context, target));
}

llvm::StoreInst *HashRearrange::setIndex(ParallelContext *context,
                                         llvm::Value *newIndex,
                                         llvm::Value *target) const {
  IRBuilder<> *Builder = context->getBuilder();
  return Builder->CreateStore(newIndex, getIndexPtr(context, target));
}

void HashRearrange::consume(ParallelContext *context,
                            const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());
  auto *oid_type = (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);

  size_t max_width = 0;
  for (const auto &e : wantedFields) {
    max_width = std::max(
        max_width,
        context->getSizeOf(e.getExpressionType()->getLLVMType(llvmContext)));
  }

  cap = blockSize / max_width;
  Value *capacity = ConstantInt::get(oid_type, cap);
  Value *last_index = ConstantInt::get(oid_type, cap - 1);

  std::map<RecordAttribute, ProteusValueMemory> variableBindings;

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  RecordAttribute tupCnt{wantedFields[0].getRegisteredRelName(), "activeCnt",
                         pg->getOIDType()};  // FIXME: OID type for blocks ?
  variableBindings[tupCnt] = context->toMem(capacity, context->createFalse());

  AllocaInst *ready_cnt =
      context->CreateEntryBlockAlloca(F, "readyN", int32_type);
  Builder->CreateStore(ConstantInt::get(int32_type, 0), ready_cnt);

  Builder->SetInsertPoint(insBB);

  // Generate target
  ExpressionGeneratorVisitor exprGenerator{context, childState};
  auto h = (expressions::HashExpression{hashExpr} % int64_t{numOfBuckets})
               .accept(exprGenerator);
  auto target = Builder->CreateTruncOrBitCast(h.value, int32_type);

  if (hashProject) {
    // Save hash in bindings
    variableBindings[*hashProject] = context->toMem(h);
  }

  vector<Type *> members;
  for (const auto &wantedField : wantedFields) {
    RecordAttribute tblock{wantedField.getRegisteredAs(), true};
    members.push_back(tblock.getLLVMType(llvmContext));
  }
  members.push_back(target->getType());

  //  auto indx = getIndex(context, target);
  //  expressions::ProteusValueExpression ij{pg->getOIDType(),
  //                                         {indx, context->createFalse()}};

  auto bb = Builder->GetInsertBlock();
  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  Value *blocks;      // = context->getStateVar(blkVar_id);
  LoadInst *blocks2;  // = context->getStateVar(blkVar_id);
  Value *alloc;
  Value *alloc_index;
  Value *alloc_oid;
  Value *sPtr;
  {
    sPtr = context->getBuilder()->CreateInBoundsGEP(
        (*context)->getStateVarPtr(),
        {context->createInt64(0),
         context->createInt32(blkVar_id.index_in_pip)});
    blocks2 = context->getBuilder()->CreateLoad(sPtr);
    auto b = Builder->CreateLoad(blocks2);

    llvm::Metadata *Args[] = {
        llvm::ValueAsMetadata::get(context->createInt64(10000))};
    blocks2->setMetadata(LLVMContext::MD_dereferenceable,
                         MDNode::get(llvmContext, Args));

    //    alloc = context->CreateEntryBlockAlloca("asd", blocks2->getType());
    //    blocks2->getType()->dump();
    //    alloc->getType()->dump();
    //    Builder->CreateStore(blocks2, alloc);
    //    blocks = Builder->CreateLoad(alloc);
    //

    blocks = context->CreateEntryBlockAlloca("asd", b->getType());
    Builder->CreateStore(b, blocks);

    auto init_index = Builder->CreateLoad(context->getStateVar(cntVar_id));
    alloc_index = context->CreateEntryBlockAlloca("asd", init_index->getType());
    Builder->CreateStore(init_index, alloc_index);

    auto init_oid = Builder->CreateLoad(context->getStateVar(oidVar_id), "oid");
    alloc_oid = context->CreateEntryBlockAlloca("oid_ptr", init_oid->getType());
    Builder->CreateStore(init_oid, alloc_oid);
  }
  Builder->SetInsertPoint(context->getEndingBlock());

  Builder->CreateStore(Builder->CreateLoad(alloc_index),
                       context->getStateVar(cntVar_id));
  Builder->CreateStore(Builder->CreateLoad(blocks), blocks2);
  Builder->CreateStore(Builder->CreateLoad(alloc_oid),
                       context->getStateVar(oidVar_id));

  Builder->SetInsertPoint(bb);

  auto indx = Builder->CreateLoad(Builder->CreateInBoundsGEP(
      alloc_index, {context->createInt32(0), target}));

  Value *curblk =
      Builder->CreateInBoundsGEP(blocks, {context->createInt32(0), target});

  llvm::Metadata *Args2[] = {nullptr};
  MDNode *n2 = MDNode::get(llvmContext, Args2);
  n2->replaceOperandWith(0, n2);

  llvm::Metadata *Args[] = {nullptr, n2};
  MDNode *n = MDNode::get(llvmContext, Args);
  n->replaceOperandWith(0, n);

  //  MDNode *ntemp = MDNode::get(llvmContext,
  //  {llvm::ValueAsMetadata::get(context->createInt32(1))}); //awful
  //  performance

  std::vector<ProteusValue> vals;
  for (auto &e : wantedFields) {
    vals.emplace_back(e.accept(exprGenerator));
  }

  auto base = next_power_of_2(wantedFields.size());
  auto vtype = llvm::VectorType::get(vals[0].value->getType(), base);
  auto vptrtype = llvm::VectorType::get(
      llvm::PointerType::getUnqual(vals[0].value->getType()), base);
  llvm::Value *v = llvm::UndefValue::get(vtype);
  llvm::Value *ptr = llvm::UndefValue::get(vptrtype);

  std::vector<Value *> els;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    auto block = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        curblk, {context->createInt32(0), context->createInt32(i)}));

    block->setMetadata(LLVMContext::MD_alias_scope, n);

    Value *el_ptr = Builder->CreateInBoundsGEP(block, indx);

    Value *el = vals[i].value;
    assert(el);
    auto ld = dyn_cast<llvm::LoadInst>(el);
    if (ld) ld->setMetadata(LLVMContext::MD_alias_scope, n);
    //    assert(ld);

    v = Builder->CreateInsertElement(v, el, i);
    ptr = Builder->CreateInsertElement(ptr, block, i);

    //      auto s = Builder->CreateStore(el, el_ptr);
    //      s->setMetadata(LLVMContext::MD_noalias, n);
    //      //    s->setMetadata("nontemporal", ntemp); //awful performance

    els.push_back(block);
  }
  ptr = Builder->CreateInBoundsGEP(vals[0].value->getType(), ptr,
                                   Builder->CreateVectorSplat(base, indx));
  std::vector<llvm::Constant *> maskc;
  for (size_t i = 0; i < base; ++i) {
    maskc.emplace_back((i < wantedFields.size()) ? context->createTrue()
                                                 : context->createFalse());
  }
  auto mask = llvm::ConstantVector::get(maskc);
  Builder->CreateMaskedScatter(
      v, ptr, context->getSizeOf(vtype->getElementType()), mask);

  //
  //  std::vector<Value *> els;
  //  for (size_t i = 0; i < wantedFields.size(); ++i) {
  //    Value *block = Builder->CreateLoad(Builder->CreateInBoundsGEP(
  //        curblk, {context->createInt32(0), context->createInt32(i)}));
  //
  //    auto btype = wantedFields[i].getExpressionType();
  //    expressions::ProteusValueExpression fptr{new type::IndexedSeq{*btype},
  //                                             {block,
  //                                             context->createFalse()}};
  //
  //    fptr[ij].assign(wantedFields[i]).accept(exprGenerator);
  //
  //    els.push_back(block);
  //  }
  //

  // if indx  >= vectorSize - 1
  Value *cond = Builder->CreateICmpUGE(indx, last_index);

  context
      ->gen_if({cond, context->createFalse()})([&]() {  // FullBB
        Value *new_oid = Builder->CreateLoad(alloc_oid, "oid");
        Builder->CreateStore(Builder->CreateAdd(new_oid, capacity), alloc_oid);

        RecordAttribute tupleIdentifier{wantedFields[0].getRegisteredRelName(),
                                        activeLoop, pg->getOIDType()};

        variableBindings[tupleIdentifier] =
            context->toMem(new_oid, context->createFalse());

        for (size_t i = 0; i < wantedFields.size(); ++i) {
          RecordAttribute tblock{wantedFields[i].getRegisteredAs(), true};

          variableBindings[tblock] =
              context->toMem(els[i], context->createFalse());
        }

        getParent()->consume(context, {*this, variableBindings});

        Function *get_buffer = context->getFunction("get_buffer");

        for (size_t i = 0; i < wantedFields.size(); ++i) {
          RecordAttribute tblock{wantedFields[i].getRegisteredAs(), true};
          Value *size = context->createSizeT(
              cap * context->getSizeOf(
                        wantedFields[i].getExpressionType()->getLLVMType(
                            llvmContext)));

          auto new_buff = Builder->CreateCall(get_buffer, {size});
          new_buff->setMetadata(LLVMContext::MD_noalias, n);

          auto new_buff2 =
              Builder->CreateBitCast(new_buff, tblock.getLLVMType(llvmContext));

          Builder->CreateStore(
              new_buff2,
              Builder->CreateInBoundsGEP(
                  curblk, {context->createInt32(0), context->createInt32(i)}));
        }

        Builder->CreateStore(
            ConstantInt::get(oid_type, 0),
            Builder->CreateInBoundsGEP(alloc_index,
                                       {context->createInt32(0), target}));
        //        auto s = setIndex(context, ConstantInt::get(oid_type, 0),
        //        target); s->setMetadata("noalias", n);
      })
      .gen_else([&]() {  // ElseBB
        for (size_t i = 0; i < wantedFields.size(); ++i) {
          Builder->CreateStore(els[i], Builder->CreateInBoundsGEP(
                                           curblk, {context->createInt32(0),
                                                    context->createInt32(i)}));
        }

        //        auto s = setIndex(
        //            context, Builder->CreateAdd(indx,
        //            ConstantInt::get(oid_type, 1)), target);
        //        s->setMetadata("noalias", n);

        Builder->CreateStore(
            Builder->CreateAdd(indx, ConstantInt::get(oid_type, 1)),
            Builder->CreateInBoundsGEP(alloc_index,
                                       {context->createInt32(0), target}));
      });
  // flush remaining elements
  consume_flush(context);
}

void HashRearrange::consume_flush(ParallelContext *context) {
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

  // std::vector<Type *> args{context->getStateVars()};

  // context->pushNewCpuPipeline();

  // for (Type * t: args) context->appendStateVar(t);

  // LLVMContext & llvmContext   = context->getLLVMContext();

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  // context->setGlobalFunction();

  // IRBuilder<> * Builder       = context->getBuilder    ();
  // BasicBlock  * insBB         = Builder->GetInsertBlock();
  // Function    * F             = insBB->getParent();
  // Get the ENTRY BLOCK
  // context->setCurrentEntryBlock(Builder->GetInsertBlock());

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "flushCond", F);

  // // Start insertion in CondBB.
  // Builder->SetInsertPoint(CondBB);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "flushBody", F);

  // Make the new basic block for the increment, inserting after current block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "flushInc", F);

  // Create the "AFTER LOOP" block and insert it.
  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "flushEnd", F);
  context->setEndingBlock(AfterBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  AllocaInst *blockN_ptr = context->CreateEntryBlockAlloca(
      F, "blockN", pg->getOIDType()->getLLVMType(llvmContext));

  IntegerType *target_type = size_type;
  if (hashProject)
    target_type = (IntegerType *)hashProject->getLLVMType(llvmContext);

  AllocaInst *mem_bucket =
      context->CreateEntryBlockAlloca(F, "target_ptr", target_type);
  Builder->CreateStore(ConstantInt::get(target_type, 0), mem_bucket);

  Builder->SetInsertPoint(CondBB);

  Value *numOfBuckets = ConstantInt::get(target_type, this->numOfBuckets);
  numOfBuckets->setName("numOfBuckets");

  Value *target = Builder->CreateLoad(mem_bucket, "target");

  Value *cond = Builder->CreateICmpSLT(target, numOfBuckets);
  // Insert the conditional branch into the end of CondBB.
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  if (hashProject) {
    // Save hash in bindings
    AllocaInst *hash_ptr =
        context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);

    Builder->CreateStore(target, hash_ptr);

    ProteusValueMemory mem_hashWrapper;
    mem_hashWrapper.mem = hash_ptr;
    mem_hashWrapper.isNull = context->createFalse();
    variableBindings[*hashProject] = mem_hashWrapper;
  }

  Value *indx = getIndex(context, target);

  Builder->CreateStore(indx, blockN_ptr);

  Value *blocks = ((ParallelContext *)context)->getStateVar(blkVar_id);
  Value *curblk = Builder->CreateInBoundsGEP(
      blocks, std::vector<Value *>{context->createInt32(0), target});

  std::vector<Value *> block_ptr_addrs;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    Value *block = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        curblk, std::vector<Value *>{context->createInt32(0),
                                     context->createInt32(i)}));
    block_ptr_addrs.push_back(block);
  }

  RecordAttribute tupCnt{wantedFields[0].getRegisteredRelName(), "activeCnt",
                         pg->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper;
  mem_cntWrapper.mem = blockN_ptr;
  mem_cntWrapper.isNull = context->createFalse();
  variableBindings[tupCnt] = mem_cntWrapper;

  Value *new_oid = Builder->CreateLoad(context->getStateVar(oidVar_id), "oid");
  Builder->CreateStore(Builder->CreateAdd(new_oid, indx),
                       context->getStateVar(oidVar_id));

  AllocaInst *new_oid_ptr =
      context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
  Builder->CreateStore(new_oid, new_oid_ptr);

  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0].getRegisteredRelName(), activeLoop, pg->getOIDType());

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = new_oid_ptr;
  mem_oidWrapper.isNull = context->createFalse();
  variableBindings[tupleIdentifier] = mem_oidWrapper;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i].getRegisteredAs(), true};

    AllocaInst *tblock_ptr = context->CreateEntryBlockAlloca(
        F, wantedFields[i].getRegisteredAttrName() + "_ptr",
        tblock.getLLVMType(llvmContext));

    Builder->CreateStore(block_ptr_addrs[i], tblock_ptr);

    ProteusValueMemory memWrapper;
    memWrapper.mem = tblock_ptr;
    memWrapper.isNull = context->createFalse();
    variableBindings[tblock] = memWrapper;
  }

  // Function * f = context->getFunction("printi");
  // Builder->CreateCall(f, indx);

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);

  Builder->SetInsertPoint(IncBB);

  Value *next = Builder->CreateAdd(target, ConstantInt::get(target_type, 1));
  Builder->CreateStore(next, mem_bucket);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateRetVoid();
}

void HashRearrange::open(Pipeline *pip) {
  auto *cnts = (size_t *)MemoryManager::mallocPinned(
      sizeof(size_t) *
      (numOfBuckets + 1));  // FIXME: is it always size_t the correct type ?

  for (int i = 0; i < numOfBuckets + 1; ++i) cnts[i] = 0;

  pip->setStateVar<size_t *>(cntVar_id, cnts);

  void **blocks = (void **)MemoryManager::mallocPinned(
      sizeof(void *) * numOfBuckets * wantedFields.size());

  for (int i = 0; i < numOfBuckets; ++i) {
    for (size_t j = 0; j < wantedFields.size(); ++j) {
      blocks[i * wantedFields.size() + j] = get_buffer(wfSizes[j] * cap);
    }
  }

  pip->setStateVar<void *>(blkVar_id, (void *)blocks);

  pip->setStateVar<size_t *>(oidVar_id, cnts + numOfBuckets);
}

void HashRearrange::close(Pipeline *pip) {
  // ((void (*)(void *)) this->flushFunc)(pip->getState());
  ((void (*)(void *))closingPip->getCompiledFunction(flushingFunc))(
      pip->getState());

  MemoryManager::freePinned(pip->getStateVar<size_t *>(cntVar_id));
  MemoryManager::freePinned(pip->getStateVar<void *>(
      blkVar_id));  // FIXME: release buffers before freeing memory!
  // oidVar is part of cntVar, so they are freed together
}
