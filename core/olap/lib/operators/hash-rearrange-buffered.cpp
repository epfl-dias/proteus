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

#include "hash-rearrange-buffered.hpp"

#include <cstdlib>

#include "lib/expressions/expressions-generator.hpp"
#include "lib/util/jit/pipeline.hpp"

#define CACHE_CAP 1024

using namespace llvm;

extern "C" {
void *get_buffer(size_t bytes);

void non_temporal_copy(char *out, char *in);
}

using namespace llvm;

void HashRearrangeBuffered::produce_(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);
  Type *cnt_type =
      PointerType::getUnqual(ArrayType::get(oid_type, numOfBuckets));
  Type *char_type = Type::getInt8Ty(context->getLLVMContext());
  Type *cptr_type = PointerType::get(PointerType::get(char_type, 0), 0);
  Type *int_type =
      PointerType::get(Type::getInt32Ty(context->getLLVMContext()), 0);

  cntVar_id = context->appendStateVar(cnt_type);

  cache_cnt_Var_id = context->appendStateVar(int_type);

  oidVar_id = context->appendStateVar(PointerType::getUnqual(oid_type));

  std::vector<Type *> block_types;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    block_types.emplace_back(
        RecordAttribute(wantedFields[i]->getRegisteredAs(), true)
            .getLLVMType(llvmContext));
    wfSizes.emplace_back(context->getSizeOf(
        wantedFields[i]->getExpressionType()->getLLVMType(llvmContext)));
  }

  Type *block_stuct = StructType::get(llvmContext, block_types);

  blkVar_id = context->appendStateVar(
      PointerType::getUnqual(ArrayType::get(block_stuct, numOfBuckets)));

  cache_Var_id = context->appendStateVar(cptr_type);

  ((ParallelContext *)context)->registerOpen(this, [this](Pipeline *pip) {
    this->open(pip);
  });
  ((ParallelContext *)context)->registerClose(this, [this](Pipeline *pip) {
    this->close(pip);
  });

  getChild()->produce(context);
}

// NOTE: no MOD hashtable_size here!
Value *HashRearrangeBuffered::hash(Value *key, Value *old_seed) {
  IRBuilder<> *Builder = context->getBuilder();

  Value *hash = key;

  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));
  hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0x85ebca6b));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 13));
  hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0xc2b2ae35));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));

  if (old_seed) {
    // boost::hash_combine
    // seed ^= hash_value(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    Value *hv = hash;

    hv = Builder->CreateAdd(hv, ConstantInt::get(hv->getType(), 0x9e3779b9));
    hv = Builder->CreateAdd(hv, Builder->CreateShl(old_seed, 6));
    hv = Builder->CreateAdd(hv, Builder->CreateLShr(old_seed, 2));
    hv = Builder->CreateXor(hv, old_seed);

    hash = hv;
  }

  return hash;
}

Value *HashRearrangeBuffered::hash(const std::vector<expression_t> &exprs,
                                   Context *const context,
                                   const OperatorState &childState) {
  ExpressionGeneratorVisitor exprGenerator(context, childState);
  Value *hash = nullptr;

  for (const auto &e : exprs) {
    if (e.getTypeID() == expressions::RECORD_CONSTRUCTION) {
      const auto &attrs =
          ((expressions::RecordConstruction *)e.getUnderlyingExpression())
              ->getAtts();

      std::vector<expression_t> exprs;
      for (const auto &attr : attrs) exprs.push_back(attr.getExpression());

      Value *hv = HashRearrangeBuffered::hash(exprs, context, childState);
      hash = HashRearrangeBuffered::hash(hv, hash);
    } else {
      ProteusValue keyWrapper = e.accept(exprGenerator);  // FIXME hash
                                                          // composite key!
      hash = HashRearrangeBuffered::hash(keyWrapper.value, hash);
    }
  }

  return hash;
}

void HashRearrangeBuffered::consume(Context *const context,
                                    const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  map<RecordAttribute, ProteusValueMemory> bindings{childState.getBindings()};

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  size_t max_width = 0;
  for (const auto &e : wantedFields) {
    std::cout << e->getExpressionType()->getType() << std::endl;
    max_width = std::max(
        max_width,
        context->getSizeOf(e->getExpressionType()->getLLVMType(llvmContext)));
  }

  cap = blockSize / max_width;
  Value *capacity = ConstantInt::get(oid_type, cap);
  Value *last_index = ConstantInt::get(oid_type, cap - 1);

  Value *cache_capacity = ConstantInt::get(int32_type, CACHE_CAP);
  Value *cache_last_index = ConstantInt::get(int32_type, CACHE_CAP - 1);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  AllocaInst *blockN_ptr =
      context->CreateEntryBlockAlloca(F, "blockN", oid_type);
  Builder->CreateStore(capacity, blockN_ptr);
  AllocaInst *ready_cnt =
      context->CreateEntryBlockAlloca(F, "readyN", int32_type);
  Builder->CreateStore(ConstantInt::get(int32_type, 0), ready_cnt);

  Value *mem_cache = ((ParallelContext *)context)->getStateVar(cache_Var_id);
  Value *mem_cache_cnt =
      ((ParallelContext *)context)->getStateVar(cache_cnt_Var_id);

  Builder->SetInsertPoint(insBB);

  map<RecordAttribute, ProteusValueMemory> *variableBindings =
      new map<RecordAttribute, ProteusValueMemory>();
  // Generate target
  ExpressionGeneratorVisitor exprGenerator{context, childState};
  Value *target = HashRearrangeBuffered::hash({hashExpr}, context, childState);
  IntegerType *target_type = (IntegerType *)target->getType();
  // Value * target            = hashExpr->accept(exprGenerator).value;
  if (hashProject) {
    // Save hash in bindings
    AllocaInst *hash_ptr =
        context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);

    Builder->CreateStore(target, hash_ptr);

    ProteusValueMemory mem_hashWrapper;
    mem_hashWrapper.mem = hash_ptr;
    mem_hashWrapper.isNull = context->createFalse();
    (*variableBindings)[*hashProject] = mem_hashWrapper;

    /*vector<Value*> ArgsV2;
    ArgsV2.push_back(target);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV2);*/
  }

  Value *numOfBucketsV = ConstantInt::get(target_type, numOfBuckets);

  target = Builder->CreateURem(target, numOfBucketsV);
  target->setName("target");

  vector<Type *> members;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
    members.push_back(tblock.getLLVMType(llvmContext));
  }
  members.push_back(target->getType());

  Value *target_cache_cnt =
      Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_cache_cnt, target));

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    Value *cache_buffer = Builder->CreateLoad(
        Builder->CreateInBoundsGEP(mem_cache, context->createInt32(i)));
    Value *cache_buffer_cast = Builder->CreatePointerCast(
        cache_buffer,
        PointerType::get(
            wantedFields[i]->getExpressionType()->getLLVMType(llvmContext), 0));

    Value *el_ptr =
        Builder->CreateInBoundsGEP(cache_buffer_cast, target_cache_cnt);

    ExpressionGeneratorVisitor exprGenerator(context, childState);
    ProteusValue valWrapper = wantedFields[i]->accept(exprGenerator);
    Value *el = valWrapper.value;

    Builder->CreateStore(el, el_ptr);
  }

  Value *target_cache_cnt_next =
      Builder->CreateAdd(target_cache_cnt, context->createInt32(1));

  Value *cache_capacity_mask =
      Builder->CreateAnd(target_cache_cnt, cache_last_index);
  Value *flush_cond =
      Builder->CreateICmpUGE(cache_capacity_mask, cache_last_index);

  BasicBlock *flushBB = BasicBlock::Create(llvmContext, "flushCache", F);
  BasicBlock *incBB = BasicBlock::Create(llvmContext, "incCache", F);
  BasicBlock *mergecBB = BasicBlock::Create(llvmContext, "mergeCache", F);

  Builder->CreateCondBr(flush_cond, flushBB, incBB);

  Builder->SetInsertPoint(incBB);
  Builder->CreateStore(target_cache_cnt_next,
                       Builder->CreateInBoundsGEP(mem_cache_cnt, target));
  Builder->CreateBr(mergecBB);

  Builder->SetInsertPoint(flushBB);
  Builder->CreateStore(Builder->CreateMul(cache_capacity, target),
                       Builder->CreateInBoundsGEP(mem_cache_cnt, target));

  StructType *partition = StructType::get(llvmContext, members);
  Value *ready = context->CreateEntryBlockAlloca(
      F, "complete_partitions", ArrayType::get(partition, 1024));

  // Value * indexes = Builder->CreateLoad(((ParallelContext *)
  // context)->getStateVar(cntVar_id), "indexes");

  // indexes->dump();
  // indexes->getType()->dump();
  // ((ParallelContext *) context)->getStateVar(cntVar_id)->getType()->dump();
  Value *indx_addr = Builder->CreateInBoundsGEP(
      ((ParallelContext *)context)->getStateVar(cntVar_id),
      std::vector<Value *>{context->createInt32(0), target});
  Value *indx = Builder->CreateLoad(indx_addr);
  // Value * indx      = Builder->Load(indx_addr);

  Value *blocks = ((ParallelContext *)context)->getStateVar(blkVar_id);
  Value *curblk = Builder->CreateInBoundsGEP(
      blocks, std::vector<Value *>{context->createInt32(0), target});

  Value *cache_offset = Builder->CreateMul(cache_capacity, target);
  AllocaInst *mem_loop_offset =
      context->CreateEntryBlockAlloca(F, "mem_loop_offset", int32_type);
  Value *loop_end = cache_capacity;

  std::vector<Value *> els;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    Value *block = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        curblk, std::vector<Value *>{context->createInt32(0),
                                     context->createInt32(i)}));
    Value *cache_buffer = Builder->CreateLoad(
        Builder->CreateInBoundsGEP(mem_cache, context->createInt32(i)));
    Value *cache_buffer_cast = Builder->CreatePointerCast(
        cache_buffer,
        PointerType::get(
            wantedFields[i]->getExpressionType()->getLLVMType(llvmContext), 0));

    Value *in_block_ptr = Builder->CreateInBoundsGEP(block, indx);
    Value *in_cache_ptr =
        Builder->CreateInBoundsGEP(cache_buffer_cast, cache_offset);

    Type *charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

    Function *copy_nt = context->getFunction("nonTemporalCopy");
    vector<Value *> argsV;
    argsV.push_back(Builder->CreatePointerCast(in_block_ptr, charPtrType));
    argsV.push_back(Builder->CreatePointerCast(in_cache_ptr, charPtrType));
    argsV.push_back(context->createInt32(wantedFields[i]
                                             ->getExpressionType()
                                             ->getLLVMType(llvmContext)
                                             ->getPrimitiveSizeInBits() /
                                         8));
    Builder->CreateCall(copy_nt, argsV);

    /*Builder->CreateStore(context->createInt32(0), mem_loop_offset);

    BasicBlock *loopCondBB  = BasicBlock::Create(llvmContext, "loopcond", F);
    BasicBlock *loopMainBB  = BasicBlock::Create(llvmContext, "loopmain", F);
    BasicBlock *loopMergeBB = BasicBlock::Create(llvmContext, "loopmerge", F);

    Builder->CreateBr(loopCondBB);

    Builder->SetInsertPoint(loopCondBB);
    Value * loop_offset = Builder->CreateLoad(mem_loop_offset);
    Value * loop_cond = Builder->CreateICmpSLT(loop_offset, loop_end);
    Builder->CreateCondBr(loop_cond, loopMainBB, loopMergeBB);

    Builder->SetInsertPoint(loopMainBB);
    Value * el_cache_ptr = Builder->CreateInBoundsGEP(in_cache_ptr,
    loop_offset); Value * el_block_ptr =
    Builder->CreateInBoundsGEP(in_block_ptr, loop_offset);
    Builder->CreateStore(Builder->CreateLoad(el_cache_ptr), el_block_ptr);
    Builder->CreateStore(Builder->CreateAdd(loop_offset,
    context->createInt32(1)), mem_loop_offset); Builder->CreateBr(loopCondBB);

    Builder->SetInsertPoint(loopMergeBB);*/

    els.push_back(block);
  }

  Builder->CreateStore(
      Builder->CreateAdd(indx, ConstantInt::get(oid_type, CACHE_CAP)),
      indx_addr);

  // if indx  >= vectorSize - 1
  BasicBlock *fullBB = BasicBlock::Create(llvmContext, "propagate", F);
  BasicBlock *elseBB = BasicBlock::Create(llvmContext, "else", F);
  BasicBlock *mergeBB = BasicBlock::Create(llvmContext, "merge", F);

  Value *cond = Builder->CreateICmpUGE(
      Builder->CreateAdd(indx, ConstantInt::get(oid_type, CACHE_CAP)),
      last_index);

  Builder->CreateCondBr(cond, fullBB, elseBB);

  Builder->SetInsertPoint(fullBB);

  RecordAttribute tupCnt =
      RecordAttribute(wantedFields[0]->getRegisteredRelName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper;
  mem_cntWrapper.mem = blockN_ptr;
  mem_cntWrapper.isNull = context->createFalse();
  (*variableBindings)[tupCnt] = mem_cntWrapper;

  Value *new_oid = Builder->CreateLoad(
      ((ParallelContext *)context)->getStateVar(oidVar_id), "oid");
  Builder->CreateStore(Builder->CreateAdd(new_oid, capacity),
                       ((ParallelContext *)context)->getStateVar(oidVar_id));

  AllocaInst *new_oid_ptr =
      context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
  Builder->CreateStore(new_oid, new_oid_ptr);

  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRegisteredRelName(), activeLoop, pg->getOIDType());

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = new_oid_ptr;
  mem_oidWrapper.isNull = context->createFalse();
  (*variableBindings)[tupleIdentifier] = mem_oidWrapper;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};

    AllocaInst *tblock_ptr = context->CreateEntryBlockAlloca(
        F, wantedFields[i]->getRegisteredAttrName() + "_ptr",
        tblock.getLLVMType(llvmContext));

    Builder->CreateStore(els[i], tblock_ptr);

    ProteusValueMemory memWrapper;
    memWrapper.mem = tblock_ptr;
    memWrapper.isNull = context->createFalse();
    (*variableBindings)[tblock] = memWrapper;
  }

  OperatorState state{*this, *variableBindings};
  getParent()->consume(context, state);

  Function *get_buffer = context->getFunction("get_buffer");

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
    Value *size = context->createSizeT(
        cap *
        context->getSizeOf(
            wantedFields[i]->getExpressionType()->getLLVMType(llvmContext)));

    Value *new_buff =
        Builder->CreateCall(get_buffer, std::vector<Value *>{size});

    new_buff =
        Builder->CreateBitCast(new_buff, tblock.getLLVMType(llvmContext));

    Builder->CreateStore(
        new_buff, Builder->CreateInBoundsGEP(
                      curblk, std::vector<Value *>{context->createInt32(0),
                                                   context->createInt32(i)}));
  }

  Builder->CreateStore(ConstantInt::get(oid_type, 0), indx_addr);

  Builder->CreateBr(mergeBB);

  // else
  Builder->SetInsertPoint(elseBB);

  Builder->CreateBr(mergeBB);

  // merge
  Builder->SetInsertPoint(mergeBB);
  Builder->CreateBr(mergecBB);

  Builder->SetInsertPoint(mergecBB);

  // context->getModule()->dump();

  // flush remaining elements
  consume_flush1();
}

void HashRearrangeBuffered::consume_flush1() {
  save_current_blocks_and_restore_at_exit_scope blks{context};
  LLVMContext &llvmContext = context->getLLVMContext();

  flushingFunc1 = (*context)->createHelperFunction(
      "flush1", std::vector<Type *>{}, std::vector<bool>{},
      std::vector<bool>{});
  closingPip1 = (context->operator->());
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();
  // Get the ENTRY BLOCK
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  size_t max_width = 0;
  for (const auto &e : wantedFields) {
    std::cout << e->getExpressionType()->getType() << std::endl;
    max_width = std::max(
        max_width,
        context->getSizeOf(e->getExpressionType()->getLLVMType(llvmContext)));
  }

  cap = blockSize / max_width;
  Value *capacity = ConstantInt::get(oid_type, cap);
  Value *last_index = ConstantInt::get(oid_type, cap - 1);

  Value *cache_capacity = ConstantInt::get(int32_type, CACHE_CAP);
  Value *cache_last_index = ConstantInt::get(int32_type, CACHE_CAP - 1);

  IntegerType *target_type = size_type;
  if (hashProject)
    target_type = (IntegerType *)hashProject->getLLVMType(llvmContext);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  AllocaInst *blockN_ptr =
      context->CreateEntryBlockAlloca(F, "blockN", oid_type);
  Builder->CreateStore(capacity, blockN_ptr);
  AllocaInst *ready_cnt =
      context->CreateEntryBlockAlloca(F, "readyN", int32_type);
  Builder->CreateStore(ConstantInt::get(int32_type, 0), ready_cnt);

  Value *mem_cache = ((ParallelContext *)context)->getStateVar(cache_Var_id);
  Value *mem_cache_cnt =
      ((ParallelContext *)context)->getStateVar(cache_cnt_Var_id);

  // Builder->SetInsertPoint(insBB);

  Value *numOfBucketsV = ConstantInt::get(target_type, numOfBuckets);
  Value *mem_target =
      context->CreateEntryBlockAlloca(F, "mem_target", int32_type);
  Builder->CreateStore(ConstantInt::get(int32_type, 0), mem_target);

  BasicBlock *LoopCondBB = BasicBlock::Create(llvmContext, "cond", F);
  BasicBlock *LoopMainBB = BasicBlock::Create(llvmContext, "main", F);
  BasicBlock *LoopMergeBB = BasicBlock::Create(llvmContext, "merge", F);

  context->setEndingBlock(LoopMergeBB);
  Builder->SetInsertPoint(LoopCondBB);
  Value *target = Builder->CreateLoad(mem_target);
  Value *loopcond = Builder->CreateICmpSLT(target, numOfBucketsV);

  Builder->CreateCondBr(loopcond, LoopMainBB, LoopMergeBB);

  Builder->SetInsertPoint(LoopMainBB);

  map<RecordAttribute, ProteusValueMemory> *variableBindings =
      new map<RecordAttribute, ProteusValueMemory>();
  // Generate target

  // Value * target            = hashExpr->accept(exprGenerator).value;
  if (hashProject) {
    // Save hash in bindings
    AllocaInst *hash_ptr =
        context->CreateEntryBlockAlloca(F, "hash_ptr", target_type);

    Builder->CreateStore(target, hash_ptr);

    ProteusValueMemory mem_hashWrapper;
    mem_hashWrapper.mem = hash_ptr;
    mem_hashWrapper.isNull = context->createFalse();
    (*variableBindings)[*hashProject] = mem_hashWrapper;
  }

  vector<Type *> members;
  members.push_back(target->getType());

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
    members.push_back(tblock.getLLVMType(llvmContext));
  }
  members.push_back(target->getType());

  StructType *partition = StructType::get(llvmContext, members);
  Value *ready = context->CreateEntryBlockAlloca(
      F, "complete_partitions", ArrayType::get(partition, 1024));

  Value *indx_addr = Builder->CreateInBoundsGEP(
      ((ParallelContext *)context)->getStateVar(cntVar_id),
      std::vector<Value *>{context->createInt32(0), target});

  // Value * indx      = Builder->Load(indx_addr);

  Value *blocks = ((ParallelContext *)context)->getStateVar(blkVar_id);
  Value *curblk = Builder->CreateInBoundsGEP(
      blocks, std::vector<Value *>{context->createInt32(0), target});

  Value *cache_offset = Builder->CreateMul(cache_capacity, target);
  AllocaInst *mem_loop_offset =
      context->CreateEntryBlockAlloca(F, "mem_loop_offset", int32_type);
  Value *loop_end = Builder->CreateSub(
      Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_cache_cnt, target)),
      cache_offset);

  Builder->CreateStore(context->createInt32(0), mem_loop_offset);

  BasicBlock *InnerLoopCondBB =
      BasicBlock::Create(llvmContext, "InnerLoopcond", F);
  BasicBlock *InnerLoopMainBB =
      BasicBlock::Create(llvmContext, "InnerLoopmain", F);
  BasicBlock *InnerLoopMergeBB =
      BasicBlock::Create(llvmContext, "InnerLoopmerge", F);

  Builder->CreateBr(InnerLoopCondBB);

  Builder->SetInsertPoint(InnerLoopCondBB);
  Value *indx = Builder->CreateLoad(indx_addr);
  Value *loop_offset = Builder->CreateLoad(mem_loop_offset);
  Value *loop_cond = Builder->CreateICmpSLT(loop_offset, loop_end);
  Builder->CreateCondBr(loop_cond, InnerLoopMainBB, InnerLoopMergeBB);

  Builder->SetInsertPoint(InnerLoopMainBB);

  std::vector<Value *> els;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    Value *block = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        curblk, std::vector<Value *>{context->createInt32(0),
                                     context->createInt32(i)}));
    Value *cache_buffer = Builder->CreateLoad(
        Builder->CreateInBoundsGEP(mem_cache, context->createInt32(i)));
    Value *cache_buffer_cast = Builder->CreatePointerCast(
        cache_buffer,
        PointerType::get(
            wantedFields[i]->getExpressionType()->getLLVMType(llvmContext), 0));

    Value *in_block_ptr = Builder->CreateInBoundsGEP(block, indx);
    Value *in_cache_ptr =
        Builder->CreateInBoundsGEP(cache_buffer_cast, cache_offset);

    Value *el_cache_ptr = Builder->CreateInBoundsGEP(in_cache_ptr, loop_offset);
    Value *el_block_ptr = in_block_ptr;
    Builder->CreateStore(Builder->CreateLoad(el_cache_ptr), el_block_ptr);

    els.push_back(block);
  }

  // if indx  >= vectorSize - 1
  BasicBlock *fullBB = BasicBlock::Create(llvmContext, "propagate", F);
  BasicBlock *elseBB = BasicBlock::Create(llvmContext, "else", F);
  BasicBlock *mergeBB = BasicBlock::Create(llvmContext, "merge", F);

  Value *cond = Builder->CreateICmpUGE(indx, last_index);

  Builder->CreateCondBr(cond, fullBB, elseBB);

  Builder->SetInsertPoint(fullBB);

  RecordAttribute tupCnt =
      RecordAttribute(wantedFields[0]->getRegisteredRelName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper;
  mem_cntWrapper.mem = blockN_ptr;
  mem_cntWrapper.isNull = context->createFalse();
  (*variableBindings)[tupCnt] = mem_cntWrapper;

  Value *new_oid = Builder->CreateLoad(
      ((ParallelContext *)context)->getStateVar(oidVar_id), "oid");
  Builder->CreateStore(Builder->CreateAdd(new_oid, capacity),
                       ((ParallelContext *)context)->getStateVar(oidVar_id));

  AllocaInst *new_oid_ptr =
      context->CreateEntryBlockAlloca(F, "new_oid_ptr", oid_type);
  Builder->CreateStore(new_oid, new_oid_ptr);

  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRegisteredRelName(), activeLoop, pg->getOIDType());

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = new_oid_ptr;
  mem_oidWrapper.isNull = context->createFalse();
  (*variableBindings)[tupleIdentifier] = mem_oidWrapper;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};

    AllocaInst *tblock_ptr = context->CreateEntryBlockAlloca(
        F, wantedFields[i]->getRegisteredAttrName() + "_ptr",
        tblock.getLLVMType(llvmContext));

    Builder->CreateStore(els[i], tblock_ptr);

    ProteusValueMemory memWrapper;
    memWrapper.mem = tblock_ptr;
    memWrapper.isNull = context->createFalse();
    (*variableBindings)[tblock] = memWrapper;
  }

  OperatorState state{*this, *variableBindings};
  getParent()->consume(context, state);

  Function *get_buffer = context->getFunction("get_buffer");

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};
    Value *size = context->createSizeT(
        cap *
        context->getSizeOf(
            wantedFields[i]->getExpressionType()->getLLVMType(llvmContext)));

    Value *new_buff =
        Builder->CreateCall(get_buffer, std::vector<Value *>{size});

    new_buff =
        Builder->CreateBitCast(new_buff, tblock.getLLVMType(llvmContext));

    Builder->CreateStore(
        new_buff, Builder->CreateInBoundsGEP(
                      curblk, std::vector<Value *>{context->createInt32(0),
                                                   context->createInt32(i)}));
  }

  Builder->CreateStore(ConstantInt::get(oid_type, 0), indx_addr);

  Builder->CreateBr(mergeBB);

  // else
  Builder->SetInsertPoint(elseBB);

  Builder->CreateStore(Builder->CreateAdd(indx, ConstantInt::get(oid_type, 1)),
                       indx_addr);

  Builder->CreateBr(mergeBB);

  // merge
  Builder->SetInsertPoint(mergeBB);
  Builder->CreateStore(Builder->CreateAdd(loop_offset, context->createInt32(1)),
                       mem_loop_offset);

  Builder->CreateBr(InnerLoopCondBB);

  Builder->SetInsertPoint(InnerLoopMergeBB);
  Builder->CreateStore(Builder->CreateAdd(target, context->createInt32(1)),
                       mem_target);
  Builder->CreateBr(LoopCondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(LoopCondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateRetVoid();

  // context->getModule()->dump();

  // flush remaining elements
  consume_flush();
}

void HashRearrangeBuffered::consume_flush() {
  save_current_blocks_and_restore_at_exit_scope blks{context};
  LLVMContext &llvmContext = context->getLLVMContext();

  flushingFunc2 = (*context)->createHelperFunction(
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
      Catalog::getInstance().getPlugin(wantedFields[0]->getRegisteredRelName());
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

  Value *indx_addr = Builder->CreateInBoundsGEP(
      ((ParallelContext *)context)->getStateVar(cntVar_id),
      std::vector<Value *>{context->createInt32(0), target});
  Value *indx = Builder->CreateLoad(indx_addr);

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

  RecordAttribute tupCnt{wantedFields[0]->getRegisteredRelName(), "activeCnt",
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
      wantedFields[0]->getRegisteredRelName(), activeLoop, pg->getOIDType());

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = new_oid_ptr;
  mem_oidWrapper.isNull = context->createFalse();
  variableBindings[tupleIdentifier] = mem_oidWrapper;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute tblock{wantedFields[i]->getRegisteredAs(), true};

    AllocaInst *tblock_ptr = context->CreateEntryBlockAlloca(
        F, wantedFields[i]->getRegisteredAttrName() + "_ptr",
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

void HashRearrangeBuffered::open(Pipeline *pip) {
  std::cout << "rerarange" << std::endl;

  size_t *cnts = (size_t *)malloc(
      sizeof(size_t) *
      (numOfBuckets + 1));  // FIXME: is it always size_t the correct type ?
  int32_t *cache_cnts = (int32_t *)malloc(sizeof(int32_t) * (numOfBuckets + 1));

  for (int i = 0; i < numOfBuckets + 1; ++i) {
    cnts[i] = 0;
    cache_cnts[i] = i * CACHE_CAP;
  }

  pip->setStateVar<size_t *>(cntVar_id, cnts);
  pip->setStateVar<int32_t *>(cache_cnt_Var_id, cache_cnts);

  void **blocks =
      (void **)malloc(sizeof(void *) * numOfBuckets * wantedFields.size());
  char **cache = (char **)malloc(sizeof(void *) * wantedFields.size());

  for (int i = 0; i < numOfBuckets; ++i) {
    for (size_t j = 0; j < wantedFields.size(); ++j) {
      blocks[i * wantedFields.size() + j] = get_buffer(wfSizes[j] * cap);
    }
  }

  for (size_t j = 0; j < wantedFields.size(); ++j) {
    // cache[j] = (char*) malloc(wfSizes[j] * CACHE_CAP * numOfBuckets + 256);
    posix_memalign((void **)&cache[j], 256,
                   wfSizes[j] * CACHE_CAP * numOfBuckets);
  }

  pip->setStateVar<void *>(blkVar_id, (void *)blocks);

  pip->setStateVar<char **>(cache_Var_id, (char **)cache);

  pip->setStateVar<size_t *>(oidVar_id, cnts + numOfBuckets);
}

void HashRearrangeBuffered::close(Pipeline *pip) {
  // ((void (*)(void *)) this->flushFunc)(pip->getState());
  ((void (*)(void *))closingPip1->getCompiledFunction(flushingFunc1))(
      pip->getState());
  ((void (*)(void *))closingPip->getCompiledFunction(flushingFunc2))(
      pip->getState());

  free(pip->getStateVar<int32_t *>(cache_cnt_Var_id));

  char **cache = pip->getStateVar<char **>(cache_Var_id);
  for (size_t j = 0; j < wantedFields.size(); ++j) free(cache[j]);
  free(cache);

  free(pip->getStateVar<size_t *>(cntVar_id));
  free(pip->getStateVar<void *>(
      blkVar_id));  // FIXME: release buffers before freeing memory!
  // oidVar is part of cntVar, so they are freed together
}
