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

#include "radix-join.hpp"

#include "lib/util/jit/pipeline.hpp"

using namespace llvm;

RadixJoin::RadixJoin(const expressions::BinaryExpression &predicate,
                     Operator *leftChild, Operator *rightChild,
                     Context *const context, const char *opLabel,
                     Materializer &matLeft, Materializer &matRight)
    : BinaryOperator(leftChild, rightChild),
      context((ParallelContext *const)context),
      htLabel(opLabel) {
  assert(dynamic_cast<ParallelContext *const>(context) &&
         "Should update caller to use the new context!");

  LLVMContext &llvmContext = context->getLLVMContext();

  // only 32bit integers keys are supported
  if (context->getSizeOf(
          predicate.getLeftOperand().getExpressionType()->getLLVMType(
              llvmContext)) != 32 / 8) {
    string error_msg = "--A-- Only INT32 keys considered atm";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  Type *int64_type = Type::getInt64Ty(llvmContext);
  Type *int32_type = Type::getInt32Ty(llvmContext);

  Type *int32_ptr_type = PointerType::getUnqual(int32_type);

  /* What the type of internal radix HT per cluster is    */
  /* (int32*, int32*, unit32_t, void*, int32) */
  vector<Type *> htRadixClusterMembers;
  htRadixClusterMembers.push_back(int32_ptr_type);
  htRadixClusterMembers.push_back(int32_ptr_type);
  /* LLVM does not make a distinction between signed and unsigned integer type:
   * Both are lowered to i32
   */
  htRadixClusterMembers.push_back(int32_type);
  htRadixClusterMembers.push_back(int32_type);
  htClusterType =
      StructType::get(context->getLLVMContext(), htRadixClusterMembers);

  /* What the type of HT entries is */
  /* (int32, size_t) */
  vector<Type *> htEntryMembers;
  htEntryMembers.push_back(int32_type);
  htEntryMembers.push_back(int64_type);
  int htEntrySize = sizeof(int) + sizeof(size_t);
  htEntryType = StructType::get(context->getLLVMContext(), htEntryMembers);

  keyType = int32_type;

  /* Arbitrary initial buffer sizes */
  /* No realloc will be required with these sizes for synthetic large-scale
   * numbers */
#ifdef LOCAL_EXEC
  size_t sizeR = 10000;
  size_t sizeS = 15000;
#else
  size_t sizeR = 20000000000;
  size_t sizeS = 30000000000;
#endif

  size_t kvSizeR = sizeR;  // * htEntrySize;
  size_t kvSizeS = sizeS;  // * htEntrySize;

  buildR =
      new RadixJoinBuild(predicate.getLeftOperand(), leftChild, this->context,
                         htLabel, matLeft, htEntryType, sizeR, kvSizeR);

  buildS =
      new RadixJoinBuild(predicate.getRightOperand(), rightChild, this->context,
                         htLabel, matRight, htEntryType, sizeS, kvSizeS);
}

RadixJoin::~RadixJoin() {
  LOG(INFO) << "Collapsing RadixJoin operator";
  //  Can't do garbage collection here, need to do it from codegen
}

int32_t *RadixJoinBuild::getClusterCounts(Pipeline *pip) {
  assert(pip->getGroup() < 128);
  int32_t *cnts = clusterCounts[pip->getGroup()];
  assert(cnts);
  return cnts;
}

void RadixJoinBuild::registerClusterCounts(Pipeline *pip, int32_t *cnts) {
  assert(pip->getGroup() < 128);
  assert(cnts);
  clusterCounts[pip->getGroup()] = cnts;
}

void *RadixJoinBuild::getHTMemKV(Pipeline *pip) {
  assert(pip->getGroup() < 128);
  void *mem_kv = ht_mem_kv[pip->getGroup()];
  assert(mem_kv);
  return mem_kv;
}

void RadixJoinBuild::registerHTMemKV(Pipeline *pip, void *mem_kv) {
  assert(pip->getGroup() < 128);
  assert(mem_kv);
  ht_mem_kv[pip->getGroup()] = mem_kv;
}

void *RadixJoinBuild::getRelationMem(Pipeline *pip) {
  assert(pip->getGroup() < 128);
  void *rel_mem = relation_mem[pip->getGroup()];
  assert(rel_mem);
  return rel_mem;
}

void RadixJoinBuild::registerRelationMem(Pipeline *pip, void *rel_mem) {
  assert(pip->getGroup() < 128);
  assert(rel_mem);
  relation_mem[pip->getGroup()] = rel_mem;
}

extern "C" {

int32_t *getClusterCounts(Pipeline *pip, RadixJoinBuild *b) {
  return b->getClusterCounts(pip);
}

void registerClusterCounts(Pipeline *pip, int32_t *cnts, RadixJoinBuild *b) {
  return b->registerClusterCounts(pip, cnts);
}

void *getHTMemKV(Pipeline *pip, RadixJoinBuild *b) {
  return b->getHTMemKV(pip);
}

void registerHTMemKV(Pipeline *pip, void *mem_kv, RadixJoinBuild *b) {
  return b->registerHTMemKV(pip, mem_kv);
}

void *getRelationMem(Pipeline *pip, RadixJoinBuild *b) {
  return b->getRelationMem(pip);
}

void registerRelationMem(Pipeline *pip, void *rel_mem, RadixJoinBuild *b) {
  return b->registerRelationMem(pip, rel_mem);
}
}

void RadixJoin::produce_(ParallelContext *context) {
  runRadix();

  context->popPipeline();

  auto flush_pip = context->removeLatestPipeline();
  flush_fun = flush_pip->getKernel();

  context->pushPipeline();

  Operator *leftChild = getLeftChild();
  leftChild->setParent(buildR);
  buildR->setParent(this);
  setLeftChild(buildR);

  buildR->produce(context);

  context->popPipeline();

  // updateRelationPointers();
  // /* XXX Place info in cache */
  // if (!cachedLeft) {
  //     placeInCache(matLeft, true);
  // }
  // if (!cachedRight) {
  //     placeInCache(matRight, false);
  // }
  // //Still need HTs...
  // // Should I mat. them too?

  // auto radix_pip = context->getCurrentPipeline();
  context->pushPipeline();

  Operator *rightChild = getRightChild();
  rightChild->setParent(buildS);
  buildS->setParent(this);
  setRightChild(buildS);

  // context->appendStateVar(llvm::Type *ptype)

  // context->appendStateVar(
  //     Type::getInt32Ty(context->getLLVMContext()),
  //     [=](llvm::Value *){
  //         return
  //         UndefValue::get(Type::getInt32Ty(context->getLLVMContext()));
  //     },
  //     [=](llvm::Value *, llvm::Value * s){
  //         IRBuilder<> * Builder = context->getBuilder();

  //         Type  * charPtrType =
  //         Type::getInt8PtrTy(context->getLLVMContext());

  //         Function * f = context->getFunction("subpipeline_consume");
  //         FunctionType * f_t  = f->getFunctionType();

  //         Type  * substate_t  = f_t->getParamType(f_t->getNumParams()-1);

  //         Value * substate    =
  //         Builder->CreateBitCast(context->getSubStateVar(), substate_t);

  //         Builder->CreateCall(f, vector<Value *>{substate});
  //     }
  // );

  context->setChainedPipeline(flush_pip);

  buildS->produce(context);
}

void RadixJoin::consume(Context *const context,
                        const OperatorState &childState) {
  assert(false &&
         "Function should not be called! Are RadixJoinBuilders "
         "correctly set as children of this operator?");
}

void RadixJoin::runRadix() const {
  LLVMContext &llvmContext = context->getLLVMContext();
  Catalog &catalog = Catalog::getInstance();

  Type *int64_type = Type::getInt64Ty(llvmContext);
  Type *int32_type = Type::getInt32Ty(llvmContext);
  PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);

  auto clusterCountR_id = context->appendStateVar(
      PointerType::getUnqual(int32_type),
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildR);
        Function *clusterCnts = context->getFunction("getClusterCounts");
        return Builder->CreateCall(clusterCnts, vector<Value *>{pip, build});
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("free");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "clusterCountR");

  auto clusterCountS_id = context->appendStateVar(
      PointerType::getUnqual(int32_type),
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildS);
        Function *clusterCnts = context->getFunction("getClusterCounts");
        return Builder->CreateCall(clusterCnts, vector<Value *>{pip, build});
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("free");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "clusterCountS");  // FIXME: read-only, we do not even have to maintain it
                         // as state variable, but where do we get pip from ?

  auto htR_mem_kv_id = context->appendStateVar(
      PointerType::getUnqual(htEntryType),
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildR);
        Function *ht_mem_kv = context->getFunction("getHTMemKV");
        Value *char_ht_mem =
            Builder->CreateCall(ht_mem_kv, vector<Value *>{pip, build});
        return Builder->CreateBitCast(char_ht_mem,
                                      PointerType::getUnqual(htEntryType));
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("releaseMemoryChunk");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "htR_mem_kv");  // FIXME: read-only, we do not even have to maintain it as
                      // state variable

  auto htS_mem_kv_id = context->appendStateVar(
      PointerType::getUnqual(htEntryType),
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildS);
        Function *ht_mem_kv = context->getFunction("getHTMemKV");
        Value *char_ht_mem =
            Builder->CreateCall(ht_mem_kv, vector<Value *>{pip, build});
        return Builder->CreateBitCast(char_ht_mem,
                                      PointerType::getUnqual(htEntryType));
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("releaseMemoryChunk");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "htS_mem_kv");  // FIXME: read-only, we do not even have to maintain it as
                      // state variable

  auto relR_mem_relation_id = context->appendStateVar(
      char_ptr_type,
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildR);
        Function *rel_mem = context->getFunction("getRelationMem");
        return Builder->CreateCall(rel_mem, vector<Value *>{pip, build});
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("releaseMemoryChunk");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "relR_mem_relation");  // FIXME: read-only, we do not even have to
                             // maintain it as state variable

  auto relS_mem_relation_id = context->appendStateVar(
      char_ptr_type,
      [=](llvm::Value *pip) {
        LLVMContext &llvmContext = context->getLLVMContext();
        IRBuilder<> *Builder = context->getBuilder();

        Value *build = context->CastPtrToLlvmPtr(char_ptr_type, buildS);
        Function *rel_mem = context->getFunction("getRelationMem");
        return Builder->CreateCall(rel_mem, vector<Value *>{pip, build});
      },
      [=](llvm::Value *, llvm::Value *s) {
        Function *f = context->getFunction("releaseMemoryChunk");
        IRBuilder<> *Builder = context->getBuilder();
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "relS_mem_relation");  // FIXME: read-only, we do not even have to
                             // maintain it as state variable

  context->setGlobalFunction();
  Function *F = context->getGlobalFunction();

  Value *val_zero = context->createInt32(0);
  Value *val_one = context->createInt32(1);
#ifdef DEBUGRADIX
  Function *debugInt = context->getFunction("printi");
  Function *debugInt64 = context->getFunction("printi64");

  vector<Value *> ArgsV0;
  ArgsV0.push_back(context->createInt32(665));
  Builder->CreateCall(debugInt, ArgsV0);
#endif
  // FIXME: NEEDED
  // /* Partition and Cluster 'R' (the corresponding htEntries) */
  Value *clusterCountR = context->getStateVar(clusterCountR_id);
  // // Partition and Cluster 'S' (the corresponding htEntries)
  Value *clusterCountS = context->getStateVar(clusterCountS_id);

#ifdef DEBUGRADIX
  ArgsV0.clear();
  ArgsV0.push_back(context->createInt32(666));
  Builder->CreateCall(debugInt, ArgsV0);
#endif

  IRBuilder<> *Builder = context->getBuilder();
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  std::string relName = getMaterializerRight()
                            .getWantedExpressions()
                            .back()
                            .getRegisteredRelName();
  Plugin *pg = Catalog::getInstance().getPlugin(relName);
  ExpressionType *oid_type = pg->getOIDType();
  IntegerType *llvm_oid_type =
      (IntegerType *)oid_type->getLLVMType(llvmContext);
  AllocaInst *mem_outCount =
      Builder->CreateAlloca(llvm_oid_type, nullptr, "join_oid");
  AllocaInst *mem_rCount = Builder->CreateAlloca(int32_type, nullptr, "rCount");
  AllocaInst *mem_sCount = Builder->CreateAlloca(int32_type, nullptr, "sCount");
  AllocaInst *mem_clusterCount =
      Builder->CreateAlloca(int32_type, nullptr, "clusterCount");
  Builder->CreateStore(ConstantInt::get(llvm_oid_type, 0), mem_outCount);
  Builder->CreateStore(val_zero, mem_rCount);
  Builder->CreateStore(val_zero, mem_sCount);
  Builder->CreateStore(val_zero, mem_clusterCount);

  uint32_t clusterNo = (1 << NUM_RADIX_BITS);
  Value *val_clusterNo = context->createInt32(clusterNo);

  /* Request memory for HT(s) construction        */
  /* Note: Does not allocate mem. for buckets too */
  size_t htSize = (1 << NUM_RADIX_BITS) * sizeof(HT);

  Builder->CreateAlloca(htClusterType, nullptr, "HTimpl");
  PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
  Function *f_getMemoryChunk = context->getFunction("getMemoryChunk");
  Value *HT_mem = Builder->CreateCall(
      f_getMemoryChunk, vector<Value *>{context->createSizeT(htSize)});
  Value *val_htPerCluster = Builder->CreateBitCast(HT_mem, htClusterPtrType);

  AllocaInst *mem_probesNo =
      Builder->CreateAlloca(int32_type, nullptr, "mem_counter");
  Builder->CreateStore(val_zero, mem_probesNo);

  /**
   * ACTUAL PROBES
   */

  /* Loop through clusters */
  /* for (i = 0; i < (1 << NUM_RADIX_BITS); i++) */

  BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
  context->CreateForLoop("clusterLoopCond", "clusterLoopBody", "clusterLoopInc",
                         "clusterLoopEnd", &loopCond, &loopBody, &loopInc,
                         &loopEnd);
  context->setEndingBlock(loopEnd);

  /* 1. Loop Condition - Unsigned integers operation */
  // Builder->CreateBr(loopCond);
  Builder->SetInsertPoint(loopCond);
  Value *val_clusterCount = Builder->CreateLoad(
      mem_clusterCount->getType()->getPointerElementType(), mem_clusterCount);
  Value *val_cond = Builder->CreateICmpULT(val_clusterCount, val_clusterNo);
  Builder->CreateCondBr(val_cond, loopBody, loopEnd);

  /* 2. Loop Body */
  Builder->SetInsertPoint(loopBody);

  /* Check cluster contents */
  /* if (R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0)
   */
  BasicBlock *ifBlock, *elseBlock;
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifNotEmptyCluster",
                              "elseEmptyCluster", &ifBlock, &elseBlock,
                              loopInc);

  {
    /* If Condition */
    Value *val_r_i_count =
        context->getArrayElem(clusterCountR, val_clusterCount);
    Value *val_s_i_count =
        context->getArrayElem(clusterCountS, val_clusterCount);
    Value *val_cond_1 = Builder->CreateICmpSGT(val_r_i_count, val_zero);
    Value *val_cond_2 = Builder->CreateICmpSGT(val_s_i_count, val_zero);
    val_cond = Builder->CreateAnd(val_cond_1, val_cond_2);

    Builder->CreateCondBr(val_cond, ifBlock, elseBlock);

    /* If clusters non-empty */
    Builder->SetInsertPoint(ifBlock);

    Value *val_rCount = Builder->CreateLoad(
        mem_rCount->getType()->getPointerElementType(), mem_rCount);
    Value *val_sCount = Builder->CreateLoad(
        mem_sCount->getType()->getPointerElementType(), mem_sCount);
#ifdef DEBUGRADIX
    /* Cluster Counts */
    vector<Value *> ArgsV0;
    ArgsV0.push_back(context->createInt32(222));
    Builder->CreateCall(debugInt, ArgsV0);

    ArgsV0.clear();
    ArgsV0.push_back(val_clusterCount);
    Builder->CreateCall(debugInt, ArgsV0);
#endif

    /* tmpR.tuples = relR->tuples + r; */
    Value *val_htR = context->getStateVar(htR_mem_kv_id);
    Value *htRshiftedPtr = Builder->CreateInBoundsGEP(
        val_htR->getType()->getNonOpaquePointerElementType(), val_htR,
        val_rCount);

    /* tmpS.tuples = relS->tuples + s; */
    Value *val_htS = context->getStateVar(htS_mem_kv_id);
    Value *htSshiftedPtr = Builder->CreateInBoundsGEP(
        val_htS->getType()->getNonOpaquePointerElementType(), val_htS,
        val_sCount);

    /* bucket_chaining_join_prepare(&tmpR, &(HT_per_cluster[i])); */
    Function *bucketChainingPrepare =
        context->getFunction("bucketChainingPrepare");

    Value *val_htPerClusterShiftedPtr = Builder->CreateInBoundsGEP(
        val_htPerCluster->getType()->getNonOpaquePointerElementType(),
        val_htPerCluster, val_clusterCount);

    // Prepare args and call function
    vector<Value *> Args;
    Args.push_back(htRshiftedPtr);
    Args.push_back(val_r_i_count);
    Args.push_back(val_htPerClusterShiftedPtr);
    Builder->CreateCall(bucketChainingPrepare, Args);

    /*
     * r += R_count_per_cluster[i];
     * s += S_count_per_cluster[i];
     */
    val_rCount = Builder->CreateAdd(val_rCount, val_r_i_count);
    val_sCount = Builder->CreateAdd(val_sCount, val_s_i_count);
    Builder->CreateStore(val_rCount, mem_rCount);
    Builder->CreateStore(val_sCount, mem_sCount);

    /* Loop over S cluster (tmpS) and use its tuples to probe HTs */
    BasicBlock *sLoopCond, *sLoopBody, *sLoopInc, *sLoopEnd;
    context->CreateForLoop("sLoopCond", "sLoopBody", "sLoopInc", "sLoopEnd",
                           &sLoopCond, &sLoopBody, &sLoopInc, &sLoopEnd);
    {
      AllocaInst *mem_j = Builder->CreateAlloca(int32_type, nullptr, "j_cnt");
      Builder->CreateStore(val_zero, mem_j);
      Builder->CreateBr(sLoopCond);

      /* Loop Condition */
      Builder->SetInsertPoint(sLoopCond);
      Value *val_j =
          Builder->CreateLoad(mem_j->getType()->getPointerElementType(), mem_j);

      val_cond = Builder->CreateICmpSLT(val_j, val_s_i_count);

      Builder->CreateCondBr(val_cond, sLoopBody, sLoopEnd);

      Builder->SetInsertPoint(sLoopBody);

#ifdef DEBUGRADIX
//          Value *val_probesNo =
//          Builder->CreateLoad(mem_probesNo->getType()->getPointerElementType(),
//          mem_probesNo); val_probesNo = Builder->CreateAdd(val_probesNo,
//          val_one); Builder->CreateStore(val_probesNo,mem_probesNo);
//
//          vector<Value*> ArgsV0;
//          ArgsV0.push_back(val_j);
//          Builder->CreateCall(debugInt,ArgsV0);
//
//          ArgsV0.clear();
//          ArgsV0.push_back(val_sCount);
//          Builder->CreateCall(debugInt,ArgsV0);
#endif
      /*
       * Break the following in pieces:
       * result += bucket_chaining_join_probe(&tmpR,
       *          &(HT_per_cluster[i]), &(tmpS.tuples[j]));
       */

      /* uint32_t idx = HASH_BIT_MODULO(s->key, ht->mask, NUM_RADIX_BITS); */
      Value *val_num_radix_bits = context->createInt32(NUM_RADIX_BITS);
      Value *val_mask = context->getStructElem(val_htPerClusterShiftedPtr, 2);
      // Get key of current s tuple (tmpS[j])
      Value *htSshiftedPtr_j = Builder->CreateInBoundsGEP(
          htSshiftedPtr->getType()->getNonOpaquePointerElementType(),
          htSshiftedPtr, val_j);
      //          Value *tuple_s_j =
      //          Builder->CreateLoad(htSshiftedPtr_j->getType()->getPointerElementType(),
      //          htSshiftedPtr_j);
      Value *val_key_s_j = context->getStructElem(htSshiftedPtr_j, 0);
      Value *val_idx =
          Builder->CreateBinOp(Instruction::And, val_key_s_j, val_mask);
      val_idx = Builder->CreateAShr(val_idx, val_num_radix_bits);

      /**
       * Checking actual hits (when applicable)
       * for(int hit = (ht->bucket)[idx]; hit > 0; hit = (ht->next)[hit-1])
       */
      BasicBlock *hitLoopCond, *hitLoopBody, *hitLoopInc, *hitLoopEnd;
      context->CreateForLoop("hitLoopCond", "hitLoopBody", "hitLoopInc",
                             "hitLoopEnd", &hitLoopCond, &hitLoopBody,
                             &hitLoopInc, &hitLoopEnd);

      {
        AllocaInst *mem_hit = Builder->CreateAlloca(int32_type, nullptr, "hit");
        //(ht->bucket)
        Value *val_bucket =
            context->getStructElem(val_htPerClusterShiftedPtr, 0);
        //(ht->bucket)[idx]
        Value *val_bucket_idx = context->getArrayElem(val_bucket, val_idx);

        Builder->CreateStore(val_bucket_idx, mem_hit);
        Builder->CreateBr(hitLoopCond);
        /* 1. Loop Condition */
        Builder->SetInsertPoint(hitLoopCond);
        Value *val_hit = Builder->CreateLoad(
            mem_hit->getType()->getPointerElementType(), mem_hit);
        val_cond = Builder->CreateICmpSGT(val_hit, val_zero);
        Builder->CreateCondBr(val_cond, hitLoopBody, hitLoopEnd);

        /* 2. Body */
        Builder->SetInsertPoint(hitLoopBody);

        /* if (s->key == Rtuples[hit - 1].key) */
        BasicBlock *ifKeyMatch;
        context->CreateIfBlock(context->getGlobalFunction(), "htMatchIfCond",
                               &ifKeyMatch, hitLoopInc);
        {
          // Rtuples[hit - 1]
          Value *val_idx_dec = Builder->CreateSub(val_hit, val_one);
          Value *htRshiftedPtr_hit = Builder->CreateInBoundsGEP(
              htRshiftedPtr->getType()->getNonOpaquePointerElementType(),
              htRshiftedPtr, val_idx_dec);
          // Rtuples[hit - 1].key
          Value *val_key_r = context->getStructElem(htRshiftedPtr_hit, 0);

          // Condition
          val_cond = Builder->CreateICmpEQ(val_key_s_j, val_key_r);
          Builder->CreateCondBr(val_cond, ifKeyMatch, hitLoopInc);

          Builder->SetInsertPoint(ifKeyMatch);

#ifdef DEBUGRADIX
          /* Printing key(s) */
//                  vector<Value*> ArgsV;
//                  ArgsV.push_back(val_key_s_j);
//                  Builder->CreateCall(debugInt, ArgsV);

//                  ArgsV.clear();
//                  ArgsV.push_back(context->createInt32(1111));
//                  Builder->CreateCall(debugInt, ArgsV);

//                  ArgsV.clear();
//                  ArgsV.push_back(val_key_r);
//                  Builder->CreateCall(debugInt, ArgsV);
#endif
          /**
           * -> RECONSTRUCT RESULTS
           * -> CALL PARENT
           */
          std::map<RecordAttribute, ProteusValueMemory> allJoinBindings;

          /* Payloads (Relative Offsets): size_t */
          /* Must be added to relR / relS accordingly */
          Value *val_payload_r_offset =
              context->getStructElem(htRshiftedPtr_hit, 1);
          Value *val_payload_s_offset =
              context->getStructElem(htSshiftedPtr_j, 1);

          /* Cast payload */
          StructType *rPayloadType = buildR->getPayloadType();
          StructType *sPayloadType = buildS->getPayloadType();
          PointerType *rPayloadPtrType = PointerType::get(rPayloadType, 0);
          PointerType *sPayloadPtrType = PointerType::get(sPayloadType, 0);

          Value *val_relR = context->getStateVar(relR_mem_relation_id);
          Value *val_relS = context->getStateVar(relS_mem_relation_id);
          Value *val_ptr_payloadR = Builder->CreateInBoundsGEP(
              val_relR->getType()->getNonOpaquePointerElementType(), val_relR,
              val_payload_r_offset);
          Value *val_ptr_payloadS = Builder->CreateInBoundsGEP(
              val_relS->getType()->getNonOpaquePointerElementType(), val_relS,
              val_payload_s_offset);
#ifdef DEBUGRADIX
          {
            /* Printing key(s) */
            vector<Value *> ArgsV;
            ArgsV.push_back(Builder->getInt32(500005));
            Builder->CreateCall(debugInt, ArgsV);
          }
#endif
          Value *mem_payload_r =
              Builder->CreateBitCast(val_ptr_payloadR, rPayloadPtrType);
          Value *val_payload_r = Builder->CreateLoad(
              mem_payload_r->getType()->getPointerElementType(), mem_payload_r);
          Value *mem_payload_s =
              Builder->CreateBitCast(val_ptr_payloadS, sPayloadPtrType);
          Value *val_payload_s = Builder->CreateLoad(
              mem_payload_s->getType()->getPointerElementType(), mem_payload_s);

          /* LEFT SIDE (RELATION R)*/
          // Retrieving activeTuple(s) from HT
          {
            int i = 0;

            for (const auto &expr2 :
                 getMaterializerLeft().getWantedExpressions()) {
              const auto &currField = expr2.getRegisteredAttrName();

              auto elem_ptr = Builder->CreateGEP(
                  mem_payload_r->getType()->getNonOpaquePointerElementType(),
                  mem_payload_r,
                  {context->createInt32(0), context->createInt32(i)});
              auto val_field = Builder->CreateLoad(
                  elem_ptr->getType()->getPointerElementType(), elem_ptr);
              auto mem_valWrapper = context->toMem(
                  val_field, context->createFalse(), "mem_" + currField);

              allJoinBindings[expr2.getRegisteredAs()] = mem_valWrapper;
              i++;
            }
          }

          /* RIGHT SIDE (RELATION S) */
          {
            int i = 0;

            for (const auto &expr2 :
                 getMaterializerRight().getWantedExpressions()) {
              const auto &currField = expr2.getRegisteredAttrName();

              auto elem_ptr = Builder->CreateGEP(
                  mem_payload_s->getType()->getNonOpaquePointerElementType(),
                  mem_payload_s,
                  {context->createInt32(0), context->createInt32(i)});
              auto val_field = Builder->CreateLoad(
                  elem_ptr->getType()->getPointerElementType(), elem_ptr);
              auto mem_valWrapper = context->toMem(
                  val_field, context->createFalse(), "mem_" + currField);

              LOG(INFO) << expr2.getRegisteredAs();
              allJoinBindings[expr2.getRegisteredAs()] = mem_valWrapper;
              i++;
            }
          }

          RecordAttribute oid{-1, relName, activeLoop, oid_type};
          ProteusValueMemory oid_mem;
          oid_mem.mem = mem_outCount;
          oid_mem.isNull = context->createFalse();
          allJoinBindings[oid] = oid_mem;

          /* Trigger Parent */
          OperatorState newState{*this, allJoinBindings};
          getParent()->consume(context, newState);

          Value *old_oid = Builder->CreateLoad(
              mem_outCount->getType()->getPointerElementType(), mem_outCount);
          Value *nxt_oid =
              Builder->CreateAdd(old_oid, ConstantInt::get(llvm_oid_type, 1));
          Builder->CreateStore(nxt_oid, mem_outCount);

          Builder->CreateBr(hitLoopInc);
        }

        /* 3. Inc: hit = (ht->next)[hit-1]) */
        Builder->SetInsertPoint(hitLoopInc);
        //(ht->next)
        Value *val_next = context->getStructElem(val_htPerClusterShiftedPtr, 1);
        val_idx = Builder->CreateSub(val_hit, val_one);
        //(ht->next)[hit-1])
        val_hit = context->getArrayElem(val_next, val_idx);
        Builder->CreateStore(val_hit, mem_hit);
        Builder->CreateBr(hitLoopCond);

        /* 4. End */
        Builder->SetInsertPoint(hitLoopEnd);
      }

      Builder->CreateBr(sLoopInc);

      Builder->SetInsertPoint(sLoopInc);
      val_j =
          Builder->CreateLoad(mem_j->getType()->getPointerElementType(), mem_j);
      val_j = Builder->CreateAdd(val_j, val_one);
      Builder->CreateStore(val_j, mem_j);
      Builder->CreateBr(sLoopCond);

      Builder->SetInsertPoint(sLoopEnd);
    }

    Builder->CreateBr(loopInc);

    /* If (either) cluster is empty */
    /*
     * r += R_count_per_cluster[i];
     * s += S_count_per_cluster[i];
     */
    Builder->SetInsertPoint(elseBlock);
    val_rCount = Builder->CreateLoad(
        mem_rCount->getType()->getPointerElementType(), mem_rCount);
    val_sCount = Builder->CreateLoad(
        mem_sCount->getType()->getPointerElementType(), mem_sCount);
    val_rCount = Builder->CreateAdd(val_rCount, val_r_i_count);
    val_sCount = Builder->CreateAdd(val_sCount, val_s_i_count);
    Builder->CreateStore(val_rCount, mem_rCount);
    Builder->CreateStore(val_sCount, mem_sCount);
    Builder->CreateBr(loopInc);
  }

  /* 3. Loop Inc. */
  Builder->SetInsertPoint(loopInc);
  val_clusterCount = Builder->CreateLoad(
      mem_clusterCount->getType()->getPointerElementType(), mem_clusterCount);
  val_clusterCount = Builder->CreateAdd(val_clusterCount, val_one);
  //#ifdef DEBUG
  //      vector<Value*> ArgsV0;
  //      ArgsV0.push_back(val_clusterCount);
  //      Builder->CreateCall(debugInt,ArgsV0);
  //#endif
  Builder->CreateStore(val_clusterCount, mem_clusterCount);

  Builder->CreateBr(loopCond);

  /* 4. Loop End */
  Builder->SetInsertPoint(loopEnd);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(loopCond);

  /* 4. Loop End */
  Builder->SetInsertPoint(context->getEndingBlock());
}
