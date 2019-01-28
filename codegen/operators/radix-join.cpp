/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "operators/radix-join.hpp"

RadixJoinBuild::RadixJoinBuild(expression_t keyExpr, RawOperator *child,
                               GpuRawContext *const context, string opLabel,
                               Materializer &mat, StructType *htEntryType,
                               size_t /* bytes */ size,
                               size_t /* bytes */ kvSize, bool is_agg)
    : UnaryRawOperator(child),
      keyExpr(std::move(keyExpr)),
      context(context),
      mat(mat),
      htEntryType(htEntryType),
      htLabel(opLabel),
      size(size),
      kvSize(kvSize),
      cached(false),
      is_agg(is_agg) {
  // TODO initializations

  pg = new OutputPlugin(context, mat, NULL);

  /* What (ht* + payload) points to: TBD */
  /* Result type specified during output plugin construction */
  payloadType = pg->getPayloadType();
}

RadixJoinBuild::~RadixJoinBuild() {
  LOG(INFO) << "Collapsing RadixJoinBuild operator";
  //  Can't do garbage collection here, need to do it from codegen
}

void RadixJoinBuild::produce() {
  initializeState();

  RawOperator *newChild = NULL;

  // TODO: enable cache
  // if (!this->child->isFiltering()) {
  //     //cout << "Checking left side for caches" << endl;
  //     newChild = findSideInCache(mat, true);
  // }

  if (newChild) {
#ifdef DEBUGCACHING
    cout << "NEW SCAN POSSIBLE!!" << endl;
#endif
    cached = true;
    this->setChild(newChild);
    newChild->setParent(this);
  }

  getChild()->produce();
}

void RadixJoinBuild::initializeState() {
  Function *F = context->getGlobalFunction();
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  Type *int64_type = Type::getInt64Ty(llvmContext);
  Type *int32_type = Type::getInt32Ty(llvmContext);
  Type *int8_type = Type::getInt8Ty(llvmContext);
  PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);
  PointerType *int32_ptr_type = PointerType::get(int32_type, 0);

  Value *zero = context->createInt64(0);

  /* Request memory to store relation             */
  rel.mem_relation_id = context->appendStateVar(
      PointerType::getUnqual(char_ptr_type),
      [=](llvm::Value *) {
        Value *mem_ptr = context->allocateStateVar(char_ptr_type);
        Function *getMemChunk = context->getFunction("getMemoryChunk");
        Value *mem_rel =
            Builder->CreateCall(getMemChunk, context->createSizeT(size));
        Builder->CreateStore(mem_rel, mem_ptr);
        return mem_ptr;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        Value *mem_rel = Builder->CreateLoad(s);
        Value *this_ptr = context->CastPtrToLlvmPtr(char_ptr_type, this);
        Function *reg = context->getFunction("registerRelationMem");
        Builder->CreateCall(reg, vector<Value *>{pip, mem_rel, this_ptr});
        context->deallocateStateVar(s);
      },
      "relation");
  rel.mem_tuplesNo_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        /* Partition and Cluster (the corresponding htEntries) */
        Value *mem_kv = context->getStateVar(ht.mem_kv_id);
        Value *clusterCount = radix_cluster_nopadding(s, mem_kv);
        Function *reg = context->getFunction("registerClusterCounts");
        Value *this_ptr = context->CastPtrToLlvmPtr(char_ptr_type, this);
        Builder->CreateCall(reg, vector<Value *>{pip, clusterCount, this_ptr});
        context->deallocateStateVar(s);
      },
      "tuples");
  rel.mem_cachedTuplesNo_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "tuplesCached");
  rel.mem_offset_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "size");
  rel.mem_size_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(context->createInt64(size), mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "offsetRel");

  // TODO: (rel.mem_relation)->setAlignment(8);

  PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

  /* Request memory to store HT entries of R */
  ht.mem_kv_id = context->appendStateVar(
      PointerType::getUnqual(htEntryPtrType),
      [=](llvm::Value *) {
        Value *mem_ptr = context->allocateStateVar(htEntryPtrType);
        Function *getMemChunk = context->getFunction("getMemoryChunk");
        Value *mem_rel =
            Builder->CreateCall(getMemChunk, context->createSizeT(kvSize));
        mem_rel = Builder->CreateBitCast(mem_rel, htEntryPtrType);
        StoreInst *store_ht = Builder->CreateStore(mem_rel, mem_ptr);
        store_ht->setAlignment(8);
        return mem_ptr;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        Value *mem_kv = Builder->CreateLoad(s);
        mem_kv = Builder->CreateBitCast(mem_kv, char_ptr_type);
        Value *this_ptr = context->CastPtrToLlvmPtr(char_ptr_type, this);
        Function *reg = context->getFunction("registerHTMemKV");
        Builder->CreateCall(reg, vector<Value *>{pip, mem_kv, this_ptr});
        context->deallocateStateVar(s);
      },
      "ht");
  ht.mem_tuplesNo_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "tuples");
  ht.mem_size_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(context->createInt64(kvSize), mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "size");
  ht.mem_offset_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "offsetRel");

  // TODO: (ht.mem_kv)->setAlignment(8);
}

void RadixJoinBuild::consume(RawContext *const context,
                             const OperatorState &childState) {
  GpuRawContext *ctx = dynamic_cast<GpuRawContext *>(context);
  assert(ctx && "Update caller to new API");
  consume(ctx, childState);
}

void RadixJoinBuild::consume(GpuRawContext *const context,
                             const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  RawCatalog &catalog = RawCatalog::getInstance();

  Type *int8_type = Type::getInt8Ty(llvmContext);
  Type *int64_type = Type::getInt64Ty(llvmContext);
  PointerType *void_ptr_type = PointerType::get(int8_type, 0);

  if (!cached) {
    const map<RecordAttribute, RawValueMemory> &bindings =
        childState.getBindings();

    pg->setBindings(&bindings);

    /* 3rd Method to calculate size */
    /* REMEMBER: PADDING DOES MATTER! */
    Value *val_payloadSize = ConstantInt::get((IntegerType *)int64_type,
                                              context->getSizeOf(payloadType));

    /* Registering payload type of HT in RAW CATALOG */
    /**
     * Must either fix (..for fully-blocking joins)
     * or remove entirely
     */
    // catalog.insertTableInfo(string(this->htLabel),payloadType);
    /*
     * Prepare payload.
     * What the 'output plugin + materializer' have decided is orthogonal
     * to this materialization policy
     * (i.e., whether we keep payload with the key, or just point to it)
     *
     * Instead of allocating space for payload and then copying it again,
     * do it ONCE on the pre-allocated buffer
     */

    Value *val_arena =
        Builder->CreateLoad(context->getStateVar(rel.mem_relation_id));
    Value *offsetInArena =
        Builder->CreateLoad(context->getStateVar(rel.mem_offset_id));
    Value *offsetPlusPayload =
        Builder->CreateAdd(offsetInArena, val_payloadSize);
    Value *arenaSize =
        Builder->CreateLoad(context->getStateVar(rel.mem_size_id));
    Value *val_tuplesNo =
        Builder->CreateLoad(context->getStateVar(rel.mem_tuplesNo_id));

    /* if(offsetInArena + payloadSize >= arenaSize) */
    BasicBlock *entryBlock = Builder->GetInsertBlock();
    BasicBlock *endBlockArenaFull =
        BasicBlock::Create(llvmContext, "IfArenaFullEnd", F);
    BasicBlock *ifArenaFull;
    context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull,
                           endBlockArenaFull);
    Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload, arenaSize);

    Builder->CreateCondBr(offsetCond, ifArenaFull, endBlockArenaFull);

    /* true => realloc() */
    Builder->SetInsertPoint(ifArenaFull);

    vector<Value *> ArgsRealloc;
    Function *reallocLLVM = context->getFunction("increaseMemoryChunk");
    AllocaInst *mem_arena_void =
        Builder->CreateAlloca(void_ptr_type, 0, "voidArenaPtr");
    Builder->CreateStore(val_arena, mem_arena_void);
    Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
    ArgsRealloc.push_back(val_arena_void);
    ArgsRealloc.push_back(arenaSize);
    Value *val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

    Builder->CreateStore(val_newArenaVoidPtr,
                         context->getStateVar(rel.mem_relation_id));
    Value *val_size =
        Builder->CreateLoad(context->getStateVar(rel.mem_size_id));
    val_size = Builder->CreateMul(val_size, context->createInt64(2));
    Builder->CreateStore(val_size, context->getStateVar(rel.mem_size_id));
    Builder->CreateBr(endBlockArenaFull);

    /* 'Normal' flow again */
    Builder->SetInsertPoint(endBlockArenaFull);

    /* Repeat load - realloc() might have occurred */
    val_arena = Builder->CreateLoad(context->getStateVar(rel.mem_relation_id));
    val_size = Builder->CreateLoad(context->getStateVar(rel.mem_size_id));

    /* XXX STORING PAYLOAD */
    /* 1. arena += (offset) */
    Value *ptr_arenaShifted =
        Builder->CreateInBoundsGEP(val_arena, offsetInArena);

    /* 2. Casting */
    PointerType *ptr_payloadType = PointerType::get(payloadType, 0);
    Value *cast_arenaShifted =
        Builder->CreateBitCast(ptr_arenaShifted, ptr_payloadType);

    /* 3. Storing payload, one field at a time */
    vector<Type *> *materializedTypes = pg->getMaterializedTypes();
    // Storing all activeTuples met so far
    int offsetInStruct =
        0;  // offset inside the struct (+current field manipulated)
    // RawValueMemory mem_activeTuple;
    // {
    // cout << "ORDER OF LEFT FIELDS MATERIALIZED"<<endl;
    // map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
    // for (memSearch = bindings.begin(); memSearch != bindings.end();
    //         memSearch++) {
    //     RecordAttribute currAttr = memSearch->first;
    //     //cout << currAttr.getRelationName() << "_" << currAttr.getAttrName()
    //     << endl; if (currAttr.getAttrName() == activeLoop) {
    //         mem_activeTuple = memSearch->second;
    //         Value* val_activeTuple = Builder->CreateLoad(
    //                 mem_activeTuple.mem);
    //         //OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON
    //         FOR EXTRA OFFSET vector<Value*> idxList = vector<Value*>();
    //         idxList.push_back(context->createInt32(0));
    //         idxList.push_back(context->createInt32(offsetInStruct));
    //         //Shift in struct ptr
    //         Value* structPtr = Builder->CreateGEP(cast_arenaShifted,
    //                 idxList);
    //         StoreInst *store_activeTuple = Builder->CreateStore(
    //                 val_activeTuple, structPtr);
    //         store_activeTuple->setAlignment(8);
    //         offsetInStruct++;
    //     }
    // }
    // }

    /* XXX Careful: Cache-aware */
    int offsetInWanted = 0;
    for (const auto &we : mat.getWantedExpressions()) {
      Value *valToMaterialize = NULL;
      // Check if cached - already done in output pg..
      bool isCached = false;
      CacheInfo info;
      CachingService &cache = CachingService::getInstance();
      /* expr does not participate in caching search, so don't need it
       * explicitly => mock */
      list<RecordAttribute *> mockAtts = list<RecordAttribute *>();
      mockAtts.push_back(new RecordAttribute(we.getRegisteredAs()));
      list<RecordAttribute> mockProjections;
      RecordType mockRec = RecordType(mockAtts);
      expression_t mockExpr = expression_t::make<expressions::InputArgument>(
          &mockRec, 0, mockProjections);
      auto e = mockExpr[we.getRegisteredAs()];
      info = cache.getCache(&e);
      if (info.structFieldNo != -1) {
        if (!cache.getCacheIsFull(&e)) {
        } else {
          isCached = true;
          cout << "[OUTPUT PG: ] *Cached* Expression found for "
               << e.getOriginalRelationName() << "."
               << e.getAttribute().getAttrName() << "!" << endl;
        }
      }

      if (isCached) {
        string activeRelation = e.getOriginalRelationName();
        string projName = e.getProjectionName();
        Plugin *plugin = catalog.getPlugin(activeRelation);
        valToMaterialize = (plugin->readCachedValue(info, bindings)).value;
      } else {
        map<RecordAttribute, RawValueMemory>::const_iterator memSearch =
            bindings.find(we.getRegisteredAs());
        if (memSearch != bindings.end()) {
          RawValueMemory currValMem = memSearch->second;
          /* FIX THE NECESSARY CONVERSIONS HERE */
          Value *currVal = Builder->CreateLoad(currValMem.mem);
          valToMaterialize =
              pg->convert(currVal->getType(),
                          materializedTypes->at(offsetInWanted), currVal);
        } else {
          ExpressionGeneratorVisitor exprGen{context, childState};
          RawValue currVal = we.accept(exprGen);
          /* FIX THE NECESSARY CONVERSIONS HERE */
          valToMaterialize = currVal.value;
        }
      }
      vector<Value *> idxList = vector<Value *>();
      idxList.push_back(context->createInt32(0));
      idxList.push_back(context->createInt32(offsetInStruct));

      // Shift in struct ptr
      Value *structPtr = Builder->CreateGEP(cast_arenaShifted, idxList);

      Builder->CreateStore(valToMaterialize, structPtr);
      offsetInStruct++;
      offsetInWanted++;
    }
    /* Backing up to incorporate caching-aware code */
    //          int offsetInWanted = 0;
    //          const vector<RecordAttribute*>& wantedFields =
    //                  matLeft.getWantedFields();
    //          for (vector<RecordAttribute*>::const_iterator it =
    //                  wantedFields.begin(); it != wantedFields.end(); ++it) {
    //              //cout << (*it)->getRelationName() << "_" <<
    //              (*it)->getAttrName() << endl; map<RecordAttribute,
    //              RawValueMemory>::const_iterator memSearch =
    //                      bindings.find(*(*it));
    //              RawValueMemory currValMem = memSearch->second;
    //              /* FIX THE NECESSARY CONVERSIONS HERE */
    //              Value* currVal = Builder->CreateLoad(currValMem.mem);
    //              Value* valToMaterialize = pg->convert(currVal->getType(),
    //                      materializedTypes->at(offsetInWanted), currVal);
    //
    //              vector<Value*> idxList = vector<Value*>();
    //              idxList.push_back(context->createInt32(0));
    //              idxList.push_back(context->createInt32(offsetInStruct));
    //
    //              //Shift in struct ptr
    //              Value* structPtr = Builder->CreateGEP(cast_arenaShifted,
    //                      idxList);
    //
    //              Builder->CreateStore(valToMaterialize, structPtr);
    //              offsetInStruct++;
    //              offsetInWanted++;
    //          }

    /* CONSTRUCT HTENTRY PAIR         */
    /* payloadPtr: relative offset from relBuffer beginning */
    /* (int32 key, int64 payloadPtr)  */
    /* Prepare key */
    ExpressionGeneratorVisitor exprGenerator{context, childState};

    RawValue key = keyExpr.accept(exprGenerator);

    PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

    BasicBlock *endBlockHTFull =
        BasicBlock::Create(llvmContext, "IfHTFullEnd", F);
    BasicBlock *ifHTFull;
    context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull, endBlockHTFull);

    LoadInst *val_ht = Builder->CreateLoad(context->getStateVar(ht.mem_kv_id));
    val_ht->setAlignment(8);

    Value *offsetInHT =
        Builder->CreateLoad(context->getStateVar(ht.mem_offset_id));
    //      Value *offsetPlusKey = Builder->CreateAdd(offsetInHT,val_keySize64);
    //      int payloadPtrSize = sizeof(size_t);
    //      Value *val_payloadPtrSize = context->createInt64(payloadPtrSize);
    //      Value *offsetPlusPayloadPtr =
    //      Builder->CreateAdd(offsetPlusKey,val_payloadPtrSize);
    Value *kvSize = ConstantInt::get((IntegerType *)int64_type,
                                     context->getSizeOf(htEntryType));
    Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);

    Value *htSize = Builder->CreateLoad(context->getStateVar(ht.mem_size_id));
    offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair, htSize);

    Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

    /* true => realloc() */
    Builder->SetInsertPoint(ifHTFull);

    /* Casting htEntry* to void* requires a cast */
    Value *cast_htEntries = Builder->CreateBitCast(val_ht, void_ptr_type);
    ArgsRealloc.clear();
    ArgsRealloc.push_back(cast_htEntries);
    ArgsRealloc.push_back(htSize);  // realloc takes old size (and doubles mem)
    Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

    Value *val_newHTPtr =
        Builder->CreateBitCast(val_newVoidHTPtr, htEntryPtrType);
    Builder->CreateStore(val_newHTPtr, context->getStateVar(ht.mem_kv_id));
    val_size = Builder->CreateLoad(context->getStateVar(ht.mem_size_id));
    val_size = Builder->CreateMul(val_size, context->createInt64(2));
    Builder->CreateStore(val_size, context->getStateVar(ht.mem_size_id));
    Builder->CreateBr(endBlockHTFull);

    /* Insert ht entry in HT */
    Builder->SetInsertPoint(endBlockHTFull);

    /* Repeat load - realloc() might have occurred */
    val_ht = Builder->CreateLoad(context->getStateVar(ht.mem_kv_id));
    val_ht->setAlignment(8);

    val_size = Builder->CreateLoad(context->getStateVar(ht.mem_size_id));

    /* 1. kv += offset */
    /* Note that we already have a htEntry ptr here */
    Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht, val_tuplesNo);

    /* 2a. kv_cast->keyPtr = &key */
    offsetInStruct = 0;
    // Shift in htEntry (struct) ptr
    vector<Value *> idxList = vector<Value *>();
    idxList.push_back(context->createInt32(0));
    idxList.push_back(context->createInt32(offsetInStruct));

    Value *structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
    StoreInst *store_key = Builder->CreateStore(key.value, structPtr);
    store_key->setAlignment(4);

    /* 2b. kv_cast->payloadPtr = &payload */
    offsetInStruct = 1;
    idxList.clear();
    idxList.push_back(context->createInt32(0));
    idxList.push_back(context->createInt32(offsetInStruct));
    structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);

    StoreInst *store_payloadPtr =
        Builder->CreateStore(offsetInArena, structPtr);
    store_payloadPtr->setAlignment(8);

    /* 4. Increment counts - both Rel and HT */
    Builder->CreateStore(offsetPlusPayload,
                         context->getStateVar(rel.mem_offset_id));
    Builder->CreateStore(offsetPlusKVPair,
                         context->getStateVar(ht.mem_offset_id));
    val_tuplesNo = Builder->CreateAdd(val_tuplesNo, context->createInt64(1));
    Builder->CreateStore(val_tuplesNo,
                         context->getStateVar(rel.mem_tuplesNo_id));
    Builder->CreateStore(val_tuplesNo,
                         context->getStateVar(ht.mem_tuplesNo_id));
  } else {
    assert(false && "Caching is disabled");  // FIMXE
  }
}

RadixJoin::RadixJoin(const expressions::BinaryExpression &predicate,
                     RawOperator *leftChild, RawOperator *rightChild,
                     RawContext *const context, const char *opLabel,
                     Materializer &matLeft, Materializer &matRight)
    : BinaryRawOperator(leftChild, rightChild),
      context((GpuRawContext *const)context),
      htLabel(opLabel) {
  assert(dynamic_cast<GpuRawContext *const>(context) &&
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

int32_t *RadixJoinBuild::getClusterCounts(RawPipeline *pip) {
  assert(pip->getGroup() < 128);
  int32_t *cnts = clusterCounts[pip->getGroup()];
  assert(cnts);
  return cnts;
}

void RadixJoinBuild::registerClusterCounts(RawPipeline *pip, int32_t *cnts) {
  assert(pip->getGroup() < 128);
  assert(cnts);
  clusterCounts[pip->getGroup()] = cnts;
}

void *RadixJoinBuild::getHTMemKV(RawPipeline *pip) {
  assert(pip->getGroup() < 128);
  void *mem_kv = ht_mem_kv[pip->getGroup()];
  assert(mem_kv);
  return mem_kv;
}

void RadixJoinBuild::registerHTMemKV(RawPipeline *pip, void *mem_kv) {
  assert(pip->getGroup() < 128);
  assert(mem_kv);
  ht_mem_kv[pip->getGroup()] = mem_kv;
}

void *RadixJoinBuild::getRelationMem(RawPipeline *pip) {
  assert(pip->getGroup() < 128);
  void *rel_mem = relation_mem[pip->getGroup()];
  assert(rel_mem);
  return rel_mem;
}

void RadixJoinBuild::registerRelationMem(RawPipeline *pip, void *rel_mem) {
  assert(pip->getGroup() < 128);
  assert(rel_mem);
  relation_mem[pip->getGroup()] = rel_mem;
}

extern "C" {

int32_t *getClusterCounts(RawPipeline *pip, RadixJoinBuild *b) {
  return b->getClusterCounts(pip);
}

void registerClusterCounts(RawPipeline *pip, int32_t *cnts, RadixJoinBuild *b) {
  return b->registerClusterCounts(pip, cnts);
}

void *getHTMemKV(RawPipeline *pip, RadixJoinBuild *b) {
  return b->getHTMemKV(pip);
}

void registerHTMemKV(RawPipeline *pip, void *mem_kv, RadixJoinBuild *b) {
  return b->registerHTMemKV(pip, mem_kv);
}

void *getRelationMem(RawPipeline *pip, RadixJoinBuild *b) {
  return b->getRelationMem(pip);
}

void registerRelationMem(RawPipeline *pip, void *rel_mem, RadixJoinBuild *b) {
  return b->registerRelationMem(pip, rel_mem);
}
}

void RadixJoin::produce() {
  runRadix();

  context->popPipeline();

  auto flush_pip = context->removeLatestPipeline();
  flush_fun = flush_pip->getKernel();

  context->pushPipeline();

  RawOperator *leftChild = getLeftChild();
  leftChild->setParent(buildR);
  buildR->setParent(this);
  setLeftChild(buildR);

  buildR->produce();

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

  RawOperator *rightChild = getRightChild();
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

  buildS->produce();
}

void RadixJoin::consume(RawContext *const context,
                        const OperatorState &childState) {
  assert(false &&
         "Function should not be called! Are RadixJoinBuilders "
         "correctly set as children of this operator?");
}

/**
 * @param rel the materialized input relation
 * @param ht  the htEntries corresp. to the relation
 *
 * @return item count per resulting cluster
 */
Value *RadixJoinBuild::radix_cluster_nopadding(Value *mem_tuplesNo,
                                               Value *mem_kv_id) const {
  LLVMContext &llvmContext = context->getLLVMContext();
  RawCatalog &catalog = RawCatalog::getInstance();
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  Function *partitionHT =
      context->getFunction(is_agg ? "partitionAggHT" : "partitionHT");
  vector<Value *> ArgsPartition;
  Value *val_tuplesNo = Builder->CreateLoad(mem_tuplesNo);
  Value *val_ht = Builder->CreateLoad(mem_kv_id);
  ArgsPartition.push_back(val_tuplesNo);
  ArgsPartition.push_back(val_ht);

  return Builder->CreateCall(partitionHT, ArgsPartition);
}

void RadixJoin::runRadix() const {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  RawCatalog &catalog = RawCatalog::getInstance();

  Type *int64_type = Type::getInt64Ty(llvmContext);
  Type *int32_type = Type::getInt32Ty(llvmContext);
  PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);

  size_t clusterCountR_id = context->appendStateVar(
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
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "clusterCountR");

  size_t clusterCountS_id = context->appendStateVar(
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
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "clusterCountS");  // FIXME: read-only, we do not even have to maintain it
                         // as state variable, but where do we get pip from ?

  size_t htR_mem_kv_id = context->appendStateVar(
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
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "htR_mem_kv");  // FIXME: read-only, we do not even have to maintain it as
                      // state variable

  size_t htS_mem_kv_id = context->appendStateVar(
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
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "htS_mem_kv");  // FIXME: read-only, we do not even have to maintain it as
                      // state variable

  size_t relR_mem_relation_id = context->appendStateVar(
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
        s = Builder->CreateBitCast(s, char_ptr_type);
        Builder->CreateCall(f, s);
      },
      "relR_mem_relation");  // FIXME: read-only, we do not even have to
                             // maintain it as state variable

  size_t relS_mem_relation_id = context->appendStateVar(
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

  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  std::string relName = getMaterializerRight()
                            .getWantedExpressions()
                            .back()
                            .getRegisteredRelName();
  Plugin *pg = RawCatalog::getInstance().getPlugin(relName);
  ExpressionType *oid_type = pg->getOIDType();
  IntegerType *llvm_oid_type =
      (IntegerType *)oid_type->getLLVMType(llvmContext);
  AllocaInst *mem_outCount =
      Builder->CreateAlloca(llvm_oid_type, 0, "join_oid");
  AllocaInst *mem_rCount = Builder->CreateAlloca(int32_type, 0, "rCount");
  AllocaInst *mem_sCount = Builder->CreateAlloca(int32_type, 0, "sCount");
  AllocaInst *mem_clusterCount =
      Builder->CreateAlloca(int32_type, 0, "clusterCount");
  Builder->CreateStore(ConstantInt::get(llvm_oid_type, 0), mem_outCount);
  Builder->CreateStore(val_zero, mem_rCount);
  Builder->CreateStore(val_zero, mem_sCount);
  Builder->CreateStore(val_zero, mem_clusterCount);

  uint32_t clusterNo = (1 << NUM_RADIX_BITS);
  Value *val_clusterNo = context->createInt32(clusterNo);

  /* Request memory for HT(s) construction        */
  /* Note: Does not allocate mem. for buckets too */
  size_t htSize = (1 << NUM_RADIX_BITS) * sizeof(HT);

  Builder->CreateAlloca(htClusterType, 0, "HTimpl");
  PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
  Function *f_getMemoryChunk = context->getFunction("getMemoryChunk");
  Value *HT_mem = Builder->CreateCall(
      f_getMemoryChunk, vector<Value *>{context->createSizeT(htSize)});
  Value *val_htPerCluster = Builder->CreateBitCast(HT_mem, htClusterPtrType);

  AllocaInst *mem_probesNo =
      Builder->CreateAlloca(int32_type, 0, "mem_counter");
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
  Value *val_clusterCount = Builder->CreateLoad(mem_clusterCount);
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

    Value *val_rCount = Builder->CreateLoad(mem_rCount);
    Value *val_sCount = Builder->CreateLoad(mem_sCount);
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
    Value *htRshiftedPtr = Builder->CreateInBoundsGEP(val_htR, val_rCount);

    /* tmpS.tuples = relS->tuples + s; */
    Value *val_htS = context->getStateVar(htS_mem_kv_id);
    Value *htSshiftedPtr = Builder->CreateInBoundsGEP(val_htS, val_sCount);

    /* bucket_chaining_join_prepare(&tmpR, &(HT_per_cluster[i])); */
    Function *bucketChainingPrepare =
        context->getFunction("bucketChainingPrepare");

    PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
    Value *val_htPerClusterShiftedPtr =
        Builder->CreateInBoundsGEP(val_htPerCluster, val_clusterCount);

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
      AllocaInst *mem_j = Builder->CreateAlloca(int32_type, 0, "j_cnt");
      Builder->CreateStore(val_zero, mem_j);
      Builder->CreateBr(sLoopCond);

      /* Loop Condition */
      Builder->SetInsertPoint(sLoopCond);
      Value *val_j = Builder->CreateLoad(mem_j);

      val_cond = Builder->CreateICmpSLT(val_j, val_s_i_count);

      Builder->CreateCondBr(val_cond, sLoopBody, sLoopEnd);

      Builder->SetInsertPoint(sLoopBody);

#ifdef DEBUGRADIX
//          Value *val_probesNo = Builder->CreateLoad(mem_probesNo);
//          val_probesNo = Builder->CreateAdd(val_probesNo, val_one);
//          Builder->CreateStore(val_probesNo,mem_probesNo);
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
      Value *htSshiftedPtr_j = Builder->CreateInBoundsGEP(htSshiftedPtr, val_j);
      //          Value *tuple_s_j = Builder->CreateLoad(htSshiftedPtr_j);
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
        AllocaInst *mem_hit = Builder->CreateAlloca(int32_type, 0, "hit");
        //(ht->bucket)
        Value *val_bucket =
            context->getStructElem(val_htPerClusterShiftedPtr, 0);
        //(ht->bucket)[idx]
        Value *val_bucket_idx = context->getArrayElem(val_bucket, val_idx);

        Builder->CreateStore(val_bucket_idx, mem_hit);
        Builder->CreateBr(hitLoopCond);
        /* 1. Loop Condition */
        Builder->SetInsertPoint(hitLoopCond);
        Value *val_hit = Builder->CreateLoad(mem_hit);
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
          Value *htRshiftedPtr_hit =
              Builder->CreateInBoundsGEP(htRshiftedPtr, val_idx_dec);
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
          map<RecordAttribute, RawValueMemory> *allJoinBindings =
              new map<RecordAttribute, RawValueMemory>();

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
          Value *val_ptr_payloadR =
              Builder->CreateInBoundsGEP(val_relR, val_payload_r_offset);
          Value *val_ptr_payloadS =
              Builder->CreateInBoundsGEP(val_relS, val_payload_s_offset);
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
          Value *val_payload_r = Builder->CreateLoad(mem_payload_r);
          Value *mem_payload_s =
              Builder->CreateBitCast(val_ptr_payloadS, sPayloadPtrType);
          Value *val_payload_s = Builder->CreateLoad(mem_payload_s);

          /* LEFT SIDE (RELATION R)*/
          // Retrieving activeTuple(s) from HT
          {
            //                         AllocaInst *mem_activeTuple = NULL;
            int i = 0;
            // //                      const set<RecordAttribute>&
            // tuplesIdentifiers =
            // //                              matLeft.getTupleIdentifiers();
            //                         for (RecordAttribute *attr:
            //                         getMaterializerLeft().getWantedOIDs()) {
            //                             mem_activeTuple =
            //                             context->CreateEntryBlockAlloca(F,
            //                                     "mem_activeTuple",
            //                                     rPayloadType->getElementType(i));
            //                             vector<Value*> idxList =
            //                             vector<Value*>();
            //                             idxList.push_back(context->createInt32(0));
            //                             idxList.push_back(context->createInt32(i));

            //                             Value *elem_ptr =
            //                             Builder->CreateGEP(mem_payload_r,
            //                                     idxList);
            //                             Value *val_activeTuple =
            //                             Builder->CreateLoad(
            //                                     elem_ptr);
            //                             StoreInst *store_activeTuple =
            //                             Builder->CreateStore(
            //                                     val_activeTuple,
            //                                     mem_activeTuple);
            //                             store_activeTuple->setAlignment(8);

            //                             RawValueMemory mem_valWrapper;
            //                             mem_valWrapper.mem = mem_activeTuple;
            //                             mem_valWrapper.isNull =
            //                             context->createFalse();
            //                             (*allJoinBindings)[*attr] =
            //                             mem_valWrapper; i++;
            //                         }

            for (const auto &expr2 :
                 getMaterializerLeft().getWantedExpressions()) {
              string currField = expr2.getRegisteredAttrName();
              AllocaInst *mem_field = context->CreateEntryBlockAlloca(
                  F, "mem_" + currField, rPayloadType->getElementType(i));
              vector<Value *> idxList = vector<Value *>();
              idxList.push_back(context->createInt32(0));
              idxList.push_back(context->createInt32(i));

              Value *elem_ptr = Builder->CreateGEP(mem_payload_r, idxList);
              Value *val_field = Builder->CreateLoad(elem_ptr);
              Builder->CreateStore(val_field, mem_field);

              RawValueMemory mem_valWrapper;
              mem_valWrapper.mem = mem_field;
              mem_valWrapper.isNull = context->createFalse();

#ifdef DEBUGRADIX
//                          vector<Value*> ArgsV;
//                          ArgsV.push_back(context->createInt32(1111));
//                          Builder->CreateCall(debugInt, ArgsV);
//                          ArgsV.clear();
//                          ArgsV.push_back(Builder->CreateLoad(mem_field));
//                          Builder->CreateCall(debugInt, ArgsV);
#endif

              (*allJoinBindings)[expr2.getRegisteredAs()] = mem_valWrapper;
              i++;
            }
          }
#ifdef DEBUGRADIX
          {
            /* Printing key(s) */
            vector<Value *> ArgsV;
            ArgsV.push_back(Builder->getInt32(500006));
            Builder->CreateCall(debugInt, ArgsV);
          }
#endif
          /* RIGHT SIDE (RELATION S) */
          {
            // AllocaInst *mem_activeTuple = NULL;
            int i = 0;
            //                         for (RecordAttribute *attr:
            //                         getMaterializerRight().getWantedOIDs()) {
            //                             mem_activeTuple =
            //                             context->CreateEntryBlockAlloca(F,
            //                                     "mem_activeTuple",
            //                                     sPayloadType->getElementType(i));
            //                             vector<Value*> idxList =
            //                             vector<Value*>();
            //                             idxList.push_back(context->createInt32(0));
            //                             idxList.push_back(context->createInt32(i));

            //                             Value *elem_ptr =
            //                             Builder->CreateGEP(mem_payload_s,
            //                                     idxList);
            //                             Value *val_activeTuple =
            //                             Builder->CreateLoad(
            //                                     elem_ptr);
            //                             StoreInst *store_activeTuple =
            //                             Builder->CreateStore(
            //                                     val_activeTuple,
            //                                     mem_activeTuple);
            //                             store_activeTuple->setAlignment(8);

            //                             RawValueMemory mem_valWrapper;
            //                             mem_valWrapper.mem = mem_activeTuple;
            //                             mem_valWrapper.isNull =
            //                             context->createFalse();
            //                             (*allJoinBindings)[*attr] =
            //                             mem_valWrapper; i++;
            // #ifdef DEBUGRADIX
            //                             {
            //                                 /* Printing key(s) */
            //                                 vector<Value*> ArgsV;
            //                                 ArgsV.push_back(val_activeTuple);
            //                                 Builder->CreateCall(debugInt64,
            //                                 ArgsV);
            //                             }
            // #endif
            //                         }

            for (const auto &expr2 :
                 getMaterializerRight().getWantedExpressions()) {
              string currField = expr2.getRegisteredAttrName();
              AllocaInst *mem_field = context->CreateEntryBlockAlloca(
                  F, "mem_" + currField, sPayloadType->getElementType(i));
              vector<Value *> idxList = vector<Value *>();
              idxList.push_back(context->createInt32(0));
              idxList.push_back(context->createInt32(i));

              Value *elem_ptr = Builder->CreateGEP(mem_payload_s, idxList);
              Value *val_field = Builder->CreateLoad(elem_ptr);
              Builder->CreateStore(val_field, mem_field);

              RawValueMemory mem_valWrapper;
              mem_valWrapper.mem = mem_field;
              mem_valWrapper.isNull = context->createFalse();
#ifdef DEBUGRADIX
//                          vector<Value*> ArgsV;
//                          ArgsV.push_back(context->createInt32(1112));
//                          Builder->CreateCall(debugInt, ArgsV);
//                          ArgsV.clear();
//                          ArgsV.push_back(Builder->CreateLoad(mem_field));
//                          Builder->CreateCall(debugInt, ArgsV);
#endif
              (*allJoinBindings)[expr2.getRegisteredAs()] = mem_valWrapper;
              i++;
            }
          }

          RecordAttribute oid{-1, relName, activeLoop, oid_type};
          RawValueMemory oid_mem;
          oid_mem.mem = mem_outCount;
          oid_mem.isNull = context->createFalse();
          (*allJoinBindings)[oid] = oid_mem;
#ifdef DEBUGRADIX
          {
            /* Printing key(s) */
            vector<Value *> ArgsV;
            ArgsV.push_back(Builder->getInt32(500008));
            Builder->CreateCall(debugInt, ArgsV);
          }
#endif
          /* Trigger Parent */
          OperatorState newState{*this, *allJoinBindings};
          getParent()->consume(context, newState);

          Value *old_oid = Builder->CreateLoad(mem_outCount);
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
      val_j = Builder->CreateLoad(mem_j);
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
    val_rCount = Builder->CreateLoad(mem_rCount);
    val_sCount = Builder->CreateLoad(mem_sCount);
    val_rCount = Builder->CreateAdd(val_rCount, val_r_i_count);
    val_sCount = Builder->CreateAdd(val_sCount, val_s_i_count);
    Builder->CreateStore(val_rCount, mem_rCount);
    Builder->CreateStore(val_sCount, mem_sCount);
    Builder->CreateBr(loopInc);
  }

  /* 3. Loop Inc. */
  Builder->SetInsertPoint(loopInc);
  val_clusterCount = Builder->CreateLoad(mem_clusterCount);
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