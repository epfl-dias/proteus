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

#include "radix-join-build.hpp"

#include <utility>

using namespace llvm;

RadixJoinBuild::RadixJoinBuild(expression_t keyExpr, Operator *child,
                               ParallelContext *context, string opLabel,
                               Materializer &mat, StructType *htEntryType,
                               size_t /* bytes */ size,
                               size_t /* bytes */ kvSize, bool is_agg)
    : UnaryOperator(child),
      keyExpr(std::move(keyExpr)),
      mat(mat),
      htEntryType(htEntryType),
      htLabel(std::move(opLabel)),
      size(size),
      kvSize(kvSize),
      cached(false),
      is_agg(is_agg) {
  // TODO initializations

  pg = new OutputPlugin(context, mat, nullptr);

  /* What (ht* + payload) points to: TBD */
  /* Result type specified during output plugin construction */
  payloadType = pg->getPayloadType();
}

RadixJoinBuild::~RadixJoinBuild() {
  LOG(INFO) << "Collapsing RadixJoinBuild operator";
  //  Can't do garbage collection here, need to do it from codegen
}

void RadixJoinBuild::produce_(ParallelContext *context) {
  initializeState(context);

  Operator *newChild = nullptr;

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

  getChild()->produce(context);
}

void RadixJoinBuild::initializeState(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

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
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem_ptr = context->allocateStateVar(char_ptr_type);
        Function *getMemChunk = context->getFunction("getMemoryChunk");
        Value *mem_rel =
            Builder->CreateCall(getMemChunk, context->createSizeT(size));
        Builder->CreateStore(mem_rel, mem_ptr);
        return mem_ptr;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        IRBuilder<> *Builder = context->getBuilder();
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
        IRBuilder<> *Builder = context->getBuilder();
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        /* Partition and Cluster (the corresponding htEntries) */
        Value *mem_kv = context->getStateVar(ht.mem_kv_id);
        Value *clusterCount = radix_cluster_nopadding(context, s, mem_kv);
        Function *reg = context->getFunction("registerClusterCounts");
        Value *this_ptr = context->CastPtrToLlvmPtr(char_ptr_type, this);
        IRBuilder<> *Builder = context->getBuilder();
        Builder->CreateCall(reg, vector<Value *>{pip, clusterCount, this_ptr});
        context->deallocateStateVar(s);
      },
      "tuples");
  rel.mem_cachedTuplesNo_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        IRBuilder<> *Builder = context->getBuilder();
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "tuplesCached");
  rel.mem_offset_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        IRBuilder<> *Builder = context->getBuilder();
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "size");
  rel.mem_size_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        Value *mem = context->allocateStateVar(int64_type);
        IRBuilder<> *Builder = context->getBuilder();
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
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem_rel =
            Builder->CreateCall(getMemChunk, context->createSizeT(kvSize));
        mem_rel = Builder->CreateBitCast(mem_rel, htEntryPtrType);
        StoreInst *store_ht = Builder->CreateStore(mem_rel, mem_ptr);
        store_ht->setAlignment(llvm::Align(8));
        return mem_ptr;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        IRBuilder<> *Builder = context->getBuilder();
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
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "tuples");
  ht.mem_size_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(context->createInt64(kvSize), mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "size");
  ht.mem_offset_id = context->appendStateVar(
      PointerType::getUnqual(int64_type),
      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem = context->allocateStateVar(int64_type);
        Builder->CreateStore(zero, mem);
        return mem;
      },
      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); },
      "offsetRel");

  // TODO: (ht.mem_kv)->setAlignment(8);
}

void RadixJoinBuild::consume(Context *const context,
                             const OperatorState &childState) {
  auto *ctx = dynamic_cast<ParallelContext *>(context);
  assert(ctx && "Update caller to new API");
  consume(ctx, childState);
}

void RadixJoinBuild::consume(ParallelContext *const context,
                             const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  Catalog &catalog = Catalog::getInstance();

  Type *int8_type = Type::getInt8Ty(llvmContext);
  Type *int64_type = Type::getInt64Ty(llvmContext);
  PointerType *void_ptr_type = PointerType::get(int8_type, 0);

  if (!cached) {
    const map<RecordAttribute, ProteusValueMemory> &bindings =
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
        Builder->CreateAlloca(void_ptr_type, nullptr, "voidArenaPtr");
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
    // ProteusValueMemory mem_activeTuple;
    // {
    // cout << "ORDER OF LEFT FIELDS MATERIALIZED"<<endl;
    // map<RecordAttribute, ProteusValueMemory>::const_iterator memSearch;
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
      Value *valToMaterialize = nullptr;
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
        valToMaterialize =
            (plugin->readCachedValue(info, childState, context)).value;
      } else {
        map<RecordAttribute, ProteusValueMemory>::const_iterator memSearch =
            bindings.find(we.getRegisteredAs());
        if (memSearch != bindings.end()) {
          ProteusValueMemory currValMem = memSearch->second;
          /* FIX THE NECESSARY CONVERSIONS HERE */
          Value *currVal = Builder->CreateLoad(currValMem.mem);
          valToMaterialize =
              pg->convert(currVal->getType(),
                          materializedTypes->at(offsetInWanted), currVal);
        } else {
          ExpressionGeneratorVisitor exprGen{context, childState};
          ProteusValue currVal = we.accept(exprGen);
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
    //              ProteusValueMemory>::const_iterator memSearch =
    //                      bindings.find(*(*it));
    //              ProteusValueMemory currValMem = memSearch->second;
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

    ProteusValue key = keyExpr.accept(exprGenerator);

    PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

    BasicBlock *endBlockHTFull =
        BasicBlock::Create(llvmContext, "IfHTFullEnd", F);
    BasicBlock *ifHTFull;
    context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull, endBlockHTFull);

    LoadInst *val_ht = Builder->CreateLoad(context->getStateVar(ht.mem_kv_id));
    val_ht->setAlignment(llvm::Align(8));

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
    val_ht->setAlignment(llvm::Align(8));

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
    store_key->setAlignment(llvm::Align(4));

    /* 2b. kv_cast->payloadPtr = &payload */
    offsetInStruct = 1;
    idxList.clear();
    idxList.push_back(context->createInt32(0));
    idxList.push_back(context->createInt32(offsetInStruct));
    structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);

    StoreInst *store_payloadPtr =
        Builder->CreateStore(offsetInArena, structPtr);
    store_payloadPtr->setAlignment(llvm::Align(8));

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

/**
 * @param mem_tuplesNo  the htEntries corresp. to the relation
 * @param mem_kv_id     the materialized input relation
 *
 * @return item count per resulting cluster
 */
Value *RadixJoinBuild::radix_cluster_nopadding(ParallelContext *context,
                                               Value *mem_tuplesNo,
                                               Value *mem_kv_id) const {
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
