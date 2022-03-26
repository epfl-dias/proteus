/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "nest.hpp"

using namespace llvm;

/**
 * Identical constructor logic with the one of Reduce
 */
Nest::Nest(Monoid acc, expressions::Expression *outputExpr,
           expressions::Expression *pred,
           const list<expressions::InputArgument> &f_grouping,
           const list<expressions::InputArgument> &g_nullToZero,
           Operator *child, char *opLabel, Materializer &mat)
    : UnaryOperator(child),
      acc(acc),
      outputExpr(outputExpr),
      g_nullToZero(g_nullToZero),
      pred(pred),
      mat(mat),
      htName(opLabel) {
  // Prepare 'f' -> Turn it into expression (record construction)
  list<expressions::AttributeConstruction> atts;
  string attrPlaceholder("attr_");
  for (const auto &it : f_grouping) {
    const auto *type = it.getExpressionType();
    int argNo = it.getArgNo();

    atts.emplace_back(attrPlaceholder, expressions::InputArgument{type, argNo});
  }
  this->f_grouping = new expressions::RecordConstruction(atts);

  // Prepare extra output binding
  this->aggregateName += string(htName);
  this->aggregateName += "_aggr";
}
/**
 * NOTE: This is a fully pipeline-breaking operator!
 * Once the HT has been build, execution must
 * stop entirely before finishing the operator's work
 * => generation takes place in two steps
 */
void Nest::produce_(ParallelContext *context) {
  getChild()->produce(context);

  generateProbe(context);
}

void Nest::consume(ParallelContext *context, const OperatorState &childState) {
  generateInsert(context, childState);
}

/**
 * This function will be launched twice, since both outer unnest + join
 * cause two code paths to be generated.
 */
void Nest::generateInsert(Context *context, const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Catalog &catalog = Catalog::getInstance();
  vector<Value *> ArgsV;
  Function *debugInt = context->getFunction("printi");

#ifdef DEBUG
//    ArgsV.clear();
//    ArgsV.push_back(context->createInt32(-6));
//    Builder->CreateCall(debugInt, ArgsV);
//    ArgsV.clear();
#endif

  /**
   * STEP: Perform aggregation. Create buckets and fill them up IF g satisfied
   * Technically, this check should be made after the buckets have been filled.
   * Is this 'rewrite' correct?
   */

  /**
   * TODO do null check based on g!!!
   * Essentially a conjunctive predicate
   */

  // 1. Compute HASH key based on grouping expression
  /**
   * XXX sth gets screwed up in f_grouping for the 6th entry
   */
  ExpressionHasherVisitor aggrExprGenerator =
      ExpressionHasherVisitor(context, childState);
  ProteusValue groupKey = f_grouping->accept(aggrExprGenerator);

  // 2. Create 'payload' --> What is to be inserted in the bucket
  LOG(INFO) << "[NEST: ] Creating payload";
  const map<RecordAttribute, ProteusValueMemory> &bindings =
      childState.getBindings();
  OutputPlugin pg(context, mat, &bindings);

  // Result type specified during output plugin construction
  llvm::StructType *payloadType = pg.getPayloadType();
  // Creating space for the payload. XXX Might have to set alignment explicitly
  AllocaInst *mem_payload = context->CreateEntryBlockAlloca(
      TheFunction, string("valueInHT"), payloadType);

  // Registering payload type of HT in RAW CATALOG
  catalog.insertTableInfo(string(this->htName), payloadType);

  // Creating and Populating Payload Struct
  int offsetInStruct =
      0;  // offset inside the struct (+current field manipulated)
  vector<Type *> *materializedTypes = pg.getMaterializedTypes();

  // Storing values in struct to be materialized in HT. Two steps
  // 2a. Materializing all 'activeTuples' (i.e. positional indices) met so far
  {
    for (const auto &memSearch : bindings) {
      RecordAttribute currAttr = memSearch.first;
      if (currAttr.getAttrName() == activeLoop) {
        auto mem_activeTuple = memSearch.second;
        Value *val_activeTuple = Builder->CreateLoad(
            mem_activeTuple.mem->getType()->getPointerElementType(),
            mem_activeTuple.mem);
        // OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR
        // EXTRA OFFSET
        // Shift in struct ptr
        Value *structPtr = Builder->CreateGEP(
            nullptr, mem_payload,
            {context->createInt32(0), context->createInt32(offsetInStruct++)});
        Builder->CreateStore(val_activeTuple, structPtr);
      }
    }
  }

  // 2b. Materializing all explicitly requested fields
  int offsetInWanted = 0;
  const vector<RecordAttribute *> &wantedFields = mat.getWantedFields();
  for (const auto &wantedField : wantedFields) {
    auto memSearch = bindings.find(*wantedField);

    Value *llvmCurrVal = nullptr;
    if (memSearch != bindings.end()) {
      ProteusValueMemory currValMem = memSearch->second;
      llvmCurrVal = Builder->CreateLoad(
          currValMem.mem->getType()->getPointerElementType(), currValMem.mem);
    } else {
      //            /* Not in bindings yet => must actively materialize
      //               This code would be relevant if materializer also
      //               supported 'expressions to be materialized     */
      //            cout << "Must actively materialize field now" << endl;
      //            const vector<expressions::Expression*>& wantedExpressions =
      //                    mat.getWantedExpressions();
      //            expressions::Expression* currExpr =
      //            wantedExpressions.at(offsetInWanted);
      //            ExpressionGeneratorVisitor exprGenerator =
      //            ExpressionGeneratorVisitor(context, childState);
      //            ProteusValue currVal = currExpr->accept(exprGenerator);
      //            llvmCurrVal = currVal.value;

      string error_msg =
          string("[NEST: ] Binding not found") + wantedField->getAttrName();
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }

    // FIXME FIX THE NECESSARY CONVERSIONS HERE
    Value *valToMaterialize =
        pg.convert(llvmCurrVal->getType(),
                   materializedTypes->at(offsetInWanted), llvmCurrVal);
    vector<Value *> idxList = vector<Value *>();
    idxList.push_back(context->createInt32(0));
    idxList.push_back(context->createInt32(offsetInStruct));
    // Shift in struct ptr
    Value *structPtr = Builder->CreateGEP(
        mem_payload->getType()->getNonOpaquePointerElementType(), mem_payload,
        idxList);
    Builder->CreateStore(valToMaterialize, structPtr);
    offsetInStruct++;
    offsetInWanted++;
  }

  // 3. Inserting payload in HT
  PointerType *voidType = PointerType::get(IntegerType::get(llvmContext, 8), 0);
  Value *voidCast =
      Builder->CreateBitCast(mem_payload, voidType, "valueVoidCast");
  Value *globalStr = context->CreateGlobalString(htName);
  // Prepare hash_insert_function arguments
  ArgsV.clear();
  ArgsV.push_back(globalStr);
  ArgsV.push_back(groupKey.value);
  ArgsV.push_back(voidCast);
  // Passing size as well
  ArgsV.push_back(context->createInt32(pg.getPayloadTypeSize()));
  Function *insert = context->getFunction("insertHT");
  Builder->CreateCall(insert, ArgsV);
#ifdef DEBUG
//            ArgsV.clear();
//            ArgsV.push_back(context->createInt32(-7));
//            Builder->CreateCall(debugInt, ArgsV);
//            ArgsV.clear();
#endif
}

void Nest::generateProbe(ParallelContext *context) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Catalog &catalog = Catalog::getInstance();
  vector<Value *> ArgsV;
  Value *globalStr = context->CreateGlobalString(htName);
  Type *int64_type = IntegerType::get(llvmContext, 64);

  /**
   * Start injecting code at previous 'ending' point
   * Must also update what is considered the ending point
   * (-> loopEndHT)
   */
  Builder->SetInsertPoint(context->getEndingBlock());

  /**
   * STEP: Foreach key, loop through corresponding bucket.
   */

  // 1. Find out each bucket's size
  // Get an array of structs containing (hashKey, bucketSize)
  // XXX Correct way to do this: have your HT implementation maintain this info
  // Result type specified
  StructType *metadataType = context->getHashtableMetadataType();
  PointerType *metadataArrayType = PointerType::get(metadataType, 0);

  // Get HT Metadata
  ArgsV.clear();
  ArgsV.push_back(globalStr);
  Function *getMetadata = context->getFunction("getMetadataHT");
  // Given that I changed the return type in raw-context,
  // no casting should be needed
  Value *metadataArray = Builder->CreateCall(getMetadata, ArgsV);

  // 2. Loop through buckets
  /**
   * foreach key in HT:
   *         ...
   */
  BasicBlock *loopCondHT, *loopBodyHT, *loopIncHT, *loopEndHT;
  context->CreateForLoop("LoopCondHT", "LoopBodyHT", "LoopIncHT", "LoopEndHT",
                         &loopCondHT, &loopBodyHT, &loopIncHT, &loopEndHT);
  context->setEndingBlock(loopEndHT);
  // Entry Block of bucket loop - Initializing counter
  BasicBlock *codeSpot = Builder->GetInsertBlock();
  PointerType *i8_ptr = PointerType::get(IntegerType::get(llvmContext, 8), 0);
  //    ConstantPointerNull* const_null = ConstantPointerNull::get(i8_ptr);
  AllocaInst *mem_bucketCounter = context->createAlloca(
      codeSpot, "mem_bucketCnt", IntegerType::get(llvmContext, 64));
  Builder->CreateStore(context->createInt64(0), mem_bucketCounter);
  Builder->CreateBr(loopCondHT);

  Builder->SetInsertPoint(loopCondHT);
  // Condition:  current bucketSize in result array positions examined is set to
  // 0
  Value *bucketCounter = Builder->CreateLoad(
      mem_bucketCounter->getType()->getPointerElementType(), mem_bucketCounter);
  Value *mem_arrayShifted =
      context->getArrayElemMem(metadataArray, bucketCounter);
  Value *bucketSize = context->getStructElem(mem_arrayShifted, 1);
  Value *htKeysEnd = Builder->CreateICmpNE(bucketSize, context->createInt64(0),
                                           "cmpMatchesEnd");
  Builder->CreateCondBr(htKeysEnd, loopBodyHT, loopEndHT);

  // Body per Key
  Builder->SetInsertPoint(loopBodyHT);

  // 3. (nested) Loop through EACH bucket chain (i.e. all results for a key)
  /**
   * foreach value in HT[key]:
   *         ...
   *
   * (Should be) very relevant to join
   */
  AllocaInst *mem_metadataStruct =
      context->CreateEntryBlockAlloca(TheFunction, "currKeyNest", metadataType);
  Value *arrayShifted = Builder->CreateLoad(
      mem_arrayShifted->getType()->getPointerElementType(), mem_arrayShifted);

  Builder->CreateStore(arrayShifted, mem_metadataStruct);
  Value *currKey = context->getStructElem(mem_metadataStruct, 0);
  Value *currBucketSize = context->getStructElem(mem_metadataStruct, 1);

  // Retrieve HT[key] (Perform the actual probe)
  ArgsV.clear();
  ArgsV.push_back(globalStr);
  ArgsV.push_back(currKey);

  Function *probe = context->getFunction("probeHT");
  Value *voidHTBindings = Builder->CreateCall(probe, ArgsV);

  BasicBlock *loopCondBucket, *loopBodyBucket, *loopIncBucket, *loopEndBucket;
  context->CreateForLoop("LoopCondBucket", "LoopBodyBucket", "LoopIncBucket",
                         "LoopEndBucket", &loopCondBucket, &loopBodyBucket,
                         &loopIncBucket, &loopEndBucket);

  // Setting up entry block
  AllocaInst *mem_valuesCounter = context->CreateEntryBlockAlloca(
      TheFunction, "ht_val_counter", int64_type);
  Builder->CreateStore(context->createInt64(0), mem_valuesCounter);
  AllocaInst *mem_accumulating = resetAccumulator(context);
  Builder->CreateBr(loopCondBucket);

  // Condition: are there any more values in the bucket?
  Builder->SetInsertPoint(loopCondBucket);
  Value *valuesCounter = Builder->CreateLoad(
      mem_valuesCounter->getType()->getPointerElementType(), mem_valuesCounter);
  Value *cond = Builder->CreateICmpEQ(valuesCounter, currBucketSize);
  Builder->CreateCondBr(cond, loopEndBucket, loopBodyBucket);

  /**
   * [BODY] Time to do work per value:
   * -> 3a. find out what this value actually is (i.e., build OperatorState)
   * -> 3b. Differentiate behavior based on monoid type. Foreach, do:
   * ->-> evaluate predicate
   * ->-> compute expression
   * ->-> partially compute operator output
   */
  Builder->SetInsertPoint(loopBodyBucket);

  // 3a. Loop through bindings and recreate/assemble OperatorState (i.e.,
  // deserialize)
  Value *currValue = context->getArrayElem(voidHTBindings, valuesCounter);

  // Result (payload) type and appropriate casts
  int typeIdx = Catalog::getInstance().getTypeIndex(string(this->htName));
  Value *idx = context->createInt32(typeIdx);
  Type *structType = Catalog::getInstance().getTypeInternal(typeIdx);
  PointerType *structPtrType = context->getPointerType(structType);
  auto *str = (llvm::StructType *)structType;

  // Casting currValue from void* back to appropriate type
  AllocaInst *mem_currValueCasted = context->CreateEntryBlockAlloca(
      TheFunction, "mem_currValueCasted", structPtrType);
  Value *currValueCasted = Builder->CreateBitCast(currValue, structPtrType);
  Builder->CreateStore(currValueCasted, mem_currValueCasted);

  unsigned elemNo = str->getNumElements();
  std::map<RecordAttribute, ProteusValueMemory> allBucketBindings;
  int i = 0;
  // Retrieving activeTuple(s) from HT
  AllocaInst *mem_activeTuple = nullptr;
  Value *activeTuple = nullptr;
  const vector<RecordAttribute *> &tuplesIdentifiers = mat.getWantedOIDs();
  for (auto attr : tuplesIdentifiers) {
    mem_activeTuple = context->CreateEntryBlockAlloca(
        TheFunction, "mem_activeTuple", str->getElementType(i));
    Value *currValueCasted = Builder->CreateLoad(
        mem_currValueCasted->getType()->getPointerElementType(),
        mem_currValueCasted);
    activeTuple = context->getStructElem(currValueCasted, i);
    Builder->CreateStore(activeTuple, mem_activeTuple);

    ProteusValueMemory mem_valWrapper{mem_activeTuple, context->createFalse()};
    allBucketBindings[*attr] = mem_valWrapper;
    i++;
  }

  const vector<RecordAttribute *> &wantedFields = mat.getWantedFields();
  Value *field = nullptr;
  for (auto wantedField : wantedFields) {
    string currField = wantedField->getName();
    AllocaInst *mem_field = context->CreateEntryBlockAlloca(
        TheFunction, currField + "mem", str->getElementType(i));

    field = context->getStructElem(mem_currValueCasted, i);
    Builder->CreateStore(field, mem_field);
    i++;

    ProteusValueMemory mem_valWrapper{mem_field, context->createFalse()};
    allBucketBindings[*wantedField] = mem_valWrapper;
    LOG(INFO) << "[HT Bucket Traversal: ] Binding name: " << currField;
  }
  OperatorState newState(*this, allBucketBindings);

#ifdef DEBUG
//    cout << "[Nest ] Bindings after probing HT:" << endl;
//    map<RecordAttribute, ProteusValueMemory>::const_iterator itt =
//    (*allBucketBindings).begin(); for( ; itt != (*allBucketBindings).end();
//    itt++)    {
//        cout << "Active att " << itt->first.getOriginalRelationName() << " - "
//        << itt->first.getAttrName() << endl;
//    }
#endif

  switch (acc) {
    case SUM:
      generateSum(context, newState, mem_accumulating);
      break;
    case MULTIPLY:
      //            generateMul(context, childState);
      //            break;
    case MAX:
      //            generateMax(context, childState);
      //            break;
    case OR:
      //            generateOr(context, childState);
      //            break;
    case AND:
      //            generateAnd(context, childState);
      //            break;
    case UNION:
      //        generateUnion(context, childState);
      //        break;
    case BAGUNION:
      //        generateBagUnion(context, childState);
      //        break;
    case APPEND:
      //        generateAppend(context, childState);
      //        break;
    default: {
      string error_msg = string("[Nest: ] Unknown accumulator");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  /**
   * [INC - HT CHAIN NO.] Increase value counter in (specific) bucket
   * Continue inner loop
   */
  Builder->CreateBr(loopIncBucket);
  Builder->SetInsertPoint(loopIncBucket);

  valuesCounter = Builder->CreateLoad(
      mem_valuesCounter->getType()->getPointerElementType(), mem_valuesCounter);
  Value *inc_valuesCounter =
      Builder->CreateAdd(valuesCounter, context->createInt64(1));
  Builder->CreateStore(inc_valuesCounter, mem_valuesCounter);
  Builder->CreateBr(loopCondBucket);

  /**
   * [END - HT CHAIN LOOP.] End inner loop
   * 4. Time to produce output tuple & forward to next operator
   */
  Builder->SetInsertPoint(loopEndBucket);

  /**
   * FIXME
   * Must forward only the fields contributing in v
   * and the result of the aggr.
   */

  Plugin *htPlugin = new BinaryInternalPlugin(context, htName);
  RecordAttribute attr_aggr =
      RecordAttribute(htName, aggregateName, outputExpr->getExpressionType());
  catalog.registerPlugin(htName, htPlugin);
  // cout << "Registering custom pg for " << htName << endl;
  ProteusValueMemory mem_aggrWrapper{mem_accumulating, context->createFalse()};
  allBucketBindings[attr_aggr] = mem_aggrWrapper;

  /* Explicit oid (i.e., bucketNo) materialization */
  ProteusValueMemory mem_oidWrapper{mem_bucketCounter, context->createFalse()};
  ExpressionType *oidType = new IntType();
  RecordAttribute attr_oid(htName, activeLoop, oidType);
  allBucketBindings[attr_oid] = mem_oidWrapper;
  //#ifdef DEBUG
  //        ArgsV.clear();
  //        Function* debugInt64 = context->getFunction("printi");
  //        Value* finalResult =
  //        Builder->CreateLoad(mem_accumulating->getType()->getPointerElementType(),
  //        mem_accumulating); ArgsV.push_back(finalResult);
  //        Builder->CreateCall(debugInt64, ArgsV);
  //        ArgsV.clear();
  //        ArgsV.push_back(context->createInt32(-7));
  //        Builder->CreateCall(debugInt64, ArgsV);
  //        ArgsV.clear();
  //#endif

  OperatorState groupState(*this, allBucketBindings);
  getParent()->consume(context, groupState);

  /**
   * [INC - HT BUCKET NO.] Continue outer loop
   */
  bucketCounter = Builder->CreateLoad(
      mem_bucketCounter->getType()->getPointerElementType(), mem_bucketCounter);
  Value *val_inc = Builder->getInt64(1);
  Value *val_new = Builder->CreateAdd(bucketCounter, val_inc);
  Builder->CreateStore(val_new, mem_bucketCounter);
  Builder->CreateBr(loopIncHT);

  Builder->SetInsertPoint(loopIncHT);
  Builder->CreateBr(loopCondHT);

  // Ending block of buckets loop
  Builder->SetInsertPoint(loopEndHT);
}

void Nest::generateSum(ParallelContext *context, const OperatorState &state,
                       AllocaInst *mem_accumulating) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Generate condition
  ExpressionGeneratorVisitor predExprGenerator =
      ExpressionGeneratorVisitor(context, state);
  ProteusValue condition = pred->accept(predExprGenerator);
  /**
   * Predicate Evaluation:
   */
  BasicBlock *entryBlock = Builder->GetInsertBlock();
  BasicBlock *endBlock =
      BasicBlock::Create(llvmContext, "nestCondEnd", TheFunction);
  BasicBlock *ifBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond", &ifBlock,
                         endBlock);

  /**
   * IF(pred) Block
   */
  ExpressionGeneratorVisitor outputExprGenerator(context, state);
  Builder->SetInsertPoint(entryBlock);
  Builder->CreateCondBr(condition.value, ifBlock, endBlock);

  Builder->SetInsertPoint(ifBlock);
  auto val_output = outputExpr->accept(outputExprGenerator);
  Value *val_accumulating = Builder->CreateLoad(
      mem_accumulating->getType()->getPointerElementType(), mem_accumulating);

  switch (outputExpr->getExpressionType()->getTypeID()) {
    case INT: {
#ifdef DEBUGNEST
//        vector<Value*> ArgsV;
//        Function* debugInt = context->getFunction("printi");
//        ArgsV.push_back(val_accumulating);
//        Builder->CreateCall(debugInt, ArgsV);
#endif
      Value *val_new = Builder->CreateAdd(val_accumulating, val_output.value);
      Builder->CreateStore(val_new, mem_accumulating);
      Builder->CreateBr(endBlock);
#ifdef DEBUGNEST
//        Builder->SetInsertPoint(endBlock);
//        vector<Value*> ArgsV;
//        Function* debugInt = context->getFunction("printi");
//        Value* finalResult =
//        Builder->CreateLoad(mem_accumulating->getType()->getPointerElementType(),
//        mem_accumulating); ArgsV.push_back(finalResult);
//        Builder->CreateCall(debugInt, ArgsV);
//        //Back to 'normal' flow
//        Builder->SetInsertPoint(ifBlock);
#endif
      break;
    }
    case FLOAT: {
      Value *val_new = Builder->CreateFAdd(val_accumulating, val_output.value);
      Builder->CreateStore(val_new, mem_accumulating);
      Builder->CreateBr(endBlock);
      break;
    }
    default: {
      string error_msg =
          string("[Nest: ] Sum accumulator operates on numerics");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  /**
   * END Block
   */
  Builder->SetInsertPoint(endBlock);
}

AllocaInst *Nest::resetAccumulator(ParallelContext *context) const {
  AllocaInst *mem_accumulating = nullptr;

  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *f = Builder->GetInsertBlock()->getParent();

  Type *int1Type = Type::getInt1Ty(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *doubleType = Type::getDoubleTy(llvmContext);

  // Deal with 'memory allocations' as per monoid type requested
  typeID outputType = outputExpr->getExpressionType()->getTypeID();
  switch (acc) {
    case SUM: {
      switch (outputType) {
        case INT: {
          mem_accumulating =
              context->CreateEntryBlockAlloca(f, string("dest_acc"), int32Type);
          Value *val_zero = Builder->getInt32(0);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        case FLOAT: {
          Type *doubleType = Type::getDoubleTy(llvmContext);
          mem_accumulating = context->CreateEntryBlockAlloca(
              f, string("dest_acc"), doubleType);
          Value *val_zero = ConstantFP::get(doubleType, 0.0);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        default: {
          string error_msg =
              string("[Nest: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
      break;
    }
    case MULTIPLY: {
      switch (outputType) {
        case INT: {
          mem_accumulating =
              context->CreateEntryBlockAlloca(f, string("dest_acc"), int32Type);
          Value *val_zero = Builder->getInt32(1);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        case FLOAT: {
          mem_accumulating = context->CreateEntryBlockAlloca(
              f, string("dest_acc"), doubleType);
          Value *val_zero = ConstantFP::get(doubleType, 1.0);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        default: {
          string error_msg =
              string("[Nest: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
      break;
    }
    case MAX: {
      switch (outputType) {
        case INT: {
          mem_accumulating =
              context->CreateEntryBlockAlloca(f, string("dest_acc"), int32Type);
          /**
           * FIXME This is not the appropriate 'zero' value for integers.
           * It is the one for naturals
           */
          Value *val_zero = Builder->getInt32(0);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        case FLOAT: {
          mem_accumulating = context->CreateEntryBlockAlloca(
              f, string("dest_acc"), doubleType);
          /**
           * FIXME This is not the appropriate 'zero' value for floats.
           * It is the one for naturals
           */
          Value *val_zero = ConstantFP::get(doubleType, 0.0);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        default: {
          string error_msg =
              string("[Nest: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
      break;
    }
    case OR: {
      switch (outputType) {
        case BOOL: {
          mem_accumulating =
              context->CreateEntryBlockAlloca(f, string("dest_acc"), int1Type);
          Value *val_zero = Builder->getInt1(false);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        default: {
          string error_msg = string("[Nest: ] Or/And operate on booleans");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
      break;
    }
    case AND: {
      switch (outputType) {
        case BOOL: {
          mem_accumulating =
              context->CreateEntryBlockAlloca(f, string("dest_acc"), int1Type);
          Value *val_zero = Builder->getInt1(true);
          Builder->CreateStore(val_zero, mem_accumulating);
          break;
        }
        default: {
          string error_msg = string("[Nest: ] Or/And operate on booleans");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
      break;
    }
    case UNION:
    case BAGUNION: {
      string error_msg = string("[Nest: ] Not implemented yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    case APPEND: {
      // XXX Reduce has some more stuff on this
      string error_msg = string("[Nest: ] Not implemented yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    default: {
      string error_msg = string("[Nest: ] Unknown accumulator");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
  return mem_accumulating;
}
