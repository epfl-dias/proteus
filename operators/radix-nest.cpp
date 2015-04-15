/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2014
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

#include "operators/radix-nest.hpp"

namespace radix	{
/**
 * Identical constructor logic with the one of Reduce
 */
Nest::Nest(RawContext* const context, vector<Monoid> accs, vector<expressions::Expression*> outputExprs, vector<string> aggrLabels,
		expressions::Expression *pred,
		expressions::Expression *f_grouping,
		expressions::Expression *g_nullToZero,
		RawOperator* const child,
		char* opLabel, Materializer& mat) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs), aggregateLabels(aggrLabels),
		pred(pred),
		g_nullToZero(g_nullToZero), f_grouping(f_grouping),
		mat(mat), htName(opLabel), context(context)
{
	Function *F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<> *Builder = context->getBuilder();

	if (accs.size() != outputExprs.size() || accs.size() != aggrLabels.size()) {
		string error_msg = string("[NEST: ] Erroneous constructor args");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	/* HT-related */
	/* Request memory for HT(s) construction 		*/
	Type* int64_type = Type::getInt64Ty(llvmContext);
	Type* int32_type = Type::getInt32Ty(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *int32_ptr_type = PointerType::get(int32_type, 0);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);
	Value *zero = context->createInt64(0);
	keyType = int32_type;
	/* Note: Does not allocate mem. for buckets too */
	size_t htSize = (1 << NUM_RADIX_BITS) * sizeof(HT);
	HT_per_cluster = (HT *) getMemoryChunk(htSize);

	/* What the type of internal radix HT per cluster is 	*/
	/* (int32*, int32*, unit32_t, void*, int32) */
	vector<Type*> htRadixClusterMembers;
	htRadixClusterMembers.push_back(int32_ptr_type);
	htRadixClusterMembers.push_back(int32_ptr_type);
	/* LLVM does not make a distinction between signed and unsigned integer type:
	 * Both are lowered to i32
	 */
	htRadixClusterMembers.push_back(int32_type);
	htRadixClusterMembers.push_back(int32_type);
	htClusterType = StructType::get(context->getLLVMContext(),
			htRadixClusterMembers);
	PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);

	/* Arbitrary initial buffer sizes */
	/* No realloc will be required with these sizes for synthetic large-scale numbers */
	//	size_t sizeR = 10000000000;
	size_t sizeR = 1000;
	Value *val_sizeR = context->createInt64(sizeR);

	/* Request memory to store relation R 			*/
	relR.mem_relation = context->CreateEntryBlockAlloca(F, string("relationR"),
			char_ptr_type);
	(relR.mem_relation)->setAlignment(8);
	relR.mem_tuplesNo = context->CreateEntryBlockAlloca(F, string("tuplesR"),
			int64_type);
	relR.mem_size = context->CreateEntryBlockAlloca(F, string("sizeR"),
			int64_type);
	relR.mem_offset = context->CreateEntryBlockAlloca(F, string("offsetRelR"),
			int64_type);
	relationR = (char*) getMemoryChunk(sizeR);
	Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type, relationR);
	Builder->CreateStore(val_relationR, relR.mem_relation);
	Builder->CreateStore(zero, relR.mem_tuplesNo);
	Builder->CreateStore(zero, relR.mem_offset);
	Builder->CreateStore(val_sizeR, relR.mem_size);

	/* XXX What the type of HT entries is */
	/* (size_t, size_t) */
	vector<Type*> htEntryMembers;
	htEntryMembers.push_back(int64_type);
	htEntryMembers.push_back(int64_type);
	int htEntrySize = sizeof(size_t) + sizeof(size_t);
	htEntryType = StructType::get(context->getLLVMContext(), htEntryMembers);
	PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

	/* Request memory to store HT entries of R */
	htR.mem_kv = context->CreateEntryBlockAlloca(F, string("htR"),
			htEntryPtrType);
	(htR.mem_kv)->setAlignment(8);

	htR.mem_tuplesNo = context->CreateEntryBlockAlloca(F, string("tuplesR"),
			int64_type);
	htR.mem_size = context->CreateEntryBlockAlloca(F, string("sizeR"),
			int64_type);
	htR.mem_offset = context->CreateEntryBlockAlloca(F, string("offsetRelR"),
			int64_type);
	int kvSizeR = sizeR; // * htEntrySize;
	kvR = (char*) getMemoryChunk(kvSizeR);
	Value *val_kvR = context->CastPtrToLlvmPtr(htEntryPtrType, kvR);

	StoreInst *store_htR = Builder->CreateStore(val_kvR, htR.mem_kv);
	store_htR->setAlignment(8);
	Builder->CreateStore(zero, htR.mem_tuplesNo);
	Builder->CreateStore(context->createInt64(kvSizeR), htR.mem_size);
	Builder->CreateStore(zero, htR.mem_offset);

	/* Defined in consume() */
	payloadType = NULL;
}

void Nest::produce()	const {
	getChild()->produce();

	generateProbe(this->context);
}

void Nest::consume(RawContext* const context, const OperatorState& childState) {
	generateInsert(context, childState);
}

/**
 * This function will be launched twice, since both outer unnest + join
 * cause two code paths to be generated.
 */
void Nest::generateInsert(RawContext* context, const OperatorState& childState)
{
	this->context = context;

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	RawCatalog& catalog = RawCatalog::getInstance();
	vector<Value*> ArgsV;
	Function* debugInt = context->getFunction("printi");

#ifdef DEBUG
//	ArgsV.clear();
//	ArgsV.push_back(context->createInt32(-6));
//	Builder->CreateCall(debugInt, ArgsV);
//	ArgsV.clear();
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

	//1. Compute HASH key based on grouping expression
	ExpressionHasherVisitor aggrExprGenerator = ExpressionHasherVisitor(context, childState);
	RawValue groupKey = f_grouping->accept(aggrExprGenerator);

	//2. Create 'payload' --> What is to be inserted in the bucket
	LOG(INFO) << "[NEST: ] Creating payload";
	const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();
	OutputPlugin *pg = new OutputPlugin(context, mat, bindings);

	//Result type specified during output plugin construction
	llvm::StructType *payloadType = pg->getPayloadType();
	//Creating space for the payload. XXX Might have to set alignment explicitly
	AllocaInst *mem_payload = context->CreateEntryBlockAlloca(TheFunction,string("valueInHT"),payloadType);

	//Registering payload type of HT in RAW CATALOG
	catalog.insertTableInfo(string(this->htName),payloadType);

	// Creating and Populating Payload Struct
	int offsetInStruct = 0; //offset inside the struct (+current field manipulated)
	vector<Type*>* materializedTypes = pg->getMaterializedTypes();



	//Storing values in struct to be materialized in HT. Two steps
	//2a. Materializing all 'activeTuples' (i.e. positional indices) met so far
	RawValueMemory mem_activeTuple;
	{
		map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
		for(memSearch = bindings.begin(); memSearch != bindings.end(); memSearch++)	{
			RecordAttribute currAttr = memSearch->first;
			if(currAttr.getAttrName() == activeLoop)	{
				mem_activeTuple = memSearch->second;
				Value* val_activeTuple = Builder->CreateLoad(mem_activeTuple.mem);
				//OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR EXTRA OFFSET
				vector<Value*> idxList = vector<Value*>();
				idxList.push_back(context->createInt32(0));
				idxList.push_back(context->createInt32(offsetInStruct++));
				//Shift in struct ptr
				Value* structPtr = Builder->CreateGEP(mem_payload, idxList);
				Builder->CreateStore(val_activeTuple,structPtr);
			}
		}
	}

	//2b. Materializing all explicitly requested fields
	int offsetInWanted = 0;
	const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
	for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it!=wantedFields.end(); ++it)
	{
		map<RecordAttribute, RawValueMemory>::const_iterator memSearch = bindings.find(*(*it));

		Value* llvmCurrVal = NULL;
		if (memSearch != bindings.end())
		{
			RawValueMemory currValMem = memSearch->second;
			llvmCurrVal = Builder->CreateLoad(currValMem.mem);
		}
		else
		{
//			/* Not in bindings yet => must actively materialize
//			   This code would be relevant if materializer also
//			   supported 'expressions to be materialized 	*/
//			cout << "Must actively materialize field now" << endl;
//			const vector<expressions::Expression*>& wantedExpressions =
//					mat.getWantedExpressions();
//			expressions::Expression* currExpr = wantedExpressions.at(offsetInWanted);
//			ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
//			RawValue currVal = currExpr->accept(exprGenerator);
//			llvmCurrVal = currVal.value;

			string error_msg = string("[NEST: ] Binding not found") + (*it)->getAttrName();
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}

		//FIXME FIX THE NECESSARY CONVERSIONS HERE
		Value* valToMaterialize = pg->convert(llvmCurrVal->getType(),materializedTypes->at(offsetInWanted),llvmCurrVal);
		vector<Value*> idxList = vector<Value*>();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));
		//Shift in struct ptr
		Value* structPtr = Builder->CreateGEP(mem_payload, idxList);
		Builder->CreateStore(valToMaterialize,structPtr);
		offsetInStruct++;
		offsetInWanted++;
	}



	//3. Inserting payload in HT
	PointerType* voidType = PointerType::get(IntegerType::get(llvmContext, 8), 0);
	Value* voidCast = Builder->CreateBitCast(mem_payload, voidType,"valueVoidCast");
	Value* globalStr = context->CreateGlobalString(htName);
	//Prepare hash_insert_function arguments
	ArgsV.clear();
	ArgsV.push_back(globalStr);
	ArgsV.push_back(groupKey.value);
	ArgsV.push_back(voidCast);
	//Passing size as well
	ArgsV.push_back(context->createInt32(pg->getPayloadTypeSize()));
	Function* insert = context->getFunction("insertHT");
	Builder->CreateCall(insert, ArgsV);
#ifdef DEBUG
//			ArgsV.clear();
//			ArgsV.push_back(context->createInt32(-7));
//			Builder->CreateCall(debugInt, ArgsV);
//			ArgsV.clear();
#endif
}

void Nest::buildHT(RawContext* context, const OperatorState& childState) {
	/* Prepare codegen utils */
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();
	Function *debugInt = context->getFunction("printi");
	Function *debugInt64 = context->getFunction("printi64");
	//Function *debug = context->getFunction("debug");

	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	Type *int64_type = Type::getInt64Ty(llvmContext);
	Type *int32_type = Type::getInt32Ty(llvmContext);

	Value *kvSize = ConstantExpr::getSizeOf(htEntryType);

	const map<RecordAttribute, RawValueMemory>& bindings =
			childState.getBindings();
	OutputPlugin* pg = new OutputPlugin(context, mat, bindings);

	/* Result type specified during output plugin construction */
	payloadType = pg->getPayloadType();

	/* Place info in cache */
	{
		CachingService& cache = CachingService::getInstance();
		bool fullRelation = !(this->getChild())->isFiltering();
		const vector<expressions::Expression*>& expsLeft =
				mat.getWantedExpressions();
		const vector<RecordAttribute*>& fieldsLeft = mat.getWantedFields();
		/* Note: wantedFields do not include activeTuple */
		vector<RecordAttribute*>::const_iterator itRec = fieldsLeft.begin();
		int fieldNo = 0;
		CacheInfo info;
		const set<RecordAttribute>& oids = mat.getTupleIdentifiers();
		set<RecordAttribute>::const_iterator itOids = oids.begin();
		for (; itOids != oids.end(); itOids++) {
			//				cout << "OID mat'ed" << endl;
			info.objectTypes.push_back(itOids->getOriginalType()->getTypeID());
		}
		for (; itRec != fieldsLeft.end(); itRec++) {
			//				cout << "Field mat'ed" << endl;
			info.objectTypes.push_back(
					(*itRec)->getOriginalType()->getTypeID());
		}
		itRec = fieldsLeft.begin();
		/* Explicit OID ('activeTuple') will be field 0 */
		if (!expsLeft.empty()) {

			/* By default, cache looks sth like custom_struct*.
			 * Is it possible to isolate cache for just ONE of the expressions??
			 * Group of expressions probably more palpable */
			vector<expressions::Expression*>::const_iterator it =
					expsLeft.begin();
			for (; it != expsLeft.end(); it++) {
				//info.objectType = rPayloadType;
				info.structFieldNo = fieldNo;
				info.payloadPtr = relationR;
				cache.registerCache(*it, info, fullRelation);

				/* Having skipped OIDs */
				if (fieldNo >= mat.getTupleIdentifiers().size()) {
					cout << "[Radix: ] Field Cached: "
							<< (*itRec)->getAttrName() << endl;
					itRec++;
				} else {
					cout << "[Radix: ] Field Cached: " << activeLoop << endl;
				}

				fieldNo++;
			}
		}
	}

	/* 3rd Method to calculate size */
	/* REMEMBER: PADDING DOES MATTER! */
	Value* val_payloadSize = ConstantExpr::getSizeOf(payloadType);

	/*
	 * Prepare payload.
	 * What the 'output plugin + materializer' have decided is orthogonal
	 * to this materialization policy
	 * (i.e., whether we keep payload with the key, or just point to it)
	 *
	 * Instead of allocating space for payload and then copying it again,
	 * do it ONCE on the pre-allocated buffer
	 */

	Value *val_arena = Builder->CreateLoad(relR.mem_relation);
	Value *offsetInArena = Builder->CreateLoad(relR.mem_offset);
	Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,
			val_payloadSize);
	Value *arenaSize = Builder->CreateLoad(relR.mem_size);
	Value* val_tuplesNo = Builder->CreateLoad(relR.mem_tuplesNo);

	/* if(offsetInArena + payloadSize >= arenaSize) */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlockArenaFull = BasicBlock::Create(llvmContext,
			"IfArenaFullEnd", F);
	BasicBlock *ifArenaFull;
	context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull,
			endBlockArenaFull);
	Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload, arenaSize);

	Builder->CreateCondBr(offsetCond, ifArenaFull, endBlockArenaFull);

	/* true => realloc() */
	Builder->SetInsertPoint(ifArenaFull);

	vector<Value*> ArgsRealloc;
	Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
	AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type, 0,
			"voidArenaPtr");
	Builder->CreateStore(val_arena, mem_arena_void);
	Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
	ArgsRealloc.push_back(val_arena_void);
	ArgsRealloc.push_back(arenaSize);
	Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

	Builder->CreateStore(val_newArenaVoidPtr, relR.mem_relation);
	Value* val_size = Builder->CreateLoad(relR.mem_size);
	val_size = Builder->CreateMul(val_size, context->createInt64(2));
	Builder->CreateStore(val_size, relR.mem_size);
	Builder->CreateBr(endBlockArenaFull);

	/* 'Normal' flow again */
	Builder->SetInsertPoint(endBlockArenaFull);

	/* Repeat load - realloc() might have occurred */
	val_arena = Builder->CreateLoad(relR.mem_relation);
	val_size = Builder->CreateLoad(relR.mem_size);

	/* XXX STORING PAYLOAD */
	/* 1. arena += (offset) */
	Value *ptr_arenaShifted = Builder->CreateInBoundsGEP(val_arena,
			offsetInArena);

	/* 2. Casting */
	PointerType *ptr_payloadType = PointerType::get(payloadType, 0);
	Value *cast_arenaShifted = Builder->CreateBitCast(ptr_arenaShifted,
			ptr_payloadType);

	/* 3. Storing payload, one field at a time */
	vector<Type*>* materializedTypes = pg->getMaterializedTypes();
	//Storing all activeTuples met so far
	int offsetInStruct = 0; //offset inside the struct (+current field manipulated)
	RawValueMemory mem_activeTuple;
	{
		map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
		for (memSearch = bindings.begin(); memSearch != bindings.end();
				memSearch++) {
			RecordAttribute currAttr = memSearch->first;
			if (currAttr.getAttrName() == activeLoop) {
				mem_activeTuple = memSearch->second;
				Value* val_activeTuple = Builder->CreateLoad(
						mem_activeTuple.mem);
				//OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR EXTRA OFFSET
				vector<Value*> idxList = vector<Value*>();
				idxList.push_back(context->createInt32(0));
				idxList.push_back(context->createInt32(offsetInStruct));
				//Shift in struct ptr
				Value* structPtr = Builder->CreateGEP(cast_arenaShifted,
						idxList);
				StoreInst *store_activeTuple = Builder->CreateStore(
						val_activeTuple, structPtr);
				store_activeTuple->setAlignment(8);
				offsetInStruct++;
			}
		}
	}

	int offsetInWanted = 0;
	const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
	for (vector<RecordAttribute*>::const_iterator it = wantedFields.begin();
			it != wantedFields.end(); ++it) {
		map<RecordAttribute, RawValueMemory>::const_iterator memSearch =
				bindings.find(*(*it));
		RawValueMemory currValMem = memSearch->second;
		/* FIX THE NECESSARY CONVERSIONS HERE */
		Value* currVal = Builder->CreateLoad(currValMem.mem);
		Value* valToMaterialize = pg->convert(currVal->getType(),
				materializedTypes->at(offsetInWanted), currVal);

		vector<Value*> idxList = vector<Value*>();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));

		//Shift in struct ptr
		Value* structPtr = Builder->CreateGEP(cast_arenaShifted, idxList);

		Builder->CreateStore(valToMaterialize, structPtr);
		offsetInStruct++;
		offsetInWanted++;
	}

	/* CONSTRUCT HTENTRY PAIR   	  */
	/* payloadPtr: relative offset from relBuffer beginning */
	/* (int64 key, int64 payloadPtr)  */
	/*  -> BOOST HASH PRODUCES INT64!! */
	/* Prepare key/pieces of key */
	/* XXX Note: key will be already hashed
	 * W/e I need to store to retrieve key will be placed in payload
	 * (if not there already)
	 */


	ExpressionHasherVisitor aggrExprGenerator = ExpressionHasherVisitor(context,childState);

	/* XXX Actually, this is work that the MATERIALIZER should do!!!
	 * Executor should remain agnostic!!!
	 * At the time of HT probing, we will go ahead and use
	 * an ExpressionGeneratorVisitor accordingly (for the dot product */
//	if(f_grouping->getTypeID() != expressions::RECORD_CONSTRUCTION)	{
//		/* Don't need to examine key piece by piece */
//	}
//	else
//	{
//		ExpressionGeneratorVisitor exprGenerator =
//				ExpressionGeneratorVisitor(context, childState);
//		/*
//		 * Need to make sure I have all that is needed to evaluate
//		 * the Record Construction
//		 */
//		expressions::RecordConstruction *recCons =
//				(expressions::RecordConstruction *) f_grouping;
//		list<expressions::AttributeConstruction>::const_iterator it;
//		for(it = recCons->getAtts().begin(); it != recCons->getAtts().end(); it++)	{
//
//			expressions::AttributeConstruction attr = *it;
//			expressions::Expression *attrExpr = attr.getExpression();
//			if(attrExpr->getTypeID() == expressions::RECORD_PROJECTION)
//			{
//
//			}
//			else
//			{
//				/* No need to do anything */
//				/* Can recreate from activeTuple */
//			}
//			/* const map<RecordAttribute, RawValueMemory>& bindings =
//			childState.getBindings();*/
//		}
//	}

	RawValue groupHashKey = f_grouping->accept(aggrExprGenerator);

	PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

	BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext, "IfHTFullEnd",
			F);
	BasicBlock *ifHTFull;
	context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull, endBlockHTFull);

	LoadInst *val_ht = Builder->CreateLoad(htR.mem_kv);
	val_ht->setAlignment(8);

	Value *offsetInHT = Builder->CreateLoad(htR.mem_offset);
	Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);

	Value *htSize = Builder->CreateLoad(htR.mem_size);
	offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair, htSize);

	Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

	/* true => realloc() */
	Builder->SetInsertPoint(ifHTFull);

	/* Casting htEntry* to void* requires a cast */
	Value *cast_htEntries = Builder->CreateBitCast(val_ht, void_ptr_type);
	ArgsRealloc.clear();
	ArgsRealloc.push_back(cast_htEntries);
	ArgsRealloc.push_back(htSize);
	Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

	Value *val_newHTPtr = Builder->CreateBitCast(val_newVoidHTPtr,
			htEntryPtrType);
	Builder->CreateStore(val_newHTPtr, htR.mem_kv);
	val_size = Builder->CreateLoad(htR.mem_size);
	val_size = Builder->CreateMul(val_size, context->createInt64(2));
	Builder->CreateStore(val_size, htR.mem_size);
	Builder->CreateBr(endBlockHTFull);

	/* Insert ht entry in HT */
	Builder->SetInsertPoint(endBlockHTFull);

	/* Repeat load - realloc() might have occurred */
	val_ht = Builder->CreateLoad(htR.mem_kv);
	val_ht->setAlignment(8);

	val_size = Builder->CreateLoad(htR.mem_size);

	/* 1. kv += offset */
	/* Note that we already have a htEntry ptr here */
	Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht, val_tuplesNo);

	/* 2a. kv_cast->keyPtr = &key */
	offsetInStruct = 0;
	//Shift in htEntry (struct) ptr
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(offsetInStruct));

	Value* structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
	StoreInst *store_key = Builder->CreateStore(groupHashKey.value, structPtr);
	store_key->setAlignment(8); //Used to be 4

	/* 2b. kv_cast->payloadPtr = &payload */
	offsetInStruct = 1;
	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(offsetInStruct));
	structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);

	StoreInst *store_payloadPtr = Builder->CreateStore(offsetInArena,
			structPtr);
	store_payloadPtr->setAlignment(8);

	/* 4. Increment counts - both Rel and HT */
	Builder->CreateStore(offsetPlusPayload, relR.mem_offset);
	Builder->CreateStore(offsetPlusKVPair, htR.mem_offset);
	val_tuplesNo = Builder->CreateAdd(val_tuplesNo, context->createInt64(1));
	Builder->CreateStore(val_tuplesNo, relR.mem_tuplesNo);
	Builder->CreateStore(val_tuplesNo, htR.mem_tuplesNo);
}

void Nest::generateProbe(RawContext* const context) const
{
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	RawCatalog& catalog = RawCatalog::getInstance();
	vector<Value*> ArgsV;
	Value* globalStr = context->CreateGlobalString(htName);
	Type* int64_type = IntegerType::get(llvmContext, 64);

	/**
	 * Start injecting code at previous 'ending' point
	 * Must also update what is considered the ending point
	 * (-> loopEndHT)
	 */
	Builder->SetInsertPoint(context->getEndingBlock());

	/**
	 * STEP: Foreach key, loop through corresponding bucket.
	 */

	//1. Find out each bucket's size
	//Get an array of structs containing (hashKey, bucketSize)
	//XXX Correct way to do this: have your HT implementation maintain this info
	//Result type specified
	StructType *metadataType = context->getHashtableMetadataType();
	PointerType *metadataArrayType = PointerType::get(metadataType, 0);

	//Get HT Metadata
	ArgsV.clear();
	ArgsV.push_back(globalStr);
	Function* getMetadata = context->getFunction("getMetadataHT");
	//Given that I changed the return type in raw-context,
	//no casting should be needed
	Value* metadataArray = Builder->CreateCall(getMetadata, ArgsV);

	//2. Loop through buckets
	/**
	 * foreach key in HT:
	 * 		...
	 */
	BasicBlock *loopCondHT, *loopBodyHT, *loopIncHT, *loopEndHT;
	context->CreateForLoop("LoopCondHT", "LoopBodyHT", "LoopIncHT", "LoopEndHT",
			&loopCondHT, &loopBodyHT, &loopIncHT, &loopEndHT);
	context->setEndingBlock(loopEndHT);
	//Entry Block of bucket loop - Initializing counter
	BasicBlock* codeSpot = Builder->GetInsertBlock();
	PointerType* i8_ptr = PointerType::get(IntegerType::get(llvmContext, 8), 0);
//	ConstantPointerNull* const_null = ConstantPointerNull::get(i8_ptr);
	AllocaInst* mem_bucketCounter = new AllocaInst(
			IntegerType::get(llvmContext, 64), "mem_bucketCnt", codeSpot);
	Builder->CreateStore(context->createInt64(0), mem_bucketCounter);
	Builder->CreateBr(loopCondHT);

	Builder->SetInsertPoint(loopCondHT);
	//Condition:  current bucketSize in result array positions examined is set to 0
	Value* bucketCounter = Builder->CreateLoad(mem_bucketCounter);
	Value* mem_arrayShifted = context->getArrayElemMem(metadataArray, bucketCounter);
	Value* bucketSize = context->getStructElem(mem_arrayShifted,1);
	Value* htKeysEnd = Builder->CreateICmpNE(bucketSize, context->createInt64(0),
			"cmpMatchesEnd");
	Builder->CreateCondBr(htKeysEnd, loopBodyHT, loopEndHT);

	//Body per Key
	Builder->SetInsertPoint(loopBodyHT);

	//3. (nested) Loop through EACH bucket chain (i.e. all results for a key)
	/**
	 * foreach value in HT[key]:
	 * 		...
	 *
	 * (Should be) very relevant to join
	 */
	AllocaInst* mem_metadataStruct = context->CreateEntryBlockAlloca(TheFunction,
				"currKeyNest", metadataType);
	Value* arrayShifted = Builder->CreateLoad(mem_arrayShifted);

	Builder->CreateStore(arrayShifted,mem_metadataStruct);
	Value* currKey = context->getStructElem(mem_metadataStruct,0);
	Value* currBucketSize = context->getStructElem(mem_metadataStruct,1);

	//Retrieve HT[key] (Perform the actual probe)
	ArgsV.clear();
	ArgsV.push_back(globalStr);
	ArgsV.push_back(currKey);

	Function* probe = context->getFunction("probeHT");
	Value* voidHTBindings = Builder->CreateCall(probe, ArgsV);

	BasicBlock *loopCondBucket, *loopBodyBucket, *loopIncBucket, *loopEndBucket;
	context->CreateForLoop("LoopCondBucket", "LoopBodyBucket", "LoopIncBucket",
			"LoopEndBucket", &loopCondBucket, &loopBodyBucket, &loopIncBucket,
			&loopEndBucket);

	//Setting up entry block
	AllocaInst* mem_valuesCounter = context->CreateEntryBlockAlloca(
							TheFunction, "ht_val_counter", int64_type);
	Builder->CreateStore(context->createInt64(0),mem_valuesCounter);


	vector<Monoid>::const_iterator itAcc = accs.begin();
	vector<expressions::Expression*>::const_iterator itExpr =
			outputExprs.begin();
	vector<AllocaInst*> mem_accumulators;
	/* Prepare accumulator FOREACH outputExpr */
	for (; itAcc != accs.end(); itAcc++, itExpr++) {
		Monoid acc = *itAcc;
		expressions::Expression *outputExpr = *itExpr;
		AllocaInst *mem_accumulator = resetAccumulator(outputExpr, acc);
		mem_accumulators.push_back(mem_accumulator);
	}
	Builder->CreateBr(loopCondBucket);

	//Condition: are there any more values in the bucket?
	Builder->SetInsertPoint(loopCondBucket);
	Value* valuesCounter = Builder->CreateLoad(mem_valuesCounter);
	Value* cond = Builder->CreateICmpEQ(valuesCounter,currBucketSize);
	Builder->CreateCondBr(cond,loopEndBucket,loopBodyBucket);

	/**
	 * [BODY] Time to do work per value:
	 * -> 3a. find out what this value actually is (i.e., build OperatorState)
	 * -> 3b. Differentiate behavior based on monoid type. Foreach, do:
	 * ->-> evaluate predicate
	 * ->-> compute expression
	 * ->-> partially compute operator output
	 */
	Builder->SetInsertPoint(loopBodyBucket);

	//3a. Loop through bindings and recreate/assemble OperatorState (i.e., deserialize)
	Value* currValue = context->getArrayElem(voidHTBindings,valuesCounter);

	//Result (payload) type and appropriate casts
	int typeIdx = RawCatalog::getInstance().getTypeIndex(string(this->htName));
	Value* idx = context->createInt32(typeIdx);
	Type* structType = RawCatalog::getInstance().getTypeInternal(typeIdx);
	PointerType* structPtrType = context->getPointerType(structType);
	StructType* str = (llvm::StructType*) structType;

	//Casting currValue from void* back to appropriate type
	AllocaInst *mem_currValueCasted = context->CreateEntryBlockAlloca(TheFunction,"mem_currValueCasted",structPtrType);
	Value* currValueCasted = Builder->CreateBitCast(currValue,structPtrType);
	Builder->CreateStore(currValueCasted,mem_currValueCasted);

	unsigned elemNo = str->getNumElements();
	map<RecordAttribute, RawValueMemory>* allBucketBindings = new map<RecordAttribute, RawValueMemory>();
	int i = 0;
	//Retrieving activeTuple(s) from HT
	AllocaInst *mem_activeTuple = NULL;
	Value *activeTuple = NULL;
	const set<RecordAttribute>& tuplesIdentifiers = mat.getTupleIdentifiers();
	for(set<RecordAttribute>::const_iterator it = tuplesIdentifiers.begin(); it!=tuplesIdentifiers.end(); it++)	{
		mem_activeTuple = context->CreateEntryBlockAlloca(TheFunction,"mem_activeTuple",str->getElementType(i));
		Value* currValueCasted = Builder->CreateLoad(mem_currValueCasted);
		activeTuple = context->getStructElem(currValueCasted,i);
		Builder->CreateStore(activeTuple,mem_activeTuple);

		RawValueMemory mem_valWrapper;
		mem_valWrapper.mem = mem_activeTuple;
		mem_valWrapper.isNull = context->createFalse();
		(*allBucketBindings)[*it] = mem_valWrapper;
		i++;
	}

	const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
	Value *field = NULL;
	for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it!= wantedFields.end(); ++it) {
		string currField = (*it)->getName();
		AllocaInst *mem_field = context->CreateEntryBlockAlloca(TheFunction,currField+"mem",str->getElementType(i));

		field = context->getStructElem(mem_currValueCasted,i);
		Builder->CreateStore(field,mem_field);
		i++;

		RawValueMemory mem_valWrapper;
		mem_valWrapper.mem = mem_field;
		mem_valWrapper.isNull = context->createFalse();
		(*allBucketBindings)[*(*it)] = mem_valWrapper;
		LOG(INFO) << "[HT Bucket Traversal: ] Binding name: "<<currField;
	}
	OperatorState newState = OperatorState(*this, *allBucketBindings);

	itAcc = accs.begin();
	itExpr = outputExprs.begin();
	vector<AllocaInst*>::const_iterator itMem = mem_accumulators.begin();
	vector<string>::const_iterator itLabels = aggregateLabels.begin();
	/* Accumulate FOREACH outputExpr */
	for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++, itLabels++) {
		Monoid acc = *itAcc;
		expressions::Expression *outputExpr = *itExpr;
		AllocaInst *mem_accumulating = *itMem;
		string aggregateName = *itLabels;

		switch (acc) {
		case SUM:
			generateSum(outputExpr, context, newState, mem_accumulating);
			break;
		case MULTIPLY:
			generateMul(outputExpr, context, newState, mem_accumulating);
			break;
		case MAX:
			generateMax(outputExpr, context, newState, mem_accumulating);
			break;
		case OR:
			generateOr(outputExpr, context, newState, mem_accumulating);
			break;
		case AND:
			generateAnd(outputExpr, context, newState, mem_accumulating);
			break;
		case UNION:
			//		generateUnion(context, childState);
			//		break;
		case BAGUNION:
			//		generateBagUnion(context, childState);
			//		break;
		case APPEND:
			//		generateAppend(context, childState);
			//		break;
		default: {
			string error_msg = string(
					"[Nest: ] Unknown / Still Unsupported accumulator");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}

		Plugin *htPlugin = new BinaryInternalPlugin(context, htName);
		RecordAttribute attr_aggr = RecordAttribute(htName, aggregateName,
				outputExpr->getExpressionType());
		catalog.registerPlugin(htName, htPlugin);
		//cout << "Registering custom pg for " << htName << endl;
		RawValueMemory mem_aggrWrapper;
		mem_aggrWrapper.mem = mem_accumulating;
		mem_aggrWrapper.isNull = context->createFalse();
		(*allBucketBindings)[attr_aggr] = mem_aggrWrapper;
	}

	/**
	 * [INC - HT CHAIN NO.] Increase value counter in (specific) bucket
	 * Continue inner loop
	 */
	Builder->CreateBr(loopIncBucket);
	Builder->SetInsertPoint(loopIncBucket);

	valuesCounter = Builder->CreateLoad(mem_valuesCounter);
	Value* inc_valuesCounter = Builder->CreateAdd(valuesCounter,
			context->createInt64(1));
	Builder->CreateStore(inc_valuesCounter, mem_valuesCounter);
	Builder->CreateBr(loopCondBucket);

	/**
	* [END - HT CHAIN LOOP.] End inner loop
	* 4. Time to produce output tuple & forward to next operator
	*/
	Builder->SetInsertPoint(loopEndBucket);

	/* Explicit oid (i.e., bucketNo) materialization */
	RawValueMemory mem_oidWrapper;
	mem_oidWrapper.mem = mem_bucketCounter;
	mem_oidWrapper.isNull = context->createFalse();
	ExpressionType *oidType = new IntType();
	RecordAttribute attr_oid = RecordAttribute(htName,activeLoop,oidType);
	(*allBucketBindings)[attr_oid] = mem_oidWrapper;
//#ifdef DEBUG
//		ArgsV.clear();
//		Function* debugInt64 = context->getFunction("printi");
//		Value* finalResult = Builder->CreateLoad(mem_accumulating);
//		ArgsV.push_back(finalResult);
//		Builder->CreateCall(debugInt64, ArgsV);
//		ArgsV.clear();
//		ArgsV.push_back(context->createInt32(-7));
//		Builder->CreateCall(debugInt64, ArgsV);
//		ArgsV.clear();
//#endif

	OperatorState *groupState = new OperatorState(*this, *allBucketBindings);
	getParent()->consume(context, *groupState);


	/**
	 * [INC - HT BUCKET NO.] Continue outer loop
	 */
	bucketCounter = Builder->CreateLoad(mem_bucketCounter);
	Value *val_inc = Builder->getInt64(1);
	Value* val_new = Builder->CreateAdd(bucketCounter, val_inc);
	Builder->CreateStore(val_new, mem_bucketCounter);
	Builder->CreateBr(loopIncHT);


	Builder->SetInsertPoint(loopIncHT);
	Builder->CreateBr(loopCondHT);

	//Ending block of buckets loop
	Builder->SetInsertPoint(loopEndHT);
}

void Nest::generateSum(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "nestCondEnd", TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond",
					&ifBlock, endBlock);

	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
#ifdef DEBUGNEST
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		ArgsV.push_back(val_accumulating);
//		Builder->CreateCall(debugInt, ArgsV);
#endif
		Value* val_new = Builder->CreateAdd(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		Builder->CreateBr(endBlock);
#ifdef DEBUGNEST
//		Builder->SetInsertPoint(endBlock);
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		Value* finalResult = Builder->CreateLoad(mem_accumulating);
//		ArgsV.push_back(finalResult);
//		Builder->CreateCall(debugInt, ArgsV);
//		//Back to 'normal' flow
//		Builder->SetInsertPoint(ifBlock);
#endif
		break;
	}
	case FLOAT: {
		Value* val_new = Builder->CreateFAdd(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		Builder->CreateBr(endBlock);
		break;
	}
	default: {
		string error_msg = string(
				"[Nest: ] Sum accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

void Nest::generateMul(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "nestCondEnd", TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond",
					&ifBlock, endBlock);

	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
#ifdef DEBUGNEST
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		ArgsV.push_back(val_accumulating);
//		Builder->CreateCall(debugInt, ArgsV);
#endif
		Value* val_new = Builder->CreateMul(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		Builder->CreateBr(endBlock);
#ifdef DEBUGNEST
//		Builder->SetInsertPoint(endBlock);
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		Value* finalResult = Builder->CreateLoad(mem_accumulating);
//		ArgsV.push_back(finalResult);
//		Builder->CreateCall(debugInt, ArgsV);
//		//Back to 'normal' flow
//		Builder->SetInsertPoint(ifBlock);
#endif
		break;
	}
	case FLOAT: {
		Value* val_new = Builder->CreateFMul(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		Builder->CreateBr(endBlock);
		break;
	}
	default: {
		string error_msg = string(
				"[Nest: ] Sum accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

void Nest::generateMax(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "nestCondEnd",
			TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond", &ifBlock,
			endBlock);

	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
		/**
		 * if(curr > max) max = curr;
		 */
		BasicBlock* ifGtMaxBlock;
		context->CreateIfBlock(context->getGlobalFunction(), "nestMaxCond",
				&ifGtMaxBlock, endBlock);
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);
		Value* maxCondition = Builder->CreateICmpSGT(val_output.value,
				val_accumulating);
		Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

		Builder->SetInsertPoint(ifGtMaxBlock);
		Builder->CreateStore(val_output.value, mem_accumulating);
		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGNEST
		vector<Value*> ArgsV;
		Function* debugInt = context->getFunction("printi");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugInt, ArgsV);
#endif
		//Back to 'normal' flow
		Builder->SetInsertPoint(ifGtMaxBlock);

		//Branch Instruction to reach endBlock will be flushed after end of switch
		break;
	}
	case FLOAT: {
		/**
		 * if(curr > max) max = curr;
		 */
		BasicBlock* ifGtMaxBlock;
		context->CreateIfBlock(context->getGlobalFunction(), "nestMaxCond",
				&ifGtMaxBlock, endBlock);
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);
		Value* maxCondition = Builder->CreateFCmpOGT(val_output.value,
				val_accumulating);
		Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

		Builder->SetInsertPoint(ifGtMaxBlock);
		Builder->CreateStore(val_output.value, mem_accumulating);
		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGNEST
		vector<Value*> ArgsV;
		Function* debugFloat = context->getFunction("printFloat");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugFloat, ArgsV);
#endif
		//Back to 'normal' flow
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Sum accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

void Nest::generateOr(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "nestCondEnd", TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond",
					&ifBlock, endBlock);

	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case BOOL: {
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

		RawValue val_output = outputExpr->accept(outputExprGenerator);
		Value* val_new = Builder->CreateOr(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);

		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		std::vector<Value*> ArgsV;
		Function* debugBoolean = context->getFunction("printBoolean");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugBoolean, ArgsV);
#endif
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Or accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);

}

void Nest::generateAnd(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "nestCondEnd",
			TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "nestIfCond", &ifBlock,
			endBlock);

	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case BOOL: {
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

		RawValue val_output = outputExpr->accept(outputExprGenerator);
		Value* val_new = Builder->CreateAnd(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);

		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		std::vector<Value*> ArgsV;
		Function* debugBoolean = context->getFunction("printBoolean");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugBoolean, ArgsV);
#endif
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Or accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

AllocaInst* Nest::resetAccumulator(expressions::Expression* outputExpr, Monoid acc) const
{
	AllocaInst* mem_accumulating = NULL;

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *f = Builder->GetInsertBlock()->getParent();

	Type* int1Type = Type::getInt1Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* doubleType = Type::getDoubleTy(llvmContext);

	//Deal with 'memory allocations' as per monoid type requested
	typeID outputType = outputExpr->getExpressionType()->getTypeID();
	switch (acc)
	{
	case SUM:
	{
		switch (outputType)
		{
		case INT:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			Value *val_zero = Builder->getInt32(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT:
		{
			Type* doubleType = Type::getDoubleTy(llvmContext);
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			Value *val_zero = ConstantFP::get(doubleType, 0.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default:
		{
			string error_msg = string(
					"[Nest: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case MULTIPLY:
	{
		switch (outputType)
		{
		case INT:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			Value *val_zero = Builder->getInt32(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			Value *val_zero = ConstantFP::get(doubleType, 1.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default:
		{
			string error_msg = string(
					"[Nest: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case MAX:
	{
		switch (outputType)
		{
		case INT:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			/**
			 * FIXME This is not the appropriate 'zero' value for integers.
			 * It is the one for naturals
			 */
			Value *val_zero = Builder->getInt32(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			/**
			 * FIXME This is not the appropriate 'zero' value for floats.
			 * It is the one for naturals
			 */
			Value *val_zero = ConstantFP::get(doubleType, 0.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default:
		{
			string error_msg = string(
					"[Nest: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case OR:
	{
		switch (outputType)
		{
		case BOOL:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int1Type);
			Value *val_zero = Builder->getInt1(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default:
		{
			string error_msg = string("[Nest: ] Or/And operate on booleans");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

		}
		break;
	}
	case AND:
	{
		switch (outputType)
		{
		case BOOL:
		{
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int1Type);
			Value *val_zero = Builder->getInt1(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default:
		{
			string error_msg = string("[Nest: ] Or/And operate on booleans");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

		}
		break;
	}
	case UNION:
	case BAGUNION:
	{
		string error_msg = string("[Nest: ] Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case APPEND:
	{
		//XXX Reduce has some more stuff on this
		string error_msg = string("[Nest: ] Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	default:
	{
		string error_msg = string("[Nest: ] Unknown accumulator");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
	return mem_accumulating;
}

}













