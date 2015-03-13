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

#include "operators/radix-join.hpp"

RadixJoin::RadixJoin(expressions::BinaryExpression* predicate,
		const RawOperator& leftChild, const RawOperator& rightChild,
		RawContext* const context, char* opLabel, Materializer& matLeft,
		Materializer& matRight) :
		BinaryRawOperator(leftChild, rightChild), pred(predicate),
		context(context), matLeft(matLeft), matRight(matRight), htLabel(opLabel) {

	Function *F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<> *Builder = context->getBuilder();

	Type* int64_type = Type::getInt64Ty(llvmContext);
	Type* int32_type = Type::getInt32Ty(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *int32_ptr_type = PointerType::get(int32_type, 0);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);
	keyType = int32_type;

	Value *zero = context->createInt64(0);

	/* Request memory for HT(s) construction 		*/
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
	htClusterType = StructType::get(context->getLLVMContext(),htRadixClusterMembers);
	PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);

	/* Arbitrary initial buffer sizes */
	size_t sizeR = 30; //1000;
	size_t sizeS = 30; //1000;
	Value *val_sizeR = context->createInt64(sizeR);
	Value *val_sizeS = context->createInt64(sizeS);

	/* Request memory to store relation R 			*/
	relR.mem_relation =
			context->CreateEntryBlockAlloca(F,string("relationR"),char_ptr_type);
	relR.mem_tuplesNo =
			context->CreateEntryBlockAlloca(F,string("tuplesR"),int64_type);
	relR.mem_size =
			context->CreateEntryBlockAlloca(F,string("sizeR"),int64_type);
	relR.mem_offset =
			context->CreateEntryBlockAlloca(F,string("offsetRelR"),int64_type);
	relationR = (char*) getMemoryChunk(sizeR);
	Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type, relationR);
	Builder->CreateStore(val_relationR,relR.mem_relation);
	Builder->CreateStore(zero,relR.mem_tuplesNo);
	Builder->CreateStore(zero,relR.mem_offset);
	Builder->CreateStore(val_sizeR,relR.mem_size);

	/* Request memory to store relation S 			*/
	relS.mem_relation =
			context->CreateEntryBlockAlloca(F,string("relationS"),char_ptr_type);
	relS.mem_tuplesNo =
			context->CreateEntryBlockAlloca(F,string("tuplesS"),int64_type);
	relS.mem_size =
			context->CreateEntryBlockAlloca(F,string("sizeS"),int64_type);
	relS.mem_offset =
			context->CreateEntryBlockAlloca(F,string("offsetRelS"),int64_type);
	relationS = (char*) getMemoryChunk(sizeS);
	Value *val_relationS = context->CastPtrToLlvmPtr(char_ptr_type, relationS);
	Builder->CreateStore(val_relationR,relS.mem_relation);
	Builder->CreateStore(zero,relS.mem_tuplesNo);
	Builder->CreateStore(zero,relS.mem_offset);
	Builder->CreateStore(val_sizeS,relS.mem_size);

	/* What the type of HT entries is */
	/* (int32, void*) */
	vector<Type*> htEntryMembers;
	htEntryMembers.push_back(int32_type);
	htEntryMembers.push_back(void_ptr_type);
	htEntryType = StructType::get(context->getLLVMContext(),htEntryMembers);
	PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

	/* Request memory to store HT entries of R */
	htR.mem_kv =
				context->CreateEntryBlockAlloca(F,string("htR"),htEntryPtrType);
	htR.mem_tuplesNo =
				context->CreateEntryBlockAlloca(F,string("tuplesR"),int64_type);
	htR.mem_size =
				context->CreateEntryBlockAlloca(F,string("sizeR"),int64_type);
	htR.mem_offset =
				context->CreateEntryBlockAlloca(F,string("offsetRelR"),int64_type);
	kvR = (char*) getMemoryChunk(sizeR);
	Value *val_kvR = context->CastPtrToLlvmPtr(htEntryPtrType, kvR);
	Builder->CreateStore(val_kvR, htR.mem_kv);
	Builder->CreateStore(zero, htR.mem_tuplesNo);
	Builder->CreateStore(context->createInt64(sizeR), htR.mem_offset);
	Builder->CreateStore(zero, htR.mem_offset);
	Builder->CreateStore(val_sizeR,htR.mem_size);

	/* Request memory to store HT entries of S */
	htS.mem_kv = context->CreateEntryBlockAlloca(F, string("htS"),
			htEntryPtrType);
	htS.mem_tuplesNo = context->CreateEntryBlockAlloca(F, string("tuplesS"),
			int64_type);
	htS.mem_size = context->CreateEntryBlockAlloca(F, string("sizeS"),
			int64_type);
	htS.mem_offset = context->CreateEntryBlockAlloca(F, string("offsetRelS"),
			int64_type);
	kvS = (char*) getMemoryChunk(sizeS);
	Value *val_kvS = context->CastPtrToLlvmPtr(htEntryPtrType, kvS);
	Builder->CreateStore(val_kvS, htS.mem_kv);
	Builder->CreateStore(zero, htS.mem_tuplesNo);
	Builder->CreateStore(context->createInt64(sizeR), htS.mem_offset);
	Builder->CreateStore(zero, htS.mem_offset);
	Builder->CreateStore(val_sizeS,htS.mem_size);
}

RadixJoin::~RadixJoin()	{
	LOG(INFO)<<"Collapsing RadixJoin operator";
//	Can't do garbage collection here, need to do it from codegen
}

void RadixJoin::freeArenas() const	{
	/* Prepare codegen utils */
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();
	Function *debugInt = context->getFunction("printi");
	Function *debugInt64 = context->getFunction("printi64");

	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	Type *int64_type = Type::getInt64Ty(llvmContext);
	Type *int32_type = Type::getInt32Ty(llvmContext);

	/* Actual Work */
	Value *val_arena = Builder->CreateLoad(relR.mem_relation);
	vector<Value*> ArgsFree;
	Function* freeLLVM = context->getFunction("releaseMemoryChunk");
	AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
	Builder->CreateStore(val_arena,mem_arena_void);
	Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
	ArgsFree.push_back(val_arena_void);
	Builder->CreateCall(freeLLVM, ArgsFree);
}

void RadixJoin::produce() const {
	getLeftChild().produce();
	getRightChild().produce();

	runRadix();

	/* Free Arenas */
	this->freeArenas();
}

void RadixJoin::runRadix() const	{

	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();

	Type *int32_type = Type::getInt32Ty(llvmContext);
	Value *val_zero = context->createInt32(0);
	Value *val_one = context->createInt32(1);

	/* Partition and Cluster 'R' (the corresponding htEntries) */
	Value *clusterCountR = radix_cluster_nopadding(relR,htR);
	/* Partition and Cluster 'S' (the corresponding htEntries) */
	Value *clusterCountS = radix_cluster_nopadding(relS,htS);


	AllocaInst *mem_rCount =
				Builder->CreateAlloca(int32_type,0,"rCount");
	AllocaInst *mem_sCount =
				Builder->CreateAlloca(int32_type,0,"sCount");
	AllocaInst *mem_clusterCount =
				Builder->CreateAlloca(int32_type,0,"clusterCount");
	Builder->CreateStore(val_zero,mem_rCount);
	Builder->CreateStore(val_zero,mem_sCount);
	Builder->CreateStore(val_zero,mem_clusterCount);

	uint32_t clusterNo = (1<<NUM_RADIX_BITS);
	Value *val_clusterNo = context->createInt32(clusterNo);

	Builder->CreateAlloca(htClusterType, 0, "HTimpl");
	PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
	Value *val_htPerCluster =
			context->CastPtrToLlvmPtr(htClusterPtrType, HT_per_cluster);

	/**
	 * ACTUAL PROBES
	 */

	/* Loop through clusters */
	/* for (i = 0; i < (1 << NUM_RADIX_BITS); i++) */

	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	context->CreateForLoop("clusterLoopCond", "clusterLoopBody",
			"clusterLoopInc", "clusterLoopEnd", &loopCond, &loopBody, &loopInc,
			&loopEnd);

	/* 1. Loop Condition - Unsigned integers operation */
	Builder->CreateBr(loopCond);
	Builder->SetInsertPoint(loopCond);
	Value *val_clusterCount = Builder->CreateLoad(mem_clusterCount);
	Value *val_cond = Builder->CreateICmpULT(val_clusterCount,val_clusterNo);
	Builder->CreateCondBr(val_cond, loopBody, loopEnd);

	/* 2. Loop Body */
	Builder->SetInsertPoint(loopBody);

	/* Check cluster contents */
	/* if (R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0)
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifNotEmptyCluster", "elseEmptyCluster",
										&ifBlock, &elseBlock,loopInc);

	{
		/* If Condition */
		Value *val_r_i_count =
				context->getArrayElem(clusterCountR,val_clusterCount);
		Value *val_s_i_count =
				context->getArrayElem(clusterCountS,val_clusterCount);
		Value *val_cond_1 = Builder->CreateICmpSGT(val_r_i_count,val_zero);
		Value *val_cond_2 = Builder->CreateICmpSGT(val_s_i_count,val_zero);
		val_cond = Builder->CreateAnd(val_cond_1,val_cond_2);
		Builder->CreateCondBr(val_cond,ifBlock,elseBlock);

		/* If clusters non-empty */
		Builder->SetInsertPoint(ifBlock);

		/* tmpR.tuples = relR->tuples + r; */
		Value *val_htR = Builder->CreateLoad(htR.mem_kv);
		Value* htRshifted = Builder->CreateInBoundsGEP(val_htR, val_r_i_count);

		/* tmpS.tuples = relS->tuples + s; */
		Value *val_htS = Builder->CreateLoad(htS.mem_kv);
		Value* htSshifted = Builder->CreateInBoundsGEP(val_htS, val_s_i_count);

		/* r += R_count_per_cluster[i];
		 * s += S_count_per_cluster[i];
		 */
		Value *val_rCount = Builder->CreateLoad(mem_rCount);
		Value *val_sCount = Builder->CreateLoad(mem_sCount);
		val_rCount = Builder->CreateAdd(val_rCount,val_r_i_count);
		val_sCount = Builder->CreateAdd(val_sCount,val_s_i_count);
		Builder->CreateStore(val_rCount,mem_rCount);
		Builder->CreateStore(val_sCount,mem_sCount);

		/* bucket_chaining_join_prepare(&tmpR, &(HT_per_cluster[i])); */
		/* TODO !!*/
//		Function *partitionHT = context->getFunction("partitionHT");
//			vector<Value*> ArgsPartition;
//			Value *val_tuplesNo = Builder->CreateLoad(rel.mem_tuplesNo);
//			Value *val_ht 		= Builder->CreateLoad(ht.mem_kv);
//			ArgsPartition.push_back(val_tuplesNo);
//			ArgsPartition.push_back(val_ht);


		Builder->CreateBr(loopInc);

		/* If (either) cluster is empty */
		/*
		 * r += R_count_per_cluster[i];
		 * s += S_count_per_cluster[i];
		 */
		Builder->SetInsertPoint(elseBlock);
		val_rCount = Builder->CreateLoad(mem_rCount);
		val_sCount = Builder->CreateLoad(mem_sCount);
		val_rCount = Builder->CreateAdd(val_rCount,val_r_i_count);
		val_sCount = Builder->CreateAdd(val_sCount,val_s_i_count);
		Builder->CreateStore(val_rCount,mem_rCount);
		Builder->CreateStore(val_sCount,mem_sCount);
		Builder->CreateBr(loopInc);
	}

	/* 3. Loop Inc. */
	Builder->SetInsertPoint(loopInc);
	val_clusterCount = Builder->CreateLoad(mem_clusterCount);
	val_clusterCount = Builder->CreateAdd(val_clusterCount,val_one);
	Builder->CreateStore(val_clusterCount, mem_clusterCount);

	Builder->CreateBr(loopCond);

	/* 4. Loop End */
	Builder->SetInsertPoint(loopEnd);

}

/**
 * @param rel the materialized input relation
 * @param ht  the htEntries corresp. to the relation
 *
 * @return item count per resulting cluster
 */
Value *RadixJoin::radix_cluster_nopadding(struct relationBuf rel, struct kvBuf ht) const	{

	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();

	Function *partitionHT = context->getFunction("partitionHT");
	vector<Value*> ArgsPartition;
	Value *val_tuplesNo = Builder->CreateLoad(rel.mem_tuplesNo);
	Value *val_ht 		= Builder->CreateLoad(ht.mem_kv);
	ArgsPartition.push_back(val_tuplesNo);
	ArgsPartition.push_back(val_ht);

	return  Builder->CreateCall(partitionHT, ArgsPartition);

}
/**
 * XXX Make sure that offset initialization happens w/o issue.
 * It 'should' take place in first 'init' block of execution
 */
void RadixJoin::consume(RawContext* const context, const OperatorState& childState) {

	/* Prepare codegen utils */
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();
	Function *debugInt = context->getFunction("printi");
	Function *debugInt64 = context->getFunction("printi64");

	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	Type *int64Type = Type::getInt64Ty(llvmContext);
	Type *int32Type = Type::getInt32Ty(llvmContext);

	int keySize = keyType->getPrimitiveSizeInBits() / 8;
	Value* val_keySize = context->createInt32(keySize);
	Value* val_keySize64 = context->createInt64(keySize);

	/* What the 2nd void* points to: TBD */
	StructType *payloadType;

	const RawOperator& caller = childState.getProducer();
	if(caller == getLeftChild())
	{

#ifdef DEBUG
		LOG(INFO)<< "[RADIX JOIN: ] Left (building) side";
#endif
		const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();
		OutputPlugin* pg = new OutputPlugin(context, matLeft, bindings);

		/* Result type specified during output plugin construction */
		payloadType = pg->getPayloadType();
		Value *val_payloadSize;

		if(pg->hasComplexTypes())	{
			val_payloadSize = pg->getRuntimePayloadTypeSize();
		}	else	{
			val_payloadSize = context->createInt64(pg->getPayloadTypeSize());
//			cout << "Left Payload: " << pg->getPayloadTypeSize()  << endl;
		}

		/* Registering payload type of HT in RAW CATALOG */
		/**
		 * Must either fix (..for fully-blocking joins)
		 * or remove entirely
		 */
		//catalog.insertTableInfo(string(this->htLabel),payloadType);

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
		Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,val_payloadSize);
		Value *arenaSize = Builder->CreateLoad(relR.mem_size);

		/* if(offsetInArena + payloadSize >= arenaSize) */
		BasicBlock* entryBlock = Builder->GetInsertBlock();
		BasicBlock *endBlockArenaFull = BasicBlock::Create(llvmContext, "IfArenaFullEnd", F);
		BasicBlock *ifArenaFull;
		context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull, endBlockArenaFull);
		Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload, arenaSize);

		Builder->CreateCondBr(offsetCond,ifArenaFull,endBlockArenaFull);

		/* true => realloc() */
		Builder->SetInsertPoint(ifArenaFull);

		vector<Value*> ArgsRealloc;
		Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
		AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
		Builder->CreateStore(val_arena,mem_arena_void);
		Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
		ArgsRealloc.push_back(val_arena_void);
		ArgsRealloc.push_back(arenaSize);
		Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

		Builder->CreateStore(val_newArenaVoidPtr,relR.mem_relation);
		Value* val_size = Builder->CreateLoad(relR.mem_size);
		val_size = Builder->CreateMul(val_size,context->createInt64(2));
		Builder->CreateStore(val_size,relR.mem_size);
		Builder->CreateBr(endBlockArenaFull);

		/* 'Normal' flow again */
		Builder->SetInsertPoint(endBlockArenaFull);

		/* Repeat load - realloc() might have occurred */
		val_arena = Builder->CreateLoad(relR.mem_relation);

		/* STORING PAYLOAD */
		/* 1. arena += (offset) */
		Value *ptr_arenaShifted = Builder->CreateInBoundsGEP(val_arena,offsetInArena);

		/* 2. Casting */
		PointerType *ptr_payloadType = PointerType::get(payloadType,0);
		Value *cast_arenaShifted = Builder->CreateBitCast(ptr_arenaShifted,ptr_payloadType);
		AllocaInst *mem_payloadCastPtr = Builder->CreateAlloca(ptr_payloadType,0,"payloadPlaceholder");
		Builder->CreateStore(cast_arenaShifted,mem_payloadCastPtr);

		/* 3. Storing payload, one field at a time */
		vector<Type*>* materializedTypes = pg->getMaterializedTypes();
		//Storing all activeTuples met so far
		int offsetInStruct = 0; //offset inside the struct (+current field manipulated)
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
					idxList.push_back(context->createInt32(offsetInStruct));
					//Shift in struct ptr
					Value* mem_payloadCast = Builder->CreateLoad(mem_payloadCastPtr);
					Value* structPtr = Builder->CreateGEP(mem_payloadCast, idxList);
					Builder->CreateStore(val_activeTuple,structPtr);

					offsetInStruct++;
				}
			}
		}

		int offsetInWanted = 0;
		const vector<RecordAttribute*>& wantedFields = matLeft.getWantedFields();
		for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it != wantedFields.end(); ++it)
		{
			map<RecordAttribute, RawValueMemory>::const_iterator memSearch = bindings.find(*(*it));
			RawValueMemory currValMem = memSearch->second;
			//FIXME FIX THE NECESSARY CONVERSIONS HERE
			Value* currVal = Builder->CreateLoad(currValMem.mem);
			Value* valToMaterialize = pg->convert(currVal->getType(),materializedTypes->at(offsetInWanted),currVal);
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			//Shift in struct ptr
			Value* mem_payloadCast = Builder->CreateLoad(mem_payloadCastPtr);
			Value* structPtr = Builder->CreateGEP(mem_payloadCast, idxList);

			Builder->CreateStore(valToMaterialize,structPtr);
			offsetInStruct++;
			offsetInWanted++;
		}

		/* 4. Increment counts */
		Builder->CreateStore(offsetPlusPayload,relR.mem_offset);
		Value* val_tuplesNo = Builder->CreateLoad(relR.mem_tuplesNo);
		val_tuplesNo = Builder->CreateAdd(val_tuplesNo,context->createInt64(1));
		Builder->CreateStore(val_tuplesNo,relR.mem_tuplesNo);

		/* CONSTRUCT HTENTRY PAIR   	  */
		/* (int32 key, void* payloadPtr)  */
		/* Prepare key */
		expressions::Expression* leftKeyExpr = this->pred->getLeftOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
		RawValue leftKey = leftKeyExpr->accept(exprGenerator);
		Type* keyType = (leftKey.value)->getType();
		//10: IntegerTyID
		if(keyType->getTypeID() != 10) {
			string error_msg = "Only INT32 keys considered atm";
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}

		PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

		BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext, "IfHTFullEnd", F);
		BasicBlock *ifHTFull;
		context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull, endBlockHTFull);

		Value *val_ht = Builder->CreateLoad(htR.mem_kv);
		Value *offsetInHT = Builder->CreateLoad(htR.mem_offset);
		int voidPtrSize = void_ptr_type->getPrimitiveSizeInBits() / 8;
		Value *val_voidPtrSize = context->createInt64(voidPtrSize);
		Value *offsetPlusKey = Builder->CreateAdd(offsetInArena,val_keySize64);

		Value *offsetPlusKeyPtr = Builder->CreateAdd(offsetInHT,val_voidPtrSize);
		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKeyPtr,val_voidPtrSize);
		Value *htSize = Builder->CreateLoad(htR.mem_size);
		offsetCond = Builder->CreateICmpSGE(offsetPlusPayloadPtr, htSize);

		Builder->CreateCondBr(offsetCond,ifHTFull,endBlockHTFull);

		/* true => realloc() */
		Builder->SetInsertPoint(ifHTFull);

		AllocaInst* mem_ht_void = Builder->CreateAlloca(void_ptr_type,0,"voidHTPtr");
		/* Casting htEntry* to void* requires a cast */
		Value *cast_htEntries =
						Builder->CreateBitCast(val_ht,void_ptr_type);
		Builder->CreateStore(cast_htEntries,mem_ht_void);
		Value *val_ht_void = Builder->CreateLoad(mem_ht_void);
		ArgsRealloc.clear();
		ArgsRealloc.push_back(val_ht_void);
		ArgsRealloc.push_back(htSize);

		Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);
		Value *val_newHTPtr =
				Builder->CreateBitCast(val_newVoidHTPtr,htEntryPtrType);
		Builder->CreateStore(val_newHTPtr,htR.mem_kv);
		val_size = Builder->CreateLoad(htR.mem_size);
		val_size = Builder->CreateMul(val_size,context->createInt64(2));
		Builder->CreateStore(val_size,htR.mem_size);
		Builder->CreateBr(endBlockHTFull);

		/* Insert ht entry in HT */
		Builder->SetInsertPoint(endBlockHTFull);

		/* Repeat load - realloc() might have occurred */
		val_ht = Builder->CreateLoad(htR.mem_kv);

		/* 1. kv += offset */
		Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,offsetInHT);

		/* 2. htEntry* kv_cast = (htEntry)* kv */
		Value *cast_kvShifted =
				Builder->CreateBitCast(ptr_kvShifted,htEntryPtrType);
		AllocaInst *mem_htEntryCastPtr = Builder->CreateAlloca(htEntryPtrType,0,"htEntryPlaceholder");
		Builder->CreateStore(cast_kvShifted,mem_htEntryCastPtr);

		/* 3a. kv_cast->keyPtr = &key */

		offsetInStruct = 0;
		//Shift in htEntry (struct) ptr
		vector<Value*> idxList = vector<Value*>();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));

		Value* mem_htEntryCast = Builder->CreateLoad(mem_htEntryCastPtr);
		Value* structPtr = Builder->CreateGEP(mem_htEntryCast, idxList);
		Builder->CreateStore(leftKey.value,structPtr);

		/* 3b. kv_cast->payloadPtr = &payload */
		offsetInStruct = 1;
		idxList.clear();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));
		structPtr = Builder->CreateGEP(mem_htEntryCast, idxList);

		Value *cast_payload = Builder->CreateBitCast(mem_payloadCastPtr,void_ptr_type);
		Builder->CreateStore(cast_payload,structPtr);

		/* 4. Increment counts */
		Builder->CreateStore(offsetPlusPayloadPtr,htR.mem_offset);
		Builder->CreateStore(val_tuplesNo,htR.mem_tuplesNo);
	}
	else
	{

#ifdef DEBUG
		LOG(INFO)<< "[RADIX JOIN: ] Right (also building!) side";
#endif
		const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();
		OutputPlugin* pg = new OutputPlugin(context, matRight, bindings);

		/* Result type specified during output plugin construction */
		payloadType = pg->getPayloadType();
		Value *val_payloadSize;

		if(pg->hasComplexTypes()) {
			val_payloadSize = pg->getRuntimePayloadTypeSize();
		} else {
			val_payloadSize = context->createInt64(pg->getPayloadTypeSize());
			//			cout << "Right Payload: " << pg->getPayloadTypeSize()  << endl;
		}

		/* Registering payload type of HT in RAW CATALOG */
		/**
		 * Must either fix (..for fully-blocking joins)
		 * or remove entirely
		 */
		//catalog.insertTableInfo(string(this->htLabel),payloadType);
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
		Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,val_payloadSize);
		Value *arenaSize = Builder->CreateLoad(relR.mem_size);

		/* if(offsetInArena + payloadSize >= arenaSize) */
		BasicBlock* entryBlock = Builder->GetInsertBlock();
		BasicBlock *endBlockArenaFull = BasicBlock::Create(llvmContext, "IfArenaFullEnd", F);
		BasicBlock *ifArenaFull;
		context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull, endBlockArenaFull);
		Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload, arenaSize);

		Builder->CreateCondBr(offsetCond,ifArenaFull,endBlockArenaFull);

		/* true => realloc() */
		Builder->SetInsertPoint(ifArenaFull);

		vector<Value*> ArgsRealloc;
		Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
		AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
		Builder->CreateStore(val_arena,mem_arena_void);
		Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
		ArgsRealloc.push_back(val_arena_void);
		ArgsRealloc.push_back(arenaSize);
		Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

		Builder->CreateStore(val_newArenaVoidPtr,relR.mem_relation);
		Value* val_size = Builder->CreateLoad(relR.mem_size);
		val_size = Builder->CreateMul(val_size,context->createInt64(2));
		Builder->CreateStore(val_size,relR.mem_size);
		Builder->CreateBr(endBlockArenaFull);

		/* 'Normal' flow again */
		Builder->SetInsertPoint(endBlockArenaFull);

		/* Repeat load - realloc() might have occurred */
		val_arena = Builder->CreateLoad(relR.mem_relation);

		/* STORING PAYLOAD */
		/* 1. arena += (offset) */
		Value *ptr_arenaShifted = Builder->CreateInBoundsGEP(val_arena,offsetInArena);

		/* 2. Casting */
		PointerType *ptr_payloadType = PointerType::get(payloadType,0);
		Value *cast_arenaShifted = Builder->CreateBitCast(ptr_arenaShifted,ptr_payloadType);
		AllocaInst *mem_payloadCastPtr = Builder->CreateAlloca(ptr_payloadType,0,"payloadPlaceholder");
		Builder->CreateStore(cast_arenaShifted,mem_payloadCastPtr);

		/* 3. Storing payload, one field at a time */
		vector<Type*>* materializedTypes = pg->getMaterializedTypes();
		//Storing all activeTuples met so far
		int offsetInStruct = 0;//offset inside the struct (+current field manipulated)
		RawValueMemory mem_activeTuple;
		{
			map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
			for(memSearch = bindings.begin(); memSearch != bindings.end(); memSearch++) {
				RecordAttribute currAttr = memSearch->first;
				if(currAttr.getAttrName() == activeLoop) {
					mem_activeTuple = memSearch->second;
					Value* val_activeTuple = Builder->CreateLoad(mem_activeTuple.mem);
					//OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR EXTRA OFFSET
					vector<Value*> idxList = vector<Value*>();
					idxList.push_back(context->createInt32(0));
					idxList.push_back(context->createInt32(offsetInStruct));
					//Shift in struct ptr
					Value* mem_payloadCast = Builder->CreateLoad(mem_payloadCastPtr);
					Value* structPtr = Builder->CreateGEP(mem_payloadCast, idxList);
					Builder->CreateStore(val_activeTuple,structPtr);

					offsetInStruct++;
				}
			}
		}

		int offsetInWanted = 0;
		const vector<RecordAttribute*>& wantedFields = matRight.getWantedFields();
		for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it != wantedFields.end(); ++it)
		{
			map<RecordAttribute, RawValueMemory>::const_iterator memSearch = bindings.find(*(*it));
			RawValueMemory currValMem = memSearch->second;
			//FIXME FIX THE NECESSARY CONVERSIONS HERE
			Value* currVal = Builder->CreateLoad(currValMem.mem);
			Value* valToMaterialize = pg->convert(currVal->getType(),materializedTypes->at(offsetInWanted),currVal);
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			//Shift in struct ptr
			Value* mem_payloadCast = Builder->CreateLoad(mem_payloadCastPtr);
			Value* structPtr = Builder->CreateGEP(mem_payloadCast, idxList);

			Builder->CreateStore(valToMaterialize,structPtr);
			offsetInStruct++;
			offsetInWanted++;
		}

		/* 4. Increment counts */
		Builder->CreateStore(offsetPlusPayload,relR.mem_offset);
		Value* val_tuplesNo = Builder->CreateLoad(relR.mem_tuplesNo);
		val_tuplesNo = Builder->CreateAdd(val_tuplesNo,context->createInt64(1));
		Builder->CreateStore(val_tuplesNo,relR.mem_tuplesNo);

		/* CONSTRUCT HTENTRY PAIR   	  */
		/* (int32 key, void* payloadPtr)  */
		/* Prepare key */
		expressions::Expression* rightKeyExpr = this->pred->getRightOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
		RawValue rightKey = rightKeyExpr->accept(exprGenerator);
		Type* keyType = (rightKey.value)->getType();
		//10: IntegerTyID
		if(keyType->getTypeID() != 10) {
			string error_msg = "Only INT32 keys considered atm";
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}

		PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

		BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext, "IfHTFullEnd", F);
		BasicBlock *ifHTFull;
		context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull, endBlockHTFull);

		Value *val_ht = Builder->CreateLoad(htS.mem_kv);
		Value *offsetInHT = Builder->CreateLoad(htS.mem_offset);
		int voidPtrSize = void_ptr_type->getPrimitiveSizeInBits() / 8;
		Value *val_voidPtrSize = context->createInt64(voidPtrSize);
		Value *offsetPlusKey = Builder->CreateAdd(offsetInArena,val_keySize64);

		Value *offsetPlusKeyPtr = Builder->CreateAdd(offsetInHT,val_voidPtrSize);
		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKeyPtr,val_voidPtrSize);
		Value *htSize = Builder->CreateLoad(htS.mem_size);
		offsetCond = Builder->CreateICmpSGE(offsetPlusPayloadPtr, htSize);

		Builder->CreateCondBr(offsetCond,ifHTFull,endBlockHTFull);

		/* true => realloc() */
		Builder->SetInsertPoint(ifHTFull);

		AllocaInst* mem_ht_void = Builder->CreateAlloca(void_ptr_type,0,"voidHTPtr");
		/* Casting htEntry* to void* requires a cast */
		Value *cast_htEntries =
		Builder->CreateBitCast(val_ht,void_ptr_type);
		Builder->CreateStore(cast_htEntries,mem_ht_void);
		Value *val_ht_void = Builder->CreateLoad(mem_ht_void);
		ArgsRealloc.clear();
		ArgsRealloc.push_back(val_ht_void);
		ArgsRealloc.push_back(htSize);

		Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);
		Value *val_newHTPtr =
		Builder->CreateBitCast(val_newVoidHTPtr,htEntryPtrType);
		Builder->CreateStore(val_newHTPtr,htS.mem_kv);
		val_size = Builder->CreateLoad(htS.mem_size);
		val_size = Builder->CreateMul(val_size,context->createInt64(2));
		Builder->CreateStore(val_size,htS.mem_size);
		Builder->CreateBr(endBlockHTFull);

		/* Insert ht entry in HT */
		Builder->SetInsertPoint(endBlockHTFull);

		/* Repeat load - realloc() might have occurred */
		val_ht = Builder->CreateLoad(htS.mem_kv);

		/* 1. kv += offset */
		Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,offsetInHT);

		/* 2. htEntry* kv_cast = (htEntry)* kv */
		Value *cast_kvShifted =
		Builder->CreateBitCast(ptr_kvShifted,htEntryPtrType);
		AllocaInst *mem_htEntryCastPtr = Builder->CreateAlloca(htEntryPtrType,0,"htEntryPlaceholder");
		Builder->CreateStore(cast_kvShifted,mem_htEntryCastPtr);

		/* 3a. kv_cast->keyPtr = &key */

		offsetInStruct = 0;
		//Shift in htEntry (struct) ptr
		vector<Value*> idxList = vector<Value*>();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));

		Value* mem_htEntryCast = Builder->CreateLoad(mem_htEntryCastPtr);
		Value* structPtr = Builder->CreateGEP(mem_htEntryCast, idxList);
		Builder->CreateStore(rightKey.value,structPtr);

		/* 3b. kv_cast->payloadPtr = &payload */
		offsetInStruct = 1;
		idxList.clear();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(offsetInStruct));
		structPtr = Builder->CreateGEP(mem_htEntryCast, idxList);

		Value *cast_payload = Builder->CreateBitCast(mem_payloadCastPtr,void_ptr_type);
		Builder->CreateStore(cast_payload,structPtr);

		/* 4. Increment counts */
		Builder->CreateStore(offsetPlusPayloadPtr,htS.mem_offset);
		Builder->CreateStore(val_tuplesNo,htS.mem_tuplesNo);
	}
};



