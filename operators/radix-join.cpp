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

/**
 * This version operates over int keys.
 * Alternate versions would also hash the key (..as radix-nest does)
 */
RadixJoin::RadixJoin(expressions::BinaryExpression* predicate,
		RawOperator *leftChild, RawOperator *rightChild,
		RawContext* const context, const char* opLabel, Materializer& matLeft,
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
	/* No realloc will be required with these sizes for synthetic large-scale numbers */
//	size_t sizeR = 10000000000;
//	size_t sizeS = 15000000000;

//	size_t sizeR = 1000;
//	size_t sizeS = 1500;
	size_t sizeR = 50000;
	size_t sizeS = 50000;

	//size_t sizeR = 100000000;
	//size_t sizeS = 100000000;

//	Still segfaults with these
//	size_t sizeR = 40000000000; //000;//0; //30; //1000;
//	size_t sizeS = 40000000000;//0; //1000;
	Value *val_sizeR = context->createInt64(sizeR);
	Value *val_sizeS = context->createInt64(sizeS);

	/* Request memory to store relation R 			*/
	relR.mem_relation =
			context->CreateEntryBlockAlloca(F,string("relationR"),char_ptr_type);
	(relR.mem_relation)->setAlignment(8);
	relR.mem_tuplesNo =
			context->CreateEntryBlockAlloca(F,string("tuplesR"),int64_type);
	relR.mem_cachedTuplesNo =
					context->CreateEntryBlockAlloca(F,string("tuplesRcached"),int64_type);
	relR.mem_size =
			context->CreateEntryBlockAlloca(F,string("sizeR"),int64_type);
	relR.mem_offset =
			context->CreateEntryBlockAlloca(F,string("offsetRelR"),int64_type);
	relationR = (char*) getMemoryChunk(sizeR);
	ptr_relationR = (char**) malloc(sizeof(char*));
	*ptr_relationR = relationR;
	Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type, relationR);
	Builder->CreateStore(val_relationR,relR.mem_relation);
	Builder->CreateStore(zero,relR.mem_tuplesNo);
	Builder->CreateStore(zero,relR.mem_cachedTuplesNo);
	Builder->CreateStore(zero,relR.mem_offset);
	Builder->CreateStore(val_sizeR,relR.mem_size);

	/* Request memory to store relation S 			*/
	relS.mem_relation =
			context->CreateEntryBlockAlloca(F,string("relationS"),char_ptr_type);
	(relS.mem_relation)->setAlignment(8);
	relS.mem_tuplesNo =
			context->CreateEntryBlockAlloca(F,string("tuplesS"),int64_type);
	relS.mem_cachedTuplesNo =
				context->CreateEntryBlockAlloca(F,string("tuplesScached"),int64_type);
	relS.mem_size =
			context->CreateEntryBlockAlloca(F,string("sizeS"),int64_type);
	relS.mem_offset =
			context->CreateEntryBlockAlloca(F,string("offsetRelS"),int64_type);
	relationS = (char*) getMemoryChunk(sizeS);
	ptr_relationS = (char**) malloc(sizeof(char*));
	*ptr_relationS = relationS;
	Value *val_relationS = context->CastPtrToLlvmPtr(char_ptr_type, relationS);
	Builder->CreateStore(val_relationS,relS.mem_relation);
	Builder->CreateStore(zero,relS.mem_tuplesNo);
	Builder->CreateStore(zero,relS.mem_cachedTuplesNo);
	Builder->CreateStore(zero,relS.mem_offset);
	Builder->CreateStore(val_sizeS,relS.mem_size);

	/* What the type of HT entries is */
	/* (int32, size_t) */
	vector<Type*> htEntryMembers;
	htEntryMembers.push_back(int32_type);
	htEntryMembers.push_back(int64_type);
	int htEntrySize = sizeof(int) + sizeof(size_t);
	htEntryType = StructType::get(context->getLLVMContext(),htEntryMembers);
	PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

	/* Request memory to store HT entries of R */
	htR.mem_kv =
				context->CreateEntryBlockAlloca(F,string("htR"),htEntryPtrType);
	(htR.mem_kv)->setAlignment(8);

	htR.mem_tuplesNo =
				context->CreateEntryBlockAlloca(F,string("tuplesR"),int64_type);
	htR.mem_size =
				context->CreateEntryBlockAlloca(F,string("sizeR"),int64_type);
	htR.mem_offset =
				context->CreateEntryBlockAlloca(F,string("offsetRelR"),int64_type);
	int kvSizeR = sizeR;// * htEntrySize;
	kvR = (char*) getMemoryChunk(kvSizeR);
	Value *val_kvR = context->CastPtrToLlvmPtr(htEntryPtrType, kvR);
	StoreInst *store_htR = Builder->CreateStore(val_kvR, htR.mem_kv);
	store_htR->setAlignment(8);

	Builder->CreateStore(zero, htR.mem_tuplesNo);
	Builder->CreateStore(context->createInt64(kvSizeR),htR.mem_size);
	Builder->CreateStore(zero, htR.mem_offset);


	/* Request memory to store HT entries of S */
	htS.mem_kv = context->CreateEntryBlockAlloca(F, string("htS"),
			htEntryPtrType);
	htS.mem_tuplesNo = context->CreateEntryBlockAlloca(F, string("tuplesS"),
			int64_type);
	htS.mem_size = context->CreateEntryBlockAlloca(F, string("sizeS"),
			int64_type);
	htS.mem_offset = context->CreateEntryBlockAlloca(F, string("offsetRelS"),
			int64_type);
	int kvSizeS = sizeS;// * htEntrySize;
	kvS = (char*) getMemoryChunk(kvSizeS);
	Value *val_kvS = context->CastPtrToLlvmPtr(htEntryPtrType, kvS);
	Builder->CreateStore(val_kvS, htS.mem_kv);
	Builder->CreateStore(zero, htS.mem_tuplesNo);
	Builder->CreateStore(context->createInt64(kvSizeS),htS.mem_size);
	Builder->CreateStore(zero, htS.mem_offset);

	/* Defined in consume() */
	rPayloadType = NULL;
	sPayloadType = NULL;

	/* Caches */
	cachedLeft = cachedRight = false;
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
	Function* freeLLVM = context->getFunction("releaseMemoryChunk");

	Value *val_arena = Builder->CreateLoad(relR.mem_relation);
	vector<Value*> ArgsFree;
	AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
	Builder->CreateStore(val_arena,mem_arena_void);
	Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
	ArgsFree.push_back(val_arena_void);
	Builder->CreateCall(freeLLVM, ArgsFree);

	val_arena = Builder->CreateLoad(relS.mem_relation);
	ArgsFree.clear();
	mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
	Builder->CreateStore(val_arena,mem_arena_void);
	val_arena_void = Builder->CreateLoad(mem_arena_void);
	ArgsFree.push_back(val_arena_void);
	Builder->CreateCall(freeLLVM, ArgsFree);

	val_arena = Builder->CreateLoad(htR.mem_kv);
	ArgsFree.clear();
	Value *payload_r_void = Builder->CreateBitCast(val_arena, void_ptr_type);
	ArgsFree.push_back(payload_r_void);
	Builder->CreateCall(freeLLVM, ArgsFree);

	val_arena = Builder->CreateLoad(htS.mem_kv);
	ArgsFree.clear();
	Value *payload_s_void = Builder->CreateBitCast(val_arena, void_ptr_type);
	ArgsFree.push_back(payload_s_void);
	Builder->CreateCall(freeLLVM, ArgsFree);
}

/* NULL if no exact match found, otherwise relationPtr */
/* Note that this works only with caches produced by radix atm
 * Expression caches can't be used verbatim
 *
 * I think this ends up matching columns produced by binary-col-pg
 * instead of rows produced by radix */
Scan* RadixJoin::findSideInCache(Materializer &mat, bool isLeft) const {
	CachingService& cache = CachingService::getInstance();
	bool found = true;
	string relName;

	Function *F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<> *Builder = context->getBuilder();
	PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);

	/* Is the relation already materialized?? */
	const vector<expressions::Expression*>& exps =
			mat.getWantedExpressions();

	if (!exps.empty()) {
		vector<expressions::Expression*>::const_iterator it = exps.begin();
		int failedNo = 0;
		CacheInfo info;
		for (; it != exps.end(); it++) {
			expressions::Expression *expr = *it;
			info = cache.getCache(expr);
			if (info.structFieldNo == -1) {
				//cout << failedNo << " did not match!" << endl;
				return NULL;
			}
			if (!cache.getCacheIsFull(expr)) {
				//cout << failedNo << " did not match!" << endl;
				return NULL;
			}
			failedNo++;
		}
		if (found) {
			/* Also check for KEY */
			if (isLeft) {
				expressions::Expression *expr = this->pred->getLeftOperand();
				info = cache.getCache(expr);
				if (info.structFieldNo == -1) {
					return NULL;
				}
			} else {
				expressions::Expression *expr = this->pred->getRightOperand();
				info = cache.getCache(expr);
				if (info.structFieldNo == -1) {
					return NULL;
				}
			}
			cout << "Relation side with " << info.objectTypes.size()
					<< " fields is READY" << endl;

			const vector<RecordAttribute*>& fields = mat.getWantedFields();

			const vector<RecordAttribute*>& OIDs = mat.getWantedOIDs();

			RecordType rec = RecordType(fields);
			BinaryInternalPlugin *pg = new BinaryInternalPlugin(context, rec,
					htLabel, OIDs, fields, info);

			if(isLeft)
			{
				Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type,
						*(info.payloadPtr));
				Value *val_size = context->createInt64(*(info.itemCount));

				Builder->CreateStore(val_relationR, relR.mem_relation);
				Builder->CreateStore(val_size, relR.mem_cachedTuplesNo);
				Builder->CreateStore(val_size, htR.mem_tuplesNo);
			}
			else
			{
				Value *val_relationS = context->CastPtrToLlvmPtr(char_ptr_type,
						*(info.payloadPtr));
				Value *val_size = context->createInt64(*(info.itemCount));

				Builder->CreateStore(val_relationS, relS.mem_relation);
				Builder->CreateStore(val_size, relS.mem_cachedTuplesNo);
				Builder->CreateStore(val_size, htS.mem_tuplesNo);
			}
			Scan *newScan = new Scan(context,*pg);
			return newScan;
		}
	}
	cout << "No expressions to check for" << endl;
	return NULL;
}

void RadixJoin::placeInCache(Materializer &mat, bool isLeft) const {
	IRBuilder<> *Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	PointerType *int64PtrType = Type::getInt64PtrTy(llvmContext);
	char** ptr_relation;
	Value *val_tuplesNo;
	//Value *val_buffer;

	if(isLeft)	{
		ptr_relation = ptr_relationR;
		val_tuplesNo = Builder->CreateLoad(relR.mem_tuplesNo);
	}
	else
	{
		ptr_relation = ptr_relationS;
		val_tuplesNo = Builder->CreateLoad(relS.mem_tuplesNo);
	}

	CachingService& cache = CachingService::getInstance();
	bool fullRelation;
	if(isLeft)
	{
		fullRelation = !(this->leftChild)->isFiltering();
	}
	else
	{
		fullRelation = !(this->rightChild)->isFiltering();
	}
	const vector<expressions::Expression*>& exps =
			mat.getWantedExpressions();
	const vector<RecordAttribute*>& fields = mat.getWantedFields();
	/* Note: wantedFields do not include activeTuple */
	vector<RecordAttribute*>::const_iterator itRec = fields.begin();
	int fieldNo = 0;
	CacheInfo info;
	const vector<RecordAttribute*>& oids = mat.getWantedOIDs();
	//const set<RecordAttribute>& oids = mat.getTupleIdentifiers();
	vector<RecordAttribute*>::const_iterator itOids = oids.begin();
	for (; itOids != oids.end(); itOids++) {
		RecordAttribute *attr = *itOids;
		//cout << "OID mat'ed" << endl;
		info.objectTypes.push_back(attr->getOriginalType()->getTypeID());
	}
	for (; itRec != fields.end(); itRec++) {
		//cout << "Field mat'ed" << endl;
		info.objectTypes.push_back((*itRec)->getOriginalType()->getTypeID());
	}
	itRec = fields.begin();
	/* Explicit OID ('activeTuple') will be field 0 */
	if (!exps.empty()) {

		/* By default, cache looks sth like custom_struct*.
		 * Is it possible to isolate cache for just ONE of the expressions??
		 * Group of expressions probably more palpable */
		vector<expressions::Expression*>::const_iterator it = exps.begin();
		for (; it != exps.end(); it++) {
			//info.objectType = rPayloadType;
			info.structFieldNo = fieldNo;
			info.payloadPtr = ptr_relation;
			//XXX Have LLVM exec. fill this up!
			info.itemCount = new size_t[1];
			Value *mem_itemCount = context->CastPtrToLlvmPtr(int64PtrType,(void*)info.itemCount);
			Builder->CreateStore(val_tuplesNo,mem_itemCount);
			cache.registerCache(*it, info, fullRelation);

			/* Having skipped OIDs */
			if (fieldNo >= mat.getWantedOIDs().size()) {
//				cout << "Field Cached: " << (*itRec)->getAttrName()
//						<< endl;
				itRec++;
			} else {
//				cout << "Field Cached: " << activeLoop << endl;
			}

			fieldNo++;
		}
	}
}

void RadixJoin::updateRelationPointers() const {
	Function *F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<> *Builder = context->getBuilder();
	PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);
	PointerType *char_ptr_ptr_type = PointerType::get(char_ptr_type, 0);

	Value *val_ptrRelationR = context->CastPtrToLlvmPtr(char_ptr_ptr_type,
			ptr_relationR);
	Value *val_relationR = Builder->CreateLoad(relR.mem_relation);
	Builder->CreateStore(val_relationR, val_ptrRelationR);

	Value *val_ptrRelationS = context->CastPtrToLlvmPtr(char_ptr_ptr_type,
			ptr_relationS);
	Value *val_relationS = Builder->CreateLoad(relS.mem_relation);
	Builder->CreateStore(val_relationS, val_ptrRelationS);
}

//void RadixJoin::produce() const {
//	getLeftChild().produce();
//	getRightChild().produce();
//
//	updateRelationPointers();
//
//	runRadix();
//}


void RadixJoin::produce() {
	IRBuilder<> *Builder = context->getBuilder();
	Function *func_startTime = context->getFunction("resetTime");
	Function *func_endTime = context->getFunction("calculateTime");
	//vector empty on purpose
	vector<Value*> ArgsTime;

	RawOperator *newChildLeft, *newChildRight;
	newChildLeft = NULL;
	newChildRight = NULL;

	if (!this->leftChild->isFiltering()) {
		//cout << "Checking left side for caches" << endl;
		newChildLeft = findSideInCache(matLeft, true);
	}
	if (newChildLeft == NULL) {
		//getLeftChild().produce();
		//cout << "Traditional LEFT side" << endl;
		getLeftChild()->produce();
	} else {
		cout << "NEW LEFT SCAN POSSIBLE!!" << endl;
		//this->setLeftChild(*newChildLeft);
		cachedLeft = true;
		this->setLeftChild(newChildLeft);
		newChildLeft->setParent(this);
		newChildLeft->produce();
	}

	if (!this->rightChild->isFiltering()) {
		//cout << "Checking right side for caches" << endl;
		newChildRight = findSideInCache(matRight, false);
	}
	if (newChildRight == NULL) {
		//getRightChild().produce();
		//cout << "Traditional RIGHT side" << endl;
		getRightChild()->produce();
	} else {
		cout << "NEW RIGHT SCAN POSSIBLE!!" << endl;
		cachedRight = true;
		this->setRightChild(newChildRight);
		newChildRight->setParent(this);
		newChildRight->produce();
	}


	updateRelationPointers();
	/* XXX Place info in cache */
	if (!cachedLeft) {
		placeInCache(matLeft, true);
	}
	if (!cachedRight) {
		placeInCache(matRight, false);
	}
	//Still need HTs...
	// Should I mat. them too?
	runRadix();
}

//void RadixJoin::produceNoCache() {
//	leftChild.produce();
//	rightChild.produce();
//	runRadix();
//}


void RadixJoin::runRadix() const	{

	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();

	Function* debugInt = context->getFunction("printi");
	Function* debugInt64 = context->getFunction("printi64");

	Type *int32_type = Type::getInt32Ty(llvmContext);
	Value *val_zero = context->createInt32(0);
	Value *val_one = context->createInt32(1);
#ifdef DEBUGRADIX
		vector<Value*> ArgsV0;
		ArgsV0.push_back(context->createInt32(665));
		Builder->CreateCall(debugInt,ArgsV0);
#endif
	/* Partition and Cluster 'R' (the corresponding htEntries) */
	Value *clusterCountR = radix_cluster_nopadding(relR,htR);
	/* Partition and Cluster 'S' (the corresponding htEntries) */
	Value *clusterCountS = radix_cluster_nopadding(relS,htS);

#ifdef DEBUGRADIX
		ArgsV0.clear();
		ArgsV0.push_back(context->createInt32(666));
		Builder->CreateCall(debugInt,ArgsV0);
#endif

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

	AllocaInst *mem_probesNo = Builder->CreateAlloca(int32_type,0,"mem_counter");
	Builder->CreateStore(val_zero,mem_probesNo);

	/**
	 * ACTUAL PROBES
	 */

	/* Loop through clusters */
	/* for (i = 0; i < (1 << NUM_RADIX_BITS); i++) */

	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	context->CreateForLoop("clusterLoopCond", "clusterLoopBody",
			"clusterLoopInc", "clusterLoopEnd", &loopCond, &loopBody, &loopInc,
			&loopEnd);
	context->setEndingBlock(loopEnd);

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

		Value *val_rCount = Builder->CreateLoad(mem_rCount);
		Value *val_sCount = Builder->CreateLoad(mem_sCount);
#ifdef DEBUGRADIX
		/* Cluster Counts */
		vector<Value*> ArgsV0;
		ArgsV0.push_back(context->createInt32(222));
		Builder->CreateCall(debugInt,ArgsV0);

		ArgsV0.clear();
		ArgsV0.push_back(val_clusterCount);
		Builder->CreateCall(debugInt,ArgsV0);
#endif


		/* tmpR.tuples = relR->tuples + r; */
		Value *val_htR = Builder->CreateLoad(htR.mem_kv);
		Value* htRshiftedPtr = Builder->CreateInBoundsGEP(val_htR, val_rCount);

		/* tmpS.tuples = relS->tuples + s; */
		Value *val_htS = Builder->CreateLoad(htS.mem_kv);
		Value* htSshiftedPtr = Builder->CreateInBoundsGEP(val_htS, val_sCount);

		/* bucket_chaining_join_prepare(&tmpR, &(HT_per_cluster[i])); */
		Function *bucketChainingPrepare = context->getFunction("bucketChainingPrepare");

		PointerType *htClusterPtrType = PointerType::get(htClusterType, 0);
		Value *val_htPerClusterShiftedPtr = Builder->CreateInBoundsGEP(
				val_htPerCluster, val_clusterCount);

		//Prepare args and call function
		vector<Value*> Args;
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
		Builder->CreateStore(val_sCount,mem_sCount);


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

			val_cond = Builder->CreateICmpSLT(val_j,val_s_i_count);

			Builder->CreateCondBr(val_cond, sLoopBody, sLoopEnd);

			Builder->SetInsertPoint(sLoopBody);

#ifdef DEBUGRADIX
//			Value *val_probesNo = Builder->CreateLoad(mem_probesNo);
//			val_probesNo = Builder->CreateAdd(val_probesNo, val_one);
//			Builder->CreateStore(val_probesNo,mem_probesNo);
//
//			vector<Value*> ArgsV0;
//			ArgsV0.push_back(val_j);
//			Builder->CreateCall(debugInt,ArgsV0);
//
//			ArgsV0.clear();
//			ArgsV0.push_back(val_sCount);
//			Builder->CreateCall(debugInt,ArgsV0);
#endif
			/*
			 * Break the following in pieces:
			 * result += bucket_chaining_join_probe(&tmpR,
			 *			&(HT_per_cluster[i]), &(tmpS.tuples[j]));
			 */

			/* uint32_t idx = HASH_BIT_MODULO(s->key, ht->mask, NUM_RADIX_BITS); */
			Value *val_num_radix_bits = context->createInt32(NUM_RADIX_BITS);
			Value *val_mask = context->getStructElem(val_htPerClusterShiftedPtr,2);
			//Get key of current s tuple (tmpS[j])
			Value *htSshiftedPtr_j = Builder->CreateInBoundsGEP(htSshiftedPtr, val_j);
//			Value *tuple_s_j = Builder->CreateLoad(htSshiftedPtr_j);
			Value *val_key_s_j = context->getStructElem(htSshiftedPtr_j,0);
			Value *val_idx = Builder->CreateBinOp(Instruction::And,val_key_s_j,val_mask);
			val_idx = Builder->CreateAShr(val_idx,val_num_radix_bits);

			/**
			 * Checking actual hits (when applicable)
			 * for(int hit = (ht->bucket)[idx]; hit > 0; hit = (ht->next)[hit-1])
			 */
			BasicBlock *hitLoopCond, *hitLoopBody, *hitLoopInc, *hitLoopEnd;
			context->CreateForLoop("hitLoopCond", "hitLoopBody", "hitLoopInc",
					"hitLoopEnd", &hitLoopCond, &hitLoopBody, &hitLoopInc,
					&hitLoopEnd);

			{
				AllocaInst *mem_hit = Builder->CreateAlloca(int32_type, 0, "hit");
				//(ht->bucket)
				Value *val_bucket = context->getStructElem(val_htPerClusterShiftedPtr,0);
				//(ht->bucket)[idx]
				Value *val_bucket_idx = context->getArrayElem(val_bucket,val_idx);

				Builder->CreateStore(val_bucket_idx, mem_hit);
				Builder->CreateBr(hitLoopCond);
				/* 1. Loop Condition */
				Builder->SetInsertPoint(hitLoopCond);
				Value *val_hit = Builder->CreateLoad(mem_hit);
				val_cond = Builder->CreateICmpSGT(val_hit,val_zero);
				Builder->CreateCondBr(val_cond,hitLoopBody,hitLoopEnd);

				/* 2. Body */
				Builder->SetInsertPoint(hitLoopBody);

				/* if (s->key == Rtuples[hit - 1].key) */
				BasicBlock *ifKeyMatch;
				context->CreateIfBlock(context->getGlobalFunction(), "htMatchIfCond",
									&ifKeyMatch, hitLoopInc);
				{
					//Rtuples[hit - 1]
					Value *val_idx_dec = Builder->CreateSub(val_hit,val_one);
					Value *htRshiftedPtr_hit = Builder->CreateInBoundsGEP(htRshiftedPtr, val_idx_dec);
					//Rtuples[hit - 1].key
					Value *val_key_r = context->getStructElem(htRshiftedPtr_hit,0);

					//Condition
					val_cond = Builder->CreateICmpEQ(val_key_s_j,val_key_r);
					Builder->CreateCondBr(val_cond,ifKeyMatch,hitLoopInc);

					Builder->SetInsertPoint(ifKeyMatch);


#ifdef DEBUGRADIX
					/* Printing key(s) */
//					vector<Value*> ArgsV;
//					ArgsV.push_back(val_key_s_j);
//					Builder->CreateCall(debugInt, ArgsV);

//					ArgsV.clear();
//					ArgsV.push_back(context->createInt32(1111));
//					Builder->CreateCall(debugInt, ArgsV);

//					ArgsV.clear();
//					ArgsV.push_back(val_key_r);
//					Builder->CreateCall(debugInt, ArgsV);
#endif
					/**
					 * -> RECONSTRUCT RESULTS
					 * -> CALL PARENT
					 */
					map<RecordAttribute, RawValueMemory>* allJoinBindings =
							new map<RecordAttribute, RawValueMemory>();

					/* Payloads (Relative Offsets): size_t */
					/* Must be added to relR / relS accordingly */
					Value *val_payload_r_offset =
							context->getStructElem(htRshiftedPtr_hit, 1);
					Value *val_payload_s_offset =
							context->getStructElem(htSshiftedPtr_j, 1);

					/* Cast payload */
					PointerType *rPayloadPtrType =
							PointerType::get(rPayloadType, 0);
					PointerType *sPayloadPtrType =
							PointerType::get(sPayloadType, 0);

					Value *val_relR = Builder->CreateLoad(relR.mem_relation);
					Value *val_relS = Builder->CreateLoad(relS.mem_relation);
					Value *val_ptr_payloadR =
							Builder->CreateInBoundsGEP(val_relR,val_payload_r_offset);
					Value *val_ptr_payloadS =
							Builder->CreateInBoundsGEP(val_relS,val_payload_s_offset);
#ifdef DEBUGRADIX
					{
					/* Printing key(s) */
					vector<Value*> ArgsV;
					ArgsV.push_back(Builder->getInt32(500005));
					Builder->CreateCall(debugInt, ArgsV);
					}
#endif
					Value *mem_payload_r = Builder->CreateBitCast(
							val_ptr_payloadR, rPayloadPtrType);
					Value *val_payload_r = Builder->CreateLoad(mem_payload_r);
					Value *mem_payload_s = Builder->CreateBitCast(
							val_ptr_payloadS, sPayloadPtrType);
					Value *val_payload_s = Builder->CreateLoad(mem_payload_s);

					/* LEFT SIDE (RELATION R)*/
					//Retrieving activeTuple(s) from HT
					{
						AllocaInst *mem_activeTuple = NULL;
						int i = 0;
//						const set<RecordAttribute>& tuplesIdentifiers =
//								matLeft.getTupleIdentifiers();
						const vector<RecordAttribute*>& tuplesIdentifiers =
								matLeft.getWantedOIDs();
						vector<RecordAttribute*>::const_iterator it =
								tuplesIdentifiers.begin();
						for (; it != tuplesIdentifiers.end(); it++) {
							RecordAttribute *attr = *it;
							mem_activeTuple = context->CreateEntryBlockAlloca(F,
									"mem_activeTuple",
									rPayloadType->getElementType(i));
							vector<Value*> idxList = vector<Value*>();
							idxList.push_back(context->createInt32(0));
							idxList.push_back(context->createInt32(i));

							Value *elem_ptr = Builder->CreateGEP(mem_payload_r,
									idxList);
							Value *val_activeTuple = Builder->CreateLoad(
									elem_ptr);
							StoreInst *store_activeTuple = Builder->CreateStore(
									val_activeTuple, mem_activeTuple);
							store_activeTuple->setAlignment(8);

							RawValueMemory mem_valWrapper;
							mem_valWrapper.mem = mem_activeTuple;
							mem_valWrapper.isNull = context->createFalse();
							(*allJoinBindings)[*attr] = mem_valWrapper;
							i++;
						}

						AllocaInst *mem_field = NULL;
						const vector<RecordAttribute*>& wantedFields =
								matLeft.getWantedFields();
						vector<RecordAttribute*>::const_iterator it2 =
								wantedFields.begin();
						for (; it2 != wantedFields.end(); it2++) {
							string currField = (*it2)->getName();
							mem_field = context->CreateEntryBlockAlloca(F,
									"mem_" + currField,
									rPayloadType->getElementType(i));
							vector<Value*> idxList = vector<Value*>();
							idxList.push_back(context->createInt32(0));
							idxList.push_back(context->createInt32(i));

							Value *elem_ptr = Builder->CreateGEP(mem_payload_r,
									idxList);
							Value *val_field = Builder->CreateLoad(elem_ptr);
							Builder->CreateStore(val_field, mem_field);

							RawValueMemory mem_valWrapper;
							mem_valWrapper.mem = mem_field;
							mem_valWrapper.isNull = context->createFalse();

#ifdef DEBUGRADIX
//							vector<Value*> ArgsV;
//							ArgsV.push_back(context->createInt32(1111));
//							Builder->CreateCall(debugInt, ArgsV);
//							ArgsV.clear();
//							ArgsV.push_back(Builder->CreateLoad(mem_field));
//							Builder->CreateCall(debugInt, ArgsV);
#endif

							(*allJoinBindings)[*(*it2)] = mem_valWrapper;
							i++;
						}
					}
#ifdef DEBUGRADIX
					{
					/* Printing key(s) */
					vector<Value*> ArgsV;
					ArgsV.push_back(Builder->getInt32(500006));
					Builder->CreateCall(debugInt, ArgsV);
					}
#endif
					/* RIGHT SIDE (RELATION S) */
					{
						AllocaInst *mem_activeTuple = NULL;
						int i = 0;
//						const set<RecordAttribute>& tuplesIdentifiers =
//								matRight.getTupleIdentifiers();
						const vector<RecordAttribute*>& tuplesIdentifiers =
														matRight.getWantedOIDs();
						vector<RecordAttribute*>::const_iterator it =
								tuplesIdentifiers.begin();
						for (; it != tuplesIdentifiers.end(); it++) {
							RecordAttribute *attr = *it;
							mem_activeTuple = context->CreateEntryBlockAlloca(F,
									"mem_activeTuple",
									sPayloadType->getElementType(i));
							vector<Value*> idxList = vector<Value*>();
							idxList.push_back(context->createInt32(0));
							idxList.push_back(context->createInt32(i));

							Value *elem_ptr = Builder->CreateGEP(mem_payload_s,
									idxList);
							Value *val_activeTuple = Builder->CreateLoad(
									elem_ptr);
							StoreInst *store_activeTuple = Builder->CreateStore(
									val_activeTuple, mem_activeTuple);
							store_activeTuple->setAlignment(8);

							RawValueMemory mem_valWrapper;
							mem_valWrapper.mem = mem_activeTuple;
							mem_valWrapper.isNull = context->createFalse();
							(*allJoinBindings)[*attr] = mem_valWrapper;
							i++;
#ifdef DEBUGRADIX
							{
								/* Printing key(s) */
								vector<Value*> ArgsV;
								ArgsV.push_back(val_activeTuple);
								Builder->CreateCall(debugInt64, ArgsV);
							}
#endif
						}

						AllocaInst *mem_field = NULL;
						const vector<RecordAttribute*>& wantedFields =
								matRight.getWantedFields();
						vector<RecordAttribute*>::const_iterator it2 =
								wantedFields.begin();
						for (; it2 != wantedFields.end(); it2++) {
							string currField = (*it2)->getName();
							mem_field = context->CreateEntryBlockAlloca(F,
									"mem_" + currField,
									sPayloadType->getElementType(i));
							vector<Value*> idxList = vector<Value*>();
							idxList.push_back(context->createInt32(0));
							idxList.push_back(context->createInt32(i));

							Value *elem_ptr = Builder->CreateGEP(mem_payload_s,
									idxList);
							Value *val_field = Builder->CreateLoad(elem_ptr);
							Builder->CreateStore(val_field, mem_field);

							RawValueMemory mem_valWrapper;
							mem_valWrapper.mem = mem_field;
							mem_valWrapper.isNull = context->createFalse();
#ifdef DEBUGRADIX
//							vector<Value*> ArgsV;
//							ArgsV.push_back(context->createInt32(1112));
//							Builder->CreateCall(debugInt, ArgsV);
//							ArgsV.clear();
//							ArgsV.push_back(Builder->CreateLoad(mem_field));
//							Builder->CreateCall(debugInt, ArgsV);
#endif
							(*allJoinBindings)[*(*it2)] = mem_valWrapper;
							i++;
						}
					}
#ifdef DEBUGRADIX
					{
					/* Printing key(s) */
					vector<Value*> ArgsV;
					ArgsV.push_back(Builder->getInt32(500008));
					Builder->CreateCall(debugInt, ArgsV);
					}
#endif
					/* Trigger Parent */
					OperatorState* newState = new OperatorState(*this, *allJoinBindings);
					getParent()->consume(context, *newState);

					Builder->CreateBr(hitLoopInc);
				}

				/* 3. Inc: hit = (ht->next)[hit-1]) */
				Builder->SetInsertPoint(hitLoopInc);
				//(ht->next)
				Value *val_next = context->getStructElem(val_htPerClusterShiftedPtr,1);
				val_idx = Builder->CreateSub(val_hit,val_one);
				//(ht->next)[hit-1])
				val_hit = context->getArrayElem(val_next,val_idx);
				Builder->CreateStore(val_hit,mem_hit);
				Builder->CreateBr(hitLoopCond);

				/* 4. End */
				Builder->SetInsertPoint(hitLoopEnd);
			}



			Builder->CreateBr(sLoopInc);

			Builder->SetInsertPoint(sLoopInc);
			val_j = Builder->CreateLoad(mem_j);
			val_j = Builder->CreateAdd(val_j,val_one);
			Builder->CreateStore(val_j,mem_j);
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
//#ifdef DEBUG
//		vector<Value*> ArgsV0;
//		ArgsV0.push_back(val_clusterCount);
//		Builder->CreateCall(debugInt,ArgsV0);
//#endif
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

void RadixJoin::consume(RawContext* const context, const OperatorState& childState) {

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

	/* What (ht* + payload) points to: TBD */
	StructType *payloadType;

	const RawOperator& caller = childState.getProducer();
	if(caller == *(getLeftChild()))
	{

#ifdef DEBUG
		LOG(INFO)<< "[RADIX JOIN: ] Left (building) side";
#endif
		if(!cachedLeft)
		{
			const map<RecordAttribute, RawValueMemory>& bindings =
					childState.getBindings();
			OutputPlugin* pg = new OutputPlugin(context, matLeft, bindings);

			/* Result type specified during output plugin construction */
			payloadType = pg->getPayloadType();
			rPayloadType = payloadType;

			/* 3rd Method to calculate size */
			/* REMEMBER: PADDING DOES MATTER! */
			Value* val_payloadSize = ConstantExpr::getSizeOf(rPayloadType);

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
			Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload,
					arenaSize);

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
			Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);

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
				//cout << "ORDER OF LEFT FIELDS MATERIALIZED"<<endl;
				map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
				for (memSearch = bindings.begin(); memSearch != bindings.end();
						memSearch++) {
					RecordAttribute currAttr = memSearch->first;
					//cout << currAttr.getRelationName() << "_" << currAttr.getAttrName() << endl;
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
			const vector<RecordAttribute*>& wantedFields =
					matLeft.getWantedFields();
			for (vector<RecordAttribute*>::const_iterator it =
					wantedFields.begin(); it != wantedFields.end(); ++it) {
				//cout << (*it)->getRelationName() << "_" << (*it)->getAttrName() << endl;
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
				Value* structPtr = Builder->CreateGEP(cast_arenaShifted,
						idxList);

				Builder->CreateStore(valToMaterialize, structPtr);
				offsetInStruct++;
				offsetInWanted++;
			}

			/* CONSTRUCT HTENTRY PAIR   	  */
			/* payloadPtr: relative offset from relBuffer beginning */
			/* (int32 key, int64 payloadPtr)  */
			/* Prepare key */
			expressions::Expression* leftKeyExpr = this->pred->getLeftOperand();
			ExpressionGeneratorVisitor exprGenerator =
					ExpressionGeneratorVisitor(context, childState);
			RawValue leftKey = leftKeyExpr->accept(exprGenerator);
			Type* keyType = (leftKey.value)->getType();
			//10: IntegerTyID
			if (keyType->getTypeID() != 10) {
				string error_msg = "Only INT32 keys considered atm";
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}

			PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

			BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext,
					"IfHTFullEnd", F);
			BasicBlock *ifHTFull;
			context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull,
					endBlockHTFull);

			LoadInst *val_ht = Builder->CreateLoad(htR.mem_kv);
			val_ht->setAlignment(8);

			Value *offsetInHT = Builder->CreateLoad(htR.mem_offset);
//		Value *offsetPlusKey = Builder->CreateAdd(offsetInHT,val_keySize64);
//		int payloadPtrSize = sizeof(size_t);
//		Value *val_payloadPtrSize = context->createInt64(payloadPtrSize);
//		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKey,val_payloadPtrSize);
			Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);

			Value *htSize = Builder->CreateLoad(htR.mem_size);
			offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair, htSize);

			Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

			/* true => realloc() */
			Builder->SetInsertPoint(ifHTFull);

			/* Casting htEntry* to void* requires a cast */
			Value *cast_htEntries = Builder->CreateBitCast(val_ht,
					void_ptr_type);
			ArgsRealloc.clear();
			ArgsRealloc.push_back(cast_htEntries);
			ArgsRealloc.push_back(htSize);
			Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);

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
			Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,
					val_tuplesNo);

			/* 2a. kv_cast->keyPtr = &key */
			offsetInStruct = 0;
			//Shift in htEntry (struct) ptr
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			Value* structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_key = Builder->CreateStore(leftKey.value,
					structPtr);
			store_key->setAlignment(4);

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
			val_tuplesNo = Builder->CreateAdd(val_tuplesNo,
					context->createInt64(1));
			Builder->CreateStore(val_tuplesNo, relR.mem_tuplesNo);
			Builder->CreateStore(val_tuplesNo, htR.mem_tuplesNo);
		}
		else
		{
			const map<RecordAttribute, RawValueMemory>& bindings =
					childState.getBindings();

			/* Peek */
			map<RecordAttribute, RawValueMemory>::const_iterator it =
					bindings.begin();
			for (; it != bindings.end(); it++) {
				RecordAttribute attr = it->first;
				RawValueMemory memWrap = it->second;
				if (attr.getAttrName() == activeLoop) {
#ifdef DEBUGRADIX
					{
						Function* debugInt = context->getFunction("printi64");
						vector<Value*> ArgsV;
						Value *val_oid = Builder->CreateLoad(memWrap.mem);
						ArgsV.push_back(val_oid);
						Builder->CreateCall(debugInt, ArgsV);
					}
#endif
				} else {
#ifdef DEBUGRADIX
					{
						Function* debugInt = context->getFunction("printi");
						vector<Value*> ArgsV;
						Value *val_oid = Builder->CreateLoad(memWrap.mem);
						ArgsV.push_back(val_oid);
						Builder->CreateCall(debugInt, ArgsV);
					}
#endif
				}
			}

			OutputPlugin* pg = new OutputPlugin(context, matLeft, bindings);

			/* Result type specified during output plugin construction */
			payloadType = pg->getPayloadType();
			rPayloadType = payloadType;

			Value* val_payloadSize = ConstantExpr::getSizeOf(rPayloadType);

			Value *val_arena = Builder->CreateLoad(relR.mem_relation);
			Value *offsetInArena = Builder->CreateLoad(relR.mem_offset);
			Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,
					val_payloadSize);

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

			//			Value *val_arena = Builder->CreateLoad(relR.mem_relation);
			Value* val_tuplesNo = Builder->CreateLoad(relR.mem_tuplesNo);

			/* CONSTRUCT HTENTRY PAIR   	  */
			/* payloadPtr: relative offset from relBuffer beginning */
			/* (int32 key, int64 payloadPtr)  */
			/* Prepare key */
			expressions::Expression* leftKeyExpr =
					this->pred->getLeftOperand();
			ExpressionGeneratorVisitor exprGenerator =
					ExpressionGeneratorVisitor(context, childState);
			RawValue leftKey = leftKeyExpr->accept(exprGenerator);
			Type* keyType = (leftKey.value)->getType();
			//10: IntegerTyID
			if (keyType->getTypeID() != 10) {
				string error_msg = "Only INT32 keys considered atm";
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
#ifdef DEBUGRADIX
			{
				Function* debugInt = context->getFunction("printi");
				vector<Value*> ArgsV;
				//		ArgsV.push_back(context->createInt32(-6));
				ArgsV.push_back(leftKey.value);
				Builder->CreateCall(debugInt, ArgsV);
				ArgsV.clear();
				ArgsV.push_back(context->createInt32(-10001));
				Builder->CreateCall(debugInt, ArgsV);
			}
#endif

			PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

			BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext,
					"IfHTFullEnd", F);
			BasicBlock *ifHTFull;
			context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull,
					endBlockHTFull);

			LoadInst *val_ht = Builder->CreateLoad(htR.mem_kv);
			val_ht->setAlignment(8);

			Value *offsetInHT = Builder->CreateLoad(htR.mem_offset);
			//		Value *offsetPlusKey = Builder->CreateAdd(offsetInHT,val_keySize64);
			//		int payloadPtrSize = sizeof(size_t);
			//		Value *val_payloadPtrSize = context->createInt64(payloadPtrSize);
			//		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKey,val_payloadPtrSize);
			Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);

			Value *htRize = Builder->CreateLoad(htR.mem_size);
			Value *offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair,
					htRize);

			Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

			/* true => realloc() */
			Builder->SetInsertPoint(ifHTFull);

			/* Casting htEntry* to void* requires a cast */
			Value *cast_htEntries = Builder->CreateBitCast(val_ht,
					void_ptr_type);
			vector<Value*> ArgsRealloc;
			Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
			ArgsRealloc.clear();
			ArgsRealloc.push_back(cast_htEntries);
			ArgsRealloc.push_back(htRize);
			Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);

			Value *val_newHTPtr = Builder->CreateBitCast(val_newVoidHTPtr,
					htEntryPtrType);
			Builder->CreateStore(val_newHTPtr, htR.mem_kv);
			Value *val_size = Builder->CreateLoad(htR.mem_size);
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
			Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,
					val_tuplesNo);

			/* 2a. kv_cast->keyPtr = &key */
			int offsetInStruct = 0;
			//Shift in htEntry (struct) ptr
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			Value* structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_key = Builder->CreateStore(leftKey.value,
					structPtr);
			store_key->setAlignment(4);

			/* 2b. kv_cast->payloadPtr = &payload */
			offsetInStruct = 1;
			idxList.clear();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));
			structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_payloadPtr = Builder->CreateStore(offsetInArena,
					structPtr);
			store_payloadPtr->setAlignment(8);

			/* 4. Increment counts - HT */
			Builder->CreateStore(offsetPlusKVPair, htR.mem_offset);
			Builder->CreateStore(offsetPlusPayload, relR.mem_offset);
			val_tuplesNo = Builder->CreateAdd(val_tuplesNo,
					context->createInt64(1));
			Builder->CreateStore(val_tuplesNo, relR.mem_tuplesNo);

		}

	}
	else
	{
		/* XXX comparisons with left and right child are off */
#ifdef DEBUG
		LOG(INFO)<< "[RADIX JOIN: ] Right (also building!) side";
#endif
		if (!cachedRight) {
			const map<RecordAttribute, RawValueMemory>& bindings =
					childState.getBindings();
			OutputPlugin* pg = new OutputPlugin(context, matRight, bindings);

			/* Result type specified during output plugin construction */
			payloadType = pg->getPayloadType();
			sPayloadType = payloadType;

			/* 3rd Method to calculate size */
			/* REMEMBER: PADDING DOES MATTER! */
			Value* val_payloadSize = ConstantExpr::getSizeOf(sPayloadType);

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

			Value *val_arena = Builder->CreateLoad(relS.mem_relation);
			Value *offsetInArena = Builder->CreateLoad(relS.mem_offset);
			Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,
					val_payloadSize);
			Value *arenaSize = Builder->CreateLoad(relS.mem_size);
			Value* val_tuplesNo = Builder->CreateLoad(relS.mem_tuplesNo);

			/* if(offsetInArena + payloadSize >= arenaSize) */
			BasicBlock* entryBlock = Builder->GetInsertBlock();
			BasicBlock *endBlockArenaFull = BasicBlock::Create(llvmContext,
					"IfArenaFullEnd", F);
			BasicBlock *ifArenaFull;
			context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull,
					endBlockArenaFull);
			Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload,
					arenaSize);

			Builder->CreateCondBr(offsetCond, ifArenaFull, endBlockArenaFull);

			/* true => realloc() */
			Builder->SetInsertPoint(ifArenaFull);

			vector<Value*> ArgsRealloc;
			Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
			AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type, 0,
					"voidArenaPtrS");
			Builder->CreateStore(val_arena, mem_arena_void);
			Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
			ArgsRealloc.push_back(val_arena_void);
			ArgsRealloc.push_back(arenaSize);
			Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);

			Builder->CreateStore(val_newArenaVoidPtr, relS.mem_relation);
			Value* val_size = Builder->CreateLoad(relS.mem_size);
			val_size = Builder->CreateMul(val_size, context->createInt64(2));
			Builder->CreateStore(val_size, relS.mem_size);
			Builder->CreateBr(endBlockArenaFull);

			/* 'Normal' flow again */
			Builder->SetInsertPoint(endBlockArenaFull);

			/* Repeat load - realloc() might have occurred */
			val_arena = Builder->CreateLoad(relS.mem_relation);

			/* STORING PAYLOAD */
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
			int offsetInStruct = 0;	//offset inside the struct (+current field manipulated)
			RawValueMemory mem_activeTuple;
			{
				//cout << "ORDER OF RIGHT FIELDS MATERIALIZED"<<endl;
				map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
				for (memSearch = bindings.begin(); memSearch != bindings.end();
						memSearch++) {
					RecordAttribute currAttr = memSearch->first;
					if (currAttr.getAttrName() == activeLoop) {
						//cout << currAttr.getRelationName() << "_" << currAttr.getAttrName() << endl;
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
						Builder->CreateStore(val_activeTuple, structPtr);
#ifdef DEBUGRADIX
						{
							Function* debugInt = context->getFunction(
									"printi64");
							vector<Value*> ArgsV;

							ArgsV.push_back(val_activeTuple);
							Builder->CreateCall(debugInt, ArgsV);

						}
#endif

						offsetInStruct++;
					}
				}
			}

			int offsetInWanted = 0;
			const vector<RecordAttribute*>& wantedFields =
					matRight.getWantedFields();
			for (vector<RecordAttribute*>::const_iterator it =
					wantedFields.begin(); it != wantedFields.end(); ++it) {
				//cout << (*it)->getRelationName() << "___" << (*it)->getAttrName() << endl;
				map<RecordAttribute, RawValueMemory>::const_iterator memSearch =
						bindings.find(*(*it));
				RawValueMemory currValMem = memSearch->second;
				Value* currVal = Builder->CreateLoad(currValMem.mem);
				Value* valToMaterialize = pg->convert(currVal->getType(),
						materializedTypes->at(offsetInWanted), currVal);
				vector<Value*> idxList = vector<Value*>();
				idxList.push_back(context->createInt32(0));
				idxList.push_back(context->createInt32(offsetInStruct));
				//Shift in struct ptr
				Value* structPtr = Builder->CreateGEP(cast_arenaShifted,
						idxList);
				Builder->CreateStore(valToMaterialize, structPtr);
#ifdef DEBUGRADIX
				{
					Function* debugInt = context->getFunction("printi");
					vector<Value*> ArgsV;

					ArgsV.push_back(valToMaterialize);
					Builder->CreateCall(debugInt, ArgsV);

				}
#endif
				offsetInStruct++;
				offsetInWanted++;
			}

#ifdef DEBUGRADIX
				{
					Function* debugInt = context->getFunction("printi");
					vector<Value*> ArgsV;

					ArgsV.push_back(Builder->getInt32(200002));
					Builder->CreateCall(debugInt, ArgsV);

				}
#endif

			/* CONSTRUCT HTENTRY PAIR   	  */
			/* payloadPtr: relative offset from relBuffer beginning */
			/* (int32 key, int64 payloadPtr)  */
			/* Prepare key */
			expressions::Expression* rightKeyExpr =
					this->pred->getRightOperand();
			ExpressionGeneratorVisitor exprGenerator =
					ExpressionGeneratorVisitor(context, childState);
			RawValue rightKey = rightKeyExpr->accept(exprGenerator);
			Type* keyType = (rightKey.value)->getType();

			//10: IntegerTyID
			if (keyType->getTypeID() != 10) {
				string error_msg = "Only INT32 keys considered atm";
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}

			PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

			BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext,
					"IfHTFullEnd", F);
			BasicBlock *ifHTFull;
			context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull,
					endBlockHTFull);

			LoadInst *val_ht = Builder->CreateLoad(htS.mem_kv);
			val_ht->setAlignment(8);

			Value *offsetInHT = Builder->CreateLoad(htS.mem_offset);
//		Value *offsetPlusKey = Builder->CreateAdd(offsetInHT,val_keySize64);
//		int payloadPtrSize = sizeof(size_t);
//		Value *val_payloadPtrSize = context->createInt64(payloadPtrSize);
//		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKey,val_payloadPtrSize);
			Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);
			Value *htSize = Builder->CreateLoad(htS.mem_size);
			offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair, htSize);

			Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

			/* true => realloc() */
			Builder->SetInsertPoint(ifHTFull);

			/* Casting htEntry* to void* requires a cast */
			Value *cast_htEntries = Builder->CreateBitCast(val_ht,
					void_ptr_type);
			ArgsRealloc.clear();
			ArgsRealloc.push_back(cast_htEntries);
			ArgsRealloc.push_back(htSize);
			Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);
			Value *val_newHTPtr = Builder->CreateBitCast(val_newVoidHTPtr,
					htEntryPtrType);
			Builder->CreateStore(val_newHTPtr, htS.mem_kv);
			val_size = Builder->CreateLoad(htS.mem_size);
			val_size = Builder->CreateMul(val_size, context->createInt64(2));
			Builder->CreateStore(val_size, htS.mem_size);
			Builder->CreateBr(endBlockHTFull);

			/* Insert ht entry in HT */
			Builder->SetInsertPoint(endBlockHTFull);

			/* Repeat load - realloc() might have occurred */
			val_ht = Builder->CreateLoad(htS.mem_kv);
			val_ht->setAlignment(8);

			val_size = Builder->CreateLoad(htS.mem_size);

			/* 1. kv += offset */
			/* Note that we already have a htEntry ptr here */
			Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,
					val_tuplesNo);

			/* 2a. kv_cast->keyPtr = &key */
			offsetInStruct = 0;
			//Shift in htEntry (struct) ptr
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			Value* structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_key = Builder->CreateStore(rightKey.value,
					structPtr);
			store_key->setAlignment(4);

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
			Builder->CreateStore(offsetPlusPayload, relS.mem_offset);
			Builder->CreateStore(offsetPlusKVPair, htS.mem_offset);
			val_tuplesNo = Builder->CreateAdd(val_tuplesNo,
					context->createInt64(1));
			Builder->CreateStore(val_tuplesNo, relS.mem_tuplesNo);
			Builder->CreateStore(val_tuplesNo, htS.mem_tuplesNo);
		}
		/* RIGHT SIDE CACHED */
		else
		{

			//cout << "RIGHT SIDE IS READY (CACHED)" << endl;
			const map<RecordAttribute, RawValueMemory>& bindings =
					childState.getBindings();

			/* Peek */
			map<RecordAttribute,RawValueMemory>::const_iterator it = bindings.begin();
			for(; it != bindings.end(); it++)	{
				RecordAttribute attr = it->first;
				RawValueMemory memWrap = it->second;
				if (attr.getAttrName() == activeLoop) {
#ifdef DEBUGRADIX
					{
						Function* debugInt = context->getFunction("printi64");
						vector<Value*> ArgsV;
						Value *val_oid = Builder->CreateLoad(memWrap.mem);
						ArgsV.push_back(val_oid);
						Builder->CreateCall(debugInt, ArgsV);
					}
#endif
				} else {
#ifdef DEBUGRADIX
					{
						Function* debugInt = context->getFunction("printi");
						vector<Value*> ArgsV;
						Value *val_oid = Builder->CreateLoad(memWrap.mem);
						ArgsV.push_back(val_oid);
						Builder->CreateCall(debugInt, ArgsV);
					}
#endif
				}
			}


			OutputPlugin* pg = new OutputPlugin(context, matRight, bindings);

			/* Result type specified during output plugin construction */
			payloadType = pg->getPayloadType();
			sPayloadType = payloadType;

			Value* val_payloadSize = ConstantExpr::getSizeOf(sPayloadType);

			Value *val_arena = Builder->CreateLoad(relS.mem_relation);
			Value *offsetInArena = Builder->CreateLoad(relS.mem_offset);
			Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena,
					val_payloadSize);

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

//			Value *val_arena = Builder->CreateLoad(relS.mem_relation);
			Value* val_tuplesNo = Builder->CreateLoad(relS.mem_tuplesNo);

			/* CONSTRUCT HTENTRY PAIR   	  */
			/* payloadPtr: relative offset from relBuffer beginning */
			/* (int32 key, int64 payloadPtr)  */
			/* Prepare key */
			expressions::Expression* rightKeyExpr =
					this->pred->getRightOperand();
			ExpressionGeneratorVisitor exprGenerator =
					ExpressionGeneratorVisitor(context, childState);
			RawValue rightKey = rightKeyExpr->accept(exprGenerator);
			Type* keyType = (rightKey.value)->getType();
			//10: IntegerTyID
			if (keyType->getTypeID() != 10) {
				string error_msg = "Only INT32 keys considered atm";
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
			#ifdef DEBUGRADIX
			{
				Function* debugInt = context->getFunction("printi");
				vector<Value*> ArgsV;
				//		ArgsV.push_back(context->createInt32(-6));
				ArgsV.push_back(rightKey.value);
				Builder->CreateCall(debugInt, ArgsV);
				ArgsV.clear();
				ArgsV.push_back(context->createInt32(-10001));
				Builder->CreateCall(debugInt, ArgsV);
			}
			#endif

			PointerType *htEntryPtrType = PointerType::get(htEntryType, 0);

			BasicBlock *endBlockHTFull = BasicBlock::Create(llvmContext,
					"IfHTFullEnd", F);
			BasicBlock *ifHTFull;
			context->CreateIfBlock(F, "IfHTFullCond", &ifHTFull,
					endBlockHTFull);

			LoadInst *val_ht = Builder->CreateLoad(htS.mem_kv);
			val_ht->setAlignment(8);

			Value *offsetInHT = Builder->CreateLoad(htS.mem_offset);
			//		Value *offsetPlusKey = Builder->CreateAdd(offsetInHT,val_keySize64);
			//		int payloadPtrSize = sizeof(size_t);
			//		Value *val_payloadPtrSize = context->createInt64(payloadPtrSize);
			//		Value *offsetPlusPayloadPtr = Builder->CreateAdd(offsetPlusKey,val_payloadPtrSize);
			Value *offsetPlusKVPair = Builder->CreateAdd(offsetInHT, kvSize);

			Value *htSize = Builder->CreateLoad(htS.mem_size);
			Value *offsetCond = Builder->CreateICmpSGE(offsetPlusKVPair,
					htSize);

			Builder->CreateCondBr(offsetCond, ifHTFull, endBlockHTFull);

			/* true => realloc() */
			Builder->SetInsertPoint(ifHTFull);

			/* Casting htEntry* to void* requires a cast */
			Value *cast_htEntries = Builder->CreateBitCast(val_ht,
					void_ptr_type);
			vector<Value*> ArgsRealloc;
			Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
			ArgsRealloc.clear();
			ArgsRealloc.push_back(cast_htEntries);
			ArgsRealloc.push_back(htSize);
			Value *val_newVoidHTPtr = Builder->CreateCall(reallocLLVM,
					ArgsRealloc);

			Value *val_newHTPtr = Builder->CreateBitCast(val_newVoidHTPtr,
					htEntryPtrType);
			Builder->CreateStore(val_newHTPtr, htS.mem_kv);
			Value *val_size = Builder->CreateLoad(htS.mem_size);
			val_size = Builder->CreateMul(val_size, context->createInt64(2));
			Builder->CreateStore(val_size, htS.mem_size);
			Builder->CreateBr(endBlockHTFull);

			/* Insert ht entry in HT */
			Builder->SetInsertPoint(endBlockHTFull);

			/* Repeat load - realloc() might have occurred */
			val_ht = Builder->CreateLoad(htS.mem_kv);
			val_ht->setAlignment(8);

			val_size = Builder->CreateLoad(htS.mem_size);

			/* 1. kv += offset */
			/* Note that we already have a htEntry ptr here */
			Value *ptr_kvShifted = Builder->CreateInBoundsGEP(val_ht,
					val_tuplesNo);

			/* 2a. kv_cast->keyPtr = &key */
			int offsetInStruct = 0;
			//Shift in htEntry (struct) ptr
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));

			Value* structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_key = Builder->CreateStore(rightKey.value,
					structPtr);
			store_key->setAlignment(4);

			/* 2b. kv_cast->payloadPtr = &payload */
			offsetInStruct = 1;
			idxList.clear();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(offsetInStruct));
			structPtr = Builder->CreateGEP(ptr_kvShifted, idxList);
			StoreInst *store_payloadPtr = Builder->CreateStore(offsetInArena,
					structPtr);
			store_payloadPtr->setAlignment(8);

			/* 4. Increment counts - HT */
			Builder->CreateStore(offsetPlusKVPair, htS.mem_offset);
			Builder->CreateStore(offsetPlusPayload, relS.mem_offset);
			val_tuplesNo = Builder->CreateAdd(val_tuplesNo,
					context->createInt64(1));
			Builder->CreateStore(val_tuplesNo, relS.mem_tuplesNo);

		}
	}
};



