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

#include "operators/nest.hpp"


/**
 * NOTE: This is a fully pipeline-breaking operator!
 * Once the HT has been build, execution must
 * stop entirely before finishing the operator's work
 * => generation takes place in two steps
 */
void Nest::produce()	const {
	getChild()->produce();
	generateProbe(this->context, *childState);
}

void Nest::consume(RawContext* const context, const OperatorState& childState) const {
	generateInsert(context, childState);
}

void Nest::generateInsert(RawContext* const context, const OperatorState& childState) const
{
	this->childState = childState;
	this->context = context;

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	RawCatalog& catalog = RawCatalog::getInstance();

	/**
	 * STEP: Perform aggregation. Create buckets and fill them up IF g satisfied
	 * Technically, this check should be made after the buckets have been filled.
	 * Is this 'rewrite' correct?
	 */

	/**
	 * TODO do null check based on g!!!
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
	std::vector<Type*>* materializedTypes = pg->getMaterializedTypes();

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
		RawValueMemory currValMem = memSearch->second;
		//FIXME FIX THE NECESSARY CONVERSIONS HERE
		Value* currVal = Builder->CreateLoad(currValMem.mem);
		Value* valToMaterialize = pg->convert(currVal->getType(),materializedTypes->at(offsetInWanted),currVal);
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
	std::vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(globalStr);
	ArgsV.push_back(groupKey.value);
	ArgsV.push_back(voidCast);
	//Passing size as well
	ArgsV.push_back(context->createInt32(pg->getPayloadTypeSize()));
	Function* insert = context->getFunction("insertHT");
	Builder->CreateCall(insert, ArgsV);
}

void Nest::generateProbe(RawContext* const context,
		const OperatorState& childState) const
{

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	RawCatalog& catalog = RawCatalog::getInstance();
	std::vector<Value*> ArgsV;
	Value* globalStr = context->CreateGlobalString(htName);

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
	//Entry Block of bucket loop - Initializing counter
	BasicBlock* codeSpot = Builder->GetInsertBlock();
	PointerType* i8_ptr = PointerType::get(IntegerType::get(llvmContext, 8), 0);
	ConstantPointerNull* const_null = ConstantPointerNull::get(i8_ptr);
	AllocaInst* mem_bucketCounter = new AllocaInst(
			IntegerType::get(llvmContext, 64), "mem_bucketCnt", codeSpot);
	Builder->CreateStore(context->createInt32(0), mem_bucketCounter);

	Builder->CreateBr(loopCondHT);

	//Condition:  current position in result array is NULL
	Value* bucketCounter = Builder->CreateLoad(mem_bucketCounter);
	Value* arrayShifted = context->getArrayElem(metadataArray,
			metadataArrayType, bucketCounter);
	Value* htKeysEnd = Builder->CreateICmpNE(arrayShifted, const_null,
			"cmpMatchesEnd");
	Builder->CreateCondBr(htKeysEnd, loopBodyHT, loopEndHT);

	//Body per Key
	Builder->SetInsertPoint(loopBodyHT);

	//	//3. (nested) Loop through EACH bucket chain (i.e. all results for a key)
	//	BasicBlock *loopCondBucket, *loopBodyBucket, *loopIncBucket, *loopEndBucket;
	//		context->CreateForLoop("LoopCondBucket","LoopBodyBucket","LoopIncBucket","LoopEndBucket",
	//									&loopCondBucket,&loopBodyBucket,&loopIncBucket,&loopEndBucket);

	//XXX Return type affects what we need to do here

	//	switch (acc) {
	//		case SUM:
	//		case MULTIPLY:
	//		case MAX:
	//		case OR:
	//		case AND:
	//			break;
	//		//TODO TODO TODO
	//		case UNION:
	//		case BAGUNION:
	//		case APPEND:
	//		default: {
	//			string error_msg = string("[Nest: ] Unknown/Unsupported accumulator");
	//			LOG(ERROR)<< error_msg;
	//			throw runtime_error(error_msg);
	//		}
	//		}

	//Inc block
	Builder->CreateBr(loopIncHT);
	bucketCounter = Builder->CreateLoad(mem_bucketCounter);
	Value *val_inc = Builder->getInt64(1);
	Value* val_new = Builder->CreateAdd(bucketCounter, val_inc);
	Builder->CreateStore(val_new, mem_bucketCounter);

	Builder->SetInsertPoint(loopIncHT);
	Builder->CreateBr(loopCondHT);

	//Ending block of buckets loop
	Builder->SetInsertPoint(loopEndHT);

}

















