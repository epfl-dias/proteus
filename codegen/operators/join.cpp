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

#include "operators/join.hpp"

void Join::produce() {
	getLeftChild()->produce();
	getRightChild()->produce();
}

//TODO For now, materializing in a struct
//Note: Pointer manipulation for this struct is not the same as char* manipulation
void Join::consume(RawContext* const context, const OperatorState& childState) {

	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function* TheFunction = context->getGlobalFunction();
	IRBuilder<>* Builder = context->getBuilder();

	//Name of hashtable: Used from both sides of the join
	//Value* globalStr = context->CreateGlobalString(htName);
	//ID of hashtable: Used from both sides of the join
	int idHashtable = catalog.getIntHashTableID(htName);
	Value *val_idHashtable = context->createInt32(idHashtable);

	llvm::StructType* payloadType;
	Function* debugInt = context->getFunction("printi");
	Function* debugInt64 = context->getFunction("printi64");

	const RawOperator& caller = childState.getProducer();
	if(caller == *(getLeftChild()))
	{

#ifdef DEBUG
		LOG(INFO) << "[JOIN: ] Building side";
#endif
		const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();
		OutputPlugin* pg = new OutputPlugin(context, mat, bindings);

		//Result type specified during output plugin construction
		payloadType = pg->getPayloadType();
		//Creating space for the payload. XXX Might have to set alignment explicitly
		AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,string("valueInHT"),payloadType);
		//Registering payload type of HT in RAW CATALOG
		catalog.insertTableInfo(string(this->htName),payloadType);


		// Creating and Populating Payload Struct
		int offsetInStruct = 0; //offset inside the struct (+current field manipulated)
		std::vector<Type*>* materializedTypes = pg->getMaterializedTypes();

		//Materializing all activeTuples met so far
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
					Value* structPtr = Builder->CreateGEP(Alloca, idxList);
					Builder->CreateStore(val_activeTuple,structPtr);
				}
			}
		}

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
			Value* structPtr = Builder->CreateGEP(Alloca, idxList);
			Builder->CreateStore(valToMaterialize,structPtr);
			offsetInStruct++;
			offsetInWanted++;
		}

		//PREPARE KEY
		expressions::Expression* leftKeyExpr = this->pred->getLeftOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
		RawValue leftKey = leftKeyExpr->accept(exprGenerator);


		//INSERT VALUES IN HT
		//Not sure whether I need to store these args in memory as well
		PointerType* voidType = PointerType::get(IntegerType::get(llvmContext, 8), 0);
		Value* voidCast = Builder->CreateBitCast(Alloca, voidType,"valueVoidCast");
		//Prepare hash_insert_function arguments
		std::vector<Value*> ArgsV;
		ArgsV.clear();
		ArgsV.push_back(val_idHashtable);
		ArgsV.push_back(leftKey.value);
		ArgsV.push_back(voidCast);
		//Passing size as well
		ArgsV.push_back(context->createInt32(pg->getPayloadTypeSize()));
		Function* insert;

		//Pick appropriate insertion function based on key type
		//FIXME Key type should not matter -> Use expression hasher here, key will always be an integer
		Type* keyType = (leftKey.value)->getType();
		switch(keyType->getTypeID())
		{
		case 	10: //IntegerTyID:
			insert = context->getFunction("insertInt");
			break;
		case 	2: //FloatTyID:
		case 	3: //DoubleTyID:
		case	13: //ArrayTyID:
		case 	14: //PointerTyID:
			throw runtime_error(string("Type of key for join not supported (yet)!!"));
		default:	throw runtime_error(string("Type of key for join not supported!!"));
		}
		Builder->CreateCall(insert, ArgsV/*,"insertInt"*/);
	}
	else
	{

#ifdef DEBUG
		LOG(INFO) << "[JOIN: ] Probing side";
#endif

		//PREPARE KEY
		expressions::Expression* rightKeyExpr = this->pred->getRightOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
		RawValue rightKey = rightKeyExpr->accept(exprGenerator);
		int typeIdx = RawCatalog::getInstance().getTypeIndex(string(this->htName));
		Value* idx = context->createInt32(typeIdx);

		//Prepare hash_probe_function arguments
		std::vector<Value*> ArgsV;
		ArgsV.clear();
		ArgsV.push_back(val_idHashtable);
		ArgsV.push_back(rightKey.value);
		ArgsV.push_back(idx);
		Function* probe;

		//Pick appropriate probing function based on key type
		Type* keyType = (rightKey.value)->getType();
		switch(keyType->getTypeID())
		{
		case 	10: //IntegerTyID:
			probe = context->getFunction("probeInt");
			break;
		case 	2: //FloatTyID:
		case 	3: //DoubleTyID:
		case	13: //ArrayTyID:
		case 	14: //PointerTyID:
			throw runtime_error(string("Type of key for join not supported (yet)!!"));
		default:	throw runtime_error(string("Type of key for join not supported!!"));
		}

		Value* voidJoinBindings = Builder->CreateCall(probe, ArgsV/*,"probeInt"*/);

		//LOOP THROUGH RESULTS (IF ANY)

		//Entry block: the one I am currently in
		BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
		context->CreateForLoop("joinResultCond","joinResultBody","joinResultInc","joinResultEnd",
								&loopCond,&loopBody,&loopInc,&loopEnd);

		//ENTRY BLOCK OF RESULT LOOP
		PointerType* i8_ptr = PointerType::get(IntegerType::get(llvmContext, 8), 0);
		ConstantPointerNull* const_null = ConstantPointerNull::get(i8_ptr);

		BasicBlock* codeSpot = Builder->GetInsertBlock();
		AllocaInst* ptr_i = new AllocaInst(IntegerType::get(llvmContext, 32), "i_mem",codeSpot);
		ptr_i->setAlignment(4);
		StoreInst* store_i = new StoreInst(context->createInt32(0), ptr_i, false,codeSpot);
		store_i->setAlignment(4);
		Builder->CreateBr(loopCond);

		//CONDITION OF RESULT LOOP
		LoadInst* load_cnt = new LoadInst(ptr_i, "", false);
		load_cnt->setAlignment(4);
		loopCond->getInstList().push_back(load_cnt);
		CastInst* int64_idxprom = new SExtInst(load_cnt, IntegerType::get(llvmContext, 64), "idxprom");
		loopCond->getInstList().push_back(int64_idxprom);
		//Normally I would load the void array here. But I have it ready by the call previously
		GetElementPtrInst* ptr_arrayidx = GetElementPtrInst::Create(voidJoinBindings->getType()->getPointerElementType(), voidJoinBindings, int64_idxprom, "arrayidx");
		loopCond->getInstList().push_back(ptr_arrayidx);
		LoadInst* arrayShifted = new LoadInst(ptr_arrayidx, "", false);
		arrayShifted->setAlignment(8);
		loopCond->getInstList().push_back(arrayShifted);
		//Ending condition: current position in result array is NULL
		ICmpInst* int_cmp = new ICmpInst(*loopCond, ICmpInst::ICMP_NE, arrayShifted, const_null, "cmpMatchesEnd");
		BranchInst::Create(loopBody, loopEnd, int_cmp, loopCond);

		//BODY OF RESULT LOOP
		LoadInst* load_cnt_body = new LoadInst(ptr_i, "", false, loopBody);
		load_cnt_body->setAlignment(4);
		CastInst* int64_idxprom_body = new SExtInst(load_cnt_body, IntegerType::get(context->getLLVMContext(), 64), "idxprom1", loopBody);
		GetElementPtrInst* ptr_arrayidx_body = GetElementPtrInst::Create(IntegerType::get(llvmContext, 64), voidJoinBindings, int64_idxprom_body, "arrayidx2", loopBody);
		LoadInst* arrayShiftedBody = new LoadInst(ptr_arrayidx_body, "", false, loopBody);
		arrayShiftedBody->setAlignment(8);

		//Result (payload) type and appropriate casts
		Type* structType = RawCatalog::getInstance().getTypeInternal(typeIdx);
		PointerType* structPtrType = context->getPointerType(structType);

		CastInst* result_cast = new BitCastInst(arrayShiftedBody, structPtrType, "", loopBody);
		AllocaInst* ptr_result = context->CreateEntryBlockAlloca(TheFunction,string("htValue"),structPtrType);
		StoreInst* store_result = new StoreInst(result_cast, ptr_result, false, loopBody);
		store_result->setAlignment(8);

		//Need to store each part of result --> Not the overall struct, just the components
		StructType* str = (llvm::StructType*) structType;
		//str->dump();
		unsigned elemNo = str->getNumElements();
		LOG(INFO) << "[JOIN: ] Elements in result struct: "<<elemNo;
		map<RecordAttribute, RawValueMemory>* allJoinBindings = new map<RecordAttribute, RawValueMemory>();

		int i = 0;
		//Retrieving activeTuple(s) from HT
		AllocaInst *mem_activeTuple = NULL;
		const vector<RecordAttribute*>& tuplesIdentifiers = mat.getWantedOIDs();
		for(vector<RecordAttribute*>::const_iterator it = tuplesIdentifiers.begin(); it!=tuplesIdentifiers.end(); it++)	{
			RecordAttribute *attr = *it;
			mem_activeTuple = context->CreateEntryBlockAlloca(TheFunction,"mem_activeTuple",str->getElementType(i));
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(i));
			GetElementPtrInst* elem_ptr = GetElementPtrInst::Create(str->getElementType(i), result_cast, idxList, "ptr_activeTuple", loopBody);
			stringstream ss;
			ss<<activeLoop;
			ss<<i;
			LoadInst* field = new LoadInst(elem_ptr,ss.str(), false, loopBody);
			StoreInst* store_field = new StoreInst(field, mem_activeTuple, false, loopBody);

			RawValueMemory mem_valWrapper;
			mem_valWrapper.mem = mem_activeTuple;
			mem_valWrapper.isNull = context->createFalse();
			(*allJoinBindings)[*attr] = mem_valWrapper;
			i++;
		}

		const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
		for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it!= wantedFields.end(); ++it) {
			string currField = (*it)->getName();
			AllocaInst *memForField = context->CreateEntryBlockAlloca(TheFunction,currField+"mem",str->getElementType(i));
			vector<Value*> idxList = vector<Value*>();
			idxList.push_back(context->createInt32(0));
			idxList.push_back(context->createInt32(i));
			GetElementPtrInst* elem_ptr = GetElementPtrInst::Create(str->getElementType(i), result_cast, idxList, currField+"ptr", loopBody);
			LoadInst* field = new LoadInst(elem_ptr,currField, false, loopBody);
			StoreInst* store_field = new StoreInst(field, memForField, false, loopBody);
			i++;

			RawValueMemory mem_valWrapper;
			mem_valWrapper.mem = memForField;
			mem_valWrapper.isNull = context->createFalse();
			(*allJoinBindings)[*(*it)] = mem_valWrapper;

#ifdef DEBUG
			LOG(INFO) << "[JOIN: ] Lhs Binding name: "<<currField;
#endif
		}
		//Forwarding/pipelining bindings of rhs too
		const map<RecordAttribute, RawValueMemory>& rhsBindings = childState.getBindings();
		for(map<RecordAttribute, RawValueMemory>::const_iterator it = rhsBindings.begin(); it!= rhsBindings.end(); ++it) {
#ifdef DEBUG
			LOG(INFO) << "[JOIN: ] Rhs Binding name: "<<(it->first).getAttrName();
#endif
			(*allJoinBindings).insert(*it);
		}

#ifdef DEBUG
		LOG(INFO) << "[JOIN: ] Number of all join bindings: "<<allJoinBindings->size();
#endif
		//TRIGGER PARENT
		Builder->SetInsertPoint(loopBody);

		OperatorState* newState = new OperatorState(*this, *allJoinBindings);
		getParent()->consume(context, *newState);

		Builder->CreateBr(loopInc);

		// Block for.inc (label_for_inc)
		LoadInst* cnt_curr = new LoadInst(ptr_i, "", false, loopInc);
		cnt_curr->setAlignment(4);
		BinaryOperator* cnt_inc1 = BinaryOperator::Create(Instruction::Add, cnt_curr, context->createInt32(1), "i_inc", loopInc);
		StoreInst* store_cnt = new StoreInst(cnt_inc1, ptr_i, false, loopInc);
		store_cnt->setAlignment(4);

		BranchInst::Create(loopCond, loopInc);

		Builder->SetInsertPoint(loopEnd);
	}
};
