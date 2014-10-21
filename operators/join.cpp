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

void Join::produce() const {
	getLeftChild().produce();
	getRightChild().produce();
}

//TODO For now, materializing in a struct
//Note: Pointer manipulation for this struct is not the same as char* manipulation
void Join::consume(RawContext* const context, const OperatorState& childState) const {

	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function* TheFunction = context->getGlobalFunction();
	IRBuilder<>* TheBuilder = context->getBuilder();
	//Name of hashtable: Used from both sides of the join
	Value* globalStr = context->CreateGlobalString(htName);
	llvm::StructType* payloadType;

	const RawOperator& caller = childState.getProducer();
	if(caller == getLeftChild())
	{
		LOG(INFO) << "[JOIN: ] Building side";
		const std::map<string, AllocaInst*>& bindings = childState.getBindings();
		OutputPlugin* pg = new OutputPlugin(context, getMaterializer(), bindings);

		//Result type specified during output plugin construction
		payloadType = pg->getPayloadType();
		//Creating space for the payload. XXX Might have to set alignment explicitly
		AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,string("valueInHT"),payloadType);
		//Registering payload type of HT in RAW CATALOG
		catalog.insertTableInfo(string(this->htName),payloadType);


		// Creating and Populating Payload Struct
		int offset = 0; //offset inside the struct (+current field manipulated)
		std::vector<Type*>* materializedTypes = pg->getMaterializedTypes();

		//Materializing activeTuple
		AllocaInst* mem_activeTuple = NULL;
		{
			std::map<string, AllocaInst*>::const_iterator memSearch = bindings.find(activeTuple);
			if(memSearch != bindings.end())	{
				mem_activeTuple = memSearch->second;
				Value* val_activeTuple = TheBuilder->CreateLoad(mem_activeTuple);
				std::vector<Value*> idxList {context->createInt32(0),context->createInt32(offset)};
				//Shift in struct ptr
				Value* structPtr = TheBuilder->CreateGEP(Alloca, idxList);
				TheBuilder->CreateStore(val_activeTuple,structPtr);
				//OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR EXTRA OFFSET
				offset+=1;
			}	else	{
				string error_msg = string("[Join: ] Could not find tuple information");
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		}

		const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
		for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it!=wantedFields.end(); ++it)
		{
			std::map<string, AllocaInst*>::const_iterator memSearch = bindings.find((*it)->getName());
			AllocaInst* currValMem = memSearch->second;
			//FIXME FIX THE NECESSARY CONVERSIONS HERE
			Value* currVal = TheBuilder->CreateLoad(currValMem);
			Value* valToMaterialize = pg->convert(currVal->getType(),materializedTypes->at(offset),currVal);
			std::vector<Value*> idxList {context->createInt32(0),context->createInt32(offset)};
			//Shift in struct ptr
			Value* structPtr = TheBuilder->CreateGEP(Alloca, idxList);
			TheBuilder->CreateStore(valToMaterialize,structPtr);
			//OFFSET OF 1 MOVES TO THE NEXT MEMBER OF THE STRUCT - NO REASON FOR EXTRA OFFSET
			offset+=1;
		}

		//PREPARE KEY
		expressions::Expression* leftKeyExpr = this->pred->getLeftOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState, getLeftPlugin());
		Value* leftKey = leftKeyExpr->accept(exprGenerator);


		//INSERT VALUES IN HT
		//Not sure whether I need to store these args in memory as well
		PointerType* voidType = PointerType::get(IntegerType::get(llvmContext, 8), 0);
		Value* voidCast = TheBuilder->CreateBitCast(Alloca, voidType,"valueVoidCast");
		//Prepare hash_insert_function arguments
		std::vector<Value*> ArgsV;
		ArgsV.clear();
		ArgsV.push_back(globalStr);
		ArgsV.push_back(leftKey);
		ArgsV.push_back(voidCast);
		//Passing size as well
		ArgsV.push_back(context->createInt32(pg->getPayloadTypeSize()));
		Function* insert;

		//Pick appropriate insertion function based on key type
		Type* keyType = leftKey->getType();
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
		TheBuilder->CreateCall(insert, ArgsV/*,"insertInt"*/);
	}
	else
	{
		LOG(INFO) << "[JOIN: ] Probing side";

		//PREPARE KEY
		expressions::Expression* rightKeyExpr = this->pred->getRightOperand();
		ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState, getRightPlugin());
		Value* rightKey = rightKeyExpr->accept(exprGenerator);
		int typeIdx = RawCatalog::getInstance().getTypeIndex(string(this->htName));
		Value* idx = context->createInt32(typeIdx);

		//Prepare hash_probe_function arguments
		std::vector<Value*> ArgsV;
		ArgsV.clear();
		ArgsV.push_back(globalStr);
		ArgsV.push_back(rightKey);
		ArgsV.push_back(idx);
		Function* probe;

		//Pick appropriate probing function based on key type
		Type* keyType = rightKey->getType();
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
		Value* voidJoinBindings = TheBuilder->CreateCall(probe, ArgsV/*,"probeInt"*/);


		//LOOP THROUGH RESULTS (IF ANY)

		//Entry block: the one I am currently in
		BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
		context->CreateForLoop("joinResultCond","joinResultBody","joinResultInc","joinResultEnd",
								&loopCond,&loopBody,&loopInc,&loopEnd);

		//ENTRY BLOCK OF RESULT LOOP
		PointerType* i8_ptr = PointerType::get(IntegerType::get(llvmContext, 8), 0);
		ConstantPointerNull* const_null = ConstantPointerNull::get(i8_ptr);

		BasicBlock* codeSpot = TheBuilder->GetInsertBlock();
		AllocaInst* ptr_i = new AllocaInst(IntegerType::get(llvmContext, 32), "i_mem",codeSpot);
		ptr_i->setAlignment(4);
		StoreInst* store_i = new StoreInst(context->createInt32(0), ptr_i, false,codeSpot);
		store_i->setAlignment(4);
		TheBuilder->CreateBr(loopCond);

		//CONDITION OF RESULT LOOP
		LoadInst* load_cnt = new LoadInst(ptr_i, "", false);
		load_cnt->setAlignment(4);
		loopCond->getInstList().push_back(load_cnt);
		CastInst* int64_idxprom = new SExtInst(load_cnt, IntegerType::get(llvmContext, 64), "idxprom");
		loopCond->getInstList().push_back(int64_idxprom);
		//Normally I would load the void array here. But I have it ready by the call previously
		GetElementPtrInst* ptr_arrayidx = GetElementPtrInst::Create(voidJoinBindings, int64_idxprom, "arrayidx");
		loopCond->getInstList().push_back(ptr_arrayidx);
		LoadInst* arrayShifted = new LoadInst(ptr_arrayidx, "", false);
		arrayShifted->setAlignment(8);
		loopCond->getInstList().push_back(arrayShifted);
		ICmpInst* int_cmp = new ICmpInst(*loopCond, ICmpInst::ICMP_NE, arrayShifted, const_null, "cmpMatchesEnd");
		BranchInst::Create(loopBody, loopEnd, int_cmp, loopCond);

		//BODY OF RESULT LOOP
		LoadInst* load_cnt_body = new LoadInst(ptr_i, "", false, loopBody);
		load_cnt_body->setAlignment(4);
		CastInst* int64_idxprom_body = new SExtInst(load_cnt_body, IntegerType::get(context->getLLVMContext(), 64), "idxprom1", loopBody);
		GetElementPtrInst* ptr_arrayidx_body = GetElementPtrInst::Create(voidJoinBindings, int64_idxprom_body, "arrayidx2", loopBody);
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
		std::map<std::string, AllocaInst*>* allJoinBindings = new std::map<std::string, AllocaInst*>();


		int i = 0;
		//Retrieving activeTuple
		AllocaInst *mem_activeTuple = NULL;
		{
			mem_activeTuple = context->CreateEntryBlockAlloca(TheFunction,"mem_activeTuple",str->getElementType(i));
			std::vector<Value*> idxList {context->createInt32(0),context->createInt32(i)};
			GetElementPtrInst* elem_ptr = GetElementPtrInst::Create(result_cast, idxList, "ptr_activeTuple", loopBody);
			LoadInst* field = new LoadInst(elem_ptr,activeTuple, false, loopBody);
			StoreInst* store_field = new StoreInst(field, mem_activeTuple, false, loopBody);
			i++;
			(*allJoinBindings)[activeTuple] = mem_activeTuple;
		}

		const vector<RecordAttribute*>& wantedFields = mat.getWantedFields();
		for(std::vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it!= wantedFields.end(); ++it) {
			string currField = (*it)->getName();
			AllocaInst *memForField = context->CreateEntryBlockAlloca(TheFunction,currField+"mem",str->getElementType(i));
			std::vector<Value*> idxList {context->createInt32(0),context->createInt32(i)};
			GetElementPtrInst* elem_ptr = GetElementPtrInst::Create(result_cast, idxList, currField+"ptr", loopBody);
			LoadInst* field = new LoadInst(elem_ptr,currField, false, loopBody);
			StoreInst* store_field = new StoreInst(field, memForField, false, loopBody);
			i++;

			(*allJoinBindings)[currField] = memForField;
			LOG(INFO) << "[JOIN: ] Lhs Binding name: "<<currField;
		}

		//Forwarding/pipelining bindings of rhs too
		const std::map<string, AllocaInst*>& rhsBindings = childState.getBindings();
		for(std::map<string, AllocaInst*>::const_iterator it = rhsBindings.begin(); it!= rhsBindings.end(); ++it) {
			LOG(INFO) << "[JOIN: ] Rhs Binding name: "<<it->first;
			(*allJoinBindings).insert(*it);
		}
		LOG(INFO) << "[JOIN: ] Number of all join bindings: "<<allJoinBindings->size();

		//TRIGGER PARENT
		TheBuilder->SetInsertPoint(loopBody);
		OperatorState* newState = new OperatorState(*this, *allJoinBindings);
		getParent()->consume(context, *newState);

		TheBuilder->CreateBr(loopInc);

		// Block for.inc (label_for_inc)
		LoadInst* cnt_curr = new LoadInst(ptr_i, "", false, loopInc);
		cnt_curr->setAlignment(4);
		BinaryOperator* cnt_inc1 = BinaryOperator::Create(Instruction::Add, cnt_curr, context->createInt32(1), "i_inc", loopInc);
		StoreInst* store_cnt = new StoreInst(cnt_inc1, ptr_i, false, loopInc);
		store_cnt->setAlignment(4);

		BranchInst::Create(loopCond, loopInc);

		TheBuilder->SetInsertPoint(loopEnd);
	}
};



