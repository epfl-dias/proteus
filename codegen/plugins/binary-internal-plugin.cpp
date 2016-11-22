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

#include "plugins/binary-internal-plugin.hpp"

BinaryInternalPlugin::BinaryInternalPlugin(RawContext* const context,
		string structName) :
		context(context), structName(structName)
{
	val_entriesNo = context->createInt64(0);
	mem_cnt = NULL;
	mem_buffer = NULL;
	mem_pos = NULL;
	val_structBufferPtr = NULL;
	payloadType = NULL;
}

BinaryInternalPlugin::BinaryInternalPlugin(RawContext* const context,
		RecordType rec, string structName,
		vector<RecordAttribute*> wantedOIDs,
		vector<RecordAttribute*> wantedFields,
		CacheInfo info) :
		rec(rec), structName(structName), OIDs(wantedOIDs), fields(wantedFields), context(context) {

	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type *int64Type = Type::getInt64Ty(llvmContext);
	PointerType* charPtrType = Type::getInt8PtrTy(llvmContext);

	payloadType = context->ReproduceCustomStruct(info.objectTypes);
	PointerType* payloadPtrType = PointerType::get(payloadType,0);
	char *rawBuffer = *(info.payloadPtr);
	//cout << "HOW MANY CACHED ENTRIES? " << *(info.itemCount) << endl;
	val_entriesNo = context->createInt64(*(info.itemCount));
	int fieldsNumber = OIDs.size() + fields.size();
	if (fieldsNumber <= 0) {
		string error_msg = string(
				"[Binary Internal Plugin]: Invalid number of fields");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	mem_buffer = context->CreateEntryBlockAlloca(F,string("binBuffer"),charPtrType);
	Value *val_buffer = context->CastPtrToLlvmPtr(charPtrType,rawBuffer);
	Builder->CreateStore(val_buffer,mem_buffer);
	mem_pos = context->CreateEntryBlockAlloca(F,string("currPos"),int64Type);
	mem_cnt = context->CreateEntryBlockAlloca(F,string("currTupleNo"),int64Type);
	Value *val_zero = Builder->getInt64(0);
	Builder->CreateStore(val_zero, mem_pos);
	Builder->CreateStore(val_zero, mem_cnt);

	val_structBufferPtr = context->CastPtrToLlvmPtr(payloadPtrType,rawBuffer);
//	cout << "Internal Binary PG creation - " << info.objectTypes.size() << " fields" << endl;
//	payloadPtrType->dump();
//	cout << endl;
}

BinaryInternalPlugin::~BinaryInternalPlugin() {}

void BinaryInternalPlugin::init()	{};

void BinaryInternalPlugin::generate(const RawOperator &producer) {
	if(mem_pos == NULL || mem_buffer == NULL)
	{
		/* XXX Later on, populate this function to simplify Nest */
		string error_msg = string(
				"[BinaryInternalPlugin: ] Unexpected use of pg.");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	else
	{
		/* Triggered by radix atm */
		scanStruct(producer);
	}
}

void BinaryInternalPlugin::scanStruct(const RawOperator& producer)
{
	//cout << "Internal Binary PG scan (struct)" << endl;
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Container for the variable bindings
	map<RecordAttribute, RawValueMemory>* variableBindings =
			new map<RecordAttribute, RawValueMemory>();

	//Get the ENTRY BLOCK
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	context->setCurrentEntryBlock(Builder->GetInsertBlock());

	BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", TheFunction);

	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(CondBB);
	// Start insertion in CondBB.
	Builder->SetInsertPoint(CondBB);

	/**
	 * while(currItemNo < itemsNo)
	 */
	Value* lhs = Builder->CreateLoad(mem_cnt);
	Value* rhs = val_entriesNo;
	Value *cond = Builder->CreateICmpSLT(lhs,rhs);

	// Make the new basic block for the loop header (BODY), inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", TheFunction);

	// Create the "AFTER LOOP" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", TheFunction);
	context->setEndingBlock(AfterBB);

	// Insert the conditional branch into the end of CondBB.
	Builder->CreateCondBr(cond, LoopBB, AfterBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);

	Value *val_cacheShiftedPtr = context->getArrayElemMem(val_structBufferPtr,
			lhs);

	int posInStruct = 0;
	for (vector<RecordAttribute*>::iterator it = OIDs.begin(); it != OIDs.end();
			it++) {
		RecordAttribute attr = *(*it);
		Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
				posInStruct);

		AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(
				TheFunction, "currOID", val_cachedField->getType());
		Builder->CreateStore(val_cachedField,mem_currResult);
		RawValueMemory mem_valWrapper;
		mem_valWrapper.mem = mem_currResult;
		mem_valWrapper.isNull = context->createFalse();
		(*variableBindings)[attr] = mem_valWrapper;
		posInStruct++;
#ifdef DEBUGBINCACHE
		{
			Function* debugInt = context->getFunction("printi64");
			vector<Value*> ArgsV;
			ArgsV.push_back(val_cachedField);
			Builder->CreateCall(debugInt, ArgsV);
			ArgsV.clear();
		}
#endif
	}

	for (vector<RecordAttribute*>::iterator it = fields.begin(); it != fields.end();
			it++) {
		RecordAttribute attr = *(*it);
		Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
				posInStruct);
#ifdef DEBUGBINCACHE
		{
			Function* debugInt = context->getFunction("printi");
			vector<Value*> ArgsV;
			ArgsV.push_back(val_cachedField);
			Builder->CreateCall(debugInt, ArgsV);
			ArgsV.clear();
		}
#endif
		AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(
				TheFunction, "currResult", val_cachedField->getType());
		Builder->CreateStore(val_cachedField,mem_currResult);
		RawValueMemory mem_valWrapper;
		mem_valWrapper.mem = mem_currResult;
		mem_valWrapper.isNull = context->createFalse();
		(*variableBindings)[attr] = mem_valWrapper;
		posInStruct++;
	}
#ifdef DEBUGBINCACHE
		{
			Function* debugInt = context->getFunction("printi64");
			vector<Value*> ArgsV;
			ArgsV.push_back(Builder->getInt64(30003));
			Builder->CreateCall(debugInt, ArgsV);
			ArgsV.clear();
		}
#endif

	// Make the new basic block for the increment, inserting after current block.
	BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

	// Insert an explicit fall through from the current (body) block to IncBB.
	Builder->CreateBr(IncBB);
	// Start insertion in IncBB.
	Builder->SetInsertPoint(IncBB);
	Value *val_cnt = Builder->CreateLoad(mem_cnt);
	Value *val_1 = Builder->getInt64(1);
	val_cnt = Builder->CreateAdd(val_cnt,val_1);
	Builder->CreateStore(val_cnt,mem_cnt);

	//Triggering parent
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context,*state);

	Builder->CreateBr(CondBB);

	//	Finish up with end (the AfterLoop)
	// 	Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);
}

void BinaryInternalPlugin::scan(const RawOperator& producer)
{
	cout << "Internal Binary PG scan" << endl;
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Container for the variable bindings
	map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();

	//Get the ENTRY BLOCK
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	context->setCurrentEntryBlock(Builder->GetInsertBlock());

	BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", TheFunction);

	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(CondBB);
	// Start insertion in CondBB.
	Builder->SetInsertPoint(CondBB);

	/**
	 * Equivalent:
	 * while(pos < fsize)
	 */
	Value* lhs = Builder->CreateLoad(mem_cnt);
	Value* rhs = val_entriesNo;
	Value *cond = Builder->CreateICmpSLT(lhs,rhs);

	// Make the new basic block for the loop header (BODY), inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", TheFunction);

	// Create the "AFTER LOOP" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", TheFunction);
	context->setEndingBlock(AfterBB);

	// Insert the conditional branch into the end of CondBB.
	Builder->CreateCondBr(cond, LoopBB, AfterBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);
#ifdef DEBUG
	{
		Function* debugInt = context->getFunction("printi64");
		vector<Value*> ArgsV;
		Value *val_pos = Builder->CreateLoad(mem_pos);
		ArgsV.push_back(val_pos);
		Builder->CreateCall(debugInt, ArgsV);
		ArgsV.clear();
	}
#endif
//	//Get the starting position of each record and pass it along.
//	//More general/lazy plugins will only perform this action,
//	//instead of eagerly 'converting' fields
//	ExpressionType *oidType = new IntType();
//	RecordAttribute tupleIdentifier = RecordAttribute(structName,activeLoop,oidType);
//
//	RawValueMemory mem_posWrapper;
//	mem_posWrapper.mem = mem_pos;
//	mem_posWrapper.isNull = context->createFalse();
//	(*variableBindings)[tupleIdentifier] = mem_posWrapper;

	//	BYTECODE
	//	for.body:                                         ; preds = %for.cond
	//	  br label %for.inc

	//Actual Work (Loop through attributes etc.)
	int cur_col = 0;

	int lastFieldNo = -1;

	Function* debugChar 	= context->getFunction("printc");
	Function* debugInt 		= context->getFunction("printi");
	Function* debugFloat 	= context->getFunction("printFloat");

	size_t offset = 0;
	list<RecordAttribute*> args = rec.getArgs();
	list<RecordAttribute*>::iterator iterSchema = args.begin();
	/* XXX No skipping atm!! */
	for (vector<RecordAttribute*>::iterator it = OIDs.begin(); it != OIDs.end();
			it++) {
		RecordAttribute attr = *(*it);
		switch (attr.getOriginalType()->getTypeID()) {

		case BOOL:
			readAsBooleanLLVM(attr, *variableBindings);
			offset = sizeof(bool);
			break;
		case STRING:
			readAsStringLLVM(attr, *variableBindings);
			offset = 5;
			break;
		case FLOAT:
			readAsFloatLLVM(attr, *variableBindings);
			offset = sizeof(float);
			break;
		case INT:
			readAsIntLLVM(attr, *variableBindings);
			offset = sizeof(int);
			break;
		case INT64:
			readAsInt64LLVM(attr, *variableBindings);
			offset = sizeof(size_t);
			break;
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Binary row files do not contain collections";
			throw runtime_error(
					string(
							"[BinaryInternalPlugin: ] Binary row files do not contain collections"));
			case RECORD:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Binary row files do not contain record-valued attributes";
			throw runtime_error(
					string(
							"[BinaryInternalPlugin: ] Binary row files do not contain record-valued attributes"));
			default:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Unknown datatype";
			throw runtime_error(
					string("[BinaryInternalPlugin: ] Unknown datatype"));

		}
		Value* val_offset = context->createInt64(offset);
		skipLLVM(val_offset);
	}
	for (vector<RecordAttribute*>::iterator it = fields.begin();
			it != fields.end(); it++) {
		RecordAttribute attr = *(*it);
		switch (attr.getOriginalType()->getTypeID()) {

		case BOOL:
			readAsBooleanLLVM(attr, *variableBindings);
			offset = sizeof(bool);
			break;
		case STRING:
			readAsStringLLVM(attr, *variableBindings);
			offset = 5;
			break;
		case FLOAT:
			readAsFloatLLVM(attr, *variableBindings);
			offset = sizeof(float);
			break;
		case INT:
			readAsIntLLVM(attr, *variableBindings);
			offset = sizeof(int);
			break;
		case INT64:
			readAsInt64LLVM(attr, *variableBindings);
			offset = sizeof(size_t);
			break;
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Binary row files do not contain collections";
			throw runtime_error(
					string(
							"[BinaryInternalPlugin: ] Binary row files do not contain collections"));
			case RECORD:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Binary row files do not contain record-valued attributes";
			throw runtime_error(
					string(
							"[BinaryInternalPlugin: ] Binary row files do not contain record-valued attributes"));
			default:
			LOG(ERROR)<< "[BinaryInternalPlugin: ] Unknown datatype";
			throw runtime_error(
					string("[BinaryInternalPlugin: ] Unknown datatype"));

		}
		Value* val_offset = context->createInt64(offset);
		skipLLVM(val_offset);
	}


	// Make the new basic block for the increment, inserting after current block.
	BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

	// Insert an explicit fall through from the current (body) block to IncBB.
	Builder->CreateBr(IncBB);
	// Start insertion in IncBB.
	Builder->SetInsertPoint(IncBB);
	Value *val_cnt = Builder->CreateLoad(mem_cnt);
	Value *val_1 = Builder->getInt64(1);
	val_cnt = Builder->CreateAdd(val_cnt,val_1);
	Builder->CreateStore(val_cnt,mem_cnt);

	//Triggering parent
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context,*state);

	Builder->CreateBr(CondBB);

	//	Finish up with end (the AfterLoop)
	// 	Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);
}

RawValueMemory BinaryInternalPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr)	{
	RawValueMemory mem_valWrapper;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
		RecordAttribute tmpKey = RecordAttribute(structName,pathVar,this->getOIDType());
		map<RecordAttribute, RawValueMemory>::const_iterator it;
		it = binProjections.find(tmpKey);
			if (it == binProjections.end()) {
				string error_msg = string("[BinaryInternalPlugin - readPath ]: Unknown variable name ")+pathVar;
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		mem_valWrapper = it->second;
	}
	return mem_valWrapper;
}

RawValueMemory BinaryInternalPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type)	{
	return mem_value;
}

RawValue BinaryInternalPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type)	{
	IRBuilder<>* Builder = context->getBuilder();
	switch (type->getTypeID())
	{
	case BOOL:
	{
		Function *hashBoolean = context->getFunction("hashBoolean");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashBoolean,
				ArgsV, "hashBoolean");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case STRING:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] String datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] String datatypes not supported yet"));
	}
	case FLOAT:
	{
		Function *hashDouble = context->getFunction("hashDouble");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashDouble, ArgsV, "hashDouble");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case INT:
	{
		Function *hashInt = context->getFunction("hashInt");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashInt, ArgsV, "hashInt");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case BAG:
	case LIST:
	case SET:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Collection datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Collection datatypes not supported yet"));
	}
	case RECORD:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Record-valued datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Record-valued datatypes not supported yet"));
	}
	default:
	{
		LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
		throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
	}
}
}

RawValue BinaryInternalPlugin::hashValueEager(RawValue valWrapper,
		const ExpressionType* type) {
	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();
	Value *tmp = valWrapper.value;
	AllocaInst *mem_tmp = context->CreateEntryBlockAlloca(F, "mem_cachedToHash",
			tmp->getType());
	Builder->CreateStore(tmp, mem_tmp);
	RawValueMemory mem_tmpWrapper = { mem_tmp, valWrapper.isNull };
	return hashValue(mem_tmpWrapper, type);
}

void BinaryInternalPlugin::flushValue(RawValueMemory mem_value, const ExpressionType *type,
		Value* fileName)
{
	IRBuilder<>* Builder = context->getBuilder();
	Function *flushFunc;
	Value* val_attr = Builder->CreateLoad(mem_value.mem);
	switch (type->getTypeID())
	{
	case BOOL:
	{
		flushFunc = context->getFunction("flushBoolean");
		vector<Value*> ArgsV;
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return;
	}
	case STRING:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] String datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] String datatypes not supported yet"));
	}
	case FLOAT:
	{
		flushFunc = context->getFunction("flushDouble");
		vector<Value*> ArgsV;
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc,ArgsV);
		return;
	}
	case INT:
	{
		vector<Value*> ArgsV;
		flushFunc = context->getFunction("flushInt");
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc,ArgsV);
		return;
	}
	case BAG:
	case LIST:
	case SET:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Collection datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Collection datatypes not supported yet"));
	}
	case RECORD:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Record-valued datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Record-valued datatypes not supported yet"));
	}
	default:
	{
		LOG(ERROR) << "[BinaryInternalPlugin: ] Unknown datatype";
		throw runtime_error(string("[BinaryInternalPlugin: ] Unknown datatype"));
	}
}
}

void BinaryInternalPlugin::flushValueEager(RawValue valWrapper,
		const ExpressionType *type, Value* fileName) {
	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();
	Value *tmp = valWrapper.value;
	AllocaInst *mem_tmp = context->CreateEntryBlockAlloca(F, "mem_cachedToFlush",
			tmp->getType());
	Builder->CreateStore(tmp, mem_tmp);
	RawValueMemory mem_tmpWrapper = { mem_tmp, valWrapper.isNull };
	return flushValue(mem_tmpWrapper, type, fileName);
}

void BinaryInternalPlugin::finish()	{

}

Value* BinaryInternalPlugin::getValueSize(RawValueMemory mem_value,
		const ExpressionType* type) {
	switch (type->getTypeID()) {
	case BOOL:
	case INT:
	case FLOAT: {
		Type *explicitType = (mem_value.mem)->getAllocatedType();
		return ConstantExpr::getSizeOf(explicitType);
		//return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
	}
	case STRING: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Strings not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case BAG:
	case LIST:
	case SET: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Collections not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case RECORD: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Records not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	default: {
		string error_msg = string("[BinaryInternalPlugin]: Unknown datatype");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	}
}

void BinaryInternalPlugin::skipLLVM(Value* offset)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Increment and store back
	Value* val_curr_pos = Builder->CreateLoad(mem_pos);
	Value* val_new_pos = Builder->CreateAdd(val_curr_pos,offset);
	Builder->CreateStore(val_new_pos,mem_pos);
}

void BinaryInternalPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	Value *val_pos = Builder->CreateLoad(mem_pos);
	Value* bufPtr = Builder->CreateLoad(mem_buffer, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int32);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYCACHE - READ INT: ] Read Successful";
#ifdef DEBUGBINCACHE
	{
		Function* debugInt = context->getFunction("printi");
		vector<Value*> ArgsV;
		//		ArgsV.push_back(context->createInt32(-6));
		ArgsV.push_back(parsedInt);
		Builder->CreateCall(debugInt, ArgsV);

	}
#endif
	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryInternalPlugin::readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	Value *val_pos = Builder->CreateLoad(mem_pos);
	Value* bufPtr = Builder->CreateLoad(mem_buffer, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int64);
	Value *parsedInt = Builder->CreateLoad(mem_result);
#ifdef DEBUGBINCACHE
	{
		Function* debugInt = context->getFunction("printi64");
		vector<Value*> ArgsV;
		ArgsV.push_back(parsedInt);
		Builder->CreateCall(debugInt, ArgsV);
	}
#endif
	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int64Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYCACHE - READ INT64: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryInternalPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();

	Value *val_pos = Builder->CreateLoad(mem_pos);
	Value* bufPtr = Builder->CreateLoad(mem_buffer, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);

	StructType* strObjType = context->CreateStringStruct();
	AllocaInst* mem_strObj = context->CreateEntryBlockAlloca(F, "currResult",
				strObjType);

	//Populate string object
	Value *val_0 = context->createInt32(0);
	Value *val_1 = context->createInt32(1);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(val_0);
	idxList.push_back(val_0);
	Value* structPtr = Builder->CreateGEP(mem_strObj,idxList);
	Builder->CreateStore(bufShiftedPtr,structPtr);

	idxList.clear();
	idxList.push_back(val_0);
	idxList.push_back(val_1);
	structPtr = Builder->CreateGEP(mem_strObj,idxList);
	Builder->CreateStore(context->createInt32(5),structPtr);

	LOG(INFO) << "[BINARYCACHE - READ STRING: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_strObj;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryInternalPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int1Type = Type::getInt1Ty(llvmContext);
	PointerType* ptrType_bool = PointerType::get(int1Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	Value *val_pos = Builder->CreateLoad(mem_pos);
	Value* bufPtr = Builder->CreateLoad(mem_buffer, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_bool);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int1Type);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYCACHE - READ BOOL: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryInternalPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* doubleType = Type::getDoubleTy(llvmContext);
	PointerType* ptrType_double = PointerType::get(doubleType, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	Value *val_pos = Builder->CreateLoad(mem_pos);
	Value* bufPtr = Builder->CreateLoad(mem_buffer, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_double);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYCACHE - READ FLOAT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

