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

#include "plugins/binary-row-plugin.hpp"

BinaryRowPlugin::BinaryRowPlugin(RawContext* const context, string& fname, RecordType& rec, vector<RecordAttribute*>& whichFields)
	: fname(fname), rec(rec), wantedFields(whichFields), context(context), posVar("offset"), bufVar("buf"), fsizeVar("fileSize") {

	fd = -1;
	buf = NULL;

	LOG(INFO) << "[BinaryRowPlugin: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;

	fd = open(name_c, O_RDONLY);
	if (fd == -1) {
		throw runtime_error(string("binary-row.open"));
	}
}

BinaryRowPlugin::~BinaryRowPlugin() {}

void BinaryRowPlugin::init()	{

	buf = (char*) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
	if (buf == MAP_FAILED) {
		throw runtime_error(string("csv.mmap"));
	}

	//Preparing the codegen part

	//(Can probably wrap some of these calls in one function)
	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Allocating memory
	AllocaInst *offsetMem = context->CreateEntryBlockAlloca(F,std::string(posVar),Type::getInt64Ty(llvmContext));
	AllocaInst *bufMem = context->CreateEntryBlockAlloca(F,std::string(bufVar),charPtrType);
	AllocaInst *fsizeMem = context->CreateEntryBlockAlloca(F,std::string(fsizeVar),Type::getInt64Ty(llvmContext));
	Value* offsetVal = Builder->getInt64(0);
	Builder->CreateStore(offsetVal,offsetMem);
	NamedValuesBinaryRow[posVar] = offsetMem;

	Value* fsizeVal = Builder->getInt64(fsize);
	Builder->CreateStore(fsizeVal,fsizeMem);
	NamedValuesBinaryRow[fsizeVar] = fsizeMem;

	//Typical way to pass a pointer via the LLVM API
	AllocaInst *AllocaPtr = context->CreateEntryBlockAlloca(F,std::string("charPtr"),charPtrType);
	Value* ptrVal = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* unshiftedPtr = Builder->CreateIntToPtr(ptrVal,charPtrType);
	Builder->CreateStore(unshiftedPtr,bufMem);
	NamedValuesBinaryRow[bufVar] = bufMem;
};

void BinaryRowPlugin::generate(const RawOperator &producer) {
	return scan(producer, context->getGlobalFunction());
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory BinaryRowPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar)	{
	RawValueMemory mem_projection;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
		RecordAttribute tmpKey = RecordAttribute(fname,pathVar);
		map<RecordAttribute, RawValueMemory>::const_iterator it;
		it = binProjections.find(tmpKey);
			if (it == binProjections.end()) {
				string error_msg = string("[Binary Row plugin - readPath ]: Unknown variable name ")+pathVar;
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		mem_projection = it->second;
	}
	return mem_projection;
}

RawValueMemory BinaryRowPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type)	{
	return mem_value;
}

RawValue BinaryRowPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type)	{
	IRBuilder<>* Builder = context->getBuilder();
	RawValue value;
	value.isNull = mem_value.isNull;
	value.value = Builder->CreateLoad(mem_value.mem);
	return value;
}

void BinaryRowPlugin::finish()	{
	close(fd);
	munmap(buf,fsize);
}

void BinaryRowPlugin::skipLLVM(Value* offset)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

#ifdef DEBUG
//		std::vector<Value*> ArgsV;
//		ArgsV.clear();
//		ArgsV.push_back(offset);
//		Function* debugInt = context->getFunction("printi64");
//		Builder->CreateCall(debugInt, ArgsV, "printi64");
#endif

	//Fetch values from symbol table
	AllocaInst* mem_pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		mem_pos = it->second;
	}

	//Increment and store back
	Value* val_curr_pos = Builder->CreateLoad(mem_pos);
	Value* val_new_pos = Builder->CreateAdd(val_curr_pos,offset);
	Builder->CreateStore(val_new_pos,mem_pos);
}

void BinaryRowPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(bufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int32);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYROW - READ INT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryRowPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(bufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
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

	LOG(INFO) << "[BINARYROW - READ STRING: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_strObj;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryRowPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int1Type = Type::getInt1Ty(llvmContext);
	PointerType* ptrType_bool = PointerType::get(int1Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(bufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_bool);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int1Type);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYROW - READ BOOL: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryRowPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* doubleType = Type::getDoubleTy(llvmContext);
	PointerType* ptrType_double = PointerType::get(doubleType, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(bufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + bufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_double);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYROW - READ FLOAT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryRowPlugin::scan(const RawOperator& producer, Function *f)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Container for the variable bindings
	map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();

	//Fetch value from symbol table
	AllocaInst* pos;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(posVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + posVar);
		}
		pos = it->second;
	}
	AllocaInst* fsizePtr;
	{
		std::map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(fsizeVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + fsizeVar);
		}
		fsizePtr = it->second;
	}

	//Get the ENTRY BLOCK
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", TheFunction);

	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(CondBB);
	// Start insertion in CondBB.
	Builder->SetInsertPoint(CondBB);

	/**
	 * Equivalent:
	 * while(pos < fsize)
	 */
	Value* lhs = Builder->CreateLoad(pos);
	Value* rhs = Builder->CreateLoad(fsizePtr);
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

	//Get the starting position of each record and pass it along.
	//More general/lazy plugins will only perform this action,
	//instead of eagerly 'converting' fields
	RecordAttribute tupleIdentifier = RecordAttribute(fname,activeLoop);

	RawValueMemory mem_posWrapper;
	mem_posWrapper.mem = pos;
	mem_posWrapper.isNull = context->createFalse();
	(*variableBindings)[tupleIdentifier] = mem_posWrapper;

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
	list<RecordAttribute*>& args = rec.getArgs();
	list<RecordAttribute*>::iterator iterSchema = args.begin();
	for (vector<RecordAttribute*>::iterator it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		RecordAttribute attr = *(*it);

		for (; (*iterSchema)->getAttrNo() < attr.getAttrNo(); iterSchema++) {
			switch ((*iterSchema)->getOriginalType()->getTypeID()) {
			case BOOL:
				offset += sizeof(bool);
				break;
			case STRING: {
				offset += 5 * sizeof(char);
				break;
			}
			case FLOAT:
				offset += sizeof(float);
				break;
			case INT:
				offset += sizeof(int);
				break;
			case BAG:
			case LIST:
			case SET:
				LOG(ERROR)<< "[BINARY ROW PLUGIN: ] Binary row files do not contain collections";
				throw runtime_error(string("[BINARY ROW PLUGIN: ] Binary row files do not contain collections"));
				case RECORD:
				LOG(ERROR) << "[BINARY ROW PLUGIN: ] Binary row files do not contain record-valued attributes";
				throw runtime_error(string("[BINARY ROW PLUGIN: ] Binary row files do not contain record-valued attributes"));
				default:
				LOG(ERROR) << "[BINARY ROW PLUGIN: ] Unknown datatype";
				throw runtime_error(string("[BINARY ROW PLUGIN: ] Unknown datatype"));
			}
		}

		//Move to appropriate position for reading
		Value* val_offset = context->createInt64(offset);
		skipLLVM(val_offset);

		//Read wanted field
		switch ((*it)->getOriginalType()->getTypeID()) {
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
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR)<< "[BINARY ROW PLUGIN: ] Binary row files do not contain collections";
			throw runtime_error(string("[BINARY ROW PLUGIN: ] Binary row files do not contain collections"));
		case RECORD:
			LOG(ERROR) << "[BINARY ROW PLUGIN: ] Binary row files do not contain record-valued attributes";
			throw runtime_error(string("[BINARY ROW PLUGIN: ] Binary row files do not contain record-valued attributes"));
		default:
			LOG(ERROR) << "[BINARY ROW PLUGIN: ] Unknown datatype";
			throw runtime_error(string("[BINARY ROW PLUGIN: ] Unknown datatype"));
		}
		iterSchema++;
	}

	for (; iterSchema != args.end(); iterSchema++) {
		switch ((*iterSchema)->getOriginalType()->getTypeID()) {
		case BOOL:
			offset += sizeof(bool);
			break;
		case STRING: {
			offset += 5 * sizeof(char);
			break;
		}
		case FLOAT:
			offset += sizeof(float);
			break;
		case INT:
			offset += sizeof(int);
			break;
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR)<< "[BINARY ROW PLUGIN: ] Binary row files do not contain collections";
			throw runtime_error(string("[CSV PLUGIN: ] Binary row files do not contain collections"));
			case RECORD:
			LOG(ERROR) << "[BINARY ROW PLUGIN: ] Binary row files do not contain record-valued attributes";
			throw runtime_error(string("[CSV PLUGIN: ] Binary row files do not contain record-valued attributes"));
			default:
			LOG(ERROR) << "[BINARY ROW PLUGIN: ] Unknown datatype";
			throw runtime_error(string("[BINARY ROW PLUGIN: ] Unknown datatype"));
		}
	}

	if(offset)	{
		//Move to appropriate position for start of 'newline'
		Value* val_offset = context->createInt64(offset);
		skipLLVM(val_offset);
	}

	// Make the new basic block for the increment, inserting after current block.
	BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

	// Insert an explicit fall through from the current (body) block to IncBB.
	Builder->CreateBr(IncBB);
	// Start insertion in IncBB.
	Builder->SetInsertPoint(IncBB);


	//Triggering parent
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context,*state);

	Builder->CreateBr(CondBB);

	//	Finish up with end (the AfterLoop)
	// 	Any new code will be inserted in AfterBB.
	Builder->SetInsertPoint(AfterBB);
}
