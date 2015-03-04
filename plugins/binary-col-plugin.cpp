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

#include "plugins/binary-col-plugin.hpp"

BinaryColPlugin::BinaryColPlugin(RawContext* const context, string& fnamePrefix, RecordType& rec, vector<RecordAttribute*>& whichFields)
	: fnamePrefix(fnamePrefix), rec(rec), wantedFields(whichFields), context(context),
	  posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size") {

	int fieldsNumber = wantedFields.size();
	if(fieldsNumber <= 0)	{
		string error_msg = string("[Binary Col Plugin]: Invalid number of fields");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}

	fd = (int*) malloc(fieldsNumber * sizeof(int));
	if(fd == NULL)	{
		string error_msg = string("[Binary Col Plugin]: Malloc Failed");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	buf = (char**) malloc(fieldsNumber * sizeof(char*));
	if(buf == NULL)	{
		string error_msg = string("[Binary Col Plugin]: Malloc Failed");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}

	colFilesize = (off_t*) malloc(fieldsNumber * sizeof(off_t));
	if (colFilesize == NULL)	{
		string error_msg = string("[Binary Col Plugin]: Malloc Failed");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	vector<RecordAttribute*>::iterator it;
	int cnt = 0;
	for(it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		string fileName = fnamePrefix + "." + (*it)->attrName;
		LOG(INFO) << "[BinaryColPlugin: ] " << fnamePrefix;

		struct stat statbuf;
		const char* name_c = fileName.c_str();
		stat(name_c, &statbuf);
		colFilesize[cnt] = statbuf.st_size;
		fd[cnt] = open(name_c, O_RDONLY);
		if (fd[cnt] == -1) {
			string error_msg = string("[Binary Col Plugin]: Opening column failed");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

		if((*it)->getOriginalType()->getTypeID() == STRING)	{
			//Initialize dictionary
			string dictionaryPath = fileName + ".dict";
			const char* dictionaryPath_c = dictionaryPath.c_str();
			int dictionaryFd = open(name_c, O_RDONLY);
			if(dictionaryFd == -1)	{
				string error_msg = string("[Binary Col Plugin]: Opening column dictionary failed");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
			dictionaries[cnt] = dictionaryFd;
			struct stat statbufDict;
			stat(dictionaryPath_c, &statbuf);
			dictionaryFilesizes[cnt] = statbufDict.st_size;

		}
		cnt++;
	}
//	dictionariesBuf = (char*) malloc(dictionaries.size() * sizeof(char*));
//	if(dictionariesBuf == NULL)	{
//		string error_msg = string("[Binary Col Plugin]: Malloc Failed");
//		LOG(ERROR) << error_msg;
//		throw runtime_error(error_msg);
//	}

}

BinaryColPlugin::~BinaryColPlugin() {

}


void BinaryColPlugin::init()	{

	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	vector<RecordAttribute*>::iterator it;
	int cnt = 0;
	for (it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		buf[cnt] = (char*) mmap(NULL, colFilesize[cnt], PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd[cnt], 0);
		if (buf[cnt] == MAP_FAILED )	{
			throw runtime_error(string("csv.mmap"));
		}

		//Allocating memory for each field / column involved
		string attrName = (*it)->getAttrName();
		string currPosVar = string(posVar) + "." + attrName();
		string currBufVar = string(bufVar) + "." + attrName();

		AllocaInst *offsetMem = context->CreateEntryBlockAlloca(F,currPosVar,Type::getInt64Ty(llvmContext));
		AllocaInst *bufMem = context->CreateEntryBlockAlloca(F,currBufVar,charPtrType);

		Value* offsetVal = Builder->getInt64(0);
		Builder->CreateStore(offsetVal,offsetMem);
		NamedValuesBinaryRow[currPosVar] = offsetMem;

		//Typical way to pass a pointer via the LLVM API
		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,string("mem_bufPtr"),charPtrType);
		Value* val_bufPtr = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf[cnt])));
		//i8*
		Value* unshiftedPtr = Builder->CreateIntToPtr(val_bufPtr,charPtrType);
		Builder->CreateStore(unshiftedPtr,bufMem);
		NamedValuesBinaryRow[currBufVar] = bufMem;

		if(wantedFields.at(cnt)->getOriginalType()->getTypeID() == STRING)	{
			char *dictBuf = (char*) mmap(NULL, colFilesize[cnt], PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd[cnt], 0);
			dictionariesBuf[cnt] = dictBuf;

			string currDictVar = string(bufVar) + "." + attrName + ".dict";
			AllocaInst *dictMem = context->CreateEntryBlockAlloca(F,currDictVar,charPtrType);

			//Typical way to pass a pointer via the LLVM API
			AllocaInst *mem_dictPtr = context->CreateEntryBlockAlloca(F,string("mem_dictPtr"),charPtrType);
			Value* val_dictPtr = ConstantInt::get(llvmContext, APInt(64,((uint64_t)dictionariesBuf[cnt])));
			//i8*
			Value* unshiftedPtr = Builder->CreateIntToPtr(val_dictPtr,charPtrType);
			Builder->CreateStore(unshiftedPtr,dictMem);
			NamedValuesBinaryRow[currDictVar] = bufMem;
		}
	}

};

void BinaryColPlugin::generate(const RawOperator &producer) {
	return scan(producer, context->getGlobalFunction());
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory BinaryColPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar)	{
	RawValueMemory mem_projection;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
		//TODO Make sure that using fnamePrefix in this search does not cause issues
		RecordAttribute tmpKey = RecordAttribute(fnamePrefix,pathVar);
		map<RecordAttribute, RawValueMemory>::const_iterator it;
		it = binProjections.find(tmpKey);
			if (it == binProjections.end()) {
				string error_msg = string("[Binary Col. plugin - readPath ]: Unknown variable name ")+pathVar;
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		mem_projection = it->second;
	}
	return mem_projection;
}

RawValueMemory BinaryColPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type)	{
	return mem_value;
}

RawValue BinaryColPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type)	{
	IRBuilder<>* Builder = context->getBuilder();
	RawValue value;
	value.isNull = mem_value.isNull;
	value.value = Builder->CreateLoad(mem_value.mem);
	return value;
}

void BinaryColPlugin::finish()	{
	vector<RecordAttribute*>::iterator it;
	int cnt = 0;
	for (it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		close(fd[cnt]);
		munmap(buf[cnt], colFilesize[cnt]);

		if ((*it)->getOriginalType()->getTypeID() == STRING)	{
			int dictionaryFd = dictionaries[cnt];
			close(dictionaryFd);
			munmap(dictionariesBuf[cnt], dictionaryFilesizes[cnt]);
		}
	}
}

void BinaryColPlugin::skipLLVM(string attName, Value* offset)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Fetch values from symbol table
	AllocaInst* mem_pos;
	{
		map<string, AllocaInst*>::iterator it;
		string currPosVar = posVar + "." + attName;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}

	//Increment and store back
	Value* val_curr_pos = Builder->CreateLoad(mem_pos);
	Value* val_new_pos = Builder->CreateAdd(val_curr_pos,offset);
	Builder->CreateStore(val_new_pos,mem_pos);
}

void BinaryColPlugin::readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string currPosVar = posVar + "." + attName;
	string currBufVar = bufVar + "." + attName;

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currBufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int32);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYCOL - READ INT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryColPlugin::readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string currPosVar = posVar + "." + attName;
	string currBufVar = bufVar + "." + attName;

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currBufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int64);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int64Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYCOL - READ INT64: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

Value* BinaryColPlugin::readAsInt64LLVM(RecordAttribute attName)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string currPosVar = posVar + "." + attName;
	string currBufVar = bufVar + "." + attName;

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currBufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_int64);
	Value *parsedInt64 = Builder->CreateLoad(mem_result);

	return parsedInt64;
}


//TODO TODO TODO Needs to use dictionary
void BinaryColPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	string error_msg = string("[Binary Col Plugin]: Strings not supported yet");
	LOG(ERROR) << error_msg;
	throw runtime_error(error_msg);

//	//Prepare
//	LLVMContext& llvmContext = context->getLLVMContext();
//	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//	Type* int32Type = Type::getInt32Ty(llvmContext);
//	Type* int64Type = Type::getInt64Ty(llvmContext);
//	PointerType* ptrType_int32 = PointerType::get(int32Type, 0);
//
//	IRBuilder<>* Builder = context->getBuilder();
//	Function *F = Builder->GetInsertBlock()->getParent();
//
//	//Fetch values from symbol table
//	AllocaInst *mem_pos;
//	{
//		std::map<std::string, AllocaInst*>::iterator it;
//		it = NamedValuesBinaryRow.find(posVar);
//		if (it == NamedValuesBinaryRow.end()) {
//			throw runtime_error(string("Unknown variable name: ") + posVar);
//		}
//		mem_pos = it->second;
//	}
//	Value *val_pos = Builder->CreateLoad(mem_pos);
//
//	AllocaInst* buf;
//	{
//		std::map<std::string, AllocaInst*>::iterator it;
//		it = NamedValuesBinaryRow.find(bufVar);
//		if (it == NamedValuesBinaryRow.end()) {
//			throw runtime_error(string("Unknown variable name: ") + bufVar);
//		}
//		buf = it->second;
//	}
//	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
//	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
//
//	StructType* strObjType = context->CreateStringStruct();
//	AllocaInst* mem_strObj = context->CreateEntryBlockAlloca(F, "currResult",
//				strObjType);
//
//	//Populate string object
//	Value *val_0 = context->createInt32(0);
//	Value *val_1 = context->createInt32(1);
//
//	vector<Value*> idxList = vector<Value*>();
//	idxList.push_back(val_0);
//	idxList.push_back(val_0);
//	Value* structPtr = Builder->CreateGEP(mem_strObj,idxList);
//	Builder->CreateStore(bufShiftedPtr,structPtr);
//
//	idxList.clear();
//	idxList.push_back(val_0);
//	idxList.push_back(val_1);
//	structPtr = Builder->CreateGEP(mem_strObj,idxList);
//	Builder->CreateStore(context->createInt32(5),structPtr);
//
//	LOG(INFO) << "[BINARYROW - READ STRING: ] Read Successful";
//
//	RawValueMemory mem_valWrapper;
//	mem_valWrapper.mem = mem_strObj;
//	mem_valWrapper.isNull = context->createFalse();
//	variables[attName] = mem_valWrapper;
}

void BinaryColPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int1Type = Type::getInt1Ty(llvmContext);
	PointerType* ptrType_bool = PointerType::get(int1Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string currPosVar = posVar + "." + attName;
	string currBufVar = bufVar + "." + attName;

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currBufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_bool);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int1Type);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYCOL - READ BOOL: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryColPlugin::readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* doubleType = Type::getDoubleTy(llvmContext);
	PointerType* ptrType_double = PointerType::get(doubleType, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string currPosVar = posVar + "." + attName;
	string currBufVar = bufVar + "." + attName;

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currPosVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryRow.find(currBufVar);
		if (it == NamedValuesBinaryRow.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value* mem_result = Builder->CreateBitCast(bufShiftedPtr,ptrType_double);
	Value *parsedInt = Builder->CreateLoad(mem_result);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
	Builder->CreateStore(parsedInt,currResult);
	LOG(INFO) << "[BINARYCOL - READ FLOAT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryColPlugin::scan(const RawOperator& producer, Function *f)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);

	Value* val_itemCtr = context->createInt64(0);
	AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(TheFunction, "itemCtr", int64Type);
	Builder->CreateStore(val_itemCtr,mem_itemCtr);

	vector<RecordAttribute*>::iterator it;
	Value* val_size = NULL;
	//Reminder: Every column starts with an entry of its size
	for(it = wantedFields.begin(); it != wantedFields.end(); it++)	{

		string currBufVar = bufVar + "." + (*it)->getAttrName();
		RecordAttribute *attr = *it;

		if(it == wantedFields.begin())	{
			val_size = readAsInt64LLVM(*attr);
		}

		//Move all buffer pointers to the actual data
		Value* val_offset = context->createInt64(sizeof(size_t));
		skipLLVM((*it)->getAttrName(), val_offset);
	}

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
	 * while(itemCtr < size)
	 */
	Value* lhs = Builder->CreateLoad(mem_itemCtr);
	Value* rhs = Builder->CreateLoad(val_size);
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

	//Get the 'oid' of each record and pass it along.
	//More general/lazy plugins will only perform this action,
	//instead of eagerly 'converting' fields
	//FIXME This action corresponds to materializing the oid. Do we want this?
	RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix,activeLoop);

	RawValueMemory mem_posWrapper;
	mem_posWrapper.mem = mem_itemCtr;
	mem_posWrapper.isNull = context->createFalse();
	(*variableBindings)[tupleIdentifier] = mem_posWrapper;

	//FIXME FIXME FIXME Actual Work (Loop through attributes etc.)
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
		//skipLLVM(val_offset);

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
		//skipLLVM(val_offset);
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
