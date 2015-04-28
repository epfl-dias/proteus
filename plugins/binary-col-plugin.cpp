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

BinaryColPlugin::BinaryColPlugin(RawContext* const context, string fnamePrefix, RecordType rec, vector<RecordAttribute*>& whichFields)
	: fnamePrefix(fnamePrefix), rec(rec), wantedFields(whichFields), context(context),
	  posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr") {

	isCached = false;
	val_size = NULL;
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
	LOG(INFO) << "[BinaryColPlugin: ] " << fnamePrefix;
	for(it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		string fileName = fnamePrefix + "." + (*it)->getAttrName();

		struct stat statbuf;
		const char* name_c = fileName.c_str();
		stat(name_c, &statbuf);
		colFilesize[cnt] = statbuf.st_size;
		fd[cnt] = open(name_c, O_RDONLY);
		if (fd[cnt] == -1) {
			string error_msg = string("[Binary Col Plugin]: Opening column failed -> "+fileName);
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

/* No STRING yet in this mode */
//BinaryColPlugin::BinaryColPlugin(RawContext* const context, vector<RecordAttribute*>& whichFields, vector<CacheInfo> whichCaches)
//	: rec(rec), wantedFields(whichFields), whichCaches(whichCaches), context(context), fnamePrefix(""),
//	  posVar("offset"), bufVar("buf"), fsizeVar("fileSize"), sizeVar("size"), itemCtrVar("itemCtr") {
//
//	isCached = true;
//	val_size = NULL;
//	int fieldsNumber = wantedFields.size();
//	if(fieldsNumber <= 0)	{
//		string error_msg = string("[Binary Col Plugin]: Invalid number of fields");
//		LOG(ERROR) << error_msg;
//		throw runtime_error(error_msg);
//	}
//
//	if (whichFields.size() != whichCaches.size()) {
//		string error_msg = string(
//				"[Binary Col Plugin]: Failed attempt to use caches");
//		LOG(ERROR)<< error_msg;
//		throw runtime_error(error_msg);
//	}
//	vector<RecordAttribute*>::iterator it;
//	vector<CacheInfo>::iterator itCaches;
//	LOG(INFO) << "[BinaryColPlugin *as cache*]";
//}

BinaryColPlugin::~BinaryColPlugin() {}

/* No STRING yet in this mode */
/* The constructor of a previous colPlugin has taken care of the populated buffers we need,
 * the val. size, etc. */
//void BinaryColPlugin::initCached()	{
//
//	Function* F = context->getGlobalFunction();
//	LLVMContext& llvmContext = context->getLLVMContext();
//	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//	Type* int64Type = Type::getInt64Ty(llvmContext);
//	IRBuilder<>* Builder = context->getBuilder();
//
//	int size = *(whichCaches.at(0).itemCount);
//	if(size <= 0)	{
//		string error_msg = string(
//				"[Binary Col Plugin]: Failed attempt to use caches");
//		LOG(ERROR)<< error_msg;
//		throw runtime_error(error_msg);
//	}
//	else {
//		val_size = context->createInt32(size);
//	}
//
//	vector<CacheInfo>::iterator itCaches;
//	vector<RecordAttribute*>::iterator it;
//	int cnt = 0;
//	for (it = wantedFields.begin(); it != wantedFields.end(); it++,itCaches++)	{
//		CacheInfo info = *itCaches;
//		buf[cnt] = *info.payloadPtr;
//
//		RecordAttribute *attr = *it;
//		//Allocating memory for each field / column involved
//		string attrName = attr->getAttrName();
//		string currPosVar = string(posVar) + "." + attrName;
//		string currBufVar = string(bufVar) + "." + attrName;
//
//		AllocaInst *offsetMem = context->CreateEntryBlockAlloca(F,currPosVar,int64Type);
//		AllocaInst *bufMem = context->CreateEntryBlockAlloca(F,currBufVar,charPtrType);
//
//		Value* offsetVal = Builder->getInt64(0);
//		Builder->CreateStore(offsetVal,offsetMem);
//		NamedValuesBinaryCol[currPosVar] = offsetMem;
//
//		//Typical way to pass a pointer via the LLVM API
//		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,string("mem_bufPtr"),charPtrType);
//		Value* val_bufPtr = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf[cnt])));
//		//i8*
//		Value* unshiftedPtr = Builder->CreateIntToPtr(val_bufPtr,charPtrType);
//		Builder->CreateStore(unshiftedPtr,bufMem);
//		NamedValuesBinaryCol[currBufVar] = bufMem;
//
//		cnt++;
//	}
//	//Global item counter
//	Value* val_itemCtr = context->createInt64(0);
//	AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(F, itemCtrVar, int64Type);
//	Builder->CreateStore(val_itemCtr,mem_itemCtr);
//	NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;
//}


void BinaryColPlugin::init()	{

	Function* F = context->getGlobalFunction();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	PointerType* int64PtrType = Type::getInt64PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	/* XXX Very silly conversion */
	list<RecordAttribute*>::iterator attrIter = rec.getArgs().begin();
	list<RecordAttribute> attrList;
	RecordAttribute projTuple = RecordAttribute(fnamePrefix, activeLoop,
			this->getOIDType());
	attrList.push_back(projTuple);
	for (vector<RecordAttribute*>::iterator it = wantedFields.begin();
			it != wantedFields.end(); it++) {
		attrList.push_back(*(*it));
	}
	expressions::InputArgument arg = expressions::InputArgument(&rec, 0,
			attrList);
	/*******/

	vector<RecordAttribute*>::iterator it;
	int cnt = 0;
	for (it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		buf[cnt] = (char*) mmap(NULL, colFilesize[cnt], PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd[cnt], 0);
		if (buf[cnt] == MAP_FAILED )	{
			throw runtime_error(string("csv.mmap"));
		}

		RecordAttribute *attr = *it;
		//Allocating memory for each field / column involved
		string attrName = attr->getAttrName();
		string currPosVar = string(posVar) + "." + attrName;
		string currBufVar = string(bufVar) + "." + attrName;

		AllocaInst *offsetMem = context->CreateEntryBlockAlloca(F,currPosVar,int64Type);
		AllocaInst *bufMem = context->CreateEntryBlockAlloca(F,currBufVar,charPtrType);

		Value* offsetVal = Builder->getInt64(0);
		Builder->CreateStore(offsetVal,offsetMem);
		NamedValuesBinaryCol[currPosVar] = offsetMem;

		//Typical way to pass a pointer via the LLVM API
		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,string("mem_bufPtr"),charPtrType);
		Value* val_bufPtr = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf[cnt])));
		//i8*
		Value* unshiftedPtr = Builder->CreateIntToPtr(val_bufPtr,charPtrType);
		Builder->CreateStore(unshiftedPtr,bufMem);
		NamedValuesBinaryCol[currBufVar] = bufMem;

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
			NamedValuesBinaryCol[currDictVar] = bufMem;
		}
		cnt++;

		/* Deal with preparation of input arrays too */
		string bufVarStr = string(bufVar);
		//Reminder: Every column starts with an entry of its size



		if (it == wantedFields.begin()) {
			val_size = readAsInt64LLVM(*attr);
#ifdef DEBUG
			vector<Value*> ArgsV;
			ArgsV.push_back(val_size);
			Function* debugInt = context->getFunction("printi64");
			Builder->CreateCall(debugInt, ArgsV, "printi64");
			Builder->CreateCall(debugInt, ArgsV, "printi64");
#endif
		}

		/* Move all buffer pointers to the actual data
		 * and cast appropriately
		 */
		prepareArray(*attr);

		/* What is the point of caching what is already converted and compact? */
//		{
//			/* Make columns available to caching service!
//			 * NOTE: Not 1-1 applicable in radix case
//			 * Reason: OID not explicitly materialized */
//			const ExpressionType *fieldType = (*it)->getOriginalType();
//			const RecordAttribute& thisAttr = *(*it);
//			expressions::Expression* thisField =
//					new expressions::RecordProjection(fieldType, &arg,
//							thisAttr);
//
//			/* Place info about col. in cache! */
//			cout << "[Binary Col. Plugin:] Register (bin col.) in cache" << endl;
//			CachingService& cache = CachingService::getInstance();
//			bool fullRelation = true;
//
//			CacheInfo info;
//			info.objectTypes.push_back(attr->getOriginalType()->getTypeID());
//			info.structFieldNo = 0;
//			//Skipping the offset that contains the size of the column!
//			char* ptr_rawBuffer = buf[cnt] + sizeof(size_t);
//			info.payloadPtr = &ptr_rawBuffer;
//			//MUST fill this up!
//			info.itemCount = new size_t[1];
//			Value *mem_itemCount = context->CastPtrToLlvmPtr(int64PtrType,
//					(void*) info.itemCount);
//			Builder->CreateStore(val_size, mem_itemCount);
//			cache.registerCache(thisField, info, fullRelation);
//		}

	}
	//cout << "[BinaryColPlugin: ] Initialization Successful for " << fnamePrefix << endl;
	//Global item counter
	Value* val_itemCtr = context->createInt64(0);
	AllocaInst *mem_itemCtr = context->CreateEntryBlockAlloca(F, itemCtrVar, int64Type);
	Builder->CreateStore(val_itemCtr,mem_itemCtr);
	NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;
}

void BinaryColPlugin::generate(const RawOperator &producer) {
	return scan(producer);
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory BinaryColPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar)	{
	RawValueMemory mem_projection;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
		//XXX Make sure that using fnamePrefix in this search does not cause issues
		RecordAttribute tmpKey = RecordAttribute(fnamePrefix,pathVar,this->getOIDType());
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

/* FIXME Differentiate between operations that need the code and the ones needing the materialized string */
RawValueMemory BinaryColPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type)	{
	return mem_value;
}

//RawValue BinaryColPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type)	{
//	IRBuilder<>* Builder = context->getBuilder();
//	RawValue value;
//	value.isNull = mem_value.isNull;
//	value.value = Builder->CreateLoad(mem_value.mem);
//	return value;
//}
RawValue BinaryColPlugin::hashValue(RawValueMemory mem_value,
		const ExpressionType* type) {
	IRBuilder<>* Builder = context->getBuilder();
	switch (type->getTypeID()) {
	case BOOL: {
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
	case STRING: {
		LOG(ERROR)<< "[CSV PLUGIN: ] String datatypes not supported yet";
		throw runtime_error(string("[CSV PLUGIN: ] String datatypes not supported yet"));
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
	LOG(ERROR) << "[BinaryColPlugin: ] Cannot contain collections";
	throw runtime_error(string("[BinaryColPlugin: ] Cannot contain collections"));
	case RECORD:
	LOG(ERROR) << "[BinaryColPlugin: ] Cannot contain record-valued attributes";
	throw runtime_error(string("[BinaryColPlugin: ] Cannot contain record-valued attributes"));
	default:
	LOG(ERROR) << "[BinaryColPlugin: ] Unknown datatype";
	throw runtime_error(string("[BinaryColPlugin: ] Unknown datatype"));
}
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
		cnt++;
	}
}

Value* BinaryColPlugin::getValueSize(RawValueMemory mem_value,
		const ExpressionType* type) {
	switch (type->getTypeID()) {
	case BOOL:
	case INT:
	case FLOAT: {
		Type *explicitType = (mem_value.mem)->getAllocatedType();
		return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
	}
	case STRING: {
		string error_msg = string(
				"[Binary Col Plugin]: Strings not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case BAG:
	case LIST:
	case SET: {
		string error_msg = string(
				"[Binary Col Plugin]: Cannot contain collections");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case RECORD: {
		string error_msg = string(
				"[Binary Col Plugin]: Cannot contain records");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	default: {
		string error_msg = string("[Binary Col Plugin]: Unknown datatype");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	}
}

void BinaryColPlugin::skipLLVM(RecordAttribute attName, Value* offset)
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
		string posVarStr = string(posVar);
		string currPosVar = posVarStr + "." + attName.getAttrName();
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}

	//Increment and store back
	Value* val_curr_pos = Builder->CreateLoad(mem_pos);
	Value* val_new_pos = Builder->CreateAdd(val_curr_pos,offset);
	Builder->CreateStore(val_new_pos,mem_pos);

}

void BinaryColPlugin::nextEntry()	{

	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Necessary because it's the itemCtr that affects the scan loop
	AllocaInst* mem_itemCtr;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(itemCtrVar);
		if (it == NamedValuesBinaryCol.end())
		{
			throw runtime_error(string("Unknown variable name: ") + itemCtrVar);
		}
		mem_itemCtr = it->second;
	}

	//Increment and store back
	Value* val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);
	Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr,
			context->createInt64(1));
	Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

/* Operates over int*! */
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


	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value *parsedInt = Builder->CreateLoad(bufShiftedPtr);

	AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
	Builder->CreateStore(parsedInt,mem_currResult);
	LOG(INFO) << "[BINARYCOL - READ INT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;

#ifdef DEBUGBINCOL
//		vector<Value*> ArgsV;
//		ArgsV.clear();
//		ArgsV.push_back(parsedInt);
//		Function* debugInt = context->getFunction("printi");
//		Builder->CreateCall(debugInt, ArgsV, "printi");
#endif
}

/* Operates over char*! */
void BinaryColPlugin::readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
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

/* Operates over char*! */
Value* BinaryColPlugin::readAsInt64LLVM(RecordAttribute attName)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptrType_int64 = PointerType::get(int64Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
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


/*
 * FIXME Needs to be aware of dictionary (?).
 * Probably readValue() is the appropriate place for this.
 * I think forwarding the dict. code (int32) is sufficient here.
 */
void BinaryColPlugin::readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	readAsIntLLVM(attName, variables);
}

void BinaryColPlugin::readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int1Type = Type::getInt1Ty(llvmContext);
	PointerType* ptrType_bool = PointerType::get(int1Type, 0);

	IRBuilder<>* Builder = context->getBuilder();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<std::string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value *parsedInt = Builder->CreateLoad(bufShiftedPtr);

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

	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	//Fetch values from symbol table
	AllocaInst *mem_pos;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}
	Value *val_pos = Builder->CreateLoad(mem_pos);

	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
	Value *parsedFloat = Builder->CreateLoad(bufShiftedPtr);

	AllocaInst *currResult = context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
	Builder->CreateStore(parsedFloat,currResult);
	LOG(INFO) << "[BINARYCOL - READ FLOAT: ] Read Successful";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = currResult;
	mem_valWrapper.isNull = context->createFalse();
	variables[attName] = mem_valWrapper;
}

void BinaryColPlugin::prepareArray(RecordAttribute attName)	{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//	Type* floatPtrType = Type::getFloatPtrTy(llvmContext);
	Type* doublePtrType = Type::getDoublePtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32PtrType = Type::getInt32PtrTy(llvmContext);
	Type* int8PtrType = Type::getInt8PtrTy(llvmContext);

	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();

	string posVarStr = string(posVar);
	string currPosVar = posVarStr + "." + attName.getAttrName();
	string bufVarStr = string(bufVar);
	string currBufVar = bufVarStr + "." + attName.getAttrName();

	/* Code equivalent to skip(size_t) */
	Value* val_offset = context->createInt64(sizeof(size_t));
	AllocaInst* mem_pos;
	{
		map<string, AllocaInst*>::iterator it;
		string posVarStr = string(posVar);
		string currPosVar = posVarStr + "." + attName.getAttrName();
		it = NamedValuesBinaryCol.find(currPosVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currPosVar);
		}
		mem_pos = it->second;
	}

	//Increment and store back
	Value* val_curr_pos = Builder->CreateLoad(mem_pos);
	Value* val_new_pos = Builder->CreateAdd(val_curr_pos,val_offset);
	/* Not storing this 'offset' - we want the cast buffer to
	 * conceptually start from 0 */
	//	Builder->CreateStore(val_new_pos,mem_pos);

	/* Get relevant char* rawBuf */
	AllocaInst* buf;
	{
		map<string, AllocaInst*>::iterator it;
		it = NamedValuesBinaryCol.find(currBufVar);
		if (it == NamedValuesBinaryCol.end()) {
			throw runtime_error(string("Unknown variable name: ") + currBufVar);
		}
		buf = it->second;
	}
	Value* bufPtr = Builder->CreateLoad(buf, "bufPtr");
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_new_pos);

	/* Cast to appropriate form */
	typeID id = attName.getOriginalType()->getTypeID();
	switch (id) {
	case BOOL: {
		//No need to do sth - char* and int8* are interchangeable
		break;
	}
	case FLOAT: {
		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
				string("mem_bufPtr"), doublePtrType);
		Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, doublePtrType);
		Builder->CreateStore(val_bufPtr, mem_bufPtr);
		NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
		break;
	}
	case INT: {
		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
				string("mem_bufPtr"), int32PtrType);
		Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, int32PtrType);
		Builder->CreateStore(val_bufPtr, mem_bufPtr);
		NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
		break;
	}
	case STRING: {
		/* String representation comprises the code and the dictionary
		 * Codes are (will be) int32, so can again treat like int32* */
		AllocaInst *mem_bufPtr = context->CreateEntryBlockAlloca(F,
				string("mem_bufPtr"), int32PtrType);
		Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, int32PtrType);
		Builder->CreateStore(val_bufPtr, mem_bufPtr);
		NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
		break;
	}
	case RECORD:
	case LIST:
	case BAG:
	case SET:
	default: {
		string error_msg = string("[Binary Col PG: ] Unsupported Record");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
}

void BinaryColPlugin::scan(const RawOperator& producer)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<>* Builder = context->getBuilder();
	Function *F = Builder->GetInsertBlock()->getParent();

	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);

	//Container for the variable bindings
	map<RecordAttribute, RawValueMemory>* variableBindings = new map<RecordAttribute, RawValueMemory>();

	//Get the ENTRY BLOCK
	context->setCurrentEntryBlock(Builder->GetInsertBlock());

	BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

	// Insert an explicit fall through from the current (entry) block to the CondBB.
	Builder->CreateBr(CondBB);
	// Start insertion in CondBB.
	Builder->SetInsertPoint(CondBB);

	/**
	 * Equivalent:
	 * while(itemCtr < size)
	 */
	AllocaInst *mem_itemCtr = NamedValuesBinaryCol[itemCtrVar];
	Value *lhs = Builder->CreateLoad(mem_itemCtr);
	Value *rhs = val_size;
	Value *cond = Builder->CreateICmpSLT(lhs,rhs);

	// Make the new basic block for the loop header (BODY), inserting after current block.
	BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);

	// Create the "AFTER LOOP" block and insert it.
	BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
	context->setEndingBlock(AfterBB);

	// Insert the conditional branch into the end of CondBB.
	Builder->CreateCondBr(cond, LoopBB, AfterBB);

	// Start insertion in LoopBB.
	Builder->SetInsertPoint(LoopBB);

	//Get the 'oid' of each record and pass it along.
	//More general/lazy plugins will only perform this action,
	//instead of eagerly 'converting' fields
	//FIXME This action corresponds to materializing the oid. Do we want this?
	RecordAttribute tupleIdentifier = RecordAttribute(fnamePrefix,activeLoop,this->getOIDType());

	RawValueMemory mem_posWrapper;
	mem_posWrapper.mem = mem_itemCtr;
	mem_posWrapper.isNull = context->createFalse();
	(*variableBindings)[tupleIdentifier] = mem_posWrapper;

	//Actual Work (Loop through attributes etc.)
	for (vector<RecordAttribute*>::iterator it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		RecordAttribute attr = *(*it);
		size_t offset = 0;

		/* Read wanted field.
		 * Reminder: Primitive, scalar types have the (raw) buffer
		 * already cast to appr. type*/
		switch ((*it)->getOriginalType()->getTypeID()) {
		case BOOL:
			readAsBooleanLLVM(attr, *variableBindings);
			offset = 1;
			break;
		case STRING:
		{
			readAsStringLLVM(attr, *variableBindings);
			offset = 1;
			break;
		}
		case FLOAT:
			readAsFloatLLVM(attr, *variableBindings);
			offset = 1;
			break;
		case INT:
			readAsIntLLVM(attr, *variableBindings);
			offset = 1;
			break;
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR)<< "[BINARY COL PLUGIN: ] Binary col files do not contain collections";
			throw runtime_error(string("[BINARY COL PLUGIN: ] Binary col files do not contain collections"));
		case RECORD:
			LOG(ERROR) << "[BINARY COL PLUGIN: ] Binary col files do not contain record-valued attributes";
			throw runtime_error(string("[BINARY COL PLUGIN: ] Binary col files do not contain record-valued attributes"));
		default:
			LOG(ERROR) << "[BINARY COL PLUGIN: ] Unknown datatype";
			throw runtime_error(string("[BINARY COL PLUGIN: ] Unknown datatype"));
		}

		//Move to next position
		Value* val_offset = context->createInt64(offset);
		skipLLVM(attr, val_offset);
	}
	nextEntry();

	// Make the new basic block for the increment, inserting after current block.
	BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

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


