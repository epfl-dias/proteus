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

#include "plugins/json-plugin.hpp"

//"Should" be enough on a per-row basis
#define MAXTOKENS 1000

namespace jsonPipelined
{

#define TOKEN_PRINT(t) \
	printf("start: %d, end: %d, type: %d, size: %d\n", \
			(t).start, (t).end, (t).type, (t).size)

#define TOKEN_STRING(js, t, s) \
	(strncmp(js+(t).start, s, (t).end - (t).start) == 0 \
	 && strlen(s) == (t).end - (t).start)

JSONPlugin::JSONPlugin(RawContext* const context, string& fname,
		ExpressionType* schema) :
		context(context), fname(fname), schema(schema), var_buf("bufPtr"), var_tokenPtr(
				"tokens"), var_tokenOffset("tokenOffset"), var_tokenOffsetHash("tokenOffsetHash")
{
	cache = false;
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int32_type = Type::getInt32Ty(llvmContext);
	Type* int64_type = Type::getInt64Ty(llvmContext);

	//Memory mapping etc
	LOG(INFO)<< "[JSONPlugin - jsmn: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;
	fd = open(name_c, O_RDONLY);
	if (fd == -1)
	{
		throw runtime_error(string("json.open"));
	}
	buf = (const char*) mmap(NULL, fsize, PROT_READ | PROT_WRITE , MAP_PRIVATE, fd, 0);
	if (buf == MAP_FAILED)
	{
		throw runtime_error(string("json.mmap"));
	}

	//Retrieving schema - not needed yet
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.registerFileJSON(fname,schema);

	//Preparing structures and variables for codegen part
	Function* F = context->getGlobalFunction();
	IRBuilder<>* Builder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);

	//Buffer holding the entire JSON document
	AllocaInst *mem_buf = context->CreateEntryBlockAlloca(F,string("jsFilePtr"),charPtrType);
	Value* val_buf_i64 = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* val_buf = Builder->CreateIntToPtr(val_buf_i64,charPtrType);
	Builder->CreateStore(val_buf,mem_buf);
	NamedValuesJSON[var_buf] = mem_buf;

	/* TODO Realloc will (eventually) take care of potential requests for resize */
	lines = 1000;

	tokenType = context->CreateJSMNStruct();


	PointerType *tokenPtrType = PointerType::get(tokenType,0);
	PointerType *token2DPtrType = PointerType::get(tokenPtrType,0);

	/* PM */
	CachingService& cache = CachingService::getInstance();

	char* pmCast = cache.getPM(fname);
	Value *cast_tokenArray = NULL;
	if (pmCast == NULL) {
		cout << "NEW (JSON) PM" << endl;
		tokenBuf = (char*) malloc(lines * sizeof(jsmntok_t*));
		if (tokenBuf == NULL) {
			string msg = string(
					"[JSON Plugin: ]: Failed to allocate token arena");
			LOG(ERROR)<< msg;
			throw runtime_error(msg);
		}
		tokens = (jsmntok_t**) tokenBuf;
		mem_tokenArray = context->CreateEntryBlockAlloca(F, "jsTokenArray",
				token2DPtrType);
		cast_tokenArray = context->CastPtrToLlvmPtr(token2DPtrType, tokenBuf);

		/* Store PM in cache */
		/* To be used by subsequent queries */
		cache.registerPM(fname, tokenBuf);
	} else {
		cout << "(JSON) PM REUSE" << endl;
		jsmntok_t **tokens = (jsmntok_t **) pmCast;
		this->tokens = tokens;
		tokenBuf = NULL;

		mem_tokenArray = context->CreateEntryBlockAlloca(F, "jsTokenArray",
				token2DPtrType);
		cast_tokenArray = context->CastPtrToLlvmPtr(token2DPtrType,
				this->tokens);
	}
	Builder->CreateStore(cast_tokenArray, mem_tokenArray);

}

JSONPlugin::JSONPlugin(RawContext* const context, string& fname,
		ExpressionType* schema, size_t linehint) :
		context(context), fname(fname), schema(schema), var_buf("bufPtr"), var_tokenPtr(
				"tokens"), var_tokenOffset("tokenOffset"), var_tokenOffsetHash("tokenOffsetHash")
{
	cache = false;
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int32_type = Type::getInt32Ty(llvmContext);
	Type* int64_type = Type::getInt64Ty(llvmContext);

	//Memory mapping etc
	LOG(INFO)<< "[JSONPlugin - jsmn: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;
	fd = open(name_c, O_RDONLY);
	if (fd == -1)
	{
		throw runtime_error(string("json.open"));
	}
	buf = (const char*) mmap(NULL, fsize, PROT_READ | PROT_WRITE , MAP_PRIVATE, fd, 0);
	if (buf == MAP_FAILED)
	{
		throw runtime_error(string("json.mmap"));
	}

	//Retrieving schema - not needed yet
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.registerFileJSON(fname,schema);

	//Preparing structures and variables for codegen part
	Function* F = context->getGlobalFunction();
	IRBuilder<>* Builder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);

	//Buffer holding the entire JSON document
	AllocaInst *mem_buf = context->CreateEntryBlockAlloca(F,string("jsFilePtr"),charPtrType);
	Value* val_buf_i64 = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* val_buf = Builder->CreateIntToPtr(val_buf_i64,charPtrType);
	Builder->CreateStore(val_buf,mem_buf);
	NamedValuesJSON[var_buf] = mem_buf;

	lines = linehint;

	tokenType = context->CreateJSMNStruct();
	PointerType *tokenPtrType = PointerType::get(tokenType,0);
	PointerType *token2DPtrType = PointerType::get(tokenPtrType,0);

	/* PM */
	CachingService& cache = CachingService::getInstance();

	char* pmCast = cache.getPM(fname);
	Value *cast_tokenArray = NULL;
	if (pmCast == NULL) {
		cout << "NEW (JSON) PM" << endl;
		tokenBuf = (char*) malloc(lines * sizeof(jsmntok_t*));
		if (tokenBuf == NULL) {
			string msg = string(
					"[JSON Plugin: ]: Failed to allocate token arena");
			LOG(ERROR)<< msg;
			throw runtime_error(msg);
		}
		tokens = (jsmntok_t**) tokenBuf;
		mem_tokenArray = context->CreateEntryBlockAlloca(F, "jsTokenArray",
				token2DPtrType);
		cast_tokenArray = context->CastPtrToLlvmPtr(token2DPtrType, tokenBuf);

		/* Store PM in cache */
		/* To be used by subsequent queries */
		cache.registerPM(fname, tokenBuf);
	} else {
		cout << "(JSON) PM REUSE" << endl;
		jsmntok_t **tokens = (jsmntok_t **) pmCast;
		this->tokens = tokens;
		tokenBuf = NULL;

		mem_tokenArray = context->CreateEntryBlockAlloca(F, "jsTokenArray",
				token2DPtrType);
		cast_tokenArray = context->CastPtrToLlvmPtr(token2DPtrType,
				this->tokens);
	}
	Builder->CreateStore(cast_tokenArray, mem_tokenArray);
}

JSONPlugin::JSONPlugin(RawContext* const context, string& fname,
		ExpressionType* schema, size_t linehint, jsmntok_t **tokens) :
		context(context), fname(fname), schema(schema), var_buf("bufPtr"), var_tokenPtr(
				"tokens"), var_tokenOffset("tokenOffset"), var_tokenOffsetHash("tokenOffsetHash")
{
	cache = true;
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int32_type = Type::getInt32Ty(llvmContext);
	Type* int64_type = Type::getInt64Ty(llvmContext);

	//Memory mapping etc
	LOG(INFO)<< "[JSONPlugin - jsmn: ] " << fname;
	struct stat statbuf;
	const char* name_c = fname.c_str();
	stat(name_c, &statbuf);
	fsize = statbuf.st_size;
	fd = open(name_c, O_RDONLY);
	if (fd == -1)
	{
		throw runtime_error(string("json.open"));
	}
	buf = (const char*) mmap(NULL, fsize, PROT_READ | PROT_WRITE , MAP_PRIVATE, fd, 0);
	if (buf == MAP_FAILED)
	{
		throw runtime_error(string("json.mmap"));
	}

	//Retrieving schema - not needed yet
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.registerFileJSON(fname,schema);

	//Preparing structures and variables for codegen part
	Function* F = context->getGlobalFunction();
	IRBuilder<>* Builder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);

	//Buffer holding the entire JSON document
	AllocaInst *mem_buf = context->CreateEntryBlockAlloca(F,string("jsFilePtr"),charPtrType);
	Value* val_buf_i64 = ConstantInt::get(llvmContext, APInt(64,((uint64_t)buf)));
	//i8*
	Value* val_buf = Builder->CreateIntToPtr(val_buf_i64,charPtrType);
	Builder->CreateStore(val_buf,mem_buf);
	NamedValuesJSON[var_buf] = mem_buf;

	lines = linehint;

	tokenType = context->CreateJSMNStruct();

	PointerType *tokenPtrType = PointerType::get(tokenType,0);
	PointerType *token2DPtrType = PointerType::get(tokenPtrType,0);

	/* XXX */
	this->tokens = tokens;
	tokenBuf = NULL;

	mem_tokenArray = context->CreateEntryBlockAlloca(F,"jsTokenArray",token2DPtrType);
	Value *cast_tokenArray = context->CastPtrToLlvmPtr(token2DPtrType, this->tokens);
	Builder->CreateStore(cast_tokenArray, mem_tokenArray);


}

RawValueMemory JSONPlugin::initCollectionUnnest(RawValue parentTokenId)
{
	Function* F = context->getGlobalFunction();
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	vector<Value*> ArgsV;
	Value *val_parentTokenId = parentTokenId.value;
	AllocaInst *mem_parentTokenId =
			context->CreateEntryBlockAlloca(F,"mem_parentTokenTmp",val_parentTokenId->getType());
	Builder->CreateStore(val_parentTokenId,mem_parentTokenId);
	Value *val_offset = context->getStructElem(mem_parentTokenId, 0);
	Value *val_rowId = context->getStructElem(mem_parentTokenId, 1);
	Value *val_parentTokenNo = context->getStructElem(mem_parentTokenId, 2);


#ifdef DEBUGJSON
//	{
//	vector<Value*> ArgsV;
//	Function* debugInt = context->getFunction("printi64");
//
//	ArgsV.push_back(val_parentTokenNo);
//	Builder->CreateCall(debugInt, ArgsV);
//	}
#endif

	AllocaInst* mem_currentTokenId = context->CreateEntryBlockAlloca(F,
			string("currentTokenUnnested"), val_parentTokenId->getType());
	Value* val_1 = Builder->getInt64(1);
	Value* val_currentTokenNo = Builder->CreateAdd(val_parentTokenNo,
			val_1);

	/* Populating 'activeTuple'/oid struct */
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(0));
	//Shift in struct ptr
	Value* structPtr = Builder->CreateGEP(mem_currentTokenId, idxList);
	StoreInst *store_offset = Builder->CreateStore(val_offset,structPtr);

	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(1));
	//Shift in struct ptr
	structPtr = Builder->CreateGEP(mem_currentTokenId, idxList);
	StoreInst *store_rowId = Builder->CreateStore(val_rowId,structPtr);

	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(2));
	//Shift in struct ptr
	structPtr = Builder->CreateGEP(mem_currentTokenId, idxList);
	StoreInst *store_currentToken = Builder->CreateStore(val_currentTokenNo,structPtr);

#ifdef DEBUGJSON
	{
//	vector<Value*> ArgsV;
//	Function* debugInt = context->getFunction("printi64");
//
//	ArgsV.push_back(val_parentTokenNo);
//	Builder->CreateCall(debugInt, ArgsV);
//	ArgsV.clear();
//	ArgsV.push_back(val_currentTokenNo);
//	Builder->CreateCall(debugInt, ArgsV);
	}
#endif
	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_currentTokenId;
	mem_valWrapper.isNull = parentTokenId.isNull;
	return mem_valWrapper;
}

/**
 * tokens[i].end <= tokens[parentToken].end && tokens[i].end != 0
 */
RawValue JSONPlugin::collectionHasNext(RawValue parentTokenId,
		RawValueMemory mem_currentTokenId)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int8Type = Type::getInt8Ty(llvmContext);
	llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

#ifndef JSON_TIGHT
	Value *val_0 = Builder->getInt32(0);
#endif
#ifdef JSON_TIGHT
	Value *val_0 = Builder->getInt16(0);
#endif

	/* Parent Token */
	vector<Value*> ArgsV;
	Value *val_parentTokenId = parentTokenId.value;
	AllocaInst *mem_parentTokenId = context->CreateEntryBlockAlloca(F,
			"mem_parentTokenTmp", val_parentTokenId->getType());
	Builder->CreateStore(val_parentTokenId,mem_parentTokenId);
	Value *val_offset = context->getStructElem(mem_parentTokenId, 0);
	Value *val_rowId = context->getStructElem(mem_parentTokenId, 1);
	Value *val_parentTokenNo = context->getStructElem(mem_parentTokenId, 2);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
	AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
				string(var_tokenPtr), context->CreateJSMNStruct());
	Value* parentToken = context->getArrayElem(mem_tokens,
			val_parentTokenNo);
	Builder->CreateStore(parentToken, mem_tokens_shifted);
	Value* parent_token_end_rel = context->getStructElem(mem_tokens_shifted, 2);
//	Value* parent_token_end_rel64 =
//					Builder->CreateSExt(parent_token_end_rel,int64Type);
//	Value* parent_token_end = Builder->CreateAdd(parent_token_end_rel64,val_offset);

	/* Current Token */
	/* Offset is the same as the one of parent */
	Value *val_currentTokenNo = context->getStructElem(mem_currentTokenId.mem, 2);

	Value* currentToken = context->getArrayElem(mem_tokens, val_currentTokenNo);
	Builder->CreateStore(currentToken, mem_tokens_shifted);
	Value* current_token_end_rel =
			context->getStructElem(mem_tokens_shifted,2);
//	Value* current_token_end_rel64 = Builder->CreateSExt(current_token_end_rel,
//			int64Type);
//	Value* current_token_end = Builder->CreateAdd(current_token_end_rel64,
//			val_offset);

	Value* endCond1 = Builder->CreateICmpSLE(current_token_end_rel,
			parent_token_end_rel);
	Value* endCond2 = Builder->CreateICmpNE(current_token_end_rel, val_0);
	Value *endCond = Builder->CreateAnd(endCond1, endCond2);
	Value *endCond_isNull = context->createFalse();


#ifdef DEBUGJSON
//	{
//	vector<Value*> ArgsV;
//	ArgsV.clear();
//	Value *test = Builder->CreateNot(endCond);
//	ArgsV.push_back(endCond);
//	Function* debugBoolean = context->getFunction("printBoolean");
//	Builder->CreateCall(debugBoolean, ArgsV);
//	}
#endif
	RawValue valWrapper;
	valWrapper.value  = endCond;
	valWrapper.isNull = endCond_isNull;
	return valWrapper;
}

//RawValueMemory JSONPlugin::collectionGetNext(RawValueMemory mem_currentTokenId)
//{
//	LLVMContext& llvmContext = context->getLLVMContext();
//	IRBuilder<>* Builder = context->getBuilder();
//	Type* int64Type = Type::getInt64Ty(llvmContext);
//	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
//	Function* F = context->getGlobalFunction();
//	vector<Value*> ArgsV;
//	Function* debugInt = context->getFunction("printi");
//	Function* debugInt64 = context->getFunction("printi64");
//
//	Value* val_currentTokenId = Builder->CreateLoad(mem_currentTokenId.mem);
//	Value *val_offset = context->getStructElem(mem_currentTokenId.mem, 0);
//	Value *val_rowId = context->getStructElem(mem_currentTokenId.mem, 1);
//	Value *val_currentTokenNo = context->getStructElem(mem_currentTokenId.mem, 2);
//#ifdef DEBUGJSON
//	{
//		//Printing the active token that will be forwarded
//		vector<Value*> ArgsV;
//		ArgsV.clear();
//		ArgsV.push_back(val_currentTokenNo);
//		Function* debugInt = context->getFunction("printi64");
//		Builder->CreateCall(debugInt, ArgsV);
//	}
//#endif
//	Type *idType = (mem_currentTokenId.mem)->getAllocatedType();
//	AllocaInst *mem_tokenToReturnId =
//				context->CreateEntryBlockAlloca(F,"mem_NestedToken",idType);
//	/**
//	 * Reason for this:
//	 * Need to return 'i', but also need to increment it before returning
//	 */
//	Builder->CreateStore(val_currentTokenId,mem_tokenToReturnId);
//
//	/**
//	 * int i_contents = i+1;
//	 * while(tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start != 0)	{
//	 *			i_contents++;
//	 *	}
//	 *	i = i_contents;
//	 */
//	BasicBlock *skipContentsCond, *skipContentsBody, *skipContentsInc,
//			*skipContentsEnd;
//	context->CreateForLoop("skipContentsCond", "skipContentsBody",
//			"skipContentsInc", "skipContentsEnd", &skipContentsCond,
//			&skipContentsBody, &skipContentsInc, &skipContentsEnd);
//	/**
//	 * Entry Block:
//	 * int i_contents = i+1;
//	 */
//	Value *val_1 = Builder->getInt64(1);
//	AllocaInst* mem_i_contents = context->CreateEntryBlockAlloca(F,
//				string("i_contents"), int64Type);
//	Value *val_i_contents = Builder->CreateAdd(val_currentTokenNo, val_1);
//	Builder->CreateStore(val_i_contents, mem_i_contents);
//	Builder->CreateBr(skipContentsCond);
//
//	/**
//	 * tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start != 0
//	 */
//	Builder->SetInsertPoint(skipContentsCond);
//	//Prepare tokens[i_contents].start
//	Value *val_0 = Builder->getInt32(0);
//	val_i_contents = Builder->CreateLoad(mem_i_contents);
//
//	//tokens**
//	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
//	//shifted tokens**
//	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
//			val_rowId);
//	//tokens*
//	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
//
//	AllocaInst* mem_tokens_i_contents_shifted = context->CreateEntryBlockAlloca(
//			F, string(var_tokenPtr), context->CreateJSMNStruct());
//	Value* token_i_contents = context->getArrayElem(mem_tokens, val_i_contents);
//	Builder->CreateStore(token_i_contents, mem_tokens_i_contents_shifted);
//	Value* token_i_contents_start_rel = context->getStructElem(
//			mem_tokens_i_contents_shifted, 1);
////	Value* token_i_contents_start_rel64 = Builder->CreateSExt(token_i_contents_start_rel,int64Type);
////	Value* token_i_contents_start = Builder->CreateAdd(token_i_contents_start_rel64,val_offset);
//
//	//Prepare tokens[i].end
//	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
//			string(var_tokenPtr), context->CreateJSMNStruct());
//	Value* token_i = context->getArrayElem(mem_tokens, val_currentTokenNo);
//	Builder->CreateStore(token_i, mem_tokens_i_shifted);
//	Value* token_i_end_rel = context->getStructElem(mem_tokens_i_shifted, 2);
////	Value* token_i_end_rel64 = Builder->CreateSExt(token_i_end_rel,int64Type);
////	Value* token_i_end = Builder->CreateAdd(token_i_end_rel64,val_offset);
//
//	//Prepare condition
//	Value* endCond1 = Builder->CreateICmpSLE(token_i_contents_start_rel,
//			token_i_end_rel);
//	Value* endCond2 = Builder->CreateICmpNE(token_i_contents_start_rel, val_0);
//	Value *endCond = Builder->CreateAnd(endCond1, endCond2);
//	BranchInst::Create(skipContentsBody, skipContentsEnd, endCond,
//			skipContentsCond);
//
//	/**
//	 * BODY:
//	 * i_contents++;
//	 */
//	Builder->SetInsertPoint(skipContentsBody);
//	Value* val_i_contents_1 = Builder->CreateAdd(val_i_contents, val_1);
//	Builder->CreateStore(val_i_contents_1, mem_i_contents);
//	val_i_contents = Builder->CreateLoad(mem_i_contents);
//	Builder->CreateBr(skipContentsInc);
//
//	/**
//	 * INC:
//	 * Nothing to do
//	 */
//	Builder->SetInsertPoint(skipContentsInc);
//	Builder->CreateBr(skipContentsCond);
//
//	/**
//	 * END:
//	 * i = i_contents;
//	 */
//	Builder->SetInsertPoint(skipContentsEnd);
//	val_i_contents = Builder->CreateLoad(mem_i_contents);
//
//	/* rowId and offset still the same */
////	AllocaInst *mem_tmp = context->CreateEntryBlockAlloca(F,"mem_unnest_inc",idType);
//	vector<Value*> idxList = vector<Value*>();
////	idxList.push_back(context->createInt32(0));
////	idxList.push_back(context->createInt32(0));
////	//Shift in struct ptr
////	Value* structPtr = Builder->CreateGEP(mem_tmp, idxList);
////	StoreInst *store_offset = Builder->CreateStore(val_offset,structPtr);
//////
////	idxList.clear();
////	idxList.push_back(context->createInt32(0));
////	idxList.push_back(context->createInt32(1));
////	//Shift in struct ptr
////	structPtr = Builder->CreateGEP(mem_tmp, idxList);
////	StoreInst *store_rowId = Builder->CreateStore(val_rowId,structPtr);
////
//	idxList.clear();
//	idxList.push_back(context->createInt32(0));
//	idxList.push_back(context->createInt32(2));
//	//Shift in struct ptr
//	Value *structPtr = Builder->CreateGEP(mem_currentTokenId.mem, idxList);
//	StoreInst *store_tokenNo = Builder->CreateStore(val_i_contents,structPtr);
//
////	Value *val_tmp = Builder->CreateLoad(mem_tmp);
////	Builder->CreateStore(val_tmp, mem_tokenToReturnId);
//
//#ifdef DEBUGJSON
//	{
//		//Printing the active token that will be forwarded
//		vector<Value*> ArgsV;
//		ArgsV.clear();
//		ArgsV.push_back(val_i_contents);
//		Value *val_currentTokenNo = context->getStructElem(mem_currentTokenId.mem, 2);
//		Function* debugInt = context->getFunction("printi64");
//		Builder->CreateCall(debugInt, ArgsV);
//	}
//#endif
//	RawValueMemory mem_newTokenIdWrap;
//	mem_newTokenIdWrap.mem = mem_tokenToReturnId;
//	mem_newTokenIdWrap.isNull = context->createFalse();
//	return mem_newTokenIdWrap;
//}

RawValueMemory JSONPlugin::collectionGetNext(RawValueMemory mem_currentTokenId)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	IRBuilder<>* Builder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(llvmContext);
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
	Function* F = context->getGlobalFunction();
	vector<Value*> ArgsV;
	Function* debugInt = context->getFunction("printi");
	Function* debugInt64 = context->getFunction("printi64");

#ifndef JSON_TIGHT
	Value *val_0 = Builder->getInt32(0);
#endif
#ifdef JSON_TIGHT
	Value *val_0 = Builder->getInt16(0);
#endif

	Value* val_currentTokenId = Builder->CreateLoad(mem_currentTokenId.mem);
	Value *val_offset = context->getStructElem(mem_currentTokenId.mem, 0);
	Value *val_rowId = context->getStructElem(mem_currentTokenId.mem, 1);
	Value *val_currentTokenNo = context->getStructElem(mem_currentTokenId.mem, 2);

	Type *idType = (mem_currentTokenId.mem)->getAllocatedType();
	AllocaInst* mem_tokenToReturn = context->CreateEntryBlockAlloca(F,
				std::string("tokenToUnnest"), idType);
	Value* tokenToReturn_isNull = context->createFalse();
	Builder->CreateStore(val_currentTokenId, mem_tokenToReturn);

	/**
	 * int i_contents = i+1;
	 * while(tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start != 0)	{
	 *			i_contents++;
	 *	}
	 *	i = i_contents;
	 */
	BasicBlock *skipContentsCond, *skipContentsBody, *skipContentsInc,
			*skipContentsEnd;
	context->CreateForLoop("skipContentsCond", "skipContentsBody",
			"skipContentsInc", "skipContentsEnd", &skipContentsCond,
			&skipContentsBody, &skipContentsInc, &skipContentsEnd);
	/**
	 * Entry Block:
	 * int i_contents = i+1;
	 */
	Value *val_1 = Builder->getInt64(1);
	AllocaInst* mem_i_contents = context->CreateEntryBlockAlloca(F,
			std::string("i_contents"), int64Type);
	Value *val_i_contents = Builder->CreateAdd(val_currentTokenNo, val_1);
	Builder->CreateStore(val_i_contents, mem_i_contents);
	Builder->CreateBr(skipContentsCond);

	/**
	 * tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start != 0
	 */
	Builder->SetInsertPoint(skipContentsCond);
	//Prepare tokens[i_contents].start
	val_i_contents = Builder->CreateLoad(mem_i_contents);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);

	AllocaInst* mem_tokens_i_contents_shifted = context->CreateEntryBlockAlloca(
			F, std::string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_i_contents = context->getArrayElem(mem_tokens, val_i_contents);
	Builder->CreateStore(token_i_contents, mem_tokens_i_contents_shifted);
	Value* token_i_contents_start_rel = context->getStructElem(
			mem_tokens_i_contents_shifted, 1);

	//Prepare tokens[i].end
	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
			string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, val_currentTokenNo);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);
	Value* token_i_end_rel = context->getStructElem(mem_tokens_i_shifted, 2);

	//Prepare condition
	Value* endCond1 = Builder->CreateICmpSLE(token_i_contents_start_rel,
			token_i_end_rel);
	Value* endCond2 = Builder->CreateICmpNE(token_i_contents_start_rel, val_0);
	Value *endCond = Builder->CreateAnd(endCond1, endCond2);
	BranchInst::Create(skipContentsBody, skipContentsEnd, endCond,
			skipContentsCond);

	/**
	 * BODY:
	 * i_contents++;
	 */
	Builder->SetInsertPoint(skipContentsBody);
	Value* val_i_contents_1 = Builder->CreateAdd(val_i_contents, val_1);
	Builder->CreateStore(val_i_contents_1, mem_i_contents);
	val_i_contents = Builder->CreateLoad(mem_i_contents);
	Builder->CreateBr(skipContentsInc);

	/**
	 * INC:
	 * Nothing to do
	 */
	Builder->SetInsertPoint(skipContentsInc);
	Builder->CreateBr(skipContentsCond);

	/**
	 * END:
	 * i = i_contents;
	 */
	Builder->SetInsertPoint(skipContentsEnd);
	val_i_contents = Builder->CreateLoad(mem_i_contents);


	/* rowId and offset still the same */
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(2));
	//Shift in struct ptr
	Value* structPtr = Builder->CreateGEP(mem_currentTokenId.mem, idxList);
	StoreInst *store_tokenId = Builder->CreateStore(val_i_contents,structPtr);

	RawValueMemory mem_wrapperVal;
	mem_wrapperVal.mem = mem_tokenToReturn;
	mem_wrapperVal.isNull = tokenToReturn_isNull;
	return mem_wrapperVal;
}

void JSONPlugin::scanObjects(const RawOperator& producer, Function* debug)
{
	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();

	Function* debugInt64 = context->getFunction("printi64");
	Function* newLine = context->getFunction("newline");
	Function* parseLineJSON = context->getFunction("parseLineJSON");

	Value *val_zero = context->createInt64(0);
	Value *val_one = context->createInt64(1);
	//Container for the variable bindings
	map<RecordAttribute, RawValueMemory>* variableBindings = new map<
			RecordAttribute, RawValueMemory>();

	AllocaInst *mem_buf = NamedValuesJSON[var_buf];
	Value *val_buf = Builder->CreateLoad(mem_buf);
	Value *val_fsize = context->createInt64(fsize);
	AllocaInst *mem_lineCnt = context->CreateEntryBlockAlloca(F,"jsonLineCtr",int64Type);
	AllocaInst *mem_offset = context->CreateEntryBlockAlloca(F,"jsonBufCtr",int64Type);
	Builder->CreateStore(val_zero,mem_offset);
	Builder->CreateStore(val_zero,mem_lineCnt);

	/**
	 * Loop is now dictated by # of rows
	 * Foreach row i:
	 * 1. Find newline -> AVX
	 * 2. Parse JSON object in line -> Store in token_t *pm[i] -> Func
	 * 3. Fwd activeLoop: (i,0) to parent
	 */
	BasicBlock *jsonScanCond, *jsonScanBody, *jsonScanInc, *jsonScanEnd;
	context->CreateForLoop("jsonScanCond", "jsonScanBody", "jsonScanInc",
			"jsonScanEnd", &jsonScanCond, &jsonScanBody, &jsonScanInc, &jsonScanEnd);
	context->setCurrentEntryBlock(Builder->GetInsertBlock());
	context->setEndingBlock(jsonScanEnd);
	vector<Value*> ArgsV;


	/* while( offset != length ) */
	Builder->CreateBr(jsonScanCond);
	Builder->SetInsertPoint(jsonScanCond);

	Value *val_offset = Builder->CreateLoad(mem_offset);
	Value *cond = Builder->CreateICmpSLT(val_offset,val_fsize);

//#ifdef DEBUGJSON
//	ArgsV.clear();
//	ArgsV.push_back(val_offset);
//	Builder->CreateCall(debugInt64, ArgsV);
//	ArgsV.clear();
//	ArgsV.push_back(val_fsize);
//	Builder->CreateCall(debugInt64, ArgsV);
//#endif
	Builder->CreateCondBr(cond,jsonScanBody,jsonScanEnd);

	/* Body */
	Builder->SetInsertPoint(jsonScanBody);

	Value *val_shiftedBuf = Builder->CreateInBoundsGEP(val_buf, val_offset);
	Value *val_len = Builder->CreateSub(val_fsize,val_offset);

	/* Find newline -> AVX */
	/* Is it worth the function call?
	 * Y, there are some savings, even for short(ish) entries */
	ArgsV.clear();
	ArgsV.push_back(val_shiftedBuf);
	ArgsV.push_back(val_fsize);
	Value *idx_newlineRelative = Builder->CreateCall(newLine, ArgsV);

	/* Ending of current JSON object */
	Value *idx_newlineAbsolute = Builder->CreateAdd(idx_newlineRelative,
			val_offset);

	/* Parse line into JSON */
	Value *val_lineCnt = Builder->CreateLoad(mem_lineCnt);
	if(!cache)	{
		Value *val_tokenArray = Builder->CreateLoad(mem_tokenArray);
		ArgsV.clear();
		ArgsV.push_back(val_buf);
		ArgsV.push_back(val_offset);
		ArgsV.push_back(idx_newlineAbsolute);
		ArgsV.push_back(val_tokenArray);
		ArgsV.push_back(val_lineCnt);
		Builder->CreateCall(parseLineJSON, ArgsV);
	}

	/* Triggering Parent */
	RecordAttribute tupleIdentifier = RecordAttribute(fname, activeLoop,this->getOIDType());
	RawValueMemory mem_tokenWrapper;

	/* Struct forwarded: (offsetInFile, rowId, tokenNo)*/
	vector<Type*> tokenIdMembers;
	tokenIdMembers.push_back(int64Type);
	tokenIdMembers.push_back(int64Type);
	tokenIdMembers.push_back(int64Type);

	StructType *tokenIdType = StructType::get(context->getLLVMContext(),tokenIdMembers);
	AllocaInst *mem_tokenId = context->CreateEntryBlockAlloca(F,"tokenId",tokenIdType);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(0));
	//Shift in struct ptr
	Value* structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_offset = Builder->CreateStore(val_offset,structPtr);

	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(1));
	structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_rowId = Builder->CreateStore(val_lineCnt,structPtr);

	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(2));
	structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_tokenNo = Builder->CreateStore(val_zero,structPtr);

	mem_tokenWrapper.mem = mem_tokenId;
	mem_tokenWrapper.isNull = context->createFalse();
	(*variableBindings)[tupleIdentifier] = mem_tokenWrapper;
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context, *state);

	/* Beginning of next JSON object */
	idx_newlineAbsolute = Builder->CreateAdd(idx_newlineAbsolute, val_one);


	Builder->CreateBr(jsonScanInc);

	/* Inc */
	Builder->SetInsertPoint(jsonScanInc);
	Builder->CreateStore(idx_newlineAbsolute,mem_offset);

	val_lineCnt = Builder->CreateAdd(val_lineCnt,val_one);
	Builder->CreateStore(val_lineCnt,mem_lineCnt);
	Builder->CreateBr(jsonScanCond);

	Builder->SetInsertPoint(jsonScanEnd);



}

/**
 *  while(tokens[i].start < tokens[curr].end && tokens[i].start != 0)	i++;
 */
void JSONPlugin::skipToEnd()
{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];
	AllocaInst* mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
	Value* val_offset = Builder->CreateLoad(mem_tokenOffset);

	Value* val_curr = Builder->CreateLoad(mem_tokenOffset);
	AllocaInst* mem_tokens_curr_shifted = context->CreateEntryBlockAlloca(F,
			std::string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_curr = context->getArrayElem(mem_tokens, val_curr);
	Builder->CreateStore(token_curr, mem_tokens_curr_shifted);
	Value* token_curr_end = context->getStructElem(mem_tokens_curr_shifted, 2);

	/**
	 * LOOP BLOCKS
	 */
	BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
	context->CreateForLoop("jsTokenSkipCond", "jsTokenSkipBody",
			"jsTokenSkipInc", "jsTokenSkipEnd", &tokenSkipCond, &tokenSkipBody,
			&tokenSkipInc, &tokenSkipEnd);

	/**
	 * Entry Block: Simply jumping to condition part
	 */
	Builder->CreateBr(tokenSkipCond);

	/**
	 * Condition: tokens[i].start < tokens[curr].end && tokens[i].start != 0
	 */
	Builder->SetInsertPoint(tokenSkipCond);
	val_offset = Builder->CreateLoad(mem_tokenOffset);
	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
			std::string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, val_offset);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);

	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_start = context->getStructElem(mem_tokens_i_shifted, 1);
	Value* rhs = context->createInt32(0);

	Value *endCond1 = Builder->CreateICmpSLT(token_i_start, token_curr_end);
	Value *endCond2 = Builder->CreateICmpNE(token_i_start, rhs);
	Value *endCond = Builder->CreateAnd(endCond1, endCond2);

	BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

	/**
	 * BODY:
	 * i++
	 */
	Builder->SetInsertPoint(tokenSkipBody);

	val_offset = Builder->CreateLoad(mem_tokenOffset);
	Value *val_step = Builder->getInt64(1);
	//CastInst* token_i_start64 = new SExtInst(token_i_start, int64Type, "i_64", tokenSkipBody);
	//cout<<int64_conv->getType()->getTypeID()<< " vs " << val_step->getType()->getTypeID();

	//Builder->CreateCall(debugInt, ArgsV, "printi");
	Value *token_i_inc = Builder->CreateAdd(val_offset, val_step, "i_inc");
	Builder->CreateStore(token_i_inc, mem_tokenOffset);

#ifdef DEBUG
	//std::vector<Value*> ArgsV;
	//ArgsV.clear();
	//ArgsV.push_back(token_i_start);
	//Function* debugInt = context->getFunction("printi");
	//Builder->CreateCall(debugInt, ArgsV);
#endif

	Builder->CreateBr(tokenSkipInc);

	/**
	 * INC:
	 * Nothing to do
	 * (in principle, job done in body could be done here)
	 */
	Builder->SetInsertPoint(tokenSkipInc);
	Builder->CreateBr(tokenSkipCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(tokenSkipEnd);
	LOG(INFO)<< "[Scan - JSON: ] End of skiptoEnd()";
}

RawValueMemory JSONPlugin::readPath(string activeRelation,
		Bindings wrappedBindings, const char* path)
{
	/**
	 * FIXME Add an extra (generated) check here
	 * Only objects are relevant to path expressions
	 * These types of validation should be applied as high as possible
	 * Still, probably unavoidable here
	 *	if(tokens[parentTokenNo].type != JSMN_OBJECT)	{
	 *		string msg = string("[JSON Plugin - jsmn: ]: Path traversal is only applicable to objects");
	 *		LOG(ERROR) << msg;
	 *		throw runtime_error(msg);
	 *	}
	 */

//	Value *val_zero = context->createInt64(0);
	const OperatorState& state = *(wrappedBindings.state);
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
	vector<Value*> argsV;

	//Get relevant ROW number
	RecordAttribute rowIdentifier = RecordAttribute(activeRelation,
			activeLoop,this->getOIDType());
	const map<RecordAttribute, RawValueMemory>& bindings = state.getBindings();
	map<RecordAttribute, RawValueMemory>::const_iterator it = bindings.find(
			rowIdentifier);
	if (it == bindings.end())
	{
		string error_msg =
				"[JSONPlugin - jsmn: ] Current tuple binding not found";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	/* scanObjects now forwards (offset, rowId) structs */
	RawValueMemory mem_tokenIdWrapper = (it->second);

	Value* val_offset = context->getStructElem(mem_tokenIdWrapper.mem, 0);
	Value* val_rowId = context->getStructElem(mem_tokenIdWrapper.mem, 1);
	Value* parentTokenNo = context->getStructElem(mem_tokenIdWrapper.mem, 2);

	//Preparing default return value (i.e., path not found)
	AllocaInst* mem_return = context->CreateEntryBlockAlloca(F,
			std::string("pathReturn"), int64Type);
	Value *minus_1 = context->createInt64(-1);
	Builder->CreateStore(minus_1, mem_return);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray, val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);

	AllocaInst* mem_tokens_parent_shifted = context->CreateEntryBlockAlloca(F,
			string(var_tokenPtr), context->CreateJSMNStruct());

	Value* token_parent = context->getArrayElem(mem_tokens, parentTokenNo);
	Builder->CreateStore(token_parent, mem_tokens_parent_shifted);
	Value* token_parent_end_rel =
			context->getStructElem(mem_tokens_parent_shifted,2);
//	Value* token_parent_end_rel64 =
//				Builder->CreateSExt(token_parent_end_rel,int64Type);
//	Value* token_parent_end = Builder->CreateAdd(token_parent_end_rel64,val_offset);
#ifdef DEBUGJSON
//	{
//	vector<Value*> ArgsV;
//	ArgsV.clear();
//	Function* debugInt = context->getFunction("printi64");
//	ArgsV.push_back(val_offset);
//	Builder->CreateCall(debugInt, ArgsV);
//	ArgsV.clear();
//
//	ArgsV.push_back(val_rowId);
//	Builder->CreateCall(debugInt, ArgsV);
//	ArgsV.clear();
//
//	Value *tmp = context->createInt64(1001);
//	ArgsV.push_back(tmp);
//	Builder->CreateCall(debugInt, ArgsV);
//	ArgsV.clear();
//	}
#endif
	/**
	 * LOOP BLOCKS
	 */
	BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
	context->CreateForLoop("path_tokenSkipCond", "path_tokenSkipBody",
			"path_tokenSkipInc", "path_tokenSkipEnd", &tokenSkipCond,
			&tokenSkipBody, &tokenSkipInc, &tokenSkipEnd);

	/**
	 * Entry Block:
	 */

	Value *val_1 = Builder->getInt64(1);
	Value *val_i = Builder->CreateAdd(parentTokenNo, val_1);
	AllocaInst* mem_i = context->CreateEntryBlockAlloca(F, std::string("tmp_i"),
			int64Type);
	Builder->CreateStore(val_i, mem_i);
	Builder->CreateBr(tokenSkipCond);

	/**
	 * tokens[i].end <= tokens[parentToken].end
	 */

	Builder->SetInsertPoint(tokenSkipCond);
	val_i = Builder->CreateLoad(mem_i);

	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
					string(var_tokenPtr), context->CreateJSMNStruct());

	Value* token_i = context->getArrayElem(mem_tokens, val_i);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);

	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_end_rel = context->getStructElem(mem_tokens_i_shifted, 2);
	Value* token_i_end_rel64 = Builder->CreateSExt(token_i_end_rel,int64Type);
	Value* token_i_end = Builder->CreateAdd(token_i_end_rel64,val_offset);
	Value *endCond = Builder->CreateICmpSLE(token_i_end_rel, token_parent_end_rel);
	BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

	/**
	 * BODY:
	 */
	Builder->SetInsertPoint(tokenSkipBody);

	/**
	 * IF-ELSE inside body:
	 * if(TOKEN_STRING(buf,tokens[i],key.c_str()))
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifTokenEq",
			"elseTokenEq", &ifBlock, &elseBlock, tokenSkipInc);

	Value* token_i_start_rel = context->getStructElem(mem_tokens_i_shifted, 1);
	Value* token_i_start_rel64 = Builder->CreateSExt(token_i_start_rel,int64Type);
	Value* token_i_start = Builder->CreateAdd(token_i_start_rel64,val_offset);

	int len = strlen(path) + 1;
	char* pathCopy = (char*) malloc(len * sizeof(char));
	strcpy(pathCopy, path);
	pathCopy[len - 1] = '\0';
	Value* globalStr = context->CreateGlobalString(pathCopy);
	Value* buf = Builder->CreateLoad(NamedValuesJSON[var_buf]);
	//Preparing custom 'strcmp'
	argsV.push_back(buf);
	argsV.push_back(token_i_start);
	argsV.push_back(token_i_end);
	argsV.push_back(globalStr);
	Function* tokenCmp = context->getFunction("compareTokenString64");
	Value* tokenEq = Builder->CreateCall(tokenCmp, argsV);
	Value* rhs = context->createInt32(1);
	Value *cond = Builder->CreateICmpEQ(tokenEq, rhs);
	Builder->CreateCondBr(cond, ifBlock, elseBlock);

	/**
	 * IF BLOCK
	 * TOKEN_PRINT(tokens[i+1]);
	 */
	Builder->SetInsertPoint(ifBlock);

	Value* val_i_1 = Builder->CreateAdd(val_i, val_1);
	AllocaInst* mem_tokens_i_1_shifted = context->CreateEntryBlockAlloca(F,
			std::string(var_tokenPtr), context->CreateJSMNStruct());

	Value* token_i_1 = context->getArrayElem(mem_tokens, val_i_1);
	Builder->CreateStore(token_i_1, mem_tokens_i_1_shifted);

	//Storing return value (i+1)
	Builder->CreateStore(val_i_1, mem_return);

#ifdef DEBUGJSON
//		Function* debugInt = context->getFunction("printi");
//		argsV.clear();
//		Value *tmp = context->createInt32(100);
//		argsV.push_back(tmp);
//		Builder->CreateCall(debugInt, argsV);
//
//		argsV.clear();
//		Value* token_i_1_start =
//				context->getStructElem(mem_tokens_i_1_shifted, 1);
//		argsV.push_back(token_i_1_start);
//		Builder->CreateCall(debugInt, argsV);
#endif

	Builder->CreateBr(tokenSkipEnd);

	/**
	 * ELSE BLOCK
	 */
	Builder->SetInsertPoint(elseBlock);
	Builder->CreateBr(tokenSkipInc);

	/**
	 * (Back to LOOP)
	 * INC:
	 * i += 2
	 */
	Builder->SetInsertPoint(tokenSkipInc);
	val_i = Builder->CreateLoad(mem_i);
	Value* val_2 = Builder->getInt64(2);
	Value* val_i_2 = Builder->CreateAdd(val_i, val_2);
	Builder->CreateStore(val_i_2, mem_i);

	token_i = context->getArrayElem(mem_tokens, val_i_2);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);

	Builder->CreateBr(tokenSkipCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(tokenSkipEnd);
	LOG(INFO)<< "[Scan - JSON: ] End of readPath()";

	RawValueMemory mem_valWrapper;

	/* Struct returned: (offsetInFile, rowId, tokenId)*/
	vector<Type*> tokenIdMembers;
	tokenIdMembers.push_back(int64Type);
	tokenIdMembers.push_back(int64Type);
	tokenIdMembers.push_back(int64Type);

	StructType *tokenIdType = StructType::get(context->getLLVMContext(),tokenIdMembers);
	AllocaInst *mem_tokenId = context->CreateEntryBlockAlloca(F,"tokenId",tokenIdType);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(0));
	//Shift in struct ptr
	Value* structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_offset = Builder->CreateStore(val_offset, structPtr);

	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(1));
	structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_rowId = Builder->CreateStore(val_rowId, structPtr);

	Value *val_return = Builder->CreateLoad(mem_return);
	idxList.clear();
	idxList.push_back(context->createInt32(0));
	idxList.push_back(context->createInt32(2));
	structPtr = Builder->CreateGEP(mem_tokenId, idxList);
	StoreInst *store_tokenId = Builder->CreateStore(val_return, structPtr);

	mem_valWrapper.mem = mem_tokenId;
	mem_valWrapper.isNull = context->createFalse();
	return mem_valWrapper;
}

RawValueMemory JSONPlugin::readPathInternal(RawValueMemory mem_parentTokenId,
		const char* path)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
	vector<Value*> argsV;
	Function* debugInt = context->getFunction("printi64");
	Function* debugInt32 = context->getFunction("printi");

	Value* val_offset = context->getStructElem(mem_parentTokenId.mem, 0);
	Value* val_rowId = context->getStructElem(mem_parentTokenId.mem, 1);
	Value* parentTokenNo = context->getStructElem(mem_parentTokenId.mem, 2);

	//Preparing default return value (i.e., path not found)
	AllocaInst* mem_return = context->CreateEntryBlockAlloca(F,
			std::string("pathReturn"), int64Type);
	Value *minus_1 = context->createInt64(-1);
	Builder->CreateStore(minus_1, mem_return);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
	AllocaInst* mem_tokens_parent_shifted = context->CreateEntryBlockAlloca(F,
			string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_parent = context->getArrayElem(mem_tokens, parentTokenNo);
	Builder->CreateStore(token_parent, mem_tokens_parent_shifted);
	Value* token_parent_end_rel =
			context->getStructElem(mem_tokens_parent_shifted,2);
	Value* token_parent_end_rel64 =
				Builder->CreateSExt(token_parent_end_rel,int64Type);
	Value* token_parent_end = Builder->CreateAdd(token_parent_end_rel64,val_offset);

	/**
	 * LOOP BLOCKS
	 */
	BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
	context->CreateForLoop("path_tokenSkipCondInternal",
			"path_tokenSkipBodyInternal", "path_tokenSkipIncInternal",
			"path_tokenSkipEndInternal", &tokenSkipCond, &tokenSkipBody,
			&tokenSkipInc, &tokenSkipEnd);

	/**
	 * Entry Block:
	 */

	Value *val_1 = Builder->getInt64(1);
	Value *val_i = Builder->CreateAdd(parentTokenNo, val_1);
	AllocaInst* mem_i = context->CreateEntryBlockAlloca(F, std::string("tmp_i"),
			int64Type);
	Builder->CreateStore(val_i, mem_i);
	Builder->CreateBr(tokenSkipCond);

	/**
	 * tokens[i].end <= tokens[parentToken].end
	 */

	Builder->SetInsertPoint(tokenSkipCond);
	val_i = Builder->CreateLoad(mem_i);
	AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
				string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_i = context->getArrayElem(mem_tokens, val_i);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);

	// 0: jsmntype_t type;
	// 1: int start;
	// 2: int end;
	// 3: int size;
	Value* token_i_end_rel = context->getStructElem(mem_tokens_i_shifted, 2);
	Value* token_i_end_rel64 = Builder->CreateSExt(token_i_end_rel,int64Type);
	Value* token_i_end = Builder->CreateAdd(token_i_end_rel64,val_offset);
#ifdef DEBUG
//	argsV.clear();
//	argsV.push_back(token_i_end);
//	Builder->CreateCall(debugInt32, argsV);
//	argsV.clear();
//
//	argsV.push_back(token_parent_end);
//	Builder->CreateCall(debugInt32, argsV);
//	argsV.clear();
#endif
	Value *endCond = Builder->CreateICmpSLE(token_i_end, token_parent_end);
	BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

	/**
	 * BODY:
	 */
	Builder->SetInsertPoint(tokenSkipBody);
#ifdef DEBUG
//	argsV.clear();
//	argsV.push_back(context->createInt64(666));
//	Builder->CreateCall(debugInt, argsV);
//	argsV.clear();
#endif

	/**
	 * IF-ELSE inside body:
	 * if(TOKEN_STRING(buf,tokens[i],key.c_str()))
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifTokenEq",
			"elseTokenEq", &ifBlock, &elseBlock, tokenSkipInc);

	Value* token_i_start_rel = context->getStructElem(mem_tokens_i_shifted, 1);
	Value* token_i_start_rel64 = Builder->CreateSExt(token_i_start_rel,int64Type);
	Value* token_i_start = Builder->CreateAdd(token_i_start_rel64,val_offset);

	int len = strlen(path) + 1;
	char* pathCopy = (char*) malloc(len * sizeof(char));
	strcpy(pathCopy, path);
	pathCopy[len] = '\0';
	Value* globalStr = context->CreateGlobalString(pathCopy);
	Value* buf = Builder->CreateLoad(NamedValuesJSON[var_buf]);
	//Preparing custom 'strcmp'
	argsV.push_back(buf);
	argsV.push_back(token_i_start);
	argsV.push_back(token_i_end);
	argsV.push_back(globalStr);
	Function* tokenCmp = context->getFunction("compareTokenString");
	Value* tokenEq = Builder->CreateCall(tokenCmp, argsV);
	Value* rhs = context->createInt32(1);
	Value *cond = Builder->CreateICmpEQ(tokenEq, rhs);
	Builder->CreateCondBr(cond, ifBlock, elseBlock);

	/**
	 * IF BLOCK
	 * TOKEN_PRINT(tokens[i+1]);
	 */
	Builder->SetInsertPoint(ifBlock);

	Value* val_i_1 = Builder->CreateAdd(val_i, val_1);
	AllocaInst* mem_tokens_i_1_shifted = context->CreateEntryBlockAlloca(F,
				string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token_i_1 = context->getArrayElem(mem_tokens, val_i_1);
	Builder->CreateStore(token_i_1, mem_tokens_i_1_shifted);

	//Storing return value (i+1)
	Builder->CreateStore(val_i_1, mem_return);

	Builder->CreateBr(tokenSkipEnd);

	/**
	 * ELSE BLOCK
	 */
	Builder->SetInsertPoint(elseBlock);
	Builder->CreateBr(tokenSkipInc);

	/**
	 * (Back to LOOP)
	 * INC:
	 * i += 2
	 */
	Builder->SetInsertPoint(tokenSkipInc);
	val_i = Builder->CreateLoad(mem_i);
	Value* val_2 = Builder->getInt64(2);
	Value* val_i_2 = Builder->CreateAdd(val_i, val_2);
	Builder->CreateStore(val_i_2, mem_i);

	token_i = context->getArrayElem(mem_tokens, val_i_2);
	Builder->CreateStore(token_i, mem_tokens_i_shifted);

	Builder->CreateBr(tokenSkipCond);

	/**
	 * END:
	 */
	Builder->SetInsertPoint(tokenSkipEnd);
	LOG(INFO)<< "[Scan - JSON: ] End of readPathInternal()";

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_return;
	mem_valWrapper.isNull = context->createFalse();
	return mem_valWrapper;
}

RawValueMemory JSONPlugin::readValue(RawValueMemory mem_value,
		const ExpressionType* type)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int8Type = Type::getInt8Ty(llvmContext);
	Type* int1Type = Type::getInt1Ty(llvmContext);
	llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	vector<Value*> ArgsV;
	Value* val_offset = context->getStructElem(mem_value.mem, 0);
	Value* val_rowId = context->getStructElem(mem_value.mem, 1);
	Value* tokenNo = context->getStructElem(mem_value.mem, 2);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
	AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
			string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token = context->getArrayElem(mem_tokens, tokenNo);
	Builder->CreateStore(token, mem_tokens_shifted);
	Value* token_start_rel = context->getStructElem(mem_tokens_shifted, 1);
	Value* token_end_rel = context->getStructElem(mem_tokens_shifted, 2);
	Value* token_start_rel64 = Builder->CreateSExt(token_start_rel,int64Type);
	Value* token_end_rel64 = Builder->CreateSExt(token_end_rel,int64Type);

	Value* token_start = Builder->CreateAdd(token_start_rel64,val_offset);
	Value* token_end = Builder->CreateAdd(token_end_rel64,val_offset);

	Value* bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);

	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, token_start);

	Function* conversionFunc = NULL;

	AllocaInst* mem_convertedValue = NULL;
	AllocaInst* mem_convertedValue_isNull = NULL;
	Value* convertedValue = NULL;

	mem_convertedValue_isNull = context->CreateEntryBlockAlloca(F,
			string("value_isNull"), int1Type);
	Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
	string error_msg;
	switch (type->getTypeID()) {
		case STRING:
		case RECORD:
		case LIST: {
			mem_convertedValue = context->CreateEntryBlockAlloca(F,
					string("existingObject"), (mem_value.mem)->getAllocatedType());
			break;
		}
		case SET: {
			error_msg = string("[JSON Plugin - jsmn: ]: SET datatype cannot occur");
			LOG(ERROR)<< error_msg;
			throw runtime_error(string(error_msg));
		}
		case BAG: {
			error_msg = string("[JSON Plugin - jsmn: ]: BAG datatype cannot occur");
			LOG(ERROR)<< error_msg;
		}
		throw runtime_error(string(error_msg));
		case BOOL:
		{
			mem_convertedValue = context->CreateEntryBlockAlloca(F,
					string("convertedBool"), int8Type);
			break;
		}
		case FLOAT:
		{
			mem_convertedValue = context->CreateEntryBlockAlloca(F,
					string("convertedFloat"), doubleType);
			break;
		}
		case INT:
		{
			mem_convertedValue = context->CreateEntryBlockAlloca(F,
					string("convertedInt"), int32Type);
			break;
		}
		default:
		{
			error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
			LOG(ERROR)<< error_msg;
			throw runtime_error(string(error_msg));
		}
	}

/**
 * Return (nil) for cases path was not found
 */
	BasicBlock *ifBlock, *elseBlock, *endBlock;
	endBlock = BasicBlock::Create(llvmContext, "afterReadValue", F);
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifPath",
			"elsePathNullEq", &ifBlock, &elseBlock, endBlock);
	Value* minus_1 = context->createInt64(-1);
	Value *cond = Builder->CreateICmpNE(tokenNo, minus_1);
	Builder->CreateCondBr(cond, ifBlock, elseBlock);

	/**
	 * IF BLOCK (tokenNo != -1)
	 */
	Builder->SetInsertPoint(ifBlock);

	switch (type->getTypeID())
	{
	case STRING:
		//For now, passing 'object' (tokenNo actually) along
	case RECORD:
		//Object
	case LIST:
	{
		/* rowId and offset still the same */
		vector<Value*> idxList = vector<Value*>();
		idxList.push_back(context->createInt32(0));
		idxList.push_back(context->createInt32(2));
		//Shift in struct ptr
		Value* structPtr = Builder->CreateGEP(mem_value.mem, idxList);
		StoreInst *store_offset = Builder->CreateStore(tokenNo,structPtr);
		//Array
		mem_convertedValue = mem_value.mem;
		break;
	}
	case BOOL:
	{
		ArgsV.push_back(bufPtr);
		ArgsV.push_back(token_start);
		ArgsV.push_back(token_end);
		conversionFunc = context->getFunction("convertBoolean");
		convertedValue = Builder->CreateCall(conversionFunc, ArgsV,
				"convertBoolean");
		Builder->CreateStore(convertedValue, mem_convertedValue);
		Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
		break;
	}
	case FLOAT:
	{
		conversionFunc = context->getFunction("atof");
		ArgsV.push_back(bufShiftedPtr);
		convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atof");
		Builder->CreateStore(convertedValue, mem_convertedValue);
		Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
		break;
	}
	case INT:
	{
		Value *val_len = Builder->CreateSub(token_end,token_start);
		Value *val_len32 = Builder->CreateTrunc(val_len,int32Type);
		atois(bufShiftedPtr,val_len32,mem_convertedValue,context);
		Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
#ifdef DEBUGJSON
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		convertedValue = Builder->CreateLoad(mem_convertedValue);
//		ArgsV.push_back(convertedValue);
//		Builder->CreateCall(debugInt, ArgsV);
//		ArgsV.clear();
//		Value *tmp = context->createInt32(888);
//		ArgsV.push_back(tmp);
//		Builder->CreateCall(debugInt, ArgsV);
#endif
		break;
	}
	default:
	{
		error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
		LOG(ERROR)<< error_msg;
		throw runtime_error(string(error_msg));
	}
	}
	Builder->CreateBr(endBlock);

	/**
	 * ELSE BLOCK
	 * return "(NULL)"
	 */
	Builder->SetInsertPoint(elseBlock);
#ifdef DEBUG //Invalid / NULL!
//	ArgsV.clear();
//	Function* debugInt = context->getFunction("printi");
//	Value *tmp = context->createInt32(-111);
//	ArgsV.push_back(tmp);
//	Builder->CreateCall(debugInt, ArgsV);
#endif
	Value* undefValue = Constant::getNullValue(
			mem_convertedValue->getAllocatedType());
	Builder->CreateStore(undefValue, mem_convertedValue);
	Builder->CreateStore(context->createTrue(), mem_convertedValue_isNull);
	Builder->CreateBr(endBlock);

	Builder->SetInsertPoint(endBlock);

	RawValueMemory mem_valWrapper;
	mem_valWrapper.mem = mem_convertedValue;
	mem_valWrapper.isNull = Builder->CreateLoad(mem_convertedValue_isNull);
#ifdef DEBUG //Invalid / NULL!
//	ArgsV.clear();
//	Function* debugBoolean = context->getFunction("printBoolean");
//	ArgsV.push_back(mem_valWrapper.isNull);
//	Builder->CreateCall(debugBoolean, ArgsV);
#endif
	return mem_valWrapper;
}

RawValue JSONPlugin::readCachedValue(CacheInfo info,
		const OperatorState& currState) {
	IRBuilder<>* const Builder = context->getBuilder();
	Function *F = context->getGlobalFunction();

	/* Need OID to retrieve corresponding value from bin. cache */
	RecordAttribute tupleIdentifier = RecordAttribute(fname, activeLoop,
			getOIDType());

	map<RecordAttribute, RawValueMemory>::const_iterator it =
			currState.getBindings().find(tupleIdentifier);
	if (it == currState.getBindings().end()) {
		string error_msg =
				"[Expression Generator: ] Current tuple binding / OID not found";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	RawValueMemory mem_oidCompositeWrapper = it->second;

	/*Reminder: JSON plugin's OID is composite*/
	Value *val_oid = context->getStructElem(mem_oidCompositeWrapper.mem, 1);

	StructType *cacheType = context->ReproduceCustomStruct(info.objectTypes);
	Value *typeSize = ConstantExpr::getSizeOf(cacheType);
	char* rawPtr = info.payloadPtr;
	int posInStruct = info.structFieldNo;

	/* Cast to appr. type */
	PointerType *ptr_cacheType = PointerType::get(cacheType, 0);
	Value *val_cachePtr = context->CastPtrToLlvmPtr(ptr_cacheType, rawPtr);

	Value *val_cacheShiftedPtr = context->getArrayElemMem(val_cachePtr,
			val_oid);
	Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
			posInStruct);
	Type *fieldType = val_cachedField->getType();

	/* This Alloca should not appear in optimized code */
	AllocaInst *mem_cachedField = context->CreateEntryBlockAlloca(F,
			"tmpCachedField", fieldType);
	Builder->CreateStore(val_cachedField, mem_cachedField);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateLoad(mem_cachedField);
	valWrapper.isNull = context->createFalse();
#ifdef DEBUGJSON
	{
		/* Obviously only works to peek integer fields */
		vector<Value*> ArgsV;

		Function* debugSth = context->getFunction("printi");
		ArgsV.push_back(val_cachedField);
		Builder->CreateCall(debugSth, ArgsV);
	}
#endif
	return valWrapper;
}

RawValue JSONPlugin::hashValue(RawValueMemory mem_value,
		const ExpressionType* type)
{
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int8Type = Type::getInt8Ty(llvmContext);
	Type* int1Type = Type::getInt1Ty(llvmContext);
	llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	Function* debugInt64 = context->getFunction("printi64");
	Function* debugInt = context->getFunction("printi");

	vector<Value*> ArgsV;
	Value* val_offset = context->getStructElem(mem_value.mem, 0);
	Value* val_rowId = context->getStructElem(mem_value.mem, 1);
	Value* tokenNo = context->getStructElem(mem_value.mem, 2);

	//tokens**
	Value *mem_tokenArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(mem_tokenArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
	AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
			string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token = context->getArrayElem(mem_tokens, tokenNo);
	Builder->CreateStore(token, mem_tokens_shifted);

	Value* token_start_rel = context->getStructElem(mem_tokens_shifted, 1);
	Value* token_end_rel = context->getStructElem(mem_tokens_shifted, 2);
	Value* token_start_rel64 = Builder->CreateSExt(token_start_rel, int64Type);
	Value* token_end_rel64 = Builder->CreateSExt(token_end_rel, int64Type);

	Value* token_start = Builder->CreateAdd(token_start_rel64, val_offset);
	Value* token_end = Builder->CreateAdd(token_end_rel64, val_offset);

	Value* bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);
	Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, token_start);

	Function* conversionFunc = NULL;
	Function* hashFunc = NULL;
	AllocaInst* mem_hashedValue = NULL;
	AllocaInst* mem_hashedValue_isNull = NULL;
	Value* hashedValue = NULL;
	Value* convertedValue = NULL;

	//Preparing hasher state
	NamedValuesJSON[var_tokenOffsetHash] = mem_value.mem;
	mem_hashedValue = context->CreateEntryBlockAlloca(F,
			std::string("hashValue"), int64Type);
	mem_hashedValue_isNull = context->CreateEntryBlockAlloca(F,
			std::string("hashvalue_isNull"), int1Type);
	Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
	Builder->CreateStore(context->createInt64(0), mem_hashedValue);
	string error_msg;

	/**
	 * Return (nil) for cases path was not found
	 */
	BasicBlock *ifBlock, *elseBlock, *endBlock;
	endBlock = BasicBlock::Create(llvmContext, "afterReadValueHash", F);
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifPathHash",
			"elsePathNullEqHash", &ifBlock, &elseBlock, endBlock);
	Value* minus_1 = context->createInt64(-1);
	Value *cond = Builder->CreateICmpNE(tokenNo, minus_1);

	Builder->CreateCondBr(cond, ifBlock, elseBlock);

	/**
	 * IF BLOCK (tokenNo != -1)
	 */
	Builder->SetInsertPoint(ifBlock);

	switch (type->getTypeID())
	{
	case STRING:
	{
		hashFunc = context->getFunction("hashStringC");
		ArgsV.clear();
		ArgsV.push_back(bufPtr);
		//FIXME tmp! - must make token_start and token_end datatype size_t
		Type* int64Type = Type::getInt64Ty(llvmContext);
		ArgsV.push_back(token_start);
		ArgsV.push_back(token_end);

		hashedValue = Builder->CreateCall(hashFunc, ArgsV, "hashStringC");
		Builder->CreateStore(hashedValue, mem_hashedValue);
		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

		Value* plus_1 = context->createInt64(1);
		Value* tokenNo_inc = Builder->CreateAdd(tokenNo,plus_1);
		Builder->CreateStore(tokenNo_inc,NamedValuesJSON[var_tokenOffsetHash]);
		break;
	}
	case RECORD:
	{
		//Object
		AllocaInst* mem_seed = context->CreateEntryBlockAlloca(F,
				std::string("hashSeed"), int64Type);
		Builder->CreateStore(context->createInt64(0), mem_seed);

		RawValueMemory recordElem;
		recordElem.mem = mem_value.mem;
		recordElem.isNull = context->createFalse();

		Function *hashCombine = context->getFunction("combineHashes");
		hashedValue = context->createInt64(0);

		list<RecordAttribute*>& args = ((RecordType*) type)->getArgs();
		list<RecordAttribute*>::iterator it = args.begin();

		//Not efficient -> duplicate work performed since fields are not visited incrementally
		//BUT: Attributes might not be in sequence, so they need to be visited in a generic way
		for (; it != args.end(); it++)
		{
			RecordAttribute* attr = *it;
			RawValueMemory mem_path = readPathInternal(recordElem,
					attr->getAttrName().c_str());
			NamedValuesJSON[var_tokenOffsetHash] = mem_path.mem;

			//CAREFUL: It's generated code that has to be stitched
			hashedValue = Builder->CreateLoad(mem_hashedValue);
			RawValue partialHash = hashValue(mem_path, attr->getOriginalType());
			ArgsV.clear();
			ArgsV.push_back(hashedValue);
			ArgsV.push_back(partialHash.value);
			//XXX Why loading now?
			//hashedValue = Builder->CreateLoad(mem_hashedValue);
			hashedValue = Builder->CreateCall(hashCombine, ArgsV,
					"combineHashesResJSON");
			Builder->CreateStore(hashedValue, mem_hashedValue);
		}

		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
		break;
	}
	case LIST:
	{
		//Array
		const ExpressionType& nestedType =
				((CollectionType*) type)->getNestedType();
		AllocaInst* mem_seed = context->CreateEntryBlockAlloca(F,
				std::string("hashSeed"), int64Type);
		Builder->CreateStore(context->createInt64(0), mem_seed);

		//Initializing: i = tokenNo + 1;
		AllocaInst* mem_tokenCnt = context->CreateEntryBlockAlloca(F,
				std::string("tokenCnt"), int64Type);
		Value* val_tokenCnt = Builder->CreateAdd(context->createInt64(1),
				tokenNo);
		Builder->CreateStore(val_tokenCnt, mem_tokenCnt);

		//while (tokens[i].start < tokens[tokenNo].end && tokens[i].start != 0)
		Function *hashCombine = context->getFunction("combineHashes");
		hashedValue = context->createInt64(0);
		Builder->CreateStore(hashedValue, mem_hashedValue);
		BasicBlock *unrollCond, *unrollBody, *unrollInc, *unrollEnd;
		context->CreateForLoop("hashListCond", "hashListBody", "hashListInc",
				"hashListEnd", &unrollCond, &unrollBody, &unrollInc,
				&unrollEnd);

		Builder->CreateBr(unrollCond);
		Builder->SetInsertPoint(unrollCond);
		AllocaInst* mem_tokens_i_shifted = context->CreateEntryBlockAlloca(F,
				string(var_tokenPtr), context->CreateJSMNStruct());
		val_tokenCnt = Builder->CreateLoad(mem_tokenCnt);
		Value* token_i = context->getArrayElem(mem_tokens, val_tokenCnt);
		Builder->CreateStore(token_i, mem_tokens_i_shifted);

		// 0: jsmntype_t type;
		// 1: int start;
		// 2: int end;
		// 3: int size;
		Value* token_i_start_rel = context->getStructElem(mem_tokens_i_shifted, 1);
		Value* token_i_start_rel64 = Builder->CreateSExt(token_i_start_rel,int64Type);
		Value* token_i_start = Builder->CreateAdd(token_i_start_rel64,val_offset);

		Value* rhs = context->createInt32(0);

		Value *endCond1 = Builder->CreateICmpSLT(token_i_start, token_end);
		Value *endCond2 = Builder->CreateICmpNE(token_i_start, rhs);
		Value *endCond = Builder->CreateAnd(endCond1, endCond2);

		Builder->CreateCondBr(endCond, unrollBody, unrollEnd);
		/**
		 * BODY:
		 * readValueEagerInterpreted(i, &nestedType);
		 * i++
		 */
		Builder->SetInsertPoint(unrollBody);

		RawValueMemory listElem;
		listElem.mem = mem_tokenCnt;
		listElem.isNull = context->createFalse();

		//CAREFUL: It's generated code that has to be stitched
		//XXX in the general case, nested type may vary between elements..
		RawValue partialHash = hashValue(listElem, &nestedType);

		ArgsV.clear();
		ArgsV.push_back(hashedValue);
		ArgsV.push_back(partialHash.value);
		hashedValue = Builder->CreateLoad(mem_hashedValue);
		hashedValue = Builder->CreateCall(hashCombine, ArgsV,
				"combineHashesResJSON");
		Builder->CreateStore(hashedValue, mem_hashedValue);

		//new offset due to recursive application of hashedValue
		Value *newTokenOffset = Builder->CreateLoad(NamedValuesJSON[var_tokenOffsetHash]);
		Builder->CreateStore(newTokenOffset, mem_tokenCnt);

		//Performing just a +1 addition would cause a lot of extra work
		//and defeat the purpose of recursion
//		val_tokenCnt = Builder->CreateAdd(Builder->CreateLoad(mem_tokenCnt),
//				context->createInt64(1));
//		Builder->CreateStore(val_tokenCnt, mem_tokenCnt);

		Builder->CreateBr(unrollInc);

		/**
		 * INC:
		 * Nothing to do
		 * (in principle, job done in body could be done here)
		 */
		Builder->SetInsertPoint(unrollInc);
		Builder->CreateBr(unrollCond);

		/**
		 * END:
		 */
		Builder->SetInsertPoint(unrollEnd);

		//Not needed to store again -> body took care of it
		//Builder->CreateStore(hashedValue, mem_hashedValue);
		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
		break;
	}
	case BOOL:
	{
		conversionFunc = context->getFunction("convertBoolean");
		ArgsV.push_back(bufPtr);
		ArgsV.push_back(token_start);
		ArgsV.push_back(token_end);
		convertedValue = Builder->CreateCall(conversionFunc, ArgsV,
				"convertBoolean");

		hashFunc = context->getFunction("hashBoolean");
		ArgsV.clear();
		ArgsV.push_back(convertedValue);
		hashedValue = Builder->CreateCall(conversionFunc, ArgsV, "hashBoolean");
		Builder->CreateStore(hashedValue, mem_hashedValue);
		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

		Value* plus_1 = context->createInt64(1);
		Value* tokenNo_inc = Builder->CreateAdd(tokenNo,plus_1);
		Builder->CreateStore(tokenNo_inc,NamedValuesJSON[var_tokenOffsetHash]);
		break;
	}
	case FLOAT:
	{
		conversionFunc = context->getFunction("atof");
		ArgsV.push_back(bufShiftedPtr);
		convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atof");

		hashFunc = context->getFunction("hashDouble");
		ArgsV.clear();
		ArgsV.push_back(convertedValue);
		hashedValue = Builder->CreateCall(conversionFunc, ArgsV, "hashDouble");
		Builder->CreateStore(hashedValue, mem_hashedValue);
		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

		Value* plus_1 = context->createInt64(1);
		Value* tokenNo_inc = Builder->CreateAdd(tokenNo,plus_1);
		Builder->CreateStore(tokenNo_inc,NamedValuesJSON[var_tokenOffsetHash]);
		break;
	}
	case INT:
	{
		conversionFunc = context->getFunction("atoi");
		ArgsV.push_back(bufShiftedPtr);
		convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atoi");
		#ifdef DEBUG
//			ArgsV.clear();
//			ArgsV.push_back(context->createInt32(-60));
//			Builder->CreateCall(debugInt, ArgsV);
//			ArgsV.clear();
//			ArgsV.push_back(convertedValue);
//			Builder->CreateCall(debugInt, ArgsV);
//			ArgsV.clear();
		#endif
		hashFunc = context->getFunction("hashInt");
		ArgsV.clear();
		ArgsV.push_back(convertedValue);
		hashedValue = Builder->CreateCall(hashFunc, ArgsV, "hashInt");
		Builder->CreateStore(hashedValue, mem_hashedValue);
		Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

		Value* plus_1 = context->createInt64(1);
		Value* tokenNo_inc = Builder->CreateAdd(tokenNo,plus_1);
		Builder->CreateStore(tokenNo_inc,NamedValuesJSON[var_tokenOffsetHash]);
		break;
	}
	default:
	{
		error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
		LOG(ERROR)<< error_msg;
		throw runtime_error(string(error_msg));
	}
	}
	Builder->CreateBr(endBlock);

	/**
	 * ELSE BLOCK
	 * "NULL" case
	 * TODO What should the behavior be in this case?
	 *
	 * c-p from Sybase manual: If the grouping column contains a null value,
	 * that row becomes its own group in the results.
	 * If the grouping column contains more than one null value,
	 * the null values form a single group.
	 */
	Builder->SetInsertPoint(elseBlock);

	Value* undefValue = Constant::getNullValue(
			mem_hashedValue->getAllocatedType());
	Builder->CreateStore(undefValue, mem_hashedValue);
	Builder->CreateStore(context->createTrue(), mem_hashedValue_isNull);
	Builder->CreateBr(endBlock);

	Builder->SetInsertPoint(endBlock);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateLoad(mem_hashedValue);
	valWrapper.isNull = Builder->CreateLoad(mem_hashedValue_isNull);
	return valWrapper;
}

void JSONPlugin::flushChunk(RawValueMemory mem_value, Value* fileName)
{

	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int8Type = Type::getInt8Ty(llvmContext);
	Type* int1Type = Type::getInt1Ty(llvmContext);
	llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();
	Function* F = context->getGlobalFunction();
	PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();

	//Preparing arguments
	vector<Value*> ArgsV;
	//buffer
	Value* bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);

	Value* val_offset = context->getStructElem(mem_value.mem, 0);
	Value* val_rowId = context->getStructElem(mem_value.mem, 1);
	Value* tokenNo = context->getStructElem(mem_value.mem, 2);

	//tokens**
	Value *val_token2DArray = Builder->CreateLoad(mem_tokenArray);
	//shifted tokens**
	Value *mem_tokenArrayShift = Builder->CreateInBoundsGEP(val_token2DArray,
			val_rowId);
	//tokens*
	Value *mem_tokens = Builder->CreateLoad(mem_tokenArrayShift);
	AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
			std::string(var_tokenPtr), context->CreateJSMNStruct());
	Value* token = context->getArrayElem(mem_tokens, tokenNo);
	Builder->CreateStore(token, mem_tokens_shifted);
	Value* token_start_rel = context->getStructElem(mem_tokens_shifted, 1);
	Value* token_end_rel = context->getStructElem(mem_tokens_shifted, 2);
	//where to flush? -> got it ready

#ifdef DEBUG
//		ArgsV.clear();
//		debugInt = context->getFunction("printi");
//		ArgsV.push_back(token_start);
//		Builder->CreateCall(debugInt, ArgsV);
//		ArgsV.clear();
//
//		ArgsV.push_back(token_end);
//		Builder->CreateCall(debugInt, ArgsV);
//		ArgsV.clear();
#endif

	Value* token_start_rel64 = Builder->CreateSExt(token_start_rel, int64Type);
	Value* token_end_rel64 = Builder->CreateSExt(token_end_rel, int64Type);

	Value* token_start = Builder->CreateAdd(token_start_rel64, val_offset);
	Value* token_end = Builder->CreateAdd(token_end_rel64, val_offset);
	ArgsV.push_back(bufPtr);
	ArgsV.push_back(token_start);
	ArgsV.push_back(token_end);
	ArgsV.push_back(fileName);
	Function *flushFunc = context->getFunction("flushStringC");
	Builder->CreateCall(flushFunc, ArgsV);
}

void JSONPlugin::generate(const RawOperator& producer)
{
	return scanObjects(producer, context->getGlobalFunction());
}

void JSONPlugin::finish()
{
	close(fd);
	munmap((void*) buf, fsize);
}

Value* JSONPlugin::getValueSize(RawValueMemory mem_value,
		const ExpressionType* type) {
	switch (type->getTypeID()) {
	case BOOL:
	case INT:
	case FLOAT: {
		Type *explicitType = (mem_value.mem)->getAllocatedType();
		return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
	}
	case STRING:
	case BAG:
	case LIST:
	case SET:
	case RECORD: {
		IRBuilder<>* Builder = context->getBuilder();
		Function* F = context->getGlobalFunction();
		/* mem_value contains the address of a tokenNo */
		/* Must return tokens[tokenNo].end - tokens[tokenNo].start */

		AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];
		Value* val_offset = Builder->CreateLoad(mem_value.mem);

		PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
		AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
					string(var_tokenPtr), context->CreateJSMNStruct());
		Value* token_i = context->getArrayElem(mem_tokens, val_offset);
		Builder->CreateStore(token_i, mem_tokens_shifted);
		// 0: jsmntype_t type;
		// 1: int start;
		// 2: int end;
		// 3: int size;
		Value* token_i_start = context->getStructElem(mem_tokens_shifted, 1);
		Value* token_i_end = context->getStructElem(mem_tokens_shifted, 2);
		return Builder->CreateSub(token_i_end,token_i_start);
	}
	default: {
		string error_msg = string("[JSON Plugin]: Unknown datatype");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	}
}

JSONPlugin::~JSONPlugin()
{
//	delete[] tokens;
}

}
