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
/** Original code of the following functions derived from
 *  the source code of Impala (https://github.com/cloudera/Impala/):
 *  CreateForLoop,
 *  CreateIfElseBlocks,
 *  CastPtrToLlvmPtr,
 *  CodegenMemcpy
 *
 *  Original code of the following functions derived from and calling
 *  the source code of semi_index (https://github.com/ot/semi_index):
 *  getJSONPositions
 */

#ifndef RAW_CONTEXT_HPP_
#define RAW_CONTEXT_HPP_

#include "common/common.hpp"
#include "util/joins/radix-join.hpp"
#include "memory/memory-allocator.hpp"

#include <x86intrin.h>

//#ifdef DEBUG
#define DEBUGCTX
//#endif

//Forward Declaration
class JSONObject;

class RawContext {
public:

	RawContext(const string& moduleName);
	~RawContext() {
		LOG(WARNING)<< "[RawContext: ] Destructor";
		//XXX Has to be done in an appropriate sequence - segfaults otherwise
		delete Builder;
//			delete TheFPM;
//			delete TheExecutionEngine;
//			delete TheFunction;
//			delete llvmContext;
//			delete TheFunction;
	}
	LLVMContext& getLLVMContext() {
		return *llvmContext;
	}

	void prepareFunction(Function *F);
	void* jit(Function* F);

	ExecutionEngine* getExecEngine() {return TheExecutionEngine;}

	Function* const getGlobalFunction() const {return TheFunction;}
	Module* const getModule() const {return TheModule;}
	IRBuilder<>* const getBuilder() const {return Builder;}
	Function* const getFunction(string funcName) const;

	ConstantInt* createInt8(char val);
	ConstantInt* createInt32(int val);
	ConstantInt* createInt64(int val);
	ConstantInt* createInt64(size_t val);
	ConstantInt* createTrue();
	ConstantInt* createFalse();

	Type* CreateCustomType(char* typeName);
	StructType* CreateJSMNStruct();
	StructType* CreateStringStruct();
	PointerType* CreateJSMNStructPtr();
	StructType* CreateJSONPosStruct();
	StructType* CreateCustomStruct(std::vector<Type*> innerTypes);
	/**
	 * Does not involve AllocaInst, but still is a memory position
	 */
	Value* getStructElem(Value* mem_struct, int elemNo);
	Value* getStructElem(AllocaInst* mem_struct, int elemNo);
	Value* CreateGlobalString(char* str);
	Value* CreateGlobalString(const char* str);
	PointerType* getPointerType(Type* type);

	//Utility functions, similar to ones from Impala
	AllocaInst* CreateEntryBlockAlloca(Function *TheFunction,
			const std::string &VarName,
			Type* varType);
	void CreateForLoop(const string& cond, const string& body,
			const string& inc, const string& end, BasicBlock** cond_block,
			BasicBlock** body_block, BasicBlock** inc_block,
			BasicBlock** end_block, BasicBlock* insert_before = NULL);

	void CreateIfElseBlocks(Function* fn, const string& if_name,
			const string& else_name, BasicBlock** if_block,
			BasicBlock** else_block, BasicBlock* insert_before = NULL);
	void CreateIfBlock(Function* fn, const string& if_name,
			BasicBlock** if_block,
			BasicBlock* insert_before = NULL);
	Value* CastPtrToLlvmPtr(PointerType* type, const void* ptr);
	Value* getArrayElem(AllocaInst* mem_ptr, Value* offset);
	Value* getArrayElem(Value* val_ptr, Value* offset);
	/**
	 * Does not involve AllocaInst, but still returns a memory position
	 */
	Value* getArrayElemMem(Value* val_ptr, Value* offset);

	//Not used atm
	void CodegenMemcpy(Value* dst, Value* src, int size);

	void registerFunction(const char*, Function*);
	BasicBlock* getEndingBlock() {return codeEnd;}
	void setEndingBlock(BasicBlock* codeEnd) {this->codeEnd = codeEnd;}
	BasicBlock* getCurrentEntryBlock() {return currentCodeEntry;}
	void setCurrentEntryBlock(BasicBlock* codeEntry) {this->currentCodeEntry = codeEntry;}

	/**
	 * Not sure the HT methods belong here
	 */
	//Metadata maintained per bucket.
	//Will probably use an array of such structs per HT
	StructType* getHashtableMetadataType() {
		vector<Type*> types_htMetadata = vector<Type*>();
		Type* int64_type = Type::getInt64Ty(*llvmContext);
		Type* keyType = int64_type;
		Type* bucketSizeType = int64_type;
		types_htMetadata.push_back(keyType);
		types_htMetadata.push_back(bucketSizeType);
		int htMetadataSize = (keyType->getPrimitiveSizeInBits() / 8);
		htMetadataSize += (bucketSizeType->getPrimitiveSizeInBits() / 8);

		//Result type specified
		StructType *metadataType = llvm::StructType::get(*llvmContext,types_htMetadata);
		return metadataType;
	}

	Value* getMemResultCtr() {return mem_resultCtr;}

private:
	LLVMContext *llvmContext;
	Module *TheModule;
	IRBuilder<> *Builder;
	//Used to include optimization passes
	FunctionPassManager *TheFPM;
	//JIT Driver
	ExecutionEngine *TheExecutionEngine;
	Function* TheFunction;
	std::map<std::string, Function*> availableFunctions;
	//Last (current) basic block. This changes every time a new scan is triggered
	BasicBlock* codeEnd;
	//Current entry basic block. This changes every time a new scan is triggered
	BasicBlock* currentCodeEntry;

	/**
	 * Basic stats / info to be used during codegen
	 */
	//XXX used to keep a counter of final output results
	//and be utilized in actions such as flushing out delimiters
	//NOTE: Must check whether sth similar is necessary for nested collections
	Value* mem_resultCtr;

};

typedef struct StringObject {
	char* start;
	int len;
} StringObject;

typedef struct HashtableBucketMetadata {
	size_t hashKey;
	size_t bucketSize;
} HashtableBucketMetadata;

#endif /* RAW_CONTEXT_HPP_ */
