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
#include "values/expressionTypes.hpp"
#include "memory/memory-allocator.hpp"

#include "llvm/IR/LegacyPassManager.h"

#define MODULEPASS 0

//#ifdef DEBUG
#define DEBUGCTX
//#endif

//Forward Declaration
class JSONObject;

class RawContext {
public:

	RawContext(const string& moduleName, bool setGlobalFunction = true);
	~RawContext() {
		LOG(WARNING)<< "[RawContext: ] Destructor";
		//XXX Has to be done in an appropriate sequence - segfaults otherwise
//		delete Builder;
//			delete TheFPM;
//			delete TheExecutionEngine;
//			delete TheFunction;
//			delete llvmContext;
//			delete TheFunction;
	}

	LLVMContext& getLLVMContext() {
		return TheContext;
	}

	virtual void prepareFunction(Function *F);

	ExecutionEngine const * const getExecEngine() {return TheExecutionEngine;}

	virtual void setGlobalFunction(Function *F);
	Function * getGlobalFunction() const {return TheFunction;}
	Module * getModule() const {return TheModule;}
	IRBuilder<> * getBuilder() const {return TheBuilder;}
	Function* const getFunction(string funcName) const;

	ConstantInt* createInt8(char val);
	ConstantInt* createInt32(int val);
	ConstantInt* createInt64(int val);
	ConstantInt* createInt64(size_t val);
	ConstantInt* createTrue();
	ConstantInt* createFalse();
	
    virtual size_t getSizeOf(llvm::Type  * type) const;
    virtual size_t getSizeOf(llvm::Value * val ) const;

	Type* CreateCustomType(char* typeName);
	StructType* CreateJSMNStruct();
	StructType* CreateStringStruct();
	PointerType* CreateJSMNStructPtr();
	StructType* CreateJSONPosStruct();
	StructType* CreateCustomStruct(vector<Type*> innerTypes);
	StructType* ReproduceCustomStruct(list<typeID> innerTypes);
	/**
	 * Does not involve AllocaInst, but still is a memory position
	 * NOTE: 1st elem of Struct is 0!!
	 */
	Value* getStructElem(Value* mem_struct, int elemNo);
	Value* getStructElem(AllocaInst* mem_struct, int elemNo);
	void updateStructElem(Value *toStore, Value* mem_struct, int elemNo);
	Value* getStructElemMem(Value* mem_struct, int elemNo);
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

	BasicBlock * CreateIfBlock(Function* fn,
								const string& if_label,
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
	void CodegenMemcpy(Value* dst, Value* src, Value* size);

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
		Type* int64_type = Type::getInt64Ty(getLLVMContext());
		Type* keyType = int64_type;
		Type* bucketSizeType = int64_type;
		types_htMetadata.push_back(keyType);
		types_htMetadata.push_back(bucketSizeType);
		int htMetadataSize = (keyType->getPrimitiveSizeInBits() / 8);
		htMetadataSize += (bucketSizeType->getPrimitiveSizeInBits() / 8);

		//Result type specified
		StructType *metadataType = llvm::StructType::get(getLLVMContext(),types_htMetadata);
		return metadataType;
	}

	Value * const getMemResultCtr() {return mem_resultCtr;}

protected:
	LLVMContext TheContext;
	Module * TheModule;
	IRBuilder<> * TheBuilder;

	//Used to include optimization passes
	legacy::FunctionPassManager * TheFPM;
#if MODULEPASS
	ModulePassManager * TheMPM;
#endif

	//JIT Driver
	ExecutionEngine * TheExecutionEngine;
	Function * TheFunction;
	map<string, Function*> availableFunctions;

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
	Value * mem_resultCtr;

	/**
	 * Helper function to create the LLVM objects required for JIT execution. */
	virtual void createJITEngine();
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
