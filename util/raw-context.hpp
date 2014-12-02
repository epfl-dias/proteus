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
#include "util/raw-catalog.hpp"

//Forward Declaration
class JSONObject;

class RawContext {
public:

	RawContext(const string& moduleName);
	~RawContext() {
		LOG(WARNING) << "[RawContext: ] Destructor";
		//XXX Has to be done in an appropriate sequence - segfaults otherwise
			delete TheBuilder;
//			delete TheFPM;
//			delete TheExecutionEngine;
//			delete TheFunction;
			delete llvmContext;
			//delete TheFunction;
	}
	LLVMContext& getLLVMContext() {
		return *llvmContext;
	}

	void prepareFunction(Function *F);
	void* jit(Function* F);

	ExecutionEngine* getExecEngine() 						{ return TheExecutionEngine; }

	Function* const getGlobalFunction() 		 const		{ return TheFunction; }
	Module* const getModule() 					 const 		{ return TheModule; }
	IRBuilder<>* const getBuilder() 			 const 		{ return TheBuilder; }
	Function* const getFunction(string funcName) const;

	ConstantInt* createInt32(int val);
	ConstantInt* createInt64(int val);
	ConstantInt* createTrue();
	ConstantInt* createFalse();

	Type* CreateCustomType(char* typeName);
	StructType* CreateJSMNStruct();
	StructType* CreateStringStruct();
	PointerType* CreateJSMNStructPtr();
	StructType* CreateJSONPosStruct();
	StructType* CreateCustomStruct(std::vector<Type*> innerTypes);
	Value* getStructElem(AllocaInst* mem_struct, int elemNo);
	Value* CreateGlobalString(char* str);
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
	Value* getArrayElem(AllocaInst* mem_ptr, PointerType* type, Value* offset);

	//Not used atm
	void CodegenMemcpy(Value* dst, Value* src, int size);

	void registerFunction(const char*, Function*);
	BasicBlock* getEndingBlock()							{ return codeEnd; }
	void setEndingBlock(BasicBlock* codeEnd)				{ this->codeEnd = codeEnd; }
private:
	LLVMContext *llvmContext;
	Module *TheModule;
	IRBuilder<> *TheBuilder;
	//Used to include optimization passes
	FunctionPassManager *TheFPM;
	//JIT Driver
	ExecutionEngine *TheExecutionEngine;
	Function* TheFunction;
	std::map<std::string, Function*> availableFunctions;
	//
	BasicBlock* codeEnd;
};

typedef struct StringObject	{
	char* start;
	int len;
} StringObject;

int compareTokenString64(const char* buf, size_t start, size_t end, const char* candidate);
void registerFunctions(RawContext& context);
//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

extern "C" int printi(int X);

extern "C" int printi64(size_t X);

extern "C" int printFloat(double X);

extern "C" int printc(char* X);

extern "C" void printBoolean(bool X);

extern "C" int atoi_llvm(const char* X);

extern "C" void insertIntKeyToHT(char* HTname, int key, void* value,
		int type_size);

extern "C"
void** probeIntHT(char* HTname, int key, int typeIndex);

extern "C" int compareTokenString(const char* buf, int start, int end, const char* candidate);

extern "C" bool equalStrings(StringObject obj1, StringObject obj2);

extern "C" bool convertBoolean(const char* buf, int start, int end);

extern "C" bool convertBoolean64(const char* buf, size_t start, size_t end);

extern "C" int atois(const char* buf, int len);

extern "C" size_t hashInt(int toHash);

extern "C" size_t hashDouble(double toHash);

extern "C" size_t hashStringC(char* toHash, size_t start, size_t end);

extern "C" size_t hashString(string toHash);

extern "C" size_t hashStringObject(StringObject obj);

extern "C" size_t hashBoolean(bool toHash);

extern "C" size_t combineHashes(size_t hash1, size_t hash2);

extern "C" size_t combineHashesNoOrder(size_t hash1, size_t hash2);

#endif /* RAW_CONTEXT_HPP_ */
