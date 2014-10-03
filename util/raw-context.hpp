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

#include "jsoncpp/json/json.h"
#include "semi_index/json_semi_index.hpp"
#include "semi_index/path_parser.hpp"

#include "common/atois.hpp"
#include "common/common.hpp"
#include "util/raw-catalog.hpp"

using json::path::path_element_t;
using json::path::path_t;
using json::path::path_list_t;
using semi_index::json_semi_index;

//Forward Declaration
class JSONObject;

class RawContext {
public:

	RawContext(const string& moduleName);
	~RawContext() {
		LOG(WARNING) << "[RawContext: ] Hollow Destructor";
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

	AllocaInst* const CreateEntryBlockAlloca(Function *TheFunction,
											 const std::string &VarName,
											 Type* varType) const;

	ExecutionEngine* getExecEngine() { return TheExecutionEngine; }

	Function* const getGlobalFunction() const;
	Module* const getModule() const;
	IRBuilder<>* const getBuilder() const;
	Function* const getFunction(std::string funcName) const;

	ConstantInt* createInt32(int val);
	ConstantInt* createInt64(int val);
	ConstantInt* createTrue();
	ConstantInt* createFalse();
	//Stub
	Type* CreateCustomType(char* typeName);
	StructType* CreateCustomStruct(std::vector<Type*> innerTypes);
	StructType* CreateJSONPosStruct();
	Value* CreateGlobalString(char* str);
	PointerType* getPointerType(Type* type);

	//Utility functions, similar to ones from Impala
	void CreateForLoop(const string& cond, const string& body,
				const string& inc, const string& end, BasicBlock** cond_block,
				BasicBlock** body_block, BasicBlock** inc_block,
				BasicBlock** end_block, BasicBlock* insert_before = NULL);

	void CreateIfElseBlocks(Function* fn, const string& if_name,
				const string& else_name, BasicBlock** if_block,
				BasicBlock** else_block, BasicBlock* insert_before = NULL);

	Value* CastPtrToLlvmPtr(PointerType* type, const void* ptr);
	//Not used atm
	void CodegenMemcpy(Value* dst, Value* src, int size);

	void registerFunction(const char*, Function*);
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
};

void registerFunctions(RawContext& context);
//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

/// putchard - putchar that takes a double and returns 0.
extern "C" double putchari(int X);

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" int printi(int X);

extern "C" int printFloat(double X);

extern "C" int printc(char* X);

extern "C" int atoi_llvm(const char* X);

extern "C" void insertIntKeyToHT(char* HTname, int key, void* value,
		int type_size);

extern "C"
void** probeIntHT(char* HTname, int key, int typeIndex);

extern "C" bool eofJSON(char* jsonName);

extern "C" JSONObject getJSONPositions(char* jsonName, int attrNo);

extern "C" int getJSONInt(char* jsonName, int attrNo);

extern "C" double getJSONDouble(char* jsonName, int attrNo);

#endif /* RAW_CONTEXT_HPP_ */
