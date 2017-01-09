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

#include "util/raw-context.hpp"

void RawContext::createJITEngine() {
	LLVMLinkInMCJIT();
	LLVMInitializeNativeTarget();
	LLVMInitializeNativeAsmPrinter();
	LLVMInitializeNativeAsmParser();

	// Create the JIT.  This takes ownership of the module.
	string ErrStr;
	TheExecutionEngine =
		EngineBuilder(std::unique_ptr<Module>(TheModule)).setErrorStr(&ErrStr).create();
	if (TheExecutionEngine == nullptr) {
		fprintf(stderr, "Could not create ExecutionEngine: %s\n",
				ErrStr.c_str());
		exit(1);
	}
}

static void __attribute__((unused)) addOptimizerPipelineDefault(legacy::FunctionPassManager * TheFPM) {
	//Provide basic AliasAnalysis support for GVN.
	TheFPM->add(createBasicAAWrapperPass());
	// Promote allocas to registers.
	TheFPM->add(createPromoteMemoryToRegisterPass());
	//Do simple "peephole" optimizations and bit-twiddling optzns.
	TheFPM->add(createInstructionCombiningPass());
	// Reassociate expressions.
	TheFPM->add(createReassociatePass());
	// Eliminate Common SubExpressions.
	TheFPM->add(createGVNPass());
	// Simplify the control flow graph (deleting unreachable blocks, etc).
	TheFPM->add(createCFGSimplificationPass());
	// Aggressive Dead Code Elimination. Make sure work takes place
	TheFPM->add(createAggressiveDCEPass());
}

static void __attribute__((unused)) addOptimizerPipelineInlining(legacy::FunctionPassManager * TheFPM) {
	/* Inlining: Not sure it works */
	TheFPM->add(createFunctionInliningPass());
	TheFPM->add(createAlwaysInlinerPass());
}

static void __attribute__((unused)) addOptimizerPipelineVectorization(legacy::FunctionPassManager * TheFPM) {
	/* Vectorization */
	TheFPM->add(createBBVectorizePass());
	TheFPM->add(createLoopVectorizePass());
	TheFPM->add(createSLPVectorizerPass());
}

RawContext::RawContext(const string& moduleName) {
	TheModule = new Module(moduleName, getLLVMContext());
	TheBuilder = new IRBuilder<>(getLLVMContext());

	TheExecutionEngine = nullptr;
	TheFunction = nullptr;
	codeEnd = nullptr;
	//availableFunctions = map<string, Function*>();

	/* OPTIMIZER PIPELINE */
	//TheFPM = new FunctionPassManager(getModule());
	TheFPM = new legacy::FunctionPassManager(getModule());

	addOptimizerPipelineDefault(TheFPM);
	//addOptimizerPipelineInlining(TheFPM);
	//addOptimizerPipelineVectorization(TheFPM);

	TheFPM->doInitialization();

	llvm::Type* int_type = Type::getInt32Ty(getLLVMContext());
	vector<Type*> Ints(1,int_type);
	FunctionType *FT = FunctionType::get(Type::getInt32Ty(getLLVMContext()),Ints, false);
	Function *F = Function::Create(FT, Function::ExternalLinkage,
		moduleName, getModule());

	//Setting the 'global' function
	TheFunction = F;
	// Create a new basic block to start insertion into.
	BasicBlock *BB = BasicBlock::Create(getLLVMContext(), "entry", F);
	getBuilder()->SetInsertPoint(BB);

	/**
	 * Preparing global info to be maintained
	 */
	llvm::Type* int64_type = Type::getInt64Ty(getLLVMContext());
	mem_resultCtr = this->CreateEntryBlockAlloca(F,"resultCtr",int64_type);
	getBuilder()->CreateStore(this->createInt64(0),mem_resultCtr);

	createJITEngine();
}

void RawContext::prepareFunction(Function *F) {

	//FIXME Have a (tmp) return value for now at this point
	getBuilder()->CreateRet(getBuilder()->getInt32(114));

	LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
#ifdef DEBUGCTX
//	getModule()->dump();
#endif
	// Validate the generated code, checking for consistency.
	verifyFunction(*F);

	// Optimize the function.
	TheFPM->run(*F);

	// JIT the function, returning a function pointer.
	TheExecutionEngine->finalizeObject();
	void *FPtr = TheExecutionEngine->getPointerToFunction(F);

	int (*FP)(void) = (int (*)(void))FPtr;
	assert(FP != nullptr && "Code generation failed!");


	//TheModule->dump();
	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	int jitFuncResult = FP();
	//LOG(INFO) << "Mock return value of generated function " << FP(11);
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("(Already compiled) Execution took %f seconds\n",diff(t0, t1));
	cout << "Return flag: " << jitFuncResult << endl;

	TheFPM = 0;
	//Dump to see final (optimized) form
#ifdef DEBUGCTX
//	F->dump();
#endif
}

Function* const RawContext::getFunction(string funcName) const {
	map<string, Function*>::const_iterator it;
	it = availableFunctions.find(funcName);
	if (it == availableFunctions.end()) {
			throw runtime_error(string("Unknown function name: ") + funcName);
	}
	return it->second;
}

void RawContext::CodegenMemcpy(Value* dst, Value* src, int size) {
	LLVMContext& ctx = getLLVMContext();
	// Cast src/dst to int8_t*.  If they already are, this will get optimized away
	//  DCHECK(PointerType::classof(dst->getType()));
	//  DCHECK(PointerType::classof(src->getType()));
	llvm::PointerType* ptr_type = nullptr;

	Value* false_value_ = ConstantInt::get(ctx,
			APInt(1, false, true));
	Value* size_ = ConstantInt::get(ctx, APInt(32, size));
	Value* zero = ConstantInt::get(ctx, APInt(32, 0));

	dst = getBuilder()->CreateBitCast(dst, ptr_type);
	src = getBuilder()->CreateBitCast(src, ptr_type);

	// Get intrinsic function.
	Function* memcpy_fn = availableFunctions[string("memcpy")];
	if (memcpy_fn == NULL) {
		throw runtime_error(string("Could not load memcpy intrinsic"));
	}

	// The fourth argument is the alignment.  For non-zero values, the caller
	// must guarantee that the src and dst values are aligned to that byte boundary.
	// TODO: We should try to take advantage of this since our tuples are well aligned.
	Type* intType = Type::getInt32Ty(ctx);
	Value* args[] = { dst, src, size_, zero, false_value_  // is_volatile.
			};
	getBuilder()->CreateCall(memcpy_fn, args);
}

void RawContext::CodegenMemcpy(Value* dst, Value* src, Value* size) {
	LLVMContext& ctx = getLLVMContext();
	// Cast src/dst to int8_t*.  If they already are, this will get optimized away
	//  DCHECK(PointerType::classof(dst->getType()));
	//  DCHECK(PointerType::classof(src->getType()));
	llvm::PointerType* ptr_type;

	Value* false_value_ = ConstantInt::get(ctx,
			APInt(1, false, true));

	Value* zero = ConstantInt::get(ctx, APInt(32, 0));

	dst->getType()->dump();
	cout << endl;
	src->getType()->dump();
//	dst = getBuilder()->CreateBitCast(dst, ptr_type);
//	src = getBuilder()->CreateBitCast(src, ptr_type);

	// Get intrinsic function.
	Function* memcpy_fn = availableFunctions[string("memcpy")];
	if (memcpy_fn == NULL) {
		throw runtime_error(string("Could not load memcpy intrinsic"));
	}

	// The fourth argument is the alignment.  For non-zero values, the caller
	// must guarantee that the src and dst values are aligned to that byte boundary.
	// TODO: We should try to take advantage of this since our tuples are well aligned.
	Type* intType = Type::getInt32Ty(ctx);
	Value* args[] = { dst, src, size, zero, false_value_  // is_volatile.
			};
	getBuilder()->CreateCall(memcpy_fn, args);
}

ConstantInt* RawContext::createInt8(char val) {
	return ConstantInt::get(getLLVMContext(), APInt(8, val));
}

ConstantInt* RawContext::createInt32(int val) {
	return ConstantInt::get(getLLVMContext(), APInt(32, val));
}

ConstantInt* RawContext::createInt64(int val) {
	return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt* RawContext::createInt64(size_t val) {
	return ConstantInt::get(getLLVMContext(), APInt(64, val));
}

ConstantInt* RawContext::createTrue() {
	return ConstantInt::get(getLLVMContext(), APInt(1, 1));
}

ConstantInt* RawContext::createFalse() {
	return ConstantInt::get(getLLVMContext(), APInt(1, 0));
}

Value* RawContext::CastPtrToLlvmPtr(PointerType* type, const void* ptr) {
	Constant* const_int = ConstantInt::get(Type::getInt64Ty(getLLVMContext()),(uint64_t) ptr);
	Value* llvmPtr = ConstantExpr::getIntToPtr(const_int, type);
	return llvmPtr;
}

Value* RawContext::getArrayElem(AllocaInst* mem_ptr, Value* offset)	{
	Value* val_ptr = getBuilder()->CreateLoad(mem_ptr, "mem_ptr");
	Value* shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
	Value* val_shifted = getBuilder()->CreateLoad(shiftedPtr,"val_shifted");
	return val_shifted;
}

Value* RawContext::getArrayElem(Value* val_ptr, Value* offset)	{
	Value* shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
	Value* val_shifted = getBuilder()->CreateLoad(shiftedPtr,"val_shifted");
	return val_shifted;
}

Value* RawContext::getArrayElemMem(Value* val_ptr, Value* offset)	{
	Value* shiftedPtr = getBuilder()->CreateInBoundsGEP(val_ptr, offset);
	return shiftedPtr;
}

Value* RawContext::getStructElem(Value* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
	Value* val_struct_shifted =  getBuilder()->CreateLoad(mem_struct_shifted);
	return val_struct_shifted;
}

Value* RawContext::getStructElemMem(Value* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
	return mem_struct_shifted;
}

Value* RawContext::getStructElem(AllocaInst* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = getBuilder()->CreateGEP(mem_struct, idxList);
	Value* val_struct_shifted =  getBuilder()->CreateLoad(mem_struct_shifted);
	return val_struct_shifted;
}

void RawContext::updateStructElem(Value *toStore, Value* mem_struct,
		int elemNo) {
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* structPtr = getBuilder()->CreateGEP(mem_struct, idxList);
	getBuilder()->CreateStore(toStore, structPtr);
}

void RawContext::CreateForLoop(const string& cond, const string& body,
		const string& inc, const string& end, BasicBlock** cond_block,
		BasicBlock** body_block, BasicBlock** inc_block, BasicBlock** end_block,
		BasicBlock* insert_before) {
	Function* fn = TheFunction;
	LLVMContext& ctx = getLLVMContext();
	*cond_block = BasicBlock::Create(ctx, string(cond), fn,	insert_before);
	*body_block = BasicBlock::Create(ctx, string(body), fn,	insert_before);
	*inc_block = BasicBlock::Create(ctx, string(inc), fn, insert_before);
	*end_block = BasicBlock::Create(ctx, string(end), fn, insert_before);
}

void RawContext::CreateIfElseBlocks(Function* fn, const string& if_label,
		const string& else_label, BasicBlock** if_block, BasicBlock** else_block,
		BasicBlock* insert_before) {
	LLVMContext& ctx = getLLVMContext();
	*if_block = BasicBlock::Create(ctx, if_label, fn, insert_before);
	*else_block = BasicBlock::Create(ctx, else_label, fn, insert_before);
}

void RawContext::CreateIfBlock(Function* fn, const string& if_label,
		BasicBlock** if_block, BasicBlock* insert_before) {
	*if_block = BasicBlock::Create(getLLVMContext(), if_label, fn, insert_before);
}

AllocaInst* RawContext::CreateEntryBlockAlloca(Function *TheFunction,
		const string &VarName, Type* varType) {
	IRBuilder<> TmpBuilder(&TheFunction->getEntryBlock(),
			TheFunction->getEntryBlock().begin());
	return TmpBuilder.CreateAlloca(varType, 0, VarName.c_str());
}

Value* RawContext::CreateGlobalString(char* str) {
	LLVMContext& ctx = getLLVMContext();
	ArrayType* ArrayTy_0 = ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

	GlobalVariable* gvar_array__str = new GlobalVariable(*getModule(),
	/*Type=*/ArrayTy_0,
	/*isConstant=*/true,
	/*Linkage=*/GlobalValue::PrivateLinkage,
	/*Initializer=*/0, // has initializer, specified below
			/*Name=*/".str");

	Constant *tmpHTname = ConstantDataArray::getString(ctx, str,true);
	PointerType* charPtrType = PointerType::get(IntegerType::get(ctx, 8), 0);
	AllocaInst* AllocaName = CreateEntryBlockAlloca(TheFunction, string("globalStr"), charPtrType);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(0));
	Constant* shifted = ConstantExpr::getGetElementPtr(ArrayTy_0,gvar_array__str,idxList);
	gvar_array__str->setInitializer(tmpHTname);


	LOG(INFO) << "[CreateGlobalString: ] " << str;
	getBuilder()->CreateStore(shifted, AllocaName);
	Value* globalStr = getBuilder()->CreateLoad(AllocaName);
	return globalStr;
}

Value* RawContext::CreateGlobalString(const char* str) {
	LLVMContext& ctx = getLLVMContext();
	ArrayType* ArrayTy_0 = ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

	GlobalVariable* gvar_array__str = new GlobalVariable(*getModule(),
	/*Type=*/ArrayTy_0,
	/*isConstant=*/true,
	/*Linkage=*/GlobalValue::PrivateLinkage,
	/*Initializer=*/0, // has initializer, specified below
			/*Name=*/".str");

	Constant *tmpHTname = ConstantDataArray::getString(ctx, str,true);
	PointerType* charPtrType = PointerType::get(IntegerType::get(ctx, 8), 0);
	AllocaInst* AllocaName = CreateEntryBlockAlloca(TheFunction, string("globalStr"), charPtrType);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(0));
	Constant* shifted = ConstantExpr::getGetElementPtr(ArrayTy_0,gvar_array__str,idxList);
	gvar_array__str->setInitializer(tmpHTname);


	LOG(INFO) << "[CreateGlobalString: ] " << str;
	getBuilder()->CreateStore(shifted, AllocaName);
	Value* globalStr = getBuilder()->CreateLoad(AllocaName);
	return globalStr;
}

PointerType* RawContext::getPointerType(Type* type) {
	return PointerType::get(type, 0);
}

StructType* RawContext::CreateCustomStruct(vector<Type*> innerTypes) {
	llvm::StructType* valueType = llvm::StructType::get(getLLVMContext(),innerTypes);
	return valueType;
}

StructType* RawContext::ReproduceCustomStruct(list<typeID> innerTypes) {
	LLVMContext& ctx = getLLVMContext();
	vector<Type*> llvmTypes;
	list<typeID>::iterator it;
	for (it = innerTypes.begin(); it != innerTypes.end(); it++) {
		switch (*it) {
		case INT: {
			Type* int32_type = Type::getInt32Ty(ctx);
			llvmTypes.push_back(int32_type);
			break;
		}
		case BOOL: {
			Type* int1_type = Type::getInt1Ty(ctx);
			llvmTypes.push_back(int1_type);
			break;
		}
		case FLOAT: {
			Type* float_type = Type::getDoubleTy(ctx);
			llvmTypes.push_back(float_type);
			break;
		}
		case INT64: {
			Type* int64_type = Type::getInt64Ty(ctx);
			llvmTypes.push_back(int64_type);
			break;
		}
		case STRING:
		case RECORD:
		case LIST:
		case BAG:
		case SET:
		case COMPOSITE:
		default: {
			string error_msg = "No explicit caching support for this type yet";
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
	}
	llvm::StructType* valueType = llvm::StructType::get(ctx, llvmTypes);
	return valueType;
}

StructType* RawContext::CreateJSONPosStruct() {
	llvm::Type* int64_type = Type::getInt64Ty(getLLVMContext());
	vector<Type*> json_pos_types;
	json_pos_types.push_back(int64_type);
	json_pos_types.push_back(int64_type);
	return CreateCustomStruct(json_pos_types);
}

PointerType* RawContext::CreateJSMNStructPtr()	{
	Type* jsmnStructType = CreateJSMNStruct();
	PointerType* ptr_jsmnStructType = PointerType::get(jsmnStructType,0);
	return ptr_jsmnStructType;
}

StructType* RawContext::CreateJSMNStruct() {
	LLVMContext& ctx = getLLVMContext();
	llvm::Type* int32_type = Type::getInt32Ty(ctx);
	llvm::Type* int8_type = Type::getInt8Ty(ctx);
	llvm::Type* int16_type = Type::getInt16Ty(ctx);
#ifndef JSON_TIGHT
	vector<Type*> jsmn_pos_types;
	jsmn_pos_types.push_back(int32_type);
	jsmn_pos_types.push_back(int32_type);
	jsmn_pos_types.push_back(int32_type);
	jsmn_pos_types.push_back(int32_type);
#endif
#ifdef JSON_TIGHT
	vector<Type*> jsmn_pos_types;
	jsmn_pos_types.push_back(int8_type);
	jsmn_pos_types.push_back(int16_type);
	jsmn_pos_types.push_back(int16_type);
	jsmn_pos_types.push_back(int8_type);
#endif
	return CreateCustomStruct(jsmn_pos_types);
}

StructType* RawContext::CreateStringStruct() {
	LLVMContext& ctx = getLLVMContext();
	llvm::Type* int32_type = Type::getInt32Ty(ctx);
	llvm::Type* char_type = Type::getInt8Ty(ctx);
	PointerType* ptr_char_type = PointerType::get(char_type,0);
	vector<Type*> string_obj_types;
	string_obj_types.push_back(ptr_char_type);
	string_obj_types.push_back(int32_type);

	return CreateCustomStruct(string_obj_types);
}

//Provide support for some extern functions
void RawContext::registerFunction(const char* funcName, Function* func)	{
	availableFunctions[funcName] = func;
}
