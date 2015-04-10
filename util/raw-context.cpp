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

RawContext::RawContext(const string& moduleName) {
	llvmContext = new LLVMContext();
	LLVMContext& ctx = *llvmContext;
	Builder = new IRBuilder<>(ctx);
	TheFPM = 0;
	TheExecutionEngine = 0;
	TheFunction = 0;
	codeEnd = NULL;
	availableFunctions = map<string, Function*>();

	InitializeNativeTarget();
	TheModule = new Module(moduleName, ctx);

	// Create the JIT.  This takes ownership of the module.
	string ErrStr;
	TheExecutionEngine = 
		EngineBuilder(TheModule).setErrorStr(&ErrStr).create();
	if (!TheExecutionEngine) {
		fprintf(stderr, "Could not create ExecutionEngine: %s\n",
				ErrStr.c_str());
		exit(1);
	}

	PassManager mpm;
	FunctionPassManager* OurFPM = new FunctionPassManager(TheModule);

	PassManagerBuilder pmb;
	pmb.OptLevel=0;
	pmb.populateModulePassManager(mpm);
	pmb.populateFunctionPassManager(*OurFPM);

	/* OPTIMIZER PIPELINE */
	// Set up the optimizer pipeline.  Start with registering info about how the
	// target lays out data structures.
	OurFPM->add(new DataLayout(*TheExecutionEngine->getDataLayout()));
	//Provide basic AliasAnalysis support for GVN.
	OurFPM->add(createBasicAliasAnalysisPass());
	// Promote allocas to registers.
	OurFPM->add(createPromoteMemoryToRegisterPass());
	//Do simple "peephole" optimizations and bit-twiddling optzns.
	OurFPM->add(createInstructionCombiningPass());
	// Reassociate expressions.
	OurFPM->add(createReassociatePass());
	// Eliminate Common SubExpressions.
	OurFPM->add(createGVNPass());
	// Simplify the control flow graph (deleting unreachable blocks, etc).
	OurFPM->add(createCFGSimplificationPass());
	// Aggressive Dead Code Elimination. Make sure work takes place
	OurFPM->add(createAggressiveDCEPass());

	/* Inlining: Not sure it works */
	mpm.add(createFunctionInliningPass());
	mpm.add(createAlwaysInlinerPass());
	/* Vectorization */
	mpm.add(createBBVectorizePass());
	mpm.add(createLoopVectorizePass());
	mpm.add(createSLPVectorizerPass());

	mpm.run(*TheModule);
	OurFPM->doInitialization();
	TheFPM = OurFPM;

	llvm::Type* int_type = Type::getInt32Ty(ctx);
	vector<Type*> Ints(1,int_type);
	FunctionType *FT = FunctionType::get(Type::getInt32Ty(ctx),Ints, false);
	//registerFunctions(*this);
	Function *F = Function::Create(FT, Function::ExternalLinkage, 
		moduleName, TheModule);

	//Setting the 'global' function
	TheFunction = F;
	// Create a new basic block to start insertion into.
	BasicBlock *BB = BasicBlock::Create(ctx, "entry", F);
	Builder->SetInsertPoint(BB);

	/**
	 * Preparing global info to be maintained
	 */
	llvm::Type* int64_type = Type::getInt64Ty(ctx);
	mem_resultCtr = this->CreateEntryBlockAlloca(F,"resultCtr",int64_type);
	Builder->CreateStore(this->createInt64(0),mem_resultCtr);
}

void RawContext::prepareFunction(Function *F) {

	//FIXME Have a (tmp) return value for now at this point
	Builder->CreateRet(Builder->getInt32(114));

	LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
#ifdef DEBUGCTX
//	getModule()->dump();
#endif
	// Validate the generated code, checking for consistency.
	verifyFunction(*F);

	// Optimize the function.
	TheFPM->run(*F);

	// JIT the function, returning a function pointer.
	void *FPtr = TheExecutionEngine->getPointerToFunction(F);

	int (*FP)(int) = (int (*)(int))(intptr_t)FPtr;


	//	TheModule->dump();
	//Run function
	struct timespec t0, t1;
	clock_gettime(CLOCK_REALTIME, &t0);
	FP(11);
	//LOG(INFO) << "Mock return value of generated function " << FP(11);
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("(Already compiled) Execution took %f seconds\n",diff(t0, t1));

	TheFPM = 0;
	//Dump to see final (optimized) form
#ifdef DEBUGCTX
//	F->dump();
#endif

}

void* RawContext::jit(Function* F) {
	// JIT the function, returning a function pointer.
	//void *FPtr = TheExecutionEngine->getPointerToFunction(F);
	return TheExecutionEngine->getPointerToFunction(F);
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
	LLVMContext& ctx = *llvmContext;
	// Cast src/dst to int8_t*.  If they already are, this will get optimized away
	//  DCHECK(PointerType::classof(dst->getType()));
	//  DCHECK(PointerType::classof(src->getType()));
	llvm::PointerType* ptr_type;

	Value* false_value_ = ConstantInt::get(ctx,
			APInt(1, false, true));
	Value* size_ = ConstantInt::get(ctx, APInt(32, size));
	Value* zero = ConstantInt::get(ctx, APInt(32, 0));

	dst = Builder->CreateBitCast(dst, ptr_type);
	src = Builder->CreateBitCast(src, ptr_type);

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
	Builder->CreateCall(memcpy_fn, args);
}

ConstantInt* RawContext::createInt8(char val) {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(8, val));
}

ConstantInt* RawContext::createInt32(int val) {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(32, val));
}

ConstantInt* RawContext::createInt64(int val) {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(64, val));
}

ConstantInt* RawContext::createInt64(size_t val) {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(64, val));
}

ConstantInt* RawContext::createTrue() {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(1, 1));
}

ConstantInt* RawContext::createFalse() {
	LLVMContext& ctx = *llvmContext;
	return ConstantInt::get(ctx, APInt(1, 0));
}

Value* RawContext::CastPtrToLlvmPtr(PointerType* type, const void* ptr) {
	LLVMContext& ctx = *llvmContext;
	Constant* const_int = ConstantInt::get(Type::getInt64Ty(ctx),(uint64_t) ptr);
	Value* llvmPtr = ConstantExpr::getIntToPtr(const_int, type);
	return llvmPtr;
}

Value* RawContext::getArrayElem(AllocaInst* mem_ptr, Value* offset)	{
	LLVMContext& ctx = *llvmContext;

	Value* val_ptr = Builder->CreateLoad(mem_ptr, "mem_ptr");
	Value* shiftedPtr = Builder->CreateInBoundsGEP(val_ptr, offset);
	Value* val_shifted = Builder->CreateLoad(shiftedPtr,"val_shifted");
	return val_shifted;
}

Value* RawContext::getArrayElem(Value* val_ptr, Value* offset)	{
	LLVMContext& ctx = *llvmContext;

	Value* shiftedPtr = Builder->CreateInBoundsGEP(val_ptr, offset);
	Value* val_shifted = Builder->CreateLoad(shiftedPtr,"val_shifted");
	return val_shifted;
}

Value* RawContext::getArrayElemMem(Value* val_ptr, Value* offset)	{
	LLVMContext& ctx = *llvmContext;

	Value* shiftedPtr = Builder->CreateInBoundsGEP(val_ptr, offset);
	return shiftedPtr;
}

Value* RawContext::getStructElem(Value* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = Builder->CreateGEP(mem_struct, idxList);
	Value* val_struct_shifted =  Builder->CreateLoad(mem_struct_shifted);
	return val_struct_shifted;
}

Value* RawContext::getStructElemMem(Value* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = Builder->CreateGEP(mem_struct, idxList);
	return mem_struct_shifted;
}

Value* RawContext::getStructElem(AllocaInst* mem_struct, int elemNo)	{
	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(createInt32(0));
	idxList.push_back(createInt32(elemNo));
	//Shift in struct ptr
	Value* mem_struct_shifted = Builder->CreateGEP(mem_struct, idxList);
	Value* val_struct_shifted =  Builder->CreateLoad(mem_struct_shifted);
	return val_struct_shifted;
}

void RawContext::CreateForLoop(const string& cond, const string& body,
		const string& inc, const string& end, BasicBlock** cond_block,
		BasicBlock** body_block, BasicBlock** inc_block, BasicBlock** end_block,
		BasicBlock* insert_before) {
	Function* fn = TheFunction;
	LLVMContext& ctx = *llvmContext;
	*cond_block = BasicBlock::Create(ctx, string(cond), fn,	insert_before);
	*body_block = BasicBlock::Create(ctx, string(body), fn,	insert_before);
	*inc_block = BasicBlock::Create(ctx, string(inc), fn, insert_before);
	*end_block = BasicBlock::Create(ctx, string(end), fn, insert_before);
}

void RawContext::CreateIfElseBlocks(Function* fn, const string& if_label,
		const string& else_label, BasicBlock** if_block, BasicBlock** else_block,
		BasicBlock* insert_before) {
	LLVMContext& ctx = *llvmContext;
	*if_block = BasicBlock::Create(ctx, if_label, fn, insert_before);
	*else_block = BasicBlock::Create(ctx, else_label, fn, insert_before);
}

void RawContext::CreateIfBlock(Function* fn, const string& if_label,
		BasicBlock** if_block, BasicBlock* insert_before) {
	LLVMContext& ctx = *llvmContext;
	*if_block = BasicBlock::Create(ctx, if_label, fn, insert_before);
}

AllocaInst* RawContext::CreateEntryBlockAlloca(Function *TheFunction,
		const string &VarName, Type* varType) {
	IRBuilder<> TmpBuilder(&TheFunction->getEntryBlock(),
			TheFunction->getEntryBlock().begin());
	return TmpBuilder.CreateAlloca(varType, 0, VarName.c_str());
}

Value* RawContext::CreateGlobalString(char* str) {
	LLVMContext& ctx = *llvmContext;
	ArrayType* ArrayTy_0 = ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

	GlobalVariable* gvar_array__str = new GlobalVariable(*TheModule,
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
	Constant* shifted = ConstantExpr::getGetElementPtr(gvar_array__str,idxList);
	gvar_array__str->setInitializer(tmpHTname);


	LOG(INFO) << "[CreateGlobalString: ] " << str;
	Builder->CreateStore(shifted, AllocaName);
	Value* globalStr = Builder->CreateLoad(AllocaName);
	return globalStr;
}

Value* RawContext::CreateGlobalString(const char* str) {
	LLVMContext& ctx = *llvmContext;
	ArrayType* ArrayTy_0 = ArrayType::get(IntegerType::get(ctx, 8), strlen(str) + 1);

	GlobalVariable* gvar_array__str = new GlobalVariable(*TheModule,
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
	Constant* shifted = ConstantExpr::getGetElementPtr(gvar_array__str,idxList);
	gvar_array__str->setInitializer(tmpHTname);


	LOG(INFO) << "[CreateGlobalString: ] " << str;
	Builder->CreateStore(shifted, AllocaName);
	Value* globalStr = Builder->CreateLoad(AllocaName);
	return globalStr;
}

PointerType* RawContext::getPointerType(Type* type) {
	return PointerType::get(type, 0);
}

StructType* RawContext::CreateCustomStruct(vector<Type*> innerTypes) {
	LLVMContext& ctx = *llvmContext;
	llvm::StructType* valueType = llvm::StructType::get(ctx,innerTypes);
	return valueType;
}

StructType* RawContext::ReproduceCustomStruct(list<typeID> innerTypes) {
	LLVMContext& ctx = *llvmContext;
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
	LLVMContext& ctx = *llvmContext;
	llvm::Type* int64_type = Type::getInt64Ty(ctx);
	vector<Type*> json_pos_types;
	json_pos_types.push_back(int64_type);
	json_pos_types.push_back(int64_type);
	return CreateCustomStruct(json_pos_types);
}

PointerType* RawContext::CreateJSMNStructPtr()	{
	LLVMContext& ctx = *llvmContext;
	Type* jsmnStructType = CreateJSMNStruct();
	PointerType* ptr_jsmnStructType = PointerType::get(jsmnStructType,0);
	return ptr_jsmnStructType;
}

StructType* RawContext::CreateJSMNStruct() {
	LLVMContext& ctx = *llvmContext;
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
	LLVMContext& ctx = *llvmContext;
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


