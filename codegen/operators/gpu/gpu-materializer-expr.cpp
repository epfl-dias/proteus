/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
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

#include "operators/gpu/gpu-materializer-expr.hpp"




GpuExprMaterializer::GpuExprMaterializer(const std::vector<GpuMatExpr> &toMat,
		const std::vector<size_t> &packet_widths,
		RawOperator* const child, GpuRawContext* const context, string opLabel) :
		UnaryRawOperator(child), matExpr(toMat),
		context(context), 
		opLabel(opLabel),
		packet_widths(packet_widths) {

}

// GpuExprMaterializer::GpuExprMaterializer(expressions::Expression* toMat, int linehint,
// 		RawOperator* const child, RawContext* const context, char* opLabel, Value * out_ptr, Value * out_cnt) :
// 		UnaryRawOperator(child), toMat(toMat), context(context), opLabel(opLabel), out_ptr(out_ptr), out_cnt(out_cnt) {


	// Function *F = context->getGlobalFunction();
	// LLVMContext& llvmContext = context->getLLVMContext();
	// IRBuilder<> *Builder = context->getBuilder();

	// Type* int64_type = Type::getInt64Ty(llvmContext);
	// Type* int32_type = Type::getInt32Ty(llvmContext);
	// Type *int8_type = Type::getInt8Ty(llvmContext);
	// PointerType *int32_ptr_type = PointerType::get(int32_type, 0);
	// PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	// PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);

	// Value *zero = context->createInt64(0);

	// /* Arbitrary initial buffer sizes */
	// /* No realloc should be required with these sizes for synthetic large-scale numbers */
	// //	size_t sizeBuff = 10000000000;

	// /* 'linehint' * sizeof(expr)
	//  * -> sizeof(expr) not trivial to be found :) */
	// size_t sizeBuffer = linehint + 1;
	// switch (toMat->getExpressionType()->getTypeID()) {
	// case BOOL: {
	// 	break;
	// }
	// case STRING: {
	// 	/* Conservative - might require realloc */
	// 	sizeBuffer *= 8;
	// 	break;
	// }
	// case FLOAT: {
	// 	/* Conservative - might require realloc */
	// 	sizeBuffer *= sizeof(double);
	// 	break;
	// }
	// case INT: {
	// 	/* Conservative - might require realloc */
	// 	sizeBuffer *= sizeof(int);
	// 	break;
	// }
	// case INT64: {
	// 	/* Conservative - might require realloc */
	// 	sizeBuffer *= sizeof(size_t);
	// 	break;
	// }
	// case RECORD:
	// case LIST:
	// case BAG:
	// case SET:
	// case COMPOSITE: {
	// 	/* Conservative - might require realloc */
	// 	sizeBuffer *= sizeof(double);
	// 	break;
	// }
	// default: {
	// 	string error_msg = "[GpuExprMaterializer: ] Unknown type to mat.";
	// 	LOG(ERROR)<< error_msg;
	// 	throw runtime_error(error_msg);
	// }
	// }

	// Value *val_sizeBuffer = context->createInt64(sizeBuffer);
	// /* Request memory to store relation R 			*/
	// opBuffer.mem_buffer = context->CreateEntryBlockAlloca(F,
	// 		string("cacheBuffer"), char_ptr_type);
	// (opBuffer.mem_buffer)->setAlignment(8);
	// opBuffer.mem_tuplesNo = context->CreateEntryBlockAlloca(F,
	// 		string("tuplesR"), int64_type);
	// opBuffer.mem_size = context->CreateEntryBlockAlloca(F, string("sizeR"),
	// 		int64_type);
	// opBuffer.mem_offset = context->CreateEntryBlockAlloca(F,
	// 		string("offsetRelR"), int64_type);
	// rawBuffer = (char*) getMemoryChunk(sizeBuffer);
	// ptr_rawBuffer = (char**) malloc(sizeof(char*));
	// *ptr_rawBuffer = rawBuffer;
	// Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type, rawBuffer);
	// Builder->CreateStore(val_relationR, opBuffer.mem_buffer);
	// Builder->CreateStore(zero, opBuffer.mem_tuplesNo);
	// Builder->CreateStore(zero, opBuffer.mem_offset);
	// Builder->CreateStore(val_sizeBuffer, opBuffer.mem_size);

	// /* Note: In principle, it's not necessary to store payload as struct.
	//  * This way, however, it is uniform across all caching cases. */
	// toMatType = NULL;
// }


GpuExprMaterializer::~GpuExprMaterializer()	{
	LOG(INFO)<<"Collapsing Gpu Materializer operator";
//	Can't do garbage collection here, need to do it from codegen
}

// void GpuExprMaterializer::freeArenas() const	{
// 	/* Prepare codegen utils */
// 	LLVMContext& llvmContext = context->getLLVMContext();
// 	RawCatalog& catalog = RawCatalog::getInstance();
// 	Function *F = context->getGlobalFunction();
// 	IRBuilder<> *Builder = context->getBuilder();
// 	Function *debugInt = context->getFunction("printi");
// 	Function *debugInt64 = context->getFunction("printi64");

// 	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
// 	Type *int8_type = Type::getInt8Ty(llvmContext);
// 	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
// 	Type *int64_type = Type::getInt64Ty(llvmContext);
// 	Type *int32_type = Type::getInt32Ty(llvmContext);

// 	/* Actual Work */
// 	Function* freeLLVM = context->getFunction("releaseMemoryChunk");

// 	Value *val_arena = Builder->CreateLoad(opBuffer.mem_buffer);
// 	vector<Value*> ArgsFree;
// 	AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type,0,"voidArenaPtr");
// 	Builder->CreateStore(val_arena,mem_arena_void);
// 	Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
// 	ArgsFree.push_back(val_arena_void);
// 	Builder->CreateCall(freeLLVM, ArgsFree);
// }

void GpuExprMaterializer::produce() {
	std::sort(matExpr.begin(), matExpr.end(), [](const GpuMatExpr& a, const GpuMatExpr& b){
		return a.packet < b.packet || a.bitoffset < b.bitoffset;
	});

	size_t i = 0;

	for (size_t p = 0 ; p < packet_widths.size() ; ++p){
		// Type * t     = PointerType::get(IntegerType::getIntNTy(context->getLLVMContext(), packet_widths[p]), /* address space */ 1);

		size_t bindex  = 0;
		size_t packind = 0;

		std::vector<Type *> body;
		while (i < matExpr.size() && matExpr[i].packet == p){
			if (matExpr[i].bitoffset != bindex){
				//insert space
				assert(matExpr[i].bitoffset > bindex);
				body.push_back(Type::getIntNTy(context->getLLVMContext(), (matExpr[i].bitoffset - bindex)));
				++packind;
			}

			const ExpressionType * out_type = matExpr[i].expr->getExpressionType();

			if (!out_type->isPrimitive()){
				string error_msg("[GpuExprMaterializer: ] Currently only supports materialization of primitive types");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}

			Type * llvm_type = ((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext());

			body.push_back(llvm_type);
			bindex = matExpr[i].bitoffset + llvm_type->getPrimitiveSizeInBits();
			matExpr[i].packind = packind++;
			++i;
		}
		assert(packet_widths[p] >= bindex);

		if (packet_widths[p] > bindex) {
			body.push_back(Type::getIntNTy(context->getLLVMContext(), (packet_widths[p] - bindex)));
		}

		Type * t     = StructType::create(body, opLabel + "_struct_" + std::to_string(p), true);
		Type * t_ptr = PointerType::get(t, /* address space */ 1);

		out_param_ids.push_back(context->appendParameter(t_ptr, true, false));
	}
	assert(i == matExpr.size());

	// Type * t     = PointerType::get(((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext()), /* address space */ 1);
	// out_param_id = context->appendParameter(t    , true, false);

	Type * int32_type = Type::getInt32Ty(context->getLLVMContext());

	Type * t_cnt = PointerType::get(int32_type, /* address space */ 1);
	cnt_param_id = context->appendParameter(t_cnt, true, false);

	// Function *F = context->getGlobalFunction();
	// LLVMContext& llvmContext = context->getLLVMContext();
	// IRBuilder<> *Builder = context->getBuilder();

	// Type* int64_type = Type::getInt64Ty(llvmContext);
	// Type* int32_type = Type::getInt32Ty(llvmContext);
	// Type *int8_type = Type::getInt8Ty(llvmContext);
	// PointerType *int32_ptr_type = PointerType::get(int32_type, 0);
	// PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	// PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);

	// Value *zero = context->createInt64(0);

	// /* Arbitrary initial buffer sizes */
	// /* No realloc will be required with these sizes for synthetic large-scale numbers */
	// //	size_t sizeBuff = 10000000000;

	// /* 'linehint' if need be*/
	// size_t sizeBuffer = 1000;

	// Value *val_sizeBuffer = context->createInt64(sizeBuffer);
	// /* Request memory to store relation R 			*/
	// opBuffer.mem_buffer = context->CreateEntryBlockAlloca(F,
	// 		string("cacheBuffer"), char_ptr_type);
	// (opBuffer.mem_buffer)->setAlignment(8);
	// opBuffer.mem_tuplesNo = context->CreateEntryBlockAlloca(F,
	// 		string("tuplesR"), int64_type);
	// opBuffer.mem_size = context->CreateEntryBlockAlloca(F, string("sizeR"),
	// 		int64_type);
	// opBuffer.mem_offset = context->CreateEntryBlockAlloca(F,
	// 		string("offsetRelR"), int64_type);
	// rawBuffer = (char*) getMemoryChunk(sizeBuffer);
	// ptr_rawBuffer = (char**) malloc(sizeof(char*));
	// *ptr_rawBuffer = rawBuffer;
	// Value *val_relationR = context->CastPtrToLlvmPtr(char_ptr_type, rawBuffer);
	// Builder->CreateStore(val_relationR, opBuffer.mem_buffer);
	// Builder->CreateStore(zero, opBuffer.mem_tuplesNo);
	// Builder->CreateStore(zero, opBuffer.mem_offset);
	// Builder->CreateStore(val_sizeBuffer, opBuffer.mem_size);

	// /* Note: In principle, it's not necessary to store payload as struct.
	//  * This way, however, it is uniform across all caching cases. */
	// toMatType = NULL;


	getChild()->produce();

	/* Free Arenas */
	/*this->freeArenas();*/
}

// void GpuExprMaterializer::updateRelationPointers() const {
// 	Function *F = context->getGlobalFunction();
// 	LLVMContext& llvmContext = context->getLLVMContext();
// 	IRBuilder<> *Builder = context->getBuilder();
// 	PointerType *char_ptr_type = Type::getInt8PtrTy(llvmContext);
// 	PointerType *char_ptr_ptr_type = PointerType::get(char_ptr_type, 0);

// 	Value *val_ptrRawBuffer = context->CastPtrToLlvmPtr(char_ptr_ptr_type,
// 			ptr_rawBuffer);
// 	Value *val_rawBuffer = Builder->CreateLoad(this->opBuffer.mem_buffer);
// 	Builder->CreateStore(val_rawBuffer, val_ptrRawBuffer);
// }

void GpuExprMaterializer::consume(RawContext* const context, const OperatorState& childState) {

	/* Prepare codegen utils */
	LLVMContext& llvmContext = context->getLLVMContext();
	RawCatalog& catalog = RawCatalog::getInstance();
	Function *F = context->getGlobalFunction();
	IRBuilder<> *Builder = context->getBuilder();
	// Function *debugInt = context->getFunction("printi");
	// Function *debugInt64 = context->getFunction("printi64");


	Argument * out_cnt = ((const GpuRawContext *) context)->getArgument(cnt_param_id);
	out_cnt->setName(opLabel + "_cnt_ptr");


	Value * old_cnt = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
												out_cnt,
												ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
												llvm::AtomicOrdering::Monotonic);



	std::vector<Value *> out_ptrs;
	std::vector<Value *> out_vals;

	for (size_t i = 0 ; i < out_param_ids.size() ; ++i) {
		Argument * out_ptr = ((const GpuRawContext *) context)->getArgument(out_param_ids[i]);
		if (out_param_ids.size() != 1){
			out_ptr->setName(opLabel + "_out" + std::to_string(i) + "_ptr");
		} else {
			out_ptr->setName(opLabel + "_out_ptr");
		}
		// out_ptrs.push_back(out_ptr);

		out_ptrs.push_back(Builder->CreateInBoundsGEP(out_ptr, old_cnt));
		out_vals.push_back(UndefValue::get(out_ptr->getType()->getPointerElementType()));
	}

	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
	Type *int8_type = Type::getInt8Ty(llvmContext);
	PointerType *void_ptr_type = PointerType::get(int8_type, 0);
	Type *int64_type = Type::getInt64Ty(llvmContext);
	Type *int32_type = Type::getInt32Ty(llvmContext);

	// const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();

	// std::vector<Value *> vals_toMat;

	for (const GpuMatExpr &mexpr: matExpr){
		ExpressionGeneratorVisitor exprGenerator(context, childState);
		RawValue valWrapper = mexpr.expr->accept(exprGenerator);
		
		out_vals[mexpr.packet] = Builder->CreateInsertValue(out_vals[mexpr.packet], valWrapper.value, mexpr.packind);
		// std::cout << out_vals[mexpr.packind]->getType() << std::endl;
		// break;
	}

	/* Creating the 'payload' type */
	// vector<Type*> types;
	// types.push_back(val_toMat->getType());
	// toMatType = context->CreateCustomStruct(types);

	// Value* val_exprSize = ConstantExpr::getSizeOf(val_toMat->getType());

// #ifdef DEBUG
// 	{
// 	vector<Value*> ArgsV;
// 	Function* debugInt = context->getFunction("printi64");

// 	ArgsV.push_back(val_exprSize);
// 	Builder->CreateCall(debugInt, ArgsV);
// 	}
// #endif

	for (size_t i = 0 ; i < out_ptrs.size() ; ++i){
		// Builder->CreateStore(out_vals[i], out_ptrs[i]);
		Builder->CreateAlignedStore(out_vals[i], out_ptrs[i], packet_widths[i]/8);
	}

	// Value *val_arena = Builder->CreateLoad(opBuffer.mem_buffer);
	// Value *offsetInArena = Builder->CreateLoad(opBuffer.mem_offset);
	// Value *offsetPlusPayload = Builder->CreateAdd(offsetInArena, val_exprSize);
	// Value *arenaSize = Builder->CreateLoad(opBuffer.mem_size);
	// Value* val_tuplesNo = Builder->CreateLoad(opBuffer.mem_tuplesNo);

	/* if(offsetInArena + payloadSize >= arenaSize) */
	// BasicBlock* entryBlock = Builder->GetInsertBlock();
	// BasicBlock *endBlockArenaFull = BasicBlock::Create(llvmContext,
	// 		"IfArenaFullEnd", F);
	// BasicBlock *ifArenaFull;
	// context->CreateIfBlock(F, "IfArenaFullCond", &ifArenaFull,
	// 		endBlockArenaFull);
	// Value *offsetCond = Builder->CreateICmpSGE(offsetPlusPayload, arenaSize);

	// Builder->CreateCondBr(offsetCond, ifArenaFull, endBlockArenaFull);

	// /* true => realloc() */
	// Builder->SetInsertPoint(ifArenaFull);

	// vector<Value*> ArgsRealloc;
	// Function* reallocLLVM = context->getFunction("increaseMemoryChunk");
	// AllocaInst* mem_arena_void = Builder->CreateAlloca(void_ptr_type, 0,
	// 		"voidArenaPtr");
	// Builder->CreateStore(val_arena, mem_arena_void);
	// Value *val_arena_void = Builder->CreateLoad(mem_arena_void);
	// ArgsRealloc.push_back(val_arena_void);
	// ArgsRealloc.push_back(arenaSize);
	// Value* val_newArenaVoidPtr = Builder->CreateCall(reallocLLVM, ArgsRealloc);

// 	// Builder->CreateStore(val_newArenaVoidPtr, opBuffer.mem_buffer);
// 	Value* val_size = Builder->CreateLoad(opBuffer.mem_size);
// 	val_size = Builder->CreateMul(val_size, context->createInt64(2));
// 	// Builder->CreateStore(val_size, opBuffer.mem_size);
// 	// Builder->CreateBr(endBlockArenaFull);

// 	// /* 'Normal' flow again */
// 	// Builder->SetInsertPoint(endBlockArenaFull);

// 	/* Repeat load - realloc() might have occurred */
// 	val_arena = Builder->CreateLoad(opBuffer.mem_buffer);
// 	val_size = Builder->CreateLoad(opBuffer.mem_size);

// 	/* XXX STORING PAYLOAD */
// 	/* 1. arena += (offset) */
// 	Value *ptr_arenaShifted = Builder->CreateInBoundsGEP(val_arena,offsetInArena);

// 	/* 2. Casting */
// 	PointerType *ptr_payloadType = PointerType::get(toMatType, 0);
// 	Value *cast_arenaShifted = Builder->CreateBitCast(ptr_arenaShifted,
// 			ptr_payloadType);

// 	/* 3. Storing payload */
// //	vector<Value*> idxList = vector<Value*>();
// //	idxList.push_back(context->createInt32(0));
// //	idxList.push_back(context->createInt32(0));
// //	//Shift in struct ptr
// //	Value* structPtr = Builder->CreateGEP(cast_arenaShifted, idxList);
// 	Value* structPtr = context->getStructElemMem(cast_arenaShifted,0);
// 	StoreInst *store_activeTuple = Builder->CreateStore(val_toMat, structPtr);

// 	/* 4. Increment counts */
// 	Builder->CreateStore(offsetPlusPayload, opBuffer.mem_offset);
// 	val_tuplesNo = Builder->CreateAdd(val_tuplesNo, context->createInt64(1));
// 	Builder->CreateStore(val_tuplesNo, opBuffer.mem_tuplesNo);
// 	Builder->CreateStore(val_tuplesNo,opBuffer.mem_tuplesNo);

	/*
	 * Control logic:
	 * XXX Register in cache
	 */
// 	{
// #ifdef DEBUGCACHING
// 		cout << "[Materializer:] Register in cache" << endl;
// #endif
// 		CachingService& cache = CachingService::getInstance();
// 		bool fullRelation = !(this->getChild())->isFiltering();

// 		CacheInfo info;
// 		info.objectTypes.push_back(toMat->getExpressionType()->getTypeID());
// 		info.structFieldNo = 0;
// 		info.payloadPtr = ptr_rawBuffer;
// 		//Have LLVM exec. fill itemCount up!
// 		info.itemCount = new size_t[1];
// 		*(info.itemCount) = 0;		
// 		PointerType *int64PtrType = Type::getInt64PtrTy(llvmContext);
// 		Value *mem_itemCount = context->CastPtrToLlvmPtr(int64PtrType,(void*)info.itemCount);
// 		Builder->CreateStore(val_tuplesNo, mem_itemCount);
		
// 		cache.registerCache(toMat, info, fullRelation);
// 	}

	/* 5. Triggering parent */
	// OperatorState* newState = new OperatorState(*this,
	// 		childState.getBindings());
	// getParent()->consume(context, *newState);



}



