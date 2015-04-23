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

#include "operators/reduce-opt-nopred.hpp"

namespace opt {
ReduceNoPred::ReduceNoPred(vector<Monoid> accs,
		vector<expressions::Expression*> outputExprs, RawOperator* const child,
		RawContext* context) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs), context(
				context) {

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *f = Builder->GetInsertBlock()->getParent();

	Type* int1Type = Type::getInt1Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* doubleType = Type::getDoubleTy(llvmContext);

	if (accs.size() != outputExprs.size()) {
		string error_msg = string("[REDUCE: ] Erroneous constructor args");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	vector<Monoid>::const_iterator itAcc;
	vector<expressions::Expression*>::const_iterator itExpr;
	itAcc = accs.begin();
	itExpr = outputExprs.begin();
	/* Prepare accumulator FOREACH outputExpr */
	for (; itAcc != accs.end(); itAcc++, itExpr++) {
		Monoid acc = *itAcc;
		expressions::Expression *outputExpr = *itExpr;
		AllocaInst *mem_accumulator = resetAccumulator(outputExpr, acc);
		mem_accumulators.push_back(mem_accumulator);
	}
}

void ReduceNoPred::produce() {
	getChild()->produce();
}

void ReduceNoPred::consume(RawContext* const context,
		const OperatorState& childState) {

	generate(context, childState);
	//flushResult();
}

void ReduceNoPred::flushResult() {
//	StringBuffer s;
//	Writer<StringBuffer> writer(s);
//	writer.StartObject();
//
//	switch (acc) {
//		case SUM:
//
//			break;
//		case MULTIPLY:
//
//			break;
//		case MAX:
//
//			break;
//		case OR:
//
//			break;
//		case AND:
//
//			break;
//		case UNION:
//		case BAGUNION:
//		case APPEND:
//		default: {
//			string error_msg = string("[Reduce: ] Unknown accumulator");
//			LOG(ERROR)<< error_msg;
//			throw runtime_error(error_msg);
//		}
//		}
//
//	writer.EndObject();
//	cout << s.GetString() << endl;
}

void ReduceNoPred::generate(RawContext* const context,
		const OperatorState& childState) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	vector<Monoid>::const_iterator itAcc;
	vector<expressions::Expression*>::const_iterator itExpr;
	vector<AllocaInst*>::const_iterator itMem;
	/* Time to Compute Aggs */
	itAcc = accs.begin();
	itExpr = outputExprs.begin();
	itMem = mem_accumulators.begin();

	for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
		Monoid acc = *itAcc;
		expressions::Expression *outputExpr = *itExpr;
		AllocaInst *mem_accumulating = *itMem;

		switch (acc) {
		case SUM:
			generateSum(outputExpr, context, childState, mem_accumulating);
			break;
		case MULTIPLY:
			generateMul(outputExpr, context, childState, mem_accumulating);
			break;
		case MAX:
			generateMax(outputExpr, context, childState, mem_accumulating);
			break;
		case OR:
			generateOr(outputExpr, context, childState, mem_accumulating);
			break;
		case AND:
			generateAnd(outputExpr, context, childState, mem_accumulating);
			break;
		case UNION:
			generateUnion(outputExpr, context, childState, mem_accumulating);
			break;
		case BAGUNION:
			//		generateBagUnion(context, childState);
			//		break;
		case APPEND:
			//		generateAppend(context, childState);
			//		break;
		default: {
			string error_msg = string(
					"[Reduce: ] Unknown / Still Unsupported accumulator");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}

	}

}

void ReduceNoPred::generateSum(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();


	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;

	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
		Value* val_new = Builder->CreateAdd(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);

#ifdef DEBUGREDUCE
		{
			Builder->SetInsertPoint(context->getEndingBlock());
			vector<Value*> ArgsV;
			Function* debugInt = context->getFunction("printi");
			Value* finalResult = Builder->CreateLoad(mem_accumulating);
			ArgsV.push_back(finalResult);
			Builder->CreateCall(debugInt, ArgsV);
			//Back to 'normal' flow
			Builder->SetInsertPoint(entryBlock);
		}
#endif
		break;
	}
	case FLOAT: {
		Value* val_new = Builder->CreateFAdd(val_accumulating,
				val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		vector<Value*> ArgsV;
		Function* debugFloat = context->getFunction("printFloat");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugFloat, ArgsV);
#endif
		//Back to 'normal' flow
		Builder->SetInsertPoint(entryBlock);
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Sum accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
}

void ReduceNoPred::generateMul(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
#ifdef DEBUGREDUCE
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		ArgsV.push_back(val_accumulating);
//		Builder->CreateCall(debugInt, ArgsV);
#endif
		Value* val_new = Builder->CreateMul(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
#ifdef DEBUGREDUCE
//		Builder->SetInsertPoint(endBlock);
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi");
//		Value* finalResult = Builder->CreateLoad(mem_accumulating);
//		ArgsV.push_back(finalResult);
//		Builder->CreateCall(debugInt, ArgsV);
//		//Back to 'normal' flow
//		Builder->SetInsertPoint(ifBlock);
#endif
		break;
	}
	case FLOAT: {
		Value* val_new = Builder->CreateFMul(val_accumulating,
				val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Sum accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
}

void ReduceNoPred::generateMax(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "reduceEnd", TheFunction);
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case INT: {
		/**
		 * if(curr > max) max = curr;
		 */
		BasicBlock* ifGtMaxBlock;
		context->CreateIfBlock(context->getGlobalFunction(), "reduceMaxCond",
				&ifGtMaxBlock, endBlock);
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);
		Value* maxCondition = Builder->CreateICmpSGT(val_output.value,
				val_accumulating);
		Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

		Builder->SetInsertPoint(ifGtMaxBlock);
		Builder->CreateStore(val_output.value, mem_accumulating);
		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		vector<Value*> ArgsV;
		Function* debugInt = context->getFunction("printi");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugInt, ArgsV);
#endif
		//Back to 'normal' flow
		Builder->SetInsertPoint(ifGtMaxBlock);

		//Branch Instruction to reach endBlock will be flushed after end of switch
		break;
	}
	case FLOAT: {
		/**
		 * if(curr > max) max = curr;
		 */
		BasicBlock* ifGtMaxBlock;
		context->CreateIfBlock(context->getGlobalFunction(), "reduceMaxCond",
				&ifGtMaxBlock, endBlock);
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);
		Value* maxCondition = Builder->CreateFCmpOGT(val_output.value,
				val_accumulating);
		Builder->CreateCondBr(maxCondition, ifGtMaxBlock, endBlock);

		Builder->SetInsertPoint(ifGtMaxBlock);
		Builder->CreateStore(val_output.value, mem_accumulating);
		Builder->CreateBr(endBlock);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		vector<Value*> ArgsV;
		Function* debugFloat = context->getFunction("printFloat");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugFloat, ArgsV);
#endif
		//Back to 'normal' flow
		Builder->SetInsertPoint(ifGtMaxBlock);
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Max accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(endBlock);
}

void ReduceNoPred::generateOr(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case BOOL: {
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

		RawValue val_output = outputExpr->accept(outputExprGenerator);
		Value* val_new = Builder->CreateOr(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);

		//Prepare final result output
		Builder->SetInsertPoint(context->getEndingBlock());
#ifdef DEBUGREDUCE
		std::vector<Value*> ArgsV;
		Function* debugBoolean = context->getFunction("printBoolean");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugBoolean, ArgsV);
#endif
		Builder->SetInsertPoint(entryBlock);
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Or accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}

}

void ReduceNoPred::generateAnd(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	/**
	 * IF(pred) Block
	 */
	ExpressionGeneratorVisitor outputExprGenerator = ExpressionGeneratorVisitor(
			context, state);
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
	val_output = outputExpr->accept(outputExprGenerator);
	Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

	switch (outputExpr->getExpressionType()->getTypeID()) {
	case BOOL: {
		Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

		RawValue val_output = outputExpr->accept(outputExprGenerator);
		Value* val_new = Builder->CreateAnd(val_accumulating, val_output.value);
		Builder->CreateStore(val_new, mem_accumulating);

		//Prepare final result output
#ifdef DEBUGREDUCE
		Builder->SetInsertPoint(context->getEndingBlock());
		vector<Value*> ArgsV;
		Function* debugBoolean = context->getFunction("printBoolean");
		Value* finalResult = Builder->CreateLoad(mem_accumulating);
		ArgsV.push_back(finalResult);
		Builder->CreateCall(debugBoolean, ArgsV);
		Builder->SetInsertPoint(entryBlock);
#endif
		break;
	}
	default: {
		string error_msg = string(
				"[Reduce: ] Or accumulator operates on numerics");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
}

//Flush out whatever you received
//FIXME Need 'output plugin' / 'serializer'
void ReduceNoPred::generateUnion(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Backing up insertion block
	BasicBlock *currBlock = Builder->GetInsertBlock();
	currBlock->dump();
	char outFilename[] = "out.json";
	ExpressionFlusherVisitor flusher = ExpressionFlusherVisitor(context, state,
			outFilename);

	//Preparing collection output (e.g., flushing out '{' in the case of JSON)
	BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();
	Builder->SetInsertPoint(loopEntryBlock->getTerminator());
	flusher.beginList();

	//Restoring
	Builder->SetInsertPoint(currBlock);

	//Generate condition
	Value *condition_ = ConstantInt::get(llvmContext,
			APInt(8, StringRef("1"), 10));
	Value *condition = Builder->CreateTrunc(condition_,
			IntegerType::get(llvmContext, 1));
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();

	BasicBlock *flushBlock = BasicBlock::Create(llvmContext, "flushBlock",
			TheFunction);
	BasicBlock *flushEnd = BasicBlock::Create(llvmContext, "flushEnd",
			TheFunction);

	Builder->CreateBr(flushBlock);
	Builder->SetInsertPoint(flushBlock);

	//results so far
	Value* mem_resultCtr = context->getMemResultCtr();
	Value* resultCtr = Builder->CreateLoad(mem_resultCtr);

	//flushing out delimiter (IF NEEDED)
	flusher.flushDelim(resultCtr);

	outputExpr->accept(flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,
			context->createInt64(1));
	Builder->CreateStore(resultCtrInc, mem_resultCtr);

	Builder->CreateBr(flushEnd);

	//Prepare final result output (e.g., flushing out '}' in the case of JSON)
	Builder->SetInsertPoint(context->getEndingBlock());
	flusher.endList();
	flusher.flushOutput();

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(flushEnd);
}

//Flush out whatever you received
//FIXME Need 'output plugin' / 'serializer'
void ReduceNoPred::generateBagUnion(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Backing up insertion block
	BasicBlock *currBlock = Builder->GetInsertBlock();
	//currBlock->dump();
	char outFilename[] = "out.json";
	ExpressionFlusherVisitor flusher = ExpressionFlusherVisitor(context, state,
			outFilename);

	//Preparing collection output (e.g., flushing out '{' in the case of JSON)
	BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();
	Builder->SetInsertPoint(loopEntryBlock->getTerminator());
	flusher.beginList();

	//Restoring
	Builder->SetInsertPoint(currBlock);

	//Generate condition
	Value *condition_ = ConstantInt::get(llvmContext,
			APInt(8, StringRef("1"), 10));
	Value *condition = Builder->CreateTrunc(condition_,
			IntegerType::get(llvmContext, 1));
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();

	BasicBlock *flushBlock = BasicBlock::Create(llvmContext, "flushBlock",
			TheFunction);
	BasicBlock *flushEnd = BasicBlock::Create(llvmContext, "flushEnd",
			TheFunction);

	Builder->CreateBr(flushBlock);
	Builder->SetInsertPoint(flushBlock);

	//results so far
	Value* mem_resultCtr = context->getMemResultCtr();
	Value* resultCtr = Builder->CreateLoad(mem_resultCtr);

	//flushing out delimiter (IF NEEDED)
	flusher.flushDelim(resultCtr);

	outputExpr->accept(flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,
			context->createInt64(1));
	Builder->CreateStore(resultCtrInc, mem_resultCtr);

	Builder->CreateBr(flushEnd);

	//Prepare final result output (e.g., flushing out '}' in the case of JSON)
	Builder->SetInsertPoint(context->getEndingBlock());
	flusher.endList();
	flusher.flushOutput();

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(flushEnd);
}

//Materializes collection (in HT?)
//Use the catalog for the materialization
void ReduceNoPred::generateAppend(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {

}

AllocaInst* ReduceNoPred::resetAccumulator(expressions::Expression* outputExpr,
		Monoid acc) const {
	AllocaInst* mem_accumulating = NULL;

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *f = Builder->GetInsertBlock()->getParent();

	Type* int1Type = Type::getInt1Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* doubleType = Type::getDoubleTy(llvmContext);

	//Deal with 'memory allocations' as per monoid type requested
	typeID outputType = outputExpr->getExpressionType()->getTypeID();
	switch (acc) {
	case SUM: {
		switch (outputType) {
		case INT: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			Value *val_zero = Builder->getInt32(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT: {
			Type* doubleType = Type::getDoubleTy(llvmContext);
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			Value *val_zero = ConstantFP::get(doubleType, 0.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default: {
			string error_msg = string(
					"[Reduce: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case MULTIPLY: {
		switch (outputType) {
		case INT: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			Value *val_zero = Builder->getInt32(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			Value *val_zero = ConstantFP::get(doubleType, 1.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default: {
			string error_msg = string(
					"[Reduce: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case MAX: {
		switch (outputType) {
		case INT: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int32Type);
			/**
			 * FIXME This is not the appropriate 'zero' value for integers.
			 * It is the one for naturals
			 */
			Value *val_zero = Builder->getInt32(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		case FLOAT: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), doubleType);
			/**
			 * FIXME This is not the appropriate 'zero' value for floats.
			 * It is the one for naturals
			 */
			Value *val_zero = ConstantFP::get(doubleType, 0.0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default: {
			string error_msg = string(
					"[Reduce: ] Sum/Multiply/Max operate on numerics");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		break;
	}
	case OR: {
		switch (outputType) {
		case BOOL: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int1Type);
			Value *val_zero = Builder->getInt1(0);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default: {
			string error_msg = string("[Reduce: ] Or/And operate on booleans");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

		}
		break;
	}
	case AND: {
		switch (outputType) {
		case BOOL: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int1Type);
			Value *val_zero = Builder->getInt1(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
		default: {
			string error_msg = string("[Reduce: ] Or/And operate on booleans");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

		}
		break;
	}
	case UNION:
	case BAGUNION: {
		string error_msg = string("[Reduce: ] Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case APPEND: {
		//XXX Reduce has some more stuff on this
		string error_msg = string("[Reduce: ] Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	default: {
		string error_msg = string("[Reduce: ] Unknown accumulator");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
	return mem_accumulating;
}

}

