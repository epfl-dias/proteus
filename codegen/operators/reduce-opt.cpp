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

#include "operators/reduce-opt.hpp"

#pragma push_macro("DEBUG") //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side
#undef DEBUG //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side
#pragma push_macro("DEBUGREDUCE") //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side
#undef DEBUGREDUCE //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side

namespace opt {
Reduce::Reduce(vector<Monoid> accs,
		vector<expressions::Expression*> outputExprs,
		expressions::Expression* pred, RawOperator* const child,
		RawContext* context, bool flushResults, const char *outPath) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs), pred(
				pred), context(context), flushResults(flushResults), outPath(outPath) {
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
}

void Reduce::produce() {
	getChild()->produce();
}

void Reduce::consume(RawContext* const context,
		const OperatorState& childState) {
	if (mem_accumulators.empty()){
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
	generate(context, childState);
	//flushResult();
}

void Reduce::flushResult() {
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

void Reduce::generate(RawContext* const context,
		const OperatorState& childState) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	ExpressionFlusherVisitor flusher{context, childState, outPath};

	int aggsNo = accs.size();
	if (flushResults && aggsNo > 1) {
		/* Start result flushing: Opening brace */
		//Backing up insertion block
		BasicBlock * currBlock = Builder->GetInsertBlock();

		//Preparing collection output (e.g., flushing out '{' in the case of JSON)
		BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();

		Builder->SetInsertPoint(loopEntryBlock);
		flusher.beginList();

		//Restoring
		Builder->SetInsertPoint(currBlock);
	}

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator{context, childState};
	RawValue condition = pred->accept(predExprGenerator);
	/**
	 * Predicate Evaluation:
	 */
	BasicBlock* entryBlock = Builder->GetInsertBlock();
	BasicBlock *endBlock = BasicBlock::Create(llvmContext, "reduceCondEnd",
			TheFunction);
	BasicBlock *ifBlock;
	context->CreateIfBlock(context->getGlobalFunction(), "reduceIfCond",
			&ifBlock, endBlock);

	/**
	 * IF(pred) Block
	 */
	RawValue val_output;
	Builder->SetInsertPoint(entryBlock);
//	{
//		Function* debugInt = context->getFunction("printi");
//		vector<Value*> ArgsV;
//		ArgsV.push_back(context->createInt32(777));
//		Builder->CreateCall(debugInt, ArgsV, "printi");
//	}
	Builder->CreateCondBr(condition.value, ifBlock, endBlock);

	Builder->SetInsertPoint(ifBlock);

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
		case MULTIPLY:
		case MAX:
		case OR:
		case AND:{
			ExpressionGeneratorVisitor outputExprGenerator{context, childState};

			// Load accumulator -> acc_value
			RawValue acc_value;
			acc_value.value  = Builder->CreateLoad(mem_accumulating);
			acc_value.isNull = context->createFalse();

			// new_value = acc_value op outputExpr
			expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc_value);
			expressions::Expression * upd = toExpression(acc, val, outputExpr);
			assert(upd && "Monoid is not convertible to expression!");
			RawValue new_val = upd->accept(outputExprGenerator);

			// store new_val to accumulator
			Builder->CreateStore(new_val.value, mem_accumulating);
			break;
		}
		case BAGUNION:
			generateBagUnion(outputExpr, context, childState);
			break;
		case APPEND:
			//		generateAppend(context, childState);
			//		break;
		case UNION:
		default: {
			string error_msg = string(
					"[Reduce: ] Unknown / Still Unsupported accumulator");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		}
		if (flushResults){
			BasicBlock * currEndingBlock = Builder->GetInsertBlock();
			/* Flushing Output */
			bool flushDelim = (aggsNo > 1) && (itAcc != accs.end() - 1);
			switch (acc) {
			case SUM:
			case MULTIPLY:
			case MAX:
			case OR:
			case AND:{
				Builder->SetInsertPoint(context->getEndingBlock());
				Value* val_acc = Builder->CreateLoad(mem_accumulating);
				flusher.flushValue(val_acc, outputExpr->getExpressionType()->getTypeID());
				if (flushDelim) flusher.flushDelim();

				break;
			}
			case BAGUNION:{
				//nothing to flush for bagunion
				break;
			}
			case APPEND:
			case UNION:
			default: {
				string error_msg = string(
						"[Reduce: ] Unknown / Still Unsupported accumulator");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
			}
			/* Back to normal flow */
			Builder->SetInsertPoint(currEndingBlock);
		}
		/* Need to group aggregate values together if > 1 accumulators */
//			Builder->SetInsertPoint(context->getEndingBlock());
//			if (aggsNo != 1) {
//				if (itAcc == accs.end() - 1) {
//					flusher.endList();
//				} else {
//					flusher.flushDelim();
//				}
//			}
//			Builder->SetInsertPoint(currEndingBlock);

	}

	Builder->CreateBr(endBlock);

	/**
	 * END Block
	 */
	if (flushResults) {
		if (aggsNo > 1) {
			//Prepare final result output (e.g., flushing out ']' in the case of JSON)
			Builder->SetInsertPoint(context->getEndingBlock());
			flusher.endList();
		}
		Builder->SetInsertPoint(context->getEndingBlock());
		flusher.flushOutput();
	}

	Builder->SetInsertPoint(endBlock);
}

void Reduce::generateSum(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	ExpressionGeneratorVisitor outputExprGenerator{context, state};

	RawValue acc;
	acc.value  = Builder->CreateLoad(mem_accumulating);
	acc.isNull = context->createFalse();

	expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc);
	expressions::Expression * upd = new expressions::AddExpression    (val, outputExpr);
	RawValue new_val = upd->accept(outputExprGenerator);

	Builder->CreateStore(new_val.value, mem_accumulating);
}

void Reduce::generateMul(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	ExpressionGeneratorVisitor outputExprGenerator{context, state};

	RawValue acc;
	acc.value  = Builder->CreateLoad(mem_accumulating);
	acc.isNull = context->createFalse();

	expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc);
	expressions::Expression * upd = new expressions::MultExpression    (val, outputExpr);
	RawValue new_val = upd->accept(outputExprGenerator);

	Builder->CreateStore(new_val.value, mem_accumulating);
}

void Reduce::generateMax(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	ExpressionGeneratorVisitor outputExprGenerator{context, state};

	RawValue acc;
	acc.value  = Builder->CreateLoad(mem_accumulating);
	acc.isNull = context->createFalse();

	expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc);
	expressions::Expression * upd = new expressions::MaxExpression     (val, outputExpr);
	RawValue new_val = upd->accept(outputExprGenerator);

	Builder->CreateStore(new_val.value, mem_accumulating);
}

void Reduce::generateOr(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	ExpressionGeneratorVisitor outputExprGenerator{context, state};

	RawValue acc;
	acc.value  = Builder->CreateLoad(mem_accumulating);
	acc.isNull = context->createFalse();

	expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc);
	expressions::Expression * upd = new expressions::OrExpression      (val, outputExpr);
	RawValue new_val = upd->accept(outputExprGenerator);

	Builder->CreateStore(new_val.value, mem_accumulating);
}

void Reduce::generateAnd(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {
	IRBuilder<>* Builder = context->getBuilder();
	ExpressionGeneratorVisitor outputExprGenerator{context, state};

	RawValue acc;
	acc.value  = Builder->CreateLoad(mem_accumulating);
	acc.isNull = context->createFalse();

	expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc);
	expressions::Expression * upd = new expressions::AndExpression     (val, outputExpr);
	RawValue new_val = upd->accept(outputExprGenerator);

	Builder->CreateStore(new_val.value, mem_accumulating);
}

//Flush out whatever you received
//FIXME Need 'output plugin' / 'serializer'
void Reduce::generateBagUnion(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state, ExpressionFlusherVisitor * bflusher) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	ExpressionFlusherVisitor * flusher;
	if (!bflusher) flusher = new ExpressionFlusherVisitor(context, state, outPath);
	else           flusher = bflusher;

	//Backing up insertion block
	BasicBlock *currBlock = Builder->GetInsertBlock();

	//Preparing collection output (e.g., flushing out '{' in the case of JSON)
	BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();

	Builder->SetInsertPoint(loopEntryBlock);
	flusher->beginList();

	//Restoring
	Builder->SetInsertPoint(currBlock);

	//results so far
	Value* mem_resultCtr = context->getMemResultCtr();
	Value* resultCtr = Builder->CreateLoad(mem_resultCtr);

	//flushing out delimiter (IF NEEDED)
	flusher->flushDelim(resultCtr);

	outputExpr->accept(*flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,
			context->createInt64(1));
	Builder->CreateStore(resultCtrInc, mem_resultCtr);

	//Backing up insertion block
	currBlock = Builder->GetInsertBlock();

	//Prepare final result output (e.g., flushing out '}' in the case of JSON)
	Builder->SetInsertPoint(context->getEndingBlock());
	flusher->endList();
	flusher->flushOutput();

	if (!bflusher) delete flusher;

	/**
	 * END Block
	 */
	Builder->SetInsertPoint(currBlock);
}

//Materializes collection (in HT?)
//Use the catalog for the materialization
void Reduce::generateAppend(expressions::Expression* outputExpr,
		RawContext* const context, const OperatorState& state,
		AllocaInst *mem_accumulating) const {

}

AllocaInst* Reduce::resetAccumulator(expressions::Expression* outputExpr,
		Monoid acc) const {
	AllocaInst* mem_accumulating = NULL;

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *f = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	Builder->SetInsertPoint(context->getCurrentEntryBlock());

	Type* int1Type = Type::getInt1Ty(llvmContext);
	Type* int32Type = Type::getInt32Ty(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	Type* doubleType = Type::getDoubleTy(llvmContext);

	//Deal with 'memory allocations' as per monoid type requested
	typeID outputType = outputExpr->getExpressionType()->getTypeID();
	switch (acc) {
	case SUM: {
		switch (outputType) {
		case INT64: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int64Type);
			Value *val_zero = Builder->getInt64(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
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
		case INT64: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int64Type);
			Value *val_zero = Builder->getInt64(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
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
		case INT64: {
			mem_accumulating = context->CreateEntryBlockAlloca(f,
					string("dest_acc"), int64Type);
			Value *val_zero = Builder->getInt64(1);
			Builder->CreateStore(val_zero, mem_accumulating);
			break;
		}
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
	case UNION: {
		string error_msg = string("[Reduce: ] Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case BAGUNION:
	case APPEND: {
		/*XXX Bags and Lists can be processed in streaming fashion -> No accumulator needed */
		mem_accumulating = NULL;
		break;
	}
	default: {
		string error_msg = string("[Reduce: ] Unknown accumulator");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}


	Builder->SetInsertPoint(entryBlock);
	return mem_accumulating;
}

}

#pragma pop_macro("DEBUG") //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side
#pragma pop_macro("DEBUGREDUCE") //FIXME: REMOVE!!! used to disable prints, as they are currently undefined for the gpu side
