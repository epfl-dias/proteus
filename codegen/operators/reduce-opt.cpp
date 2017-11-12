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

namespace opt {
Reduce::Reduce(vector<Monoid> accs,
		vector<expressions::Expression*> outputExprs,
		expressions::Expression* pred, RawOperator* const child,
		RawContext* context, bool flushResults, const char *outPath) :
		UnaryRawOperator(child), accs(accs), outputExprs(outputExprs), pred(
				pred), context(context), flushResults(flushResults), outPath(outPath) {
	if (accs.size() != outputExprs.size()) {
		string error_msg = string("[REDUCE: ] Erroneous constructor args");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
}

void Reduce::produce() {
	getChild()->produce();
}

void Reduce::consume(RawContext* const context, const OperatorState& childState) {
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

//Flush out whatever you received
//FIXME Need 'output plugin' / 'serializer'
void Reduce::generateBagUnion(expressions::Expression* outputExpr,
				RawContext* const context, const OperatorState& state) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	ExpressionFlusherVisitor flusher{context, state, outPath};

	//Backing up insertion block
	BasicBlock *currBlock = Builder->GetInsertBlock();

	//Preparing collection output (e.g., flushing out '{' in the case of JSON)
	BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();

	Builder->SetInsertPoint(loopEntryBlock);
	flusher.beginList();

	//Restoring
	Builder->SetInsertPoint(currBlock);

	//results so far
	Value* mem_resultCtr = context->getMemResultCtr();
	Value* resultCtr = Builder->CreateLoad(mem_resultCtr);

	//flushing out delimiter (IF NEEDED)
	flusher.flushDelim(resultCtr);

	outputExpr->accept(flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,context->createInt64(1));
	Builder->CreateStore(resultCtrInc, mem_resultCtr);

	//Backing up insertion block
	currBlock = Builder->GetInsertBlock();

	//Prepare final result output (e.g., flushing out '}' in the case of JSON)
	Builder->SetInsertPoint(context->getEndingBlock());
	flusher.endList();
	flusher.flushOutput();

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
	Function *f = Builder->GetInsertBlock()->getParent();

	BasicBlock* entryBlock = Builder->GetInsertBlock();
	Builder->SetInsertPoint(context->getCurrentEntryBlock());

	//Deal with 'memory allocations' as per monoid type requested
	switch (acc) {
		case SUM:
		case MULTIPLY:
		case MAX:
		case OR:
		case AND: {
			Constant * val_id = getIdentityElementIfSimple(
												acc, 
												outputExpr->getExpressionType(),
												context
							);
			mem_accumulating  = context->CreateEntryBlockAlloca(
												f, 
												string("dest_acc"), 
												val_id->getType()
							);
			Builder->CreateStore(val_id, mem_accumulating);
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
