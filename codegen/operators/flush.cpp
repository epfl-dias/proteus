/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2018
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

#include "operators/flush.hpp"
#include "util/raw-memory-manager.hpp"
#include "util/gpu/gpu-raw-context.hpp"

Flush::Flush(vector<expressions::Expression*> outputExprs,
		RawOperator* const child,
		RawContext* context,
		const char *outPath) :
		UnaryRawOperator(child), context(context), outPath(outPath) {
	list<expressions::AttributeConstruction> *attrs = new list<expressions::AttributeConstruction>();
	for (auto expr: outputExprs){
		assert(expr->isRegistered() && "All output expressions must be registered!");
		expressions::AttributeConstruction *newAttr =
										new expressions::AttributeConstruction(
											expr->getRegisteredAttrName(),
											expr
										);
		attrs->push_back(*newAttr);
	}

	outputExpr = new expressions::RecordConstruction(new RecordType(), *attrs);
}

void Flush::produce() {
	IntegerType * t = Type::getInt64Ty(context->getLLVMContext());
	result_cnt_id = context->appendStateVar(
		PointerType::getUnqual(t),
		[=](llvm::Value *){
			IRBuilder<> * Builder = context->getBuilder();

			Value * mem_acc = context->allocateStateVar(t);

			Builder->CreateStore(context->createInt64(0), mem_acc);


			OperatorState childState{*this, map<RecordAttribute, RawValueMemory>{}};
			ExpressionFlusherVisitor flusher{context, childState, outPath};
			flusher.beginList();

			return mem_acc;
		},

		[=](llvm::Value *, llvm::Value * s){
			OperatorState childState{*this, map<RecordAttribute, RawValueMemory>{}};
			ExpressionFlusherVisitor flusher{context, childState, outPath};
			flusher.endList();
			flusher.flushOutput();
			context->deallocateStateVar(s);
		}
	);

	getChild()->produce();
}

void Flush::consume(RawContext* const context, const OperatorState& childState) {
	generate(context, childState);
}

void Flush::generate(RawContext* const context,
		const OperatorState& childState) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	ExpressionFlusherVisitor flusher{context, childState, outPath};
	
	//results so far
	Value* mem_resultCtr = context->getStateVar(result_cnt_id);
	Value* resultCtr = Builder->CreateLoad(mem_resultCtr);
	
	//flushing out delimiter (IF NEEDED)
	flusher.flushDelim(resultCtr);
	
	outputExpr->accept(flusher);

	//increase result ctr
	Value* resultCtrInc = Builder->CreateAdd(resultCtr,context->createInt64(1));
	Builder->CreateStore(resultCtrInc, mem_resultCtr);
}

