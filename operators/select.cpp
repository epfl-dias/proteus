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

#include "operators/select.hpp"

void Select::produce()	const { getChild()->produce(); }

void Select::consume(RawContext* const context, const OperatorState& childState) const {
	generate(context, childState);
}

void Select::generate(RawContext* const context, const OperatorState& childState) const
{
	IRBuilder<>* TheBuilder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();

	//Generate condition
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
	RawValue condition = this->expr->accept(exprGenerator);

	//Get entry point
	Function *TheFunction = TheBuilder->GetInsertBlock()->getParent();

	// Create blocks for the then cases.  Insert the 'then' block at the end of the function.
	// Note: No 'else' in this case
	BasicBlock *ThenBB = BasicBlock::Create(llvmContext, "selectionThen", TheFunction);
	BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "selectionIfCont");

	TheBuilder->CreateCondBr(condition.value, ThenBB, MergeBB);
	TheBuilder->SetInsertPoint(ThenBB);

	//Triggering parent
	OperatorState* newState = new OperatorState(*this,childState.getBindings());
	getParent()->consume(context, *newState);

	TheBuilder->CreateBr(MergeBB);

	TheFunction->getBasicBlockList().push_back(MergeBB);
	TheBuilder->SetInsertPoint(MergeBB);
}
