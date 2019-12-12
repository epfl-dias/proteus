/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#include "operators/null-filter.hpp"

using namespace llvm;

void NullFilter::produce_(ParallelContext *context) {
  getChild()->produce(context);
}

void NullFilter::consume(Context *const context,
                         const OperatorState &childState) {
  generate(context, childState);
}

void NullFilter::generate(Context *const context,
                          const OperatorState &childState) const {
  IRBuilder<> *TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();

  // Generate condition
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, childState);
  ProteusValue condition = this->expr->accept(exprGenerator);
  Value *isNotNull = TheBuilder->CreateNot(condition.isNull);

  // Get entry point
  Function *TheFunction = TheBuilder->GetInsertBlock()->getParent();

  // Create blocks for the then cases.  Insert the 'then' block at the end of
  // the function. Note: No 'else' in this case
  BasicBlock *ThenBB =
      BasicBlock::Create(llvmContext, "notNullFilterThen", TheFunction);
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "notNullFilterCont");

  TheBuilder->CreateCondBr(isNotNull, ThenBB, MergeBB);
  TheBuilder->SetInsertPoint(ThenBB);

  // Triggering parent
  OperatorState *newState = new OperatorState(*this, childState.getBindings());
  getParent()->consume(context, *newState);

  TheBuilder->CreateBr(MergeBB);

  TheFunction->getBasicBlockList().push_back(MergeBB);
  TheBuilder->SetInsertPoint(MergeBB);
}
