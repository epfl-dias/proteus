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

#include "operators/unnest.hpp"

void Unnest::produce()	const { getChild()->produce(); }

void Unnest::consume(RawContext* const context, const OperatorState& childState) const {
	generate(context, childState);
}

void Unnest::generate(RawContext* const context, const OperatorState& childState) const
{
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Generate path. Value returned must be a collection
	ExpressionGeneratorVisitor pathExprGenerator = ExpressionGeneratorVisitor(context, childState);
	expressions::RecordProjection* pathProj = path.get();
	Value* nestedValueAll = pathProj->accept(pathExprGenerator);

	/**
	 * foreach val in nestedValue:
	 * 		if(condition)
	 * 			...
	 */

	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	context->CreateForLoop("unnestChildLoopCond","unnestChildLoopBody","unnestChildLoopInc","unnestChildLoopEnd",
							&loopCond,&loopBody,&loopInc,&loopEnd);
	/**
	 * ENTRY:
	 * init the vars used by the plugin
	 */
	Plugin* pg = path.getRelevantPlugin();
	AllocaInst* mem_currentObjId = pg->initCollectionUnnest(nestedValueAll);
	Builder->CreateBr(loopCond);

	Builder->SetInsertPoint(loopCond);
	Value* endCond = pg->collectionHasNext(nestedValueAll,mem_currentObjId);
	Builder->CreateCondBr(endCond, loopBody, loopEnd);

	Builder->SetInsertPoint(loopBody);
	AllocaInst* nestedValueItem =  pg->collectionGetNext(mem_currentObjId);

	#ifdef DEBUG
//		vector<Value*> ArgsV;
//		Function* debugInt = context->getFunction("printi64");
//		Value* val = Builder->CreateLoad(nestedValueItem);
//		ArgsV.push_back(val);
//		Builder->CreateCall(debugInt, ArgsV);
//		ArgsV.clear();
//		Value *tmp = context->createInt64(1000);
//		ArgsV.push_back(tmp);
//		Builder->CreateCall(debugInt, ArgsV);
	#endif

	//Preparing call to parent
	map<RecordAttribute, AllocaInst*>* unnestBindings = new map<RecordAttribute, AllocaInst*>(childState.getBindings());
	RawCatalog& catalog = RawCatalog::getInstance();
	LOG(INFO) << "[Unnest: ] Registering plugin of "<< path.toString();
	catalog.registerPlugin(path.toString(),pg);

	//attrNo does not make a difference
	RecordAttribute unnestedAttr = RecordAttribute(2,
			path.toString(),
			activeLoop,
			pathProj->getExpressionType());

	(*unnestBindings)[unnestedAttr] = nestedValueItem;
	OperatorState* newState = new OperatorState(*this,*unnestBindings);

	/**
	 * Predicate Evaluation:
	 */
	BasicBlock *ifBlock, *elseBlock;
	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifUnnestCond", "elseUnnestCond",
									&ifBlock, &elseBlock,loopInc);

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, *newState);
	Value* condition = pred->accept(predExprGenerator);
	Builder->CreateCondBr(condition,ifBlock,elseBlock);

	/*
	 * IF BLOCK
	 * CALL NEXT OPERATOR, ADDING nestedValueItem binding
	 */
	Builder->SetInsertPoint(ifBlock);
	//Triggering parent
	getParent()->consume(context, *newState);
	Builder->CreateBr(loopInc);

	/**
	 * ELSE BLOCK
	 * Just branch to the INC part of unnest loop
	 */
	Builder->SetInsertPoint(elseBlock);
	Builder->CreateBr(loopInc);

	Builder->SetInsertPoint(loopInc);
	Builder->CreateBr(loopCond);

	Builder->SetInsertPoint(loopEnd);
}
