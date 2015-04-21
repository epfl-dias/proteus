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

void Unnest::produce()	{ getChild()->produce(); }

void Unnest::consume(RawContext* const context, const OperatorState& childState) {
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
	RawValue nestedValueAll = pathProj->accept(pathExprGenerator);

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
	RawValueMemory mem_currentObjId = pg->initCollectionUnnest(nestedValueAll);
	Builder->CreateBr(loopCond);

	Builder->SetInsertPoint(loopCond);
	RawValue endCond = pg->collectionHasNext(nestedValueAll,mem_currentObjId);
	Value *val_hasNext = endCond.value;
#ifdef DEBUGUNNEST
	{
	//Printing the active token that will be forwarded
	vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(val_hasNext);
	Function* debugBoolean = context->getFunction("printBoolean");
	Builder->CreateCall(debugBoolean, ArgsV);
	}
#endif
	Builder->CreateCondBr(val_hasNext, loopBody, loopEnd);

	Builder->SetInsertPoint(loopBody);
#ifdef DEBUGUNNEST
//	{
//	//Printing the active token that will be forwarded
//	vector<Value*> ArgsV;
//	ArgsV.clear();
//	ArgsV.push_back(context->createInt64(111));
//	Function* debugInt = context->getFunction("printi64");
//	Builder->CreateCall(debugInt, ArgsV);
//	}
#endif
	RawValueMemory nestedValueItem =  pg->collectionGetNext(mem_currentObjId);
#ifdef DEBUGUNNEST
	{	Function* debugInt = context->getFunction("printi64");

		Value* val_currentTokenId = Builder->CreateLoad(nestedValueItem.mem);
		Value *val_offset = context->getStructElem(nestedValueItem.mem, 0);
		Value *val_rowId = context->getStructElem(nestedValueItem.mem, 1);
		Value *val_currentTokenNo = context->getStructElem(nestedValueItem.mem, 2);
	//Printing the active token that will be forwarded
	vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(val_offset);
	Builder->CreateCall(debugInt, ArgsV);

	ArgsV.clear();
	ArgsV.push_back(val_rowId);
	Builder->CreateCall(debugInt, ArgsV);

	ArgsV.clear();
	ArgsV.push_back(val_currentTokenNo);
	Builder->CreateCall(debugInt, ArgsV);
	}
#endif

	//Preparing call to parent
	map<RecordAttribute, RawValueMemory>* unnestBindings = new map<RecordAttribute, RawValueMemory>(childState.getBindings());
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
	RawValue condition = pred->accept(predExprGenerator);
	Builder->CreateCondBr(condition.value,ifBlock,elseBlock);

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
#ifdef DEBUGUNNEST
	{
	//Printing the active token that will be forwarded
	vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(context->createInt64(222));
	Function* debugInt = context->getFunction("printi64");
	Builder->CreateCall(debugInt, ArgsV);
	}
#endif
}
