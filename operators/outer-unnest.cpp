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

#include "operators/outer-unnest.hpp"

void OuterUnnest::produce() const {
	getChild()->produce();
}

void OuterUnnest::consume(RawContext* const context,
		const OperatorState& childState) const {
	generate(context, childState);
}

/**
 * { (v',w) | v <- 	X,
 * 			  w <-	if and{ not p(v,w') | v != NULL, w' <- path(v)}
 * 			  		then [NULL]
 * 			  		else { w' | w' <- path(v), p(v,w') }
 */
void OuterUnnest::generate(RawContext* const context,
		const OperatorState& childState) const {
	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	Type* boolType = Type::getInt1Ty(llvmContext);
	Type* type_w = NULL;

	/**
	 *  Preparing if(v != NULL)
	 *  Comparing with NULL is a hassle
	 */
	BasicBlock *IfNotNull;
	BasicBlock *ElseIsNull = BasicBlock::Create(llvmContext, "else_null",
			TheFunction);
	{
		context->CreateIfBlock(TheFunction, "if_notNull", &IfNotNull,
				ElseIsNull);

		//Retrieving v using the projection -> Keeping v out of v.w
		expressions::RecordProjection* pathProj = path.get();
		expressions::Expression* expr_v = pathProj->getExpr();

		//Evaluating v

		ExpressionGeneratorVisitor vGenerator = ExpressionGeneratorVisitor(
				context, childState, pathProj->getOriginalRelationName());
		Value *val_v = expr_v->accept(vGenerator);

		PointerType* ptr_type_v = PointerType::get(val_v->getType(), 0);
		ConstantPointerNull* null_ptr_v = ConstantPointerNull::get(ptr_type_v);

		//XXX Used to get memory of a value v
		//TODO Check if generated code simplifies it!
		//BUG BUG BUG HERE??
		AllocaInst* mem_tmp_v = context->CreateEntryBlockAlloca(TheFunction,
				"mem_tmp_v", val_v->getType());
		Builder->CreateStore(val_v, mem_tmp_v);

		Value* nullCond = Builder->CreateICmpNE(mem_tmp_v, null_ptr_v);
		Builder->CreateCondBr(nullCond, IfNotNull, ElseIsNull);
	}

	/**
	 * if( v != NULL )
	 */
	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	AllocaInst *mem_accumulating = NULL;
	AllocaInst *nestedValueItem = NULL;
	{
		Builder->SetInsertPoint(IfNotNull);

		//Generate path. Value returned must be a collection
		ExpressionGeneratorVisitor pathExprGenerator = ExpressionGeneratorVisitor(context, childState);
		expressions::RecordProjection* pathProj = path.get();
		Value* nestedValueAll = pathProj->accept(pathExprGenerator);

		/**
		 * and{ not p(v,w') | v != NULL, w' <- path(v)}
		 *
		 * aka:
		 * foreach val in nestedValue:
		 * 		acc = acc AND not(if(condition))
		 * 			...
		 */
		mem_accumulating = context->CreateEntryBlockAlloca(
				TheFunction, "mem_acc", boolType);
		Value *val_zero = Builder->getInt1(1);
		Builder->CreateStore(val_zero, mem_accumulating);

		//w' <- path(v)
		context->CreateForLoop("outUnnestLoopCond",
				"outUnnestLoopBody", "outUnnestLoopInc",
				"outUnnestLoopEnd", &loopCond, &loopBody, &loopInc,
				&loopEnd);

		//ENTRY: init the vars used by the plugin
		Plugin* pg = path.getRelevantPlugin();
		AllocaInst* mem_currentObjId = pg->initCollectionUnnest(nestedValueAll);
		Builder->CreateBr(loopCond);

		Builder->SetInsertPoint(loopCond);
		Value* endCond = pg->collectionHasNext(nestedValueAll,
				mem_currentObjId);
		Builder->CreateCondBr(endCond, loopBody, loopEnd);

		Builder->SetInsertPoint(loopBody);
		nestedValueItem = pg->collectionGetNext(mem_currentObjId);
		type_w = nestedValueItem->getAllocatedType();

		map<RecordAttribute, AllocaInst*>* unnestBindings =
				new map<RecordAttribute, AllocaInst*>(childState.getBindings());
		RawCatalog& catalog = RawCatalog::getInstance();
		LOG(INFO)<< "[Outer Unnest: ] Registering plugin of "<< path.toString();
		catalog.registerPlugin(path.toString(), pg);

		//attrNo does not make a difference
		RecordAttribute unnestedAttr = RecordAttribute(2,
				path.toString(),
				activeLoop,
				pathProj->getExpressionType());

		(*unnestBindings)[unnestedAttr] = nestedValueItem;
		OperatorState* newState = new OperatorState(*this, *unnestBindings);

		//Predicate Evaluation: Generate condition
		ExpressionGeneratorVisitor predExprGenerator =
				ExpressionGeneratorVisitor(context, *newState);
		Value* condition = pred->accept(predExprGenerator);
		Value* invert_condition = Builder->CreateNot(condition);
		Value* val_acc_current = Builder->CreateLoad(mem_accumulating);
		Value* val_acc_new = Builder->CreateAnd(invert_condition,
				val_acc_current);
		Builder->CreateStore(val_acc_new, mem_accumulating);
		//DELETEME
#ifdef DEBUG
		//TEMP
		{
			std::vector<Value*> ArgsV;
			Function* debugBool = context->getFunction("printBoolean");
			Value* acc_ = Builder->CreateLoad(mem_accumulating);
			ArgsV.push_back(acc_);
			Builder->CreateCall(debugBool, ArgsV);
		}
		getParent()->consume(context, *newState);
#endif
		//END OF DELETEME
		(*unnestBindings).erase(unnestedAttr);
		Builder->CreateBr(loopInc);
	}

	//else -> v == NULL
	//Just branch to the INC part of unnest loop
	{
		Builder->SetInsertPoint(loopInc);
		Builder->CreateBr(loopCond);
		Builder->SetInsertPoint(loopEnd);
		Builder->CreateBr(ElseIsNull);
	}

//	//TEMP
//	{
//		Builder->SetInsertPoint(loopEnd);
//#ifdef DEBUG
//		std::vector<Value*> ArgsV;
//		Function* debugBool = context->getFunction("printBoolean");
//		Value* acc_ = Builder->CreateLoad(mem_accumulating);
//		ArgsV.push_back(acc_);
//		Builder->CreateCall(debugBool, ArgsV);
//#endif
//
//		map<RecordAttribute, AllocaInst*>* unnestBindings =
//						new map<RecordAttribute, AllocaInst*>(childState.getBindings());
//		RecordAttribute unnestedAttr = RecordAttribute(2, path.toString(),
//						activeLoop, path.get()->getExpressionType());
//		(*unnestBindings)[unnestedAttr] = nestedValueItem;
//		OperatorState* newState = new OperatorState(*this, *unnestBindings);
//		getParent()->consume(context, *newState);
//	}
	//send upwards to debug!
	//END OF TEMP

//
//	/*
//	 * Time to generate "if-then-else"
//	 *
//	 * w <-	IF and{ not p(v,w') | v != NULL, w' <- path(v)}
//	 * 		THEN [NULL]
//	 * 		ELSE { w' | w' <- path(v), p(v,w') }
//	 *
//	 * FIXME Code as is will loop twice over path(v) -> Optimize
//	 */
//	Builder->SetInsertPoint(loopEnd);
//
//	BasicBlock *ifOuterNull, *elseOuterNotNull;
//
//	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifOuterUnnestNull", "elseOuterNotNull",
//										&ifOuterNull, &elseOuterNotNull);
//
//	Value* ifCond = Builder->CreateLoad(mem_accumulating);
//	Builder->CreateCondBr(ifCond,ifOuterNull,elseOuterNotNull);
//
//	//'if' --> NULL
//	Builder->SetInsertPoint(ifOuterNull);
//	PointerType* ptr_type_w = PointerType::get(type_w, 0);
//	ConstantPointerNull* null_ptr_w = ConstantPointerNull::get(ptr_type_w);
//	(*unnestBindings)[unnestedAttr] = null_ptr_w;
//	OperatorState* newStateNull = new OperatorState(*this,*unnestBindings);
//	(*unnestBindings).erase(unnestedAttr);
//	//Triggering parent
//	getParent()->consume(context, *newState);
//	Builder->CreateBr(ElseIsNull);
//
//	//'else - outerNotNull'
//	Builder->SetInsertPoint(elseOuterNotNull);
//	//XXX Same as unnest!
//	//foreach val in nestedValue: if(condition) ...
//	BasicBlock *loopCond2, *loopBody2, *loopInc2, *loopEnd2;
//	context->CreateForLoop("unnestChildLoopCond2","unnestChildLoopBody2","unnestChildLoopInc2","unnestChildLoopEnd2",
//							&loopCond2,&loopBody2,&loopInc2,&loopEnd2);
//	/**
//	 * ENTRY:
//	 * init the vars used by the plugin
//	 */
//	AllocaInst* mem_currentObjId2 = pg->initCollectionUnnest(nestedValueAll);
//	Builder->CreateBr(loopCond2);
//
//	Builder->SetInsertPoint(loopCond2);
//	Value* endCond = pg->collectionHasNext(nestedValueAll,mem_currentObjId);
//	Builder->CreateCondBr(endCond, loopBody2, loopEnd2);
//
//	Builder->SetInsertPoint(loopBody2);
//	AllocaInst* nestedValueItem2 =  pg->collectionGetNext(mem_currentObjId2);
//	//Preparing call to parent
//	map<RecordAttribute, AllocaInst*>* unnestBindings2 = new map<RecordAttribute, AllocaInst*>(childState.getBindings());
//
//	//attrNo does not make a difference
//	RecordAttribute unnestedAttr = RecordAttribute(2,
//			path.toString(),
//			activeLoop,
//			pathProj->getExpressionType());
//
//	(*unnestBindings)[unnestedAttr] = nestedValueItem2;
//	OperatorState* newState2 = new OperatorState(*this,*unnestBindings);
//
//	/**
//	 * Predicate Evaluation:
//	 */
//	BasicBlock *ifBlock, *elseBlock;
//	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifOuterUnnestCond", "elseOuterUnnestCond",
//									&ifBlock, &elseBlock,loopInc2);
//
//	//Generate condition
//	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, *newState2);
//	Value* condition2 = pred->accept(predExprGenerator);
//	Builder->CreateCondBr(condition2,ifBlock,elseBlock);
//
//	/*
//	 * IF BLOCK
//	 * CALL NEXT OPERATOR, ADDING nestedValueItem binding
//	 */
//	Builder->SetInsertPoint(ifBlock);
//	//Triggering parent
//	getParent()->consume(context, *newState);
//	Builder->CreateBr(loopInc2);
//
//	/**
//	 * ELSE BLOCK
//	 * Just branch to the INC part of unnest loop
//	 */
//	Builder->SetInsertPoint(elseBlock);
//	Builder->CreateBr(loopInc2);
//
//	Builder->SetInsertPoint(loopInc2);
//	Builder->CreateBr(loopCond2);
//
//	Builder->SetInsertPoint(loopEnd2);
//
//	Builder->CreateBr(ElseIsNull);

//	/**
//	 * else -> What to do?
//	 * 		-> Nothing; just keep the insert point there
//	 */
	Builder->SetInsertPoint(ElseIsNull);

}
