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
		RawValue val_v = expr_v->accept(vGenerator);
		Value* val_v_isNull = val_v.isNull;

		Value* nullCond = Builder->CreateICmpNE(val_v_isNull, context->createTrue());
		Builder->CreateCondBr(nullCond, IfNotNull, ElseIsNull);
	}

	/**
	 * if( v != NULL )
	 */
//	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
//	AllocaInst *mem_accumulating = NULL;
//	RawValueMemory nestedValueItem;
//	map<RecordAttribute, RawValueMemory>* unnestBindings =
//					new map<RecordAttribute, RawValueMemory>(childState.getBindings());
//	//Generate path. Value returned must be a collection
//	expressions::RecordProjection* pathProj = path.get();
//	ExpressionGeneratorVisitor pathExprGenerator = ExpressionGeneratorVisitor(context, childState);
//	RawValue nestedValueAll = pathProj->accept(pathExprGenerator);
//	Plugin* pg = path.getRelevantPlugin();
//
//	//attrNo does not make a difference
//	RecordAttribute unnestedAttr = RecordAttribute(2, path.toString(),
//			activeLoop, pathProj->getExpressionType());
//	{
//		Builder->SetInsertPoint(IfNotNull);
//
//		/**
//		 * and{ not p(v,w') | v != NULL, w' <- path(v)}
//		 *
//		 * aka:
//		 * foreach val in nestedValue:
//		 * 		acc = acc AND not(if(condition))
//		 * 			...
//		 */
//		mem_accumulating = context->CreateEntryBlockAlloca(
//				TheFunction, "mem_acc", boolType);
//		Value *val_zero = Builder->getInt1(1);
//		Builder->CreateStore(val_zero, mem_accumulating);
//
//		//w' <- path(v)
//		context->CreateForLoop("outUnnestLoopCond",
//				"outUnnestLoopBody", "outUnnestLoopInc",
//				"outUnnestLoopEnd", &loopCond, &loopBody, &loopInc,
//				&loopEnd);
//
//		//ENTRY: init the vars used by the plugin
//		RawValueMemory mem_currentObjId = pg->initCollectionUnnest(nestedValueAll);
//		Builder->CreateBr(loopCond);
//
//		Builder->SetInsertPoint(loopCond);
//		RawValue endCond = pg->collectionHasNext(nestedValueAll,
//				mem_currentObjId);
//		Builder->CreateCondBr(endCond.value, loopBody, loopEnd);
//
//		Builder->SetInsertPoint(loopBody);
//		nestedValueItem = pg->collectionGetNext(mem_currentObjId);
//		type_w = (nestedValueItem.mem)->getAllocatedType();
//
//		RawCatalog& catalog = RawCatalog::getInstance();
//		LOG(INFO)<< "[Outer Unnest: ] Registering plugin of "<< path.toString();
//		catalog.registerPlugin(path.toString(), pg);
//
//		(*unnestBindings)[unnestedAttr] = nestedValueItem;
//		OperatorState* newState = new OperatorState(*this, *unnestBindings);
//
//		//Predicate Evaluation: Generate condition
//		ExpressionGeneratorVisitor predExprGenerator =
//				ExpressionGeneratorVisitor(context, *newState);
//		RawValue condition = pred->accept(predExprGenerator);
//		Value* invert_condition = Builder->CreateNot(condition.value);
//		Value* val_acc_current = Builder->CreateLoad(mem_accumulating);
//		Value* val_acc_new = Builder->CreateAnd(invert_condition,
//				val_acc_current);
//		Builder->CreateStore(val_acc_new, mem_accumulating);
//
//		getParent()->consume(context, *newState);
//
//		(*unnestBindings).erase(unnestedAttr);
//		Builder->CreateBr(loopInc);
//	}
//
//	//else -> v == NULL
//	//Just branch to the INC part of unnest loop
//	{
//		Builder->SetInsertPoint(loopInc);
//		Builder->CreateBr(loopCond);
//		Builder->SetInsertPoint(loopEnd);
//		Builder->CreateBr(ElseIsNull);
//	}
	BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
	AllocaInst *mem_accumulating = NULL;
	RawValueMemory nestedValueItem;
	Plugin* pg = path.getRelevantPlugin();
	{
		Builder->SetInsertPoint(IfNotNull);

		//Generate path. Value returned must be a collection
		ExpressionGeneratorVisitor pathExprGenerator = ExpressionGeneratorVisitor(context, childState);
		expressions::RecordProjection* pathProj = path.get();
		RawValue nestedValueAll = pathProj->accept(pathExprGenerator);

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
		RawValueMemory mem_currentObjId = pg->initCollectionUnnest(nestedValueAll);
		Builder->CreateBr(loopCond);

		Builder->SetInsertPoint(loopCond);
		RawValue endCond = pg->collectionHasNext(nestedValueAll,
				mem_currentObjId);
		Builder->CreateCondBr(endCond.value, loopBody, loopEnd);

		Builder->SetInsertPoint(loopBody);
		nestedValueItem = pg->collectionGetNext(mem_currentObjId);
		type_w = (nestedValueItem.mem)->getAllocatedType();

		map<RecordAttribute, RawValueMemory>* unnestBindings =
				new map<RecordAttribute, RawValueMemory>(childState.getBindings());
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
		RawValue condition = pred->accept(predExprGenerator);
		Value* invert_condition = Builder->CreateNot(condition.value);
		Value* val_acc_current = Builder->CreateLoad(mem_accumulating);
		Value* val_acc_new = Builder->CreateAnd(invert_condition,
				val_acc_current);
		Builder->CreateStore(val_acc_new, mem_accumulating);

		//getParent()->consume(context, *newState);
		(*unnestBindings).erase(unnestedAttr);
		Builder->CreateBr(loopInc);
	}

	//else -> v == NULL
	//Just branch to the INC part of unnest loop
	{
		Builder->SetInsertPoint(loopInc);
		Builder->CreateBr(loopCond);
		Builder->SetInsertPoint(loopEnd);
		//Builder->CreateBr(ElseIsNull);
	}

	/*
	 * Time to generate "if-then-else"
	 *
	 * w <-	IF and{ not p(v,w') | v != NULL, w' <- path(v)}
	 * 		THEN [NULL]
	 * 		ELSE { w' | w' <- path(v), p(v,w') }
	 *
	 * FIXME Code as is will loop twice over path(v) -> Optimize
	 */
	Builder->SetInsertPoint(loopEnd);
	BasicBlock *ifOuterNull, *elseOuterNotNull;

	context->CreateIfElseBlocks(context->getGlobalFunction(), "ifOuterUnnestNull", "elseOuterNotNull",
										&ifOuterNull, &elseOuterNotNull);

	Value* ifCond = Builder->CreateLoad(mem_accumulating);
	Builder->CreateCondBr(ifCond,ifOuterNull,elseOuterNotNull);

	//attrNo does not make a difference
	RecordAttribute unnestedAttr = RecordAttribute(2, path.toString(),
			activeLoop, path.get()->getExpressionType());
	map<RecordAttribute, RawValueMemory>* unnestBindings =
							new map<RecordAttribute, RawValueMemory>(childState.getBindings());
	//'if' --> NULL
	{
		Builder->SetInsertPoint(ifOuterNull);
		Value* undefValue = Constant::getNullValue(type_w);
		RawValueMemory valWrapper;
		AllocaInst* mem_undefValue = context->CreateEntryBlockAlloca(
						TheFunction, "val_undef", type_w);
		Builder->CreateStore(undefValue,mem_undefValue);
		valWrapper.mem = mem_undefValue;
		valWrapper.isNull = context->createTrue();

		(*unnestBindings)[unnestedAttr] = valWrapper;
		OperatorState *newStateNull = new OperatorState(*this,*unnestBindings);

		//Triggering parent
		getParent()->consume(context, *newStateNull);
		Builder->CreateBr(ElseIsNull);
	}

	//'else - outerNotNull'
	{
		Builder->SetInsertPoint(elseOuterNotNull);
		//XXX Same as unnest!
		//foreach val in nestedValue: if(condition) ...
		BasicBlock *loopCond2, *loopBody2, *loopInc2, *loopEnd2;
		context->CreateForLoop("unnestChildLoopCond2","unnestChildLoopBody2","unnestChildLoopInc2","unnestChildLoopEnd2",
							&loopCond2,&loopBody2,&loopInc2,&loopEnd2);
		/**
		 * ENTRY:
		 * init the vars used by the plugin
		 */
		ExpressionGeneratorVisitor pathExprGenerator = ExpressionGeneratorVisitor(context, childState);
		RawValue nestedValueAll = path.get()->accept(pathExprGenerator);
		RawValueMemory mem_currentObjId2 = pg->initCollectionUnnest(nestedValueAll);
		Builder->CreateBr(loopCond2);

		Builder->SetInsertPoint(loopCond2);
		RawValue endCond = pg->collectionHasNext(nestedValueAll,mem_currentObjId2);
		Builder->CreateCondBr(endCond.value, loopBody2, loopEnd2);

		//body
		{
			Builder->SetInsertPoint(loopBody2);
			RawValueMemory nestedValueItem2 =  pg->collectionGetNext(mem_currentObjId2);
			//Preparing call to parent
			map<RecordAttribute, RawValueMemory>* unnestBindings2 = new map<
					RecordAttribute, RawValueMemory>(childState.getBindings());

			(*unnestBindings2)[unnestedAttr] = nestedValueItem2;


			OperatorState* newState2 = new OperatorState(*this,*unnestBindings2);


			/**
			 * Predicate Evaluation:
			 */
			BasicBlock *ifBlock, *elseBlock;
			context->CreateIfElseBlocks(context->getGlobalFunction(), "ifOuterUnnestCond", "elseOuterUnnestCond",
									&ifBlock, &elseBlock,loopInc2);

			//Generate condition
			ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, *newState2);
			RawValue condition2 = pred->accept(predExprGenerator);
			Builder->CreateCondBr(condition2.value,ifBlock,elseBlock);

			/*
			 * IF BLOCK
			 * CALL NEXT OPERATOR, ADDING nestedValueItem binding
			 */
			{
				Builder->SetInsertPoint(ifBlock);
				//Triggering parent
				getParent()->consume(context, *newState2);
				Builder->CreateBr(loopInc2);
			}

			/**
			 * ELSE BLOCK
			 * Just branch to the INC part of unnest loop
			 */
			{
				Builder->SetInsertPoint(elseBlock);
				Builder->CreateBr(loopInc2);
			}

			{
				Builder->SetInsertPoint(loopInc2);
				Builder->CreateBr(loopCond2);
			}
		}

		{
			Builder->SetInsertPoint(loopEnd2);
			Builder->CreateBr(ElseIsNull);
		}
	}

	/**
	 * else -> What to do?
	 * 		-> Nothing; just keep the insert point there
	 */
	Builder->SetInsertPoint(ElseIsNull);
}
