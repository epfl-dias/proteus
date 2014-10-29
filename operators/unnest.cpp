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

	//Generate condition
	ExpressionGeneratorVisitor predExprGenerator = ExpressionGeneratorVisitor(context, childState);
	Value* condition = pred->accept(predExprGenerator);

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
	/**
	 * EVALUATE PREDICATE
	 * IF QUALIFYING:
	 * CALL NEXT OPERATOR, ADDING nestedValueItem binding
	 *
	 */
	map<RecordAttribute, AllocaInst*>* unnestBindings = new map<RecordAttribute, AllocaInst*>(childState.getBindings());
	RawCatalog& catalog = RawCatalog::getInstance();
	cout << "Getting plugin of " << path.toString() << endl;
	catalog.registerPlugin(path.toString(),pg);


	RecordAttribute unnestedAttr = RecordAttribute(2,
			path.toString(),
			path.toString(),
			activeLoop,//path.getNestedName(),
			pathProj->getExpressionType());
//	RecordAttribute unnestedAttr = RecordAttribute(-1,
//				pathProj->getOriginalRelationName(),
//				path.toString(),
//				activeLoop,
//				pathProj->getExpressionType());

	//Triggering parent
//	(*unnestBindings)[unnestedAttr] = nestedValueItem;
	(*unnestBindings).insert(std::pair<RecordAttribute,AllocaInst*>(unnestedAttr,nestedValueItem));
	OperatorState* newState = new OperatorState(*this,*unnestBindings);
	getParent()->consume(context, *newState);

	Builder->CreateBr(loopInc);
	Builder->SetInsertPoint(loopInc);
	Builder->CreateBr(loopCond);

	Builder->SetInsertPoint(loopEnd);


//	TheBuilder->CreateBr(MergeBB);
//
//	TheFunction->getBasicBlockList().push_back(MergeBB);
//	TheBuilder->SetInsertPoint(MergeBB);
}

string Path::toString() {
	stringstream ss;
	ss << desugarizedPath->getRelationName();
	expressions::Expression* currExpr = desugarizedPath;
	while (currExpr->getTypeId() == expressions::RECORD_PROJECTION) {
		expressions::RecordProjection* const currProj =
				(expressions::RecordProjection*) currExpr;
		ss << ".";
		ss << currProj->getProjectionName();
		currExpr = currProj->getExpr();
	}
	LOG(INFO) << "[Unnest: ] path.toString = " << ss.str();
	return ss.str();
}


