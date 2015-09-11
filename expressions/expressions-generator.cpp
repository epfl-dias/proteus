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

#include "expressions/expressions-generator.hpp"

RawValue ExpressionGeneratorVisitor::visit(expressions::IntConstant *e) {
	RawValue valWrapper;
	valWrapper.value = ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::FloatConstant *e) {
	RawValue valWrapper;
	valWrapper.value = ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::BoolConstant *e) {
	RawValue valWrapper;
	valWrapper.value = ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::StringConstant *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();

	char* str = new char[e->getVal().length() + 1];
	strcpy(str,e->getVal().c_str());
	Value* globalStr = context->CreateGlobalString(str);

	StructType* strObjType = context->CreateStringStruct();
	Function *F = context->getGlobalFunction();
	AllocaInst* mem_strObj = context->CreateEntryBlockAlloca(F, e->getVal(),
			strObjType);

	Value *val_0 = context->createInt32(0);
	Value *val_1 = context->createInt32(1);

	vector<Value*> idxList = vector<Value*>();
	idxList.push_back(val_0);
	idxList.push_back(val_0);
	Value* structPtr = TheBuilder->CreateGEP(mem_strObj,idxList);
	TheBuilder->CreateStore(globalStr,structPtr);

	idxList.clear();
	idxList.push_back(val_0);
	idxList.push_back(val_1);
	structPtr = TheBuilder->CreateGEP(mem_strObj,idxList);
	TheBuilder->CreateStore(context->createInt32(e->getVal().length()),structPtr);

	Value* val_strObj = TheBuilder->CreateLoad(mem_strObj);
	RawValue valWrapper;
	valWrapper.value = val_strObj;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::InputArgument *e) {
	IRBuilder<>* const Builder = context->getBuilder();
	RawCatalog& catalog 			= RawCatalog::getInstance();
	AllocaInst* argMem = NULL;
	Value* isNull;

	/* No caching logic here, because InputArgument generally
	 * does not result in materializing / converting / etc. actions! */
	{
		const map<RecordAttribute, RawValueMemory>& activeVars = currState.getBindings();
		map<RecordAttribute, RawValueMemory>::const_iterator it;

		//A previous visitor has indicated which relation is relevant
		if(activeRelation != "")	{
			Plugin* pg = catalog.getPlugin(activeRelation);
			RecordAttribute relevantAttr = RecordAttribute(activeRelation,activeLoop,pg->getOIDType());
			it = activeVars.find(relevantAttr);
			if (it == activeVars.end()) {
				string error_msg = string("[Expression Generator: ] Could not find tuple information for ") + activeRelation;
				//cout << activeVars.size() << endl;
//				map<RecordAttribute, RawValueMemory>::const_iterator it = activeVars.begin();
//				for(; it != activeVars.end(); it++)	{
//					cout << it->first.getRelationName() << "-" << it->first.getAttrName() << endl;
//				}
				LOG(ERROR) << error_msg;
			 	throw runtime_error(error_msg);
			}	else	{
				argMem = (it->second).mem;
				isNull = (it->second).isNull;
			}
		}
		else
		{
			list<RecordAttribute> projections = e->getProjections();
			list<RecordAttribute>::iterator it = projections.begin();

			for (; it != projections.end(); it++) {
				/* OID info */
				if (it->getAttrName() == activeLoop) {
					map<RecordAttribute, RawValueMemory>::const_iterator itBindings;
					for (itBindings = activeVars.begin();
							itBindings != activeVars.end(); itBindings++) {

						RecordAttribute currAttr = itBindings->first;
						if (currAttr.getRelationName() == it->getRelationName()
								&& currAttr.getAttrName() == activeLoop) {
							/* Found info needed! */
//							cout << "Binding found!!!" << endl;
//							cout << currAttr.getRelationName() << "_" << currAttr.getAttrName() << endl;
							RawValueMemory mem_activeTuple = itBindings->second;
							argMem = mem_activeTuple.mem;
							isNull = mem_activeTuple.isNull;
						}
					}
				}
			}
		}
		/*else	{
			LOG(WARNING) << "[Expression Generator: ] No active relation found - Non-record case (OR e IS A TOPMOST EXPR.!)";
			 Have seen this occurring in Nest
			int relationsCount = 0;
			for(it = activeVars.begin(); it != activeVars.end(); it++)	{
				RecordAttribute currAttr = it->first;
				cout << currAttr.getRelationName() <<" and "<< currAttr.getAttrName() << endl;
				//XXX Does 1st part of check ever get satisfied? activeRelation is empty here
				if(currAttr.getRelationName() == activeRelation && currAttr.getAttrName() == activeLoop)	{

					argMem = (it->second).mem;
					isNull = (it->second).isNull;
					relationsCount++;
				}
			}
			if (!relationsCount) {
				string error_msg = string("[Expression Generator: ] Could not find tuple information");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			} else if (relationsCount > 1) {
				string error_msg =
						string("[Expression Generator: ] Could not distinguish appropriate bindings");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
		}*/
	}

	RawValue valWrapper;
	/* XXX Should this load be removed?? Will an optimization pass realize it is not needed? */
	valWrapper.value = Builder->CreateLoad(argMem);
	valWrapper.isNull = isNull;//context->createFalse();

#ifdef DEBUG
	{
		Function* debugInt = context->getFunction("printi64");
		vector<Value*> ArgsV;

		ArgsV.push_back(valWrapper.value);
		Builder->CreateCall(debugInt, ArgsV);
		ArgsV.clear();
		ArgsV.push_back(context->createInt64(100000001));
		Builder->CreateCall(debugInt, ArgsV);
	}
#endif

	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::RecordProjection *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const Builder		= context->getBuilder();
	activeRelation 					= e->getOriginalRelationName();
	string projName 				= e->getProjectionName();

	Plugin* plugin 					= catalog.getPlugin(activeRelation);

	{
		//if (plugin->getPluginType() != PGBINARY) {
			/* Cache Logic */
			/* XXX Apply in other visitors too! */
			CachingService& cache = CachingService::getInstance();
//			cout << "LOOKING FOR STUFF FROM "<<e->getRelationName() << "."
//					<< e->getAttribute().getAttrName()<< endl;
			CacheInfo info = cache.getCache(e);
			if (info.structFieldNo != -1) {
#ifdef DEBUGCACHING
				cout << "[Generator: ] Expression found for "
						<< e->getOriginalRelationName() << "."
						<< e->getAttribute().getAttrName() << "!" << endl;
#endif
				if (!cache.getCacheIsFull(e)) {
#ifdef DEBUGCACHING
					cout << "...but is not useable " << endl;
#endif
				} else {
					return plugin->readCachedValue(info, currState);
				}
			} else {
#ifdef DEBUGCACHING
			cout << "[Generator: ] No cache found for "
					<< e->getOriginalRelationName() << "."
					<< e->getAttribute().getAttrName() << "!" << endl;
#endif
			}
		//}
	}

	RawValue record					= e->getExpr()->accept(*this);
	//Resetting activeRelation here would break nested-record-projections
	//activeRelation = "";
	if(plugin == NULL)	{
		string error_msg = string("[Expression Generator: ] No plugin provided");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}	else	{
		Bindings bindings = { &currState, record };
		RawValueMemory mem_path;
		RawValueMemory mem_val;
//		cout << "Active RelationProj: " << activeRelation << "_" << e->getProjectionName() << endl;
		if (e->getProjectionName() != activeLoop) {
			//Path involves a projection / an object
			mem_path = plugin->readPath(activeRelation, bindings,
					e->getProjectionName().c_str(),e->getAttribute());
		} else {
			//Path involves a primitive datatype
			//(e.g., the result of unnesting a list of primitives)
			//cout << "PROJ: " << activeRelation << endl;
			Plugin* pg = catalog.getPlugin(activeRelation);
			RecordAttribute tupleIdentifier = RecordAttribute(activeRelation,
					activeLoop,pg->getOIDType());
			map<RecordAttribute, RawValueMemory>::const_iterator it =
					currState.getBindings().find(tupleIdentifier);
			if (it == currState.getBindings().end()) {
				string error_msg =
						"[Expression Generator: ] Current tuple binding not found";
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}
			mem_path = it->second;
		}
		mem_val = plugin->readValue(mem_path, e->getExpressionType());
		Value *val = Builder->CreateLoad(mem_val.mem);
#ifdef DEBUG
		{
			/* Printing the pos. to be marked */
//			if(e->getProjectionName() == "age") {
//				cout << "AGE! " << endl;
//			if (e->getProjectionName() == "c2") {
//				cout << "C2! " << endl;
			if (e->getProjectionName() == "dim") {
				cout << "dim! " << endl;
				map<RecordAttribute, RawValueMemory>::const_iterator it =
						currState.getBindings().begin();
				for (; it != currState.getBindings().end(); it++) {
					string relname = it->first.getRelationName();
					string attrname = it->first.getAttrName();
					cout << "Available: " << relname << "_" << attrname << endl;
				}

				/* Radix treats this as int64 (!) */
//				val->getType()->dump();
				Function* debugInt = context->getFunction("printi");
				Function* debugInt64 = context->getFunction("printi64");
				vector<Value*> ArgsV;
				ArgsV.clear();
				ArgsV.push_back(val);
				Builder->CreateCall(debugInt, ArgsV);
			} else {
				cout << "Other projection - " << e->getProjectionName() << endl;
			}
		}
#endif
		RawValue valWrapper;
		valWrapper.value = val;
		valWrapper.isNull = mem_val.isNull;
		return valWrapper;
	}
}

RawValue ExpressionGeneratorVisitor::visit(expressions::IfThenElse *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	LLVMContext& llvmContext		= context->getLLVMContext();
	Function *F 					= TheBuilder->GetInsertBlock()->getParent();

	RawValue ifCond 		= e->getIfCond()->accept(*this);

	//Prepare result
//	AllocaInst* mem_result = context->CreateEntryBlockAlloca(F, "ifElseResult", (ifResult.value)->getType());
//	AllocaInst* mem_result_isNull = context->CreateEntryBlockAlloca(F, "ifElseResultIsNull", (ifResult.isNull)->getType());
	AllocaInst* mem_result = NULL;
	AllocaInst* mem_result_isNull = NULL;

	//Prepare blocks
	BasicBlock *ThenBB;
	BasicBlock *ElseBB;
	BasicBlock *MergeBB = BasicBlock::Create(llvmContext,  "ifExprCont", F);
	context->CreateIfElseBlocks(F,"ifExprThen","ifExprElse",&ThenBB,&ElseBB,MergeBB);

	//if
	TheBuilder->CreateCondBr(ifCond.value, ThenBB, ElseBB);

	//then
	TheBuilder->SetInsertPoint(ThenBB);
	RawValue ifResult = e->getIfResult()->accept(*this);
	mem_result = context->CreateEntryBlockAlloca(F, "ifElseResult", (ifResult.value)->getType());
	mem_result_isNull = context->CreateEntryBlockAlloca(F, "ifElseResultIsNull", (ifResult.isNull)->getType());

	TheBuilder->CreateStore(ifResult.value,mem_result);
	TheBuilder->CreateStore(ifResult.isNull,mem_result_isNull);
	TheBuilder->CreateBr(MergeBB);

	//else
	TheBuilder->SetInsertPoint(ElseBB);
	RawValue elseResult = e->getElseResult()->accept(*this);
	TheBuilder->CreateStore(elseResult.value,mem_result);
	TheBuilder->CreateStore(elseResult.isNull,mem_result_isNull);
	TheBuilder->CreateBr(MergeBB);

	//cont.
	TheBuilder->SetInsertPoint(MergeBB);
	RawValue valWrapper;
	valWrapper.value = TheBuilder->CreateLoad(mem_result);
	valWrapper.isNull = TheBuilder->CreateLoad(mem_result_isNull);

	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::EqExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();
		switch (id) {
		case INT:
		{
#ifdef DEBUG
{
			vector<Value*> ArgsV;
			ArgsV.clear();
			ArgsV.push_back(left.value);
			Function* debugInt = context->getFunction("printi");
			TheBuilder->CreateCall(debugInt, ArgsV);
}
#endif
			valWrapper.value = TheBuilder->CreateICmpEQ(left.value, right.value);
			return valWrapper;
		}
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpOEQ(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpEQ(left.value, right.value);
			return valWrapper;
		//XXX Does this code work if we are iterating over a json primitive array?
		//Example: ["alpha","beta","gamma"]
		case STRING: {
			vector<Value*> ArgsV;
			ArgsV.push_back(left.value);
			ArgsV.push_back(right.value);
			Function* stringEquality = context->getFunction("equalStringObjs");
			valWrapper.value = TheBuilder->CreateCall(stringEquality, ArgsV,
					"equalStringObjsCall");
			return valWrapper;
		}
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::NeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateICmpNE(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpONE(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpNE(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::GeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateICmpSGE(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpOGE(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpSGE(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::GtExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateICmpSGT(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpOGT(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpSGT(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::LeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateICmpSLE(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpOLE(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpSLE(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::LtExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
#ifdef DEBUG
{
			vector<Value*> ArgsV;
			ArgsV.clear();
			ArgsV.push_back(left.value);
			Function* debugInt = context->getFunction("printi");
			TheBuilder->CreateCall(debugInt, ArgsV);
}
#endif
			valWrapper.value = TheBuilder->CreateICmpSLT(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFCmpOLT(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateICmpSLT(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::AddExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateAdd(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFAdd(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateAdd(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::SubExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateSub(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFSub(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateSub(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::MultExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateMul(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFMul(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateMul(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::DivExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			valWrapper.value = TheBuilder->CreateSDiv(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = TheBuilder->CreateFDiv(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = TheBuilder->CreateSDiv(left.value, right.value);
			return valWrapper;
		case STRING:
			LOG(ERROR)<< "[ExpressionGeneratorVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionGeneratorVisitor]: Unknown Input"));
		}
	}
	throw runtime_error(string("[ExpressionGeneratorVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionGeneratorVisitor::visit(expressions::AndExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	RawValue left = e->getLeftOperand()->accept(*this);
	RawValue right = e->getRightOperand()->accept(*this);
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();
	valWrapper.value = TheBuilder->CreateAnd(left.value, right.value);
	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::OrExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
		RawValue left = e->getLeftOperand()->accept(*this);
		RawValue right = e->getRightOperand()->accept(*this);
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();
		valWrapper.value = TheBuilder->CreateOr(left.value, right.value);
		return valWrapper;
}


