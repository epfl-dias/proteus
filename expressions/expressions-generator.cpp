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

	return valWrapper;
}

RawValue ExpressionGeneratorVisitor::visit(expressions::RecordProjection *e) {
	RawCatalog& catalog = RawCatalog::getInstance();
	IRBuilder<>* const Builder = context->getBuilder();
	activeRelation = e->getOriginalRelationName();
	string projName = e->getProjectionName();

	Plugin* plugin = catalog.getPlugin(activeRelation);

	if (plugin != NULL) {
		/* Cache Logic */
		/* XXX Apply in other visitors too! */
		CachingService& cache = CachingService::getInstance();
		CacheInfo info = cache.getCache(e);

		/* Must also make sure that no explicit binding exists => No duplicate work */
		map<RecordAttribute, RawValueMemory>::const_iterator it =
				currState.getBindings().find(e->getAttribute());
#ifdef DEBUGCACHING
		if(it != currState.getBindings().end())	{
			cout << "Even if cached, binding's already there!" << endl;
		}
#endif
		if (info.structFieldNo != -1 && it == currState.getBindings().end()) {
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

	RawValue record = e->getExpr()->accept(*this);
	//Resetting activeRelation here would break nested-record-projections
	if (plugin == NULL) {
		string error_msg = string(
				"[Expression Generator: ] No plugin provided");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	} else {
		Bindings bindings = { &currState, record };
		RawValueMemory mem_path;
		RawValueMemory mem_val;
		//cout << "Active RelationProj: " << activeRelation << "_" << e->getProjectionName() << endl;
		if (e->getProjectionName() != activeLoop) {
			//Path involves a projection / an object
			mem_path = plugin->readPath(activeRelation, bindings,
					e->getProjectionName().c_str(), e->getAttribute());
		} else {
			//Path involves a primitive datatype
			//(e.g., the result of unnesting a list of primitives)
			//cout << "PROJ: " << activeRelation << endl;
			Plugin* pg = catalog.getPlugin(activeRelation);
			RecordAttribute tupleIdentifier = RecordAttribute(activeRelation,
					activeLoop, pg->getOIDType());
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
		RawValue valWrapper;
		valWrapper.value = val;
		valWrapper.isNull = mem_val.isNull;
		return valWrapper;
	}
}


RawValue ExpressionGeneratorVisitor::visit(expressions::RecordConstruction *e) {

	RawCatalog& catalog 	   = RawCatalog::getInstance();
	IRBuilder<>* const Builder = context->getBuilder();
	Function *F 			   = Builder->GetInsertBlock()->getParent();

	/* Evaluate new attribute expressions and keep their types */
	vector<RawValue> valuesForStruct;
	vector<Type*> typesForStruct;
	list<expressions::AttributeConstruction>::const_iterator it = e->getAtts().begin();
	for(; it != e->getAtts().end(); it++)
	{
		RawValue val = (it->getExpression())->accept(*this);
		valuesForStruct.push_back(val);
		typesForStruct.push_back(val.value->getType());
	}

	/* Construct struct type to hold record */
	StructType* recStructType = context->CreateCustomStruct(typesForStruct);
	AllocaInst* mem_struct =
			context->CreateEntryBlockAlloca(F, "recConstr",recStructType);

	vector<RawValue>::const_iterator itVal = valuesForStruct.begin();
	int i = 0;
	for (; itVal != valuesForStruct.end(); itVal++) {
		context->updateStructElem(itVal->value,mem_struct,i++);
	}
	RawValue recStruct;
	recStruct.value = Builder->CreateLoad(mem_struct);
	recStruct.isNull = context->createFalse();

	return recStruct;
}

//RawValue ExpressionGeneratorVisitor::visit(expressions::RecordProjection *e) {
//	RawCatalog& catalog 			= RawCatalog::getInstance();
//	IRBuilder<>* const Builder		= context->getBuilder();
//	activeRelation 					= e->getOriginalRelationName();
//	string projName 				= e->getProjectionName();
//
//	Plugin* plugin 					= catalog.getPlugin(activeRelation);
//
//	{
//		//if (plugin->getPluginType() != PGBINARY) {
//		/* Cache Logic */
//		/* XXX Apply in other visitors too! */
//		CachingService& cache = CachingService::getInstance();
//		//cout << "LOOKING FOR STUFF FROM "<<e->getRelationName() << "."
//		//<< e->getAttribute().getAttrName()<< endl;
//		CacheInfo info = cache.getCache(e);
//		if (info.structFieldNo != -1) {
//#ifdef DEBUGCACHING
//			cout << "[Generator: ] Expression found for "
//					<< e->getOriginalRelationName() << "."
//					<< e->getAttribute().getAttrName() << "!" << endl;
//#endif
//			if (!cache.getCacheIsFull(e)) {
//#ifdef DEBUGCACHING
//				cout << "...but is not useable " << endl;
//#endif
//			} else {
//				return plugin->readCachedValue(info, currState);
//			}
//		} else {
//#ifdef DEBUGCACHING
//			cout << "[Generator: ] No cache found for "
//					<< e->getOriginalRelationName() << "."
//					<< e->getAttribute().getAttrName() << "!" << endl;
//#endif
//		}
//		//}
//	}
//
//	RawValue record					= e->getExpr()->accept(*this);
//	//Resetting activeRelation here would break nested-record-projections
//	//activeRelation = "";
//	if(plugin == NULL)	{
//		string error_msg = string("[Expression Generator: ] No plugin provided");
//		LOG(ERROR) << error_msg;
//		throw runtime_error(error_msg);
//	}	else	{
//		Bindings bindings = { &currState, record };
//		RawValueMemory mem_path;
//		RawValueMemory mem_val;
//		//cout << "Active RelationProj: " << activeRelation << "_" << e->getProjectionName() << endl;
//		if (e->getProjectionName() != activeLoop) {
//			//Path involves a projection / an object
//			mem_path = plugin->readPath(activeRelation, bindings,
//					e->getProjectionName().c_str(),e->getAttribute());
//		} else {
//			//Path involves a primitive datatype
//			//(e.g., the result of unnesting a list of primitives)
//			//cout << "PROJ: " << activeRelation << endl;
//			Plugin* pg = catalog.getPlugin(activeRelation);
//			RecordAttribute tupleIdentifier = RecordAttribute(activeRelation,
//					activeLoop,pg->getOIDType());
//			map<RecordAttribute, RawValueMemory>::const_iterator it =
//					currState.getBindings().find(tupleIdentifier);
//			if (it == currState.getBindings().end()) {
//				string error_msg =
//						"[Expression Generator: ] Current tuple binding not found";
//				LOG(ERROR)<< error_msg;
//				throw runtime_error(error_msg);
//			}
//			mem_path = it->second;
//		}
//		mem_val = plugin->readValue(mem_path, e->getExpressionType());
//		Value *val = Builder->CreateLoad(mem_val.mem);
//#ifdef DEBUG
//		{
//			/* Printing the pos. to be marked */
////			if(e->getProjectionName() == "age") {
////				cout << "AGE! " << endl;
////			if (e->getProjectionName() == "c2") {
////				cout << "C2! " << endl;
//			if (e->getProjectionName() == "dim") {
//				cout << "dim! " << endl;
//				map<RecordAttribute, RawValueMemory>::const_iterator it =
//						currState.getBindings().begin();
//				for (; it != currState.getBindings().end(); it++) {
//					string relname = it->first.getRelationName();
//					string attrname = it->first.getAttrName();
//					cout << "Available: " << relname << "_" << attrname << endl;
//				}
//
//				/* Radix treats this as int64 (!) */
////				val->getType()->dump();
//				Function* debugInt = context->getFunction("printi");
//				Function* debugInt64 = context->getFunction("printi64");
//				vector<Value*> ArgsV;
//				ArgsV.clear();
//				ArgsV.push_back(val);
//				Builder->CreateCall(debugInt, ArgsV);
//			} else {
//				cout << "Other projection - " << e->getProjectionName() << endl;
//			}
//		}
//#endif
//		RawValue valWrapper;
//		valWrapper.value = val;
//		valWrapper.isNull = mem_val.isNull;
//		return valWrapper;
//	}
//}

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
	Function *F = TheBuilder->GetInsertBlock()->getParent();

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
//		case STRING: {
//			vector<Value*> ArgsV;
//			ArgsV.push_back(left.value);
//			ArgsV.push_back(right.value);
//			Function* stringEquality = context->getFunction("equalStringObjs");
//			valWrapper.value = TheBuilder->CreateCall(stringEquality, ArgsV,
//					"equalStringObjsCall");
//			return valWrapper;
//		}
		case STRING: {
//			vector<Value*> ArgsV;
//			ArgsV.push_back(left.value);
//			ArgsV.push_back(right.value);
//			Function* stringEquality = context->getFunction("equalStringObjs");
//			valWrapper.value = TheBuilder->CreateCall(stringEquality, ArgsV,
//					"equalStringObjsCall");
//			return valWrapper;

			//{ i8*, i32 }
			StructType *stringObjType = context->CreateStringStruct();
			AllocaInst *mem_str_obj_left = context->CreateEntryBlockAlloca(F,
								string("revertedStringObjLeft"), stringObjType);
			AllocaInst *mem_str_obj_right = context->CreateEntryBlockAlloca(F,
											string("revertedStringObjRight"), stringObjType);
			TheBuilder->CreateStore(left.value,mem_str_obj_left);
			TheBuilder->CreateStore(right.value,mem_str_obj_right);

			Value *ptr_s1 = context->getStructElem(mem_str_obj_left,0);
			Value *ptr_s2 = context->getStructElem(mem_str_obj_right,0);
			Value *len = context->getStructElem(mem_str_obj_left,1);

			Value *toCmp = context->createInt32(0);
			valWrapper.value = TheBuilder->CreateICmpEQ(mystrncmp(ptr_s1,ptr_s2,len).value,toCmp);
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

int mystrncmp(const char *s1, const char *s2, size_t n)
{
    for ( ; n > 0; s1++, s2++, --n)
	if (*s1 != *s2)
	    return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
	else if (*s1 == '\0')
	    return 0;
    return 0;
}


//RawValue ExpressionGeneratorVisitor::mystrncmp(Value *ptr_s1, Value *ptr_s2,
//		Value *int64_n) {
//
//	IRBuilder<>* Builder = context->getBuilder();
//	LLVMContext& llvmContext = context->getLLVMContext();
//	Function *F = Builder->GetInsertBlock()->getParent();
//	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
//
//	// Constant Definitions
//	ConstantInt* const_int32_5 = ConstantInt::get(llvmContext,
//			APInt(32, StringRef("1"), 10));
//	ConstantInt* const_int32_6 = ConstantInt::get(llvmContext,
//			APInt(32, StringRef("-2"), 10));
//	ConstantInt* const_int64_7 = ConstantInt::get(llvmContext,
//			APInt(64, StringRef("0"), 10));
//	ConstantInt* const_int32_8 = ConstantInt::get(llvmContext,
//			APInt(32, StringRef("-1"), 10));
//	ConstantInt* const_int32_9 = ConstantInt::get(llvmContext,
//			APInt(32, StringRef("0"), 10));
//	ConstantInt* const_int64_10 = ConstantInt::get(llvmContext,
//			APInt(64, StringRef("-1"), 10));
//
//	BasicBlock* label_entry = BasicBlock::Create(llvmContext, "entry", F, 0);
//	BasicBlock* label_for_cond = BasicBlock::Create(llvmContext, "for.cond", F,
//			0);
//	BasicBlock* label_for_body = BasicBlock::Create(llvmContext, "for.body", F,
//			0);
//	BasicBlock* label_if_then = BasicBlock::Create(llvmContext, "if.then", F,
//			0);
//	BasicBlock* label_if_else = BasicBlock::Create(llvmContext, "if.else", F,
//			0);
//	BasicBlock* label_if_then8 = BasicBlock::Create(llvmContext, "if.then8", F,
//			0);
//	BasicBlock* label_if_end = BasicBlock::Create(llvmContext, "if.end", F, 0);
//	BasicBlock* label_if_end9 = BasicBlock::Create(llvmContext, "if.end9", F,
//			0);
//	BasicBlock* label_for_inc = BasicBlock::Create(llvmContext, "for.inc", F,
//			0);
//	BasicBlock* label_for_end = BasicBlock::Create(llvmContext, "for.end", F,
//			0);
//	BasicBlock* label_if_then12 = BasicBlock::Create(llvmContext, "if.then12",
//			F, 0);
//	BasicBlock* label_if_end13 = BasicBlock::Create(llvmContext, "if.end13", F,
//			0);
//
//	/* Connect w. previous */
//	Builder->CreateBr(label_entry);
//
//
//	// Block entry (label_entry)
//	AllocaInst* ptr_s1_addr = new AllocaInst(charPtrType, "s1.addr",
//			label_entry);
//	ptr_s1_addr->setAlignment(8);
//	AllocaInst* ptr_s2_addr = new AllocaInst(charPtrType, "s2.addr",
//			label_entry);
//	ptr_s2_addr->setAlignment(8);
//	AllocaInst* ptr_n_addr = new AllocaInst(IntegerType::get(llvmContext, 64),
//			"n.addr", label_entry);
//	ptr_n_addr->setAlignment(8);
//	AllocaInst* ptr_result = new AllocaInst(IntegerType::get(llvmContext, 32),
//			"result", label_entry);
//	ptr_result->setAlignment(4);
//	StoreInst* void_11 = new StoreInst(ptr_s1, ptr_s1_addr, false, label_entry);
//	void_11->setAlignment(8);
//	StoreInst* void_12 = new StoreInst(ptr_s2, ptr_s2_addr, false, label_entry);
//	void_12->setAlignment(8);
//	StoreInst* void_13 = new StoreInst(int64_n, ptr_n_addr, false, label_entry);
//	void_13->setAlignment(8);
//	StoreInst* void_14 = new StoreInst(const_int32_6, ptr_result, false,
//			label_entry);
//	void_14->setAlignment(4);
//	BranchInst::Create(label_for_cond, label_entry);
//
//	// Block for.cond (label_for_cond)
//	LoadInst* int64_16 = new LoadInst(ptr_n_addr, "", false, label_for_cond);
//	int64_16->setAlignment(8);
//	ICmpInst* int1_cmp = new ICmpInst(*label_for_cond, ICmpInst::ICMP_UGT,
//			int64_16, const_int64_7, "cmp");
//	BranchInst::Create(label_for_body, label_for_end, int1_cmp, label_for_cond);
//
//	// Block for.body (label_for_body)
//	LoadInst* ptr_18 = new LoadInst(ptr_s1_addr, "", false, label_for_body);
//	ptr_18->setAlignment(8);
//	LoadInst* int8_19 = new LoadInst(ptr_18, "", false, label_for_body);
//	int8_19->setAlignment(1);
//	CastInst* int32_conv = new SExtInst(int8_19,
//			IntegerType::get(llvmContext, 32), "conv", label_for_body);
//	LoadInst* ptr_20 = new LoadInst(ptr_s2_addr, "", false, label_for_body);
//	ptr_20->setAlignment(8);
//	LoadInst* int8_21 = new LoadInst(ptr_20, "", false, label_for_body);
//	int8_21->setAlignment(1);
//	CastInst* int32_conv1 = new SExtInst(int8_21,
//			IntegerType::get(llvmContext, 32), "conv1", label_for_body);
//	ICmpInst* int1_cmp2 = new ICmpInst(*label_for_body, ICmpInst::ICMP_NE,
//			int32_conv, int32_conv1, "cmp2");
//	BranchInst::Create(label_if_then, label_if_else, int1_cmp2, label_for_body);
//
//	// Block if.then (label_if_then)
//	LoadInst* ptr_23 = new LoadInst(ptr_s1_addr, "", false, label_if_then);
//	ptr_23->setAlignment(8);
//	LoadInst* int8_24 = new LoadInst(ptr_23, "", false, label_if_then);
//	int8_24->setAlignment(1);
//	CastInst* int32_conv3 = new ZExtInst(int8_24,
//			IntegerType::get(llvmContext, 32), "conv3", label_if_then);
//	LoadInst* ptr_25 = new LoadInst(ptr_s2_addr, "", false, label_if_then);
//	ptr_25->setAlignment(8);
//	LoadInst* int8_26 = new LoadInst(ptr_25, "", false, label_if_then);
//	int8_26->setAlignment(1);
//	CastInst* int32_conv4 = new ZExtInst(int8_26,
//			IntegerType::get(llvmContext, 32), "conv4", label_if_then);
//	ICmpInst* int1_cmp5 = new ICmpInst(*label_if_then, ICmpInst::ICMP_SLT,
//			int32_conv3, int32_conv4, "cmp5");
//	SelectInst* int32_cond = SelectInst::Create(int1_cmp5, const_int32_8,
//			const_int32_5, "cond", label_if_then);
//	StoreInst* void_27 = new StoreInst(int32_cond, ptr_result, false,
//			label_if_then);
//	void_27->setAlignment(4);
//	BranchInst::Create(label_for_end, label_if_then);
//
//	// Block if.else (label_if_else)
//	LoadInst* ptr_29 = new LoadInst(ptr_s1_addr, "", false, label_if_else);
//	ptr_29->setAlignment(8);
//	LoadInst* int8_30 = new LoadInst(ptr_29, "", false, label_if_else);
//	int8_30->setAlignment(1);
//	CastInst* int32_conv6 = new SExtInst(int8_30,
//			IntegerType::get(llvmContext, 32), "conv6", label_if_else);
//	ICmpInst* int1_cmp7 = new ICmpInst(*label_if_else, ICmpInst::ICMP_EQ,
//			int32_conv6, const_int32_9, "cmp7");
//	BranchInst::Create(label_if_then8, label_if_end, int1_cmp7, label_if_else);
//
//	// Block if.then8 (label_if_then8)
//	StoreInst* void_32 = new StoreInst(const_int32_9, ptr_result, false,
//			label_if_then8);
//	void_32->setAlignment(4);
//	BranchInst::Create(label_for_end, label_if_then8);
//
//	// Block if.end (label_if_end)
//	BranchInst::Create(label_if_end9, label_if_end);
//
//	// Block if.end9 (label_if_end9)
//	BranchInst::Create(label_for_inc, label_if_end9);
//
//	// Block for.inc (label_for_inc)
//	LoadInst* ptr_36 = new LoadInst(ptr_s1_addr, "", false, label_for_inc);
//	ptr_36->setAlignment(8);
////	GetElementPtrInst* ptr_incdec_ptr = GetElementPtrInst::Create(
////			IntegerType::get(llvmContext, 8), ptr_36, { const_int32_5 },
////			"incdec.ptr", label_for_inc);
////	GetElementPtrInst* ptr_incdec_ptr = GetElementPtrInst::Create(
////				ptr_36, { const_int32_5 },
////				"incdec.ptr", label_for_inc);
//	GetElementPtrInst* ptr_incdec_ptr = GetElementPtrInst::Create(
//					ptr_36, const_int32_5 ,
//					"incdec.ptr", label_for_inc);
//	StoreInst* void_37 = new StoreInst(ptr_incdec_ptr, ptr_s1_addr, false,
//			label_for_inc);
//	void_37->setAlignment(8);
//	LoadInst* ptr_38 = new LoadInst(ptr_s2_addr, "", false, label_for_inc);
//	ptr_38->setAlignment(8);
////	GetElementPtrInst* ptr_incdec_ptr10 = GetElementPtrInst::Create(
////			IntegerType::get(llvmContext, 8), ptr_38, { const_int32_5 },
////			"incdec.ptr10", label_for_inc);
////	GetElementPtrInst* ptr_incdec_ptr10 = GetElementPtrInst::Create(
////				ptr_38, { const_int32_5 },"incdec.ptr10", label_for_inc);
//	GetElementPtrInst* ptr_incdec_ptr10 = GetElementPtrInst::Create(
//					ptr_38, const_int32_5 ,"incdec.ptr10", label_for_inc);
//	StoreInst* void_39 = new StoreInst(ptr_incdec_ptr10, ptr_s2_addr, false,
//			label_for_inc);
//	void_39->setAlignment(8);
//	LoadInst* int64_40 = new LoadInst(ptr_n_addr, "", false, label_for_inc);
//	int64_40->setAlignment(8);
//	BinaryOperator* int64_dec = BinaryOperator::Create(Instruction::Add,
//			int64_40, const_int64_10, "dec", label_for_inc);
//	StoreInst* void_41 = new StoreInst(int64_dec, ptr_n_addr, false,
//			label_for_inc);
//	void_41->setAlignment(8);
//	BranchInst::Create(label_for_cond, label_for_inc);
//
//	// Block for.end (label_for_end)
//	LoadInst* int32_43 = new LoadInst(ptr_result, "", false, label_for_end);
//	int32_43->setAlignment(4);
//	ICmpInst* int1_cmp11 = new ICmpInst(*label_for_end, ICmpInst::ICMP_EQ,
//			int32_43, const_int32_6, "cmp11");
//	BranchInst::Create(label_if_then12, label_if_end13, int1_cmp11,
//			label_for_end);
//
//	// Block if.then12 (label_if_then12)
//	StoreInst* void_45 = new StoreInst(const_int32_9, ptr_result, false,
//			label_if_then12);
//	void_45->setAlignment(4);
//	BranchInst::Create(label_if_end13, label_if_then12);
//
//	// Block if.end13 (label_if_end13)
//	LoadInst* int32_47 = new LoadInst(ptr_result, "", false, label_if_end13);
//	int32_47->setAlignment(4);
////	ReturnInst::Create(llvmContext, int32_47, label_if_end13);
//	RawValue valWrapper;
//	valWrapper.isNull = context->createFalse();
//	valWrapper.value = int32_47;
//
//	Builder->SetInsertPoint(label_if_end13);
//
//	return valWrapper;
//}

/* returns 0/1/-1 */
RawValue ExpressionGeneratorVisitor::mystrncmp(Value *ptr_s1, Value *ptr_s2,
		Value *int32_n) {

	IRBuilder<>* Builder = context->getBuilder();
	LLVMContext& llvmContext = context->getLLVMContext();
	Function *F = Builder->GetInsertBlock()->getParent();
	PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);

	// Constant Definitions
	ConstantInt* const_int32_4 = ConstantInt::get(llvmContext,
			APInt(32, StringRef("1"), 10));
	ConstantInt* const_int32_5 = ConstantInt::get(llvmContext,
			APInt(32, StringRef("-2"), 10));
	ConstantInt* const_int32_6 = ConstantInt::get(llvmContext,
			APInt(32, StringRef("0"), 10));
	ConstantInt* const_int32_7 = ConstantInt::get(llvmContext,
			APInt(32, StringRef("-1"), 10));

	BasicBlock* label_entry = BasicBlock::Create(llvmContext, "entry", F, 0);
	BasicBlock* label_for_cond = BasicBlock::Create(llvmContext, "for.cond", F,
			0);
	BasicBlock* label_for_body = BasicBlock::Create(llvmContext, "for.body", F,
			0);
	BasicBlock* label_if_then = BasicBlock::Create(llvmContext, "if.then", F,
			0);
	BasicBlock* label_if_else = BasicBlock::Create(llvmContext, "if.else", F,
			0);
	BasicBlock* label_if_then11 = BasicBlock::Create(llvmContext, "if.then11",
			F, 0);
	BasicBlock* label_if_end = BasicBlock::Create(llvmContext, "if.end", F, 0);
	BasicBlock* label_if_end12 = BasicBlock::Create(llvmContext, "if.end12", F,
			0);
	BasicBlock* label_for_inc = BasicBlock::Create(llvmContext, "for.inc", F,
			0);
	BasicBlock* label_for_end = BasicBlock::Create(llvmContext, "for.end", F,
			0);
	BasicBlock* label_if_then16 = BasicBlock::Create(llvmContext, "if.then16",
			F, 0);
	BasicBlock* label_if_end17 = BasicBlock::Create(llvmContext, "if.end17", F,
			0);

	/* Connect w. previous */
	Builder->CreateBr(label_entry);

	// Block entry (label_entry)
	AllocaInst* ptr_s1_addr = new AllocaInst(charPtrType, "s1.addr",
			label_entry);
	ptr_s1_addr->setAlignment(8);
	AllocaInst* ptr_s2_addr = new AllocaInst(charPtrType, "s2.addr",
			label_entry);
	ptr_s2_addr->setAlignment(8);
	AllocaInst* ptr_n_addr = new AllocaInst(IntegerType::get(llvmContext, 32),
			"n.addr", label_entry);
	ptr_n_addr->setAlignment(4);
	AllocaInst* ptr_result = new AllocaInst(IntegerType::get(llvmContext, 32),
			"result", label_entry);
	ptr_result->setAlignment(4);
	StoreInst* void_8 = new StoreInst(ptr_s1, ptr_s1_addr, false, label_entry);
	void_8->setAlignment(8);
	StoreInst* void_9 = new StoreInst(ptr_s2, ptr_s2_addr, false, label_entry);
	void_9->setAlignment(8);
	StoreInst* void_10 = new StoreInst(int32_n, ptr_n_addr, false, label_entry);
	void_10->setAlignment(4);
	StoreInst* void_11 = new StoreInst(const_int32_5, ptr_result, false,
			label_entry);
	void_11->setAlignment(4);
	BranchInst::Create(label_for_cond, label_entry);

	// Block for.cond (label_for_cond)
	LoadInst* int32_13 = new LoadInst(ptr_n_addr, "", false, label_for_cond);
	int32_13->setAlignment(4);
	ICmpInst* int1_cmp = new ICmpInst(*label_for_cond, ICmpInst::ICMP_SGT,
			int32_13, const_int32_6, "cmp");
	BranchInst::Create(label_for_body, label_for_end, int1_cmp, label_for_cond);

	// Block for.body (label_for_body)
	LoadInst* ptr_15 = new LoadInst(ptr_s1_addr, "", false, label_for_body);
	ptr_15->setAlignment(8);
	LoadInst* int8_16 = new LoadInst(ptr_15, "", false, label_for_body);
	int8_16->setAlignment(1);
	CastInst* int32_conv = new SExtInst(int8_16,
			IntegerType::get(llvmContext, 32), "conv", label_for_body);
	LoadInst* ptr_17 = new LoadInst(ptr_s2_addr, "", false, label_for_body);
	ptr_17->setAlignment(8);
	LoadInst* int8_18 = new LoadInst(ptr_17, "", false, label_for_body);
	int8_18->setAlignment(1);
	CastInst* int32_conv1 = new SExtInst(int8_18,
			IntegerType::get(llvmContext, 32), "conv1", label_for_body);
	ICmpInst* int1_cmp2 = new ICmpInst(*label_for_body, ICmpInst::ICMP_NE,
			int32_conv, int32_conv1, "cmp2");
	BranchInst::Create(label_if_then, label_if_else, int1_cmp2, label_for_body);

	// Block if.then (label_if_then)
	LoadInst* ptr_20 = new LoadInst(ptr_s1_addr, "", false, label_if_then);
	ptr_20->setAlignment(8);
	LoadInst* int8_21 = new LoadInst(ptr_20, "", false, label_if_then);
	int8_21->setAlignment(1);
	CastInst* int32_conv4 = new ZExtInst(int8_21,
			IntegerType::get(llvmContext, 32), "conv4", label_if_then);
	LoadInst* ptr_22 = new LoadInst(ptr_s2_addr, "", false, label_if_then);
	ptr_22->setAlignment(8);
	LoadInst* int8_23 = new LoadInst(ptr_22, "", false, label_if_then);
	int8_23->setAlignment(1);
	CastInst* int32_conv5 = new ZExtInst(int8_23,
			IntegerType::get(llvmContext, 32), "conv5", label_if_then);
	ICmpInst* int1_cmp6 = new ICmpInst(*label_if_then, ICmpInst::ICMP_SLT,
			int32_conv4, int32_conv5, "cmp6");
	SelectInst* int32_cond = SelectInst::Create(int1_cmp6, const_int32_7,
			const_int32_4, "cond", label_if_then);
	StoreInst* void_24 = new StoreInst(int32_cond, ptr_result, false,
			label_if_then);
	void_24->setAlignment(4);
	BranchInst::Create(label_for_end, label_if_then);

	// Block if.else (label_if_else)
	LoadInst* ptr_26 = new LoadInst(ptr_s1_addr, "", false, label_if_else);
	ptr_26->setAlignment(8);
	LoadInst* int8_27 = new LoadInst(ptr_26, "", false, label_if_else);
	int8_27->setAlignment(1);
	CastInst* int32_conv8 = new SExtInst(int8_27,
			IntegerType::get(llvmContext, 32), "conv8", label_if_else);
	ICmpInst* int1_cmp9 = new ICmpInst(*label_if_else, ICmpInst::ICMP_EQ,
			int32_conv8, const_int32_6, "cmp9");
	BranchInst::Create(label_if_then11, label_if_end, int1_cmp9, label_if_else);

	// Block if.then11 (label_if_then11)
	StoreInst* void_29 = new StoreInst(const_int32_6, ptr_result, false,
			label_if_then11);
	void_29->setAlignment(4);
	BranchInst::Create(label_for_end, label_if_then11);

	// Block if.end (label_if_end)
	BranchInst::Create(label_if_end12, label_if_end);

	// Block if.end12 (label_if_end12)
	BranchInst::Create(label_for_inc, label_if_end12);

	// Block for.inc (label_for_inc)
	LoadInst* ptr_33 = new LoadInst(ptr_s1_addr, "", false, label_for_inc);
	ptr_33->setAlignment(8);
	GetElementPtrInst* ptr_incdec_ptr = GetElementPtrInst::Create(ptr_33,
			const_int32_4, "incdec.ptr", label_for_inc);
	StoreInst* void_34 = new StoreInst(ptr_incdec_ptr, ptr_s1_addr, false,
			label_for_inc);
	void_34->setAlignment(8);
	LoadInst* ptr_35 = new LoadInst(ptr_s2_addr, "", false, label_for_inc);
	ptr_35->setAlignment(8);
	GetElementPtrInst* ptr_incdec_ptr13 = GetElementPtrInst::Create(ptr_35,
			const_int32_4, "incdec.ptr13", label_for_inc);
	StoreInst* void_36 = new StoreInst(ptr_incdec_ptr13, ptr_s2_addr, false,
			label_for_inc);
	void_36->setAlignment(8);
	LoadInst* int32_37 = new LoadInst(ptr_n_addr, "", false, label_for_inc);
	int32_37->setAlignment(4);
	BinaryOperator* int32_dec = BinaryOperator::Create(Instruction::Add,
			int32_37, const_int32_7, "dec", label_for_inc);
	StoreInst* void_38 = new StoreInst(int32_dec, ptr_n_addr, false,
			label_for_inc);
	void_38->setAlignment(4);
	BranchInst::Create(label_for_cond, label_for_inc);

	// Block for.end (label_for_end)
	LoadInst* int32_40 = new LoadInst(ptr_result, "", false, label_for_end);
	int32_40->setAlignment(4);
	ICmpInst* int1_cmp14 = new ICmpInst(*label_for_end, ICmpInst::ICMP_EQ,
			int32_40, const_int32_5, "cmp14");
	BranchInst::Create(label_if_then16, label_if_end17, int1_cmp14,
			label_for_end);

	// Block if.then16 (label_if_then16)
	StoreInst* void_42 = new StoreInst(const_int32_6, ptr_result, false,
			label_if_then16);
	void_42->setAlignment(4);
	BranchInst::Create(label_if_end17, label_if_then16);

	// Block if.end17 (label_if_end17)
	LoadInst* int32_44 = new LoadInst(ptr_result, "", false, label_if_end17);
	int32_44->setAlignment(4);

	/* OUTPUT */
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();
	valWrapper.value = int32_44;

	Builder->SetInsertPoint(label_if_end17);

	return valWrapper;
}

/* returns true / false!!*/

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
#ifdef DEBUG
{
			vector<Value*> ArgsV;
			ArgsV.clear();
			ArgsV.push_back(left.value);
			Function* debugF = context->getFunction("printFloat");
			TheBuilder->CreateCall(debugF, ArgsV);
}
#endif
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
#ifdef DEBUG
{
			vector<Value*> ArgsV;
			ArgsV.clear();
			ArgsV.push_back(left.value);
			Function* debugF = context->getFunction("printFloat");
			TheBuilder->CreateCall(debugF, ArgsV);
}
#endif
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

