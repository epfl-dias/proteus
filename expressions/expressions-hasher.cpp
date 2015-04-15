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

#include "expressions/expressions-hasher.hpp"

RawValue ExpressionHasherVisitor::visit(expressions::IntConstant *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *hashInt = context->getFunction("hashInt");
	std::vector<Value*> ArgsV;
	Value *val_int = ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
	ArgsV.push_back(val_int);
	Value *hashResult = context->getBuilder()->CreateCall(hashInt, ArgsV, "hashInt");

	RawValue valWrapper;
	valWrapper.value = hashResult;
	valWrapper.isNull = context->createFalse();

#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif

	return valWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::FloatConstant *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *hashDouble = context->getFunction("hashDouble");
	std::vector<Value*> ArgsV;
	Value *val_double = ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
	ArgsV.push_back(val_double);
	Value *hashResult = context->getBuilder()->CreateCall(hashDouble, ArgsV, "hashDouble");

	RawValue valWrapper;
	valWrapper.value = hashResult;
	valWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return valWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::BoolConstant *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *hashBoolean = context->getFunction("hashBoolean");
	std::vector<Value*> ArgsV;
	Value *val_boolean = ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
	ArgsV.push_back(val_boolean);
	Value *hashResult = context->getBuilder()->CreateCall(hashBoolean, ArgsV, "hashBoolean");

	RawValue valWrapper;
	valWrapper.value = hashResult;
	valWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return valWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::StringConstant *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	size_t hashResultC = hashString(e->getVal());
	Value* hashResult = ConstantInt::get(context->getLLVMContext(), APInt(64, hashResultC));

	RawValue valWrapper;
	valWrapper.value = hashResult;
	valWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return valWrapper;
}


RawValue ExpressionHasherVisitor::visit(expressions::InputArgument *e)
{
	RawCatalog& catalog = RawCatalog::getInstance();
	Function* const F  = context->getGlobalFunction();
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(context->getLLVMContext());

	Function *hashCombine = context->getFunction("combineHashes");
	Value* hashedValue = context->createInt64(0);
	Function* debugInt = context->getFunction("printi");
	vector<Value*> ArgsV;

	const map<RecordAttribute, RawValueMemory>& activeVars =
					currState.getBindings();

	list<RecordAttribute> projections = e->getProjections();
	list<RecordAttribute>::iterator it = projections.begin();

	//Initializing resulting hashed value
	AllocaInst* mem_hashedValue = context->CreateEntryBlockAlloca(F,
			string("hashValue"), int64Type);
	TheBuilder->CreateStore(hashedValue, mem_hashedValue);

	//Is there a case that I am doing extra work?
	//Example: Am I hashing a value that is nested
	//in some value that I have already hashed?

	for(; it != projections.end(); it++)	{
		/* Explicitly looking for OID!!! */
		if(it->getAttrName() == activeLoop)	{
			map<RecordAttribute, RawValueMemory>::const_iterator itBindings;
			for(itBindings = activeVars.begin(); itBindings != activeVars.end(); itBindings++)
			{
				RecordAttribute currAttr = itBindings->first;
				if (currAttr.getRelationName() == it->getRelationName()
						&& currAttr.getAttrName() == activeLoop)
				{
					//Hash value now
					RawValueMemory mem_activeTuple = itBindings->second;

					Plugin* plugin = catalog.getPlugin(currAttr.getRelationName());
					if(plugin == NULL)	{
						string error_msg = string("[Expression Hasher: ] No plugin provided");
						LOG(ERROR) << error_msg;
						throw runtime_error(error_msg);
					}

					//Combine with previous hashed values (if applicable)
					//Does order matter?
					//Does result of retrieving tuple1->tuple2 differ from tuple2->tuple1???
					//(Probably should)
					RawValue partialHash = plugin->hashValue(mem_activeTuple,
							e->getExpressionType());
					ArgsV.clear();
					ArgsV.push_back(hashedValue);
					ArgsV.push_back(partialHash.value);

					hashedValue = TheBuilder->CreateCall(hashCombine, ArgsV,"combineHashesRes");
					TheBuilder->CreateStore(hashedValue, mem_hashedValue);
					break;
				}
			}
		}
	}

	RawValue hashValWrapper;
	hashValWrapper.value = TheBuilder->CreateLoad(mem_hashedValue);
	hashValWrapper.isNull = context->createFalse();
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::RecordProjection *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	activeRelation 					= e->getOriginalRelationName();

	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	/* Need this 'hint' before launching generator,
	 * otherwise (potential) InputArg visitor will crash */
	exprGenerator.setActiveRelation(activeRelation);
	RawValue record					= e->getExpr()->accept(exprGenerator);
	Plugin* plugin 					= catalog.getPlugin(activeRelation);

	//Resetting activeRelation here would break nested-record-projections
	//activeRelation = "";
	if(plugin == NULL)	{
		string error_msg = string("[Expression Generator: ] No plugin provided");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}	else	{
		Bindings bindings = { &currState, record };
		RawValueMemory mem_path;
		RawValue mem_val;
		//cout << "Active Relation: " << e->getProjectionName() << endl;
		if (e->getProjectionName() != activeLoop) {
			//Path involves a projection / an object
			mem_path = plugin->readPath(activeRelation, bindings,
					e->getProjectionName().c_str());
		} else {
			//Path involves a primitive datatype
			//(e.g., the result of unnesting a list of primitives)
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
		mem_val = plugin->hashValue(mem_path, e->getExpressionType());

		return mem_val;
	}
}

RawValue ExpressionHasherVisitor::visit(expressions::IfThenElse *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	LLVMContext& llvmContext		= context->getLLVMContext();
	Function *F 					= TheBuilder->GetInsertBlock()->getParent();
	Type* int64Type = Type::getInt64Ty(llvmContext);

	RawValue hashResult;
	AllocaInst* mem_hashResult = context->CreateEntryBlockAlloca(F, "hashResult", int64Type);

	//Need to evaluate, not hash!
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue ifCond = e->getIfCond()->accept(exprGenerator);

	//Prepare blocks
	BasicBlock *ThenBB;
	BasicBlock *ElseBB;
	BasicBlock *MergeBB = BasicBlock::Create(llvmContext,  "ifExprCont", F);
	context->CreateIfElseBlocks(F,"ifExprThen","ifExprElse",&ThenBB,&ElseBB,MergeBB);

	//if
	TheBuilder->CreateCondBr(ifCond.value, ThenBB, ElseBB);

	//then
	TheBuilder->SetInsertPoint(ThenBB);
	hashResult = e->getIfResult()->accept(*this);
	TheBuilder->CreateStore(hashResult.value,mem_hashResult);
	TheBuilder->CreateBr(MergeBB);

	//else
	TheBuilder->SetInsertPoint(ElseBB);
	hashResult = e->getElseResult()->accept(*this);
	TheBuilder->CreateStore(hashResult.value,mem_hashResult);

	TheBuilder->CreateBr(MergeBB);

	//cont.
	TheBuilder->SetInsertPoint(MergeBB);
	RawValue valWrapper;
	valWrapper.value = TheBuilder->CreateLoad(mem_hashResult);
	valWrapper.isNull = context->createFalse();

	return valWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::EqExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::NeExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::GeExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::GtExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::LeExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::LtExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::AndExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::OrExpression *e)	{
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *hashFunc = context->getFunction("hashBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

	RawValue hashValWrapper;
	hashValWrapper.value = hashResult;
	hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
	return hashValWrapper;
}

RawValue ExpressionHasherVisitor::visit(expressions::AddExpression *e) {
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *hashFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("hashInt");
			break;
		case FLOAT:
			instructionLabel = string("hashDouble");
			break;
		case BOOL:
			instructionLabel = string("hashBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionHasherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionHasherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionHasherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionHasherVisitor]: Unknown Input"));
		}
		hashFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

		RawValue hashValWrapper;
		hashValWrapper.value = hashResult;
		hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
		return hashValWrapper;
	}
	throw runtime_error(string("[ExpressionHasherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionHasherVisitor::visit(expressions::SubExpression *e) {
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *hashFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("hashInt");
			break;
		case FLOAT:
			instructionLabel = string("hashDouble");
			break;
		case BOOL:
			instructionLabel = string("hashBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionHasherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionHasherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionHasherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionHasherVisitor]: Unknown Input"));
		}
		hashFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

		RawValue hashValWrapper;
		hashValWrapper.value = hashResult;
		hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
		return hashValWrapper;
	}
	throw runtime_error(string("[ExpressionHasherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionHasherVisitor::visit(expressions::MultExpression *e) {
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *hashFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("hashInt");
			break;
		case FLOAT:
			instructionLabel = string("hashDouble");
			break;
		case BOOL:
			instructionLabel = string("hashBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionHasherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionHasherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionHasherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionHasherVisitor]: Unknown Input"));
		}
		hashFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

		RawValue hashValWrapper;
		hashValWrapper.value = hashResult;
		hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
		return hashValWrapper;
	}
	throw runtime_error(string("[ExpressionHasherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionHasherVisitor::visit(expressions::DivExpression *e) {
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *hashFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("hashInt");
			break;
		case FLOAT:
			instructionLabel = string("hashDouble");
			break;
		case BOOL:
			instructionLabel = string("hashBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionHasherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionHasherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionHasherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionHasherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionHasherVisitor]: Unknown Input"));
		}
		hashFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		Value *hashResult = context->getBuilder()->CreateCall(hashFunc, ArgsV,"hashBoolean");

		RawValue hashValWrapper;
		hashValWrapper.value = hashResult;
		hashValWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
	vector<Value*> argsV;
	argsV.clear();
	argsV.push_back(hashResult);
	Function* debugInt64 = context->getFunction("printi64");
	TheBuilder->CreateCall(debugInt64, argsV);
#endif
		return hashValWrapper;
	}
	throw runtime_error(string("[ExpressionHasherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionHasherVisitor::visit(expressions::RecordConstruction *e) {

	RawCatalog& catalog = RawCatalog::getInstance();
	Function* const F = context->getGlobalFunction();
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(context->getLLVMContext());
	AllocaInst* argMem = NULL;
	Value* isNull = NULL;

	Function *hashCombine = context->getFunction("combineHashes");
	Value* hashedValue = context->createInt64(0);
	std::vector<Value*> ArgsV;
	//Initializing resulting hashed value
	AllocaInst* mem_hashedValue = context->CreateEntryBlockAlloca(F,
			std::string("hashValue"), int64Type);
	TheBuilder->CreateStore(hashedValue, mem_hashedValue);

	const list<expressions::AttributeConstruction> atts = e->getAtts();
	list<expressions::AttributeConstruction>::const_iterator it;
	for (it = atts.begin(); it != atts.end(); it++)
	{
		expressions::Expression* expr = (*it).getExpression();
		RawValue partialHash = expr->accept(*this);

		ArgsV.clear();
		ArgsV.push_back(hashedValue);
		ArgsV.push_back(partialHash.value);
		hashedValue = TheBuilder->CreateCall(hashCombine, ArgsV,
											"combineHashesRes");
		TheBuilder->CreateStore(hashedValue, mem_hashedValue);
	}

	RawValue hashValWrapper;
	hashValWrapper.value = TheBuilder->CreateLoad(mem_hashedValue);
	hashValWrapper.isNull = context->createFalse();
	return hashValWrapper;
}



























