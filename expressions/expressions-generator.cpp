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

Value* ExpressionGeneratorVisitor::visit(expressions::IntConstant *e) {
	return ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
}

Value* ExpressionGeneratorVisitor::visit(expressions::FloatConstant *e) {
	return ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
}

Value* ExpressionGeneratorVisitor::visit(expressions::BoolConstant *e) {
	return ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
}

Value* ExpressionGeneratorVisitor::visit(expressions::StringConstant *e) {
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
	return val_strObj;
}

Value* ExpressionGeneratorVisitor::visit(expressions::InputArgument *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	AllocaInst* argMem = NULL;
	{
		const map<RecordAttribute, AllocaInst*>& activeVars = currState.getBindings();
		map<RecordAttribute, AllocaInst*>::const_iterator it;

		//A previous visitor has indicated which relation is relevant
		if(activeRelation != "")	{
			RecordAttribute relevantAttr = RecordAttribute(activeRelation,activeLoop);
			it = activeVars.find(relevantAttr);
			if (it == activeVars.end()) {
				string error_msg = string("[Expression Generator: ] Could not find tuple information for ") + activeRelation;
				LOG(ERROR) << error_msg;
			 	throw runtime_error(error_msg);
			}	else	{
				argMem = it->second;
			}
		}	else	{
			LOG(INFO) << "[Expression Generator: ] No active relation found - Non-record case";
			int relationsCount = 0;
			for(it = activeVars.begin(); it != activeVars.end(); it++)	{
				RecordAttribute currAttr = it->first;
				cout << currAttr.getRelationName() <<" and "<< currAttr.getAttrName() << endl;

				//Does 1st part of check ever get satisfied? activeRelation is empty here
				if(currAttr.getRelationName() == activeRelation && currAttr.getAttrName() == activeLoop)	{
					//cout << "Found " << currAttr.getRelationName() << " " << currAttr.getAttrName() << endl;
					argMem = it->second;
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
		}
	}
	return TheBuilder->CreateLoad(argMem);
}

Value* ExpressionGeneratorVisitor::visit(expressions::RecordProjection *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	activeRelation 					= e->getOriginalRelationName();
	Value* record 					= e->getExpr()->accept(*this);
	Plugin* plugin 					= catalog.getPlugin(activeRelation);

	//Resetting activeRelation here would break nested-record-projections
	//activeRelation = "";
	if(plugin == NULL)	{
		string error_msg = string("[Expression Generator: ] No plugin provided");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}	else	{
		Bindings bindings = { &currState, record };
		AllocaInst *mem_path = NULL;
		AllocaInst *mem_val = NULL;

		if (e->getProjectionName() != activeLoop) {
			//Path involves a projection / an object
			mem_path = plugin->readPath(activeRelation, bindings,
					e->getProjectionName().c_str());

		} else {
			//Path involves a primitive datatype
			//(e.g., the result of unnesting a list of primitives)
			RecordAttribute tupleIdentifier = RecordAttribute(activeRelation,
					activeLoop);
			map<RecordAttribute, AllocaInst*>::const_iterator it =
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
		Value *val = TheBuilder->CreateLoad(mem_val);

		return val;
	}
}

Value* ExpressionGeneratorVisitor::visit(expressions::IfThenElse *e) {
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	LLVMContext& llvmContext		= context->getLLVMContext();
	Function *F 					= TheBuilder->GetInsertBlock()->getParent();

	Value* ifCond 		= e->getIfCond()->accept(*this);
	Value* ifResult		= e->getIfResult()->accept(*this);
	Value* elseResult 	= e->getElseResult()->accept(*this);

	//Prepare result
	AllocaInst* mem_result = context->CreateEntryBlockAlloca(F, "ifElseResult", ifResult->getType());

	//Prepare blocks
	BasicBlock *ThenBB;
	BasicBlock *ElseBB;
	BasicBlock *MergeBB = BasicBlock::Create(llvmContext,  "ifExprCont", F);
	context->CreateIfElseBlocks(F,"ifExprThen","ifExprElse",&ThenBB,&ElseBB,MergeBB);

	//if
	TheBuilder->CreateCondBr(ifCond, ThenBB, ElseBB);

	//then
	TheBuilder->SetInsertPoint(ThenBB);
	TheBuilder->CreateStore(ifResult,mem_result);
	TheBuilder->CreateBr(MergeBB);

	//else
	TheBuilder->SetInsertPoint(ElseBB);
	TheBuilder->CreateStore(elseResult,mem_result);
	TheBuilder->CreateBr(MergeBB);

	//cont.
	TheBuilder->SetInsertPoint(MergeBB);
	Value* result = TheBuilder->CreateLoad(mem_result);
	return result;
}

Value* ExpressionGeneratorVisitor::visit(expressions::EqExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpEQ(left, right);
		case FLOAT:
			return TheBuilder->CreateFCmpOEQ(left, right);
		case BOOL:
			return TheBuilder->CreateICmpEQ(left, right);
		case STRING: {
			std::vector<Value*> ArgsV;
			ArgsV.push_back(left);
			ArgsV.push_back(right);
			Function* stringEquality = context->getFunction("equalStrings");
			return TheBuilder->CreateCall(stringEquality, ArgsV,
					"equalStringsCall");
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

Value* ExpressionGeneratorVisitor::visit(expressions::NeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpNE(left, right);
		case FLOAT:
			return TheBuilder->CreateFCmpONE(left, right);
		case BOOL:
			return TheBuilder->CreateICmpNE(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::GeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpSGE(left, right);
		case FLOAT:
			return TheBuilder->CreateFCmpOGE(left, right);
		case BOOL:
			return TheBuilder->CreateICmpSGE(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::GtExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpSGT(left, right);
		case FLOAT:
		{
#ifdef DEBUG
//		std::vector<Value*> ArgsV;
//		Function* debugFloat = context->getFunction("printFloat");
//		ArgsV.push_back(left);
//		TheBuilder->CreateCall(debugFloat, ArgsV);
#endif
			return TheBuilder->CreateFCmpOGT(left, right);
		}
		case BOOL:
			return TheBuilder->CreateICmpSGT(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::LeExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpSLE(left, right);
		case FLOAT:
			return TheBuilder->CreateFCmpOLE(left, right);
		case BOOL:
			return TheBuilder->CreateICmpSLE(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::LtExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateICmpSLT(left, right);
		case FLOAT:
			return TheBuilder->CreateFCmpOLT(left, right);
		case BOOL:
			return TheBuilder->CreateICmpSLT(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::AddExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateAdd(left, right);
		case FLOAT:
			return TheBuilder->CreateFAdd(left, right);
		case BOOL:
			return TheBuilder->CreateAdd(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::SubExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateSub(left, right);
		case FLOAT:
			return TheBuilder->CreateFSub(left, right);
		case BOOL:
			return TheBuilder->CreateSub(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::MultExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateMul(left, right);
		case FLOAT:
			return TheBuilder->CreateFMul(left, right);
		case BOOL:
			return TheBuilder->CreateMul(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::DivExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);

	ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();

		switch (id) {
		case INT:
			return TheBuilder->CreateSDiv(left, right);
		case FLOAT:
			return TheBuilder->CreateFDiv(left, right);
		case BOOL:
			return TheBuilder->CreateSDiv(left, right);
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

Value* ExpressionGeneratorVisitor::visit(expressions::AndExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);
	return TheBuilder->CreateAnd(left, right);
}

Value* ExpressionGeneratorVisitor::visit(expressions::OrExpression *e) {
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* left = e->getLeftOperand()->accept(*this);
	Value* right = e->getRightOperand()->accept(*this);
	return TheBuilder->CreateOr(left, right);
}


