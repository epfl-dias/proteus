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

#include "expressions/expressions-flusher.hpp"

RawValue ExpressionFlusherVisitor::visit(expressions::IntConstant *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *flushInt = context->getFunction("flushInt");
	vector<Value*> ArgsV;
	Value *val_int = ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
	ArgsV.push_back(val_int);
	ArgsV.push_back(outputFileLLVM);
	context->getBuilder()->CreateCall(flushInt, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::FloatConstant *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Value* fileName = context->CreateGlobalString(this->outputFile);
	Function *flushDouble = context->getFunction("flushDouble");
	vector<Value*> ArgsV;
	Value *val_double = ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
	ArgsV.push_back(val_double);
	ArgsV.push_back(outputFileLLVM);
	context->getBuilder()->CreateCall(flushDouble, ArgsV);
	return placeholder;

}

RawValue ExpressionFlusherVisitor::visit(expressions::BoolConstant *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *flushBoolean = context->getFunction("floatBoolean");
	vector<Value*> ArgsV;
	Value *val_boolean = ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
	ArgsV.push_back(val_boolean);
	ArgsV.push_back(outputFileLLVM);
	context->getBuilder()->CreateCall(flushBoolean, ArgsV);
	return placeholder;

}

RawValue ExpressionFlusherVisitor::visit(expressions::StringConstant *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Function *flushStringC = context->getFunction("flushStringC");

	string toFlush = e->getVal();
	size_t start = 0;
	size_t end = toFlush.length();
	const char* str = toFlush.c_str();
	Value* strLLVM = context->CreateGlobalString(str);

	vector<Value*> ArgsV;
	ArgsV.push_back(strLLVM);
	ArgsV.push_back(context->createInt64(start));
	ArgsV.push_back(context->createInt64(end));
	ArgsV.push_back(outputFileLLVM);

	context->getBuilder()->CreateCall(flushStringC, ArgsV);
	return placeholder;

}

/**
 * Always flush the full entry in this case!
 * Reason: This visitor does not actually recurse -
 * for any recursion, the ExpressionGeneratorVisitor is used
 */
RawValue ExpressionFlusherVisitor::visit(expressions::InputArgument *e)
{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	RawCatalog& catalog = RawCatalog::getInstance();
	Function* const F  = context->getGlobalFunction();
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(context->getLLVMContext());

	std::vector<Value*> ArgsV;

	const map<RecordAttribute, RawValueMemory>& activeVars =
					currState.getBindings();
	list<RecordAttribute> projections = e->getProjections();
	list<RecordAttribute>::iterator it = projections.begin();

		//Is there a case that I am doing extra work?
	//Example: Am I flushing a value that is nested
	//in some value that I have already hashed?
	for(; it != projections.end(); it++)	{
		if(it->getAttrName() == activeLoop)	{
			map<RecordAttribute, RawValueMemory>::const_iterator itBindings;
			for(itBindings = activeVars.begin(); itBindings != activeVars.end(); itBindings++)
			{
				RecordAttribute currAttr = itBindings->first;
				if (currAttr.getRelationName() == it->getRelationName()
						&& currAttr.getAttrName() == activeLoop)
				{
					//Flush value now
					RawValueMemory mem_activeTuple = itBindings->second;

					Plugin* plugin = catalog.getPlugin(currAttr.getRelationName());
					if(plugin == NULL)	{
						string error_msg = string("[Expression Flusher: ] No plugin provided");
						LOG(ERROR) << error_msg;
						throw runtime_error(error_msg);
					}

					plugin->flushTuple(mem_activeTuple,outputFileLLVM);
				}
			}
		}
	}
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::RecordProjection *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	activeRelation 					= e->getOriginalRelationName();

	/**
	 *  Missing connection apparently ('activeRelation')
	*/
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState, activeRelation);

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
					e->getProjectionName().c_str(),e->getAttribute());
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
		plugin->flushValue(mem_path, e->getExpressionType(),outputFileLLVM);
	}
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::IfThenElse *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	RawCatalog& catalog 			= RawCatalog::getInstance();
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	LLVMContext& llvmContext		= context->getLLVMContext();
	Function *F 					= TheBuilder->GetInsertBlock()->getParent();
	Type* int64Type = Type::getInt64Ty(llvmContext);

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
	e->getIfResult()->accept(*this);
	TheBuilder->CreateBr(MergeBB);

	//else
	TheBuilder->SetInsertPoint(ElseBB);
	e->getElseResult()->accept(*this);

	TheBuilder->CreateBr(MergeBB);

	//cont.
	TheBuilder->SetInsertPoint(MergeBB);
	return placeholder;

}

RawValue ExpressionFlusherVisitor::visit(expressions::EqExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;

}

RawValue ExpressionFlusherVisitor::visit(expressions::NeExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;

}

RawValue ExpressionFlusherVisitor::visit(expressions::GeExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::GtExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::LeExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::LtExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::AndExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::OrExpression *e)	{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);

	RawValue exprResult = e->accept(exprGenerator);
	Function *flushFunc = context->getFunction("flushBoolean");
	vector<Value*> ArgsV;
	ArgsV.push_back(exprResult.value);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}

RawValue ExpressionFlusherVisitor::visit(expressions::AddExpression *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *flushFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("flushInt");
			break;
		case FLOAT:
			instructionLabel = string("flushDouble");
			break;
		case BOOL:
			instructionLabel = string("flushBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionFlusherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionFlusherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionFlusherVisitor]: Unknown Input"));
		}
		flushFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return placeholder;
	}
	throw runtime_error(string("[ExpressionFlusherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionFlusherVisitor::visit(expressions::SubExpression *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *flushFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("flushInt");
			break;
		case FLOAT:
			instructionLabel = string("flushDouble");
			break;
		case BOOL:
			instructionLabel = string("flushBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionFlusherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionFlusherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionFlusherVisitor]: Unknown Input"));
		}
		flushFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return placeholder;

	}
	throw runtime_error(string("[ExpressionFlusherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionFlusherVisitor::visit(expressions::MultExpression *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *flushFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("flushInt");
			break;
		case FLOAT:
			instructionLabel = string("flushDouble");
			break;
		case BOOL:
			instructionLabel = string("flushBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionFlusherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionFlusherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionFlusherVisitor]: Unknown Input"));
		}
		flushFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return placeholder;
	}
	throw runtime_error(string("[ExpressionFlusherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionFlusherVisitor::visit(expressions::DivExpression *e) {
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	IRBuilder<>* const TheBuilder	= context->getBuilder();
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, currState);
	RawValue exprResult = e->accept(exprGenerator);

	const ExpressionType* childType = e->getLeftOperand()->getExpressionType();
	Function *flushFunc = NULL;
	string instructionLabel;

	if (childType->isPrimitive()) {
		typeID id = childType->getTypeID();
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();

		switch (id) {
		case INT:
			instructionLabel = string("flushInt");
			break;
		case FLOAT:
			instructionLabel = string("flushDouble");
			break;
		case BOOL:
			instructionLabel = string("flushBoolean");
			break;
		case STRING:
			LOG(ERROR)<< "[ExpressionFlusherVisitor]: string operations not supported yet";
			throw runtime_error(string("[ExpressionFlusherVisitor]: string operations not supported yet"));
		case BAG:
		case LIST:
		case SET:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		case RECORD:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
			throw runtime_error(string("[ExpressionFlusherVisitor]: invalid expression type"));
		default:
			LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
			throw runtime_error(string("[ExpressionFlusherVisitor]: Unknown Input"));
		}
		flushFunc = context->getFunction(instructionLabel);
		vector<Value*> ArgsV;
		ArgsV.push_back(exprResult.value);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return placeholder;
	}
	throw runtime_error(string("[ExpressionFlusherVisitor]: input of binary expression can only be primitive"));
}

RawValue ExpressionFlusherVisitor::visit(expressions::RecordConstruction *e)
{
	outputFileLLVM = context->CreateGlobalString(this->outputFile);
	RawCatalog& catalog = RawCatalog::getInstance();
	Function* const F = context->getGlobalFunction();
	IRBuilder<>* const TheBuilder = context->getBuilder();
	Type* int64Type = Type::getInt64Ty(context->getLLVMContext());
	AllocaInst* argMem = NULL;
	Value* isNull = NULL;
	char delim = ',';

	Function *flushStr = context->getFunction("flushStringCv2");
	Function *flushFunc = context->getFunction("flushChar");
	vector<Value*> ArgsV;

	const list<expressions::AttributeConstruction> atts = e->getAtts();
	list<expressions::AttributeConstruction>::const_iterator it;
	//Start 'record'
	ArgsV.push_back(context->createInt8('{'));
	ArgsV.push_back(outputFileLLVM);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	for (it = atts.begin(); it != atts.end();)
	{
		//attrName
		Value* val_attr = context->CreateGlobalString(
				it->getBindingName().c_str());
		ArgsV.clear();
		ArgsV.push_back(val_attr);
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushStr, ArgsV);

		//:
		ArgsV.clear();
		ArgsV.push_back(context->createInt8(':'));
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);

		//value
		expressions::Expression* expr = (*it).getExpression();
		RawValue partialFlush = expr->accept(*this);

		//comma, if needed
		it++;
		if (it != atts.end())
		{
			ArgsV.clear();
			ArgsV.push_back(context->createInt8(delim));
			ArgsV.push_back(outputFileLLVM);
			context->getBuilder()->CreateCall(flushFunc, ArgsV);
		}
	}
	ArgsV.clear();
	ArgsV.push_back(context->createInt8('}'));
	ArgsV.push_back(outputFileLLVM);
	context->getBuilder()->CreateCall(flushFunc, ArgsV);
	return placeholder;
}



























