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

#include "expressions/expressions-dot-evaluator.hpp"
#include "expressions/expressions-generator.hpp"

RawValue ExpressionDotVisitor::visit(expressions::IntConstant *e1,
		expressions::IntConstant *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_int1 = ConstantInt::get(context->getLLVMContext(),
			APInt(32, e1->getVal()));
	Value *val_int2 = ConstantInt::get(context->getLLVMContext(),
			APInt(32, e2->getVal()));

	Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::DStringConstant *e1,
		expressions::DStringConstant *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_int1 = ConstantInt::get(context->getLLVMContext(),
			APInt(32, e1->getVal()));
	Value *val_int2 = ConstantInt::get(context->getLLVMContext(),
			APInt(32, e2->getVal()));

	Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::Int64Constant *e1,
		expressions::Int64Constant *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_int1 = ConstantInt::get(context->getLLVMContext(),
			APInt(64, e1->getVal()));
	Value *val_int2 = ConstantInt::get(context->getLLVMContext(),
			APInt(64, e2->getVal()));

	Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::DateConstant *e1,
		expressions::DateConstant *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_int1 = ConstantInt::get(context->getLLVMContext(),
			APInt(64, e1->getVal()));
	Value *val_int2 = ConstantInt::get(context->getLLVMContext(),
			APInt(64, e2->getVal()));

	Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::FloatConstant *e1,
		expressions::FloatConstant *e2)	{
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_double1 = ConstantFP::get(context->getLLVMContext(),
			APFloat(e1->getVal()));
	Value *val_double2 = ConstantFP::get(context->getLLVMContext(),
			APFloat(e2->getVal()));

	Value *val_result = Builder->CreateFCmpOEQ(val_double1, val_double2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;

}

RawValue ExpressionDotVisitor::visit(expressions::BoolConstant *e1,
		expressions::BoolConstant *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_int1 = ConstantInt::get(context->getLLVMContext(),
			APInt(8, e1->getVal()));
	Value *val_int2 = ConstantInt::get(context->getLLVMContext(),
			APInt(8, e2->getVal()));

	Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
	RawValue valWrapper;
	valWrapper.value = val_result;
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::StringConstant *e1,
		expressions::StringConstant *e2) {

	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	vector<Value*> ArgsV;
	ArgsV.push_back(left.value);
	ArgsV.push_back(right.value);
	Function* stringEquality = context->getFunction("equalStringObjs");
	valWrapper.value = Builder->CreateCall(stringEquality, ArgsV,
			"equalStringObjsCall");

	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::EqExpression *e1,
		expressions::EqExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::NeExpression *e1,
		expressions::NeExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::GeExpression *e1,
		expressions::GeExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::GtExpression *e1,
		expressions::GtExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::LeExpression *e1,
		expressions::LeExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::LtExpression *e1,
		expressions::LtExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::AndExpression *e1,
		expressions::AndExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::OrExpression *e1,
		expressions::OrExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}


RawValue ExpressionDotVisitor::visit(expressions::AddExpression *e1,
		expressions::AddExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::SubExpression *e1,
		expressions::SubExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::MultExpression *e1,
		expressions::MultExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::DivExpression *e1,
		expressions::DivExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}


//XXX Careful here
RawValue ExpressionDotVisitor::visit(expressions::RecordConstruction *e1,
		expressions::RecordConstruction *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	Value *val_true = context->createTrue();
	Value *val_result = val_true;


	const list<expressions::AttributeConstruction>& atts1 = e1->getAtts();
	const list<expressions::AttributeConstruction>& atts2 = e2->getAtts();
	list<expressions::AttributeConstruction>::const_iterator it1 = atts1.begin();
	list<expressions::AttributeConstruction>::const_iterator it2 = atts2.begin();

	for(; it1 != atts1.end() && it2 != atts2.end(); it1++ , it2++ )	{
		expressions::Expression* expr1 = it1->getExpression();
		expressions::Expression* expr2 = it2->getExpression();

		RawValue val_partialWrapper = expr1->acceptTandem(*this,expr2);
		val_result = Builder->CreateAnd(val_result,val_partialWrapper.value);
	}
	RawValue val_resultWrapper;
	val_resultWrapper.value = val_result;
	val_resultWrapper.isNull = context->createFalse();
	return val_resultWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::HashExpression *e1,
		expressions::HashExpression *e2)	{
	assert(false && "This does not really make sense");
	return e1->getExpr()->acceptTandem(*this, e2->getExpr());
}


RawValue ExpressionDotVisitor::visit(expressions::InputArgument *e1,
		expressions::InputArgument *e2) {
	/* What would be needed is a per-pg 'dotCmp' method
	 * -> compare piece by piece at this level, don't reconstruct here*/
	string error_msg = string(
			"[Expression Dot: ] No explicit InputArg support yet");
	LOG(ERROR)<< error_msg;
	throw runtime_error(error_msg);
}

RawValue ExpressionDotVisitor::visit(expressions::RawValueExpression *e1,
		expressions::RawValueExpression *e2) {
	// left or right should not matter for RawValueExpressions, as 
	// the bindings will not be used
	ExpressionGeneratorVisitor visitor{context, currStateLeft};
	return expressions::EqExpression{e1, e2}.accept(visitor);
}

/* Probably insufficient for complex datatypes */
RawValue ExpressionDotVisitor::visit(expressions::RecordProjection *e1,
		expressions::RecordProjection *e2) {
	IRBuilder<>* const Builder = context->getBuilder();

	typeID id = e1->getExpressionType()->getTypeID();
	bool primitive =  id == INT || id == FLOAT || id == BOOL || id == INT64 || id == STRING;
	if (primitive) {
		const OperatorState& currState1 = currStateLeft;
		ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
				context, currState1);
		RawValue left = e1->accept(exprGenerator1);

		const OperatorState& currState2 = currStateRight;
		ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
				context, currState2);
		RawValue right = e2->accept(exprGenerator2);

		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();
		switch (id) {
		case INT:
//#ifdef DEBUG
//		{
//			/* Printing the pos. to be marked */
//			if(e1->getProjectionName() == "age") {
//				cout << "AGE! " << endl;
//				Function* debugInt = context->getFunction("printi");
//				vector<Value*> ArgsV;
//				ArgsV.clear();
//				ArgsV.push_back(left.value);
//				Builder->CreateCall(debugInt, ArgsV);
//			}
//			else
//			{
//				cout << "Other projection - " << e1->getProjectionName() << endl;
//			}
//		}
//#endif
			valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
			return valWrapper;
		case FLOAT:
			valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
			return valWrapper;
		case BOOL:
			valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
			return valWrapper;
		case STRING:
		{
			vector<Value*> ArgsV;
			ArgsV.push_back(left.value);
			ArgsV.push_back(right.value);
			Function* stringEquality = context->getFunction("equalStringObjs");
			valWrapper.value = Builder->CreateCall(stringEquality, ArgsV,
					"equalStringObjsCall");
			return valWrapper;
		}
		default:
			LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
			throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
		}
		return valWrapper;
	}
	else
	{
		/* XXX
		 * Stick to returning hash of result for now
		 * Obviously can cause false positives
		 *
		 * What would be needed is a per-pg 'dotCmp' method
		 * -> compare piece by piece at this level, don't reconstruct here
		 */
		const OperatorState& currState1 = currStateLeft;
		ExpressionHasherVisitor aggrExprGenerator1 = ExpressionHasherVisitor(context, currState1);
		RawValue hashLeft = e1->accept(aggrExprGenerator1);

		const OperatorState& currState2 = currStateRight;
		ExpressionHasherVisitor aggrExprGenerator2 = ExpressionHasherVisitor(
				context, currState2);
		RawValue hashRight = e2->accept(aggrExprGenerator2);
		RawValue valWrapper;
		valWrapper.isNull = context->createFalse();
		valWrapper.value = Builder->CreateICmpEQ(hashLeft.value, hashRight.value);
		return valWrapper;
	}
}

RawValue ExpressionDotVisitor::visit(expressions::IfThenElse *e1,
		expressions::IfThenElse *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();
	switch (id) {
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case STRING: {
		vector<Value*> ArgsV;
		ArgsV.push_back(left.value);
		ArgsV.push_back(right.value);
		Function* stringEquality = context->getFunction("equalStringObjs");
		valWrapper.value = Builder->CreateCall(stringEquality, ArgsV,
				"equalStringObjsCall");
		return valWrapper;
	}
	default: {
		string error_msg = string(
				"[Expression Dot: ] No explicit non-primitive support yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	}
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::MaxExpression *e1,
		expressions::MaxExpression *e2) {
	return e1->getCond()->acceptTandem(*this, e2->getCond());
}

RawValue ExpressionDotVisitor::visit(expressions::MinExpression *e1,
		expressions::MinExpression *e2) {
	return e1->getCond()->acceptTandem(*this, e2->getCond());
}

RawValue ExpressionDotVisitor::visit(expressions::NegExpression *e1,
		expressions::NegExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1 = ExpressionGeneratorVisitor(
			context, currState1);
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2 = ExpressionGeneratorVisitor(
			context, currState2);
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case DSTRING:
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::ExtractExpression *e1,
		expressions::ExtractExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1{context, currState1};
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2{context, currState2};
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.isNull = context->createFalse();

	switch (id) {
	case DSTRING:
	case INT:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	case FLOAT:
		valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
		return valWrapper;
	case BOOL:
		valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
		return valWrapper;
	default:
		LOG(ERROR)<< "[ExpressionDotVisitor]: Invalid Input";
		throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
	}

	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::TestNullExpression *e1,
		expressions::TestNullExpression *e2) {
	IRBuilder<>* const Builder = context->getBuilder();
	const OperatorState& currState1 = currStateLeft;
	ExpressionGeneratorVisitor exprGenerator1{context, currState1};
	RawValue left = e1->accept(exprGenerator1);

	const OperatorState& currState2 = currStateRight;
	ExpressionGeneratorVisitor exprGenerator2{context, currState2};
	RawValue right = e2->accept(exprGenerator2);

	typeID id = e1->getExpressionType()->getTypeID();
	RawValue valWrapper;
	valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
	valWrapper.isNull = context->createFalse();
	return valWrapper;
}

RawValue ExpressionDotVisitor::visit(expressions::CastExpression *e1,
		expressions::CastExpression *e2) {
	return e1->getExpr()->acceptTandem(*this, e2->getExpr());
}
