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

#ifndef EXPRESSIONS_FLUSHER_VISITOR_HPP_
#define EXPRESSIONS_FLUSHER_VISITOR_HPP_

#include "common/common.hpp"
#include "plugins/plugins.hpp"
#include "expressions/expressions-generator.hpp"

#ifdef DEBUG
#define DEBUG_FLUSH
#endif

//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for eagerly evaluating and flushing an Expression
//===---------------------------------------------------------------------------===//
/**
 * TODO Also need the concept of a serializer / output plugin.
 * Atm, this flusher is 'hardcoded' to JSON logic
 */
class ExpressionFlusherVisitor: public ExprVisitor
{
public:
	ExpressionFlusherVisitor(RawContext* const context,
			const OperatorState& currState, const char* outputFile) :
			context(context), currState(currState), outputFile(outputFile),
			activeRelation("")
	{
		//Only used as a token return value that is passed along by each visitor
		placeholder.isNull = context->createTrue();
		placeholder.value = NULL;
		outputFileLLVM = NULL;
	}
	ExpressionFlusherVisitor(RawContext* const context,
			const OperatorState& currState, char* outputFile,
			string activeRelation) :
			context(context), currState(currState), outputFile(outputFile),
			activeRelation(activeRelation)
	{
		placeholder.isNull = context->createTrue();
		placeholder.value = NULL;
		outputFileLLVM = NULL;
	}
	RawValue visit(expressions::IntConstant *e);
	RawValue visit(expressions::Int64Constant *e);
	RawValue visit(expressions::FloatConstant *e);
	RawValue visit(expressions::BoolConstant *e);
	RawValue visit(expressions::StringConstant *e);
	RawValue visit(expressions::InputArgument *e);
	RawValue visit(expressions::RecordProjection *e);
	RawValue visit(expressions::IfThenElse *e);
	//XXX Do binary operators require explicit handling of NULL?
	RawValue visit(expressions::EqExpression *e);
	RawValue visit(expressions::NeExpression *e);
	RawValue visit(expressions::GeExpression *e);
	RawValue visit(expressions::GtExpression *e);
	RawValue visit(expressions::LeExpression *e);
	RawValue visit(expressions::LtExpression *e);
	RawValue visit(expressions::AddExpression *e);
	RawValue visit(expressions::SubExpression *e);
	RawValue visit(expressions::MultExpression *e);
	RawValue visit(expressions::DivExpression *e);
	RawValue visit(expressions::AndExpression *e);
	RawValue visit(expressions::OrExpression *e);
	RawValue visit(expressions::RecordConstruction *e);
	RawValue visit(expressions::RawValueExpression *e);
	RawValue visit(expressions::MinExpression *e);
	RawValue visit(expressions::MaxExpression *e);
	/* Reduce produces accumulated value internally.
	 * It makes no sense to probe a plugin in order to flush this value out */
	void flushValue(Value *val, typeID val_type);

	void setActiveRelation(string relName)		{ activeRelation = relName; }
	string getActiveRelation(string relName)	{ return activeRelation; }

	/**
	 * TODO Push these functions to Serializer
	 * NOTE: Hard-coded to JSON case atm
	 */
	void beginList()
	{	outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushChar");
		vector<Value*> ArgsV;
		//Start 'array'
		ArgsV.push_back(context->createInt8('['));
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}
	void beginBag()
	{
		string error_msg = string(
				"[ExpressionFlusherVisitor]: Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	void beginSet()
	{
		string error_msg = string(
				"[ExpressionFlusherVisitor]: Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	void endList()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushChar");
		vector<Value*> ArgsV;
		//Start 'array'
		ArgsV.push_back(context->createInt8(']'));
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}
	void endBag()
	{
		string error_msg = string(
				"[ExpressionFlusherVisitor]: Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	void endSet()
	{
		string error_msg = string(
				"[ExpressionFlusherVisitor]: Not implemented yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	void flushOutput()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushOutput");
		vector<Value*> ArgsV;
		//Start 'array'
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}
	void flushDelim(Value* resultCtr)
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushDelim");
		vector<Value*> ArgsV;
		ArgsV.push_back(resultCtr);
		//XXX JSON-specific -> Serializer business to differentiate
		ArgsV.push_back(context->createInt8(','));
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}
	void flushDelim() {
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushChar");
		vector<Value*> ArgsV;
		//XXX JSON-specific -> Serializer business to differentiate
		ArgsV.push_back(context->createInt8(','));
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}

private:
	RawContext* const context;
	const OperatorState& currState;
	const char *outputFile;
	Value* outputFileLLVM;

	RawValue placeholder;
	string activeRelation;

};

#endif /* EXPRESSIONS_FLUSHER_VISITOR_HPP_ */
