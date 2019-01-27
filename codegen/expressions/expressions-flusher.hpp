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
#include "plugins/json-plugin.hpp"
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
	//TODO: should we remove this constructor ?
	ExpressionFlusherVisitor(RawContext* const context,
			const OperatorState& currState, const char* outputFile) :
			context(context), currState(currState), outputFile(outputFile),
			activeRelation(""), pg(NULL)
	{
		//Only used as a token return value that is passed along by each visitor
		placeholder.isNull = context->createTrue();
		placeholder.value = NULL;
		outputFileLLVM = NULL;
		pg = new jsonPipelined::JSONPlugin(context, outputFile, NULL);
	}
	ExpressionFlusherVisitor(RawContext* const context,
			const OperatorState& currState, const char* outputFile,
			string activeRelation) :
			context(context), currState(currState), outputFile(outputFile),
			activeRelation(activeRelation)
	{
		placeholder.isNull = context->createTrue();
		placeholder.value = NULL;
		outputFileLLVM = NULL;
		pg = RawCatalog::getInstance().getPlugin(activeRelation);
	}
	RawValue visit(const expressions::IntConstant *e);
	RawValue visit(const expressions::Int64Constant *e);
	RawValue visit(const expressions::DateConstant *e);
	RawValue visit(const expressions::FloatConstant *e);
	RawValue visit(const expressions::BoolConstant *e);
	RawValue visit(const expressions::StringConstant *e);
	RawValue visit(const expressions::DStringConstant *e);
	RawValue visit(const expressions::InputArgument *e);
	RawValue visit(const expressions::RecordProjection *e);
	RawValue visit(const expressions::IfThenElse *e);
	//XXX Do binary operators require explicit handling of NULL?
	RawValue visit(const expressions::EqExpression *e);
	RawValue visit(const expressions::NeExpression *e);
	RawValue visit(const expressions::GeExpression *e);
	RawValue visit(const expressions::GtExpression *e);
	RawValue visit(const expressions::LeExpression *e);
	RawValue visit(const expressions::LtExpression *e);
	RawValue visit(const expressions::AddExpression *e);
	RawValue visit(const expressions::SubExpression *e);
	RawValue visit(const expressions::MultExpression *e);
	RawValue visit(const expressions::DivExpression *e);
	RawValue visit(const expressions::AndExpression *e);
	RawValue visit(const expressions::OrExpression *e);
	RawValue visit(const expressions::RecordConstruction *e);
	RawValue visit(const expressions::RawValueExpression *e);
	RawValue visit(const expressions::MinExpression *e);
	RawValue visit(const expressions::MaxExpression *e);
	RawValue visit(const expressions::HashExpression *e);
	RawValue visit(const expressions::NegExpression *e);
	RawValue visit(const expressions::ExtractExpression *e);
	RawValue visit(const expressions::TestNullExpression *e);
	RawValue visit(const expressions::CastExpression *e);
	/* Reduce produces accumulated value internally.
	 * It makes no sense to probe a plugin in order to flush this value out */
	void flushValue(llvm::Value *val, typeID val_type);

	void setActiveRelation(string relName)		{ 
		activeRelation = relName; 
		if (relName != ""){
			pg = RawCatalog::getInstance().getPlugin(activeRelation);
		} else {
			pg = RawCatalog::getInstance().getPlugin(outputFile);
		}
	}
	string getActiveRelation(string relName)	{ return activeRelation; }

	/**
	 * TODO Push these functions to Serializer
	 * NOTE: Hard-coded to JSON case atm
	 */
	void beginList()
	{	outputFileLLVM = context->CreateGlobalString(this->outputFile);
		if (!pg){
			//TODO: remove. deprecated exectuion path
			Function *flushFunc = context->getFunction("flushChar");
			vector<llvm::Value*> ArgsV;
			//Start 'array'
			ArgsV.push_back(context->createInt8('['));
			ArgsV.push_back(outputFileLLVM);
			context->getBuilder()->CreateCall(flushFunc, ArgsV);
		} else {
			pg->flushBeginList(outputFileLLVM);
		}
	}
	void beginBag()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		pg->flushBeginBag(outputFileLLVM);
	}
	void beginSet()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		pg->flushBeginSet(outputFileLLVM);
	}
	void endList()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		if (!pg){
			//TODO: remove. deprecated exectuion path
			Function *flushFunc = context->getFunction("flushChar");
			vector<llvm::Value*> ArgsV;
			//Start 'array'
			ArgsV.push_back(context->createInt8(']'));
			ArgsV.push_back(outputFileLLVM);
			context->getBuilder()->CreateCall(flushFunc, ArgsV);
		} else {
			pg->flushEndList(outputFileLLVM);
		}
	}
	void endBag()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		pg->flushEndBag(outputFileLLVM);
	}
	void endSet()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		pg->flushEndSet(outputFileLLVM);
	}
	void flushOutput()
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		Function *flushFunc = context->getFunction("flushOutput");
		vector<llvm::Value*> ArgsV;
		//Start 'array'
		ArgsV.push_back(outputFileLLVM);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}
	void flushDelim(llvm::Value* resultCtr, int depth = 0)
	{
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		if (!pg){
			//TODO: remove. deprecated exectuion path
			Function *flushFunc = context->getFunction("flushDelim");
			vector<llvm::Value*> ArgsV;
			ArgsV.push_back(resultCtr);
			//XXX JSON-specific -> Serializer business to differentiate
			ArgsV.push_back(context->createInt8(','));
			ArgsV.push_back(outputFileLLVM);
			context->getBuilder()->CreateCall(flushFunc, ArgsV);
		} else {
			pg->flushDelim(resultCtr, outputFileLLVM, depth);
		}
	}
	void flushDelim(int depth = 0) {
		outputFileLLVM = context->CreateGlobalString(this->outputFile);
		if (!pg){
			//TODO: remove. deprecated exectuion path
			Function *flushFunc = context->getFunction("flushChar");
			vector<llvm::Value*> ArgsV;
			//XXX JSON-specific -> Serializer business to differentiate
			ArgsV.push_back(context->createInt8(','));
			ArgsV.push_back(outputFileLLVM);
			context->getBuilder()->CreateCall(flushFunc, ArgsV);
		} else {
			pg->flushDelim(outputFileLLVM, depth);
		}
	}

private:
	RawContext* const context;
	const OperatorState& currState;
	const char *outputFile;
	llvm::Value* outputFileLLVM;

	RawValue placeholder;
	string activeRelation;
	Plugin * pg;
};

#endif /* EXPRESSIONS_FLUSHER_VISITOR_HPP_ */
