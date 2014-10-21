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

#include "operators/print.hpp"

void Print::produce() const { getChild()->produce(); }

void Print::consume (RawContext* const context, const OperatorState& childState) const {

	const std::map<std::string, AllocaInst*>& activeVars = childState.getBindings();
	LOG(INFO) << "[Print:] Printing variable " << arg->getProjectionName();

	//Load argument of print
	AllocaInst* mem_value = NULL;
	{
		std::map<std::string, AllocaInst*>::const_iterator it;
		it = activeVars.find(activeTuple);
		if(it == activeVars.end())	{
			string error_msg = string("[PrintOp: ] Wrong handling of active tuple");
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}
		mem_value = it->second;
	}
	//Generate condition
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState, getInputPlugin());
	Value* toPrint = arg->accept(exprGenerator);

	//Call print
	IRBuilder<>* TheBuilder = context->getBuilder();
	std::vector<Value*> ArgsV;
	ArgsV.push_back(toPrint);
	TheBuilder->CreateCall(print, ArgsV,"printi");

	//Trigger parent
	OperatorState *newState = new OperatorState(*this, childState.getBindings());
	getParent()->consume(context, *newState);
}
