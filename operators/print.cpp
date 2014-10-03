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
	LOG(INFO) << "[Print:] Printing variable " << arg->getArgName();

	//Load argument of print
	std::map<std::string, AllocaInst*>::const_iterator it;
	it = activeVars.find(arg->getArgName());
	if( it == activeVars.end())	{
			throw runtime_error(string("Unknown variable name: ")+arg->getArgName());
	}
	AllocaInst* argMem = it->second;
	IRBuilder<>* TheBuilder = context->getBuilder();
	BasicBlock* codeSpot = TheBuilder->GetInsertBlock();
	LoadInst* loadResult = TheBuilder->CreateLoad(argMem, codeSpot);

	//Call print
	std::vector<Value*> ArgsV;
	ArgsV.clear();
	ArgsV.push_back(loadResult);
	TheBuilder->CreateCall(this->print, ArgsV,"printi");

	//Trigger parent
	OperatorState *newState = new OperatorState(*this, childState.getBindings());
	getParent()->consume(context, *newState);
}
