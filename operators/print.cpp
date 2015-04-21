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

void Print::produce() { getChild()->produce(); }

void Print::consume (RawContext* const context, const OperatorState& childState) {

	IRBuilder<>* TheBuilder = context->getBuilder();
	vector<Value*> ArgsV;

	const map<RecordAttribute, RawValueMemory>& activeVars = childState.getBindings();
	LOG(INFO) << "[Print:] Printing variable " << arg->getProjectionName();
//	cout << "[Print:] Printing variable " << arg->getRelationName() << "." << arg->getProjectionName();

	map<RecordAttribute, RawValueMemory>::const_iterator it = activeVars.begin();

#ifdef DEBUG
	for(; it != activeVars.end(); it++)	{
		RecordAttribute attr = it->first;
		cout << "Original relname: " << attr.getOriginalRelationName() << endl;
		cout << "Current relname: " << attr.getRelationName() << endl;
		cout << "Attribute name: " << attr.getAttrName() << endl;
		cout << "---" << endl;
	}
#endif

	//Load argument of print
//	AllocaInst* mem_value = NULL;
//	{
//		string relName = arg->getOriginalRelationName();
//		/*Active Tuple not always of same type
//		  => Deprecated Constructor			*/
//		RecordAttribute attr = RecordAttribute(relName,activeLoop);
//
//		map<RecordAttribute, RawValueMemory>::const_iterator it;
//		it = activeVars.find(attr);
//		if(it == activeVars.end())	{
//			string error_msg = string("[PrintOp: ] Wrong handling of active tuple");
//			LOG(ERROR) << error_msg;
//			throw runtime_error(error_msg);
//		}
//		mem_value = (it->second).mem;
//	}

#ifdef DEBUG
//		Value* value = TheBuilder->CreateLoad(mem_value);
//		ArgsV.push_back(value);
//		Function* debugInt = context->getFunction("printi64");
//		TheBuilder->CreateCall(debugInt, ArgsV);
#endif

	//Generate condition
	ExpressionGeneratorVisitor exprGenerator = ExpressionGeneratorVisitor(context, childState);
	RawValue toPrint = arg->accept(exprGenerator);

	//Call print

	ArgsV.clear();
	ArgsV.push_back(toPrint.value);
	TheBuilder->CreateCall(print, ArgsV);

	//Trigger parent
	OperatorState *newState = new OperatorState(*this, childState.getBindings());
	getParent()->consume(context, *newState);
}
