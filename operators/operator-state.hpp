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

#ifndef OPERATOR_STATE_HPP_
#define OPERATOR_STATE_HPP_

#include "operators/operators.hpp"
#include "values/expressionTypes.hpp"

//Forward declaration
class RawOperator;

/**
 * Wrappers for LLVM Value and Alloca.
 * Maintain information such as whether the corresponding value is 'NULL'
 * LLVM's interpretation of 'NULL' for primitive types is not sufficient
 * (e.g., lvvm_null(int) = 0
 */
typedef struct RawValueMemory {
	AllocaInst* mem;
	Value* isNull;
} RawValueMemory;

typedef struct RawValue {
	Value* value;
	Value* isNull;
} RawValue;



class OperatorState {
public:
	OperatorState(const RawOperator& producer,
			const map<RecordAttribute, RawValueMemory>& vars) :
			producer(producer), activeVariables(vars)			{}
	OperatorState(const OperatorState &opState) :
			producer(opState.producer),
			activeVariables(opState.activeVariables)			{ LOG(INFO)<< "[Operator State: ] Copy Constructor"; }

	const map<RecordAttribute, RawValueMemory>& getBindings() 	const 	{ return activeVariables; }
	const RawOperator& getProducer() 							const 	{ return producer; }
private:
	const RawOperator& producer;
	//Variable bindings produced by operator and provided to its parent
	//const map<string, AllocaInst*>& activeVariables;
	const map<RecordAttribute, RawValueMemory>& activeVariables;
};
#endif /* OPERATOR_STATE_HPP_ */
