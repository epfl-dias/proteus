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

#ifndef OPERATORS_HPP_
#define OPERATORS_HPP_

#include "common/common.hpp"
#include "expressions/expressions.hpp"
#include "plugins/plugins.hpp"
#include "plugins/output/plugins-output.hpp"

//Fwd declaration
class Plugin;
class OperatorState;

class RawOperator {
public:
	RawOperator()	:	parent(NULL)				{}
	virtual ~RawOperator() 							{ LOG(INFO) << "Collapsing operator"; }
	void setParent(RawOperator* parent)				{ this->parent = parent; }
	RawOperator* const getParent() const			{ return parent; }
	//Overloaded operator used in checks for children of Join op. More complex cases may require different handling
	bool operator == (const RawOperator& i) const 	{ /*if(this != &i) LOG(INFO) << "NOT EQUAL OPERATORS"<<this<<" vs "<<&i;*/ return this == &i; }
	virtual void produce() const = 0;
	/**
	 * Consume is not a const method because Nest does need to keep some state info.
	 * RawContext needs to be passed from the consuming to the producing side
	 * to kickstart execution once an HT has been built
	 */
	virtual void consume(RawContext* const context, const OperatorState& childState) = 0;
	/* Used by caching service. Aim is finding whether data to be cached has been filtered
	 * by some of the children operators of the plan */
	virtual bool isFiltering() = 0;

private:
	RawOperator* parent;
};

class UnaryRawOperator : public RawOperator {
public:
	UnaryRawOperator(RawOperator* const child) :
			RawOperator(), child(child) 										{}
	virtual ~UnaryRawOperator() 												{ LOG(INFO) << "Collapsing unary operator"; }

	RawOperator* const getChild() 		const									{ return child; }
private:
	RawOperator* const child;
};

class BinaryRawOperator : public RawOperator {
public:
	BinaryRawOperator(const RawOperator& leftChild,	const RawOperator& rightChild) :
			RawOperator(), leftChild(leftChild), rightChild(rightChild) 						{}
	BinaryRawOperator(const RawOperator& leftChild, const RawOperator& rightChild,
			Plugin* const leftPlugin, Plugin* const rightPlugin) :
			RawOperator(), leftChild(leftChild), rightChild(rightChild)	{}
	virtual ~BinaryRawOperator() 										{ LOG(INFO) << "Collapsing binary operator"; }
	const RawOperator& getLeftChild() const								{ return leftChild; }
	const RawOperator& getRightChild() const							{ return rightChild; }
private:
	const RawOperator& leftChild;
	const RawOperator& rightChild;
};

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

#endif /* OPERATORS_HPP_ */
