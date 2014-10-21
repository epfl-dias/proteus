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

#include <iostream>
#include <stdexcept>

#include "common/common.hpp"
#include "expressions/expressions.hpp"
#include "plugins/plugins.hpp"
#include "plugins/output/plugins-output.hpp"
#include "operators/operator-state.hpp"

//Fwd declaration
class Plugin;

class RawOperator {
public:
	RawOperator()	:	parent(NULL)				{}
	virtual ~RawOperator() 							{ LOG(INFO) << "Collapsing operator"; }
	void setParent(RawOperator* parent)				{ this->parent = parent; }
	RawOperator* const getParent() const			{ return parent; }
	//Overloaded operator used in checks for children of Join op. More complex cases may require different handling
	bool operator == (const RawOperator& i) const 	{ /*if(this != &i) LOG(INFO) << "NOT EQUAL OPERATORS"<<this<<" vs "<<&i;*/ return this == &i; }
	virtual void produce() const = 0;
	virtual void consume(RawContext* const context, const OperatorState& childState) const = 0;

private:
	RawOperator* parent;
};

class UnaryRawOperator : public RawOperator {
public:
	UnaryRawOperator(RawOperator* const child) :
			RawOperator(), child(child), inputPlugin(NULL) 						{}
	UnaryRawOperator(RawOperator* const child, Plugin* const inputPlugin) :
			RawOperator(), child(child), inputPlugin(inputPlugin) 				{}
	virtual ~UnaryRawOperator() 												{ LOG(INFO) << "Collapsing unary operator"; }
	RawOperator* const getChild() 		const									{ return child; }
	Plugin* 	 const getInputPlugin() const									{ return inputPlugin; }
private:
	RawOperator* const child;
	Plugin* 	 const inputPlugin;
};

class BinaryRawOperator : public RawOperator {
public:
	BinaryRawOperator(const RawOperator& leftChild,	const RawOperator& rightChild) :
			RawOperator(), leftChild(leftChild), rightChild(rightChild),
			leftPlugin(NULL), rightPlugin(NULL) 						{}
	BinaryRawOperator(const RawOperator& leftChild, const RawOperator& rightChild,
			Plugin* const leftPlugin, Plugin* const rightPlugin) :
			RawOperator(), leftChild(leftChild), rightChild(rightChild),
			leftPlugin(leftPlugin), rightPlugin(rightPlugin) 			{}
	virtual ~BinaryRawOperator() 										{ LOG(INFO) << "Collapsing binary operator"; }
	const RawOperator& getLeftChild() const								{ return leftChild; }
	const RawOperator& getRightChild() const							{ return rightChild; }
	Plugin* const getLeftPlugin() const									{ return leftPlugin; }
	Plugin* const getRightPlugin() const								{ return rightPlugin; }
private:
	const RawOperator& leftChild;
	const RawOperator& rightChild;
	Plugin* const leftPlugin;
	Plugin* const rightPlugin;
};

#endif /* OPERATORS_HPP_ */
