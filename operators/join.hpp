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

#include "operators/operators.hpp"

class Join : public BinaryRawOperator {
public:
	Join(expressions::BinaryExpression* predicate, const RawOperator& leftChild, const RawOperator& rightChild, char* htName, const Materializer& mat)
		: BinaryRawOperator(leftChild, rightChild), pred(predicate), htName(htName), mat(mat)	 		{}
	virtual ~Join() 																					{ LOG(INFO)<<"Collapsing Join operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context, const OperatorState& childState) const;
	const Materializer& getMaterializer() const															{ return mat; }
private:
	char* htName;
	std::vector<string> leftFields;
	OperatorState* generate(RawOperator* op,  OperatorState* childState);
	expressions::BinaryExpression* pred;
	const Materializer& mat;
};
