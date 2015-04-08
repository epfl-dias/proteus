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
#include "expressions/expressions-generator.hpp"


class Join : public BinaryRawOperator {
public:
	Join(expressions::BinaryExpression* predicate, const RawOperator& leftChild,
			const RawOperator& rightChild, char* opLabel, Materializer& mat) :
			BinaryRawOperator(leftChild, rightChild), pred(predicate), htName(
					opLabel), mat(mat) {}
	virtual ~Join() {
		LOG(INFO)<<"Collapsing Join operator";}
	virtual void produce() const;
	virtual void consume(RawContext* const context, const OperatorState& childState);
	Materializer& getMaterializer() {return mat;}
	virtual bool isFiltering() const {return true;}
private:
	char* htName;
	OperatorState* generate(RawOperator* op,  OperatorState* childState);
	expressions::BinaryExpression* pred;
	Materializer& mat;
};
