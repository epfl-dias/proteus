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

#ifndef UNNEST_HPP_
#define UNNEST_HPP_

#include "operators/operators.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/path.hpp"

//#define DEBUGUNNEST
/**
 * XXX Paper comment: 'Very few ways of evaluating unnest operator -> lamdaDB only provides a nested-loop variation'
 */
class Unnest : public UnaryRawOperator {
public:
	Unnest(expressions::Expression* pred, Path& path, RawOperator* const child) :
		  UnaryRawOperator(child), path(path), pred(pred)								 		{}
	virtual ~Unnest() 																			{ LOG(INFO)<<"Collapsing Unnest operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context, const OperatorState& childState);
private:
	void generate(RawContext* const context, const OperatorState& childState) const;
	expressions::Expression* pred;
	Path& path;
};

#endif /* UNNEST_HPP_ */
