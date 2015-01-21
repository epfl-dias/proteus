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

#ifndef _NEST_HPP_
#define _NEST_HPP_

#include "operators/operators.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/path.hpp"

/**
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR REDUCE OPERATOR
 */
class Nest : public UnaryRawOperator {
public:
	Nest(expressions::Expression* pred, expressions::Expression* f_grouping,
			expressions::Expression* g_nullToZero, RawOperator* const child) :
			UnaryRawOperator(child), f_grouping(f_grouping),
			g_nullToZero(g_nullToZero), pred(pred)								 			{}
	virtual ~Nest() 																		{ LOG(INFO)<<"Collapsing Nest operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context, const OperatorState& childState) const;
private:
	void generate(RawContext* const context, const OperatorState& childState) const;
	expressions::Expression* pred;
	expressions::Expression* f_grouping;
	expressions::Expression* g_nullToZero;
};

#endif /* UNNEST_HPP_ */
