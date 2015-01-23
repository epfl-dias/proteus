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
 * Indicative query where a nest (..and an outer join) occur:
 * for (d <- Departments) yield set (D := d, E := for ( e <- Employees, e.dno = d.dno) yield set e)
 *
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR NEST OPERATOR (?)
 */
class Nest : public UnaryRawOperator {
public:
	Nest(Monoid acc, expressions::Expression* outputExpr,
		 expressions::Expression* pred, expressions::Expression* f_grouping,
		 expressions::Expression* g_nullToZero, RawOperator* const child,
		 char* opLabel, Materializer& mat) :
			UnaryRawOperator(child), acc(acc), outputExpr(outputExpr),
			f_grouping(f_grouping),	g_nullToZero(g_nullToZero),
			pred(pred), mat(mat), htName(opLabel), context(NULL), childState(NULL)			{}
	virtual ~Nest() 																		{ LOG(INFO)<<"Collapsing Nest operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context, const OperatorState& childState) const;
	Materializer& getMaterializer()		 													{ return mat; }
private:
	void generateInsert(RawContext* const context, const OperatorState& childState) const;
	void generateProbe(RawContext* const context, const OperatorState& childState) const;
	Monoid acc;
	expressions::Expression* outputExpr;
	expressions::Expression* pred;
	expressions::Expression* f_grouping;
	expressions::Expression* g_nullToZero;

	char* htName;
	Materializer& mat;

	RawContext* context;
	OperatorState* childState;
};

#endif /* UNNEST_HPP_ */
