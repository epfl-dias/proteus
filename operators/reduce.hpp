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

#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "operators/operators.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"

//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

/**
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR REDUCE OPERATOR
 * ADD 'SERIALIZER' SUPPORT IN ALL CASES, NOT ONLY LIST/BAG AS NOW
 */
class Reduce : public UnaryRawOperator {
public:
	Reduce(Monoid acc, expressions::Expression* outputExpr,
			expressions::Expression* pred, RawOperator* const child,
			RawContext* context);
	virtual ~Reduce()												{ LOG(INFO)<<"Collapsing Reduce operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
private:
	RawContext* context;

	Monoid acc;
	expressions::Expression* outputExpr;
	expressions::Expression* pred;
	AllocaInst* mem_accumulating;

	void generate(RawContext* const context, const OperatorState& childState) const;
	void generateSum(RawContext* const context, const OperatorState& childState) const;
	void generateMul(RawContext* const context, const OperatorState& childState) const;
	void generateMax(RawContext* const context, const OperatorState& childState) const;
	void generateAnd(RawContext* const context, const OperatorState& childState) const;
	void generateOr(RawContext* const context, const OperatorState& childState) const;
	void generateUnion(RawContext* const context, const OperatorState& childState) const;
	void generateBagUnion(RawContext* const context, const OperatorState& childState) const;
	void generateAppend(RawContext* const context, const OperatorState& childState) const;

	void flushResult();
};

#endif /* REDUCE_HPP_ */
