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

#ifndef REDUCENOPRED_HPP_
#define REDUCENOPRED_HPP_

#include "operators/operators.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"

//#ifdef DEBUG
#define DEBUGREDUCENOPRED
//#endif

/**
 * In many cases, the plan that is produced includes
 * a Reduce Operator with predicate = true.
 *
 * This simplified operator implementation does not perform
 * whether p is true
 */
class ReduceNoPred : public UnaryRawOperator {
public:
	ReduceNoPred(Monoid acc, expressions::Expression* outputExpr,
			RawOperator* const child, RawContext* context);
	virtual ~ReduceNoPred()												{ LOG(INFO)<<"Collapsing Reduce operator"; }
	virtual void produce() const;
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() {return getChild()->isFiltering();}
private:
	RawContext* context;

	Monoid acc;
	expressions::Expression* outputExpr;
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

#endif /* REDUCENOPRED_HPP_ */
