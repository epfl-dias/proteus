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

#ifndef REDUCE_OPT_HPP_
#define REDUCE_OPT_HPP_

#include "operators/operators.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"

namespace opt {
//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

/* MULTIPLE ACCUMULATORS SUPPORTED */
class Reduce: public UnaryRawOperator {
public:
	Reduce(vector<Monoid> accs, vector<expressions::Expression*> outputExprs,
			expressions::Expression* pred, RawOperator* const child,
			RawContext* context);
	virtual ~Reduce() {
		LOG(INFO)<<"Collapsing Reduce operator";}
	virtual void produce();
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() const {return true;}
private:
	RawContext* context;

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* pred;
	vector<AllocaInst*> mem_accumulators;

	void generate(RawContext* const context, const OperatorState& childState) const;
	void generateSum(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateMul(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateMax(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateOr(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateAnd(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateUnion(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateBagUnion(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;
	void generateAppend(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;

	AllocaInst* resetAccumulator(expressions::Expression* outputExpr, Monoid acc) const;

	void flushResult();
};
}

#endif /* REDUCE_HPP_ */
