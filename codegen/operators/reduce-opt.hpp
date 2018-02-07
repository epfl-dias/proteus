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
			RawContext* context, bool flushResults = false, const char *outPath = "out.json");
	virtual ~Reduce() {
		LOG(INFO)<<"Collapsing Reduce operator";}
	virtual void produce();
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() const {return true;}

	Value * getAccumulator(int index){return context->getStateVar(mem_accumulators[index]);}
protected:
	RawContext* context;

	vector<Monoid> accs;
	vector<expressions::Expression*> outputExprs;
	expressions::Expression* pred;
	vector<size_t> mem_accumulators;
	
	const char *outPath;
	bool flushResults;

	void * flush_fun;

	// RawPipelineGen *                flush_pip;

	virtual void generate_flush();

	virtual size_t resetAccumulator(expressions::Expression* outputExpr, Monoid acc, bool flushDelim = false, bool is_first = false, bool is_last = false) const;

private:
	void generate(RawContext* const context, const OperatorState& childState) const;
	//Used to enable chaining with subsequent operators
	void generateBagUnion(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, Value * cnt_mem) const;
	void generateAppend(expressions::Expression* outputExpr, RawContext* const context,
			const OperatorState& state, AllocaInst *mem_accumulating) const;

	void flushResult();
};
}

#endif /* REDUCE_HPP_ */
