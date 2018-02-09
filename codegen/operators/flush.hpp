/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2018
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

#ifndef FLUSH_HPP_
#define FLUSH_HPP_

#include "operators/operators.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class Flush: public UnaryRawOperator {
public:
	Flush(	vector<expressions::Expression*> outputExprs,
			RawOperator* const child,
			RawContext* context,
			const char *outPath = "out.json");
	virtual ~Flush() {
		LOG(INFO)<<"Collapsing Flush operator";}
	virtual void produce();
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() const {return getChild()->isFiltering();}

protected:
	RawContext *context;
	size_t      result_cnt_id;

	expressions::Expression *outputExpr;
	
	const char *outPath;
private:
	void generate(RawContext* const context, const OperatorState& childState) const;
};

#endif /* REDUCE_HPP_ */
