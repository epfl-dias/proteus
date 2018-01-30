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

#ifndef PROJECT_HPP_
#define PROJECT_HPP_

#include "operators/operators.hpp"
#include "operators/monoids.hpp"
#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-flusher.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class Project: public UnaryRawOperator {
public:
	Project(vector<expressions::Expression*> outputExprs,
			string             relName,
			RawOperator* const child,
			RawContext* context);
	virtual ~Project() {LOG(INFO)<<"Collapsing Project operator";}
	virtual void produce();
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() const {return getChild()->isFiltering();}

protected:
	RawContext *context;
	size_t      oid_id ;
	string      relName;

	vector<expressions::Expression*> outputExprs;
	
	const char *outPath;
private:
	void generate(RawContext* const context, const OperatorState& childState) const;

	void open (RawPipeline * pip) const;
	void close(RawPipeline * pip) const;
};

#endif /* PROJECT_HPP_ */
