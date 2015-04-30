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

#ifndef _MATERIALIZER_EXPR_HPP_
#define _MATERIALIZER_EXPR_HPP_

#include "operators/operators.hpp"
#include "expressions/expressions-generator.hpp"
#include "util/raw-functions.hpp"
#include "util/raw-caching.hpp"

struct matBuf	{
	/* Mem layout:
	 * Consecutive <payload> chunks - payload type defined at runtime
	 */
	AllocaInst *mem_buffer;
	/* Size in bytes */
	AllocaInst *mem_tuplesNo;
	/* Size in bytes */
	AllocaInst *mem_size;
	/* (Current) Offset in bytes */
	AllocaInst *mem_offset;
};

class ExprMaterializer: public UnaryRawOperator {
public:
	ExprMaterializer(expressions::Expression* expr, RawOperator* const child,
			RawContext* const context, char* opLabel);
	virtual ~ExprMaterializer();
	virtual void produce();
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	virtual bool isFiltering() const {return false;}
private:
	void freeArenas() const;
	void updateRelationPointers() const;

	StructType *toMatType;
	struct matBuf opBuffer;
	char *rawBuffer;
	char **ptr_rawBuffer;

	RawContext* const context;
	expressions::Expression* toMat;
	string opLabel;
};

#endif
