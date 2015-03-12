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

#include "operators/operators.hpp"
#include "expressions/expressions-generator.hpp"
#include "util/joins/radix-join.hpp"


typedef struct htEntry	{
	void *keyPtr;
	void *valuePtr;
} htEntry;

struct relationBuf	{
	/* Mem layout:
	 * Pairs of (size_t key, arbitrary - defined at runtime - payload)
	 */
	AllocaInst *mem_relation;
	/* Size in bytes */
	AllocaInst *mem_tuplesNo;
	/* Size in bytes */
	AllocaInst *mem_size;
	/* (Current) Offset in bytes */
	AllocaInst *mem_offset;
};

struct kvBuf	{
	/* Mem layout:
	 * Pairs of (void *keyPtr, void *payloadPtr)
	 */
	AllocaInst *mem_kv;
	/* Size in bytes */
	AllocaInst *mem_tuplesNo;
	/* Size in bytes */
	AllocaInst *mem_size;
	/* (Current) Offset in bytes */
	AllocaInst *mem_offset;
};



class RadixJoin: public BinaryRawOperator {
public:
	RadixJoin(expressions::BinaryExpression* predicate,
			const RawOperator& leftChild, const RawOperator& rightChild,
			RawContext* const context, char* opLabel, Materializer& matLeft,
			Materializer& matRight);
	virtual ~RadixJoin() ;
	virtual void produce() const;
	virtual void consume(RawContext* const context,
			const OperatorState& childState);
	Materializer& getMaterializerLeft() {return matLeft;}
	Materializer& getMaterializerRight() {return matRight;}
private:
	OperatorState* generate(RawOperator* op, OperatorState* childState);

	void freeArenas() const;
	struct relationBuf relR;
	struct relationBuf relS;

	struct kvBuf htR;
	struct kvBuf htS;

	HT *HT_per_cluster;

	char *relationR;
	char *relationS;
	char *kvR;
	char *kvS;

	string htLabel;
	RawContext* const context;
	expressions::BinaryExpression* pred;
	Materializer& matLeft;
	Materializer& matRight;
};
