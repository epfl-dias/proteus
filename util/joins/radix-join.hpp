#ifndef RADIX_JOIN_STATIC_HPP_
#define RADIX_JOIN_STATIC_HPP_

#include "common/common.hpp"
#include "util/joins/types.h" /* relation_t */
#include "util/joins/prj_params.h"
#include "util/joins/rdtsc.h"

#ifdef DEBUG
//#define DEBUGRADIX
#endif

typedef struct HT	{
	int *bucket;
	int *next;
	uint32_t mask;
	int count;
} HT;

/**
 * RJ: Radix Join.
 *
 * The "Radix Join" implementation denoted as RJ implements
 * the single-threaded original multipass radix cluster join idea
 * by Manegold et al.
 *
 * @param relR  input relation R
 * @param relS  input relation S
 *
 * @return number of result tuples
 */
int64_t
RJStepwise(relation_t * relR, relation_t * relS);

/**
 * Splitting functionality in chunks to be called from generated code
 */
int *
partitionHT(size_t num_tuples, tuple_t *inTuples);

void
bucket_chaining_join_prepare(const tuple_t * const tuplesR, int num_tuples,
		HT * ht);


#endif /* RADIX_JOIN_STATIC_HPP_ */
