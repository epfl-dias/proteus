#ifndef RADIX_JOIN_STATIC_HPP_
#define RADIX_JOIN_STATIC_HPP_

#include "common/common.hpp"
#include "util/radix/joins/types.h" /* relation_t */
#include "util/radix/joins/prj_params.h"
#include "util/radix/rdtsc.h"

#ifdef DEBUG
//#define DEBUGRADIX
#endif

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
RJStepwise(joins::relation_t * relR, joins::relation_t * relS);

/**
 * Splitting functionality in chunks to be called from generated code
 */
int *
partitionHT(size_t num_tuples, joins::tuple_t *inTuples);

void
bucket_chaining_join_prepare(const joins::tuple_t * const tuplesR, int num_tuples,
		HT * ht);


#endif /* RADIX_JOIN_STATIC_HPP_ */
