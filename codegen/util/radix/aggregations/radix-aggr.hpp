#ifndef RADIX_AGG_STATIC_HPP_
#define RADIX_AGG_STATIC_HPP_

#include "common/common.hpp"
#include "util/radix/aggregations/types.h" /* relation_t */
#include "util/radix/aggregations/prj_params.h"

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
RJStepwise(agg::relation_t * relR, agg::relation_t * relS);

/**
 * Splitting functionality in chunks to be called from generated code
 */
int *
partitionHT(size_t num_tuples, agg::tuple_t *inTuples);

void
bucket_chaining_agg_prepare(const agg::tuple_t * const tuplesR, int num_tuples,
		HT * ht);


#endif /* RADIX_AGG_STATIC_HPP_ */
