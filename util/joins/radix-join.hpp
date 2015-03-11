#ifndef RADIX_JOIN_HPP_
#define RADIX_JOIN_HPP_

#include "common/common.hpp"
#include "types.h" /* relation_t */
#include "prj_params.h"
#include "rdtsc.h"

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
#endif /* RADIX_JOIN_HPP_ */
