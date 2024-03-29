/**
 * @file    prj_params.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue May 22 14:03:52 2012
 * @version $Id: prj_params.h 3017 2012-12-07 10:56:20Z bcagri $
 *
 * @brief  Constant parameters used by Parallel Radix Join implementations.
 *
 */

#ifndef PRJ_PARAMS_AGG_H
#define PRJ_PARAMS_AGG_H

#include "platform/util/radix/prj_params.h"

/** number of tuples fitting into L1 */
#define L1_CACHE_TUPLES_AGG (L1_CACHE_SIZE / sizeof(agg::tuple_t))

/** \internal some padding space is allocated for relations in order to
 *  avoid L1 conflict misses and PADDING_TUPLES is placed between
 *  partitions in pass-1 of partitioning and SMALL_PADDING_TUPLES is placed
 *  between partitions in pass-2 of partitioning. 3 is a magic number.
 */

/**
 * Put an odd number of cache lines between partitions in pass-2:
 * Here we put 3 cache lines.
 */
#define SMALL_PADDING_TUPLES_AGG (3 * CACHE_LINE_SIZE / sizeof(agg::tuple_t))
#define PADDING_TUPLES_AGG (SMALL_PADDING_TUPLES_AGG * (FANOUT_PASS2 + 1))

/** @warning This padding must be allocated at the end of relation */
#define RELATION_PADDING_AGG \
  (PADDING_TUPLES_AGG * FANOUT_PASS1 * sizeof(agg::tuple_t))

/** \endinternal */

#endif /* PRJ_PARAMS_AGG_H */
