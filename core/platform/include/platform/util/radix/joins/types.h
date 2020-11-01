/**
 * @file    types.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue May 22 16:43:30 2012
 * @version $Id: types.h 3017 2012-12-07 10:56:20Z bcagri $
 *
 * @brief  Provides general type definitions used by all join algorithms.
 *
 *
 */
#ifndef JOIN_TYPES_H
#define JOIN_TYPES_H

#include "platform/util/radix/types.h"

namespace joins {
/**
 * @defgroup Types Common Types
 * Common type definitions used by all join implementations.
 * @{
 */
#ifdef KEY_8B /* 64-bit key/value, 16B tuples */
typedef int64_t intkey_t;
typedef int64_t value_t;
#else /* 32-bit key / 64-bit value, 12B tuples */
typedef int32_t intkey_t;
typedef int64_t value_t;
// typedef void* value_t;
#endif

/** Type definition for a tuple, depending on KEY_8B a tuple can be 16B or 8B */
typedef struct tuple_t {
  intkey_t key;
  value_t payload;
} tuple_t;

/**
 * Type definition for a relation.
 * It consists of an array of tuples and a size of the relation.
 */
typedef struct relation_t {
  tuple_t* tuples;
  uint32_t num_tuples;
} relation_t;

/** @} */
}  // namespace joins

#endif /* TYPES_H */
