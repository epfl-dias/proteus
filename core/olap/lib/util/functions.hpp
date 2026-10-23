/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
        Data Intensive Applications and Systems Laboratory (DIAS)
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

#ifndef FUNCTIONS_HPP_
#define FUNCTIONS_HPP_

#include <deque>
#include <platform/util/string-object.hpp>
#include <platform/util/timing.hpp>
#include <string>

#include "lib/util/radix/aggregations/radix-aggr.hpp"
#include "lib/util/radix/joins/radix-join.hpp"

class OlapParallelContext;
struct HashtableBucketMetadata;

OlapParallelContext *prepareOlapContext(string moduleName,
                                        bool gpuRoot = false);
//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

extern "C" int atoi_llvm(const char *X);

extern "C" void insertIntKeyToHT(int htIdentifier, int key, void *value,
                                 int type_size);

extern "C" void **probeIntHT(int htIdentifier, int key, int typeIndex);

extern "C" void insertToHT(char *HTname, size_t key, void *value,
                           int type_size);

extern "C" void **probeHT(char *HTname, size_t key);

extern "C" HashtableBucketMetadata *getMetadataHT(char *HTname);

/**
 * Radix hashing
 */

extern "C" int *partitionHTLLVM(size_t num_tuples, joins::tuple_t *inTuples);
extern "C" void bucket_chaining_join_prepareLLVM(
    const joins::tuple_t *const tuplesR, int num_tuples, HT *ht);
extern "C" int *partitionAggHTLLVM(size_t num_tuples, agg::tuple_t *inTuples);
extern "C" void bucket_chaining_agg_prepareLLVM(
    const agg::tuple_t *const tuplesR, int num_tuples, HT *ht);
/**
 * Flushing data
 */

extern "C" void flushObjectStart(char *fileName);

extern "C" void flushObjectEnd(char *fileName);

extern "C" void flushArrayStart(char *fileName);

extern "C" void flushArrayEnd(char *fileName);

extern "C" void flushInt(int toFlush, char *fileName);

extern "C" void flushDString(int toFlush, void *dict, char *fileName);

extern "C" void flushIntDequeAsBag(std::deque<int32_t> *toFlush,
                                   char *fileName);

extern "C" void flushInt64(int64_t toFlush, char *fileName);

extern "C" void flushDate(int64_t toFlush, char *fileName);

extern "C" void flushDouble(double toFlush, char *fileName);

extern "C" void flushBoolean(bool toFlush, char *fileName);

// Used for debugging purposes, ptrs are generally meaningless when flushed
extern "C" void flushPtr(uintptr_t ptr, char *fileName);

extern "C" void flushStringC(char *toFlush, size_t start, size_t end,
                             char *fileName);

// Used for pre-existing, well-formed strings (e.g. Record attributes)
extern "C" void flushStringReady(char *toFlush, char *fileName);

extern "C" void flushStringObject(StringObject toFlush, char *fileName);

extern "C" void flushChar(char whichChar, char *fileName);

extern "C" void flushOutput(char *fileName);
extern "C" void flushBinaryOutput(char *fileName, std::ostream *strBuffer);

extern "C" void flushDelim(size_t resultCtr, char whichDelim, char *fileName);

#endif /* FUNCTIONS_HPP_ */
