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

#ifndef RAW_FUNCTIONS_HPP_
#define RAW_FUNCTIONS_HPP_

#include "util/raw-catalog.hpp"
#include "util/raw-context.hpp"
#include "util/raw-timing.hpp"
#include "util/radix/joins/radix-join.hpp"
#include "util/radix/aggregations/radix-aggr.hpp"

class RawContext;

RawContext * prepareContext(string moduleName);
//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

extern "C" int printi(int X);

extern "C" int printShort(short X);

extern "C" int printi64(size_t X);

extern "C" int printFloat(double X);

extern "C" int printc(char* X);

extern "C" void printBoolean(bool X);

extern "C" int atoi_llvm(const char* X);

extern "C" void insertIntKeyToHT(int htIdentifier, int key, void* value,
		int type_size);

extern "C" void** probeIntHT(int htIdentifier, int key, int typeIndex);

extern "C" void insertToHT(char* HTname, size_t key, void* value,
		int type_size);

extern "C" void** probeHT(char* HTname, size_t key);

extern "C" HashtableBucketMetadata* getMetadataHT(char* HTname);

extern "C" int compareTokenString(const char* buf, size_t start, size_t end,
		const char* candidate);

extern "C" int compareTokenString64(const char* buf, size_t start, size_t end,
		const char* candidate);

extern "C" bool equalStringObjs(StringObject obj1, StringObject obj2);

extern "C" bool equalStrings(char *str1, char *str2);

extern "C" bool convertBoolean(const char* buf, int start, int end);

extern "C" bool convertBoolean64(const char* buf, size_t start, size_t end);

extern "C" int atois(const char* buf, int len);

/**
 * Hashing
 */

extern "C" size_t hashInt(int toHash);

extern "C" size_t hashDouble(double toHash);

extern "C" size_t hashStringC(char* toHash, size_t start, size_t end);

//extern "C" size_t hashString(string toHash);
size_t hashString(string toHash);

extern "C" size_t hashStringObject(StringObject obj);

extern "C" size_t hashBoolean(bool toHash);

extern "C" size_t combineHashes(size_t hash1, size_t hash2);

extern "C" size_t combineHashesNoOrder(size_t hash1, size_t hash2);

/**
 * Radix hashing
 */

extern "C" int *partitionHTLLVM(size_t num_tuples, joins::tuple_t *inTuples);
extern "C" void bucket_chaining_join_prepareLLVM(const joins::tuple_t * const tuplesR,
		int num_tuples, HT * ht);
extern "C" int *partitionAggHTLLVM(size_t num_tuples, agg::tuple_t *inTuples);
extern "C" void bucket_chaining_agg_prepareLLVM(const agg::tuple_t * const tuplesR,
		int num_tuples, HT * ht);
/**
 * Flushing data
 */

extern "C" void flushObjectStart(char* fileName);

extern "C" void flushObjectEnd(char* fileName);

extern "C" void flushArrayStart(char* fileName);

extern "C" void flushArrayEnd(char* fileName);

extern "C" void flushInt(int toFlush, char* fileName);

extern "C" void flushInt64(size_t toFlush, char* fileName);

extern "C" void flushDouble(double toFlush, char* fileName);

extern "C" void flushBoolean(bool toFlush, char* fileName);

extern "C" void flushStringC(char* toFlush, size_t start, size_t end,
		char* fileName);

//Used for pre-existing, well-formed strings (e.g. Record attributes)
extern "C" void flushStringReady(char* toFlush, char* fileName);

extern "C" void flushStringObject(StringObject toFlush, char* fileName);

extern "C" void flushChar(char whichChar, char* fileName);

extern "C" void flushOutput(char* fileName);

extern "C" void flushDelim(size_t resultCtr, char whichDelim, char* fileName);

/**
 * Memory mgmt
 */
extern "C" void* getMemoryChunk(size_t chunkSize);
extern "C" void* increaseMemoryChunk(void* chunk, size_t chunkSize);
extern "C" void releaseMemoryChunk(void* chunk);

/**
 * Parsing
 */
extern "C" size_t newlineAVX(const char* const target, size_t targetLength);
extern "C" void parseLineJSON(char *buf, size_t start, size_t end,
		jsmntok_t** tokens, size_t line);

/**
 * Timing
 */
extern "C" void resetTime();

extern "C" void calculateTime();

#endif /* RAW_FUNCTIONS_HPP_ */
