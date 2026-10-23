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
#ifndef PROTEUS_FUNCTIONS_HPP
#define PROTEUS_FUNCTIONS_HPP

#include <cstddef>
#include <cstdint>
#include <platform/util/string-object.hpp>
#include <string>

// #define JSON_TIGHT
#include "jsmn.h"

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

extern "C" double putchari(int X);

extern "C" void nonTemporalCopy(char *out, char *in, int n);

extern "C" int printi(int X);

extern "C" int printShort(short X);

extern "C" int printi64(size_t X);

extern "C" int printFloat(double X);

extern "C" void printptr(void *ptr);

extern "C" int printc(char *X);

extern "C" void printBoolean(bool X);

extern "C" int compareTokenString(const char *buf, size_t start, size_t end,
                                  const char *candidate);

extern "C" int compareTokenString64(const char *buf, size_t start, size_t end,
                                    const char *candidate);

// Definition is in the platform/lib/memory/buffer-manager.cu
extern "C" bool equalStringObjs(StringObject obj1, StringObject obj2);

extern "C" bool equalStrings(char *str1, char *str2);

extern "C" bool convertBoolean(const char *buf, int start, int end);

extern "C" bool convertBoolean64(const char *buf, size_t start, size_t end);

extern "C" int atois(const char *buf, int len);

/**
 * Hashing
 */

extern "C" size_t hashInt(int toHash);

extern "C" size_t hashInt64(int64_t toHash);

extern "C" size_t hashDouble(double toHash);

extern "C" size_t hashStringC(char *toHash, size_t start, size_t end);

size_t hashString(std::string toHash);

extern "C" size_t hashStringObject(StringObject obj);

extern "C" size_t hashBoolean(bool toHash);

extern "C" size_t combineHashes(size_t hash1, size_t hash2);

extern "C" size_t combineHashesNoOrder(size_t hash1, size_t hash2);

/**
 * Memory mgmt
 */

extern "C" void *getMemoryChunk(size_t chunkSize);

extern "C" void *increaseMemoryChunk(void *chunk, size_t chunkSize);

extern "C" void releaseMemoryChunk(void *chunk);

/**
 * Parsing
 */

extern "C" size_t newlineAVX(const char *const target, size_t targetLength);

extern "C" void parseLineJSON(char *buf, size_t start, size_t end,
                              jsmntok_t **tokens, size_t line);

#endif  // PROTEUS_FUNCTIONS_HPP
