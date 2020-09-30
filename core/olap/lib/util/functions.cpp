/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "functions.hpp"

#include <chrono>
#include <common/error-handling.hpp>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <storage/storage-manager.hpp>

#include "catalog.hpp"
#include "olap/util/parallel-context.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
// TODO: remove as soon as the default GCC moves filesystem out of experimental
//  GCC 8.3 has made the transition, but the default GCC in Ubuntu 18.04 is 7.4
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

//#define JSON_TIGHT
#include "jsmn.h"

// Remember to add these functions as extern in .hpp too!
extern "C" double putchari(int X) {
  putchar((char)X);
  return 0;
}

#define CACHE_CAP 1024

void nonTemporalCopy(char *out, char *in, int n) {
  // TODO: use more portable code for non temporal copy and/or bigger registers
  for (int i = 0; i < n * CACHE_CAP; i += 32) {
    typedef int __v4di_aligned __attribute__((__vector_size__(32)))
    __attribute__((aligned(32)));
    __builtin_nontemporal_store(*((__v4di_aligned *)(out + i)),
                                (__v4di_aligned *)(in + i));
  }
}

void printBoolean(bool in) {
  if (in) {
    printf("True\n");
  } else {
    printf("False\n");
  }
}

// printd - printf that takes a double prints it as "%f\n", returning 0.
int printi(int X) {
#ifdef DEBUG
  printf("[printi:] Generated code called %d\n", X);
#else
  printf("%d\n", X);
#endif
  return 0;
}

int printShort(short X) {
  printf("[printShort:] Generated code called %d\n", X);
  return 0;
}

int printFloat(double X) {
#ifdef DEBUG
  printf("[printFloat:] Generated code called %f\n", X);
#else
  printf("%f\n", X);
#endif

  return 0;
}

int printi64(size_t X) {
  printf("[printi64:] Debugging int64, not size_t: %ld\n", X);

  //    printf("[printi64:] Generated code called %lu\n", X);

  // This is the appropriate one...
  //    printf("[printi64:] Generated code called %zu\n", X);
  // cout <<"[printi64:] Generated code called "<< X<< endl;
  return 0;
}

extern "C" void printptr(void *ptr) {
  printf("[printptr:] Generated code called %p\n", ptr);
}

int printc(char *X) {
  printf("[printc:] Generated code -- char read: %c\n", X[0]);
  return 0;
}

// int s(const char* X) {
//    //printf("Generated code -- char read: %c\n", X[0]);
//    return atoi(X);
//}

void insertToHT(char *HTname, size_t key, void *value, int type_size) {
  Catalog &catalog = Catalog::getInstance();
  // still, one unneeded indirection..... is there a quicker way?
  multimap<size_t, void *> *HT = catalog.getHashTable(string(HTname));

  void *valMaterialized = malloc(type_size);
  memcpy(valMaterialized, value, type_size);

  HT->insert(pair<size_t, void *>(key, valMaterialized));

  //    HT->insert(pair<int,void*>(key,value));
  LOG(INFO) << "[Insert: ] Hash key " << key << " inserted successfully";

  LOG(INFO) << "[INSERT: ] There are " << HT->count(key)
            << " elements with key " << key << ":";
}

void **probeHT(char *HTname, size_t key) {
  string name = string(HTname);
  Catalog &catalog = Catalog::getInstance();

  // same indirection here as above.
  multimap<size_t, void *> *HT = catalog.getHashTable(name);

  auto results = HT->equal_range(key);

  void **bindings = nullptr;
  int count = HT->count(key);
  LOG(INFO) << "[PROBE:] There are " << HT->count(key)
            << " elements with hash key " << key;
  if (count) {
    //+1 used to set last position to null and know when to terminate
    bindings = new void *[count + 1];
    bindings[count] = nullptr;
  } else {
    bindings = new void *[1];
    bindings[0] = nullptr;
    return bindings;
  }

  int curr = 0;
  for (multimap<size_t, void *>::iterator it = results.first;
       it != results.second; ++it) {
    bindings[curr] = it->second;
    curr++;
  }
  return bindings;
}

/**
 * TODO
 * Obviously extremely inefficient.
 * Once having replaced multimap for our own code,
 * we also need to gather this metadata at build time.
 *
 * Examples: Number of buckets (keys) / elements in each bucket
 */
HashtableBucketMetadata *getMetadataHT(char *HTname) {
  string name = string(HTname);
  Catalog &catalog = Catalog::getInstance();

  // same indirection here as above.
  multimap<size_t, void *> *HT = catalog.getHashTable(name);

  vector<size_t> keys;
  for (multimap<size_t, void *>::iterator it = HT->begin(), end = HT->end();
       it != end; it = HT->upper_bound(it->first)) {
    keys.push_back(it->first);
    // cout << it->first << ' ' << it->second << endl;
  }
  HashtableBucketMetadata *metadata =
      new HashtableBucketMetadata[keys.size() + 1];
  size_t pos = 0;
  for (auto &it : keys) {
    metadata[pos].hashKey = it;
    metadata[pos].bucketSize = HT->count(it);
    pos++;
  }
  // XXX Silly stopping condition..
  metadata[pos].bucketSize = 0;
  return metadata;
}

/* Deprecated */
void insertIntKeyToHT(int htIdentifier, int key, void *value, int type_size) {
  Catalog &catalog = Catalog::getInstance();
  // still, one unneeded indirection..... is there a quicker way?
  multimap<int, void *> *HT = catalog.getIntHashTable(htIdentifier);

  void *valMaterialized = malloc(type_size);
  // FIXME obviously expensive, but probably cannot be helped
  memcpy(valMaterialized, value, type_size);

  HT->insert(pair<int, void *>(key, valMaterialized));
  //    cout << "INSERTED KEY " << key << endl;

#ifdef DEBUG
//    LOG(INFO) << "[Insert: ] Integer key " << key << " inserted successfully";
//
//    LOG(INFO) << "[INSERT: ] There are " << HT->count(key)
//            << " elements with key " << key << ":";
#endif
}

/* Deprecated */
void **probeIntHT(int htIdentifier, int key, int typeIndex) {
  //    string name = string(HTname);
  Catalog &catalog = Catalog::getInstance();

  // same indirection here as above.
  multimap<int, void *> *HT = catalog.getIntHashTable(htIdentifier);

  pair<multimap<int, void *>::iterator, multimap<int, void *>::iterator>
      results;
  results = HT->equal_range(key);

  void **bindings = nullptr;
  int count = HT->count(key);

  if (count) {
    //+1 used to set last position to null and know when to terminate
    bindings = new void *[count + 1];
    bindings[count] = nullptr;
  } else {
    bindings = new void *[1];
    bindings[0] = nullptr;
    return bindings;
  }

  int curr = 0;
  for (multimap<int, void *>::iterator it = results.first; it != results.second;
       ++it) {
    bindings[curr] = it->second;
    curr++;
  }
#ifdef DEBUG
  LOG(INFO) << "[PROBE INT:] There are " << HT->count(key)
            << " elements with key " << key;
#endif
  return bindings;
}

bool equalStrings(char *str1, char *str2) { return strcmp(str1, str2) == 0; }

int compareTokenString(const char *buf, size_t start, size_t end,
                       const char *candidate) {
  //    cout << "Candidate?? " << candidate << endl;
  //    cout << "Buf?" << start << " " << end << endl;
  return (strncmp(buf + start, candidate, end - start) == 0 &&
          strlen(candidate) == end - start);
}

int compareTokenString64(const char *buf, size_t start, size_t end,
                         const char *candidate) {
  //    cout << "Start? " << start << endl;
  //    cout << "End? " << end << endl;
  //    cout << "Candidate?? " << candidate << endl;
  //    char *deleteme = (char*) malloc(end - start +1);
  //    memcpy(deleteme,buf+start,end-start);
  //    deleteme[end-start] = '\0';
  //    cout << "From file: " << deleteme << endl;
  return (strncmp(buf + start, candidate, end - start) == 0 &&
          strlen(candidate) == end - start);
}

bool convertBoolean(const char *buf, int start, int end) {
  if (compareTokenString(buf, start, end, "true") == 1 ||
      compareTokenString(buf, start, end, "TRUE") == 1 ||
      compareTokenString(buf, start, end, "1") == 1) {
    return true;
  } else if (compareTokenString(buf, start, end, "false") == 1 ||
             compareTokenString(buf, start, end, "FALSE") == 1 ||
             compareTokenString(buf, start, end, "0") == 1) {
    return false;
  } else {
    string error_msg = string("[convertBoolean: Error - unknown input]");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
}

bool convertBoolean64(const char *buf, size_t start, size_t end) {
  if (compareTokenString64(buf, start, end, "true") == 1 ||
      compareTokenString64(buf, start, end, "TRUE") == 1 ||
      compareTokenString64(buf, start, end, "1") == 1) {
    return true;
  } else if (compareTokenString64(buf, start, end, "false") == 1 ||
             compareTokenString64(buf, start, end, "FALSE") == 1 ||
             compareTokenString64(buf, start, end, "0") == 1) {
    return false;
  } else {
    string error_msg = string("[convertBoolean64: Error - unknown input]");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
}

size_t hashInt(int toHash) {
  std::hash<int> hasher;
  return hasher(toHash);
}

size_t hashInt64(int64_t toHash) {
  std::hash<int64_t> hasher;
  return hasher(toHash);
}

size_t hashDouble(double toHash) {
  std::hash<double> hasher;
  return hasher(toHash);
}

size_t hashString(string toHash) {
  std::hash<string> hasher;
  size_t result = hasher(toHash);
  return result;
}

// XXX Copy string? Or edit in place?
size_t hashStringC(char *toHash, size_t start, size_t end) {
  char tmp = toHash[end];
  toHash[end] = '\0';
  std::hash<string> hasher;
  size_t result = hasher(toHash + start);
  toHash[end] = tmp;
  return result;
}

size_t hashBoolean(bool toHash) {
  std::hash<bool> hasher;
  return hasher(toHash);
}

size_t hashStringObject(StringObject obj) {
  // No '+1' needed here
  char tmp = obj.start[obj.len];
  obj.start[obj.len] = '\0';
  //    obj.start[obj.len+1] = '\0';
  // cout << "To Hash: " << obj.start << endl;
  std::hash<string> hasher;
  size_t result = hasher(obj.start);
  obj.start[obj.len + 1] = tmp;
  return result;
}

// size_t combineHashes(size_t hash1, size_t hash2) {
//     size_t seed = 0;
//     std::hash_combine(seed, hash1);
//     std::hash_combine(seed, hash2);
//     return seed;
//}
//
// template <class T>
// inline void hash_combine_no_order(size_t& seed, const T& v)
//{
//    std::hash<T> hasher;
//    seed ^= hasher(v);
//}
//
// size_t combineHashesNoOrder(size_t hash1, size_t hash2) {
//     size_t seed = 0;
//     hash_combine_no_order(seed, hash1);
//     hash_combine_no_order(seed, hash2);
//     return seed;
//}

size_t combineHashes(size_t hash1, size_t hash2) {
  hash_combine(hash1, hash2);
  return hash1;
}

template <class T>
inline void hash_combine_no_order(size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v);
}

size_t combineHashesNoOrder(size_t hash1, size_t hash2) {
  hash_combine_no_order(hash1, hash2);
  return hash1;
}

/**
 * Radix chunks of functionality
 */
int *partitionHTLLVM(size_t num_tuples, joins::tuple_t *inTuples) {
  return partitionHT(num_tuples, inTuples);
}

void bucket_chaining_join_prepareLLVM(const joins::tuple_t *const tuplesR,
                                      int num_tuples, HT *ht) {
  bucket_chaining_join_prepare(tuplesR, num_tuples, ht);
}

void bucket_chaining_agg_prepareLLVM(const agg::tuple_t *const tuplesR,
                                     int num_tuples, HT *ht) {
  bucket_chaining_agg_prepare(tuplesR, num_tuples, ht);
}

int *partitionAggHTLLVM(size_t num_tuples, agg::tuple_t *inTuples) {
  return partitionHT(num_tuples, inTuples);
}

void flushDString(int toFlush, void *dict, char *fileName) {
  assert(dict && "Dictionary should not be null!");
  map<int, std::string> *actual_dict{(map<int, std::string> *)dict};
  Catalog &catalog = Catalog::getInstance();
  string name = string(fileName);
  auto &strBuffer = catalog.getSerializer(name);
  strBuffer << '"';
  try {
    strBuffer << actual_dict->at(toFlush);
  } catch (std::out_of_range &) {
    strBuffer << "unimplemented // TODO: oltp strings";
  }
  strBuffer << '"';
}

template <typename T>
void flush(const T &toFlush, const char *fileName) {
  Catalog::getInstance().getSerializer(fileName) << toFlush;
}

void flushInt(int toFlush, char *fileName) { flush(toFlush, fileName); }

void flushInt64(size_t toFlush, char *fileName) { flush(toFlush, fileName); }

void flushDate(int64_t toFlush, char *fileName) {
  flush(time_t{toFlush}, fileName);
  // std::put_time(std::localtime(&t), L"%Y-%m-%d");
}

void flushDouble(double toFlush, char *fileName) { flush(toFlush, fileName); }

void flushBoolean(bool toFlush, char *fileName) { flush(toFlush, fileName); }

void flushStringC(char *toFlush, size_t start, size_t end, char *fileName) {
  assert(start <= end);
  auto &strBuffer = Catalog::getInstance().getSerializer(fileName);
  strBuffer.write(toFlush + start, end - start);
}

void flushStringReady(char *toFlush, char *fileName) {
  auto &strBuffer = Catalog::getInstance().getSerializer(fileName);
  strBuffer << '"' << toFlush << '"';
}

void flushStringObject(StringObject obj, char *fileName) {
  auto &strBuffer = Catalog::getInstance().getSerializer(fileName);
  strBuffer << '"';
  strBuffer.write(obj.start, obj.len);
  strBuffer << '"';
}

void flushObjectStart(char *fileName) { flush('{', fileName); }

void flushArrayStart(char *fileName) { flush('[', fileName); }

void flushObjectEnd(char *fileName) { flush('}', fileName); }

void flushArrayEnd(char *fileName) { flush(']', fileName); }

void flushChar(char toFlush, char *fileName) { flush(toFlush, fileName); }

void flushDelim(size_t resultCtr, char whichDelim, char *fileName) {
  if (likely(resultCtr > 0)) {
    flushChar(whichDelim, fileName);
  }
}

static std::map<std::ostream *, std::map<std::string, int32_t>> dicts;

void flushDictIfExists(std::ostream *ptr, const char *fileName) {
  if (!dicts.count(ptr)) return;
  auto dictName = "/dev/shm/" + std::string{fileName} + ".dict";
  LOG(INFO) << "Flushing dictionary to " << dictName << endl;
  std::ofstream out{dictName};
#ifndef NDEBUG
  auto prev = std::numeric_limits<int32_t>::lowest();
#endif
  LOG(INFO) << dicts.at(ptr).size();
  for (auto &e : dicts.at(ptr)) {
    out << e.first << ":" << e.second << '\n';
    assert(e.second > prev);
#ifndef NDEBUG
    prev = e.second;
#endif
  }
  // Invalidate file!
  StorageManager::getInstance().unloadFile(dictName);
}

void flushBinaryOutput(char *fileName, std::ostream *strBuffer) {
  LOG(INFO) << "Flushing to " << fileName << endl;
  {
    std::filesystem::path p{fileName};
    if (p.is_relative()) {
      // Here we assume that every flushOutput will result in one QueryResult
      // The shm_open keeps the file alive in shm. The release happens
      // when all file descriptors have been closed and after shm_unlink has
      // been called. Thus, we can immediately close the file descriptor
      // returned by shm_open. The QueryResult is responsible to shm_unlink
      // the file to avoid resource leakage.

      // shm_open is probably not needed here. At least it works without it,
      // but then, what are the guarantees for the lifetime of the file?
      // Is it well defined? If we never unlink it, does it persist forever?
      // int fd = linux_run(shm_open(fileName, O_CREAT | O_RDWR, S_IRWXU));
      // linux_run(close(fd));
      p = "/dev/shm" / p;
    }

    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }

    // assert(std::filesystem::exists(p) && "Too many open file descriptors?");

    std::ofstream{p, std::ios::app} << strBuffer->rdbuf();

    flushDictIfExists(strBuffer, fileName);
    // const string &tmp_str = strBuffer->str();     //more portable but it
    // creates a copy, which will be problematic for big output files...
    // write(fd, tmp_str.c_str(), tmp_str.size());

    // __gnu_cxx::stdio_filebuf<char> filebuf(fd, std::ofstream::out |
    // std::ofstream::app); std::ostream outFile(&filebuf);

    // write(fd, strBuffer->str());//->rdbuf());
    // outFile << strBuffer->rdbuf();
    // shm_unlink(fileName); //REMEMBER to unlink at the end of the test

    // Invalidate file!
    StorageManager::getInstance().unloadFile(p);
  }
  // {
  // time_block t("memfd_create: ");
  // interestingly enough, opening /dev/null takes 0ms, while opening another
  // file takes from 4 to 25 (!) ms!
  // ofstream outFile("/dev/null",std::ofstream::out | std::ofstream::app);
  // ofstream outFile;
  //     cout << "Flushing to " << fileName << endl;

  // outFile.open("/dev/null",std::ofstream::out | std::ofstream::app);
  // //    const char *toFlush = strBuffer->rdbuf()->str().c_str();
  // //    cout << "Contents being flushed: " << toFlush << endl;
  // //    cout << "Contents being flushed: " << std::endl << strBuffer->str()
  // << std::flush << std::endl;
  //     outFile << strBuffer->rdbuf();
  //     //outFile << strBuffer->rdbuf();

  //     outFile.close();
  // }
}

void flushOutput(char *fileName) {
  return flushBinaryOutput(fileName,
                           &Catalog::getInstance().getSerializer(fileName));
}

/**
 * Memory mgmt
 */

void *getMemoryChunk(size_t chunkSize) { return allocateFromRegion(chunkSize); }

void *increaseMemoryChunk(void *chunk, size_t chunkSize) {
  return increaseRegion(chunk, chunkSize);
}

void releaseMemoryChunk(void *chunk) { return freeRegion(chunk); }

/**
 * Parsing
 */
/*
 * Return position of \n
 * Code from
 * https://www.klittlepage.com/2013/12/10/accelerated-fix-processing-via-avx2-vector-instructions/
 *
 * XXX Assumption: all lines of file end with \n
 */
/*
As we're looking for simple, single character needles (newlines) we can use
bitmasking to search in lieu of SSE 4.2 string comparison functions. This simple
implementation splits a 256 bit AVX register into eight 32-bit words. Whenever
a word is non-zero (any bits are set within the word) a linear scan identifies
the position of the matching character within the 32-bit word.
*/
//__attribute__((always_inline))
// inline
size_t newlineAVX(const char *const target, size_t targetLength) {
  char nl = '\n';
#ifdef __AVX2__
  //    cout << "AVX mode ON" << endl;
  __m256i eq = _mm256_set1_epi8(nl);
  size_t strIdx = 0;
  union {
    __m256i v;
    char c[32];
  } testVec;
  union {
    __m256i v;
    uint32_t l[8];
  } mask;

  if (targetLength >= 32) {
    for (; strIdx <= targetLength - 32; strIdx += 32) {
      testVec.v = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(target + strIdx));
      mask.v = _mm256_cmpeq_epi8(testVec.v, eq);
      for (int i = 0; i < 8; ++i) {
        if (0 != mask.l[i]) {
          for (int j = 0; j < 4; ++j) {
            char c = testVec.c[4 * i + j];
            if (nl == c) {
              //                            cout << "1. NL at pos (" << strIdx
              //                            << "+" << 4*i+j << ")" << endl; cout
              //                            << "[AVX1:] Newline / End of line at
              //                            pos " << strIdx + 4 * i + j << endl;
              return strIdx + 4 * i + j;
            }
          }
        }
      }
    }
  }

  for (; strIdx < targetLength; ++strIdx) {
    const char c = target[strIdx];
    if (nl == c) {
      // cout << "2. NL at pos " << strIdx << endl;
      cout << "[AVX2:] Newline / End of line at pos " << strIdx << endl;
      return strIdx;
    }
  }

  string error_msg = string("No newline found");
  LOG(ERROR) << error_msg;
  throw runtime_error(error_msg);
#else
  // cout << "Careful: Non-AVX parsing" << endl;
  size_t i = 0;
  while (target[i] != nl && i < targetLength) {
    i++;
  }
  //    if(i == targetLength && target[i] != nl)    {
  //        string error_msg = string("No newline found");
  //            LOG(ERROR)<< error_msg;
  //    }
  //    cout << "[Non-AVX:] Newline / End of line at pos " << i << endl;
  return i;
#endif
}

// void parseLineJSON(char *buf, size_t start, size_t end, jsmntok_t** tokens,
// size_t line)    {
//
//    int error_code;
//    jsmn_parser p;
//
//    /* inputs/json/jsmnDeeper-flat.json : MAXTOKENS = 25 */
//
//    //Populating our json 'positional index'
//    jsmntok_t* tokenArray = (jsmntok_t*) calloc(MAXTOKENS,sizeof(jsmntok_t));
//    if(tokens == nullptr)
//    {
//        throw runtime_error(string("new() of tokens failed"));
//    }
//
//    jsmn_init(&p);
//    char* bufShift = buf + start;
//    char eol = buf[end];
//    buf[end] = '\0';
////    printf("JSON Raw Input: %s\n",bufShift);
////    printf("Which line? %d\n",line);
//    error_code = jsmn_parse(&p, bufShift, end - start, tokenArray, MAXTOKENS);
//    buf[end] = eol;
//    if(error_code < 0)
//    {
//        string msg = "Json (JSMN) plugin failure: ";
//        LOG(ERROR) << msg << error_code;
//        throw runtime_error(msg);
//    }
////    else
////    {
////        cout << "How many tokens?? " << error_code << endl;
////    }
//    tokens[line] = tokenArray;
////    cout << "[parseLineJSON: ] " << tokenArray[0].start << " to " <<
/// tokenArray[0].end << endl;
//}

extern "C" void parseLineJSON(char *buf, size_t start, size_t end,
                              jsmntok_t **tokens, size_t line) {
  //    cout << "[parseLineJSON: ] Entry for line " << line << " from " << start
  //    << " to " << end << endl;
  int error_code;
  jsmn_parser p;

  /* inputs/json/jsmnDeeper-flat.json : MAXTOKENS = 25 */

  // Populating our json 'positional index'

  jsmn_init(&p);
  char *bufShift = buf + start;
  char eol = buf[end];
  buf[end] = '\0';
  //    error_code = jsmn_parse(&p, bufShift, end - start, tokens[line],
  //    MAXTOKENS); printf("Before %ld %ld %ld\n",tokens,tokens + line,
  //    tokens[line]);
  size_t tokensNo = MAXTOKENS;
  error_code = jsmn_parse(&p, bufShift, end - start, &(tokens[line]), tokensNo);
  //    printf("After %ld %ld\n",tokens,tokens[line]);
  buf[end] = eol;
  //    if(line > 0 && (line +1)% 10000000 == 0)
  //    {
  //        printf("Processing line no. %ld\n",line);
  //    }
  //    if (line > 0 && (line + 1) % 100000 == 0) {
  //        printf("Processing line no. %ld\n", line);
  //    }
  if (error_code < 0) {
    string msg = "Json (JSMN) plugin failure: ";
    LOG(ERROR) << msg << error_code << " in line " << line;
    throw runtime_error(msg);
  }
  //    else
  //    {
  //        cout << "How many tokens?? " << error_code << " in line " << line <<
  //        endl;
  //    }
  //    cout << "[parseLineJSON - " << line << ": ] "
  //    << tokens[line][0].start
  //    << " to " << tokens[line][0].end << endl;
  //    cout << "[parseLineJSON - exit] "<< endl;
}

//'Inline' -> shouldn't it be placed in .hpp?
inline int atoi1(const char *buf) { return (buf[0] - '0'); }

inline int atoi2(const char *buf) {
  return ((buf[0] - '0') * 10) + (buf[1] - '0');
}

inline int atoi3(const char *buf) {
  return ((buf[0] - '0') * 100) + ((buf[1] - '0') * 10) + (buf[2] - '0');
}

inline int atoi4(const char *buf) {
  return ((buf[0] - '0') * 1000) + ((buf[1] - '0') * 100) +
         ((buf[2] - '0') * 10) + (buf[3] - '0');
}

inline int atoi5(const char *buf) {
  return ((buf[0] - '0') * 10000) + ((buf[1] - '0') * 1000) +
         ((buf[2] - '0') * 100) + ((buf[3] - '0') * 10) + (buf[4] - '0');
}

inline int atoi6(const char *buf) {
  return ((buf[0] - '0') * 100000) + ((buf[1] - '0') * 10000) +
         ((buf[2] - '0') * 1000) + ((buf[3] - '0') * 100) +
         ((buf[4] - '0') * 10) + (buf[5] - '0');
}

inline int atoi7(const char *buf) {
  return ((buf[0] - '0') * 1000000) + ((buf[1] - '0') * 100000) +
         ((buf[2] - '0') * 10000) + ((buf[3] - '0') * 1000) +
         ((buf[4] - '0') * 100) + ((buf[5] - '0') * 10) + (buf[6] - '0');
}

inline int atoi8(const char *buf) {
  return ((buf[0] - '0') * 10000000) + ((buf[1] - '0') * 1000000) +
         ((buf[2] - '0') * 100000) + ((buf[3] - '0') * 10000) +
         ((buf[4] - '0') * 1000) + ((buf[5] - '0') * 100) +
         ((buf[6] - '0') * 10) + (buf[7] - '0');
}

inline int atoi9(const char *buf) {
  return ((buf[0] - '0') * 100000000) + ((buf[1] - '0') * 10000000) +
         ((buf[2] - '0') * 1000000) + ((buf[3] - '0') * 100000) +
         ((buf[4] - '0') * 10000) + ((buf[5] - '0') * 1000) +
         ((buf[6] - '0') * 100) + ((buf[7] - '0') * 10) + (buf[8] - '0');
}

inline int atoi10(const char *buf) {
  return ((buf[0] - '0') * 1000000000) + ((buf[1] - '0') * 100000000) +
         ((buf[2] - '0') * 10000000) + ((buf[3] - '0') * 1000000) +
         ((buf[4] - '0') * 100000) + ((buf[5] - '0') * 10000) +
         ((buf[6] - '0') * 1000) + ((buf[7] - '0') * 100) +
         ((buf[8] - '0') * 10) + (buf[9] - '0');
}

int atois(const char *buf, int len) {
  switch (len) {
    case 1:
      return atoi1(buf);
    case 2:
      return atoi2(buf);
    case 3:
      return atoi3(buf);
    case 4:
      return atoi4(buf);
    case 5:
      return atoi5(buf);
    case 6:
      return atoi6(buf);
    case 7:
      return atoi7(buf);
    case 8:
      return atoi8(buf);
    case 9:
      return atoi9(buf);
    case 10:
      return atoi10(buf);
    default:
      LOG(ERROR) << "[ATOIS: ] Invalid Size " << len;
      throw runtime_error(string("[ATOIS: ] Invalid Size "));
  }
}

ParallelContext *prepareContext(string moduleName) {
  ParallelContext *ctx = new ParallelContext(moduleName);
  // registerFunctions(ctx);
  return ctx;
}

template <typename T>
void flushBinary_impl(const T &toFlush, std::ostream *fileName) {
  fileName->write(reinterpret_cast<const char *>(&toFlush), sizeof(T));
}

extern "C" void flushBinaryi8(int8_t x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinaryi16(int16_t x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinaryi32(int32_t x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinaryi64(int64_t x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinarydouble(double x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinaryfloat(float x, std::ostream *fileName) {
  flushBinary_impl(x, fileName);
}

extern "C" void flushBinarystr(const char *s, uint32_t len,
                               std::ostream *fileName) {
  auto &d = dicts[fileName];
  int32_t dkey;
  std::string key{s, len};
  auto kit = d.find(key);
  if (kit == d.end()) {
    if (d.size() < 2) {
      if (d.empty()) {
        dkey = std::numeric_limits<int32_t>::max() / 2;
      } else {
        dkey = std::numeric_limits<int32_t>::max() / 4;
        if (key > d.begin()->first) {
          dkey *= 3;
        }
      }
    } else {
      int32_t lower_key;
      int32_t greater_key;
      auto it_gt = d.upper_bound(key);
      if (it_gt == d.end()) {
        greater_key = std::numeric_limits<int32_t>::max() - 1;
      } else {
        greater_key = it_gt->second;
      }
      if (it_gt == d.begin()) {
        lower_key = 0;
      } else {
        lower_key = (--it_gt)->second;
      }
      assert(lower_key <= greater_key);
      dkey = (greater_key & lower_key) + (greater_key ^ lower_key) / 2;
      assert(lower_key < dkey);
      assert(greater_key > dkey);
    }
    d.emplace(std::move(key), dkey);
  } else {
    dkey = kit->second;
  }
  flushBinary_impl(dkey, fileName);
}
