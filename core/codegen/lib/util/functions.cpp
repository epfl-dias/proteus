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
#include <codegen/util/functions.hpp>
#include <cstdio>
#include <cstring>
#include <platform/common/common.hpp>
#include <platform/memory/memory-allocator.hpp>
#include <platform/util/glog.hpp>
#include <string>

#ifdef __AVX2__
#include <immintrin.h>
#endif

double putchari(int X) {
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

int printi64(size_t X) {
  printf("[printi64:] Debugging int64, not size_t: %ld\n", X);

  //    printf("[printi64:] Generated code called %lu\n", X);

  // This is the appropriate one...
  //    printf("[printi64:] Generated code called %zu\n", X);
  // cout <<"[printi64:] Generated code called "<< X<< endl;
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

void printptr(void *ptr) {
  printf("[printptr:] Generated code called %p\n", ptr);
}

int printc(char *X) {
#ifdef DEBUG
  printf("[printc:] Generated code -- char read: %c\n", X[0]);
#else
  printf("%c\n", X[0]);
#endif
  return 0;
}

void printBoolean(bool in) {
  if (in) {
    printf("True\n");
  } else {
    printf("False\n");
  }
}

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

bool equalStrings(char *str1, char *str2) { return strcmp(str1, str2) == 0; }

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
    std::string error_msg =
        std::string("[convertBoolean: Error - unknown input]");
    LOG(ERROR) << error_msg;
    throw std::runtime_error(error_msg);
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
    std::string error_msg =
        std::string("[convertBoolean64: Error - unknown input]");
    LOG(ERROR) << error_msg;
    throw std::runtime_error(error_msg);
  }
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
      throw std::runtime_error(std::string("[ATOIS: ] Invalid Size "));
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

// XXX Copy string? Or edit in place?
size_t hashStringC(char *toHash, size_t start, size_t end) {
  char tmp = toHash[end];
  toHash[end] = '\0';
  std::hash<std::string> hasher;
  size_t result = hasher(toHash + start);
  toHash[end] = tmp;
  return result;
}

size_t hashString(string toHash) {
  std::hash<string> hasher;
  size_t result = hasher(toHash);
  return result;
}

size_t hashBoolean(bool toHash) {
  std::hash<bool> hasher;
  return hasher(toHash);
}

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
      cout << "[AVX2:] Newline / End of line at pos " << strIdx << std::endl;
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

void parseLineJSON(char *buf, size_t start, size_t end, jsmntok_t **tokens,
                   size_t line) {
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
