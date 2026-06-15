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

#include "functions.hpp"

#include <chrono>
#include <fstream>
#include <olap/util/parallel-context.hpp>
#include <ostream>
#include <platform/common/error-handling.hpp>
#include <storage/storage-manager.hpp>

#include "catalog.hpp"
#include "lib/util/jit/olap-pipeline.hpp"

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

// #define JSON_TIGHT
#include "jsmn.h"

OlapParallelContext *prepareOlapContext(string moduleName, bool gpuRoot) {
  auto *ctx = new OlapParallelContext(moduleName, gpuRoot);
  return ctx;
}

void insertToHT(char *HTname, size_t key, void *value, int type_size) {
  Catalog &catalog = Catalog::getInstance();
  // still, one unneeded indirection..... is there a quicker way?
  std::multimap<size_t, void *> *HT = catalog.getHashTable(string(HTname));

  void *valMaterialized = malloc(type_size);
  memcpy(valMaterialized, value, type_size);

  HT->insert(std::pair<size_t, void *>(key, valMaterialized));

  //    HT->insert(pair<int,void*>(key,value));
  LOG(INFO) << "[Insert: ] Hash key " << key << " inserted successfully";

  LOG(INFO) << "[INSERT: ] There are " << HT->count(key)
            << " elements with key " << key << ":";
}

void **probeHT(char *HTname, size_t key) {
  string name = string(HTname);
  Catalog &catalog = Catalog::getInstance();

  // same indirection here as above.
  std::multimap<size_t, void *> *HT = catalog.getHashTable(name);

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
  for (std::multimap<size_t, void *>::iterator it = results.first;
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
  std::multimap<size_t, void *> *HT = catalog.getHashTable(name);

  std::vector<size_t> keys;
  for (std::multimap<size_t, void *>::iterator it = HT->begin(),
                                               end = HT->end();
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
  std::multimap<int, void *> *HT = catalog.getIntHashTable(htIdentifier);

  void *valMaterialized = malloc(type_size);
  // FIXME obviously expensive, but probably cannot be helped
  memcpy(valMaterialized, value, type_size);

  HT->insert(std::pair<int, void *>(key, valMaterialized));
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
  std::multimap<int, void *> *HT = catalog.getIntHashTable(htIdentifier);

  std::pair<std::multimap<int, void *>::iterator,
            std::multimap<int, void *>::iterator>
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
  for (std::multimap<int, void *>::iterator it = results.first;
       it != results.second; ++it) {
    bindings[curr] = it->second;
    curr++;
  }
#ifdef DEBUG
  LOG(INFO) << "[PROBE INT:] There are " << HT->count(key)
            << " elements with key " << key;
#endif
  return bindings;
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

void flushInt64(int64_t toFlush, char *fileName) { flush(toFlush, fileName); }

[[maybe_unused]] /* called by codegen */
void flushIntDequeAsBag(std::deque<int32_t> *toFlush, char *fileName) {
  auto &out = Catalog::getInstance().getSerializer(fileName);
  out << "[";
  bool first = true;
  for (const auto &x : *toFlush) {
    if (!first) out << "|";
    out << x;
    first = false;
  }
  out << "]";
}

void flushDate(int64_t toFlush, char *fileName) {
  flush(time_t{toFlush}, fileName);
  // std::put_time(std::localtime(&t), L"%Y-%m-%d");
}

void flushDouble(double toFlush, char *fileName) { flush(toFlush, fileName); }

void flushBoolean(bool toFlush, char *fileName) { flush(toFlush, fileName); }

void flushPtr(uintptr_t ptr, char *fileName) { flush((void *)ptr, fileName); }

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
  LOG(INFO) << "Flushing dictionary to " << dictName << std::endl;
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
  LOG(INFO) << "Flushing to " << fileName << std::endl;
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
