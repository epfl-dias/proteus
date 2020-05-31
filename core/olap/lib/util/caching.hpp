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

#ifndef CACHING_HPP_
#define CACHING_HPP_

#include "common/common.hpp"
#include "olap/expressions/expressions.hpp"

#define CACHING_ON
//#define DEBUGCACHING

#include "olap/util/cache-info.hpp"

class CachingService {
 public:
  static CachingService &getInstance() {
    static CachingService instance;
    return instance;
  }

  void registerPM(string fileName, char *payloadPtr) {
    auto it = pmCaches.find(fileName);
    if (it != pmCaches.end()) {
      LOG(WARNING) << "PM caches already contain " << fileName;
      cout << "*Warning* PM caches already contain " << fileName << endl;
    }
    pmCaches[fileName] = payloadPtr;
  }

  char *getPM(string fileName) {
    auto it = pmCaches.find(fileName);
    if (it == pmCaches.end()) {
      LOG(INFO) << "No PM found for " << fileName;
      /* nullptr is a valid value (PM not found) */
      return nullptr;
    }
    return it->second;
  }

  void registerCache(const expressions::Expression *expr, CacheInfo payload,
                     bool entireDataset) {
#ifdef CACHING_ON
    auto it = binCaches.find(expr);
    bool found = false;
    decltype(binCacheIsFull)::iterator itBool;
    if (it != binCaches.end()) {
      LOG(WARNING) << "Bin. caches already contain expr " << expr->getTypeID();
      found = true;
      itBool = binCacheIsFull.find(expr);
    }

    /* XXX Different degrees of 'fullness' can be placed in these methods.
     * BUT: Not sure if C++ side is the one deciding on them */
    if (found) {
      /* Replace what is cached if
       * -> the new one is the full thing
       *
       * This implies that the newer chunk will
       * replace the older one, even if the older one
       * is related to a 'struct' with more fields.
       *
       * This should not be the case
       * -> we should be running a sophisticated algo for these decisions
       * -> XXX This also causes us to run (all) queries in a
       *       certain sequence to emulate caching effects
       */
      if (entireDataset /*&& !(itBool->second)*/) {
        binCaches.erase(expr);
        binCacheIsFull.erase(expr);
        binCaches[expr] = payload;
        binCacheIsFull[expr] = entireDataset;
        // cout << "Replaced in cache " << endl;
      }

    } else {
      binCaches[expr] = payload;
      binCacheIsFull[expr] = entireDataset;
#ifdef DEBUGCACHING
      cout << "Registered in cache " << endl;
#endif
    }
#endif
  }

  CacheInfo getCache(const expressions::Expression *expr) {
    auto it = binCaches.find(expr);
    if (it == binCaches.end()) {
      // cout << "No match out of " << binCaches.size() << " entries" << endl;
      // LOG(INFO) << "No Bin Cache found for expr of type "
      //           << expr->getExpressionType()->getType();
      CacheInfo invalid;
      invalid.structFieldNo = -1;
      return invalid;
    }
    return it->second;
  }

  bool getCacheIsFull(const expressions::Expression *expr) {
    auto it = binCacheIsFull.find(expr);
    if (it == binCacheIsFull.end()) {
      /* Should not occur - method should be used only
       * after establishing cache info exists */
      string error_msg = "No Bin Cache Info found for expr ";
      LOG(ERROR) << error_msg << expr->getTypeID();
      throw runtime_error(error_msg);
    }
    return it->second;
  }

  void clear();
  void clearPM();

 private:
  struct less_map
      : std::binary_function<const expressions::Expression *,
                             const expressions::Expression *, bool> {
    bool operator()(const expressions::Expression *a,
                    const expressions::Expression *b) const {
      return *a < *b;
    }
  };
  /*
   * From expressions::Expression to (cast) cache.
   * Binary cache only probed at time of scan.
   * More sophisticated / explicit uses of the cache
   * (i.e., replacing parts of the query subtree)
   * will only be triggered if dictated by the QO.
   */
  map<const expressions::Expression *, CacheInfo, less_map> binCaches;
  /* Answers whether the entire dataset contributes
   * to the cache, or whether some operator has
   * filtered some objects/tuples */
  map<const expressions::Expression *, bool, less_map> binCacheIsFull;

  /* From filename to (cast) PM */
  map<string, char *> pmCaches;

  CachingService() {}
  ~CachingService() {}

  // Not implementing; CachingService is a singleton
  CachingService(CachingService const &);  // Don't Implement.
  void operator=(CachingService const &);  // Don't implement.
};
#endif /* CACHING_HPP_ */
