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

#ifndef RAW_CACHING_HPP_
#define RAW_CACHING_HPP_

#include "common/common.hpp"
#include "expressions/expressions.hpp"

#define CACHING_ON
//#define DEBUGCACHING

using expressions::less_map;

typedef struct CacheInfo {
	/* XXX Issue: Types depend on context
	 * Do I need to share the context for this to work?
	 * One option is mapping LLVM Types to an enum -
	 * But what about custom types?
	 * An option is that everything considered
	 * a custom / structType will be stored in CATALOG too */
	//StructType *objectType;
	list<typeID> objectTypes;
	/* Convention:
	 * Count begins from 1.
	 * Zero implies we're dealing with the whole obj. (activeLoop)
	 * Negative implies invalid entry */
	int structFieldNo;
	//Pointers to facilitate LLVM storing stuff in them
	char **payloadPtr;
	size_t *itemCount;
} CacheInfo;
class CachingService
{
public:

	static CachingService& getInstance()
	{
		static CachingService instance;
		return instance;
	}

	void registerPM(string fileName, char *payloadPtr) {
		map<string, char*>::iterator it = pmCaches.find(fileName);
		if (it != pmCaches.end()) {
			LOG(WARNING)<< "PM caches already contain " << fileName;
		}
		pmCaches[fileName] = payloadPtr;
	}

	char* getPM(string fileName) {
		map<string, char*>::iterator it = pmCaches.find(fileName);
		if (it == pmCaches.end()) {
			LOG(INFO)<< "No PM found for " << fileName;
			/* NULL is a valid value (PM not found) */
			return NULL;
		}
		return it->second;
	}

	void registerCache(expressions::Expression* expr, CacheInfo payload, bool entireDataset) {
#ifdef CACHING_ON
		map<expressions::Expression*, CacheInfo>::iterator it = binCaches.find(expr);
		bool found = false;
		map<expressions::Expression*, bool>::iterator itBool;
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
			 * 	  certain sequence to emulate caching effects
			 */
			if (entireDataset /*&& !(itBool->second)*/) {
				binCaches.erase(expr);
				binCacheIsFull.erase(expr);
				binCaches[expr] = payload;
				binCacheIsFull[expr] = entireDataset;
				//cout << "Replaced in cache " << endl;
			}

		}
		else
		{
			binCaches[expr] = payload;
			binCacheIsFull[expr] = entireDataset;
#ifdef DEBUGCACHING
			cout << "Registered in cache " << endl;
#endif
		}
#endif
	}

	CacheInfo getCache(expressions::Expression* expr) {
		map<expressions::Expression*, CacheInfo>::iterator it = binCaches.find(expr);
		if (it == binCaches.end()) {
			//cout << "No match out of " << binCaches.size() << " entries" << endl;
			LOG(INFO)<< "No Bin Cache found for expr of type " << expr->getExpressionType()->getType();
			CacheInfo invalid;
			invalid.structFieldNo = -1;
			return invalid;
		}
		return it->second;
	}

	bool getCacheIsFull(expressions::Expression* expr) {
		map<expressions::Expression*, bool>::iterator it = binCacheIsFull.find(expr);
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

private:
	/*
	 * From expressions::Expression to (cast) cache.
	 * Binary cache only probed at time of scan.
	 * More sophisticated / explicit uses of the cache
	 * (i.e., replacing parts of the query subtree)
	 * will only be triggered if dictated by the QO.
	 */
	map<expressions::Expression*, CacheInfo, less_map> binCaches;
	/* Answers whether the entire dataset contributes
	 * to the cache, or whether some operator has
	 * filtered some objects/tuples */
	map<expressions::Expression*, bool, less_map> binCacheIsFull;

	/* From filename to (cast) PM */
	map<string, char*> pmCaches;

	CachingService()  {}
	~CachingService() {}

	//Not implementing; CachingService is a singleton
	CachingService(CachingService const&); // Don't Implement.
	void operator=(CachingService const&); // Don't implement.
};
#endif /* RAW_CACHING_HPP_ */
