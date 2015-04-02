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

class CachingService
{
public:

	static CachingService& getInstance()
	{
		static CachingService instance;
		return instance;
	}

private:

	CachingService();
	~CachingService() { }

	//Not implementing; CachingService is a singleton
	CachingService(CachingService const&);     // Don't Implement.
	void operator=(CachingService const&); // Don't implement
};
#endif /* RAW_CACHING_HPP_ */
