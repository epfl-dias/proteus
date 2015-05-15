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

#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include "common/common.hpp"

void* allocateFromRegion(size_t regionSize);
void* increaseRegion(void *region, size_t currSize);
void freeRegion(void *region);

class MemoryService
{
public:

	static MemoryService& getInstance()
	{
		static MemoryService instance;
		return instance;
	}

	void registerChunk(void* mem_chunk) {
		memoryChunks.push_back(mem_chunk);
	}

	void updateChunk(void *mem_before, void *mem_after) {
		vector<void*>::iterator it = memoryChunks.begin();
		for (; it != memoryChunks.end(); it++) {
			void* currChunk = (*it);
			if (currChunk == mem_before)
				memoryChunks.erase(it);
			memoryChunks.push_back(mem_after);
			break;
		}
	}

	void clear() {
		vector<void*>::iterator it = memoryChunks.begin();
		for (; it != memoryChunks.end(); it++) {
			void* currChunk = (*it);
			free(currChunk);
		}
		memoryChunks.clear();
	}
private:
	vector<void*> memoryChunks;
	MemoryService()  {}
	~MemoryService() {}

	//Not implementing; MemoryService is a singleton
	MemoryService(MemoryService const&); // Don't Implement.
	void operator=(MemoryService const&); // Don't implement.
};
#endif /* MEMORY_ALLOCATOR_HPP_ */
