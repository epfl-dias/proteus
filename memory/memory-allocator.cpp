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

#include "memory/memory-allocator.hpp"

void* allocateFromRegion(size_t regionSize)	{
	char* arenaChunk = (char*) calloc(regionSize,sizeof(char));
	if(arenaChunk == NULL)	{
		string error_msg = string("[Memory Allocator: ] new() failed");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	LOG(INFO) << "Region Allocated" << endl;
	MemoryService& mem = MemoryService::getInstance();
	mem.registerChunk(arenaChunk);
	return arenaChunk;
}

void* increaseRegion(void* region, size_t currSize)	{
	cout << "Realloc()" << endl;
	LOG(INFO) << "Realloc()";
	currSize <<= 1;
	void* newRegion = realloc(region, currSize);
	if(newRegion != NULL)	{
		MemoryService& mem = MemoryService::getInstance();
		mem.updateChunk(region,newRegion);
		return newRegion;
	}
	else
	{
		free(region);
		string error_msg = string("[Memory Allocator: ] realloc() failed");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
}

void freeRegion(void* region)	{
	free(region);
}

