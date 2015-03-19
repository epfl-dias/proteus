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
//	char* arenaChunk = new char[regionSize];
	char* arenaChunk = (char*) calloc(regionSize,sizeof(char));
	if(arenaChunk == NULL)	{
		string error_msg = string("[Memory Allocator: ] new() failed");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	return arenaChunk;
}

void* increaseRegion(void* region, size_t currSize)	{
	currSize <<= 1;
//	cout << "Doubling Arena to " << currSize << endl;
	/* Very Hardcoded */
//	int peek = *(int*)region;
//	cout << "Peek a boo " << peek << endl;


	void* newRegion = realloc(region, currSize);
	if(newRegion != NULL)	{
		region = newRegion;
		cout << "Doubled Arena" << endl;
//		peek = *(int*)((char*)region+20);
//
//		size_t peek2 = *(size_t*)((char*)region+sizeof(int));
//		int peek3 = *(int*)((char*)region+12);
//		int peek4 = *(int*)((char*)region+16);
//		size_t peek5 = *(size_t*)((char*)region+24);
//		cout << "Peek a boo2 " << peek2 << " " << peek3 << " " << peek4 << " " << peek << " " << peek5 << endl;
		return region;
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
//	size_t peek2 = *(size_t*)((char*)region+sizeof(int));
//			int peek3 = *(int*)((char*)region+12);
//			int peek4 = *(int*)((char*)region+16);
//			int peek = *(int*)((char*)region+20);
//			size_t peek5 = *(size_t*)((char*)region+24);
//			cout << "Peek a boo2 " << peek2 << " " << peek3 << " " << peek4 << " " << peek << " " << peek5 << endl;
	free(region);
}

