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

#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include <mutex>

#include "common/common.hpp"

/**
 * TODO
 * Pre-allocating memory and providing it when requested
 * will speed things up.
 *
 * At the moment, memory is allocated at the time of the
 * request.
 */
void *allocateFromRegion(size_t regionSize);
void *increaseRegion(void *region, size_t currSize);
void freeRegion(void *region);

class MemoryService {
 public:
  static MemoryService &getInstance() {
    static MemoryService instance;
    return instance;
  }

  void registerChunk(void *mem_chunk) {
    std::lock_guard<std::mutex> lock(m);
    //        cout << "Registering " << mem_chunk << endl;
    memoryChunks.push_back(mem_chunk);
  }

  void removeChunk(void *mem_chunk) {
    std::lock_guard<std::mutex> lock(m);
    //        cout << "Removing " << mem_chunk << endl;

    vector<void *>::iterator it = memoryChunks.begin();
    for (; it != memoryChunks.end(); it++) {
      void *currChunk = (*it);
      if (currChunk == mem_chunk) {
        memoryChunks.erase(it);
        return;
      }
    }
    string error_msg =
        string("[MemoryService: ] Unknown memory chunk to be freed");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void updateChunk(void *mem_before, void *mem_after) {
    removeChunk(mem_before);
    registerChunk(mem_after);
    // vector<void*>::iterator it = memoryChunks.begin();
    // for (; it != memoryChunks.end(); it++) {
    //     void* currChunk = (*it);
    //     if (currChunk == mem_before)
    //         memoryChunks.erase(it);
    //     memoryChunks.push_back(mem_after);
    //     break;
    // }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(m);
    vector<void *>::iterator it = memoryChunks.begin();
    for (; it != memoryChunks.end(); it++) {
      void *currChunk = (*it);
      //            cout << "Freeing " << currChunk << endl;
      free(currChunk);
    }
    memoryChunks.clear();
  }

 private:
  vector<void *> memoryChunks;
  std::mutex m;

  MemoryService() {}
  ~MemoryService() {}

  // Not implementing; MemoryService is a singleton
  MemoryService(MemoryService const &);   // Don't Implement.
  void operator=(MemoryService const &);  // Don't implement.
};
#endif /* MEMORY_ALLOCATOR_HPP_ */
