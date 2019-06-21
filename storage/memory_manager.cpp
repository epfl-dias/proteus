/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#include "storage/memory_manager.hpp"
#include <numa.h>
#include <numaif.h>
#include <iostream>
#include "scheduler/topology.hpp"

/*
  TODO:
    malloc option
    shared memory option but still allowing pinning
    not own allocator but requesting a third-party allocator over UNIX mqueues.

*/

namespace storage {


void MemoryManager::init() {
  std::cout << "[MemoryManager::init] --BEGIN--" << std::endl;
  std::cout << "[MemoryManager::init] --END--" << std::endl;
}
void MemoryManager::destroy() {
  std::cout << "[MemoryManager::destroy] --BEGIN--" << std::endl;
  std::cout << "[MemoryManager::destroy] --END--" << std::endl;
}
void* MemoryManager::alloc(size_t bytes, int numa_memset_id) {
  // std::cout << "[MemoryManager::alloc] --BEGIN--" << std::endl;
  return numa_alloc_onnode(bytes, numa_memset_id);
}
void MemoryManager::free(void* mem, size_t bytes) {
  // std::cout << "[MemoryManager::free] --BEGIN--" << std::endl;
  numa_free(mem, bytes);
  // std::cout << "[MemoryManager::free] --END--" << std::endl;
}

};  // namespace storage
