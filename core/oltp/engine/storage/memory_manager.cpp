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

#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>

#include "memory/memory-manager.hpp"
#include "scheduler/topology.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

/*
  TODO:
    malloc option
    shared memory option but still allowing pinning
    not own allocator but requesting a third-party allocator over UNIX mqueues.

*/

namespace storage {

// #define QUIT fail(__FILE__, __LINE__ )
// int fail(char *filename, int linenumber) {
//   fprintf(stderr, "%s:%d %s\n", filename, linenumber, strerror(errno));
//   exit(1);
//   return 0; /*Make compiler happy */
// }


void MemoryManager::remove_shm(const std::string& key) {
  // int munmap(void *addr, size_t length);

  int ret = shm_unlink(key.c_str());

  if (ret != 0) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
  }
}

void* MemoryManager::alloc_shm(const std::string& key, const size_t size_bytes,
                               const int numa_memset_id) {
  // std::cout << "[MemoryManager::alloc_shm] key: "<< key << std::endl;
  // std::cout << "[MemoryManager::alloc_shm] size_bytes: "<< size_bytes <<
  // std::endl; std::cout << "[MemoryManager::alloc_shm] numa_memset_id: "<<
  // numa_memset_id << std::endl;

  // assert(key.length() <= 255 && key[0] == '/');

  int shm_fd = shm_open(key.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (shm_fd == -1) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return nullptr;
  }

  if (ftruncate(shm_fd, size_bytes) < 0) {  //== -1){
    shm_unlink(key.c_str());
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return nullptr;
  }

  void* mem_addr =
      mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
  if (!mem_addr) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    return nullptr;
  }

  if (numa_memset_id != -1) {
    // move memory pages to that location.

    // numa_get_mems_allowed
    const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
    numa_tonode_memory(mem_addr, size_bytes, vec[numa_memset_id].id);
  }

  close(shm_fd);
  return mem_addr;
}

void MemoryManager::init() {
  std::cout << "[MemoryManager::init] --BEGIN--" << std::endl;
  std::cout << "[MemoryManager::init] --END--" << std::endl;
}
void MemoryManager::destroy() {
  std::cout << "[MemoryManager::destroy] --BEGIN--" << std::endl;
  std::cout << "[MemoryManager::destroy] --END--" << std::endl;
}
void* MemoryManager::alloc(size_t bytes, int numa_memset_id, int mem_advice) {
  // const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
  // assert(numa_memset_id < vec.size());
  // void* ret = numa_alloc_onnode(bytes, vec[numa_memset_id].id);

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();
  set_exec_location_on_scope d{nodes[numa_memset_id]};
  void* ret = ::MemoryManager::mallocPinned(bytes);

  // assert(topo.getCpuNumaNodeAddressed(ret)->id == nodes[numa_memset_id].id);

  // if (madvise(ret, bytes, mem_advice) == -1) {
  //   fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
  //   assert(false);
  //   return nullptr;
  // }

  return ret;
  // return numa_alloc_interleaved(bytes);
}
void MemoryManager::free(void* mem) {  // numa_free(mem, bytes);
  ::MemoryManager::freePinned(mem);
}

};  // namespace storage
