/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef STORAGE_MEMORY_MANAGER_HPP_
#define STORAGE_MEMORY_MANAGER_HPP_

#include <sys/mman.h>

#include <new>

#include "memory/memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

namespace storage::memory {

class MemoryManager {
 public:
  static void *alloc(size_t bytes, int numa_memset_id,
                     int mem_advice = MADV_DOFORK | MADV_HUGEPAGE) {
    void *ret = nullptr;

    // const auto& vec = scheduler::Topology::getInstance().getCpuNumaNodes();
    // assert(numa_memset_id < vec.size());
    // void* ret = numa_alloc_onnode(bytes, vec[numa_memset_id].id);

    // if (madvise(ret, bytes, mem_advice) == -1) {
    //   fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    //   assert(false);
    //   return nullptr;
    // }

    // return numa_alloc_interleaved(bytes);

    if (numa_memset_id >= 0) {
      static const auto &nodes = topology::getInstance().getCpuNumaNodes();
      set_exec_location_on_scope d{nodes[numa_memset_id]};
      ret = ::MemoryManager::mallocPinned(bytes);
    } else {
      ret = ::MemoryManager::mallocPinned(bytes);
    }
    return ret;
  }

  static void free(void *mem) {
    // numa_free(mem, bytes);
    ::MemoryManager::freePinned(mem);
  }
};

struct mem_chunk {
  void *data;
  const size_t size;
  const int numa_id;

  mem_chunk() : data(nullptr), size(0), numa_id(-1) {}

  mem_chunk(void *data, size_t size, int numa_id)
      : data(data), size(size), numa_id(numa_id) {}
};

};  // namespace storage::memory

#endif /* STORAGE_MEMORY_MANAGER_HPP_ */
