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

#ifndef STORAGE_MEMORY_MANAGER_HPP_
#define STORAGE_MEMORY_MANAGER_HPP_

#include <sys/mman.h>


namespace storage::memory {

struct mem_chunk {
  void* data;
  const size_t size;
  const int numa_id;

  mem_chunk() : data(nullptr), size(0), numa_id(-1) {}

  mem_chunk(void* data, size_t size, int numa_id)
      : data(data), size(size), numa_id(numa_id) {}
};

class MemoryManager {
 public:
  static void* alloc(size_t bytes, int numa_memset_id,
                     int mem_advice = MADV_DOFORK | MADV_HUGEPAGE);
  static void free(void* mem);
};

};  // namespace storage

#endif /* STORAGE_MEMORY_MANAGER_HPP_ */
