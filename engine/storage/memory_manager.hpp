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

#include <map>
#include <vector>

namespace storage {

struct mem_chunk {
  void* data;
  size_t size;
  int numa_id;

  // latching or locking here?
  mem_chunk() : data(nullptr), size(0), numa_id(-1) {}

  mem_chunk(void* data, size_t size, int numa_id)
      : data(data), size(size), numa_id(numa_id) {}
};
//  mlock() ???
class MemoryManager {
 public:
  static void init();
  static void destroy();

  static void* alloc_shm_htap(const std::string& key, const size_t size_bytes,
                              const size_t unit_size, const int numa_memset_id);
  static void remove_shm_htap(const std::string& key);

  // Allocation should be managed  and linked with affinities and topology
  static void* alloc_shm(const std::string& key, const size_t size_bytes,
                         const int numa_memset_id);
  static void remove_shm(const std::string& key);

  static void* alloc(size_t bytes, int numa_memset_id);
  static void free(void* mem, size_t bytes);
};

/*class MemoryManager {
 public:
  static void init();
  static void destory();

  static mem_chunk* malloc(size_t bytes);
  static void free(mem_chunk* chunk);
};

class NUMAPinnedMemAllocator {};

}  // namespace storage*/

};  // namespace storage

#endif /* STORAGE_MEMORY_MANAGER_HPP_ */
