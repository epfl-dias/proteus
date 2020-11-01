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

#ifndef OLTP_COMMON_MEMORY_CHUNK_HPP_
#define OLTP_COMMON_MEMORY_CHUNK_HPP_

#include <sys/mman.h>

#include <new>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

namespace oltp::common {

class mem_chunk {
 public:
  void *data;
  const size_t size;
  const int numa_id;

  explicit mem_chunk() : data(nullptr), size(0), numa_id(-1) {}

  explicit mem_chunk(void *data, size_t size, int numa_id)
      : data(data), size(size), numa_id(numa_id) {}
};

}  // namespace oltp::common

#endif /* OLTP_COMMON_MEMORY_CHUNK_HPP_ */
