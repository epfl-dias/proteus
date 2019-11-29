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

#ifndef BLOCK_MANAGER_HPP
#define BLOCK_MANAGER_HPP

#include "buffer-manager.cuh"
#include "util/memory-registry.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
class BlockManager : public buffer_manager<int32_t> {
 public:
  static constexpr size_t block_size = buffer_manager<int32_t>::buffer_size;

  static __host__ __device__ __forceinline__ void release_buffer(void* buff) {
    buffer_manager<int32_t>::release_buffer((int32_t*)buff);
  }
#pragma clang diagnostic pop

  static void reg(MemoryRegistry&);
  static void unreg(MemoryRegistry& registry);
};
#endif /* BLOCK_MANAGER_HPP */
