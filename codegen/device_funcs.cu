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

#include <cinttypes>
#include <cstdio>
#include "common/gpu/gpu-common.hpp"
#include "multigpu/buffer_manager.cuh"

extern "C" {

__device__ void printi64(int64_t x) { printf("%" PRId64 "\n", x); }

__device__ int32_t *get_buffers() {
  uint32_t b = __ballot(1);
  uint32_t m = 1 << get_laneid();
  int32_t *ret;
  do {
    uint32_t leader = b & -b;

    if (leader == m) ret = buffer_manager<int32_t>::get_buffer();

    b ^= leader;
  } while (b);
  return ret;
}

__device__ void release_buffers(int32_t *buff) {
  uint32_t b = __ballot(1);
  uint32_t m = 1 << get_laneid();
  int32_t *ret;
  do {
    uint32_t leader = b & -b;

    if (leader == m) buffer_manager<int32_t>::release_buffer(buff);

    b ^= leader;
  } while (b);
}
}