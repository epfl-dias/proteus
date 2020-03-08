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

#ifndef THREADSAFE_DEVICE_STACK_CUH_
#define THREADSAFE_DEVICE_STACK_CUH_

#include <cstdint>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <vector>

#include "common/gpu/gpu-common.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

/**
 * Based on a combination of :
 * * https://github.com/memsql/lockfree-bench/blob/master/stack/lockfree.h
 * * (use with care)
 * http://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
 * * http://www.boost.org/doc/libs/1_60_0/boost/lockfree/detail/freelist.hpp
 * * http://blog.memsql.com/common-pitfalls-in-writing-lock-free-algorithms/
 */
template <typename T, T invalid_value = std::numeric_limits<T>::max()>
class threadsafe_device_stack {  // FIXME: must have a bug
 private:
  volatile uint32_t cnt;
  const uint32_t size;

  // int          lock;      //consider changing the orde of parameters to :
  // data, cnt, lock (128bits) to load the important part of the DS
  /*volatile if any op is not atomic*/ int
      lock;  // consider changing the orde of parameters to : data, cnt, lock
             // (128bits) to load the important part of the DS
  volatile T *data;  // OR cnt, size, data, lock to load the non-atomic part of
                     // the DS with a single 128bit load

 public:
  __host__ threadsafe_device_stack(uint32_t size, std::vector<T> fill,
                                   const topology::gpunode &device)
      : cnt(0), size(size), lock(0) {
    set_device_on_scope d{device};
    gpu_run(cudaMalloc(&data, size * sizeof(T)));

    if (fill.size() > 0) {
      gpu_run(cudaMemcpy(const_cast<T *>(data), fill.data(),
                         fill.size() * sizeof(T), cudaMemcpyDefault));
      cnt = fill.size();
    }
  }

  __host__ ~threadsafe_device_stack() {
    // void ** t = (void **) malloc(cnt * sizeof(T));
    // gpu(cudaMemcpy(t, (void *) data, cnt*sizeof(T), cudaMemcpyDefault));
    // for (int i = 0 ; i < cnt ; ++i){
    //     printf("%p ", t[i]);
    // }
    // printf("\n");

    gpu_run(cudaFree(const_cast<T *>(data)));

    if (cnt) {
      int dev = topology::getInstance().getActiveGpu().id;
      LOG(INFO) << "------------------------------------------------------->dev"
                << dev << "_stack: " << cnt << " " << size << std::endl;
    }
    // assert(cnt == size);
  }

 public:
#ifndef NCUDA
  __device__ __forceinline__ void push(T v) {
    // assert(__popc(__ballot(1)) == 1);

    while (atomicCAS(const_cast<int *>(&lock), 0, 1) != 0)
      ;

    // assert(cnt < size);
    data[cnt++] = v;
    // printf("pushed %p\n", v);

    // lock = 0;       //FIXME: atomicExch() is probably needed here. This
    // should have undef behavior
    __threadfence();

    atomicExch(const_cast<int *>(&lock), 0);
  }

  __device__ __forceinline__ bool try_pop(T *ret) {
    // assert(__popc(__ballot(1)) == 1);

    if (atomicCAS(const_cast<int *>(&lock), 0, 1) != 0) return false;

    if (cnt == 0) {
      // lock = 0;   //FIXME: atomicExch() is probably needed here. This should
      // have undef behavior
      atomicExch(const_cast<int *>(&lock), 0);
      // printf("popped empty\n");
      return false;
    }

    *ret = data[--cnt];
    // printf("popped %p\n", *ret);

    // lock = 0;       //FIXME: atomicExch() is probably needed here. This
    // should have undef behavior

    __threadfence();

    atomicExch(const_cast<int *>(&lock), 0);

    return true;
  }

  __device__ __forceinline__ bool pop_if_nonempty(T *ret) {  // blocking
    // assert(__popc(__ballot(1)) == 1);

    if (cnt == 0) return false;

    while (atomicCAS(const_cast<int *>(&lock), 0, 1) != 0)
      ;

    if (cnt == 0) {
      // lock = 0;   //FIXME: atomicExch() is probably needed here. This should
      // have undef behavior
      atomicExch(const_cast<int *>(&lock), 0);
      // printf("popped empty\n");
      return false;
    }

    *ret = data[--cnt];
    // printf("popped %p\n", *ret);

    // lock = 0;       //FIXME: atomicExch() is probably needed here. This
    // should have undef behavior

    __threadfence();

    atomicExch(const_cast<int *>(&lock), 0);

    return true;
  }

  __device__ __forceinline__ T pop() {  // blocking
    // assert(__popc(__ballot(1)) == 1);
    T ret;
    while (!try_pop(&ret))
      ;
    // printf("popped %p\n", ret);
    return ret;
  }

  __host__ __device__ __forceinline__ static bool is_valid(T &x) {
    return x != invalid_value;
  }

  __host__ __device__ __forceinline__ static T get_invalid() {
    return invalid_value;
  }
#else
  __device__ __forceinline__ void push(T v) { assert(false); }

  __device__ __forceinline__ bool try_pop(T *ret) {
    assert(false);
    return true;
  }

  __device__ __forceinline__ T pop() {  // blocking
    assert(false);
    T ret;
    return ret;
  }
#endif

  template <typename U>
  friend class buffer_manager;  // std::remove_pointer<T>>;
};

#endif /* THREADSAFE_DEVICE_STACK_CUH_ */
