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

#ifndef BUFFER_MANAGER_CUH_
#define BUFFER_MANAGER_CUH_

#include <sched.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "topology/affinity_manager.hpp"
#include "util/threadsafe_stack.cuh"

template <typename T, T invalid_value>
class threadsafe_device_stack;

[[deprecated("Use topology")]] inline int get_gpu_count();

[[deprecated("Use topology")]] inline int get_cpu_numa_node_count();

template <typename T>
class buffer_manager;

__global__ void release_buffer_host(void **buff, int buffs = 1);
__global__ void get_buffer_host(void **buff, int buffs = 1);

void initializeModule(CUmodule &cudaModule);

template <typename T = int32_t>
class [[deprecated("Access through BlockManager")]] buffer_manager {
  static_assert(std::is_same<T, int32_t>::value, "Not implemented yet");

 public:
  typedef T *buffer_t;
  typedef threadsafe_device_stack<T *, (T *)nullptr> pool_t;
  typedef threadsafe_stack<T *, (T *)nullptr> h_pool_t;

  static constexpr size_t buffer_size = h_vector_size * sizeof(T);

  static bool terminating;
  static std::mutex *device_buffs_mutex;
  static std::condition_variable *device_buffs_cv;
  static std::thread **device_buffs_thrds;
  static std::vector<T *> *device_buffs_pool;
  static T ***device_buff;
  static size_t device_buff_size;
  static size_t keep_threshold;
  static void **h_buff_start;
  static void **h_buff_end;

  static void **h_h_buff_start;
  static size_t h_size;

  static cudaStream_t *release_streams;

  static pool_t **h_d_pool;
  static threadsafe_stack<T *, (T *)nullptr> **h_pool;
  static threadsafe_stack<T *, (T *)nullptr> **h_pool_numa;

  static std::unordered_map<T *, std::atomic<int>> buffer_cache;

  static std::thread *buffer_logger;

  static __host__ void init(size_t size = 64, size_t h_size = 64,
                            size_t buff_buffer_size = 8,
                            size_t buff_keep_threshold = 16);

  static void dev_buff_manager(int dev);

  [[deprecated]] static __host__ T *get_buffer_numa(int numa_node) {
    T *b = h_pool[numa_node]->pop();
#ifndef NDEBUG
    int old =
#endif
        buffer_cache[b]++;
    assert(old == 0);
    return b;
  }

  static __host__ T *get_buffer_numa(const topology::cpunumanode &cpu) {
    T *b = h_pool_numa[cpu.id]->pop();
#ifndef NDEBUG
    int old =
#endif
        buffer_cache[b]++;
    assert(old == 0);
    return b;
  }

#if defined(__clang__) && defined(__CUDA__)
  static __device__ T *get_buffer();
  static __host__ T *get_buffer();
#else
  static __host__ __device__ T *get_buffer();
#endif

  static __device__ T *try_get_buffer();
  //     static __device__ bool try_get_buffer2(T **ret){
  // #ifdef __CUDA_ARCH__
  //         if (pool->try_pop(ret)){
  // #else
  //         if (h_pool[sched_getcpu()]->try_pop(ret)){
  // #endif
  //             // (*ret)->clean();
  //             return true;
  //         }
  //         return false;
  //     }

  static __host__ T *h_get_buffer(int dev);

 private:
  static __device__ void __release_buffer_device(T * buff);
  static __host__ void __release_buffer_host(T * buff);

 public:
  static __host__ __forceinline__ bool share_host_buffer(T * buff) {
    const auto &it = buffer_cache.find(buff);
    if (it == buffer_cache.end()) return true;
    (it->second)++;
    return true;
  }

#if defined(__clang__) && defined(__CUDA__)
  static __device__ __forceinline__ void release_buffer(T * buff) {
    __release_buffer_device(buff);
  }

  static __host__ __forceinline__ void release_buffer(T * buff) {
    __release_buffer_host(buff);
  }
#else
  static __host__ __device__ __forceinline__ void release_buffer(T * buff) {
#ifdef __CUDA_ARCH__
    __release_buffer_device(buff);
#else
    __release_buffer_host(buff);
#endif
  }
#endif

  static __host__ void overwrite_bytes(void *buff, const void *data,
                                       size_t bytes, cudaStream_t strm,
                                       bool blocking = true);

  static __host__ void destroy();  // FIXME: cleanup...

  static __host__ void log_buffers();
};

extern "C" {
void *get_buffer(size_t bytes);
void release_buffer(void *buff);

void *get_dev_buffer();
}

extern "C" {
__device__ void dprinti(int32_t x);
__device__ void dprinti64(int64_t x);
__device__ void dprintptr(void *x);
__device__ int32_t *get_buffers();
__device__ void release_buffers(int32_t *buff);
}

template <typename T>
threadsafe_stack<T *, (T *)nullptr> **buffer_manager<T>::h_pool;

template <typename T>
threadsafe_stack<T *, (T *)nullptr> **buffer_manager<T>::h_pool_numa;

template <typename T>
std::unordered_map<T *, std::atomic<int>> buffer_manager<T>::buffer_cache;

#endif /* BUFFER_MANAGER_CUH_ */
