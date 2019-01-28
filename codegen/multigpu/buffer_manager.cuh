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
#include "threadsafe_device_stack.cuh"
#include "threadsafe_stack.cuh"

using namespace std;

#ifndef NCUDA
__device__ __constant__
    threadsafe_device_stack<int32_t *, (int32_t *)NULL> *pool;
__device__ __constant__ int deviceId;
__device__ __constant__ void *buff_start;
__device__ __constant__ void *buff_end;
#else
constexpr threadsafe_device_stack<int32_t *, (int32_t *)NULL> *pool = nullptr;
constexpr int deviceId = 0;
constexpr void *buff_start = nullptr;
constexpr void *buff_end = nullptr;
#endif

extern int num_of_gpus;
extern int num_of_cpus;

inline int get_gpu_count();
inline int get_cpu_numa_node_count();

template <typename T>
class buffer_manager;

__global__ void release_buffer_host(void **buff, int buffs = 1);
__global__ void get_buffer_host(void **buff, int buffs = 1);

void initializeModule(CUmodule &cudaModule);

template <typename T = int32_t>
class buffer_manager {
  static_assert(std::is_same<T, int32_t>::value, "Not implemented yet");

 public:
  typedef T *buffer_t;
  typedef threadsafe_device_stack<T *, (T *)NULL> pool_t;
  typedef threadsafe_stack<T *, (T *)NULL> h_pool_t;

  static bool terminating;
  static mutex *device_buffs_mutex;
  static condition_variable *device_buffs_cv;
  static thread **device_buffs_thrds;
  static vector<T *> *device_buffs_pool;
  static T ***device_buff;
  static size_t device_buff_size;
  static size_t keep_threshold;
  static void **h_buff_start;
  static void **h_buff_end;

  static void **h_h_buff_start;
  static size_t h_size;

  static cudaStream_t *release_streams;

  static pool_t **h_d_pool;
  static threadsafe_stack<T *, (T *)NULL> **h_pool;
  static threadsafe_stack<T *, (T *)NULL> **h_pool_numa;

  static unordered_map<T *, std::atomic<int>> buffer_cache;

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
  static __device__ __forceinline__ void __release_buffer_device(T *buff) {
    if (!buff) return;
    // assert(strm == 0); //FIXME: something better ?
    // if (buff->device == deviceId) { //FIXME: remote device!
    // buff->clean();
    // __threadfence();
    if (buff >= buff_start && buff < buff_end) pool->push(buff);
    // else printf("Throwing buffer: %p\n", buff);
    // } else                          assert(false); //FIXME: IMPORTANT free
    // buffer of another device (or host)!
  }

  static __host__ __forceinline__ void __release_buffer_host(T *buff) {
    if (!buff) return;
    nvtxRangePushA("release_buffer_host");
    const auto *gpu = topology::getInstance().getGpuAddressed(buff);
    if (gpu) {
#ifndef NCUDA
      if (buff < h_buff_start[gpu->id] || buff >= h_buff_end[gpu->id]) return;

      nvtxRangePushA("release_buffer_host_devbuffer");
      set_device_on_scope d(*gpu);
      std::unique_lock<std::mutex> lock(device_buffs_mutex[gpu->id]);
      device_buffs_pool[gpu->id].push_back(buff);
      size_t size = device_buffs_pool[gpu->id].size();
      if (size > keep_threshold) {
        uint32_t devid = gpu->id;
        nvtxRangePushA("release_buffer_host_devbuffer_overflow");
        for (size_t i = 0; i < device_buff_size; ++i)
          device_buff[devid][i] = device_buffs_pool[devid][size - i - 1];
        device_buffs_pool[devid].erase(
            device_buffs_pool[devid].end() - device_buff_size,
            device_buffs_pool[devid].end());
        release_buffer_host<<<1, 1, 0, release_streams[devid]>>>(
            (void **)device_buff[devid], device_buff_size);
        gpu_run(cudaStreamSynchronize(release_streams[devid]));
        // gpu_run(cudaPeekAtLastError()  );
        // gpu_run(cudaDeviceSynchronize());
        nvtxRangePop();
      }
      device_buffs_cv[gpu->id].notify_all();
      nvtxRangePop();
#else
      assert(false);
#endif
    } else {
      nvtxRangePushA("release_buffer_host_hostbuffer");
      const auto &it = buffer_cache.find(buff);
      if (it == buffer_cache.end()) {
        nvtxRangePop(); /* release_buffer_host_hostbuffer */
        nvtxRangePop(); /* release_buffer_host */
        return;
      }
      nvtxRangePushA("release_buffer_host_actual_release");
      int occ = (it->second)--;
      if (occ == 1) {
        // assert(buff->device < 0);
        // assert(get_device(buff->data) < 0);

        const auto &topo = topology::getInstance();
        uint32_t node = topo.getCpuNumaNodeAddressed(buff)->id;
        h_pool_numa[node]->push(buff);
        // printf("%d %p %d\n", buff->device, buff->data, status[0]);
      }
      nvtxRangePop();
      nvtxRangePop();
    }
    nvtxRangePop();
  }

 public:
  static __host__ __forceinline__ bool share_host_buffer(T *buff) {
    const auto &it = buffer_cache.find(buff);
    if (it == buffer_cache.end()) return true;
    (it->second)++;
    return true;
  }

#if defined(__clang__) && defined(__CUDA__)
  static __device__ __forceinline__ void release_buffer(
      T *buff) {  //, cudaStream_t strm){
    __release_buffer_device(buff);
  }

  static __host__ __forceinline__ void release_buffer(
      T *buff) {  //, cudaStream_t strm){
    __release_buffer_host(buff);
  }
#else
  static __host__ __device__ __forceinline__ void release_buffer(
      T *buff) {  //, cudaStream_t strm){
#ifdef __CUDA_ARCH__
    __release_buffer_device(buff);
#else
    __release_buffer_host(buff);
#endif
  }
#endif

  //     static __host__ __device__ void release_buffer(buffer_t * buff){
  // #ifdef __CUDA_ARCH__
  //         release_buffer(buff, 0);
  // #else
  //         cudaStream_t strm = 0;
  //         int dev = get_device(buff);
  //         if (dev >= 0){
  //             set_device_on_score d(dev);

  //             gpu_run(cudaStreamCreateWithFlags(&strm,
  //             cudaStreamNonBlocking));
  //         }
  //         release_buffer(buff, strm);
  //         if (dev >= 0) gpu_run(cudaStreamDestroy(strm));
  // #endif
  //     }

  // static __device__ inline bool acquire_buffer_blocked_to(buffer_t** volatile
  // ret, buffer_t** replaced){
  //     assert(__ballot(1) == 1);
  //     buffer_t * outbuff = *ret;
  //     if (replaced) *replaced = NULL;
  //     while (!outbuff->may_write()){
  //         buffer_t * tmp;
  //         if (pool->try_pop(&tmp)){ //NOTE: we can not get the current buffer
  //         as answer, as it is not released yet
  //             // assert(tmp->cnt == 0);
  //             tmp->clean();
  //             __threadfence();
  //             // if (tmp == outbuff) {
  //             //     if (*ret != outbuff) release_buffer(tmp);
  //             //     // else assert(false);
  //             //     // assert(*ret != outbuff); //FIXME: Should hold! but
  //             does not!!!
  //             //     // release_buffer(tmp);
  //             // } else {
  //                 // assert(tmp != outbuff); //NOTE: we can... it may have
  //                 been released buffer_t * oldb = atomicCAS(ret, outbuff,
  //                 tmp); //FIXME: ABA problem if (oldb != outbuff)
  //                 release_buffer(tmp); else                {
  //                     // atomicSub((uint32_t *) &available_buffers, 1);
  //                     if (replaced) *replaced = outbuff;
  //                     return true;
  //                 }
  //             // }
  //         } //else if (!producers && !available_buffers) return false;
  //         outbuff = *ret;
  //     }

  static __host__ void overwrite(T *buff, const T *data, uint32_t N,
                                 cudaStream_t strm, bool blocking = true);

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
typename buffer_manager<T>::pool_t **buffer_manager<T>::h_d_pool;

template <typename T>
threadsafe_stack<T *, (T *)NULL> **buffer_manager<T>::h_pool;

template <typename T>
threadsafe_stack<T *, (T *)NULL> **buffer_manager<T>::h_pool_numa;

template <typename T>
unordered_map<T *, atomic<int>> buffer_manager<T>::buffer_cache;

template <typename T>
mutex *buffer_manager<T>::device_buffs_mutex;

template <typename T>
thread **buffer_manager<T>::device_buffs_thrds;

template <typename T>
condition_variable *buffer_manager<T>::device_buffs_cv;

template <typename T>
bool buffer_manager<T>::terminating;

template <typename T>
vector<T *> *buffer_manager<T>::device_buffs_pool;

template <typename T>
T ***buffer_manager<T>::device_buff;

template <typename T>
size_t buffer_manager<T>::device_buff_size;

template <typename T>
size_t buffer_manager<T>::keep_threshold;

template <typename T>
cudaStream_t *buffer_manager<T>::release_streams;

template <typename T>
void **buffer_manager<T>::h_buff_start;
template <typename T>
void **buffer_manager<T>::h_buff_end;

template <typename T>
void **buffer_manager<T>::h_h_buff_start;

template <typename T>
size_t buffer_manager<T>::h_size;

template <typename T>
std::thread *buffer_manager<T>::buffer_logger;
#endif /* BUFFER_MANAGER_CUH_ */