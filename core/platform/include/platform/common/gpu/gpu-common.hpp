/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef GPU_COMMON_HPP_
#define GPU_COMMON_HPP_

#ifndef NCUDA
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <nvToolsExt.h>
#include <nvml.h>

#include <platform/common/common.hpp>
#else
#define __device__
#define __host__
#define __constant__
#define __global__
#define __forceinline__ inline

#define CUfunction void *
#define cudaStream_t void *
#define CUmodule void *
#define cudaEvent_t void *

constexpr int cudaProfilerStart() { return 0; }
constexpr int cudaProfilerStop() { return 0; }

struct dim3 {
  int x;
  int y;
  int z;

  constexpr dim3(int x, int y, int z) : x(x), y(y), z(z) {}
  constexpr dim3(int x) : x(x), y(1), z(1) {}
};

struct alignas(16) int4 {
  int x, y, z, w;
};

inline void nvtxRangePushA(const char *x) {}
inline void nvtxRangePop() {}

#endif

#include <thread>

// #include "multigpu/src/common.cuh"
#include <cassert>
#include <cstdint>
#include <iostream>

#ifndef WARPSIZE
#define WARPSIZE (32)
#endif

// FIXME: remove when upgrading LLVM to >9
namespace proteus {
inline __device__ unsigned int __activemask() {
  unsigned int ret;
  asm volatile("activemask.b32 %0;" : "=r"(ret));
  return ret;
}
}  // namespace proteus

// extern int                                                 cpu_cnt;
// extern cpu_set_t                                          *gpu_affinity;
// extern cpu_set_t                                          *cpu_numa_affinity;
// extern int                                                *gpu_numa_node;

typedef size_t vid_t;
typedef uint32_t cid_t;
typedef uint32_t sel_t;
typedef uint32_t cnt_t;

constexpr uint32_t warp_size = WARPSIZE;

enum class gran_t { GRID, BLOCK, THREAD };

#ifndef NCUDA
#define gpu_run(ans)                               \
  do {                                             \
    if (topology::getInstance().getGpuCount() > 0) \
      gpuAssert((ans), __FILE__, __LINE__);        \
  } while (0)

[[noreturn]] __host__ inline void gpuAssert(const std::string &str,
                                            const char *file, int line) {
  auto msg = std::string{"GPUassert: "} + str;
  google::LogMessage(file, line, google::GLOG_ERROR).stream() << msg;
  throw std::runtime_error{msg};
}

__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file,
                                          int line) {
  if (code != cudaSuccess) {
#ifndef __CUDA_ARCH__
    gpuAssert(cudaGetErrorString(code), file, line);
#else
    printf("GPUassert: %s %s %d\n", "error", file, line);
    exit(code);
#endif
  }
}

__host__ __device__ inline void gpuAssert(CUptiResult code, const char *file,
                                          int line) {
  if (code != CUPTI_SUCCESS) {
#ifndef __CUDA_ARCH__
    const char *errstr;
    if (cuptiGetResultString(code, &errstr)) {
      gpuAssert(errstr, file, line);
    } else {
      gpuAssert("Unknown CUPTI error", file, line);
    }
#else
    printf("GPUassert: %s %s %d\n", "error", file, line);
    exit(code);
#endif
  }
}

__host__ __device__ inline void gpuAssert(CUresult code, const char *file,
                                          int line) {
  if (code != CUDA_SUCCESS) {
#ifndef __CUDA_ARCH__
    const char *msg;
    cuGetErrorString(code, &msg);
    gpuAssert(msg, file, line);
#else
    printf("GPUassert: %s %s %d\n", "error", file, line);
    exit(code);
#endif
  }
}

__host__ __device__ inline void gpuAssert(nvmlReturn_enum code,
                                          const char *file, int line) {
  if (code != NVML_SUCCESS) {
#ifndef __CUDA_ARCH__
    gpuAssert(nvmlErrorString(code), file, line);
#else
    printf("GPUassert: %s %s %d\n", "error", file, line);
    exit(code);
#endif
  }
}

extern "C" void memcpy_gpu(void *dst, const void *src, size_t size,
                           bool is_volatile = false);

// __host__ __device__ inline void gpuAssert(nvrtcResult code, const char *file,
// int line, bool abort=true){
//     if (code != NVML_SUCCESS) {
// #ifndef __CUDA_ARCH__
//         const char * msg = nvrtcGetErrorString(code);
//         fprintf(stderr,"GPUassert: %s %s %d\n", msg, file, line);
//         if (abort) exit(code);
// #else
//         printf("GPUassert: %s %s %d\n", "error", file, line);
// #endif
//     }
// }

// inline int get_num_of_gpus(){
//     int devices;
//     gpu_run(cudaGetDeviceCount(&devices));
//     return devices;
// }
#else
#define gpu_run(ans)
// inline constexpr int get_num_of_gpus() {return 0;}
#endif

// int get_device(const void *p);
// inline int get_device(){
// #ifndef NCUDA
//     int device;
//     gpu_run(cudaGetDevice(&device));
//     return device;
// #else
//     return 0;
// #endif
// }

// inline int get_current_gpu(){
//     return get_device();
// }

std::ostream &operator<<(std::ostream &out, const cpu_set_t &cpus);

#ifndef NCUDA
__device__ __forceinline__ uint32_t get_laneid() {
  uint32_t laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

// Handle missmatch of atomics for (u)int64/32_t with cuda's definitions
template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                                  int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val) {
  return (T)atomicExch((unsigned long long int *)address,
                       (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val) {
  return (T)atomicExch((unsigned int *)address, (unsigned int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                                  int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val) {
  return (T)atomicExch_block((unsigned long long int *)address,
                             (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val) {
  return (T)atomicExch_block((unsigned int *)address, (unsigned int)val);
}

template <typename T, typename std::enable_if<sizeof(T) == sizeof(int) &&
                                                  std::is_signed<T>::value,
                                              int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val) {
  return (T)atomicExch((int *)address, (int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val) {
  return (T)atomicMin((unsigned long long int *)address,
                      (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val) {
  return (T)atomicMin((unsigned int *)address, (unsigned int)val);
}

template <typename T, typename std::enable_if<sizeof(T) == sizeof(int) &&
                                                  std::is_integral<T>::value &&
                                                  std::is_signed<T>::value,
                                              int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val) {
  return (T)atomicMin((int *)address, (int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val) {
  return (T)atomicMin_block((unsigned long long int *)address,
                            (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val) {
  return (T)atomicMin_block((unsigned int *)address, (unsigned int)val);
}

template <typename T, typename std::enable_if<sizeof(T) == sizeof(int) &&
                                                  std::is_integral<T>::value &&
                                                  std::is_signed<T>::value,
                                              int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val) {
  return (T)atomicMin_block((int *)address, (int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val) {
  return (T)atomicAdd((unsigned long long int *)address,
                      (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val) {
  return (T)atomicAdd((unsigned int *)address, (unsigned int)val);
}

template <typename T, typename std::enable_if<sizeof(T) == sizeof(int) &&
                                                  std::is_integral<T>::value &&
                                                  std::is_signed<T>::value,
                                              int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val) {
  return (T)atomicAdd((int *)address, (int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val) {
  return (T)atomicAdd_block((unsigned long long int *)address,
                            (unsigned long long int)val);
}

template <typename T,
          typename std::enable_if<sizeof(T) == sizeof(unsigned int) &&
                                      std::is_integral<T>::value &&
                                      !std::is_signed<T>::value,
                                  int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val) {
  return (T)atomicAdd_block((unsigned int *)address, (unsigned int)val);
}

template <typename T, typename std::enable_if<sizeof(T) == sizeof(int) &&
                                                  std::is_integral<T>::value &&
                                                  std::is_signed<T>::value,
                                              int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val) {
  return (T)atomicAdd_block((int *)address, (int)val);
}
#endif

dim3 findDefaultGridDim();

const dim3 defaultBlockDim(1024, 1, 1);
extern dim3 defaultGridDim;

struct execution_conf {
  dim3 gridDim;
  dim3 blockDim;

  execution_conf(dim3 gridDim = defaultGridDim, dim3 blockDim = defaultBlockDim)
      : gridDim(gridDim), blockDim(blockDim) {}

  size_t gridSize() const {
    return ((size_t)gridDim.x) * gridDim.y * gridDim.z;
  }

  size_t blockSize() const {
    return ((size_t)blockDim.x) * blockDim.y * blockDim.z;
  }

  size_t threadNum() const { return blockSize() * gridSize(); }
};

void launch_kernel(CUfunction function, void **args, dim3 gridDim,
                   dim3 blockDim, cudaStream_t strm = nullptr);
void launch_kernel(CUfunction function, void **args, dim3 gridDim,
                   cudaStream_t strm = nullptr);
void launch_kernel(CUfunction function, void **args,
                   cudaStream_t strm = nullptr);
extern "C" void launch_kernel_strm(CUfunction function, void **args,
                                   cudaStream_t strm);

extern "C" {
int get_ptr_device(const void *p);
int get_ptr_device_or_rand_for_host(const void *p);
int get_rand_core_local_to_ptr(const void *p);
}

cudaStream_t createNonBlockingStream();
void syncAndDestroyStream(cudaStream_t strm);

#endif /* GPU_COMMON_HPP_ */
