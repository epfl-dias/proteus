/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
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

#ifndef GPU_COMMON_HPP_
#define GPU_COMMON_HPP_

// #include "common/common.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvml.h"
#include <numaif.h>
#include <numa.h>
#include <thread>

// #include "multigpu/src/common.cuh"
#include <cstdint>
#include <iostream>

#ifndef WARPSIZE
#define WARPSIZE (32)
#endif

#ifndef DEFAULT_BUFF_CAP
#define DEFAULT_BUFF_CAP (4*1024*1024)
#endif

extern int                                                 cpu_cnt;
extern cpu_set_t                                          *gpu_affinity;
extern cpu_set_t                                          *cpu_numa_affinity;
extern int                                                *gpu_numa_node;

typedef size_t   vid_t;
typedef uint32_t cid_t;
typedef uint32_t sel_t;
typedef uint32_t cnt_t;

constexpr uint32_t warp_size     =         WARPSIZE;
constexpr cnt_t    vector_size   =   32*4*warp_size;
constexpr cnt_t    h_vector_size = DEFAULT_BUFF_CAP;

enum class gran_t {
    GRID,
    BLOCK,
    THREAD
};

#define gpu_run(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
#ifndef __CUDA_ARCH__
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
#else
        printf("GPUassert: %s %s %d\n", "error", file, line);
#endif
    }
}

__host__ __device__ inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true){
    if (code != CUDA_SUCCESS) {
#ifndef __CUDA_ARCH__
        const char * msg;
        cuGetErrorString(code, &msg);
        fprintf(stderr,"GPUassert: %s %s %d\n", msg, file, line);
        if (abort) exit(code);
#else
        printf("GPUassert: %s %s %d\n", "error", file, line);
#endif
    }
}

__host__ __device__ inline void gpuAssert(nvmlReturn_t code, const char *file, int line, bool abort=true){
    if (code != NVML_SUCCESS) {
#ifndef __CUDA_ARCH__
        const char * msg = nvmlErrorString(code);
        fprintf(stderr,"GPUassert: %s %s %d\n", msg, file, line);
        if (abort) exit(code);
#else
        printf("GPUassert: %s %s %d\n", "error", file, line);
#endif
    }
}

// __host__ __device__ inline void gpuAssert(nvrtcResult code, const char *file, int line, bool abort=true){
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

inline int get_num_of_gpus(){
    int devices;
    gpu_run(cudaGetDeviceCount(&devices));
    return devices;
}

inline int get_current_gpu(){
    int device;
    gpu_run(cudaGetDevice(&device));
    return device;
}

std::ostream& operator<<(std::ostream& out, const cpu_set_t& cpus);

class exec_location{
private:
    int         gpu_device;
    cpu_set_t   cpus;

public:
    exec_location(){
        gpu_run(cudaGetDevice(&gpu_device));

        CPU_ZERO(&cpus);
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpus);
    }

    exec_location(int gpu): gpu_device(gpu){
        cpus = gpu_affinity[gpu_device];
    }

    exec_location(int gpu, cpu_set_t cpus): gpu_device(gpu), cpus(cpus){
    }

    exec_location(cpu_set_t cpus): cpus(cpus){
        gpu_device = -1;
    }

public:
    void activate() const{
        // std::cout << "d" << gpu_device << " " << cpus << std::endl;
        if (gpu_device >= 0) gpu_run(cudaSetDevice(gpu_device));
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpus);
    }

};

class set_device_on_scope{
private:
    int device;
public:
    inline set_device_on_scope(int set){
        gpu_run(cudaGetDevice(&device));
        if (set >= 0) gpu_run(cudaSetDevice(set));
    }

    inline ~set_device_on_scope(){
        gpu_run(cudaSetDevice(device));
    }
};


class set_exec_location_on_scope{
private:
    exec_location old;
public:
    inline set_exec_location_on_scope(int gpu){
        exec_location{gpu}.activate();
    }

    inline set_exec_location_on_scope(int gpu, cpu_set_t cpus){
        exec_location{gpu, cpus}.activate();
    }

    inline set_exec_location_on_scope(const exec_location &loc){
        loc.activate();
    }

    inline ~set_exec_location_on_scope(){
        old.activate();
    }
};

__device__ __forceinline__ uint32_t get_laneid(){
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

// Handle missmatch of atomics for (u)int64/32_t with cuda's definitions
template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned int*) address, (unsigned int) val);
}


template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((int*) address, (int) val);
}



const dim3 defaultBlockDim(1024, 1, 1);
const dim3 defaultGridDim (  40, 1, 1);

struct execution_conf {
    dim3 gridDim ;
    dim3 blockDim;

    execution_conf(dim3 gridDim = defaultGridDim, dim3 blockDim = defaultBlockDim):
            gridDim(gridDim), blockDim(blockDim){}

    size_t gridSize() const{
        return ((size_t) gridDim.x) * gridDim.y * gridDim.z;
    }
    
    size_t blockSize() const{
        return ((size_t) blockDim.x) * blockDim.y * blockDim.z;
    }

    size_t threadNum() const{
        return blockSize() * gridSize();
    }
};

void launch_kernel(CUfunction function, void ** args, dim3 gridDim, dim3 blockDim, cudaStream_t strm = 0);
void launch_kernel(CUfunction function, void ** args, dim3 gridDim, cudaStream_t strm = 0);
void launch_kernel(CUfunction function, void ** args, cudaStream_t strm = 0);

int get_device(const void *p);
inline int get_device(){
    int device;
    gpu_run(cudaGetDevice(&device));
    return device;
}

extern "C"{
    int get_ptr_device(const void *p);
    int get_ptr_device_or_rand_for_host(const void *p);
}

#endif /* GPU_COMMON_HPP_ */