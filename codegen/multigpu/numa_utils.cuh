#ifndef NUMA_UTILS_CUH_
#define NUMA_UTILS_CUH_

// #include "buffer_manager.cuh"
#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include <thread>
#include <numaif.h>
#include <numa.h>

// #define NNUMA

#define numa_assert(x) assert(x)

void inline set_affinity(cpu_set_t *aff){
#ifndef NDEBUG
    int rc =
#endif
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), aff);
    assert(rc == 0);
    // this_thread::yield();
}

template<typename T>
int get_numa_addressed(T * m){
//     T * mtest = m;//(T *) (((uintptr_t) m) & ~(((uintptr_t) getpagesize()) - 1));
//     int status[1];
//     status[0]=-1;
// #ifndef NDEBUG
//     int ret_code = 
// #endif
//     move_pages(0 /*self memory */, 1, (void **) &mtest, NULL, status, 0);
//     assert(ret_code == 0);
//     return status[0];
    int numa_node = -1;
    get_mempolicy(&numa_node, NULL, 0, (void*) m, MPOL_F_NODE | MPOL_F_ADDR);
    return numa_node;
}

void inline set_affinity_local_to_gpu(int device){
    set_affinity(&gpu_affinity[device]);
}

inline int numa_node_of_gpu(int device){
    return gpu_numa_node[device];
}

inline int calc_numa_node_of_gpu(int device){ // a portable but slow way...
    cpu_set_t cpus = gpu_affinity[device];
    for (int i = 0 ; i < cpu_cnt ; ++i) if (CPU_ISSET(i, &cpus)) return numa_node_of_cpu(i);
    assert(false);
    return -1;
}

template<typename T = char>
T * malloc_host_local_to_gpu(size_t size, int device){
    assert(device >= 0);

    T      *mem = (T *) numa_alloc_onnode(sizeof(T)*size, numa_node_of_gpu(device));
    assert(mem);
    mem[0] = 0; //force allocation of first page
    gpu_run(cudaHostRegister(mem, sizeof(T)*size, 0));

    // T * mem;
    // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

inline void * cudaMallocHost_local_to_cpu(size_t size, int device){
    assert(device >= 0);

    time_block t("TcudaMallocHost_local_to_cpu: ");

    void * mem;
    {
        time_block t("Tmmap_alloc: ");
        mem = numa_alloc_onnode(size, device);
        assert(mem);
        ((int *) mem)[0] = 0; //force allocation of first page
    }
    gpu_run(cudaHostRegister(mem, size, 0));

    // // T * mem;
    // // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

inline void * cudaMallocHost_local_to_gpu(size_t size, int device){
    return cudaMallocHost_local_to_cpu(size, numa_node_of_gpu(device));
}

inline void * cudaMallocHost_local_to_gpu(size_t size){
    return cudaMallocHost_local_to_gpu(size, get_device());
}

inline void * cudaMallocHost_local_to_cpu(size_t size){
    return cudaMallocHost_local_to_cpu(size, numa_node_of_cpu(sched_getcpu()));
}


inline void cudaFreeHost_local_to_gpu(void * mem, size_t size){
    gpu_run(cudaHostUnregister(mem));

    numa_free(mem, size);
}

inline void cudaFreeHost_local_to_cpu(void * mem, size_t size){
    gpu_run(cudaHostUnregister(mem));

    numa_free(mem, size);
}


#endif /* NUMA_UTILS_CUH_ */