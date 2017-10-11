#include <cstdio>
#include <cinttypes>
#include "common/gpu/gpu-common.hpp"
#include "multigpu/buffer_manager.cuh"

extern "C"{

__device__ void printi64(int64_t x){
    printf("%" PRId64 "\n", x);
}

__device__ int32_t * get_buffers(){
    uint32_t b = __ballot(1);
    uint32_t m = 1 << get_laneid();
    int32_t * ret;
    do {
        uint32_t leader = b & -b;

        if (leader == m) ret = buffer_manager<int32_t>::get_buffer();

        b ^= leader;
    } while (b);
    return ret;
}

__device__ void release_buffers(int32_t * buff){
    uint32_t b = __ballot(1);
    uint32_t m = 1 << get_laneid();
    int32_t * ret;
    do {
        uint32_t leader = b & -b;

        if (leader == m) buffer_manager<int32_t>::release_buffer(buff);

        b ^= leader;
    } while (b);
}

}