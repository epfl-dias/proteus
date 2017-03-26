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

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <cstdint>
#include <iostream>

constexpr uint32_t warp_size = 32;

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

#endif /* GPU_COMMON_HPP_ */