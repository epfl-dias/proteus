#include "common/gpu/gpu-common.hpp"

void launch_kernel(CUfunction function, void ** args, dim3 gridDim, dim3 blockDim){
    gpu_run(cuLaunchKernel(function, gridDim.x, gridDim.y, gridDim.z,
                                 blockDim.x, blockDim.y, blockDim.z,
                                 0, NULL, args, NULL));
}

void launch_kernel(CUfunction function, void ** args, dim3 gridDim){
    launch_kernel(function, args, gridDim, defaultBlockDim);
}

void launch_kernel(CUfunction function, void ** args){
    launch_kernel(function, args, defaultGridDim, defaultBlockDim);
}
