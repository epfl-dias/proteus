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

#include <cassert>
#include <cstdlib>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/topology/topology.hpp>

void launch_kernel(CUfunction function, void **args, dim3 gridDim,
                   dim3 blockDim, cudaStream_t strm) {
  gpu_run(cuLaunchKernel(function, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
                         blockDim.y, blockDim.z, 0, (CUstream)strm, args,
                         nullptr));
}

void launch_kernel(CUfunction function, void **args, dim3 gridDim,
                   cudaStream_t strm) {
  launch_kernel(function, args, gridDim, defaultBlockDim, strm);
}

void launch_kernel(CUfunction function, void **args, cudaStream_t strm) {
  launch_kernel(function, args, defaultGridDim, defaultBlockDim, strm);
}

extern "C" {
void launch_kernel(CUfunction function, void **args) {
  launch_kernel(function, args, defaultGridDim, defaultBlockDim, nullptr);
}

void launch_kernel_strm(CUfunction function, void **args, cudaStream_t strm) {
  launch_kernel(function, args, strm);
  gpu_run(cudaStreamSynchronize(strm));
}

void launch_kernel_strm_sized(CUfunction function, void **args,
                              cudaStream_t strm, unsigned int blockX,
                              unsigned int gridX) {
  launch_kernel(function, args, gridX, blockX, strm);
}

void launch_kernel_strm_single(CUfunction function, void **args,
                               cudaStream_t strm) {
  launch_kernel_strm_sized(function, args, strm, 1, 1);
}
}

extern "C" {
int get_ptr_device(const void *p) {
  const auto *g = topology::getInstance().getGpuAddressed(p);
  return g ? g->id : -1;
}

// FIXME: rename function...........
int get_ptr_device_or_rand_for_host(const void *p) {
  const auto *g = topology::getInstance().getGpuAddressed(p);
  if (g) return g->id;
  const auto *c = topology::getInstance().getCpuNumaNodeAddressed(p);
  size_t local_gpus = c->local_gpus.size();
  if (local_gpus == 1)
    return c->local_gpus[0];
  else if (local_gpus > 0)
    return c->local_gpus[rand() % local_gpus];
  else
    return rand();
}
}

cudaStream_t createNonBlockingStream() {
  cudaStream_t strm = nullptr;
  gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
  return strm;
}

extern "C" void memcpy_gpu(void *dst, const void *src, size_t size,
                           bool is_volatile) {
  assert(!is_volatile);
#ifndef NCUDA
  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, strm));
  syncAndDestroyStream(strm);
#else
  memcpy(dst, src, size);
#endif
}

void syncAndDestroyStream(cudaStream_t strm) {
  gpu_run(cudaStreamSynchronize(strm));
  gpu_run(cudaStreamDestroy(strm));
}

dim3 defaultGridDim = {40, 1, 1};
