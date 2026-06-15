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
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/tracing.hpp>

DECLARE_uint32(gpu_launch_trace_limit);
DEFINE_uint32(gpu_launch_trace_limit, 0,
              "Log at most N launch_kernel_strm_sized calls (0 disables)");
DECLARE_uint32(gpu_launch_trace_every);
DEFINE_uint32(gpu_launch_trace_every, 0,
              "If >0, log every Nth launch_kernel_strm_sized call");
DECLARE_uint32(gpu_launch_sync_every);
DEFINE_uint32(gpu_launch_sync_every, 0,
              "If >0, synchronize stream every N launch_kernel_strm_sized calls");

DECLARE_int32(num_threads);
DEFINE_int32(num_threads, 0,
             "Override blockDim.x for launch_kernel_strm (NONE compression "
             "path). 0 = use defaultBlockDim (1024).");

namespace {
std::mutex gpu_kernel_symbols_mu;
std::unordered_map<CUfunction, std::string> gpu_kernel_symbols;

std::string getGpuKernelSymbol(CUfunction function) {
  std::lock_guard<std::mutex> lock(gpu_kernel_symbols_mu);
  auto it = gpu_kernel_symbols.find(function);
  if (it == gpu_kernel_symbols.end()) return {};
  return it->second;
}
}  // namespace

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
void register_gpu_kernel_symbol(CUfunction function, const char *name) {
  if (!function || !name || name[0] == '\0') return;
  std::lock_guard<std::mutex> lock(gpu_kernel_symbols_mu);
  gpu_kernel_symbols[function] = name;
}

void launch_kernel(CUfunction function, void **args) {
  launch_kernel(function, args, defaultGridDim, defaultBlockDim, nullptr);
}

void launch_kernel_strm(CUfunction function, void **args, cudaStream_t strm) {
  event_range<range_log_op::RUN_GPU_KERNEL> er{{}};
  if (FLAGS_num_threads > 0) {
    launch_kernel(function, args, defaultGridDim,
                  dim3(FLAGS_num_threads, 1, 1), strm);
  } else {
    launch_kernel(function, args, strm);
  }
  // Keep legacy semantics: callers may read host-visible state immediately
  // after launch via operator close callbacks.
  gpu_run(cudaStreamSynchronize(strm));
}

void launch_kernel_strm_sized(CUfunction function, void **args,
                              cudaStream_t strm, unsigned int blockX,
                              unsigned int gridX) {
  static std::atomic<uint64_t> launch_counter{0};
  const uint64_t launch_id =
      launch_counter.fetch_add(1, std::memory_order_relaxed);
  const bool in_limit = FLAGS_gpu_launch_trace_limit > 0 &&
                        launch_id < FLAGS_gpu_launch_trace_limit;
  const bool periodic = FLAGS_gpu_launch_trace_every > 0 &&
                        (launch_id % FLAGS_gpu_launch_trace_every) == 0;
  std::string kernel_symbol;
  const bool do_sync = FLAGS_gpu_launch_sync_every > 0 &&
                       ((launch_id + 1) % FLAGS_gpu_launch_sync_every) == 0;
  if (in_limit || periodic || do_sync) {
    kernel_symbol = getGpuKernelSymbol(function);
  }
  if (in_limit || periodic) {
    LOG(INFO) << "GPU launch:sized id=" << launch_id << " stream=" << strm
              << " gridX=" << gridX << " blockX=" << blockX
              << " function=" << reinterpret_cast<void *>(function)
              << " kernel="
              << (kernel_symbol.empty() ? "<unknown>" : kernel_symbol);
  }
  launch_kernel(function, args, gridX, blockX, strm);
  if (do_sync) {
    const auto t0 = std::chrono::steady_clock::now();
    LOG(INFO) << "GPU launch:sized sync:start id=" << launch_id
              << " stream=" << strm
              << " gridX=" << gridX << " blockX=" << blockX
              << " function=" << reinterpret_cast<void *>(function)
              << " kernel="
              << (kernel_symbol.empty() ? "<unknown>" : kernel_symbol);
    uint64_t next_wait_log_ms = 1000;
    while (true) {
      const auto q = cudaStreamQuery(strm);
      if (q == cudaSuccess) break;
      if (q != cudaErrorNotReady) {
        gpu_run(q);
      }
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0)
              .count();
      if (elapsed_ms >= static_cast<long long>(next_wait_log_ms)) {
        LOG(INFO) << "GPU launch:sized sync:waiting id=" << launch_id
                  << " stream=" << strm << " elapsed_ms=" << elapsed_ms
                  << " gridX=" << gridX << " blockX=" << blockX
                  << " function=" << reinterpret_cast<void *>(function)
                  << " kernel="
                  << (kernel_symbol.empty() ? "<unknown>" : kernel_symbol);
        next_wait_log_ms += 1000;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count();
    LOG(INFO) << "GPU launch:sized sync:done id=" << launch_id
              << " stream=" << strm << " elapsed_ms=" << elapsed_ms
              << " gridX=" << gridX << " blockX=" << blockX
              << " function=" << reinterpret_cast<void *>(function)
              << " kernel="
              << (kernel_symbol.empty() ? "<unknown>" : kernel_symbol);
  }
}

void launch_kernel_strm_single(CUfunction function, void **args,
                               cudaStream_t strm) {
  launch_kernel_strm_sized(function, args, strm, 1, 1);
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

/// Placeholder defaultGridDim. The value is updated when topology initializes
dim3 defaultGridDim = {40, 1, 1};
