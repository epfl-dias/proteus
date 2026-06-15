/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnewline-eof"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#pragma clang diagnostic ignored "-Wdocumentation"
#include <nvcomp/lz4.h>
#pragma clang diagnostic pop
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/glog.hpp>
#include <platform/util/timing.hpp>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging((argv)[0]);
  FLAGS_colorlogtostderr = true;
  google::LogToStderr();

  topology::init();
  auto& topo = topology::getInstance();
  if (topo.getGpuCount() == 0) {
    LOG(FATAL) << "No GPUs found";
  }
  auto& gpu = topo.getGpus().at(0);
  auto exec_scope = gpu.set_on_scope();

  const size_t uncompressed_byte_count = 24_M;

  LOG(INFO) << "allocating and initializing " << uncompressed_byte_count
            << " bytes of data on CPU...";
  // Do not allocate memory via the topology usually, prefer using standard
  // allocators for host only data, or MemoryManager::Pinned for data that will
  // be copied to the GPU. This is just avoiding initialization of the memory
  // managers
  int* initial_data = static_cast<int*>(
      gpu.getLocalCPUNumaNode().alloc(uncompressed_byte_count));
  for (size_t i = 0; i < 24_M / sizeof(int); i++) {
    initial_data[i] = i % 1024;  // make the data a bit compressible
  }
  constexpr size_t kChunkSize = 1 << 16;

  /* Compression code.  */

  // compute chunk sizes
  size_t* host_uncompressed_chunk_sizes = nullptr;
  const size_t batch_size =
      (uncompressed_byte_count + kChunkSize - 1) / kChunkSize;

  char* device_input_data = nullptr;
  LOG(INFO) << "copying initial data to the GPU...";
  gpu_run(cudaMalloc(&device_input_data, uncompressed_byte_count));
  gpu_run(cudaMemcpy(device_input_data, static_cast<void*>(initial_data),
                     uncompressed_byte_count, cudaMemcpyHostToDevice));

  gpu_run(cudaMallocHost(&host_uncompressed_chunk_sizes,
                         sizeof(size_t) * batch_size));
  for (size_t i = 0; i < batch_size; ++i) {
    if (i + 1 < batch_size) {
      host_uncompressed_chunk_sizes[i] = kChunkSize;
    } else {
      // last chunk may be smaller
      host_uncompressed_chunk_sizes[i] =
          uncompressed_byte_count - (kChunkSize * i);
    }
  }

  void** host_uncompressed_chunk_ptrs = nullptr;
  cudaMallocHost(&host_uncompressed_chunk_ptrs, sizeof(size_t) * batch_size);
  for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
    host_uncompressed_chunk_ptrs[ix_chunk] =
        device_input_data + kChunkSize * ix_chunk;
  }

  LOG(INFO) << "copying chunk sizes and chunk ptrs to the GPU...";
  size_t* device_uncompressed_bytes = nullptr;
  void** device_uncompressed_ptrs = nullptr;
  gpu_run(cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size));
  gpu_run(cudaMalloc(&device_uncompressed_ptrs, sizeof(size_t) * batch_size));

  gpu_run(cudaMemcpy(device_uncompressed_bytes, host_uncompressed_chunk_sizes,
                     sizeof(size_t) * batch_size, cudaMemcpyHostToDevice));
  gpu_run(cudaMemcpy(device_uncompressed_ptrs, host_uncompressed_chunk_ptrs,
                     sizeof(size_t) * batch_size, cudaMemcpyHostToDevice));

  size_t temp_bytes;
  nvcompBatchedLZ4CompressGetTempSize(batch_size, kChunkSize,
                                      nvcompBatchedLZ4DefaultOpts, &temp_bytes);

  LOG(INFO) << "allocating " << temp_bytes
            << " bytes of device memory for compression temp memory... ";
  void* device_temp_ptr = nullptr;
  cudaMalloc(&device_temp_ptr, temp_bytes);

  LOG(INFO) << "calculating maximum compressed chunk size";
  size_t max_out_bytes;
  nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      kChunkSize, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

  LOG(INFO) << "allocating device memory for compressed chunks ...";
  void** host_compressed_ptrs = nullptr;
  gpu_run(cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size));
  for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
    gpu_run(cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes));
  }

  void** device_compressed_ptrs = nullptr;
  gpu_run(cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size));
  gpu_run(cudaMemcpy(device_compressed_ptrs, host_compressed_ptrs,
                     sizeof(size_t) * batch_size, cudaMemcpyHostToDevice));

  LOG(INFO) << "allocating device memory for compressed chunk sizes ...";
  size_t* device_compressed_bytes = nullptr;
  gpu_run(cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size));

  cudaStream_t stream = nullptr;
  gpu_run(cudaStreamCreate(&stream));

  LOG(INFO) << "compressing the chunks on the GPU...";
  nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
      device_uncompressed_ptrs, device_uncompressed_bytes,
      kChunkSize,  // The maximum uncompressed chunk size
      batch_size, device_temp_ptr, temp_bytes, device_compressed_ptrs,
      device_compressed_bytes, nvcompBatchedLZ4DefaultOpts, stream);

  CHECK_EQ(comp_res, nvcompSuccess) << "compression failed";

  LOG(INFO) << "syncing the stream after compression...";
  gpu_run(cudaStreamSynchronize(stream));
  gpu_run(cudaStreamDestroy(stream));

  size_t compressed_size = 0;
  size_t* compressed_chunk_sizes_host = new size_t[batch_size];
  gpu_run(cudaMemcpy(compressed_chunk_sizes_host, device_compressed_bytes,
                     batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < batch_size; i++) {
    compressed_size += compressed_chunk_sizes_host[i];
  }
  LOG(INFO) << "compressed to " << compressed_size << " bytes";

  /* Decompression code.  */

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  std::chrono::milliseconds real_decomp_time;
  {
    time_block t{[&](auto tms) { real_decomp_time = tms; }};

    cudaStreamCreate(&stream);

    gpu_run(cudaEventCreate(&start));
    gpu_run(cudaEventCreate(&end));
    gpu_run(cudaEventRecord(start, stream));

    nvcompBatchedLZ4GetDecompressSizeAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, batch_size, stream);

    LOG(INFO) << "calculating decompression temp memory size and allocating...";
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, kChunkSize,
                                          &decomp_temp_bytes);
    void* device_decomp_temp = nullptr;
    gpu_run(cudaMallocAsync(&device_decomp_temp, decomp_temp_bytes, stream));

    // allocate statuses, 1 per chunk
    nvcompStatus_t* device_statuses = nullptr;
    gpu_run(cudaMallocAsync(&device_statuses,
                            sizeof(nvcompStatus_t) * batch_size, stream));

    LOG(INFO) << "allocating device memory for uncompressed sizes";
    size_t* device_actual_uncompressed_bytes = nullptr;
    gpu_run(cudaMallocAsync(&device_actual_uncompressed_bytes,
                            sizeof(size_t) * batch_size, stream));

    LOG(INFO) << "decompressing the chunks on the GPU...";
    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, device_actual_uncompressed_bytes, batch_size,
        device_decomp_temp, decomp_temp_bytes, device_uncompressed_ptrs,
        device_statuses, stream);

    CHECK_EQ(decomp_res, nvcompSuccess) << "compression failed";

    gpu_run(cudaFreeAsync(device_decomp_temp, stream));
    gpu_run(cudaFreeAsync(device_statuses, stream));

    size_t decompressed_size = 0;
    size_t* decompressed_chunk_sizes_host = new size_t[batch_size];
    gpu_run(cudaMemcpy(decompressed_chunk_sizes_host,
                       device_actual_uncompressed_bytes,
                       batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < batch_size; i++) {
      decompressed_size += decompressed_chunk_sizes_host[i];
    }
    LOG(INFO) << "decompressed to " << decompressed_size << " bytes";
    CHECK_EQ(decompressed_size, uncompressed_byte_count)
        << "decompressed size does not match original size";

    gpu_run(cudaFreeAsync(device_actual_uncompressed_bytes, stream));

    gpu_run(cudaEventRecord(end, stream));
    LOG(INFO) << "syncing the stream after decompression...";
    gpu_run(cudaStreamSynchronize(stream));
  }
  gpu_run(cudaStreamDestroy(stream));

  float gpu_time = 0;
  gpu_run(cudaEventElapsedTime(&gpu_time, start, end));

  LOG(INFO) << "lz4 (low-level) gpu decompression GPU time : " << gpu_time
            << " ms"
            << " real time: " << real_decomp_time.count() << "ms";
}
