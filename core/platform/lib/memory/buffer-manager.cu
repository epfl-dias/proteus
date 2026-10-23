/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include <cinttypes>
#include <cstdint>
#include <cstdio>

// __device__ __constant__ threadsafe_device_stack<int32_t *, (int32_t *)
// nullptr>
// * pool;
// __device__ __constant__ int deviceId;
// __device__ __constant__ void * buff_start;
// __device__ __constant__ void * buff_end  ;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated"

#include <platform/common/common.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/memory/buffer-manager.cuh>
#include <platform/memory/decompression-config.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/util/string-object.hpp>
#include <platform/util/threadsafe_device_stack.cuh>
#include <thread>

#include <cub/cub.cuh>

#ifndef NCUDA
extern __device__ __constant__
    threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *pool;
extern __device__ __constant__
    threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *npool;
extern __device__ __constant__ int deviceId;
extern __device__ __constant__ void *buff_start;
extern __device__ __constant__ void *buff_end;

__device__ __constant__
    threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *pool;
__device__ __constant__
    threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *npool;
__device__ __constant__ int deviceId;
__device__ __constant__ void *buff_start;
__device__ __constant__ void *buff_end;
#else
constexpr threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *pool =
    nullptr;
constexpr threadsafe_device_stack<int32_t *, (int32_t *)nullptr> *npool =
    nullptr;
constexpr int deviceId = 0;
constexpr void *buff_start = nullptr;
constexpr void *buff_end = nullptr;
#endif

void buffer_manager_init(float gpu_mem_pool_percentage = 0.1,
                         float cpu_mem_pool_percentage = 0.1,
                         size_t log_buffers = false) {
  buffer_manager<int32_t>::init(gpu_mem_pool_percentage,
                                cpu_mem_pool_percentage, log_buffers, 64, 128);
}

template <typename T>
__device__ void buffer_manager<T>::__release_buffer_device(T *buff) {
  if (!buff) return;
  // assert(strm == 0); //FIXME: something better ?
  // if (buff->device == deviceId) { //FIXME: remote device!
  // buff->clean();
  // __threadfence();
  if (buff >= buff_start && buff < buff_end)
    pool->push(buff);
  else
    npool->push(buff);
  //  else printf("Throwing buffer: %p %p %p\n", (void *) buff, buff_start,
  //  buff_end);
  // } else                          assert(false); //FIXME: IMPORTANT free
  // buffer of another device (or host)!
}

template <typename T>
__host__ void buffer_manager<T>::__release_buffer_host(T *buff) {
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
    while (true) {
      size_t size = device_buffs_pool[gpu->id].size();
      if (size > keep_threshold) {
        uint32_t devid = gpu->id;
        nvtxRangePushA("release_buffer_host_devbuffer_overflow");
        for (size_t i = 0; i < device_buff_size; ++i)
          device_buff[devid][i] = device_buffs_pool[devid][size - i - 1];
        device_buffs_pool[devid].erase(
            device_buffs_pool[devid].end() - device_buff_size,
            device_buffs_pool[devid].end());
        lock.unlock();
        LOG(INFO) << "release spawned";
        release_buffer_host<<<1, 1, 0, release_streams[devid]>>>(
            (void **)device_buff[devid], device_buff_size);
        gpu_run(cudaStreamSynchronize(release_streams[devid]));
        lock.lock();
        LOG(INFO) << "release done";
        // gpu_run(cudaPeekAtLastError()  );
        // gpu_run(cudaDeviceSynchronize());
        nvtxRangePop();
      } else
        break;
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
    assert(occ >= 1 && "Buffer is already released");
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

void buffer_manager_destroy() { buffer_manager<int32_t>::destroy(); }

template <typename T, typename... Args>
__host__ T *cuda_new(const topology::gpunode &gpu, Args &&...args) {
  set_device_on_scope d{gpu};
  T *tmp = new T(std::forward<Args>(args)...);
  T *res = (T *)MemoryManager::mallocGpu(sizeof(T));
  gpu_run(cudaMemcpy(res, tmp, sizeof(T), cudaMemcpyDefault));
  gpu_run(cudaDeviceSynchronize());
  free(tmp);  // NOTE: bad practice ? we want to allocate tmp by new to
              //      trigger initialization but we want to free the
              //      corresponding memory after moving to device
              //      without triggering the destructor
  return res;
}

template <typename T, typename... Args>
__host__ void cuda_delete(T *obj, Args &&...args) {
  if (topology::getInstance().getGpuAddressed(obj)) {
    T *tmp = (T *)malloc(sizeof(T));
    gpu_run(cudaDeviceSynchronize());
    gpu_run(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
    MemoryManager::freeGpu(obj);
    delete tmp;
  } else {
    T *tmp = (T *)malloc(sizeof(T));
    gpu_run(cudaDeviceSynchronize());
    gpu_run(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
    gpu_run(cudaFreeHost(obj));
    delete tmp;
    // delete obj;
  }
}

extern "C" {

#ifndef NCUDA
__device__ void dprinti64(int64_t x) { printf("%" PRId64 "\n", x); }

namespace proteus {
__device__ int32_t *get_buffers() {
  uint32_t mask = __activemask();
  uint32_t b = __ballot_sync(mask, 1);
  uint32_t m = 1 << get_laneid();
  int32_t *ret = nullptr;
  do {
    uint32_t leader = b & -b;

    if (leader == m) ret = buffer_manager<int32_t>::get_buffer();

    b ^= leader;
  } while (b);
  __syncwarp(mask);
  return ret;
}
}  // namespace proteus

void *get_dev_buffer() {
  return buffer_manager<int32_t>::h_get_buffer(
      topology::getInstance().getActiveGpu().id);
}

namespace proteus {
__device__ void release_buffers(int32_t *buff) {
  uint32_t mask = __activemask();
  uint32_t b = __ballot_sync(mask, buff != nullptr);
  uint32_t m = 1 << get_laneid();
  do {
    uint32_t leader = b & -b;

    if (leader == m) buffer_manager<int32_t>::release_buffer(buff);

    b ^= leader;
  } while (b);
  __syncwarp(mask);
}
}  // namespace proteus

__device__ void dprinti(int32_t x) { printf("%d\n", x); }

__device__ void dprintptr(void *x) { printf("%p\n", x); }

__device__ void dprintstr(const char *x) { printf("%s\n", x); }

static constexpr int BLOCK_THREADS = proteus::rfor::cb2t::kBlockThreads;
static constexpr int ITEMS_PER_THREAD = proteus::rfor::cb2t::kItemsPerThread;
static constexpr int ITEMS_PER_GROUP = BLOCK_THREADS * ITEMS_PER_THREAD;
static constexpr int FLBP_VECTOR_SIZE = proteus::flbp::cb2t::kVectorSize;
static constexpr int FLBP_WARP_THREADS = 32;
// Scratch layout for LoadRBinPack_notemplate:
// [working area (ITEMS_PER_GROUP u32) | staged packed stream
//  (ITEMS_PER_GROUP u32)].
// One RFOR tile per CTA.

struct RLETileMetadata {
  uint32_t val_reference;
  uint32_t val_bitwidth;
  uint32_t val_count;
  uint32_t rl_reference;
  uint32_t rl_bitwidth;
  uint32_t rl_count;
};

#if defined(__clang__)
using shared_u32_ptr = uint __attribute__((address_space(3))) *;
using shared_i32_ptr = int __attribute__((address_space(3))) *;
#else
using shared_u32_ptr = uint *;
using shared_i32_ptr = int *;
#endif

__forceinline__ __device__ int decodeElementRBin(int i, const uint* data_block,
                                                  uint packed_words,
                                                  uint reference,
                                                  uint bitwidth) {
  if (bitwidth == 0) return reference;

  uint start_bitindex = (bitwidth * i);
  uint start_intindex = (start_bitindex >> 5);
  start_bitindex = start_bitindex & (32 - 1);

  if (start_intindex >= packed_words) return reference;

  const uint low_word = data_block[start_intindex];
  unsigned long long element_block = low_word;
  if (start_bitindex + bitwidth > 32 && (start_intindex + 1) < packed_words) {
    element_block |=
        (static_cast<unsigned long long>(data_block[start_intindex + 1]) << 32);
  }

  const unsigned long long mask =
      (bitwidth >= 32) ? 0xFFFFFFFFULL : ((1ULL << bitwidth) - 1ULL);
  const uint element = (element_block >> start_bitindex) & mask;
  return reference + element;
}

__forceinline__ __device__ int decodeElementRBinFastShared(int i,
                                                            shared_u32_ptr data_block,
                                                            uint packed_words,
                                                            uint reference,
                                                            uint bitwidth) {
  if (bitwidth == 0) return reference;

  uint start_bitindex = (bitwidth * i);
  uint start_intindex = (start_bitindex >> 5);
  start_bitindex = start_bitindex & (32 - 1);
  if (start_intindex >= packed_words) return reference;

  uint element = 0;
  const unsigned long long mask =
      (bitwidth == 32) ? 0xffffffffULL : ((1ULL << bitwidth) - 1ULL);

  if (start_bitindex + bitwidth <= 32) {
    element = static_cast<uint>(
        (static_cast<unsigned long long>(data_block[start_intindex]) >>
         start_bitindex) &
        mask);
  } else {
    const unsigned long long element_block =
        ((start_intindex + 1) < packed_words
             ? (static_cast<unsigned long long>(data_block[start_intindex + 1]) << 32)
             : 0ULL) |
        data_block[start_intindex];
    element = static_cast<uint>((element_block >> start_bitindex) & mask);
  }

  return reference + element;
}

__device__ __forceinline__ void group_striped_to_blocked(
    shared_u32_ptr scratch, int lane, int (&items)[ITEMS_PER_THREAD]) {
  // Convert striped layout -> blocked layout for a group.
  // Striped: item i is stored at scratch[i * BLOCK_THREADS + lane].
  // Blocked: items for a lane are contiguous (lane * ITEMS_PER_THREAD + i).
  // NOTE: scratch is a group-local slice; __syncthreads is block-wide.
  int base = lane * ITEMS_PER_THREAD;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    scratch[i * BLOCK_THREADS + lane] = static_cast<uint32_t>(items[i]);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    items[i] = static_cast<int>(scratch[base + i]);
  }
  __syncthreads();
}

__device__ __forceinline__ void group_blocked_to_striped(
    shared_u32_ptr scratch, int lane, int (&items)[ITEMS_PER_THREAD]) {
  // Convert blocked layout -> striped layout for a group.
  // This is the inverse of group_striped_to_blocked.
  int base = lane * ITEMS_PER_THREAD;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    scratch[base + i] = static_cast<uint32_t>(items[i]);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    items[i] = static_cast<int>(scratch[i * BLOCK_THREADS + lane]);
  }
  __syncthreads();
}

__device__ __forceinline__ void group_inclusive_scan(
    shared_u32_ptr scratch, int lane, int (&items)[ITEMS_PER_THREAD]) {
  // Group-wide inclusive scan over ITEMS_PER_GROUP values using shared scratch.
  // Each lane contributes ITEMS_PER_THREAD values in blocked layout.
  // We use a simple O(log N) Hillis-Steele style scan with __syncthreads().
  // Caller must provide a group-local scratch slice to avoid cross-group races.
  int base = lane * ITEMS_PER_THREAD;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    scratch[base + i] = static_cast<uint32_t>(items[i]);
  }
  __syncthreads();
  for (int offset = 1; offset < ITEMS_PER_GROUP; offset <<= 1) {
    uint32_t tmp[ITEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      int idx = base + i;
      tmp[i] = (idx >= offset) ? scratch[idx - offset] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      scratch[base + i] += tmp[i];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    items[i] = static_cast<int>(scratch[base + i]);
  }
  __syncthreads();
}

__device__ void LoadRBinPack_predecode_impl_runtime(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out,
    const uint block_threads_t,
    const uint items_per_thread_t,
    bool run_ends_are_cumulative) {
  const uint items_per_group_t = block_threads_t * items_per_thread_t;
  uint value_count = metadata->val_count;
  if (value_count > items_per_group_t) value_count = items_per_group_t;
  uint rl_count = metadata->rl_count;
  if (rl_count > items_per_group_t) rl_count = items_per_group_t;
  const uint run_count = value_count < rl_count ? value_count : rl_count;

  const uint64_t val_words_u64 =
      ((static_cast<uint64_t>(value_count) * metadata->val_bitwidth) + 31ull) >>
      5ull;
  const uint64_t rl_words_u64 =
      ((static_cast<uint64_t>(rl_count) * metadata->rl_bitwidth) + 31ull) >>
      5ull;
  const uint val_words = static_cast<uint>(
      val_words_u64 > items_per_group_t ? items_per_group_t : val_words_u64);
  const uint rl_words = static_cast<uint>(
      rl_words_u64 > items_per_group_t ? items_per_group_t : rl_words_u64);

  for (uint idx = threadIdx.x; idx < run_count; idx += block_threads_t) {
    run_values_out[idx] = decodeElementRBin(idx, val_data, val_words,
                                            metadata->val_reference,
                                            metadata->val_bitwidth);
    run_ends_out[idx] = decodeElementRBin(idx, rl_data, rl_words,
                                          metadata->rl_reference,
                                          metadata->rl_bitwidth);
  }
  __syncthreads();

  if (!run_ends_are_cumulative) {
    // Legacy RFOR datasets store individual run lengths. Convert once to
    // cumulative run ends so tuple decode can use RFOR_PREDECODE-style
    // binary search.
    if (threadIdx.x == 0) {
      int cumulative = 0;
      for (uint idx = 0; idx < run_count; ++idx) {
        cumulative += run_ends_out[idx];
        run_ends_out[idx] = cumulative;
      }
    }
    __syncthreads();
  }
}

__device__ void LoadRBinPack_predecode_notemplate(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      BLOCK_THREADS, ITEMS_PER_THREAD,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_notemplate_64x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      64, 8,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_notemplate_512x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      512, 2,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_notemplate_256x4(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      256, 4,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_notemplate_256x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      256, 2,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_notemplate_128x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      128, 8,
      /*run_ends_are_cumulative=*/false);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      BLOCK_THREADS, ITEMS_PER_THREAD,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate_64x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      64, 8,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate_512x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      512, 2,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate_256x4(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      256, 4,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate_256x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      256, 2,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_predecode_cumulative_notemplate_128x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_i32_ptr __restrict__ run_values_out,
    shared_i32_ptr __restrict__ run_ends_out) {
  LoadRBinPack_predecode_impl_runtime(
      metadata, val_data, rl_data, run_values_out, run_ends_out,
      128, 8,
      /*run_ends_are_cumulative=*/true);
}

__device__ void LoadRBinPack_notemplate(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values)
{
  typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
  // Keep decode/working + packed-stream staging in caller-provided shared
  // scratch.
  constexpr uint decode_words = ITEMS_PER_GROUP;
  constexpr uint working_offset_words = 0;
  constexpr uint stage_offset_words = working_offset_words + decode_words;
  constexpr uint stage_words = decode_words;
  constexpr uint required_scratch_words = stage_offset_words + stage_words;
  // Defensive cap to catch unexpected scratch growth at compile time.
  static_assert(required_scratch_words <= (2 * ITEMS_PER_GROUP),
                "LoadRBinPack_notemplate scratch exceeds reserved capacity");

  uint count = metadata->val_count;
  if (count > ITEMS_PER_GROUP) count = ITEMS_PER_GROUP;
  uint rl_count = metadata->rl_count;
  if (rl_count > ITEMS_PER_GROUP) rl_count = ITEMS_PER_GROUP;
  uint num_decode_val = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
  uint num_decode_rl = ((rl_count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
  const uint64_t val_words_u64 =
      ((static_cast<uint64_t>(count) * metadata->val_bitwidth) + 31ull) >> 5ull;
  const uint64_t rl_words_u64 =
      ((static_cast<uint64_t>(rl_count) * metadata->rl_bitwidth) + 31ull) >> 5ull;
  const uint val_words =
      static_cast<uint>(val_words_u64 > decode_words ? decode_words : val_words_u64);
  const uint rl_words =
      static_cast<uint>(rl_words_u64 > decode_words ? decode_words : rl_words_u64);

  int items_value[ITEMS_PER_THREAD];
  int items_run_length[ITEMS_PER_THREAD];
  shared_u32_ptr working_area = shared_buffer + working_offset_words;
  shared_u32_ptr staged_area = shared_buffer + stage_offset_words;

  // Stage values once from global memory and decode from shared.
  for (uint idx = threadIdx.x; idx < val_words; idx += BLOCK_THREADS) {
    staged_area[idx] = val_data[idx];
  }
  __syncthreads();

  uint decode_idx = threadIdx.x;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (threadIdx.x < num_decode_val && decode_idx < count) {
      items_value[i] =
          decodeElementRBinFastShared(decode_idx, staged_area, val_words,
                                      metadata->val_reference,
                                      metadata->val_bitwidth);
    } else {
      items_value[i] = 0;
    }
    decode_idx += num_decode_val;
  }
  __syncthreads();

  // Reuse staged buffer for run-length stream.
  for (uint idx = threadIdx.x; idx < rl_words; idx += BLOCK_THREADS) {
    staged_area[idx] = rl_data[idx];
  }
  __syncthreads();

  decode_idx = threadIdx.x;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (threadIdx.x < num_decode_rl && decode_idx < rl_count) {
      items_run_length[i] =
          decodeElementRBinFastShared(decode_idx, staged_area, rl_words,
                                      metadata->rl_reference,
                                      metadata->rl_bitwidth);
    } else {
      items_run_length[i] = 0;
    }
    decode_idx += num_decode_rl;
  }

  __syncthreads();

  __shared__ typename BlockScan::TempStorage temp_storage_scan;
  __shared__ typename BlockExchange::TempStorage temp_storage_exchange;

  BlockExchange(temp_storage_exchange).StripedToBlocked(items_run_length);
  __syncthreads();

  BlockScan(temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
  __syncthreads();

  const uint lane_base = threadIdx.x * ITEMS_PER_THREAD;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    working_area[lane_base + i] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (items_run_length[i] < ITEMS_PER_GROUP) {
      working_area[items_run_length[i]] = 1;
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    items_run_length[i] = working_area[lane_base + i];
  }
  __syncthreads();

  BlockScan(temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
  __syncthreads();

  BlockExchange(temp_storage_exchange).BlockedToStriped(items_run_length);
  __syncthreads();

  uint rle_offset = 0;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (threadIdx.x < num_decode_val &&
        threadIdx.x + rle_offset < ITEMS_PER_GROUP) {
      working_area[threadIdx.x + rle_offset] = items_value[i];
    }
    rle_offset += num_decode_val;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (items_run_length[i] < ITEMS_PER_GROUP) {
      items_value[i] = working_area[items_run_length[i]];
    } else {
      items_value[i] = 0;
    }
  }
  __syncthreads();

  // Write the fully expanded values directly into the caller-provided output
  // tile in striped layout: lane + i*BLOCK_THREADS.
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    out_values[threadIdx.x + i * BLOCK_THREADS] = items_value[i];
  }
}

// Shape-specialized cooperative RFOR decode. Keep packed-stream staging and
// RLE working scratch in separate shared regions: the helper is called once per
// column from generated CB2T code, and CUB/shared scratch reuse is sensitive to
// warp progress at high-occupancy shapes such as 256x4.
#define RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(BLOCK_THREADS_T, ITEMS_PER_THREAD_T) \
  typedef cub::BlockExchange<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T> BlockExchange; \
  typedef cub::BlockScan<int, BLOCK_THREADS_T> BlockScan; \
  constexpr uint ITEMS_PER_GROUP_T = BLOCK_THREADS_T * ITEMS_PER_THREAD_T; \
  constexpr uint decode_words = ITEMS_PER_GROUP_T; \
  constexpr uint working_offset_words = 0; \
  constexpr uint stage_offset_words = working_offset_words + decode_words; \
  constexpr uint stage_words = decode_words; \
  constexpr uint required_scratch_words = stage_offset_words + stage_words; \
  static_assert(required_scratch_words <= (2 * ITEMS_PER_GROUP_T), \
                "LoadRBinPack_notemplate_overlap scratch exceeds reserved capacity"); \
  uint count = metadata->val_count; \
  if (count > ITEMS_PER_GROUP_T) count = ITEMS_PER_GROUP_T; \
  uint rl_count = metadata->rl_count; \
  if (rl_count > ITEMS_PER_GROUP_T) rl_count = ITEMS_PER_GROUP_T; \
  uint num_decode_val = ((count + ITEMS_PER_THREAD_T - 1) / ITEMS_PER_THREAD_T); \
  uint num_decode_rl = ((rl_count + ITEMS_PER_THREAD_T - 1) / ITEMS_PER_THREAD_T); \
  const uint64_t val_words_u64 = \
      ((static_cast<uint64_t>(count) * metadata->val_bitwidth) + 31ull) >> 5ull; \
  const uint64_t rl_words_u64 = \
      ((static_cast<uint64_t>(rl_count) * metadata->rl_bitwidth) + 31ull) >> 5ull; \
  const uint val_words = static_cast<uint>( \
      val_words_u64 > decode_words ? decode_words : val_words_u64); \
  const uint rl_words = static_cast<uint>( \
      rl_words_u64 > decode_words ? decode_words : rl_words_u64); \
  int items_value[ITEMS_PER_THREAD_T]; \
  int items_run_length[ITEMS_PER_THREAD_T]; \
  shared_u32_ptr working_area = shared_buffer + working_offset_words; \
  shared_u32_ptr staged_area = shared_buffer + stage_offset_words; \
  for (uint idx = threadIdx.x; idx < val_words; idx += BLOCK_THREADS_T) { \
    staged_area[idx] = val_data[idx]; \
  } \
  __syncthreads(); \
  uint decode_idx = threadIdx.x; \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    if (threadIdx.x < num_decode_val && decode_idx < count) { \
      items_value[i] = decodeElementRBinFastShared( \
          decode_idx, staged_area, val_words, metadata->val_reference, \
          metadata->val_bitwidth); \
    } else { \
      items_value[i] = 0; \
    } \
    decode_idx += num_decode_val; \
  } \
  __syncthreads(); \
  for (uint idx = threadIdx.x; idx < rl_words; idx += BLOCK_THREADS_T) { \
    staged_area[idx] = rl_data[idx]; \
  } \
  __syncthreads(); \
  decode_idx = threadIdx.x; \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    if (threadIdx.x < num_decode_rl && decode_idx < rl_count) { \
      items_run_length[i] = decodeElementRBinFastShared( \
          decode_idx, staged_area, rl_words, metadata->rl_reference, \
          metadata->rl_bitwidth); \
    } else { \
      items_run_length[i] = 0; \
    } \
    decode_idx += num_decode_rl; \
  } \
  __syncthreads(); \
  __shared__ typename BlockScan::TempStorage temp_storage_scan; \
  __shared__ typename BlockExchange::TempStorage temp_storage_exchange; \
  BlockExchange(temp_storage_exchange).StripedToBlocked(items_run_length); \
  __syncthreads(); \
  BlockScan(temp_storage_scan).InclusiveSum(items_run_length, items_run_length); \
  __syncthreads(); \
  const uint lane_base = threadIdx.x * ITEMS_PER_THREAD_T; \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    working_area[lane_base + i] = 0; \
  } \
  __syncthreads(); \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    if (items_run_length[i] < ITEMS_PER_GROUP_T) { \
      working_area[items_run_length[i]] = 1; \
    } \
  } \
  __syncthreads(); \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    items_run_length[i] = working_area[lane_base + i]; \
  } \
  __syncthreads(); \
  BlockScan(temp_storage_scan).InclusiveSum(items_run_length, items_run_length); \
  __syncthreads(); \
  BlockExchange(temp_storage_exchange).BlockedToStriped(items_run_length); \
  __syncthreads(); \
  uint rle_offset = 0; \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    if (threadIdx.x < num_decode_val && \
        threadIdx.x + rle_offset < ITEMS_PER_GROUP_T) { \
      working_area[threadIdx.x + rle_offset] = items_value[i]; \
    } \
    rle_offset += num_decode_val; \
  } \
  __syncthreads(); \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; i++) { \
    if (items_run_length[i] < ITEMS_PER_GROUP_T) { \
      items_value[i] = working_area[items_run_length[i]]; \
    } else { \
      items_value[i] = 0; \
    } \
  } \
  __syncthreads(); \
  _Pragma("unroll") \
  for (int i = 0; i < ITEMS_PER_THREAD_T; ++i) { \
    out_values[threadIdx.x + i * BLOCK_THREADS_T] = items_value[i]; \
  }

__device__ void LoadRBinPack_notemplate_overlap(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(BLOCK_THREADS, ITEMS_PER_THREAD)
}

__device__ void LoadRBinPack_notemplate_overlap_64x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(64, 8)
}

__device__ void LoadRBinPack_notemplate_overlap_512x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(512, 2)
}

__device__ void LoadRBinPack_notemplate_overlap_256x4(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(256, 4)
}

__device__ void LoadRBinPack_notemplate_overlap_256x2(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(256, 2)
}

__device__ void LoadRBinPack_notemplate_overlap_128x8(
    RLETileMetadata* __restrict__ metadata,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY(128, 8)
}

#undef RFOR_LOAD_RBINPACK_NOTEMPLATE_OVERLAP_BODY

__forceinline__ __device__ uint32_t decodeElementFLBPLaneStream(
    uint32_t vector_idx, const uint32_t* __restrict__ packed_data,
    uint8_t bitwidth) {
  if (bitwidth == 0) return 0u;

  const uint32_t lane = vector_idx & 31u;     // vector_idx % 32
  const uint32_t pos = vector_idx >> 5u;      // vector_idx / 32, in [0,31]
  const uint32_t bit_pos = pos * static_cast<uint32_t>(bitwidth);
  const uint32_t word_idx = bit_pos >> 5u;    // / 32
  const uint32_t bit_off = bit_pos & 31u;     // % 32

  const uint64_t lo =
      static_cast<uint64_t>(packed_data[word_idx * FLBP_WARP_THREADS + lane]);
  uint64_t packed = lo;
  if (bit_off + bitwidth > 32u && (word_idx + 1u) < bitwidth) {
    const uint64_t hi = static_cast<uint64_t>(
        packed_data[(word_idx + 1u) * FLBP_WARP_THREADS + lane]);
    packed |= (hi << 32u);
  }

  const uint64_t mask =
      (bitwidth >= 32u) ? 0xFFFFFFFFull : ((1ull << bitwidth) - 1ull);
  return static_cast<uint32_t>((packed >> bit_off) & mask);
}

__device__ int DecodeFLBPElement_notemplate(
    uint32_t* __restrict__ metadata_words,
    uint32_t* __restrict__ val_data,
    uint32_t local_idx) {
  const volatile uint32_t* metadata =
      reinterpret_cast<const volatile uint32_t*>(metadata_words);
  const uint32_t base_value = metadata[0];
  const uint8_t bitwidth = static_cast<uint8_t>(metadata[1] & 0xffu);
  const uint32_t element_count = metadata[2];
  const uint32_t subtile_offset = metadata[3];
  const uint32_t vector_idx = subtile_offset + local_idx;

  uint32_t decoded = base_value;
  if (vector_idx < static_cast<uint32_t>(FLBP_VECTOR_SIZE) &&
      vector_idx < element_count) {
    decoded += decodeElementFLBPLaneStream(vector_idx, val_data, bitwidth);
  }
  return static_cast<int>(decoded);
}

// FLBP register-decode fast path for full-vector CB2T tiles.
//
// Safe when:
// - tile_size == FLBP_VECTOR_SIZE, so local_idx already addresses the full
//   logical vector and subtile_offset is provably 0
// - the writer pads the tail vector payload to the full vector width, so the
//   final partial vector still has decodable packed words for all 1024 lanes
//
// In that configuration tuple-loop bounds enforce semantic end-of-relation,
// so per-element element_count/subtile_offset checks are redundant.
__device__ int DecodeFLBPElement_fullvector_notemplate(
    uint32_t* __restrict__ metadata_words,
    uint32_t* __restrict__ val_data,
    uint32_t local_idx) {
  const uint32_t base_value = metadata_words[0];
  const uint8_t bitwidth = static_cast<uint8_t>(metadata_words[1] & 0xffu);
  const uint32_t decoded =
      base_value + decodeElementFLBPLaneStream(local_idx, val_data, bitwidth);
  return static_cast<int>(decoded);
}

// FLBP cooperative decode for CB2T.
// The metadata argument is a CB2T-staged pseudo-struct with layout:
// [base_value, bitwidth, element_count, subtile_offset, subtile_size, reserved].
// Decode reads FLBP packed lane-stream words directly and emits only the
// logical CB2T subtile needed by this CTA/tile_id.
__device__ void LoadFLBP_notemplate_overlap(
    uint32_t* __restrict__ metadata_words,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  (void)rl_data;
  (void)shared_buffer;

  // Snapshot the entire staged metadata block into thread-local registers
  // BEFORE the worker loop. The leader CTA thread staged these words with
  // volatile shared stores and a CTA barrier, but `metadata_words` is a
  // generic-addrspace pointer here, so the device compiler is free to reload
  // it across function-call boundaries inside the loop. A reload of [0..3]
  // per iteration is the suspected source of intermittent FLBP --decom_shared
  // decode corruption on Q3.1/Q4.1 at 256x4. Pin each word once.
  const volatile uint32_t* metadata =
      reinterpret_cast<const volatile uint32_t*>(metadata_words);
  const uint32_t base_value = metadata[0];
  const uint8_t bitwidth = static_cast<uint8_t>(metadata[1] & 0xffu);
  const uint32_t element_count = metadata[2];
  const uint32_t subtile_offset = metadata[3];
  uint32_t subtile_size = metadata[4];
  if (subtile_size == 0u || subtile_size > static_cast<uint32_t>(FLBP_VECTOR_SIZE)) {
    subtile_size = static_cast<uint32_t>(FLBP_VECTOR_SIZE);
  }
  if (subtile_offset >= static_cast<uint32_t>(FLBP_VECTOR_SIZE)) {
    return;
  }
  if (subtile_offset + subtile_size > static_cast<uint32_t>(FLBP_VECTOR_SIZE)) {
    subtile_size = static_cast<uint32_t>(FLBP_VECTOR_SIZE) - subtile_offset;
  }

  const uint32_t worker_threads =
      blockDim.x == 0 ? 1u : static_cast<uint32_t>(blockDim.x);
  for (uint32_t local_idx = static_cast<uint32_t>(threadIdx.x);
       local_idx < subtile_size; local_idx += worker_threads) {
    const uint32_t vector_idx = subtile_offset + local_idx;
    uint32_t decoded = base_value;
    if (vector_idx < static_cast<uint32_t>(FLBP_VECTOR_SIZE) &&
        vector_idx < element_count) {
      decoded += decodeElementFLBPLaneStream(vector_idx, val_data, bitwidth);
    }
    out_values[local_idx] = static_cast<int>(decoded);
  }
}

// FLBP cooperative-decode fast path for full-vector CB2T tiles.
//
// Safe under the same conditions as DecodeFLBPElement_fullvector_notemplate:
// tile_size == FLBP_VECTOR_SIZE and padded full-vector payloads.
__device__ void LoadFLBPFullVector_notemplate_overlap(
    uint32_t* __restrict__ metadata_words,
    uint32_t* __restrict__ val_data,
    uint32_t* __restrict__ rl_data,
    shared_u32_ptr __restrict__ shared_buffer,
    shared_i32_ptr __restrict__ out_values) {
  (void)rl_data;
  (void)shared_buffer;

  const uint32_t base_value = metadata_words[0];
  const uint8_t bitwidth = static_cast<uint8_t>(metadata_words[1] & 0xffu);
  const uint32_t worker_threads =
      blockDim.x == 0 ? 1u : static_cast<uint32_t>(blockDim.x);
  for (uint32_t local_idx = static_cast<uint32_t>(threadIdx.x);
       local_idx < static_cast<uint32_t>(FLBP_VECTOR_SIZE);
       local_idx += worker_threads) {
    const uint32_t decoded =
        base_value + decodeElementFLBPLaneStream(local_idx, val_data, bitwidth);
    out_values[local_idx] = static_cast<int>(decoded);
  }
}

#endif  // NCUDA
}  // extern "C"

void initializeModule(CUmodule &cudaModule) {
#ifndef NCUDA
  CUdeviceptr ptr;
  size_t bytes;
  void *mem;

  auto pool_err = cuModuleGetGlobal(&ptr, &bytes, cudaModule, "pool");
  if (pool_err != CUDA_ERROR_NOT_FOUND) {
    gpu_run(pool_err);
    gpu_run(cudaMemcpyFromSymbol(&mem, pool, sizeof(void *)));
    gpu_run(cuMemcpyHtoD(ptr, &mem, sizeof(void *)));
  }

  auto npool_err = cuModuleGetGlobal(&ptr, &bytes, cudaModule, "npool");
  if (npool_err != CUDA_ERROR_NOT_FOUND) {
    gpu_run(npool_err);
    gpu_run(cudaMemcpyFromSymbol(&mem, npool, sizeof(void *)));
    gpu_run(cuMemcpyHtoD(ptr, &mem, sizeof(void *)));
  }

  auto buff_start_err =
      cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_start");
  if (buff_start_err != CUDA_ERROR_NOT_FOUND) {
    gpu_run(buff_start_err);
    gpu_run(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void *)));
    gpu_run(cuMemcpyHtoD(ptr, &mem, sizeof(void *)));
  }

  auto buff_end_err = cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_end");
  if (buff_end_err != CUDA_ERROR_NOT_FOUND) {
    gpu_run(buff_end_err);
    gpu_run(cudaMemcpyFromSymbol(&mem, buff_end, sizeof(void *)));
    gpu_run(cuMemcpyHtoD(ptr, &mem, sizeof(void *)));
  }

  auto deviceId_err = cuModuleGetGlobal(&ptr, &bytes, cudaModule, "deviceId");
  if (deviceId_err != CUDA_ERROR_NOT_FOUND) {
    gpu_run(deviceId_err);
    gpu_run(cudaMemcpyFromSymbol(&mem, deviceId, sizeof(int)));
    gpu_run(cuMemcpyHtoD(ptr, &mem, sizeof(int)));
  }
#endif
}

#ifndef NCUDA
__global__ void release_buffer_host(void **buff, int buffs) {
  assert(blockDim.x * blockDim.y * blockDim.z == 1);
  assert(gridDim.x * gridDim.y * gridDim.z == 1);
  for (int i = 0; i < buffs; ++i)
    buffer_manager<int32_t>::release_buffer((int32_t *)buff[i]);
}

__global__ void find_released_remote_buffers(void **buff, int buffs) {
  assert(blockDim.x * blockDim.y * blockDim.z == 1);
  assert(gridDim.x * gridDim.y * gridDim.z == 1);
  for (int i = 0; i < buffs; ++i) {
    if (!npool->try_pop(((int32_t **)buff) + i)) {
      buff[i] = nullptr;
      return;
    }
  }
}

__global__ void get_buffer_host(void **buff, int buffs) {
  assert(blockDim.x * blockDim.y * blockDim.z == 1);
  assert(gridDim.x * gridDim.y * gridDim.z == 1);
  for (int i = 0; i < buffs; ++i)
    buff[i] = buffer_manager<int32_t>::try_get_buffer();
}
#endif

#if defined(__clang__) && defined(__CUDA__)
template <typename T>
__device__ T *buffer_manager<T>::get_buffer() {
  return pool->pop();
}

template <typename T>
__host__ T *buffer_manager<T>::get_buffer() {
  return get_buffer_numa(affinity::get());
}
#else
template <typename T>
__host__ __device__ T *buffer_manager<T>::get_buffer() {
#ifdef __CUDA_ARCH__
  return pool->pop();
#else
  return get_buffer_numa(affinity::get());
#endif
}
#endif

template <typename T>
__device__ T *buffer_manager<T>::try_get_buffer() {
#ifndef NCUDA
  T *b;
  bool got = pool->pop_if_nonempty(&b);
  if (!got) b = nullptr;
  return b;
#else
  return nullptr;
#endif
}

#include <platform/topology/topology.hpp>

template <typename T>
__host__ void buffer_manager<T>::init(float gpu_mem_pool_percentage,
                                      float cpu_mem_pool_percentage,
                                      size_t log_buffers,
                                      size_t buff_buffer_size,
                                      size_t buff_keep_threshold) {
  assert(buff_buffer_size < buff_keep_threshold);
  const topology &topo = topology::getInstance();
  // std::cout << topo << std::endl;

  uint32_t devices = topo.getGpuCount();
  // buffer_manager<T>::h_size =
  //     cpu_mem_pool_percentage *
  //     (topo.getCpuNumaNodes()[0].getMemorySize() / buffer_size);

  uint32_t cores = topo.getCoreCount();

  {
    uint32_t max_numa_id = 0;
    for (const auto &n : topo.getCpuNumaNodes()) {
      max_numa_id = std::max(max_numa_id, n.id);
    }

    terminating = false;
    device_buffs_mutex = new std::mutex[devices];
    device_buffs_cv = new std::condition_variable[devices];
    device_buffs_thrds = new std::thread *[devices];
    device_buffs_pool = new std::deque<T *>[devices];
    release_streams = new cudaStream_t[devices];

    h_pool = new h_pool_t *[cores];
    h_pool_numa = new h_pool_t *[max_numa_id + 1];

    h_d_pool = new pool_t *[devices];

    h_buff_start = new void *[devices];
    h_buff_end = new void *[devices];

    h_h_buff_start = new void *[max_numa_id + 1];
    h_size = new size_t[max_numa_id + 1];

    device_buff = new T **[devices];
    device_buff_size = buff_buffer_size;
    keep_threshold = buff_keep_threshold;

    buffer_cache.clear();
  }

  std::mutex buff_cache;

  std::vector<std::thread> buffer_pool_constrs;
  for (const auto &gpu : topo.getGpus()) {
    buffer_pool_constrs.emplace_back([&gpu, gpu_mem_pool_percentage,
                                      buff_buffer_size] {
      size_t size =
          gpu_mem_pool_percentage * (gpu.getMemorySize() / buffer_size);
      LOG(INFO) << "Using " << size << " " << bytes{buffer_size}
                << "-buffers in GPU " << gpu.id
                << " (Total: " << bytes{size * buffer_size} << "/"
                << bytes{gpu.getMemorySize()} << ")";
      CHECK_LT(buff_buffer_size, size);
      uint32_t j = gpu.id;

      set_exec_location_on_scope d(gpu);

      T *mem = nullptr;
      size_t pitch = 0;
      gpu_run(cudaMallocPitch(&mem, &pitch, h_vector_size * sizeof(T), size));

      std::vector<T *> buffs;

      buffs.reserve(size);
      for (size_t i = 0; i < size; ++i) {
        T *m = (T *)(((char *)mem) + i * pitch);
        // buffer_t * b = cuda_new<buffer_t>(gpu, m, gpu);
        buffs.push_back(m);

        // cout << "Device " << j << " : data = " << m << endl;
        CHECK_EQ(topology::getInstance().getGpuAddressed(m)->id, j);
      }

      pool_t *tmp = cuda_new<pool_t>(gpu, size, buffs, gpu);
      gpu_run(cudaMemcpyToSymbol(pool, &tmp, sizeof(pool_t *)));
      pool_t *tmp2 =
          cuda_new<pool_t>(gpu, size * 1024, std::vector<T *>{}, gpu);
      gpu_run(cudaMemcpyToSymbol(npool, &tmp2, sizeof(pool_t *)));
      gpu_run(cudaMemcpyToSymbol(deviceId, &j, sizeof(int)));
      gpu_run(cudaMemcpyToSymbol(buff_start, &mem, sizeof(void *)));
      void *e = (void *)(((char *)mem) + size * pitch);
      gpu_run(cudaMemcpyToSymbol(buff_end, &e, sizeof(void *)));

      h_d_pool[j] = tmp;
      h_buff_start[j] = mem;
      h_buff_end[j] = e;

      int greatest = 0;
      int lowest = 0;
      gpu_run(cudaDeviceGetStreamPriorityRange(&greatest, &lowest));
      // std::cout << greatest << " " << lowest << std::endl;
      gpu_run(cudaStreamCreateWithPriority(&(release_streams[j]),
                                           cudaStreamNonBlocking, lowest));

      T **bf = nullptr;
      gpu_run(cudaMallocHost(
          &bf, std::max(device_buff_size, keep_threshold) * sizeof(T *)));
      device_buff[j] = bf;

      device_buffs_thrds[j] = new std::thread(dev_buff_manager, j);
      pthread_setname_np(device_buffs_thrds[j]->native_handle(),
                         ("devblockmgr" + std::to_string(j)).c_str());
    });
  }

  for (const auto &cpu : topo.getCpuNumaNodes()) {
    buffer_pool_constrs.emplace_back(
        [&cpu, cpu_mem_pool_percentage, &buff_cache, &topo] {
          size_t cpu_h_size =
              cpu_mem_pool_percentage * (cpu.getMemorySize() / buffer_size);
          buffer_manager<T>::h_size[cpu.id] = cpu_h_size;
          LOG(INFO) << "Using " << h_size << " " << bytes{buffer_size}
                    << "-buffers in CPU " << cpu.id
                    << " (Total: " << bytes{cpu_h_size * buffer_size} << "/"
                    << bytes{cpu.getMemorySize()} << ")";
          set_exec_location_on_scope cu{cpu};

          size_t bytes = h_vector_size * sizeof(T) * cpu_h_size;
          T *mem = (T *)MemoryManager::mallocPinned(bytes);
          CHECK(mem);

          // T * mem;
          // gpu_run(cudaMallocHost(&mem, h_vector_size*sizeof(T)*h_size));
          LOG(INFO) << "Memory at " << mem << "is at node "
                    << topo.getCpuNumaNodeAddressed(mem)->id
                    << " (expected: " << affinity::get().id << ")";
          // assert(topo.getCpuNumaNodeAddressed(mem)->id == cpu.id); //FIXME:
          // fails on power9, should reenable after we fix it

          h_h_buff_start[cpu.id] = mem;

          std::vector<T *> buffs;
          buffs.reserve(cpu_h_size);
          for (size_t j = 0; j < cpu_h_size; ++j) {
            T *m = mem + j * h_vector_size;
            // buffer_t * b = cuda_new<buffer_t>(-1, m, -1);
            buffs.push_back(m);

            m[0] = 0;  // force allocation of first page of each buffer
            // cout << "NUMA " << topo.getCpuNumaNodeAddressed(m)->id << " :
            // data = " << m << endl; assert(topo.getCpuNumaNodeAddressed(m)->id
            // == cpu.id); //FIXME: fails on power9, should reenable after we
            // fix it
          }

          {
            std::lock_guard<std::mutex> guard(buff_cache);
            for (const auto b : buffs) buffer_cache[b] = 0;
          }

          auto *p = new h_pool_t(cpu_h_size, buffs);

          h_pool_numa[cpu.id] = p;

          for (const auto &core : cpu.local_cores) h_pool[core] = p;
        });
  }

  // h_pool_t **numa_h_pools = new h_pool_t *[cpu_numa_nodes];

  // for (int i = 0 ; i < cores ; ++i) numa_node_inited[i] = nullptr;

  // for (int i = 0 ; i < cores ; ++i){
  //     int numa_node = numa_node_of_cpu(i);

  //     if (!numa_node_inited[numa_node]){
  //         cpu_set_t cpuset;
  //         CPU_ZERO(&cpuset);
  //         CPU_SET(i, &cpuset);

  //         T      *mem;
  //         gpu_run(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

  //         std::vector<buffer_t *> buffs;
  //         for (size_t i = 0 ; i < size ; ++i)
  //         buffs.push_back(cuda_new<buffer_t>(-1, mem + i *
  //         buffer_t::capacity(), -1)); numa_node_inited[numa_node] = new
  //         h_pool_t(size, buffs);
  //     }
  //     h_pool[i] = numa_node_inited[numa_node];
  // }

  // T      *mem;
  // gpu_run(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

  // std::vector<buffer_t *> buffs;
  // for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1,
  // mem + i * buffer_t::capacity(), -1)); h_pool = new h_pool_t(size, buffs);

  for (auto &t : buffer_pool_constrs) t.join();

  if (log_buffers) {
    buffer_logger =
        new std::thread{buffer_manager<T>::log_buffers, log_buffers};
    pthread_setname_np(buffer_logger->native_handle(), "block log");
  } else {
    buffer_logger = nullptr;
  }

  // Remote-release GC is only needed when more than one GPU participates.
  // In single-GPU runs, keeping this polling thread active can surface
  // spurious async failures in the remote npool path with no functional gain.
  if (devices > 1) {
    buffer_gc = new std::thread(buffer_manager<T>::find_released_buffers, 50);
    pthread_setname_np(buffer_gc->native_handle(), "block gc");
  } else {
    buffer_gc = nullptr;
  }
}

template <typename T>
__host__ void buffer_manager<T>::destroy() {
  int devices;
  gpu_run(cudaGetDeviceCount(&devices));

  // long cores = sysconf(_SC_NPROCESSORS_ONLN);
  // assert(cores > 0);

  // int cpu_numa_nodes = numa_num_task_nodes();
  CHECK(!terminating) << "Already terminated";
  terminating = true;

  if (buffer_logger) {
    buffer_logger->join();
    delete buffer_logger;
    buffer_logger = nullptr;
  }

  if (buffer_gc) {
    buffer_gc->join();
    delete buffer_gc;
    buffer_gc = nullptr;
  }
  // device_buffs_mutex = new mutex              [devices];
  // device_buffs_pool  = new std::vector<buffer_t *> [devices];
  // release_streams    = new cudaStream_t       [devices];

  // h_pool             = new h_pool_t *         [cores  ];
  // h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

  // device_buff        = new buffer_t**[devices];
  // device_buff_size   = buff_buffer_size;
  // keep_threshold     = buff_keep_threshold;

  // gpu_affinity       = new cpu_set_t[devices];

  // mutex buff_cache;

  const auto &topo = topology::getInstance();

  std::vector<std::thread> buffer_pool_constrs;
  for (const auto &gpu : topo.getGpus()) {
#ifndef NCUDA
    buffer_pool_constrs.emplace_back([&gpu] {
      uint32_t j = gpu.id;
      set_device_on_scope d(j);

      device_buffs_cv[j].notify_all();
      device_buffs_thrds[j]->join();

      std::unique_lock<std::mutex> lock(device_buffs_mutex[j]);

      size_t size = device_buffs_pool[j].size();
      assert(size <= keep_threshold);
      for (size_t i = 0; i < size; ++i)
        device_buff[j][i] = device_buffs_pool[j][i];

      release_buffer_host<<<1, 1, 0, release_streams[j]>>>(
          (void **)device_buff[j], size);
      gpu_run(cudaStreamSynchronize(release_streams[j]));

      pool_t *tmp = nullptr;
      gpu_run(cudaMemcpyFromSymbol(&tmp, pool, sizeof(pool_t *)));
      cuda_delete(tmp);
      gpu_run(cudaMemcpyFromSymbol(&tmp, npool, sizeof(pool_t *)));
      cuda_delete(tmp);

      T *mem = nullptr;
      gpu_run(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void *)));
      gpu_run(cudaFree(mem));

      gpu_run(cudaStreamDestroy(release_streams[j]));

      gpu_run(cudaFreeHost(device_buff[j]));
    });
#else
    assert(false);
#endif
  }

  for (const auto &cpu : topo.getCpuNumaNodes()) {
    buffer_pool_constrs.emplace_back([&cpu] {
      set_exec_location_on_scope cu{cpu};
      MemoryManager::freePinned(h_h_buff_start[cpu.id]);
      delete h_pool_numa[cpu.id];
    });
  }

  for (auto &t : buffer_pool_constrs) t.join();

  //  terminating = false;
  delete[] device_buffs_mutex;
  delete[] device_buffs_cv;
  delete[] device_buffs_thrds;
  delete[] device_buffs_pool;
  delete[] release_streams;

  delete[] h_pool;
  delete[] h_pool_numa;

  delete[] h_d_pool;

  delete[] h_buff_start;
  delete[] h_buff_end;

  delete[] h_h_buff_start;
  delete[] h_size;

  delete[] device_buff;

  buffer_cache.clear();
}

extern "C" {
void *get_buffer(size_t bytes) {
  assert(
      bytes <=
      sizeof(int32_t) *
          buffer_manager<
              int32_t>::h_vector_size);  // FIMXE: buffer manager should be able
                                         // to allocate blocks of arbitary size
  return (void *)buffer_manager<int32_t>::h_get_buffer(-1);
}

void release_buffer(void *buff) {
  buffer_manager<int32_t>::release_buffer((int32_t *)buff);
}

void *get_pinned_buffer(size_t bytes) { return MemoryManager::mallocPinned(bytes); }

void release_pinned_buffer(void *buff) { MemoryManager::freePinned(buff); }
}

template <typename T>
void buffer_manager<T>::dev_buff_manager(int dev) {
#ifndef NCUDA
  set_device_on_scope d(dev);

  while (true) {
    bool sleep = false;
    int added = 0;
    {
      {
        std::unique_lock<std::mutex> lk(device_buffs_mutex[dev]);

        device_buffs_cv[dev].wait(lk, [dev] {
          return (device_buffs_pool[dev].size() < (device_buff_size / 4)) ||
                 terminating;
        });

        if (terminating) break;
      }

      get_buffer_host<<<1, 1, 0, release_streams[dev]>>>(
          (void **)device_buff[dev], device_buff_size);
      gpu_run(cudaStreamSynchronize(release_streams[dev]));

      {
        std::unique_lock<std::mutex> lk(device_buffs_mutex[dev]);

        for (size_t i = 0; i < device_buff_size; ++i) {
          if (device_buff[dev][i]) {
            device_buffs_pool[dev].push_back(device_buff[dev][i]);
            ++added;
          }
        }

        device_buffs_cv[dev].notify_all();

        sleep = device_buffs_pool[dev].empty();
      }
    }

    if (sleep) {
      std::cout << "Sleeping... (" << added << ")" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "Waking..." << std::endl;
    }
    // device_buffs_pool[dev].insert(device_buffs_pool[dev].end(),
    // device_buff[dev], device_buff[dev]+device_buff_size);

    // lk.unlock();
  }
#endif
}

template <typename T>
__host__ void buffer_manager<T>::find_released_buffers(size_t freq) {
  const auto &topo = topology::getInstance();
  uint32_t devices = topo.getGpuCount();
  if (devices <= 0) return;

  // std::ostream &out = std::cerr;
  size_t maxFetchedBuffs = 1024 * 1024;
  auto stage = (T **)MemoryManager::mallocPinned(maxFetchedBuffs * sizeof(T *) *
                                                 devices);

  cudaStream_t strs[devices];

  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    int greatest = 0;
    int lowest = 0;
    gpu_run(cudaDeviceGetStreamPriorityRange(&greatest, &lowest));
    // std::cout << greatest << " " << lowest << std::endl;
    gpu_run(cudaStreamCreateWithPriority(strs + gpu.id, cudaStreamNonBlocking,
                                         lowest));
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds{freq});

#ifndef NCUDA
    for (uint32_t i = 0; i < devices; ++i) {
      set_device_on_scope d(topo.getGpus()[i]);

      auto buffs = stage + maxFetchedBuffs * i;

      // FIXME: Probably we can "snapshot" the npool and read the snapshotted
      // version. Then propagate the npool respectively
      find_released_remote_buffers<<<1, 1, 0, strs[i]>>>((void **)buffs,
                                                         maxFetchedBuffs);
    }
#endif

    bool no_releases = true;

    for (uint32_t i = 0; i < devices; ++i) {
      set_device_on_scope d(topo.getGpus()[i]);

      auto buffs = stage + maxFetchedBuffs * i;

      gpu_run(cudaStreamSynchronize(strs[i]));

      for (size_t j = 0; j < maxFetchedBuffs; ++j) {
        auto b = buffs[j];
        if (!b) break;
        buffer_manager<T>::release_buffer(b);
      }
      no_releases = no_releases && (buffs[0] == nullptr);
    }

    if (no_releases && terminating) break;
  }

  for (uint32_t i = 0; i < devices; ++i)
    gpu_run(cudaStreamSynchronize(strs[i]));

  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    gpu_run(cudaStreamDestroy(strs[gpu.id]));
  }

  MemoryManager::freePinned(stage);
}

size_t get_avail_remote_buffs();

template <typename T>
__host__ void buffer_manager<T>::log_buffers(size_t freq) {
  const auto &topo = topology::getInstance();
  uint32_t devices = topo.getGpuCount();
  // if (devices <= 0) return;

  // std::ostream &out = std::cerr;

  uint32_t cnts[devices];
  cudaStream_t strs[devices];

  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    gpu_run(cudaStreamCreateWithFlags(strs + gpu.id, cudaStreamNonBlocking));
  }

  char progress[]{"-\\|/"};
  size_t iter = 0;

  while (!terminating) {
    std::this_thread::sleep_for(std::chrono::milliseconds{freq});
    std::stringstream out;
    for (uint32_t i = 0; i < devices; ++i) {
      gpu_run(cudaMemcpyAsync(cnts + i, (void *)&(h_d_pool[i]->cnt),
                              sizeof(decltype(pool_t::cnt)), cudaMemcpyDefault,
                              strs[i]));
    }
    out.clear();
    for (uint32_t i = 0; i < devices; ++i)
      gpu_run(cudaStreamSynchronize(strs[i]));
    // out << "\0337\033[H\r";
    for (uint32_t i = 0; i < 80; ++i) out << ' ';
    // out << "\rBuffers on device: ";
    out << "Buffers on device: ";
    for (uint32_t i = 0; i < devices; ++i)
      out << cnts[i] << "(+" << device_buffs_pool[i].size() << ") ";
    for (const auto &s : topo.getCpuNumaNodes()) {
      out << h_pool_numa[s.id]->size_unsafe() << " ";
    }
    for (const auto &node : topo.getCpuNumaNodes()) {
      out << bytes{MemoryManager::cpu_managers[node.index_in_topo]
                       ->usage_unsafe()}
          << " ";
    }
    for (uint32_t i = 0; i < devices; ++i) {
      out << bytes{MemoryManager::gpu_managers[i]->usage_unsafe()} << " ";
    }
    out << get_avail_remote_buffs() << " ";
    out << "\t\t"
        << progress[(iter++) % (sizeof(progress) - 1)];  // for null character
    // out << "\0338";
    // out.flush();
    LOG(INFO) << out.str();
  }

  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d(gpu);
    gpu_run(cudaStreamDestroy(strs[gpu.id]));
  }
}

template <typename T>
__host__ inline T *buffer_manager<T>::h_get_buffer(int dev) {
  if (dev >= 0) {
    std::unique_lock<std::mutex> lock(device_buffs_mutex[dev]);

    device_buffs_cv[dev].wait(
        lock, [dev] { return !device_buffs_pool[dev].empty(); });

    T *ret = device_buffs_pool[dev].back();
    device_buffs_pool[dev].pop_back();
    device_buffs_cv[dev].notify_all();
    return ret;
  } else {
    return get_buffer();
  }
}

template <typename T>
__host__ void buffer_manager<T>::overwrite_bytes(void *buff, const void *data,
                                                 size_t bytes,
                                                 cudaStream_t strm,
                                                 bool blocking) {
#ifndef NCUDA
  if (topology::getInstance().getGpuCount() > 0) {
    gpu_run(cudaMemcpyAsync(buff, data, bytes, cudaMemcpyDefault, strm));
    if (blocking) gpu_run(cudaStreamSynchronize(strm));
    return;
  }
#endif
  // We have to wait here! As otherwise memcpy will be executed immediately!
  gpu_run(cudaStreamSynchronize(strm));
  memcpy(buff, data, bytes);
}

template class buffer_manager<int32_t>;

#ifndef NCUDA
__global__ void GpuHashRearrange_acq_buffs(void **buffs) {
  buffs[blockIdx.x] = proteus::get_buffers();
}

__global__ void GpuHashRearrange_try_acq_buffs(void **buffs) {
  buffs[blockIdx.x] = buffer_manager<int32_t>::try_get_buffer();
}

#endif

#ifdef NCUDA
[[noreturn]]
#endif
void call_GpuHashRearrange_acq_buffs(size_t cnt, cudaStream_t strm,
                                     void **buffs) {
#ifndef NCUDA
  GpuHashRearrange_acq_buffs<<<cnt, 1, 0, strm>>>(
      buffs);  // TODO: wrap it in a nicer way, using the code-gen context
#else
  assert(false);
#endif
}

#ifdef NCUDA
[[noreturn]]
#endif
void call_GpuHashRearrange_try_acq_buffs(size_t cnt, cudaStream_t strm,
                                         void **buffs) {
#ifndef NCUDA
  // Acquire in a single thread to avoid high-contention lock spinning when
  // cnt is large (e.g. thousands of buffers for GpuHashRearrange).
  get_buffer_host<<<1, 1, 0, strm>>>(buffs, static_cast<int>(cnt));
#else
  assert(false);
#endif
}

extern "C" {
void gpu_memset(void *dst, int32_t val, size_t size) {
  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaMemsetAsync(dst, val, size, strm));
  syncAndDestroyStream(strm);
}
}

template <typename T>
typename buffer_manager<T>::pool_t **buffer_manager<T>::h_d_pool;

template <typename T>
std::mutex *buffer_manager<T>::device_buffs_mutex;

template <typename T>
std::thread **buffer_manager<T>::device_buffs_thrds;

template <typename T>
std::condition_variable *buffer_manager<T>::device_buffs_cv;

template <typename T>
bool buffer_manager<T>::terminating;

template <typename T>
std::deque<T *> *buffer_manager<T>::device_buffs_pool;

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
size_t *buffer_manager<T>::h_size;

template <typename T>
std::thread *buffer_manager<T>::buffer_logger;

template <typename T>
std::thread *buffer_manager<T>::buffer_gc;

bool __equalStringObjs_host(StringObject o1, StringObject o2) {
  return o1.len == o2.len && strncmp(o1.start, o2.start, o1.len) == 0;
}

__device__ bool __equalStringObjs_dev(StringObject o1, StringObject o2) {
  if (o1.len != o2.len) return false;
  for (size_t i = 0; i < o1.len; ++i) {
    if (o1.start[i] != o2.start[i]) return false;
  }
  return true;
}

extern "C" {
#if defined(__clang__) && defined(__CUDA__)
__device__ bool equalStringObjs(StringObject o1, StringObject o2) {
  return __equalStringObjs_dev(o1, o2);
}

__host__ bool equalStringObjs(StringObject o1, StringObject o2) {
  return __equalStringObjs_host(o1, o2);
}
#else
__host__ __device__ bool equalStringObjs(StringObject o1, StringObject o2) {
#ifdef __CUDA_ARCH__
  return __equalStringObjs_dev(o1, o2);
#else
  return __equalStringObjs_host(o1, o2);
#endif
}
#endif
}

template <typename T>
__host__ __device__ void log_impl(
    const T &x, decltype(__builtin_FILE()) file = __builtin_FILE(),
    decltype(__builtin_LINE()) line = __builtin_LINE()) {
#ifdef __CUDA_ARCH__
  static_assert(sizeof(T) <= sizeof(int64_t));
  printf("I %s:%u] %" PRId64 "\n", file, line, static_cast<int64_t>(x));
#else
  google::LogMessage(file, line, google::GLOG_INFO).stream() << x;
#endif
}

template <>
__host__ __device__ void log_impl<const char *>(
    const char *const &x, decltype(__builtin_FILE()) file,
    decltype(__builtin_LINE()) line) {
#ifdef __CUDA_ARCH__
  printf("I %s:%u] %s\n", file, line, x);
#else
  google::LogMessage(file, line, google::GLOG_INFO).stream() << x;
#endif
}

extern "C" __host__ __device__ void logi1(bool x,
                                          decltype(__builtin_FILE()) file,
                                          decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}
extern "C" __host__ __device__ void logi8(int8_t x,
                                          decltype(__builtin_FILE()) file,
                                          decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}
extern "C" __host__ __device__ void logi16(int16_t x,
                                           decltype(__builtin_FILE()) file,
                                           decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}
extern "C" __host__ __device__ void logi32(int32_t x,
                                           decltype(__builtin_FILE()) file,
                                           decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}
extern "C" __host__ __device__ void logi64(int64_t x,
                                           decltype(__builtin_FILE()) file,
                                           decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}
extern "C" __host__ __device__ void logi8ptr(const char *x,
                                             decltype(__builtin_FILE()) file,
                                             decltype(__builtin_LINE()) line) {
  log_impl(x, file, line);
}

#pragma clang diagnostic pop
