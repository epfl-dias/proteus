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
//
//#include "operators/gpu/gpu-partitioned-hash-join-chained.hpp"
//#include "codegen/memory/memory-manager.hpp"
//#include "expressions/expressions-generator.hpp"
//#include "operators/gpu/gmonoids.hpp"
//#include "util/gpu/gpu-intrinsics.hpp"

#include <cassert>

#include "common/gpu/gpu-common.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

// #define COMPACT_OFFSETS_
// //#define PARTITION_PAYLOAD

// #define SHMEM_SIZE 4096
// #define HT_LOGSIZE 10

extern __shared__ int int_shared[];

const size_t log2_bucket_size = 12;
const size_t bucket_size = 1 << log2_bucket_size;
const size_t bucket_size_mask = bucket_size - 1;

__device__ int hashd(int val) {
  val = (val >> 16) ^ val;
  val *= 0x85ebca6b;
  val = (val >> 13) ^ val;
  val *= 0xc2b2ae35;
  val = (val >> 16) ^ val;
  return val;
}

union vec4 {
  int4 vec;
  int32_t i[4];
};

__global__ void init_first(int32_t *payload, int32_t *cnt_ptr, uint32_t *chains,
                           uint32_t *buckets_used) {
  uint32_t cnt = *cnt_ptr;

#ifndef PARTITION_PAYLOAD
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < cnt;
       i += blockDim.x * gridDim.x)
    payload[i] = i;
#endif
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < (cnt + bucket_size - 1) / bucket_size - 1;
       i += blockDim.x * gridDim.x)
    chains[i] = i + 1;

  if (threadIdx.x + blockIdx.x * blockDim.x == 0)
    chains[(cnt + bucket_size - 1) / bucket_size - 1] = 0;

  *buckets_used = (cnt + bucket_size - 1) / bucket_size;
}

void call_init_first(size_t grid, size_t block, size_t shmem, cudaStream_t strm,
                     int32_t *payload, int32_t *cnt_ptr, uint32_t *chains,
                     uint32_t *buckets_used) {
  init_first<<<grid, block, shmem, strm>>>(payload, cnt_ptr, chains,
                                           buckets_used);
}

__global__ void init_metadata(uint64_t *heads, uint32_t *chains,
                              int32_t *out_cnts, uint32_t *buckets_used,
                              uint32_t parts, uint32_t buckets_num) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < buckets_num;
       i += blockDim.x * gridDim.x)
    chains[i] = 0;

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < parts;
       i += blockDim.x * gridDim.x) {
    out_cnts[i] = 0;
    heads[i] = (1 << 18) + (((uint64_t)bucket_size_mask) << 32);
  }

  if (threadIdx.x + blockIdx.x * blockDim.x == 0) *buckets_used = parts;
}

void call_init_metadata(size_t grid, size_t block, size_t shmem,
                        cudaStream_t strm, uint64_t *heads, uint32_t *chains,
                        int32_t *out_cnts, uint32_t *buckets_used,
                        uint32_t parts, uint32_t buckets_num) {
  init_metadata<<<grid, block, shmem, strm>>>(heads, chains, out_cnts,
                                              buckets_used, parts, buckets_num);
}

__global__ void compute_bucket_info(uint32_t *bucket_info, uint32_t *chains,
                                    int32_t *out_cnts, uint32_t log_parts) {
  uint32_t parts = 1 << log_parts;

  for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < parts;
       p += gridDim.x * blockDim.x) {
    uint32_t cur = p;
    int32_t cnt = out_cnts[p];

    while (cnt > 0) {
      uint32_t local_cnt = (cnt >= bucket_size) ? bucket_size : cnt;
      uint32_t val = (p << 15) + local_cnt;

      uint32_t next = chains[cur];
      bucket_info[cur] = val;

      cur = next;
      cnt -= bucket_size;
    }
  }
}

void call_compute_bucket_info(size_t grid, size_t block, size_t shmem,
                              cudaStream_t strm, uint32_t *bucket_info,
                              uint32_t *chains, int32_t *out_cnts,
                              uint32_t log_parts) {
  compute_bucket_info<<<grid, block, shmem, strm>>>(bucket_info, chains,
                                                    out_cnts, log_parts);
}

__global__ void verify_decomposition(uint32_t *bucket_info,
                                     uint32_t *buckets_used) {
  int cnt = *buckets_used;
  int sum = 0;

  for (int i = 0; i < cnt; i++) {
    if (bucket_info[i] != 0) {
      sum += bucket_info[i] & ((1 << 15) - 1);
    }
  }

  printf("%d\n", sum);
}

__global__ void verify_partitions(int *S_in, int *P_in, uint32_t *chains,
                                  int *out_cnts, int *S_out, int *P_out,
                                  int log_parts, int *error_cnt) {
  for (int p = blockIdx.x; p < (1 << log_parts); p += gridDim.x) {
    int current_bucket = p;
    int remaining = out_cnts[p];

    while (remaining > 0) {
      int cnt = (remaining > bucket_size) ? bucket_size : remaining;

      for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        int offset = current_bucket * bucket_size + i;
        int payload = P_out[offset];

        if (S_out[offset] != S_in[payload]) {
          printf("Errooooor %d %d %d %d %d %d!\n", S_out[offset], payload,
                 S_in[payload], P_in[payload], cnt, current_bucket);
          atomicAdd(error_cnt, 1);
        }
      }

      current_bucket = chains[p];
      remaining -= 4096;
    }
  }
}
__global__ void printpart(int *S_in, uint32_t *chains, int *out_cnts, int p) {
  int bucket = p;
  int cnt = out_cnts[p];

  while (cnt > 0) {
    int local_cnt = (cnt < bucket_size) ? cnt : bucket_size;

    for (int i = 0; i < local_cnt; i++)
      printf("%d\n", S_in[bucket * bucket_size + i]);

    bucket = chains[bucket];
    cnt -= bucket_size;
  }
}

__global__ void decompose_chains(uint32_t *bucket_info, uint32_t *chains,
                                 int32_t *out_cnts, uint32_t log_parts,
                                 int threshold) {
  uint32_t parts = 1 << log_parts;

  for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < parts;
       p += gridDim.x * blockDim.x) {
    uint32_t cur = p;
    int32_t cnt = out_cnts[p];
    uint32_t first_cnt = (cnt >= threshold) ? threshold : cnt;
    int32_t cutoff = 0;

    while (cnt > 0) {
      cutoff += bucket_size;
      cnt -= bucket_size;

      uint32_t next = chains[cur];

      if (cutoff >= threshold && cnt > 0) {
        uint32_t local_cnt = (cnt >= threshold) ? threshold : cnt;
        bucket_info[next] = (p << 15) + local_cnt;
        // printf("%d!!\n", next);
        chains[cur] = 0;
        cutoff = 0;
      } else if (next != 0) {
        bucket_info[next] = 0;
      }

      cur = next;
    }

    bucket_info[p] = (p << 15) + first_cnt;
  }
}

void call_decompose_chains(size_t grid, size_t block, size_t shmem,
                           cudaStream_t strm, uint32_t *bucket_info,
                           uint32_t *chains, int32_t *out_cnts,
                           uint32_t log_parts, int threshold) {
  decompose_chains<<<grid, block, shmem, strm>>>(bucket_info, chains, out_cnts,
                                                 log_parts, threshold);
}

namespace proteus {
__global__ void build_partitions(const int32_t *__restrict__ S,
                                 const int32_t *__restrict__ P,
                                 const uint32_t *__restrict__ bucket_info,
                                 uint32_t *__restrict__ buckets_used,
                                 uint64_t *heads, uint32_t *__restrict__ chains,
                                 int32_t *__restrict__ out_cnts,
                                 int32_t *__restrict__ output_S,
                                 int32_t *__restrict__ output_P,
                                 uint32_t S_log_parts, uint32_t log_parts,
                                 uint32_t first_bit, uint32_t *bucket_num_ptr) {
  assert((((size_t)bucket_size) + ((size_t)blockDim.x) * gridDim.x) <
         (((size_t)1) << 32));
  // assert((parts & (parts - 1)) == 0);
  const uint32_t S_parts = 1 << S_log_parts;
  const uint32_t parts = 1 << log_parts;
  const int32_t parts_mask = parts - 1;

  uint32_t buckets_num = *bucket_num_ptr;

  // get shared memory pointer

  uint32_t *router = (uint32_t *)int_shared;  //[1024*4 + parts];

  // initialize shared memory

  for (size_t j = threadIdx.x; j < parts; j += blockDim.x)
    router[1024 * 4 + parts + j] = 0;

  if (threadIdx.x == 0) router[0] = 0;

  __syncthreads();

  // loop over the blocks

  for (size_t i = blockIdx.x; i < buckets_num; i += gridDim.x) {
    uint32_t info = bucket_info[i];
    uint32_t cnt = info & ((1 << 15) - 1);
    uint32_t pid = info >> 15;

    vec4 thread_vals = *(
        reinterpret_cast<const vec4 *>(S + bucket_size * i + 4 * threadIdx.x));

    uint32_t thread_keys[4];

// compute local histogram
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (4 * threadIdx.x + k < cnt) {
        uint32_t partition =
            (hashd(thread_vals.i[k]) >> first_bit) & parts_mask;

        atomicAdd(router + (1024 * 4 + parts + partition), 1);

        thread_keys[k] = partition;
      } else {
        thread_keys[k] = 0;
      }
    }

    __syncthreads();

    // update bucket chain

    for (size_t j = threadIdx.x; j < parts; j += blockDim.x) {
      uint32_t cnt = router[1024 * 4 + parts + j];

      if (cnt > 0) {
        atomicAdd(out_cnts + (pid << log_parts) + j, cnt);  // Is this needed ?

        uint32_t pcnt;
        uint32_t bucket;
        uint32_t next_buck;

        // assert(cnt <= bucket_size);
        bool repeat = true;
        uint32_t thrdmask = __activemask();

        while (__any_sync(
            repeat, thrdmask)) {  // without the "repeat" variable, the compiler
                                  // probably moves the "if(pcnt < bucket_size)"
                                  // block out of the loop, which creates a
          // deadlock using the repeat variable, it should
          // convince the compiler that it should not
          if (repeat) {
            uint64_t old_heads = __atomic_fetch_add(
                heads + (pid << log_parts) + j, ((uint64_t)cnt) << 32,
                __ATOMIC_SEQ_CST);  // atomicAdd(heads + (pid << log_parts) + j,
                                    // ((uint64_t) cnt) << 32);

            atomicMin(heads + (pid << log_parts) + j,
                      ((uint64_t)(2 * bucket_size)) << 32);

            pcnt = ((uint32_t)(old_heads >> 32));
            bucket = (uint32_t)old_heads;

            // now there are two cases:
            // 2) old_heads.cnt >  bucket_size ( => locked => retry)
            // if (pcnt       >= bucket_size) continue;

            if (pcnt < bucket_size) {
              // 1) old_heads.cnt <= bucket_size

              // check if the bucket was filled
              if (pcnt + cnt >= bucket_size) {  //&& pcnt <  bucket_size
                // assert(pcnt + cnt < 2*bucket_size);
                // must replace bucket!

                if (bucket < (1 << 18)) {
                  next_buck = atomicAdd(buckets_used, 1);
                  chains[bucket] = next_buck;
                } else {
                  next_buck = (pid << log_parts) + j;
                }

                // assert(next_buck >= parts);

                // assert(pcnt + cnt - bucket_size >= 0);
                // assert(pcnt + cnt - bucket_size <  bucket_size);

                uint64_t tmp =
                    next_buck + (((uint64_t)(pcnt + cnt - bucket_size)) << 32);

                // assert(((uint32_t) (tmp >> 32)) < bucket_size);

                // atomicExch(heads + (pid << log_parts) + j, tmp); //also
                // zeroes the cnt!
                __atomic_exchange_n(heads + (pid << log_parts) + j, tmp,
                                    __ATOMIC_SEQ_CST);
              } else {
                next_buck = bucket;
              }

              repeat = false;
            }
          }

          __syncwarp(thrdmask);
        }

        // NOTE shared memory requirements can be relaxed, but when moving the
        // two last "rows" one up, we get a 10% performance penalty! This needs
        // a little bit more investigation
        router[1024 * 4 + j] = atomicAdd(router, cnt);
        router[1024 * 4 + parts + j] = 0;  // cnt;//pcnt     ;
        router[1024 * 4 + 2 * parts + j] = (bucket << log2_bucket_size) + pcnt;
        router[1024 * 4 + 3 * parts + j] = next_buck << log2_bucket_size;
      }
    }

    __syncthreads();

    uint32_t total_cnt = router[0];

    __syncthreads();

// calculate target positions for block-wise shuffle
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (4 * threadIdx.x + k < cnt)
        thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);
    }

// perform the shuffle
#pragma unroll
    for (int k = 0; k < 4; ++k)
      if (4 * threadIdx.x + k < cnt) router[thread_keys[k]] = thread_vals.i[k];

    __syncthreads();

    int32_t thread_parts[4];

// write out partition
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (threadIdx.x + 1024 * k < total_cnt) {
        int32_t val = router[threadIdx.x + 1024 * k];
        uint32_t partition = (hashd(val) >> first_bit) & parts_mask;

        uint32_t cnt = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);

        uint32_t bucket = router[1024 * 4 + 2 * parts + partition];
        // uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];

        if (((bucket + cnt) ^ bucket) & ~bucket_size_mask) {
          uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
          cnt = ((bucket + cnt) & bucket_size_mask);
          bucket = next_buck;
        }

        bucket += cnt;

        output_S[bucket] = val;

        thread_parts[k] = partition;
      }
    }

    __syncthreads();

    thread_vals = *(
        reinterpret_cast<const vec4 *>(P + i * bucket_size + 4 * threadIdx.x));

// perform the shuffle
#pragma unroll
    for (int k = 0; k < 4; ++k)
      if (4 * threadIdx.x + k < cnt) {
        router[thread_keys[k]] = thread_vals.i[k];
      }

    __syncthreads();

// write out payload
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (threadIdx.x + 1024 * k < total_cnt) {
        int32_t val = router[threadIdx.x + 1024 * k];

        int32_t partition = thread_parts[k];

        uint32_t cnt = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);

        uint32_t bucket = router[1024 * 4 + 2 * parts + partition];
        // uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];

        if (((bucket + cnt) ^ bucket) & ~bucket_size_mask) {
          uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
          cnt = ((bucket + cnt) & bucket_size_mask);
          bucket = next_buck;
        }
        bucket += cnt;

        output_P[bucket] = val;
      }
    }

    // re-init
    if (threadIdx.x == 0) router[0] = 0;
  }
}

}

void call_build_partitions(size_t grid, size_t block, size_t shmem,
                           cudaStream_t strm, const int32_t *__restrict__ S,
                           const int32_t *__restrict__ P,
                           const uint32_t *__restrict__ bucket_info,
                           uint32_t *__restrict__ buckets_used, uint64_t *heads,
                           uint32_t *__restrict__ chains,
                           int32_t *__restrict__ out_cnts,
                           int32_t *__restrict__ output_S,
                           int32_t *__restrict__ output_P, uint32_t S_log_parts,
                           uint32_t log_parts, uint32_t first_bit,
                           uint32_t *bucket_num_ptr) {
  proteus::build_partitions<<<grid, block, shmem, strm>>>(
      S, P, bucket_info, buckets_used, heads, chains, out_cnts, output_S,
      output_P, S_log_parts, log_parts, first_bit, bucket_num_ptr);
}
