//
// Created by arturo on 10/16/25.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <climits>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "platform/common/gpu/gpu-common.hpp"
#include "platform/util/profiling.hpp"

using namespace std;
using namespace cub;

static inline std::string csvEscape(const std::string& input) {
    bool needs_quotes = false;
    for (char c : input) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return input;
    }
    std::string out;
    out.reserve(input.size() + 2);
    out.push_back('"');
    for (char c : input) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

template <typename T>
static inline std::string toCsvString(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

static inline void csvOut(const std::string& section,
                          const std::string& run,
                          const std::string& field,
                          const std::string& value,
                          std::ostream& os = std::cout) {
    os << csvEscape(section) << "," << csvEscape(run) << "," << csvEscape(field) << "," << csvEscape(value) << "\n";
}

template <typename T>
static inline void csvOut(const std::string& section,
                          const std::string& run,
                          const std::string& field,
                          const T& value,
                          std::ostream& os = std::cout) {
    csvOut(section, run, field, toCsvString(value), os);
}

struct ScopedProfilingWindow {
    explicit ScopedProfilingWindow(const bool use_proteus_profiling_control)
        : use_proteus_control(use_proteus_profiling_control) {
        if (use_proteus_control) {
            profiling::resume();
        } else {
            (void)cudaProfilerStart();
        }
    }
    ~ScopedProfilingWindow() {
        if (use_proteus_control) {
            profiling::pause();
        } else {
            (void)cudaProfilerStop();
        }
    }
    ScopedProfilingWindow(const ScopedProfilingWindow&) = delete;
    ScopedProfilingWindow& operator=(const ScopedProfilingWindow&) = delete;
private:
    bool use_proteus_control;
};

// static inline void csvMessage(const std::string& section,
//                               const std::string& run,
//                               const std::string& message,
//                               std::ostream& os = std::cout) {
//     csvOut(section, run, "message", message, os);
// }

CachingDeviceAllocator g_allocator(true, false);

#define THREADS_PER_TILE 256
#define ELEMS_PER_THREAD 4
#define ELEMS_PER_TILE (THREADS_PER_TILE * ELEMS_PER_THREAD)
#define MIN_COLUMNS 4
#define NUM_RUNS 1
#define RFOR_SHARED_TILES_PER_BLOCK 4
#define D_ARRAY_ALIGNMENT 16

// Arguments
#define FIRST_COL_ARG 1

typedef unsigned long long int sum_type;
typedef unsigned long long int nsecs;
constexpr double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;
constexpr double BYTES_PER_GB = 1000.0 * 1000.0 * 1000.0;

constexpr int SSB_Q11_COL_ORDERDATE = 0;
constexpr int SSB_Q11_COL_QUANTITY = 1;
constexpr int SSB_Q11_COL_EXTENDEDPRICE = 2;
constexpr int SSB_Q11_COL_DISCOUNT = 3;
constexpr int SSB_Q11_NUM_COLS = 4;
constexpr int SSB_Q11_NUM_SUMS = 1;
constexpr int SSB_Q11_ORDERDATE_GT = 19930000;
constexpr int SSB_Q11_ORDERDATE_LT = 19940000;
constexpr int SSB_Q11_DISCOUNT_MIN = 1;
constexpr int SSB_Q11_DISCOUNT_MAX = 3;
constexpr int SSB_Q11_QUANTITY_MAX_EXCLUSIVE = 25;

template<typename T>
__host__ __device__ constexpr T constMax(const T a, const T b) {
    return (a > b) ? a : b;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__host__ __device__ constexpr size_t loadRBinPackDecodeWords() {
    return static_cast<size_t>(BLOCK_THREADS) * ITEMS_PER_THREAD;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__host__ __device__ constexpr size_t loadRBinPackWorkingOffsetWords() {
    using BlockExchange = cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
    constexpr size_t decode_words = loadRBinPackDecodeWords<BLOCK_THREADS, ITEMS_PER_THREAD>();
    constexpr size_t scan_words = (sizeof(typename BlockScan::TempStorage) + sizeof(uint) - 1) / sizeof(uint);
    constexpr size_t exchange_words = (sizeof(typename BlockExchange::TempStorage) + sizeof(uint) - 1) / sizeof(uint);
    return constMax(decode_words, constMax(scan_words, exchange_words));
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__host__ __device__ constexpr size_t loadRBinPackScratchBytes() {
    constexpr size_t decode_words = loadRBinPackDecodeWords<BLOCK_THREADS, ITEMS_PER_THREAD>();
    constexpr size_t working_offset_words = loadRBinPackWorkingOffsetWords<BLOCK_THREADS, ITEMS_PER_THREAD>();
    // Scratch layout:
    // [CUB temp storage/working base][decode working area][staged packed payload (+1 sentinel word)].
    // A single staging region is reused first for values and then for run-lengths.
    constexpr size_t payload_stage_words = decode_words + 1;
    constexpr size_t scratch_bytes =
        (working_offset_words + decode_words + payload_stage_words) * sizeof(uint);
    constexpr size_t scratch_align = alignof(sum_type);
    return ((scratch_bytes + scratch_align - 1) / scratch_align) * scratch_align;
}

enum ExecutionMode {
    RFOR_SHARED,
    RFOR_GLOBAL,
    RFOR_REGISTERS,
    RFOR_PREDECODE,
    RFOR_WARP,
    UNCOMPRESSED_COLUMN,
    UNCOMPRESSED_TUPLE
};

struct RLETileMetadata {
    uint32_t val_reference;
    uint32_t val_bitwidth;
    uint32_t val_count;
    uint32_t rl_reference;
    uint32_t rl_bitwidth;
    uint32_t rl_count;
};

struct RLEColumn {
    std::vector<std::vector<char>> compressed_chunks;
    size_t num_values;
};

__forceinline__ __device__ nsecs globaltimer64() {
    nsecs t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__host__ __device__ __forceinline__ bool passesSSBQ11(
    const int orderdate,
    const int quantity,
    const int discount) {
    return orderdate > SSB_Q11_ORDERDATE_GT &&
           orderdate < SSB_Q11_ORDERDATE_LT &&
           discount >= SSB_Q11_DISCOUNT_MIN &&
           discount <= SSB_Q11_DISCOUNT_MAX &&
           quantity < SSB_Q11_QUANTITY_MAX_EXCLUSIVE;
}

__host__ __device__ __forceinline__ sum_type revenueSSBQ11(
    const int extended_price,
    const int discount) {
    return static_cast<sum_type>(extended_price) * static_cast<sum_type>(discount);
}

tuple<nsecs, nsecs, nsecs, nsecs> longest_processing_time_first_compressed(
    vector<tuple<nsecs, nsecs, nsecs, nsecs>> &block_runtimes, const int max_blocks) {

    // Sort runtimes by the first element, which should be the total runtime of each block
    sort(block_runtimes.begin(), block_runtimes.end(), greater<>());

    vector<tuple<nsecs, nsecs, nsecs, nsecs>> slot_runtimes(max_blocks);
    fill(slot_runtimes.begin(), slot_runtimes.end(), make_tuple(0, 0, 0, 0));
    for (const auto& block_runtime : block_runtimes) {
        auto min_runtime_slot = min_element(slot_runtimes.begin(), slot_runtimes.end());
        *min_runtime_slot = make_tuple(get<0>(*min_runtime_slot) + get<0>(block_runtime),
                                       get<1>(*min_runtime_slot) + get<1>(block_runtime),
                                       get<2>(*min_runtime_slot) + get<2>(block_runtime),
                                       get<3>(*min_runtime_slot) + get<3>(block_runtime));
    }

    return *max_element(slot_runtimes.begin(), slot_runtimes.end());
}

nsecs longest_processing_time_first_uncompressed(vector<nsecs>& block_runtimes, const int max_blocks) {
    sort(block_runtimes.begin(), block_runtimes.end(), greater<>());

    vector<nsecs> slot_runtimes(max_blocks);
    fill(slot_runtimes.begin(), slot_runtimes.end(), 0);
    for (const auto& block_runtime : block_runtimes) {
        auto min_runtime_slot = min_element(slot_runtimes.begin(), slot_runtimes.end());
        *min_runtime_slot += block_runtime;
    }

    return *max_element(slot_runtimes.begin(), slot_runtimes.end());
}

__forceinline__ __device__ int decodeElementRBin(int i, const uint32_t* data_block, uint reference, uint bitwidth) {
    if (bitwidth == 0) return reference;

    uint start_bitindex = (bitwidth * i);
    uint start_intindex = (start_bitindex >> 5);
    start_bitindex = start_bitindex & (32-1);

    uint element = 0;
    const unsigned long long mask = (bitwidth == 32) ? 0xffffffffULL : ((1ULL << bitwidth) - 1ULL);

    if (start_bitindex + bitwidth <= 32) {
        element = static_cast<uint>((static_cast<unsigned long long>(data_block[start_intindex]) >> start_bitindex) & mask);
    } else {
        const unsigned long long element_block =
            (static_cast<unsigned long long>(data_block[start_intindex + 1]) << 32) | data_block[start_intindex];
        element = static_cast<uint>((element_block >> start_bitindex) & mask);
    }

    return reference + element;
}

__forceinline__ __device__ int decodeElementRBinBounded(
    int i,
    const uint32_t* data_block,
    uint packed_words,
    uint reference,
    uint bitwidth) {
    if (bitwidth == 0) return reference;

    uint start_bitindex = (bitwidth * i);
    uint start_intindex = (start_bitindex >> 5);
    start_bitindex = start_bitindex & (32 - 1);

    if (start_intindex >= packed_words) return reference;

    const uint32_t low_word = data_block[start_intindex];
    unsigned long long element_block = low_word;
    if (start_bitindex + bitwidth > 32 && (start_intindex + 1) < packed_words) {
        element_block |=
            (static_cast<unsigned long long>(data_block[start_intindex + 1]) << 32);
    }

    const unsigned long long mask =
        (bitwidth >= 32) ? 0xFFFFFFFFULL : ((1ULL << bitwidth) - 1ULL);
    const uint32_t element = static_cast<uint32_t>((element_block >> start_bitindex) & mask);
    return reference + element;
}

__forceinline__ __device__ int decodeRLEElementSequential(
    int local_idx,
    const RLETileMetadata* metadata,
    const uint32_t* val_data,
    const uint32_t* rl_data) {

    const int run_count = static_cast<int>(metadata->val_count);
    if (run_count == 0) {
        return 0;
    }

    int run_idx = 0;
    int run_end = decodeElementRBin(0, rl_data, metadata->rl_reference, metadata->rl_bitwidth);

    while (run_idx + 1 < run_count && local_idx >= run_end) {
        ++run_idx;
        run_end += decodeElementRBin(run_idx, rl_data, metadata->rl_reference, metadata->rl_bitwidth);
    }

    return decodeElementRBin(run_idx, val_data, metadata->val_reference, metadata->val_bitwidth);
}

__forceinline__ __device__ int decodeRLEElementCursor(
    int local_idx,
    const RLETileMetadata* metadata,
    const uint32_t* val_data,
    const uint32_t* rl_data,
    int* run_idx,
    int* run_end) {

    const int run_count = static_cast<int>(metadata->val_count);
    if (run_count == 0) {
        return 0;
    }

    // Cursor state for this (thread, column): current run index and exclusive run end.
    int idx = *run_idx;
    int end = *run_end;
    // Sentinel for first access in this tile/column by this thread.
    if (end == 0) {
        idx = 0;
        end = decodeElementRBin(0, rl_data, metadata->rl_reference, metadata->rl_bitwidth);
    }

    // Move forward until local_idx belongs to the current run.
    // This is the key amortization: we never rescan old runs for this thread.
    while (idx + 1 < run_count && local_idx >= end) {
        ++idx;
        end += decodeElementRBin(idx, rl_data, metadata->rl_reference, metadata->rl_bitwidth);
    }

    // Persist cursor for the next local_idx processed by this thread.
    *run_idx = idx;
    *run_end = end;
    return decodeElementRBin(idx, val_data, metadata->val_reference, metadata->val_bitwidth);
}

__forceinline__ __device__ int findRunBinarySearch(int local_idx, const int* run_ends, int run_count) {
    int left = 0;
    int right = run_count - 1;
    while (left < right) {
        int mid = left + ((right - left) >> 1);
        if (run_ends[mid] > local_idx) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

__forceinline__ __device__ int findRunCumulativeBinarySearchPacked(
    int local_idx,
    const uint32_t* cumulative_rl,
    uint32_t rl_reference,
    uint32_t rl_bitwidth,
    int run_count) {
    int left = 0;
    int right = run_count - 1;
    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        const int run_end = decodeElementRBin(mid, cumulative_rl, rl_reference, rl_bitwidth);
        if (run_end > local_idx) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

__forceinline__ __device__ int decodeRLEElementCumulativeBinarySearch(
    int local_idx,
    const RLETileMetadata* metadata,
    const uint32_t* val_data,
    const uint32_t* cumulative_rl) {
    const int run_count = static_cast<int>(metadata->val_count);
    if (run_count == 0) {
        return 0;
    }

    const int run_idx = findRunCumulativeBinarySearchPacked(
        local_idx,
        cumulative_rl,
        metadata->rl_reference,
        metadata->rl_bitwidth,
        run_count);

    return decodeElementRBin(run_idx, val_data, metadata->val_reference, metadata->val_bitwidth);
}

__forceinline__ __device__ int findRunCumulativeWithWarpCursor(
    int local_idx,
    int batch_start_local_idx,
    const uint32_t* cumulative_rl,
    uint32_t rl_reference,
    uint32_t rl_bitwidth,
    int run_count,
    int* cursor_ptr) {
    const int lane = threadIdx.x & 31;
    int cursor = *cursor_ptr;
    if (cursor < 0) cursor = 0;
    if (cursor >= run_count) cursor = run_count - 1;

    while (true) {
        const int slot = cursor + lane;
        int run_end = INT_MAX;
        if (slot < run_count) {
            run_end = decodeElementRBin(slot, cumulative_rl, rl_reference, rl_bitwidth);
        }

        // Use the batch start as the threshold so we never skip runs needed by lower lanes.
        // This gives a lower bound cursor shared by all lanes in this batch.
        const unsigned has_candidate = __ballot_sync(0xffffffff, slot < run_count && run_end > batch_start_local_idx);
        if (has_candidate != 0) {
            cursor += (__ffs(has_candidate) - 1);
            break;
        }
        if (cursor + 32 >= run_count) {
            cursor = run_count - 1;
            break;
        }
        cursor += 32;
    }

    int left = cursor;
    int right = run_count - 1;

    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        const int run_end = decodeElementRBin(mid, cumulative_rl, rl_reference, rl_bitwidth);
        if (run_end > local_idx) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    int max_run = left;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const int other = __shfl_down_sync(0xffffffff, max_run, off);
        if (other > max_run) {
            max_run = other;
        }
    }
    if (lane == 0) {
        *cursor_ptr = max_run;
    }
    __syncwarp();

    return left;
}

__forceinline__ __device__ int decodeRLEElementCumulativeWarpCursor(
    int local_idx,
    int batch_start_local_idx,
    const RLETileMetadata* metadata,
    const uint32_t* val_data,
    const uint32_t* cumulative_rl,
    int* cursor_ptr) {
    const int run_count = static_cast<int>(metadata->val_count);
    if (run_count == 0) {
        return 0;
    }

    const int run_idx = findRunCumulativeWithWarpCursor(
        local_idx,
        batch_start_local_idx,
        cumulative_rl,
        metadata->rl_reference,
        metadata->rl_bitwidth,
        run_count,
        cursor_ptr);

    return decodeElementRBin(run_idx, val_data, metadata->val_reference, metadata->val_bitwidth);
}

// TODO: Seems to be accessing rl_data out of bounds!
// metadata = 0x7c76b958803c
// val_data = 0x7c76b9588054
// rl_data = 0x7c76b9588074
// shared_buffer = 0x7c7815001000
// items_value = 0x7c7813fffbe4
// Address 0x7c76b9588084 is out of bounds
// Closest to rl_data
// TODO: We have a contiguous block of memory that looks like |metadata|val_data|rl_data|metadata|val_data|rl_data|...
// TODO: and it seems like we're unpacking past the last bit of data
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack(
    RLETileMetadata* metadata,
    uint32_t num_decode,
    uint32_t val_words,
    uint32_t rl_words,
    uint32_t* val_data,
    uint32_t* rl_data,
    uint* shared_buffer,
    int (&items_value)[ITEMS_PER_THREAD]) {

    // int check_thread = 2;
    // int check_block = 562494;

    // if (threadIdx.x == check_thread && blockIdx.x == check_block) {
    //     printf("metadata = %p\n", metadata);
    //     printf("val_data = %p\n", val_data);
    //     printf("rl_data = %p\n", rl_data);
    //     printf("shared_buffer = %p\n", shared_buffer);
    //     printf("items_value = %p\n", items_value);
    // }
    // __syncthreads();

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    uint count = metadata->val_count;

    int items_run_length[ITEMS_PER_THREAD];

    constexpr size_t decode_words = loadRBinPackDecodeWords<BLOCK_THREADS, ITEMS_PER_THREAD>();
    constexpr size_t working_offset_words = loadRBinPackWorkingOffsetWords<BLOCK_THREADS, ITEMS_PER_THREAD>();
    constexpr size_t payload_stage_offset_words = working_offset_words + decode_words;
    uint32_t* staged_data = shared_buffer + payload_stage_offset_words;

    // Stage packed value payload once from global memory, then decode from shared memory.
    for (uint idx = threadIdx.x; idx < val_words; idx += BLOCK_THREADS) {
        staged_data[idx] = val_data[idx];
    }
    if (threadIdx.x == 0) {
        staged_data[val_words] = 0;
    }
    __syncthreads();

    uint decode_idx = static_cast<uint>(threadIdx.x);
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && decode_idx < count) {
            items_value[i] = decodeElementRBin(
                decode_idx, staged_data, metadata->val_reference, metadata->val_bitwidth);
        } else {
            items_value[i] = 0;
        }
        decode_idx += num_decode;
    }
    __syncthreads();

    // Reuse the same staging region for run-length payload.
    for (uint idx = threadIdx.x; idx < rl_words; idx += BLOCK_THREADS) {
        staged_data[idx] = rl_data[idx];
    }
    if (threadIdx.x == 0) {
        staged_data[rl_words] = 0;
    }
    __syncthreads();

    decode_idx = static_cast<uint>(threadIdx.x);
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && decode_idx < count) {
            items_run_length[i] = decodeElementRBin(
                decode_idx, staged_data, metadata->rl_reference, metadata->rl_bitwidth);
        } else {
            items_run_length[i] = 0;
        }
        decode_idx += num_decode;
    }

    __syncthreads();

    typename BlockScan::TempStorage* temp_storage_scan =
        reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage* temp_storage_exchange =
        reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    uint32_t* working_area = shared_buffer + working_offset_words;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < decode_words) {
            working_area[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
        }
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < decode_words) {
            working_area[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x * ITEMS_PER_THREAD + i < decode_words) {
            items_run_length[i] = working_area[threadIdx.x * ITEMS_PER_THREAD + i];
        } else {
            items_run_length[i] = 0;
        }
    }
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();
    BlockExchange(*temp_storage_exchange).BlockedToStriped(items_run_length);
    __syncthreads();

    uint rle_offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode && threadIdx.x + rle_offset < decode_words) {
            working_area[threadIdx.x + rle_offset] = items_value[i];
        }
        rle_offset += num_decode;
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < decode_words) {
            items_value[i] = working_area[items_run_length[i]];
        } else {
            items_value[i] = 0;
        }
    }
    __syncthreads();
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_tuple(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    int* d_tuples,
    sum_type* d_sums,
    nsecs* d_block_decompress_runtimes,
    nsecs* d_block_partial_sums_runtimes,
    nsecs* d_block_reduce_runtimes,
    int filter,
    size_t tiles_per_column,
    size_t num_tuples,
    int num_columns,
    int num_sums) {
    (void)filter;
    (void)num_sums;

    nsecs time_start = 0;
    if (threadIdx.x == 0) {
        time_start = globaltimer64();
    }

    size_t tile_idx = blockIdx.x;
    if (tile_idx >= tiles_per_column) return;

    extern __shared__ unsigned char shared_buffer[];
    const size_t tuple_size = sizeof(int) * num_columns;
    const size_t tuples_size = tuple_size * ELEMS_PER_TILE;
    const size_t partial_sum_size = sizeof(sum_type) * num_sums;
    constexpr size_t load_rbin_scratch_size = loadRBinPackScratchBytes<BLOCK_THREADS, ITEMS_PER_THREAD>();
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    auto column_elements = reinterpret_cast<int*>(shared_buffer);
    auto decompression_scratch = reinterpret_cast<uint*>(shared_buffer + tuples_size);
    auto partial_sums = reinterpret_cast<sum_type*>(
        shared_buffer + tuples_size + load_rbin_scratch_size + partial_sum_size * threadIdx.x);
    auto temp_storage_reduce = reinterpret_cast<typename BlockReduce::TempStorage*>(shared_buffer);
    // Cache per-column descriptor data once per block for SSB Q1.1's fixed 4 columns.
    __shared__ RLETileMetadata s_metadata[MIN_COLUMNS];
    __shared__ uint32_t* s_val_data[MIN_COLUMNS];
    __shared__ uint32_t* s_rl_data[MIN_COLUMNS];
    __shared__ uint32_t s_num_decode[MIN_COLUMNS];
    __shared__ uint32_t s_val_words[MIN_COLUMNS];
    __shared__ uint32_t s_rl_words[MIN_COLUMNS];

    if (num_columns <= MIN_COLUMNS && threadIdx.x < num_columns) {
        const size_t ptr_index = static_cast<size_t>(threadIdx.x) * tiles_per_column + tile_idx;
        RLETileMetadata* metadata = d_metadata_ptrs[ptr_index];
        s_metadata[threadIdx.x] = *metadata;
        s_val_data[threadIdx.x] = d_val_data_ptrs[ptr_index];
        s_rl_data[threadIdx.x] = d_rl_data_ptrs[ptr_index];
        const uint32_t count = metadata->val_count;
        s_num_decode[threadIdx.x] = (count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
        s_val_words[threadIdx.x] = (count * metadata->val_bitwidth + 31) >> 5;
        s_rl_words[threadIdx.x] = (metadata->rl_count * metadata->rl_bitwidth + 31) >> 5;
    }
    __syncthreads();

    // Process columns sequentially
    for (int col = 0; col < num_columns; col++) {
        RLETileMetadata* metadata;
        uint32_t* val_data;
        uint32_t* rl_data;
        uint32_t num_decode;
        uint32_t val_words;
        uint32_t rl_words;
        if (num_columns <= MIN_COLUMNS) {
            metadata = &s_metadata[col];
            val_data = s_val_data[col];
            rl_data = s_rl_data[col];
            num_decode = s_num_decode[col];
            val_words = s_val_words[col];
            rl_words = s_rl_words[col];
        } else {
            const size_t ptr_index = static_cast<size_t>(col) * tiles_per_column + tile_idx;
            metadata = d_metadata_ptrs[ptr_index];
            val_data = d_val_data_ptrs[ptr_index];
            rl_data = d_rl_data_ptrs[ptr_index];
            const uint32_t count = metadata->val_count;
            num_decode = (count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
            val_words = (count * metadata->val_bitwidth + 31) >> 5;
            rl_words = (metadata->rl_count * metadata->rl_bitwidth + 31) >> 5;
        }

        // Decompress into thread-local registers
        int temp_elements[ITEMS_PER_THREAD];
        LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
            metadata, num_decode, val_words, rl_words,
            val_data, rl_data, decompression_scratch, temp_elements);

        size_t col_offset = col * ELEMS_PER_TILE;
        size_t first_thread_element_idx = col_offset + threadIdx.x;
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            column_elements[first_thread_element_idx + i * BLOCK_THREADS] = temp_elements[i];
        }

        __syncthreads();
    }

    // Form tuples from all decompressed columns
    size_t tile_base_offset = tile_idx * ELEMS_PER_TILE;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t global_tuple_idx = tile_base_offset + i * BLOCK_THREADS + threadIdx.x;

        if (global_tuple_idx < num_tuples) {
            size_t global_tuple_base_offset = global_tuple_idx * num_columns;

            for (int col = 0; col < num_columns; col++) {
                size_t intermediate_idx = col * ELEMS_PER_TILE + i * BLOCK_THREADS + threadIdx.x;
                d_tuples[global_tuple_base_offset + col] = column_elements[intermediate_idx];
            }
        }
    }
    __syncthreads();

    nsecs time_decompress = 0;
    if (threadIdx.x == 0) {
        time_decompress = globaltimer64();
    }

    // Hardcoded SSB Q1.1 aggregation.
    partial_sums[0] = 0;
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        size_t tuple_base_idx = i * BLOCK_THREADS + threadIdx.x;
        size_t global_tuple_idx = tile_base_offset + tuple_base_idx;

        if (global_tuple_idx < num_tuples) {
            size_t global_tuple_base_offset = global_tuple_idx * num_columns;

            const int lo_orderdate = d_tuples[global_tuple_base_offset + SSB_Q11_COL_ORDERDATE];
            const int lo_quantity = d_tuples[global_tuple_base_offset + SSB_Q11_COL_QUANTITY];
            const int lo_extendedprice = d_tuples[global_tuple_base_offset + SSB_Q11_COL_EXTENDEDPRICE];
            const int lo_discount = d_tuples[global_tuple_base_offset + SSB_Q11_COL_DISCOUNT];
            if (passesSSBQ11(lo_orderdate, lo_quantity, lo_discount)) {
                partial_sums[0] += revenueSSBQ11(lo_extendedprice, lo_discount);
            }
        }
    }
    __syncthreads();

    nsecs time_partial_sums = 0;
    if (threadIdx.x == 0) {
        time_partial_sums = globaltimer64();
    }

    // Add the tile revenue to global memory.
    const sum_type partial_sum = partial_sums[0];
    const sum_type tile_sum = BlockReduce(*temp_storage_reduce).Sum(partial_sum);

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<sum_type*>(&d_sums[0]), tile_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const nsecs time_end = globaltimer64();
        d_block_decompress_runtimes[blockIdx.x] = time_decompress - time_start;
        d_block_partial_sums_runtimes[blockIdx.x] = time_partial_sums - time_decompress;
        d_block_reduce_runtimes[blockIdx.x] = time_end - time_partial_sums;
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_tuple_registers(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    int* d_tuples,
    sum_type* d_sums,
    nsecs* d_block_decompress_runtimes,
    nsecs* d_block_partial_sums_runtimes,
    nsecs* d_block_reduce_runtimes,
    nsecs* d_registers_stage_totals,
    int filter,
    size_t tiles_per_column,
    size_t num_tuples,
    int num_columns,
    int num_sums) {
    (void)filter;

    nsecs time_start = 0;
    if (threadIdx.x == 0) {
        time_start = globaltimer64();
    }

    size_t tile_idx = blockIdx.x;
    if (tile_idx >= tiles_per_column) return;

    extern __shared__ unsigned char shared_buffer[];
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    constexpr size_t temp_storage_size = sizeof(typename BlockReduce::TempStorage);
    constexpr size_t padded_temp_storage_size =
        (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT * D_ARRAY_ALIGNMENT;
    auto temp_storage_reduce = reinterpret_cast<typename BlockReduce::TempStorage*>(shared_buffer);
    // Shared layout: [reduce temp][partial sums][cursor state].
    auto partial_sums_base = reinterpret_cast<sum_type*>(shared_buffer + padded_temp_storage_size);
    auto partial_sums = partial_sums_base + static_cast<size_t>(threadIdx.x) * num_sums;
    auto cursor_storage = reinterpret_cast<int*>(partial_sums_base + static_cast<size_t>(num_sums) * BLOCK_THREADS);
    const size_t cursor_stride = static_cast<size_t>(num_columns) * 2;
    int* cursor_base = cursor_storage + static_cast<size_t>(threadIdx.x) * cursor_stride;
    // Per-thread cursor arrays:
    //   run_idx[col] = current run for this thread/column
    //   run_end[col] = exclusive logical end of run_idx[col]
    int* run_idx = cursor_base;
    int* run_end = cursor_base + num_columns;

    (void)d_tuples;
    // Cache per-column descriptor data once per block for SSB Q1.1's fixed 4 columns.
    __shared__ RLETileMetadata s_metadata[MIN_COLUMNS];
    __shared__ const uint32_t* s_val_data[MIN_COLUMNS];
    __shared__ const uint32_t* s_rl_data[MIN_COLUMNS];

    if (num_columns <= MIN_COLUMNS && threadIdx.x < num_columns) {
        const size_t ptr_index = static_cast<size_t>(threadIdx.x) * tiles_per_column + tile_idx;
        RLETileMetadata* metadata = d_metadata_ptrs[ptr_index];
        s_metadata[threadIdx.x] = *metadata;
        s_val_data[threadIdx.x] = d_val_data_ptrs[ptr_index];
        s_rl_data[threadIdx.x] = d_rl_data_ptrs[ptr_index];
    }

    partial_sums[0] = 0;
    for (int col = 0; col < num_columns; col++) {
        run_idx[col] = 0;
        run_end[col] = 0;
    }
    __syncthreads();

    const size_t tile_base_offset = tile_idx * ELEMS_PER_TILE;
    if (tile_base_offset >= num_tuples) return;
    const RLETileMetadata* key_metadata;
    const uint32_t* key_val_data;
    const uint32_t* key_rl_data;
    const RLETileMetadata* quantity_metadata;
    const uint32_t* quantity_val_data;
    const uint32_t* quantity_rl_data;
    const RLETileMetadata* extendedprice_metadata;
    const uint32_t* extendedprice_val_data;
    const uint32_t* extendedprice_rl_data;
    const RLETileMetadata* discount_metadata;
    const uint32_t* discount_val_data;
    const uint32_t* discount_rl_data;
    if (num_columns <= MIN_COLUMNS) {
        key_metadata = &s_metadata[SSB_Q11_COL_ORDERDATE];
        key_val_data = s_val_data[SSB_Q11_COL_ORDERDATE];
        key_rl_data = s_rl_data[SSB_Q11_COL_ORDERDATE];
        quantity_metadata = &s_metadata[SSB_Q11_COL_QUANTITY];
        quantity_val_data = s_val_data[SSB_Q11_COL_QUANTITY];
        quantity_rl_data = s_rl_data[SSB_Q11_COL_QUANTITY];
        extendedprice_metadata = &s_metadata[SSB_Q11_COL_EXTENDEDPRICE];
        extendedprice_val_data = s_val_data[SSB_Q11_COL_EXTENDEDPRICE];
        extendedprice_rl_data = s_rl_data[SSB_Q11_COL_EXTENDEDPRICE];
        discount_metadata = &s_metadata[SSB_Q11_COL_DISCOUNT];
        discount_val_data = s_val_data[SSB_Q11_COL_DISCOUNT];
        discount_rl_data = s_rl_data[SSB_Q11_COL_DISCOUNT];
    } else {
        const size_t key_ptr_index = tile_idx;
        key_metadata = d_metadata_ptrs[key_ptr_index];
        key_val_data = d_val_data_ptrs[key_ptr_index];
        key_rl_data = d_rl_data_ptrs[key_ptr_index];
        const size_t quantity_ptr_index = static_cast<size_t>(SSB_Q11_COL_QUANTITY) * tiles_per_column + tile_idx;
        const size_t extendedprice_ptr_index = static_cast<size_t>(SSB_Q11_COL_EXTENDEDPRICE) * tiles_per_column + tile_idx;
        const size_t discount_ptr_index = static_cast<size_t>(SSB_Q11_COL_DISCOUNT) * tiles_per_column + tile_idx;
        quantity_metadata = d_metadata_ptrs[quantity_ptr_index];
        quantity_val_data = d_val_data_ptrs[quantity_ptr_index];
        quantity_rl_data = d_rl_data_ptrs[quantity_ptr_index];
        extendedprice_metadata = d_metadata_ptrs[extendedprice_ptr_index];
        extendedprice_val_data = d_val_data_ptrs[extendedprice_ptr_index];
        extendedprice_rl_data = d_rl_data_ptrs[extendedprice_ptr_index];
        discount_metadata = d_metadata_ptrs[discount_ptr_index];
        discount_val_data = d_val_data_ptrs[discount_ptr_index];
        discount_rl_data = d_rl_data_ptrs[discount_ptr_index];
    }
    nsecs key_decode_thread_ns = 0;
    nsecs value_decode_thread_ns = 0;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        // Thread-local logical positions are monotonic:
        // threadIdx.x, threadIdx.x + BLOCK_THREADS, ...
        // Cursor decoding relies on this monotonicity.
        const int local_idx = threadIdx.x + i * BLOCK_THREADS;
        const size_t global_tuple_idx = tile_base_offset + local_idx;

        if (global_tuple_idx < num_tuples) {
            const nsecs key_start = globaltimer64();
            const int orderdate = decodeRLEElementCursor(local_idx, key_metadata, key_val_data, key_rl_data,
                                                         &run_idx[SSB_Q11_COL_ORDERDATE], &run_end[SSB_Q11_COL_ORDERDATE]);
            key_decode_thread_ns += (globaltimer64() - key_start);

            if (orderdate > SSB_Q11_ORDERDATE_GT && orderdate < SSB_Q11_ORDERDATE_LT) {
                // Late materialization: decode remaining columns only for candidate tuples.
                const nsecs value_start = globaltimer64();
                const int quantity = decodeRLEElementCursor(local_idx, quantity_metadata, quantity_val_data,
                                                            quantity_rl_data, &run_idx[SSB_Q11_COL_QUANTITY],
                                                            &run_end[SSB_Q11_COL_QUANTITY]);
                const int discount = decodeRLEElementCursor(local_idx, discount_metadata, discount_val_data,
                                                            discount_rl_data, &run_idx[SSB_Q11_COL_DISCOUNT],
                                                            &run_end[SSB_Q11_COL_DISCOUNT]);
                if (quantity < SSB_Q11_QUANTITY_MAX_EXCLUSIVE &&
                    discount >= SSB_Q11_DISCOUNT_MIN &&
                    discount <= SSB_Q11_DISCOUNT_MAX) {
                    const int extendedprice = decodeRLEElementCursor(local_idx, extendedprice_metadata,
                                                                     extendedprice_val_data, extendedprice_rl_data,
                                                                     &run_idx[SSB_Q11_COL_EXTENDEDPRICE],
                                                                     &run_end[SSB_Q11_COL_EXTENDEDPRICE]);
                    partial_sums[0] += revenueSSBQ11(extendedprice, discount);
                }
                value_decode_thread_ns += (globaltimer64() - value_start);
            }
        }
    }
    __syncthreads();
    // Use the max thread time to model decode critical path per block.
    const nsecs key_decode_block_ns = BlockReduce(*temp_storage_reduce).Reduce(key_decode_thread_ns, cub::Max());
    __syncthreads();
    const nsecs value_decode_block_ns = BlockReduce(*temp_storage_reduce).Reduce(value_decode_thread_ns, cub::Max());
    __syncthreads();

    nsecs time_decompress = 0;
    nsecs time_partial_sums = 0;
    if (threadIdx.x == 0) {
        // Decoding and aggregation are fused in this kernel.
        time_decompress = globaltimer64();
        time_partial_sums = time_decompress;
    }

    // Add the tile revenue to global memory.
    const sum_type partial_sum = partial_sums[0];
    const sum_type tile_sum = BlockReduce(*temp_storage_reduce).Sum(partial_sum);

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<sum_type*>(&d_sums[0]), tile_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const nsecs time_end = globaltimer64();
        d_block_decompress_runtimes[blockIdx.x] = time_decompress - time_start;
        d_block_partial_sums_runtimes[blockIdx.x] = time_partial_sums - time_decompress;
        d_block_reduce_runtimes[blockIdx.x] = time_end - time_partial_sums;
        if (d_registers_stage_totals != nullptr) {
            // Aggregated diagnostics across all blocks in this launch.
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_registers_stage_totals[0]), key_decode_block_ns);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_registers_stage_totals[1]), value_decode_block_ns);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_registers_stage_totals[2]), time_end - time_partial_sums);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_registers_stage_totals[3]), time_decompress - time_start);
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_tuple_warp(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    int* d_tuples,
    sum_type* d_sums,
    nsecs* d_block_decompress_runtimes,
    nsecs* d_block_partial_sums_runtimes,
    nsecs* d_block_reduce_runtimes,
    int filter,
    size_t tiles_per_column,
    size_t num_tuples,
    int num_columns,
    int num_sums) {

    (void)d_tuples;
    (void)filter;
    (void)num_sums;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int warps_per_block = BLOCK_THREADS / 32;
    const size_t tile_idx = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_id;
    if (tile_idx >= tiles_per_column) return;

    extern __shared__ unsigned char shared_buffer[];
    auto partial_sums_base = reinterpret_cast<sum_type*>(shared_buffer);
    auto partial_sums = partial_sums_base + static_cast<size_t>(threadIdx.x) * SSB_Q11_NUM_SUMS;

    partial_sums[0] = 0;

    nsecs time_start = 0;
    if (lane == 0) {
        time_start = globaltimer64();
    }

    const size_t tile_base_offset = tile_idx * ELEMS_PER_TILE;
    if (tile_base_offset >= num_tuples) return;
    const size_t key_ptr_index = tile_idx;
    const RLETileMetadata* key_metadata = d_metadata_ptrs[key_ptr_index];
    const uint32_t* key_val_data = d_val_data_ptrs[key_ptr_index];
    const uint32_t* key_rl_data = d_rl_data_ptrs[key_ptr_index];
    const size_t quantity_ptr_index = static_cast<size_t>(SSB_Q11_COL_QUANTITY) * tiles_per_column + tile_idx;
    const size_t extendedprice_ptr_index = static_cast<size_t>(SSB_Q11_COL_EXTENDEDPRICE) * tiles_per_column + tile_idx;
    const size_t discount_ptr_index = static_cast<size_t>(SSB_Q11_COL_DISCOUNT) * tiles_per_column + tile_idx;
    const RLETileMetadata* quantity_metadata = d_metadata_ptrs[quantity_ptr_index];
    const uint32_t* quantity_val_data = d_val_data_ptrs[quantity_ptr_index];
    const uint32_t* quantity_rl_data = d_rl_data_ptrs[quantity_ptr_index];
    const RLETileMetadata* extendedprice_metadata = d_metadata_ptrs[extendedprice_ptr_index];
    const uint32_t* extendedprice_val_data = d_val_data_ptrs[extendedprice_ptr_index];
    const uint32_t* extendedprice_rl_data = d_rl_data_ptrs[extendedprice_ptr_index];
    const RLETileMetadata* discount_metadata = d_metadata_ptrs[discount_ptr_index];
    const uint32_t* discount_val_data = d_val_data_ptrs[discount_ptr_index];
    const uint32_t* discount_rl_data = d_rl_data_ptrs[discount_ptr_index];

    for (int batch = 0; batch < ELEMS_PER_TILE; batch += 32) {
        const int local_idx = batch + lane;
        const size_t global_tuple_idx = tile_base_offset + static_cast<size_t>(local_idx);
        const bool in_range = (global_tuple_idx < num_tuples);
        const unsigned active_mask = __ballot_sync(0xffffffff, in_range);
        if (active_mask == 0) continue;

        int orderdate = 0;
        if (in_range) {
            orderdate = decodeRLEElementCumulativeBinarySearch(
                local_idx,
                key_metadata,
                key_val_data,
                key_rl_data);
        }

        const bool year_pass = in_range &&
            (orderdate > SSB_Q11_ORDERDATE_GT && orderdate < SSB_Q11_ORDERDATE_LT);
        const unsigned year_pass_mask = __ballot_sync(active_mask, year_pass);
        if (year_pass_mask == 0) continue;

        int quantity = 0;
        int discount = 0;
        if (year_pass) {
            quantity = decodeRLEElementCumulativeBinarySearch(
                local_idx,
                quantity_metadata,
                quantity_val_data,
                quantity_rl_data);
            discount = decodeRLEElementCumulativeBinarySearch(
                local_idx,
                discount_metadata,
                discount_val_data,
                discount_rl_data);
        }

        const bool passes = year_pass &&
            quantity < SSB_Q11_QUANTITY_MAX_EXCLUSIVE &&
            discount >= SSB_Q11_DISCOUNT_MIN &&
            discount <= SSB_Q11_DISCOUNT_MAX;
        const unsigned pass_mask = __ballot_sync(active_mask, passes);
        if (pass_mask == 0) continue;

        if (passes) {
            const int extendedprice = decodeRLEElementCumulativeBinarySearch(
                local_idx,
                extendedprice_metadata,
                extendedprice_val_data,
                extendedprice_rl_data);
            partial_sums[0] += revenueSSBQ11(extendedprice, discount);
        }
    }

    __syncwarp();
    if (lane == 0) {
        const int warp_thread_base = warp_id * 32;
        sum_type sum_val = 0;
        for (int l = 0; l < 32; l++) {
            const int thread_idx = warp_thread_base + l;
            sum_val += partial_sums_base[static_cast<size_t>(thread_idx) * SSB_Q11_NUM_SUMS];
        }
        atomicAdd(reinterpret_cast<sum_type*>(&d_sums[0]), sum_val);
    }


    if (lane == 0) {
        const nsecs time_end = globaltimer64();
        d_block_decompress_runtimes[tile_idx] = time_end - time_start;
        d_block_partial_sums_runtimes[tile_idx] = 0;
        d_block_reduce_runtimes[tile_idx] = 0;

    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_tuple_predecode(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    int* d_tuples,
    sum_type* d_sums,
    nsecs* d_block_decompress_runtimes,
    nsecs* d_block_partial_sums_runtimes,
    nsecs* d_block_reduce_runtimes,
    int filter,
    size_t tiles_per_column,
    size_t num_tuples,
    int num_columns,
    int num_sums) {

    (void)d_tuples;
    (void)filter;
    (void)num_sums;

    nsecs time_start = 0;
    if (threadIdx.x == 0) {
        time_start = globaltimer64();
    }

    size_t tile_idx = blockIdx.x;
    if (tile_idx >= tiles_per_column) return;

    extern __shared__ unsigned char shared_buffer[];
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    constexpr size_t temp_storage_size = sizeof(typename BlockReduce::TempStorage);
    constexpr size_t padded_temp_storage_size =
        (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT * D_ARRAY_ALIGNMENT;
    auto temp_storage_reduce = reinterpret_cast<typename BlockReduce::TempStorage*>(shared_buffer);
    auto partial_sums_base = reinterpret_cast<sum_type*>(shared_buffer + padded_temp_storage_size);
    auto partial_sums = partial_sums_base + static_cast<size_t>(threadIdx.x) * SSB_Q11_NUM_SUMS;
    auto run_ends = reinterpret_cast<int*>(partial_sums_base + static_cast<size_t>(SSB_Q11_NUM_SUMS) * BLOCK_THREADS);
    auto run_values = run_ends + ELEMS_PER_TILE;
    auto pass_mask = reinterpret_cast<unsigned char*>(run_values + ELEMS_PER_TILE);
    auto discount_values = reinterpret_cast<int*>(pass_mask + ELEMS_PER_TILE);
    // Cache per-column descriptor data once per block for SSB Q1.1's fixed 4 columns.
    __shared__ RLETileMetadata s_metadata[MIN_COLUMNS];
    __shared__ const uint32_t* s_val_data[MIN_COLUMNS];
    __shared__ const uint32_t* s_rl_data[MIN_COLUMNS];

    if (num_columns <= MIN_COLUMNS && threadIdx.x < num_columns) {
        const size_t ptr_index = static_cast<size_t>(threadIdx.x) * tiles_per_column + tile_idx;
        RLETileMetadata* metadata = d_metadata_ptrs[ptr_index];
        s_metadata[threadIdx.x] = *metadata;
        s_val_data[threadIdx.x] = d_val_data_ptrs[ptr_index];
        s_rl_data[threadIdx.x] = d_rl_data_ptrs[ptr_index];
    }

    partial_sums[0] = 0;
    __syncthreads();

    const size_t tile_base_offset = tile_idx * ELEMS_PER_TILE;
    const RLETileMetadata* orderdate_metadata;
    const uint32_t* orderdate_val_data;
    const uint32_t* orderdate_rl_data;
    if (num_columns <= MIN_COLUMNS) {
        orderdate_metadata = &s_metadata[SSB_Q11_COL_ORDERDATE];
        orderdate_val_data = s_val_data[SSB_Q11_COL_ORDERDATE];
        orderdate_rl_data = s_rl_data[SSB_Q11_COL_ORDERDATE];
    } else {
        const size_t orderdate_ptr_index = tile_idx;
        orderdate_metadata = d_metadata_ptrs[orderdate_ptr_index];
        orderdate_val_data = d_val_data_ptrs[orderdate_ptr_index];
        orderdate_rl_data = d_rl_data_ptrs[orderdate_ptr_index];
    }
    const int orderdate_run_count = static_cast<int>(orderdate_metadata->val_count);

    for (int idx = threadIdx.x; idx < orderdate_run_count; idx += BLOCK_THREADS) {
        run_values[idx] = decodeElementRBin(idx, orderdate_val_data,
                                            orderdate_metadata->val_reference, orderdate_metadata->val_bitwidth);
        run_ends[idx] = decodeElementRBin(idx, orderdate_rl_data,
                                          orderdate_metadata->rl_reference, orderdate_metadata->rl_bitwidth);
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int local_idx = threadIdx.x + i * BLOCK_THREADS;
        const size_t global_tuple_idx = tile_base_offset + local_idx;
        unsigned char pass = 0;

        if (global_tuple_idx < num_tuples && orderdate_run_count > 0) {
            const int run_idx = findRunBinarySearch(local_idx, run_ends, orderdate_run_count);
            const int orderdate = run_values[run_idx];
            pass = (orderdate > SSB_Q11_ORDERDATE_GT && orderdate < SSB_Q11_ORDERDATE_LT);
        }
        pass_mask[local_idx] = pass;
    }
    __syncthreads();

    const RLETileMetadata* quantity_metadata;
    const uint32_t* quantity_val_data;
    const uint32_t* quantity_rl_data;
    if (num_columns <= MIN_COLUMNS) {
        quantity_metadata = &s_metadata[SSB_Q11_COL_QUANTITY];
        quantity_val_data = s_val_data[SSB_Q11_COL_QUANTITY];
        quantity_rl_data = s_rl_data[SSB_Q11_COL_QUANTITY];
    } else {
        const size_t quantity_ptr_index = static_cast<size_t>(SSB_Q11_COL_QUANTITY) * tiles_per_column + tile_idx;
        quantity_metadata = d_metadata_ptrs[quantity_ptr_index];
        quantity_val_data = d_val_data_ptrs[quantity_ptr_index];
        quantity_rl_data = d_rl_data_ptrs[quantity_ptr_index];
    }
    const int quantity_run_count = static_cast<int>(quantity_metadata->val_count);

    for (int idx = threadIdx.x; idx < quantity_run_count; idx += BLOCK_THREADS) {
        run_values[idx] = decodeElementRBin(idx, quantity_val_data,
                                            quantity_metadata->val_reference, quantity_metadata->val_bitwidth);
        run_ends[idx] = decodeElementRBin(idx, quantity_rl_data,
                                          quantity_metadata->rl_reference, quantity_metadata->rl_bitwidth);
    }
    __syncthreads();

    if (quantity_run_count > 0) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int local_idx = threadIdx.x + i * BLOCK_THREADS;
            const size_t global_tuple_idx = tile_base_offset + local_idx;
            if (global_tuple_idx < num_tuples && pass_mask[local_idx]) {
                const int run_idx = findRunBinarySearch(local_idx, run_ends, quantity_run_count);
                const int quantity = run_values[run_idx];
                pass_mask[local_idx] = (quantity < SSB_Q11_QUANTITY_MAX_EXCLUSIVE);
            }
        }
    }
    __syncthreads();

    const RLETileMetadata* discount_metadata;
    const uint32_t* discount_val_data;
    const uint32_t* discount_rl_data;
    if (num_columns <= MIN_COLUMNS) {
        discount_metadata = &s_metadata[SSB_Q11_COL_DISCOUNT];
        discount_val_data = s_val_data[SSB_Q11_COL_DISCOUNT];
        discount_rl_data = s_rl_data[SSB_Q11_COL_DISCOUNT];
    } else {
        const size_t discount_ptr_index = static_cast<size_t>(SSB_Q11_COL_DISCOUNT) * tiles_per_column + tile_idx;
        discount_metadata = d_metadata_ptrs[discount_ptr_index];
        discount_val_data = d_val_data_ptrs[discount_ptr_index];
        discount_rl_data = d_rl_data_ptrs[discount_ptr_index];
    }
    const int discount_run_count = static_cast<int>(discount_metadata->val_count);

    for (int idx = threadIdx.x; idx < discount_run_count; idx += BLOCK_THREADS) {
        run_values[idx] = decodeElementRBin(idx, discount_val_data,
                                            discount_metadata->val_reference, discount_metadata->val_bitwidth);
        run_ends[idx] = decodeElementRBin(idx, discount_rl_data,
                                          discount_metadata->rl_reference, discount_metadata->rl_bitwidth);
    }
    __syncthreads();

    if (discount_run_count > 0) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int local_idx = threadIdx.x + i * BLOCK_THREADS;
            const size_t global_tuple_idx = tile_base_offset + local_idx;
            if (global_tuple_idx < num_tuples && pass_mask[local_idx]) {
                const int run_idx = findRunBinarySearch(local_idx, run_ends, discount_run_count);
                const int discount = run_values[run_idx];
                const unsigned char discount_pass = discount >= SSB_Q11_DISCOUNT_MIN &&
                                                    discount <= SSB_Q11_DISCOUNT_MAX;
                pass_mask[local_idx] = discount_pass;
                if (discount_pass) {
                    discount_values[local_idx] = discount;
                }
            }
        }
    }
    __syncthreads();

    const RLETileMetadata* extendedprice_metadata;
    const uint32_t* extendedprice_val_data;
    const uint32_t* extendedprice_rl_data;
    if (num_columns <= MIN_COLUMNS) {
        extendedprice_metadata = &s_metadata[SSB_Q11_COL_EXTENDEDPRICE];
        extendedprice_val_data = s_val_data[SSB_Q11_COL_EXTENDEDPRICE];
        extendedprice_rl_data = s_rl_data[SSB_Q11_COL_EXTENDEDPRICE];
    } else {
        const size_t extendedprice_ptr_index = static_cast<size_t>(SSB_Q11_COL_EXTENDEDPRICE) * tiles_per_column + tile_idx;
        extendedprice_metadata = d_metadata_ptrs[extendedprice_ptr_index];
        extendedprice_val_data = d_val_data_ptrs[extendedprice_ptr_index];
        extendedprice_rl_data = d_rl_data_ptrs[extendedprice_ptr_index];
    }
    const int extendedprice_run_count = static_cast<int>(extendedprice_metadata->val_count);

    for (int idx = threadIdx.x; idx < extendedprice_run_count; idx += BLOCK_THREADS) {
        run_values[idx] = decodeElementRBin(idx, extendedprice_val_data,
                                            extendedprice_metadata->val_reference, extendedprice_metadata->val_bitwidth);
        run_ends[idx] = decodeElementRBin(idx, extendedprice_rl_data,
                                          extendedprice_metadata->rl_reference, extendedprice_metadata->rl_bitwidth);
    }
    __syncthreads();

    if (extendedprice_run_count > 0) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const int local_idx = threadIdx.x + i * BLOCK_THREADS;
            const size_t global_tuple_idx = tile_base_offset + local_idx;
            if (global_tuple_idx < num_tuples && pass_mask[local_idx]) {
                const int run_idx = findRunBinarySearch(local_idx, run_ends, extendedprice_run_count);
                const int extendedprice = run_values[run_idx];
                partial_sums[0] += revenueSSBQ11(extendedprice, discount_values[local_idx]);
            }
        }
    }
    __syncthreads();

    nsecs time_decompress = 0;
    nsecs time_partial_sums = 0;
    if (threadIdx.x == 0) {
        // Decoding and aggregation are fused in this kernel.
        time_decompress = globaltimer64();
        time_partial_sums = time_decompress;
    }

    const sum_type partial_sum = partial_sums[0];
    const sum_type tile_sum = BlockReduce(*temp_storage_reduce).Sum(partial_sum);

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<sum_type*>(&d_sums[0]), tile_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const nsecs time_end = globaltimer64();
        d_block_decompress_runtimes[blockIdx.x] = time_decompress - time_start;
        d_block_partial_sums_runtimes[blockIdx.x] = time_partial_sums - time_decompress;
        d_block_reduce_runtimes[blockIdx.x] = time_end - time_partial_sums;
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compressed_block_to_shared_tuples(
    RLETileMetadata** d_metadata_ptrs,
    uint32_t** d_val_data_ptrs,
    uint32_t** d_rl_data_ptrs,
    sum_type* d_sums,
    nsecs* d_block_decompress_runtimes,
    nsecs* d_block_partial_sums_runtimes,
    nsecs* d_block_reduce_runtimes,
    int filter,
    size_t tiles_per_column,
    size_t total_tuples,
    int num_columns,
    int num_sums) {
    (void)filter;
    (void)num_sums;

    // Each block processes D consecutive tiles.
    const size_t tile_group_base =
        static_cast<size_t>(blockIdx.x) * RFOR_SHARED_TILES_PER_BLOCK;
    if (tile_group_base >= tiles_per_column) return;

    extern __shared__ unsigned char shared_buffer[];
    const size_t tuple_size = sizeof(int) * num_columns;
    const size_t tuples_size = tuple_size * ELEMS_PER_TILE;
    const size_t partial_sum_size = sizeof(sum_type) * num_sums;
    constexpr size_t load_rbin_scratch_size = loadRBinPackScratchBytes<BLOCK_THREADS, ITEMS_PER_THREAD>();
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    auto column_elements = reinterpret_cast<int*>(shared_buffer);
    auto partial_sums = reinterpret_cast<sum_type*>(
        shared_buffer + tuples_size + load_rbin_scratch_size + partial_sum_size * threadIdx.x);
    auto decompression_scratch = reinterpret_cast<uint*>(shared_buffer + tuples_size);
    auto temp_storage_reduce = reinterpret_cast<typename BlockReduce::TempStorage*>(shared_buffer);

    // SSB Q1.1 uses exactly 4 columns; cache per-column descriptor data for D tiles once per block.
    __shared__ RLETileMetadata s_metadata[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];
    __shared__ uint32_t* s_val_data[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];
    __shared__ uint32_t* s_rl_data[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];
    __shared__ uint32_t s_num_decode[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];
    __shared__ uint32_t s_val_words[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];
    __shared__ uint32_t s_rl_words[MIN_COLUMNS][RFOR_SHARED_TILES_PER_BLOCK];

    if (num_columns <= MIN_COLUMNS) {
        const uint32_t slots = static_cast<uint32_t>(num_columns * RFOR_SHARED_TILES_PER_BLOCK);
        if (threadIdx.x < slots) {
            const uint32_t flat = static_cast<uint32_t>(threadIdx.x);
            const uint32_t tile_slot = flat / static_cast<uint32_t>(num_columns);
            const uint32_t col = flat % static_cast<uint32_t>(num_columns);
            const size_t tile_idx = tile_group_base + static_cast<size_t>(tile_slot);
            if (tile_idx < tiles_per_column) {
                const size_t ptr_index = static_cast<size_t>(col) * tiles_per_column + tile_idx;
                RLETileMetadata* metadata = d_metadata_ptrs[ptr_index];
                s_metadata[col][tile_slot] = *metadata;
                s_val_data[col][tile_slot] = d_val_data_ptrs[ptr_index];
                s_rl_data[col][tile_slot] = d_rl_data_ptrs[ptr_index];
                const uint32_t count = metadata->val_count;
                s_num_decode[col][tile_slot] =
                    (count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
                s_val_words[col][tile_slot] =
                    (count * metadata->val_bitwidth + 31) >> 5;
                s_rl_words[col][tile_slot] =
                    (metadata->rl_count * metadata->rl_bitwidth + 31) >> 5;
            } else {
                s_num_decode[col][tile_slot] = 0;
                s_val_words[col][tile_slot] = 0;
                s_rl_words[col][tile_slot] = 0;
                s_val_data[col][tile_slot] = nullptr;
                s_rl_data[col][tile_slot] = nullptr;
            }
        }
    }
    __syncthreads();

    for (size_t tile_slot = 0; tile_slot < RFOR_SHARED_TILES_PER_BLOCK; ++tile_slot) {
        const size_t tile_idx = tile_group_base + tile_slot;
        if (tile_idx >= tiles_per_column) break;

        const nsecs tile_start = (threadIdx.x == 0) ? globaltimer64() : 0;

        for (int col = 0; col < num_columns; col++) {
            RLETileMetadata* metadata;
            uint32_t* val_data;
            uint32_t* rl_data;
            uint32_t num_decode;
            uint32_t val_words;
            uint32_t rl_words;
            if (num_columns <= MIN_COLUMNS) {
                metadata = &s_metadata[col][tile_slot];
                val_data = s_val_data[col][tile_slot];
                rl_data = s_rl_data[col][tile_slot];
                num_decode = s_num_decode[col][tile_slot];
                val_words = s_val_words[col][tile_slot];
                rl_words = s_rl_words[col][tile_slot];
            } else {
                const size_t ptr_index = static_cast<size_t>(col) * tiles_per_column + tile_idx;
                metadata = d_metadata_ptrs[ptr_index];
                val_data = d_val_data_ptrs[ptr_index];
                rl_data = d_rl_data_ptrs[ptr_index];
                const uint32_t count = metadata->val_count;
                num_decode = (count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
                val_words = (count * metadata->val_bitwidth + 31) >> 5;
                rl_words = (metadata->rl_count * metadata->rl_bitwidth + 31) >> 5;
            }

            int temp_elements[ITEMS_PER_THREAD];
            LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
                metadata, num_decode, val_words, rl_words,
                val_data, rl_data, decompression_scratch, temp_elements);

            const size_t col_offset = col * ELEMS_PER_TILE;
            const size_t first_thread_element_idx = col_offset + threadIdx.x;
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                const size_t thread_element_idx =
                    first_thread_element_idx + i * BLOCK_THREADS;
                column_elements[thread_element_idx] = temp_elements[i];
            }
            __syncthreads();
        }

        const nsecs time_decompress =
            (threadIdx.x == 0) ? globaltimer64() : 0;

        partial_sums[0] = 0;
        __syncthreads();

        const size_t tile_base_offset = tile_idx * ELEMS_PER_TILE;

        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            const size_t tuple_base_idx = i * BLOCK_THREADS + threadIdx.x;
            const size_t global_tuple_idx = tile_base_offset + tuple_base_idx;

            if (global_tuple_idx < total_tuples) {
                const int lo_orderdate = column_elements[tuple_base_idx + SSB_Q11_COL_ORDERDATE * ELEMS_PER_TILE];
                const int lo_quantity = column_elements[tuple_base_idx + SSB_Q11_COL_QUANTITY * ELEMS_PER_TILE];
                const int lo_extendedprice = column_elements[tuple_base_idx + SSB_Q11_COL_EXTENDEDPRICE * ELEMS_PER_TILE];
                const int lo_discount = column_elements[tuple_base_idx + SSB_Q11_COL_DISCOUNT * ELEMS_PER_TILE];
                if (passesSSBQ11(lo_orderdate, lo_quantity, lo_discount)) {
                    partial_sums[0] += revenueSSBQ11(lo_extendedprice, lo_discount);
                }
            }
        }
        __syncthreads();

        const nsecs time_partial_sums =
            (threadIdx.x == 0) ? globaltimer64() : 0;

        const sum_type partial_sum = partial_sums[0];
        const sum_type tile_sum = BlockReduce(*temp_storage_reduce).Sum(partial_sum);

        if (threadIdx.x == 0) {
            atomicAdd(reinterpret_cast<sum_type*>(&d_sums[0]), tile_sum);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const nsecs time_end = globaltimer64();
            d_block_decompress_runtimes[tile_idx] = time_decompress - tile_start;
            d_block_partial_sums_runtimes[tile_idx] = time_partial_sums - time_decompress;
            d_block_reduce_runtimes[tile_idx] = time_end - time_partial_sums;
        }
        __syncthreads();
    }
}

uint compressRFORTile(int* tile_data, size_t elems_per_tile, std::vector<char>& compressed_tile, bool cumulative_rl) {
    if (elems_per_tile == 0) return 0;

    std::vector<int> unique_values;
    std::vector<int> run_lengths;

    int current_value = tile_data[0];
    int current_run_length = 1;

    for (size_t i = 1; i < elems_per_tile; i++) {
        if (tile_data[i] == current_value) {
            current_run_length++;
        } else {
            unique_values.push_back(current_value);
            run_lengths.push_back(current_run_length);
            current_value = tile_data[i];
            current_run_length = 1;
        }
    }

    unique_values.push_back(current_value);
    run_lengths.push_back(current_run_length);

    if (cumulative_rl) {
        int cumulative = 0;
        for (auto& run_length : run_lengths) {
            cumulative += run_length;
            run_length = cumulative;
        }
    }

    // print the unique values and run lengths for debugging
    // std::cout << "Unique Values: ";
    // for (const auto& val : unique_values) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\nRun Lengths: ";
    // for (const auto& rl : run_lengths)
    // {
    //     std::cout << rl << " ";
    //
    // }
    //exit(1);

    uint count = unique_values.size();

    int val_reference = *std::min_element(unique_values.begin(), unique_values.end());
    uint val_bitwidth = 1;
    for (size_t i = 0; i < unique_values.size(); i++) {
        uint for_adjusted = unique_values[i] - val_reference;
        if (for_adjusted > 0) {
            uint required_bits = static_cast<uint>(ceil(log2(for_adjusted + 1)));
            if (required_bits > val_bitwidth) {
                val_bitwidth = required_bits;
            }
        }
    }
    if (val_bitwidth == 0) val_bitwidth = 1;
    if (val_bitwidth > 32) val_bitwidth = 32;

    int rl_reference = *std::min_element(run_lengths.begin(), run_lengths.end());
    uint rl_bitwidth = 1;
    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint for_adjusted = run_lengths[i] - rl_reference;
        if (for_adjusted > 0) {
            uint required_bits = static_cast<uint>(ceil(log2(for_adjusted + 1)));
            if (required_bits > rl_bitwidth) {
                rl_bitwidth = required_bits;
            }
        }
    }
    if (rl_bitwidth == 0) rl_bitwidth = 1;
    if (rl_bitwidth > 32) rl_bitwidth = 32;

    RLETileMetadata metadata;
    metadata.val_reference = static_cast<uint32_t>(val_reference);
    metadata.val_bitwidth = val_bitwidth;
    metadata.val_count = count;
    metadata.rl_reference = static_cast<uint32_t>(rl_reference);
    metadata.rl_bitwidth = rl_bitwidth;
    metadata.rl_count = count;

    std::vector<uint32_t> val_data;
    uint current_word = 0;
    uint bits_used = 0;

    for (size_t i = 0; i < unique_values.size(); i++) {
        uint for_adjusted = unique_values[i] - val_reference;

        if (bits_used + val_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += val_bitwidth;

            if (bits_used == 32 || i == unique_values.size() - 1) {
                val_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint bits_in_current = 32 - bits_used;
            uint bits_in_next = val_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            val_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == unique_values.size() - 1 && bits_used > 0) {
                val_data.push_back(current_word);
            }
        }
    }

    std::vector<uint32_t> rl_data;
    current_word = 0;
    bits_used = 0;

    for (size_t i = 0; i < run_lengths.size(); i++) {
        uint for_adjusted = run_lengths[i] - rl_reference;

        if (bits_used + rl_bitwidth <= 32) {
            current_word |= (for_adjusted << bits_used);
            bits_used += rl_bitwidth;

            if (bits_used == 32 || i == run_lengths.size() - 1) {
                rl_data.push_back(current_word);
                current_word = 0;
                bits_used = 0;
            }
        } else {
            uint bits_in_current = 32 - bits_used;
            uint bits_in_next = rl_bitwidth - bits_in_current;

            current_word |= (for_adjusted << bits_used);
            rl_data.push_back(current_word);

            current_word = for_adjusted >> bits_in_current;
            bits_used = bits_in_next;

            if (i == run_lengths.size() - 1 && bits_used > 0) {
                rl_data.push_back(current_word);
            }
        }
    }

    size_t metadata_size = sizeof(RLETileMetadata);
    size_t val_data_size = val_data.size() * sizeof(uint32_t);
    size_t rl_data_size = rl_data.size() * sizeof(uint32_t);
    size_t total_size = metadata_size + val_data_size + rl_data_size;

    compressed_tile.resize(total_size);
    char* ptr = compressed_tile.data();

    std::memcpy(ptr, &metadata, metadata_size);
    ptr += metadata_size;
    std::memcpy(ptr, val_data.data(), val_data_size);
    ptr += val_data_size;
    std::memcpy(ptr, rl_data.data(), rl_data_size);

    return count;
}

std::vector<RLEColumn> createRLEColumns(const std::vector<std::vector<int>>& column_data,
                                        size_t total_elements, int num_columns,
                                        size_t max_uncompressed_bytes_per_block,
                                        bool cumulative_rl) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<RLEColumn> columns(num_columns);
    const size_t total_tiles = (total_elements + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE;
    const size_t max_tiles_per_block = max_uncompressed_bytes_per_block / (ELEMS_PER_TILE * sizeof(int));

    for (int col = 0; col < num_columns; col++) {
        size_t current_tile = 0;

        while (current_tile < total_tiles) {
            size_t tiles_in_block = std::min(max_tiles_per_block, total_tiles - current_tile);

            for (size_t tile_idx = 0; tile_idx < tiles_in_block; tile_idx++) {
                size_t global_tile_idx = current_tile + tile_idx;
                size_t tile_start = global_tile_idx * ELEMS_PER_TILE;

                std::vector<int> tile_data(ELEMS_PER_TILE);
                for (size_t i = 0; i < ELEMS_PER_TILE; i++) {
                    if (tile_start + i < total_elements) {
                        tile_data[i] = column_data[col][tile_start + i];
                    } else {
                        tile_data[i] = column_data[col][total_elements - 1];
                    }
                }

                std::vector<char> compressed_tile;
                compressRFORTile(tile_data.data(), ELEMS_PER_TILE, compressed_tile, cumulative_rl);
                columns[col].compressed_chunks.push_back(std::move(compressed_tile));
            }

            current_tile += tiles_in_block;
        }

        columns[col].num_values = total_elements;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(end - start).count();

    return columns;
}

float run_compressed_query(
    const vector<vector<int>>& h_cols_data,
    vector<sum_type>& h_sums,
    const size_t total_data_size,
    const size_t per_col_size,
    const int filter,
    const int num_cols,
    const int num_sums,
    const int num_tuples,
    const int num_SMs,
    const bool use_proteus_profiling_control,
    const ExecutionMode exec_mode) {
    (void)filter;

    constexpr size_t max_block_size = 2 * 1024 * 1024;

    const bool use_cumulative_rl = (exec_mode == RFOR_PREDECODE || exec_mode == RFOR_WARP);
    const auto columns = createRLEColumns(h_cols_data, num_tuples, num_cols, max_block_size, use_cumulative_rl);

    const size_t num_tiles = columns[0].compressed_chunks.size();
    const size_t num_chunks = num_cols * num_tiles;

    // Prepare separate host buffers per column
    vector<vector<char>> h_col_buffers(num_cols);
    vector<vector<size_t>> chunk_offsets_per_col(num_cols);
    vector<size_t> col_total_sizes(num_cols);
    vector<size_t> col_metadata_bytes(num_cols, 0);
    vector<size_t> col_value_bytes(num_cols, 0);
    vector<size_t> col_run_bytes(num_cols, 0);

    size_t total_compressed_bytes = 0;
    size_t total_metadata_bytes = 0;
    size_t total_value_bytes = 0;
    size_t total_run_bytes = 0;
    double compression_ratio_sum = 0.0;
    vector<double> compression_ratios(num_cols);

    for (int col = 0; col < num_cols; ++col) {
        // Calculate total size for this column
        size_t col_total_size = 0;
        for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            col_total_size += columns[col].compressed_chunks[tile_idx].size();
        }
        col_total_sizes[col] = col_total_size;

        // Compute the column's compression ratio
        const auto compression_ratio = static_cast<double>(per_col_size) / static_cast<double>(col_total_size);
        compression_ratio_sum += compression_ratio;
        compression_ratios[col] = compression_ratio;

        // Create contiguous host buffer for this column
        h_col_buffers[col].resize(col_total_size);
        size_t current_offset = 0;
        for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const auto& chunk = columns[col].compressed_chunks[tile_idx];
            total_compressed_bytes += chunk.size();

            RLETileMetadata metadata{};
            memcpy(&metadata, chunk.data(), sizeof(RLETileMetadata));
            const size_t metadata_size = sizeof(RLETileMetadata);
            const size_t value_words = (static_cast<size_t>(metadata.val_count) * metadata.val_bitwidth + 31) / 32;
            const size_t run_words = (static_cast<size_t>(metadata.rl_count) * metadata.rl_bitwidth + 31) / 32;
            const size_t value_size = value_words * sizeof(uint32_t);
            const size_t run_size = run_words * sizeof(uint32_t);

            col_metadata_bytes[col] += metadata_size;
            col_value_bytes[col] += value_size;
            col_run_bytes[col] += run_size;
            total_metadata_bytes += metadata_size;
            total_value_bytes += value_size;
            total_run_bytes += run_size;

            memcpy(h_col_buffers[col].data() + current_offset, chunk.data(), chunk.size());
            chunk_offsets_per_col[col].push_back(current_offset);
            current_offset += chunk.size();
        }
    }

    // Compute the mean and std dev per col compression ratios
    const double mean_per_col_compression_ratio = compression_ratio_sum / num_cols;
    for_each(compression_ratios.begin(), compression_ratios.end(), [mean_per_col_compression_ratio](const auto compression_ratio) {
        const auto diff = static_cast<double>(compression_ratio) - mean_per_col_compression_ratio;
        return diff * diff;
    });
    const auto std_dev_per_col_compression_ratio = sqrt(
        accumulate(compression_ratios.begin(), compression_ratios.end(), 0.0) / (
            static_cast<double>(compression_ratios.size()) - 1));

    // Compute the entire dataset compression ratio
    const auto total_compression_ratio = static_cast<double>(total_data_size) / static_cast<double>(
                                             total_compressed_bytes);

    for (int col = 0; col < num_cols; ++col) {
        csvOut("compressed_query", "all", "column_" + toCsvString(col) + "_metadata_bytes", col_metadata_bytes[col]);
        csvOut("compressed_query", "all", "column_" + toCsvString(col) + "_value_bytes", col_value_bytes[col]);
        csvOut("compressed_query", "all", "column_" + toCsvString(col) + "_run_bytes", col_run_bytes[col]);
        const size_t col_payload_bytes = col_value_bytes[col] + col_run_bytes[col];
        csvOut("compressed_query", "all", "column_" + toCsvString(col) + "_payload_bytes", col_payload_bytes);
        if (col_payload_bytes > 0) {
            const double run_share = static_cast<double>(col_run_bytes[col]) / static_cast<double>(col_payload_bytes);
            csvOut("compressed_query", "all", "column_" + toCsvString(col) + "_run_share_of_payload", run_share);
        }
    }
    csvOut("compressed_query", "summary", "total_metadata_bytes", total_metadata_bytes);
    csvOut("compressed_query", "summary", "total_value_bytes", total_value_bytes);
    csvOut("compressed_query", "summary", "total_run_bytes", total_run_bytes);
    csvOut("compressed_query", "summary", "total_compressed_bytes", total_compressed_bytes);
    const size_t total_payload_bytes = total_value_bytes + total_run_bytes;
    csvOut("compressed_query", "summary", "total_payload_bytes", total_payload_bytes);
    if (total_payload_bytes > 0) {
        const double run_share = static_cast<double>(total_run_bytes) / static_cast<double>(total_payload_bytes);
        csvOut("compressed_query", "summary", "total_run_share_of_payload", run_share);
    }

    // Compute sizes for memory requirements
    const size_t d_tuples_size = static_cast<size_t>(num_tuples) * num_cols * sizeof(int);
    const size_t tuples_size = sizeof(int) * num_cols * ELEMS_PER_TILE;
    const size_t sums_size = sizeof(sum_type) * num_sums;
    const size_t s_sums_size = sums_size * THREADS_PER_TILE;
    const size_t block_decompress_runtimes_size = sizeof(nsecs) * num_tiles;
    const size_t block_partial_sums_runtimes_size = sizeof(nsecs) * num_tiles;
    const size_t block_reduce_runtimes_size = sizeof(nsecs) * num_tiles;
    constexpr size_t load_rbin_scratch_size = loadRBinPackScratchBytes<THREADS_PER_TILE, ELEMS_PER_THREAD>();
    constexpr int warps_per_block = THREADS_PER_TILE / 32;
    size_t shared_mem_size = 0;
    if (exec_mode == RFOR_REGISTERS) {
        constexpr size_t temp_storage_size = sizeof(BlockReduce<sum_type, THREADS_PER_TILE>::TempStorage);
        constexpr size_t padded_temp_storage_size =
            (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT * D_ARRAY_ALIGNMENT;
        const size_t cursor_storage_size = sizeof(int) * 2 * static_cast<size_t>(num_cols) * THREADS_PER_TILE;
        shared_mem_size = padded_temp_storage_size + s_sums_size + cursor_storage_size;
    } else if (exec_mode == RFOR_PREDECODE) {
        constexpr size_t temp_storage_size = sizeof(BlockReduce<sum_type, THREADS_PER_TILE>::TempStorage);
        constexpr size_t padded_temp_storage_size =
            (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT * D_ARRAY_ALIGNMENT;
        const size_t run_ends_size = sizeof(int) * ELEMS_PER_TILE;
        const size_t run_values_size = sizeof(int) * ELEMS_PER_TILE;
        const size_t pass_mask_size = sizeof(unsigned char) * ELEMS_PER_TILE;
        const size_t discount_values_size = sizeof(int) * ELEMS_PER_TILE;
        shared_mem_size = padded_temp_storage_size + s_sums_size + run_ends_size + run_values_size + pass_mask_size +
                          discount_values_size;
    } else if (exec_mode == RFOR_WARP) {
        shared_mem_size = s_sums_size;
    } else {
        shared_mem_size = tuples_size + load_rbin_scratch_size + s_sums_size;
    }
    const size_t metadata_ptrs_size = num_chunks * sizeof(RLETileMetadata*);
    const size_t val_data_ptrs_size = num_chunks * sizeof(uint32_t*);
    const size_t rl_data_ptrs_size = num_chunks * sizeof(uint32_t*);
    const size_t pointer_tables_bytes = metadata_ptrs_size + val_data_ptrs_size + rl_data_ptrs_size;
    const size_t expected_kernel_reads_payload_plus_metadata_bytes = total_compressed_bytes;
    const size_t expected_kernel_reads_plus_ptr_tables_bytes =
        expected_kernel_reads_payload_plus_metadata_bytes + pointer_tables_bytes;
    csvOut("compressed_query", "summary", "pointer_tables_bytes", pointer_tables_bytes);
    csvOut("compressed_query", "summary", "expected_kernel_reads_payload_plus_metadata_bytes",
           expected_kernel_reads_payload_plus_metadata_bytes);
    csvOut("compressed_query", "summary", "expected_kernel_reads_plus_ptr_tables_bytes",
           expected_kernel_reads_plus_ptr_tables_bytes);
    csvOut("compressed_query", "summary", "expected_kernel_reads_payload_plus_metadata_gib",
           static_cast<double>(expected_kernel_reads_payload_plus_metadata_bytes) / BYTES_PER_GIB);
    csvOut("compressed_query", "summary", "expected_kernel_reads_plus_ptr_tables_gib",
           static_cast<double>(expected_kernel_reads_plus_ptr_tables_bytes) / BYTES_PER_GIB);
    csvOut("compressed_query", "summary", "expected_kernel_reads_payload_plus_metadata_gb",
           static_cast<double>(expected_kernel_reads_payload_plus_metadata_bytes) / BYTES_PER_GB);
    csvOut("compressed_query", "summary", "expected_kernel_reads_plus_ptr_tables_gb",
           static_cast<double>(expected_kernel_reads_plus_ptr_tables_bytes) / BYTES_PER_GB);

    // Query the maximum number of blocks the GPU can run given the threads per block and shared memory requirements
    int max_blocks_per_SM;
    if (exec_mode == RFOR_GLOBAL) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_tuple<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else if (exec_mode == RFOR_SHARED) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_shared_tuples<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else if (exec_mode == RFOR_REGISTERS) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_tuple_registers<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else if (exec_mode == RFOR_PREDECODE) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_tuple_predecode<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else if (exec_mode == RFOR_WARP) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_tuple_warp<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      compressed_block_to_tuple<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    }
    int max_blocks = max_blocks_per_SM * num_SMs;
    size_t launched_blocks = num_tiles;
    if (exec_mode == RFOR_SHARED) {
        launched_blocks =
            (num_tiles + RFOR_SHARED_TILES_PER_BLOCK - 1) / RFOR_SHARED_TILES_PER_BLOCK;
    }
    if (exec_mode == RFOR_WARP) {
        max_blocks *= warps_per_block;
        launched_blocks = (num_tiles + warps_per_block - 1) / warps_per_block;
    }
    const int num_waves =
        (static_cast<int>(launched_blocks) + max_blocks - 1) / max_blocks;
    csvOut("compressed_query", "all", "shared_mem_size", shared_mem_size);
    csvOut("compressed_query", "all", "max_blocks_per_SM", max_blocks_per_SM);
    csvOut("compressed_query", "all", "num_SMs", num_SMs);
    csvOut("compressed_query", "all", "max_blocks", max_blocks);
    csvOut("compressed_query", "all", "num_blocks", launched_blocks);
    csvOut("compressed_query", "all", "num_waves", num_waves);

    // Allocate GPU memory for runtime calculations
    nsecs* d_block_decompress_runtimes;
    cudaError_t result = g_allocator.DeviceAllocate(reinterpret_cast<void **>(&d_block_decompress_runtimes),
                                                    block_decompress_runtimes_size);
    if (result != cudaSuccess) {
        csvOut("compressed_query", "error", "Failed to allocate block decompress runtimes",
               cudaGetErrorString(result));
        return -1;
    }
    nsecs* d_block_partial_sums_runtimes;
    result = g_allocator.DeviceAllocate(reinterpret_cast<void **>(&d_block_partial_sums_runtimes),
                                                    block_partial_sums_runtimes_size);
    if (result != cudaSuccess) {
        csvOut("compressed_query", "error", "Failed to allocate block partial sums runtimes",
               cudaGetErrorString(result));
        return -1;
    }
    nsecs* d_block_reduce_runtimes;
    result = g_allocator.DeviceAllocate(reinterpret_cast<void **>(&d_block_reduce_runtimes),
                                                    block_reduce_runtimes_size);
    if (result != cudaSuccess) {
        csvOut("compressed_query", "error", "Failed to allocate block reduce runtimes",
               cudaGetErrorString(result));
        return -1;
    }

    // Declare CPU side timing variables
    vector<nsecs> h_block_decompress_runtimes(num_tiles);
    vector<nsecs> h_block_partial_sums_runtimes(num_tiles);
    vector<nsecs> h_block_reduce_runtimes(num_tiles);
    vector<nsecs> h_block_runtimes(num_tiles);
    constexpr int registers_stage_count = 4;
    constexpr int registers_stage_key_decode_idx = 0;
    constexpr int registers_stage_value_decode_idx = 1;
    constexpr int registers_stage_reduce_idx = 2;
    constexpr int registers_stage_decode_total_idx = 3;
    const size_t registers_stage_totals_size = registers_stage_count * sizeof(nsecs);
    thrust::device_vector<nsecs> d_registers_stage_totals;
    nsecs* d_registers_stage_totals_ptr = nullptr;
    vector<nsecs> h_registers_stage_totals(registers_stage_count, 0);
    nsecs full_lpt_runtimes_sum = 0;
    nsecs decompress_lpt_runtimes_sum = 0;
    nsecs partial_sums_lpt_runtimes_sum = 0;
    nsecs reduce_lpt_runtimes_sum = 0;
    nsecs registers_key_decode_ns_sum = 0;
    nsecs registers_value_decode_ns_sum = 0;
    nsecs registers_reduce_ns_sum = 0;
    nsecs registers_decode_total_ns_sum = 0;
    vector<float> all_elapsed_ms;

    if (exec_mode == RFOR_REGISTERS) {
        d_registers_stage_totals.resize(registers_stage_count);
        d_registers_stage_totals_ptr = d_registers_stage_totals.data().get();
    }

    for (int run = 0; run < NUM_RUNS; ++run) {
        int* d_tuples;
        sum_type* d_sums;
        vector<char*> d_col_buffers;
        vector<RLETileMetadata*> h_metadata_ptrs;
        vector<uint32_t*> h_val_data_ptrs;
        vector<uint32_t*> h_rl_data_ptrs;

        h_metadata_ptrs.resize(num_chunks);
        h_val_data_ptrs.resize(num_chunks);
        h_rl_data_ptrs.resize(num_chunks);

        result = cudaMemset(d_block_decompress_runtimes, 0, block_decompress_runtimes_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to zero out block decompress runtimes",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            return -1;
        }
        result = cudaMemset(d_block_partial_sums_runtimes, 0, block_partial_sums_runtimes_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to zero out block partial sums runtimes",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            return -1;
        }
        result = cudaMemset(d_block_reduce_runtimes, 0, block_reduce_runtimes_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to zero out block reduce runtimes",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            return -1;
        }
        if (exec_mode == RFOR_REGISTERS) {
            result = cudaMemset(d_registers_stage_totals_ptr, 0, registers_stage_totals_size);
            if (result != cudaSuccess) {
                csvOut("compressed_query", toCsvString(run), "Failed to zero out registers stage totals",
                       cudaGetErrorString(result));
                g_allocator.DeviceFree(d_block_decompress_runtimes);
                g_allocator.DeviceFree(d_block_partial_sums_runtimes);
                g_allocator.DeviceFree(d_block_reduce_runtimes);
                return -1;
            }
        }

        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_tuples), d_tuples_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to allocate output tuples",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            return -1;
        }

        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_sums), sums_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to allocate output sums",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            return -1;
        }
        result = cudaMemset(d_sums, 0, sums_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to zero out output sums",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }

        bool allocation_failed = false;

        // Allocate and upload separate buffer for each column
        for (int col = 0; col < num_cols && !allocation_failed; ++col) {
            char* d_col_buffer;
            result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_col_buffer), col_total_sizes[col]);
            if (result != cudaSuccess) {
                csvOut("compressed_query", toCsvString(run), "Failed to allocate column buffer",
                       std::string("col=") + toCsvString(col) + ", error=" + cudaGetErrorString(result));
                allocation_failed = true;
                break;
            }

            cudaMemcpy(d_col_buffer, h_col_buffers[col].data(), col_total_sizes[col], cudaMemcpyHostToDevice);
            d_col_buffers.push_back(d_col_buffer);

            // Build pointer arrays for this column
            for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                char* chunk_ptr = d_col_buffer + chunk_offsets_per_col[col][tile_idx];

                auto metadata = reinterpret_cast<RLETileMetadata*>(chunk_ptr);

                RLETileMetadata temp_metadata;
                cudaMemcpy(&temp_metadata, metadata, sizeof(RLETileMetadata), cudaMemcpyDeviceToHost);

                const uint32_t val_packed_words = (temp_metadata.val_count * temp_metadata.val_bitwidth + 31) / 32;
                const uint32_t val_section_size = sizeof(RLETileMetadata) + val_packed_words * sizeof(uint32_t);

                auto val_data = reinterpret_cast<uint32_t*>(chunk_ptr + sizeof(RLETileMetadata));
                auto rl_data = reinterpret_cast<uint32_t*>(chunk_ptr + val_section_size);

                const size_t ptr_index = col * num_tiles + tile_idx;
                h_metadata_ptrs[ptr_index] = metadata;
                h_val_data_ptrs[ptr_index] = val_data;
                h_rl_data_ptrs[ptr_index] = rl_data;
            }
        }
        if (allocation_failed) {
            for (const auto& ptr : d_col_buffers) {
                g_allocator.DeviceFree(ptr);
            }
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }

        RLETileMetadata** d_metadata_ptrs;
        uint32_t** d_val_data_ptrs;
        uint32_t** d_rl_data_ptrs;

        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_metadata_ptrs), metadata_ptrs_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to allocate metadata pointers",
                   cudaGetErrorString(result));
            for (const auto& ptr : d_col_buffers) {
                g_allocator.DeviceFree(ptr);
            }
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }
        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_val_data_ptrs), val_data_ptrs_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to allocate value data pointers",
                   cudaGetErrorString(result));
            for (const auto& ptr : d_col_buffers) {
                g_allocator.DeviceFree(ptr);
            }
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            g_allocator.DeviceFree(d_sums);
            g_allocator.DeviceFree(d_metadata_ptrs);
            return -1;
        }
        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_rl_data_ptrs), rl_data_ptrs_size);
        if (result != cudaSuccess) {
            csvOut("compressed_query", toCsvString(run), "Failed to allocate run length pointers",
                   cudaGetErrorString(result));
            for (const auto& ptr : d_col_buffers) {
                g_allocator.DeviceFree(ptr);
            }
            g_allocator.DeviceFree(d_block_decompress_runtimes);
            g_allocator.DeviceFree(d_block_partial_sums_runtimes);
            g_allocator.DeviceFree(d_block_reduce_runtimes);
            g_allocator.DeviceFree(d_tuples);
            g_allocator.DeviceFree(d_sums);
            g_allocator.DeviceFree(d_metadata_ptrs);
            g_allocator.DeviceFree(d_val_data_ptrs);
            return -1;
        }

        cudaMemcpy(d_metadata_ptrs, h_metadata_ptrs.data(), metadata_ptrs_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_data_ptrs, h_val_data_ptrs.data(), val_data_ptrs_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rl_data_ptrs, h_rl_data_ptrs.data(), rl_data_ptrs_size, cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        dim3 grid(num_tiles);
        if (exec_mode == RFOR_SHARED) {
            grid = dim3((static_cast<int>(num_tiles) +
                         RFOR_SHARED_TILES_PER_BLOCK - 1) /
                        RFOR_SHARED_TILES_PER_BLOCK);
        } else if (exec_mode == RFOR_WARP) {
            grid = dim3((static_cast<int>(num_tiles) + warps_per_block - 1) / warps_per_block);
        }
        dim3 block_dim(THREADS_PER_TILE);

        csvOut("compressed_query", toCsvString(run), "run", run);

        {
            ScopedProfilingWindow profiling_window(use_proteus_profiling_control);

            if (exec_mode == RFOR_SHARED) {
                compressed_block_to_shared_tuples<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size,
                    stream>>>(d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_sums, d_block_decompress_runtimes,
                              d_block_partial_sums_runtimes, d_block_reduce_runtimes, filter, num_tiles, num_tuples,
                              num_cols, num_sums);
            } else if (exec_mode == RFOR_REGISTERS) {
                compressed_block_to_tuple_registers<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size,
                    stream>>>(d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_tuples, d_sums,
                              d_block_decompress_runtimes, d_block_partial_sums_runtimes, d_block_reduce_runtimes,
                              d_registers_stage_totals_ptr, filter, num_tiles, num_tuples, num_cols, num_sums);
            } else if (exec_mode == RFOR_PREDECODE) {
                compressed_block_to_tuple_predecode<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size,
                    stream>>>(d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_tuples, d_sums, d_block_decompress_runtimes,
                              d_block_partial_sums_runtimes, d_block_reduce_runtimes, filter, num_tiles, num_tuples,
                              num_cols, num_sums);
            } else if (exec_mode == RFOR_WARP) {
                compressed_block_to_tuple_warp<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size,
                    stream>>>(d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_tuples, d_sums, d_block_decompress_runtimes,
                              d_block_partial_sums_runtimes, d_block_reduce_runtimes, filter, num_tiles, num_tuples,
                              num_cols, num_sums);
            } else {
                compressed_block_to_tuple<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size, stream>>>(
                    d_metadata_ptrs, d_val_data_ptrs, d_rl_data_ptrs, d_tuples, d_sums, d_block_decompress_runtimes,
                    d_block_partial_sums_runtimes, d_block_reduce_runtimes, filter, num_tiles, num_tuples, num_cols,
                    num_sums);
            }

            cudaError_t kernel_error = cudaGetLastError();
            if (kernel_error != cudaSuccess) {
                csvOut("compressed_query", toCsvString(run), "Kernel launch failed",
                       cudaGetErrorString(kernel_error));
                return -1;
            }
            kernel_error = cudaDeviceSynchronize();
            if (kernel_error != cudaSuccess) {
                csvOut("compressed_query", toCsvString(run), "Runtime error in Kernel",
                       cudaGetErrorString(kernel_error));
                return -1;
            }
        }

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        if (run == 0) {
            cudaMemcpy(h_sums.data(), d_sums, sums_size, cudaMemcpyDeviceToHost);
            csvOut("compressed_query", toCsvString(run), "warmup", "ignoring elapsed_ms and lpt_ns");
        } else {
            vector<sum_type> h_sums_compare(num_sums);
            cudaMemcpy(h_sums_compare.data(), d_sums, sums_size, cudaMemcpyDeviceToHost);
            if (h_sums != h_sums_compare) {
                csvOut("compressed_query", toCsvString(run), "Query resulted in differing sum", "true");
                return -1;
            }
            all_elapsed_ms.push_back(kernel_time_ms);
            csvOut("compressed_query", toCsvString(run), "elapsed_ms", kernel_time_ms);

            cudaMemcpy(h_block_decompress_runtimes.data(), d_block_decompress_runtimes, block_decompress_runtimes_size,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_block_partial_sums_runtimes.data(), d_block_partial_sums_runtimes,
                       block_partial_sums_runtimes_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_block_reduce_runtimes.data(), d_block_reduce_runtimes, block_reduce_runtimes_size,
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < h_block_runtimes.size(); ++i) {
                h_block_runtimes[i] = h_block_decompress_runtimes[i] + h_block_partial_sums_runtimes[i] +
                                      h_block_reduce_runtimes[i];
            }

            vector<tuple<nsecs, nsecs, nsecs, nsecs>> block_runtimes;
            if (exec_mode == RFOR_SHARED) {
                const size_t num_groups =
                    (num_tiles + RFOR_SHARED_TILES_PER_BLOCK - 1) / RFOR_SHARED_TILES_PER_BLOCK;
                block_runtimes.resize(num_groups);
                for (size_t g = 0; g < num_groups; ++g) {
                    nsecs g_full = 0;
                    nsecs g_dec = 0;
                    nsecs g_part = 0;
                    nsecs g_red = 0;
                    const size_t base = g * RFOR_SHARED_TILES_PER_BLOCK;
                    const size_t end = std::min(base + RFOR_SHARED_TILES_PER_BLOCK, num_tiles);
                    for (size_t i = base; i < end; ++i) {
                        g_full += h_block_runtimes[i];
                        g_dec += h_block_decompress_runtimes[i];
                        g_part += h_block_partial_sums_runtimes[i];
                        g_red += h_block_reduce_runtimes[i];
                    }
                    block_runtimes[g] = make_tuple(g_full, g_dec, g_part, g_red);
                }
            } else {
                block_runtimes.resize(h_block_runtimes.size());
                for (size_t i = 0; i < h_block_runtimes.size(); ++i) {
                    block_runtimes[i] = make_tuple(
                        h_block_runtimes[i], h_block_decompress_runtimes[i],
                        h_block_partial_sums_runtimes[i], h_block_reduce_runtimes[i]);
                }
            }
            const auto lpt_runtimes = longest_processing_time_first_compressed(block_runtimes, max_blocks);
            const auto full_lpt_runtime = get<0>(lpt_runtimes);
            const auto decompress_lpt_runtime = get<1>(lpt_runtimes);
            const auto partial_sums_lpt_runtime = get<2>(lpt_runtimes);
            const auto reduce_lpt_runtime = get<3>(lpt_runtimes);
            full_lpt_runtimes_sum += full_lpt_runtime;
            decompress_lpt_runtimes_sum += decompress_lpt_runtime;
            partial_sums_lpt_runtimes_sum += partial_sums_lpt_runtime;
            reduce_lpt_runtimes_sum += reduce_lpt_runtime;
            csvOut("compressed_query", toCsvString(run), "decompress_lpt_ns", decompress_lpt_runtime);
            csvOut("compressed_query", toCsvString(run), "partial_sums_lpt_ns", partial_sums_lpt_runtime);
            csvOut("compressed_query", toCsvString(run), "reduce_lpt_ns", reduce_lpt_runtime);
            csvOut("compressed_query", toCsvString(run), "full_lpt_ns", full_lpt_runtime);

            if (exec_mode == RFOR_REGISTERS) {
                cudaMemcpy(h_registers_stage_totals.data(), d_registers_stage_totals_ptr, registers_stage_totals_size,
                           cudaMemcpyDeviceToHost);
                const nsecs key_decode_stage_ns =
                    h_registers_stage_totals[registers_stage_key_decode_idx];
                const nsecs value_decode_stage_ns =
                    h_registers_stage_totals[registers_stage_value_decode_idx];
                const nsecs reduce_stage_ns =
                    h_registers_stage_totals[registers_stage_reduce_idx];
                const nsecs decode_total_stage_ns =
                    h_registers_stage_totals[registers_stage_decode_total_idx];
                registers_key_decode_ns_sum += key_decode_stage_ns;
                registers_value_decode_ns_sum += value_decode_stage_ns;
                registers_reduce_ns_sum += reduce_stage_ns;
                registers_decode_total_ns_sum += decode_total_stage_ns;
                csvOut("compressed_query", toCsvString(run), "registers_key_decode_ns_sum_of_block_max", key_decode_stage_ns);
                csvOut("compressed_query", toCsvString(run), "registers_value_decode_ns_sum_of_block_max", value_decode_stage_ns);
                csvOut("compressed_query", toCsvString(run), "registers_reduce_ns_sum", reduce_stage_ns);
                csvOut("compressed_query", toCsvString(run), "registers_decode_total_ns_sum", decode_total_stage_ns);
            }
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        // Cleanup
        for (const auto& ptr : d_col_buffers) {
            g_allocator.DeviceFree(ptr);
        }
        g_allocator.DeviceFree(d_tuples);
        g_allocator.DeviceFree(d_sums);
        g_allocator.DeviceFree(d_metadata_ptrs);
        g_allocator.DeviceFree(d_val_data_ptrs);
        g_allocator.DeviceFree(d_rl_data_ptrs);
    }

    g_allocator.DeviceFree(d_block_decompress_runtimes);
    g_allocator.DeviceFree(d_block_partial_sums_runtimes);
    g_allocator.DeviceFree(d_block_reduce_runtimes);

    const auto mean_full_lpt_runtime = static_cast<double>(full_lpt_runtimes_sum) / (NUM_RUNS - 1);
    const auto mean_decompress_lpt_runtime = static_cast<double>(decompress_lpt_runtimes_sum) / (NUM_RUNS - 1);
    const auto mean_partial_sums_lpt_runtime = static_cast<double>(partial_sums_lpt_runtimes_sum) / (NUM_RUNS - 1);
    const auto mean_reduce_lpt_runtime = static_cast<double>(reduce_lpt_runtimes_sum) / (NUM_RUNS - 1);
    const auto elapsed_ms = accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f) / static_cast<float>(
                                all_elapsed_ms.size());

    csvOut("compressed_query", "summary", "mean_decompress_lpt_ns", mean_decompress_lpt_runtime);
    csvOut("compressed_query", "summary", "mean_partial_sums_lpt_ns", mean_partial_sums_lpt_runtime);
    csvOut("compressed_query", "summary", "mean_reduce_lpt_ns", mean_reduce_lpt_runtime);
    csvOut("compressed_query", "summary", "mean_full_lpt_ns", mean_full_lpt_runtime);
    if (exec_mode == RFOR_REGISTERS) {
        const auto mean_registers_key_decode_ns =
            static_cast<double>(registers_key_decode_ns_sum) / (NUM_RUNS - 1);
        const auto mean_registers_value_decode_ns =
            static_cast<double>(registers_value_decode_ns_sum) / (NUM_RUNS - 1);
        const auto mean_registers_reduce_ns =
            static_cast<double>(registers_reduce_ns_sum) / (NUM_RUNS - 1);
        const auto mean_registers_decode_total_ns =
            static_cast<double>(registers_decode_total_ns_sum) / (NUM_RUNS - 1);
        csvOut("compressed_query", "summary", "mean_registers_key_decode_ns_sum_of_block_max",
               mean_registers_key_decode_ns);
        csvOut("compressed_query", "summary", "mean_registers_value_decode_ns_sum_of_block_max",
               mean_registers_value_decode_ns);
        csvOut("compressed_query", "summary", "mean_registers_reduce_ns_sum",
               mean_registers_reduce_ns);
        csvOut("compressed_query", "summary", "mean_registers_decode_total_ns_sum",
               mean_registers_decode_total_ns);
    }
    csvOut("compressed_query", "summary", "mean_per_col_compression_ratio", mean_per_col_compression_ratio);
    csvOut("compressed_query", "summary", "std_dev_per_col_compression_ratio", std_dev_per_col_compression_ratio);
    csvOut("compressed_query", "summary", "total_compression_ratio", total_compression_ratio);

    return elapsed_ms;
}

// TODO: Test the effects of the `#pragma unroll`s, rewrite the code to be simpler given that the loops are unrolled
// TODO: Try to make this simpler in general
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void uncompressed_tuple_query(
    int** __restrict__ d_cols_data,
    sum_type* __restrict__ d_col_sums,
    nsecs* __restrict__ d_block_runtimes,
    const int filter,
    const int num_sums,
    const int num_tuples) {
    (void)filter;
    (void)num_sums;

    nsecs time_start = 0;
    if (threadIdx.x == 0) {
        time_start = globaltimer64();
    }

    // Prepare shared memory
    extern __shared__ unsigned char shared_buffer[];
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    constexpr size_t temp_storage_size = sizeof(typename BlockReduce::TempStorage);
    constexpr size_t padded_temp_storage_size = (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT *
                                                D_ARRAY_ALIGNMENT;
    auto temp_storage = reinterpret_cast<typename BlockReduce::TempStorage*>(shared_buffer);
    (void)padded_temp_storage_size;

    const auto tile_id = static_cast<size_t>(blockIdx.x);
    const size_t tile_offset = tile_id * ELEMS_PER_TILE;
    const size_t first_tuple_idx = tile_offset + threadIdx.x;
    const int* orderdate_col = d_cols_data[SSB_Q11_COL_ORDERDATE];
    const int* quantity_col = d_cols_data[SSB_Q11_COL_QUANTITY];
    const int* extendedprice_col = d_cols_data[SSB_Q11_COL_EXTENDEDPRICE];
    const int* discount_col = d_cols_data[SSB_Q11_COL_DISCOUNT];

    sum_type thread_sum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (const size_t tuple_idx = first_tuple_idx + i * BLOCK_THREADS; tuple_idx < num_tuples) {
            const int orderdate = orderdate_col[tuple_idx];
            const int quantity = quantity_col[tuple_idx];
            const int extendedprice = extendedprice_col[tuple_idx];
            const int discount = discount_col[tuple_idx];
            if (passesSSBQ11(orderdate, quantity, discount)) {
                thread_sum += revenueSSBQ11(extendedprice, discount);
            }
        }
    }

    const sum_type tile_sum = BlockReduce(*temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<sum_type*>(d_col_sums + 0), tile_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const nsecs time_end = globaltimer64();
        d_block_runtimes[blockIdx.x] += time_end - time_start;
    }
}

// TODO: Add in and test the effects of `#pragma unroll`, rewriting to take advantage of unrolled loops
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void uncompressed_columnar_query(
    int** __restrict__ d_cols_data,
    sum_type* __restrict__ d_col_sums,
    nsecs* __restrict__ d_block_runtimes,
    const int filter,
    const int num_sums,
    const int num_tuples) {
    (void)filter;
    (void)num_sums;

    nsecs time_start = 0;
    if (threadIdx.x == 0) {
        time_start = globaltimer64();
    }

    // Prepare shared memory
    typedef BlockReduce<sum_type, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const auto tile_id = static_cast<size_t>(blockIdx.x);
    const size_t tile_offset = tile_id * ELEMS_PER_TILE;
    const size_t first_tuple_idx = tile_offset + threadIdx.x;
    const int* orderdate_col = d_cols_data[SSB_Q11_COL_ORDERDATE];
    const int* quantity_col = d_cols_data[SSB_Q11_COL_QUANTITY];
    const int* extendedprice_col = d_cols_data[SSB_Q11_COL_EXTENDEDPRICE];
    const int* discount_col = d_cols_data[SSB_Q11_COL_DISCOUNT];

    sum_type thread_sum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (const size_t tuple_idx = first_tuple_idx + i * BLOCK_THREADS; tuple_idx < num_tuples) {
            const int orderdate = orderdate_col[tuple_idx];
            const int quantity = quantity_col[tuple_idx];
            const int extendedprice = extendedprice_col[tuple_idx];
            const int discount = discount_col[tuple_idx];
            if (passesSSBQ11(orderdate, quantity, discount)) {
                thread_sum += revenueSSBQ11(extendedprice, discount);
            }
        }
    }

    const sum_type tile_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<sum_type*>(d_col_sums + 0), tile_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const nsecs time_end = globaltimer64();
        d_block_runtimes[blockIdx.x] += time_end - time_start;
    }
}

float run_uncompressed_query(
    const vector<vector<int>>& h_cols_data,
    vector<sum_type>& h_sums,
    const size_t per_col_size,
    const int filter,
    const int num_cols,
    const int num_sums,
    const int num_tuples,
    const int num_SMs,
    const bool use_proteus_profiling_control,
    const ExecutionMode exec_mode) {
    (void)filter;

    const int num_tiles = (num_tuples + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE;
    vector<float> all_elapsed_ms;

    const size_t d_col_ptrs_size = sizeof(int*) * num_cols;
    const size_t sums_size = sizeof(sum_type) * num_sums;
    const size_t block_runtimes_size = sizeof(nsecs) * num_tiles;

    size_t shared_mem_size = 0;
    int max_blocks_per_SM;
    if (exec_mode == UNCOMPRESSED_COLUMN) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      uncompressed_columnar_query<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    } else {
        constexpr size_t temp_storage_size = sizeof(typename BlockReduce<sum_type, THREADS_PER_TILE>::TempStorage);
        constexpr size_t padded_temp_storage_size =
            (temp_storage_size + D_ARRAY_ALIGNMENT - 1) / D_ARRAY_ALIGNMENT * D_ARRAY_ALIGNMENT;
        shared_mem_size = padded_temp_storage_size;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
                                                      uncompressed_tuple_query<THREADS_PER_TILE, ELEMS_PER_THREAD>,
                                                      THREADS_PER_TILE, shared_mem_size);
    }
    const int max_blocks = max_blocks_per_SM * num_SMs;
    const int num_waves = (num_tiles + max_blocks - 1) / max_blocks;
    csvOut("uncompressed_query", "all", "shared_mem_size", shared_mem_size);
    csvOut("uncompressed_query", "all", "max_blocks_per_SM", max_blocks_per_SM);
    csvOut("uncompressed_query", "all", "num_SMs", num_SMs);
    csvOut("uncompressed_query", "all", "max_blocks", max_blocks);
    csvOut("uncompressed_query", "all", "num_blocks", num_tiles);
    csvOut("uncompressed_query", "all", "num_waves", num_waves);

    nsecs* d_block_runtimes;
    cudaError_t result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_block_runtimes), block_runtimes_size);
    if (result != cudaSuccess) {
        csvOut("uncompressed_query", "error", "Failed to allocate block runtimes",
               cudaGetErrorString(result));
        return -1;
    }

    vector<nsecs> h_block_runtimes(num_tiles);
    nsecs lpt_runtimes_sum = 0;

    for (int run = 0; run < NUM_RUNS; ++run) {
        sum_type* d_sums;
        vector<int*> d_cols_data(num_cols);
        int** d_col_ptrs;

        result = cudaMemset(d_block_runtimes, 0, block_runtimes_size);
        if (result != cudaSuccess) {
            csvOut("uncompressed_query", toCsvString(run), "Failed to zero out block runtimes",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_runtimes);
            return -1;
        }

        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_sums), sums_size);
        if (result != cudaSuccess) {
            csvOut("uncompressed_query", toCsvString(run), "Failed to allocate output sums",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_runtimes);
            return -1;
        }
        result = cudaMemset(d_sums, 0, sums_size);
        if (result != cudaSuccess) {
            csvOut("uncompressed_query", toCsvString(run), "Failed to zero out output sums",
                   cudaGetErrorString(result));
            g_allocator.DeviceFree(d_block_runtimes);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }

        bool allocation_failed = false;
        for (int col = 0; col < num_cols && !allocation_failed; ++col) {
            result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_cols_data[col]), per_col_size);
            if (result != cudaSuccess) {
                csvOut("uncompressed_query", toCsvString(run), "Failed to allocate column data",
                       std::string("col=") + toCsvString(col) + ", error=" + cudaGetErrorString(result));
                allocation_failed = true;
                break;
            }
            cudaMemcpy(d_cols_data[col], h_cols_data[col].data(), per_col_size, cudaMemcpyHostToDevice);
        }
        if (allocation_failed) {
            for (int col = 0; col < num_cols; ++col) {
                if (d_cols_data[col]) {
                    g_allocator.DeviceFree(d_cols_data[col]);
                }
            }
            g_allocator.DeviceFree(d_block_runtimes);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }

        result = g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_col_ptrs), d_col_ptrs_size);
        if (result != cudaSuccess) {
            csvOut("uncompressed_query", toCsvString(run), "Failed to allocate column pointers",
                   cudaGetErrorString(result));
            for (const auto& col_data : d_cols_data) {
                g_allocator.DeviceFree(col_data);
            }
            g_allocator.DeviceFree(d_block_runtimes);
            g_allocator.DeviceFree(d_sums);
            return -1;
        }
        cudaMemcpy(d_col_ptrs, d_cols_data.data(), d_col_ptrs_size, cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        cudaStreamSynchronize(stream);

        cudaEventRecord(start_event, stream);

        dim3 grid(num_tiles);
        dim3 block_dim(THREADS_PER_TILE);

        csvOut("uncompressed_query", toCsvString(run), "run", run);

        {
            ScopedProfilingWindow profiling_window(use_proteus_profiling_control);

            if (exec_mode == UNCOMPRESSED_COLUMN) {
                uncompressed_columnar_query<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size, stream>>>(
                    d_col_ptrs, d_sums, d_block_runtimes, filter, num_sums, num_tuples);
            } else {
                uncompressed_tuple_query<THREADS_PER_TILE, ELEMS_PER_THREAD><<<grid, block_dim, shared_mem_size, stream>>>(
                    d_col_ptrs, d_sums, d_block_runtimes, filter, num_sums, num_tuples);
            }

            cudaError_t kernel_error = cudaGetLastError();
            if (kernel_error != cudaSuccess) {
                csvOut("uncompressed_query", toCsvString(run), "Kernel launch failed",
                       cudaGetErrorString(kernel_error));
                return -1;
            }
            kernel_error = cudaDeviceSynchronize();
            if (kernel_error != cudaSuccess) {
                csvOut("uncompressed_query", toCsvString(run), "Runtime error in kernel",
                       cudaGetErrorString(kernel_error));
                return -1;
            }
        }

        cudaEventRecord(stop_event, stream);
        cudaStreamSynchronize(stream);

        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

        if (run == 0) {
            cudaMemcpy(h_sums.data(), d_sums, sums_size, cudaMemcpyDeviceToHost);
            csvOut("uncompressed_query", toCsvString(run), "warmup", "ignoring elapsed_ms and lpt_ns");
        } else {
            vector<sum_type> h_sums_compare(num_sums);
            cudaMemcpy(h_sums_compare.data(), d_sums, sums_size, cudaMemcpyDeviceToHost);
            if (h_sums != h_sums_compare) {
                csvOut("uncompressed_query", toCsvString(run), "Query resulted in differing sum", "true");
                return -1;
            }
            all_elapsed_ms.push_back(kernel_time_ms);
            csvOut("uncompressed_query", toCsvString(run), "elapsed_ms", kernel_time_ms);

            cudaMemcpy(h_block_runtimes.data(), d_block_runtimes, block_runtimes_size, cudaMemcpyDeviceToHost);
            const auto lpt_runtime = longest_processing_time_first_uncompressed(h_block_runtimes, max_blocks);
            lpt_runtimes_sum += lpt_runtime;
            csvOut("uncompressed_query", toCsvString(run), "lpt_ns", lpt_runtime);
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);

        for (const auto& col_data : d_cols_data) {
            g_allocator.DeviceFree(col_data);
        }
        g_allocator.DeviceFree(d_col_ptrs);
        g_allocator.DeviceFree(d_sums);
    }
    g_allocator.DeviceFree(d_block_runtimes);

    const auto mean_lpt_runtime = static_cast<double>(lpt_runtimes_sum) / (NUM_RUNS - 1);
    const auto elapsed_ms = accumulate(all_elapsed_ms.begin(), all_elapsed_ms.end(), 0.0f) / static_cast<float>(
                                all_elapsed_ms.size());

    csvOut("uncompressed_query", "summary", "mean_lpt_ns", mean_lpt_runtime);

    return elapsed_ms;
}

template<typename T>
T* loadColumn(const string& filePath, const size_t num_elements) {
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        csvOut("main", "error", "file_open", std::string("Failed to open file: ") + filePath);
        return nullptr;
    }

    T* data = new T[num_elements];
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));

    if (file.gcount() != static_cast<streamsize>(num_elements * sizeof(T))) {
        csvOut("main", "error", "file_read", "Failed to read expected amount of data from file");
        delete[] data;
        return nullptr;
    }

    file.close();
    return data;
}

size_t getFileSize(const string& filePath) {
    ifstream file(filePath, ios::binary | ios::ate);
    if (!file.is_open()) {
        csvOut("main", "error", "file_open", std::string("Failed to open file: ") + filePath);
        return 0;
    }
    const size_t size = file.tellg();
    file.close();
    return size;
}

const char* executionModeToString(const ExecutionMode mode) {
    switch (mode) {
        case RFOR_SHARED: return "RFOR_SHARED";
        case RFOR_GLOBAL: return "RFOR_GLOBAL";
        case RFOR_REGISTERS: return "RFOR_REGISTERS";
        case RFOR_PREDECODE: return "RFOR_PREDECODE";
        case RFOR_WARP: return "RFOR_WARP";
        case UNCOMPRESSED_COLUMN: return "UNCOMPRESSED_COLUMN";
        case UNCOMPRESSED_TUPLE: return "UNCOMPRESSED_TUPLE";
    }
    return "UNKNOWN";
}

ExecutionMode parseExecutionMode(const string& mode_str) {
    if (mode_str == "RFOR_SHARED") return RFOR_SHARED;
    if (mode_str == "RFOR_GLOBAL") return RFOR_GLOBAL;
    if (mode_str == "RFOR_REGISTERS") return RFOR_REGISTERS;
    if (mode_str == "RFOR_PREDECODE") return RFOR_PREDECODE;
    if (mode_str == "RFOR_WARP") return RFOR_WARP;
    if (mode_str == "UNCOMPRESSED_COLUMN") return UNCOMPRESSED_COLUMN;
    if (mode_str == "UNCOMPRESSED_TUPLE") return UNCOMPRESSED_TUPLE;
    throw invalid_argument("Invalid mode: " + mode_str);
}

vector<ExecutionMode> allExecutionModes() {
    return {
        //UNCOMPRESSED_COLUMN,
        UNCOMPRESSED_TUPLE,
        RFOR_GLOBAL,
        RFOR_SHARED,
        RFOR_REGISTERS,
        RFOR_PREDECODE,
        //RFOR_WARP
    };
}

bool useProteusProfilingControl() {
    const char* env = std::getenv("USE_PROTEUS_PROFILING_CONTROL");
    if (env == nullptr) return false;
    const string value(env);
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

int main(int argc, char** argv) {
    csvOut("main", "all", "header", "---------------------------------------------------------------------------------------------");

    int first_col_arg = FIRST_COL_ARG;
    string mode_selector = "ALL";
    if (argc == FIRST_COL_ARG + MIN_COLUMNS) {
        // default: all modes
    } else if (argc == FIRST_COL_ARG + MIN_COLUMNS + 1) {
        mode_selector = argv[FIRST_COL_ARG];
        first_col_arg = FIRST_COL_ARG + 1;
    } else if (argc == FIRST_COL_ARG + MIN_COLUMNS + 2 && string(argv[FIRST_COL_ARG]) == "--mode") {
        mode_selector = argv[FIRST_COL_ARG + 1];
        first_col_arg = FIRST_COL_ARG + 2;
    } else {
        csvOut("main", "error", "usage",
               std::string("Usage: ") + argv[0] +
               " [--mode <ALL|RFOR_SHARED|RFOR_GLOBAL|RFOR_REGISTERS|RFOR_PREDECODE|RFOR_WARP|UNCOMPRESSED_COLUMN|UNCOMPRESSED_TUPLE>] "
               "<lo_orderdate_file> <lo_quantity_file> <lo_extendedprice_file> <lo_discount_file>");
        csvOut("main", "error", "usage_alt",
               std::string("Usage: ") + argv[0] +
               " [ALL|RFOR_SHARED|RFOR_GLOBAL|RFOR_REGISTERS|RFOR_PREDECODE|RFOR_WARP|UNCOMPRESSED_COLUMN|UNCOMPRESSED_TUPLE] "
               "<lo_orderdate_file> <lo_quantity_file> <lo_extendedprice_file> <lo_discount_file>");
        return 1;
    }

    vector<ExecutionMode> exec_modes;
    if (mode_selector == "ALL" || mode_selector == "all") {
        exec_modes = allExecutionModes();
    } else {
        try {
            exec_modes = {parseExecutionMode(mode_selector)};
        } catch (const invalid_argument& e) {
            csvOut("main", "error", "mode", e.what());
            return 1;
        }
    }
    const bool use_proteus_profiling_control = useProteusProfilingControl();

    const int num_cols = SSB_Q11_NUM_COLS;
    const int num_sums = SSB_Q11_NUM_SUMS;
    vector<string> col_files(num_cols);
    for (int col = 0; col < num_cols; ++col) {
        col_files[col] = argv[col + first_col_arg];
    }

    csvOut("main", "all", "query", "SSB Q1.1");
    csvOut("main", "all", "mode_selector", mode_selector);
    csvOut("main", "all", "use_proteus_profiling_control", use_proteus_profiling_control);
    csvOut("main", "all", "predicate_orderdate_gt", SSB_Q11_ORDERDATE_GT);
    csvOut("main", "all", "predicate_orderdate_lt", SSB_Q11_ORDERDATE_LT);
    csvOut("main", "all", "predicate_discount_min", SSB_Q11_DISCOUNT_MIN);
    csvOut("main", "all", "predicate_discount_max", SSB_Q11_DISCOUNT_MAX);
    csvOut("main", "all", "predicate_quantity_lt", SSB_Q11_QUANTITY_MAX_EXCLUSIVE);
    for (int col = 0; col < num_cols; ++col) {
        csvOut("main", "all", std::string("column_") + toCsvString(col), col_files[col]);
    }

    // Compute the size of the data set, and the number of tuples
    const size_t per_col_size = getFileSize(col_files[0]);
    for (int col = 1; col < num_cols; ++col) {
        if (const size_t col_size = getFileSize(col_files[col]); col_size < per_col_size) {
            csvOut("main", "error", "column_size", "The key (first) col must not have more elements than any other col");
            return 1;
        }
    }
    if (per_col_size % sizeof(int) != 0) {
        csvOut("main", "error", "column_size", "Column size must be a multiple of sizeof(int)");
        return 1;
    }
    const size_t total_data_size = per_col_size * num_cols;
    const int num_tuples = static_cast<int>(per_col_size / sizeof(int));

    // Extract column data from column files
    vector<vector<int>> col_data(num_cols);
    for (int col = 0; col < num_cols; ++col) {
        int* raw_data = loadColumn<int>(col_files[col], num_tuples);
        if (!raw_data) {
            csvOut("main", "error", "load_column", std::string("Failed to load column ") + toCsvString(col));
            return 1;
        }
        col_data[col].assign(raw_data, raw_data + num_tuples);
        delete[] raw_data;
    }

    // Query the GPU for the number of SMs it has
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    const int num_SMs = device_properties.multiProcessorCount;

    constexpr int unused_filter = 0;
    bool reference_set = false;
    sum_type reference_revenue = 0;

    for (const auto mode : exec_modes) {
        const char* mode_name = executionModeToString(mode);
        vector<sum_type> sums(num_sums, 0);
        float elapsed_ms = 0;

        csvOut("main", mode_name, "mode_start", mode_name);
        if (mode == UNCOMPRESSED_COLUMN || mode == UNCOMPRESSED_TUPLE) {
            elapsed_ms = run_uncompressed_query(col_data, sums, per_col_size, unused_filter, num_cols, num_sums,
                                                num_tuples, num_SMs, use_proteus_profiling_control, mode);
        } else {
            elapsed_ms = run_compressed_query(col_data, sums, total_data_size, per_col_size, unused_filter, num_cols,
                                              num_sums, num_tuples, num_SMs, use_proteus_profiling_control, mode);
        }
        if (elapsed_ms < 0) {
            csvOut("main", "error", "pipeline", std::string("Pipeline failed in mode ") + mode_name);
            return 1;
        }

        const sum_type mode_revenue = sums.at(0);
        if (!reference_set) {
            reference_revenue = mode_revenue;
            reference_set = true;
        } else if (reference_revenue != mode_revenue) {
            csvOut("main", "error", "sum_mismatch", std::string("Revenue mismatch in mode ") + mode_name);
            csvOut("main", "error", "expected_revenue", reference_revenue);
            csvOut("main", "error", "actual_revenue", mode_revenue);
            return 1;
        }

        const double total_gb = static_cast<double>(total_data_size) / 1e9;
        const double elapsed_s = elapsed_ms * 1e-3;
        const double throughput_gb_s = total_gb / elapsed_s;
        const double throughput_tuples_s = static_cast<double>(num_tuples) / elapsed_s;

        csvOut("main", mode_name, "elapsed_ms", elapsed_ms);
        csvOut("main", mode_name, "total_gb", total_gb);
        csvOut("main", mode_name, "throughput_gb_s", throughput_gb_s);
        csvOut("main", mode_name, "num_tuples", num_tuples);
        csvOut("main", mode_name, "throughput_tuples_s", throughput_tuples_s);
        csvOut("main", mode_name, "revenue", mode_revenue);
        csvOut("main", mode_name, "mode_end", mode_name);
    }

    csvOut("main", "summary", "revenue", reference_revenue);

    return 0;
}
