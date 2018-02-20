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


// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include "gtest/gtest.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"

// #include "nvToolsExt.h"

#include <ittnotify.h>

// #include "llvm/DerivedTypes.h"
// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>

#include "common/gpu/gpu-common.hpp"
#include "common/common.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "util/raw-functions.hpp"
#include "util/raw-pipeline.hpp"
#include "plan/plan-parser.hpp"
#include "util/raw-memory-manager.hpp"
#include "storage/raw-storage-manager.hpp"
#include "multigpu/numa_utils.cuh"
// #include <cuda_profiler_api.h>

#include <vector>
#include <thread>

using namespace llvm;

class RawTestEnvironment2 : public ::testing::Environment {
public:
    virtual void SetUp();
    virtual void TearDown();
};

::testing::Environment *const pools_env = ::testing::AddGlobalTestEnvironment(new RawTestEnvironment2);

void thread_warm_up(){}

void RawTestEnvironment2::SetUp(){
    setbuf(stdout, NULL);

    google::InstallFailureSignalHandler();
    // int devCount;

    // gpu_run(cuInit(0));
    // gpu_run(cuDeviceGetCount(&devCount));

    // device  = new CUdevice [devCount];
    // context = new CUcontext[devCount];

    // for (int i = 0 ; i < devCount ; ++i){
    //     gpu_run(cuDeviceGet(device  + i, i));
    //     gpu_run(cuCtxCreate(context + i, 0, device[i]));
    // }

    // gpu_run(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaFree(0));
    }
    
    // gpu_run(cudaSetDevice(0));

    gpu_run(cudaFree(0));

    // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

    std::vector<std::thread> thrds;
    for (int i = 0 ; i < 32 ; ++i) thrds.emplace_back(thread_warm_up);
    for (auto &t: thrds) t.join();

    // srand(time(0));

    RawPipelineGen::init();
    RawMemoryManager::init();

    gpu_run(cudaSetDevice(0));
}

void RawTestEnvironment2::TearDown(){
    RawMemoryManager::destroy();
}

class MultiGPUTest : public ::testing::Test {
protected:
    virtual void SetUp();
    virtual void TearDown();

    // void launch(void ** args, dim3 gridDim, dim3 blockDim);
    // void launch(void ** args, dim3 gridDim);
    // void launch(void ** args);
    
    void runAndVerify(const char *testLabel, const char* planPath);
    
    bool flushResults = true;
    const char * testPath = TEST_OUTPUTS "/tests-output/";

    const char * catalogJSON = "inputs/plans/catalog.json";
public:
    // CUdevice  *device ;
    // CUcontext *context;

    // sys::PrintStackTraceOnErrorSignal;
    // llvm::PrettyStackTraceProgram X;

    // llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
};

void MultiGPUTest::SetUp   (){
    gpu_run(cudaSetDevice(0));
}

void MultiGPUTest::TearDown(){
    StorageManager::unloadAll();
}

void MultiGPUTest::runAndVerify(const char *testLabel, const char* planPath){
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }
    __itt_resume();

    gpu_run(cudaSetDevice(0));
    
    GpuRawContext * ctx;

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel, false);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    //just to be sure...
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaDeviceSynchronize());
    }
    
    {
        time_block     t("Texecute w sync: ");

        {
            time_block t("Texecute       : ");

            for (RawPipeline * p: pipelines) {
                nvtxRangePushA("pip");
                {
                    time_block t("T: ");
    
                    p->open();
                    p->consume(0);
                    p->close();
                }
                nvtxRangePop();
            }
        }

        //just to be sure...
        for (int i = 0 ; i < devices ; ++i) {
            gpu_run(cudaSetDevice(i));
            gpu_run(cudaDeviceSynchronize());
        }
    }

    __itt_pause();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }

    gpu_run(cudaSetDevice(0));

    EXPECT_TRUE(verifyTestResult(TEST_OUTPUTS "/tests-multigpu-integration/", testLabel));
    shm_unlink(testLabel);
}


TEST_F(MultiGPUTest, gpuDriverSequential) {
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );//, GPU_RESIDENT);

    // StorageManager::loadToGpus("inputs/ssbm/date.csv.d_datekey"             );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/date.csv.d_year"                );//, GPU_RESIDENT);

    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"      , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"      , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"     , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice" , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"             , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/date.csv.d_year"                , PINNED);//GPU_RESIDENT);
    
    const char *testLabel = "gpuSSBM_Q1_1c";
    const char *planPath  = "inputs/plans/ssbm_q1_1c.json";
 
    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuDriverMultiReduce) {
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );
    // StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    // StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    // StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    // StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_discount"      , 3, 1);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_quantity"      , 3, 1);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_orderdate"     , 3, 1);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_extendedprice" , 3, 1);

    const char *testLabel = "gpuDriverMultiReduce";
    const char *planPath  = "inputs/plans/reduce-scan-multigpu.json";

    runAndVerify(testLabel, planPath);
    //SF=10 => -799879732
}

TEST_F(MultiGPUTest, gpuReduceScanRearrange) {
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"    , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice", PINNED);

    const char *testLabel = "gpuReduceScanRearrange";
    const char *planPath  = "inputs/plans/reduce-scan-rearrange-gpu.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuDriverParallel) {
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"    , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice", PINNED);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"            , PINNED);
    StorageManager::load("inputs/ssbm/date.csv.d_year"               , PINNED);

    const char *testLabel = "gpuSSBM_Q1_1_parallel";
    const char *planPath  = "inputs/plans/ssbm_q1_1_parallel.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuTest1) {
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"    , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice", PINNED);

    // StorageManager::load("inputs/ssbm/date.csv.d_datekey"            , PINNED);
    // StorageManager::load("inputs/ssbm/date.csv.d_year"               , PINNED);

    const char *testLabel = "test1";
    const char *planPath  = "inputs/plans/test1.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuTest2) {
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"    , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice", PINNED);

    // StorageManager::load("inputs/ssbm/date.csv.d_datekey"            , PINNED);
    // StorageManager::load("inputs/ssbm/date.csv.d_year"               , PINNED);

    const char *testLabel = "test2";
    const char *planPath  = "inputs/plans/test2.json";

    runAndVerify(testLabel, planPath);
}
#include <stdint.h>
#include <immintrin.h>

// Uses 64bit pdep / pext to save a step in unpacking.
__m256 compress256(__m256 src, unsigned int mask /* from movmskps */)
{
  uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  // unpack each bit to a byte
  expanded_mask *= 0xFFU;  // mask |= mask<<1 | mask<<2 | ... | mask<<7;
  // ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

  const uint64_t identity_indices = 0x0706050403020100;    // the identity shuffle for vpermps, packed to one index per byte
  uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

  __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
  __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

  return _mm256_permutevar8x32_ps(src, shufmask);
}

// Factor this out so we can get messy with pdep_u64 vs. variable-shift
static inline __m256i unpack_24b_shufmask(unsigned int packed_mask)
{
  uint64_t want_bytes = _pdep_u64(packed_mask, 0x0707070707070707);  // deposit every 3bit index in a separate byte
  __m128i bytevec = _mm_cvtsi64_si128(want_bytes);
  __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

  return shufmask;
}

// // godbolt example for clang uses -fno-unroll-loops only for human-readability
// // For real use, let clang unroll by two
// std::pair<size_t, size_t> filter_non_negative_avx2(int32_t *__restrict__ dst, int32_t *__restrict__ dst1, int32_t *__restrict__ ndst, int32_t *__restrict__ ndst1, const int32_t *__restrict__ src, const int32_t *__restrict__ src1, size_t len) {
//     const int32_t *endp = src+len;
//     int32_t *dst_start  = dst;
//     int32_t *ndst_start = ndst;
//     do {
//       __m256 sv   = _mm256_loadu_ps(reinterpret_cast<const float *>(src));
//       __m256 sv1  = _mm256_loadu_ps(reinterpret_cast<const float *>(src1));
//       __m256 target = _mm256_and_si256(sv, _mm256_set1_epi32(1));
//       __m256 keepvec = _mm256_cmpgt_epi32(target, _mm256_setzero_si256());//, _CMP_GE_OQ);  // true for src > 0.0, false for unordered and src <= 0.0
//       __m256 nkeepvec = _mm256_cmpeq_epi32(target, _mm256_setzero_si256());  // true for src >= 0.0, false for unordered and src < 0.0
//       // note that you can movemask_ps on sv directly (skipping the compare) if you don't mind considering negative-zero and -NaN as negative.
//       unsigned keep = _mm256_movemask_ps(keepvec);
//       __m256 compressed =  compress256(sv, keep);
//       _mm256_storeu_ps(reinterpret_cast<float *>(dst), compressed);
//       __m256 compressed1 =  compress256(sv1, keep);
//       _mm256_storeu_ps(reinterpret_cast<float *>(dst1), compressed1);

//       unsigned nkeep = _mm256_movemask_ps(nkeepvec);
//       __m256 ncompressed =  compress256(sv, nkeep);
//       _mm256_storeu_ps(reinterpret_cast<float *>(ndst), ncompressed);
//       __m256 ncompressed1 =  compress256(sv1, nkeep);
//       _mm256_storeu_ps(reinterpret_cast<float *>(ndst1), ncompressed1);

//       src  += 8;
//       src1 += 8;
//       dst += _mm_popcnt_u32(keep);
//       dst1 += _mm_popcnt_u32(keep);
//       ndst += _mm_popcnt_u32(nkeep);
//       ndst1 += _mm_popcnt_u32(nkeep);
//     } while (src < endp);
//     return make_pair(dst - dst_start, ndst - ndst_start);
// }

std::pair<size_t, size_t> filter_non_negative_avx2(int32_t *__restrict__ dst, int32_t *__restrict__ dst1, int32_t *__restrict__ ndst, int32_t *__restrict__ ndst1, const int32_t *__restrict__ src, const int32_t *__restrict__ src1, size_t len) {
    const int32_t *endp = src+len;
    int32_t *dst_start  = dst;
    int32_t *ndst_start = ndst;
    size_t cnt = 0;

    #pragma clang loop unroll(enable)
    do {
      __m256 sv   = _mm256_loadu_ps(reinterpret_cast<const float *>(src));
      __m256 sv1  = _mm256_loadu_ps(reinterpret_cast<const float *>(src1));
      __m256 target = _mm256_and_si256(sv, _mm256_set1_epi32(1));
      __m256 keepvec = _mm256_cmpgt_epi32(target, _mm256_setzero_si256());//, _CMP_GE_OQ);  // true for src > 0.0, false for unordered and src <= 0.0
      __m256 nkeepvec = _mm256_cmpeq_epi32(target, _mm256_setzero_si256());  // true for src >= 0.0, false for unordered and src < 0.0
      // note that you can movemask_ps on sv directly (skipping the compare) if you don't mind considering negative-zero and -NaN as negative.
      unsigned keep = _mm256_movemask_ps(keepvec);
      __m256 compressed =  compress256(sv, keep);
      _mm256_storeu_ps(reinterpret_cast<float *>(dst), compressed);
      __m256 compressed1 =  compress256(sv1, keep);
      _mm256_storeu_ps(reinterpret_cast<float *>(dst1), compressed1);

      unsigned nkeep = _mm256_movemask_ps(nkeepvec);
      __m256 ncompressed =  compress256(sv, nkeep);
      _mm256_storeu_ps(reinterpret_cast<float *>(ndst), ncompressed);
      __m256 ncompressed1 =  compress256(sv1, nkeep);
      _mm256_storeu_ps(reinterpret_cast<float *>(ndst1), ncompressed1);

      src  += 8;
      src1 += 8;
      cnt += _mm_popcnt_u32(keep);
      dst += _mm_popcnt_u32(keep);
      dst1 += _mm_popcnt_u32(keep);
      ndst += _mm_popcnt_u32(nkeep);
      ndst1 += _mm_popcnt_u32(nkeep);
    } while (src < endp);
    return make_pair(cnt, len - cnt);
}

std::pair<size_t, size_t> func(int32_t *__restrict__ dst, int32_t *__restrict__ dst1, int32_t *__restrict__ ndst, int32_t *__restrict__ ndst1, const int32_t *__restrict__ src, const int32_t *__restrict__ src1, size_t len) {
    size_t j = 0;
    size_t nj = 0;
    for (size_t i = 0 ; i < len ; ++i){
        if (src[i] & 1) {
            dst  [j   ] = src [i];
            dst1 [j++ ] = src1[i];
        } else {
            ndst [nj  ] = src [i];
            ndst1[nj++] = src1[i];
        }
    }
    return make_pair(j, nj);
}
// #ifdef __AVX512F__     // build with -mavx512f
// size_t filter_non_negative_avx512(float *__restrict__ dst, const float *__restrict__ src, size_t len) {
//     const float *endp = src+len;
//     float *dst_start = dst;
//     do {
//         __m512      sv  = _mm512_loadu_ps(src);
//         __mmask16 keep = _mm512_cmp_ps_mask(sv, _mm512_setzero_ps(), _CMP_GE_OQ);  // true for src >= 0.0, false for unordered and src < 0.0
//         _mm512_mask_compressstoreu_ps(dst, keep, sv);   // clang is missing this intrinsic, which can't be emulated with a separate store

//         src += 16;
//         dst += _mm_popcnt_u32(keep);
//     } while (src < endp);
//     return dst - dst_start;
// }
// #endif
TEST_F(MultiGPUTest, gpuTest3) {
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"     , PINNED);
    // StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"    , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice", PINNED);

    // // StorageManager::load("inputs/ssbm/date.csv.d_datekey"            , PINNED);
    // // StorageManager::load("inputs/ssbm/date.csv.d_year"               , PINNED);

    // const char *testLabel = "test2";
    // const char *planPath  = "inputs/plans/test2.json";

    // runAndVerify(testLabel, planPath);


    auto t  = StorageManager::getFile("inputs/ssbm/lineorder.csv.lo_discount"     );
    auto t1 = StorageManager::getFile("inputs/ssbm/lineorder.csv.lo_extendedprice");
    int32_t * src  = (int32_t *) t[0].data;
    int32_t * src1 = (int32_t *) t1[0].data;
    int32_t * dst  = (int32_t *) RawMemoryManager::mallocPinned(t[0].size);
    int32_t * dst1 = (int32_t *) RawMemoryManager::mallocPinned(t1[0].size);
    int32_t * ndst  = (int32_t *) RawMemoryManager::mallocPinned(t[0].size);
    int32_t * ndst1 = (int32_t *) RawMemoryManager::mallocPinned(t1[0].size);

    size_t size = t[0].size/sizeof(int32_t);
    std::cout << "Bytes: " << t[0].size << " tuples: " << size << std::endl;
    std::cout << "Bytes: " << t1[0].size << " tuples: " << t1[0].size/sizeof(int32_t) << std::endl;
    for (size_t i = 0 ; i < size ; ++i) dst[i]  = ndst[i]  = rand(); //touch to force allocation
    for (size_t i = 0 ; i < size ; ++i) dst1[i] = ndst1[i] = rand(); //touch to force allocation

    int32_t s = 0;
    {
        time_block t("T: ");
        for (size_t i = 0 ; i < size ; ++i) s += src [i] * src1[i];
    }
    std::cout << s << std::endl;

    pair<size_t, size_t> len;
    s = 0;
    {
        time_block t("Tc: ");
        {
            time_block t("T: ");
            len = filter_non_negative_avx2(dst, dst1, ndst, ndst1, src, src1, size);
        }

        for (size_t i = 0 ; i < len.first  ; ++i) s += dst [i] * dst1[i];
        for (size_t i = 0 ; i < len.second ; ++i) s += ndst [i] * ndst1[i];
    }
    std::cout << s << std::endl;
    std::cout << len.first << " " << len.second << std::endl;

    s = 0;
    for (size_t i = 0 ; i < len.first  ; ++i) s += dst [i];
    for (size_t i = 0 ; i < len.first  ; ++i) s += dst1[i];
    for (size_t i = 0 ; i < len.second ; ++i) s -= ndst [i];
    for (size_t i = 0 ; i < len.second ; ++i) s -= ndst1[i];

    std::cout << s << std::endl;
    
    s = 0;
    {
        time_block t("Tc: ");
        {
            time_block t("T: ");
            len = func(dst, dst1, ndst, ndst1, src, src1, size);
        }

        for (size_t i = 0 ; i < len.first  ; ++i) s += dst [i] * dst1[i];
        for (size_t i = 0 ; i < len.second ; ++i) s += ndst [i] * ndst1[i];
    }
    std::cout << s << std::endl;
    std::cout << len.first << " " << len.second << std::endl;
    s = 0;
    for (size_t i = 0 ; i < len.first  ; ++i) s += dst [i];
    for (size_t i = 0 ; i < len.first  ; ++i) s += dst1[i];
    for (size_t i = 0 ; i < len.second ; ++i) s -= ndst [i];
    for (size_t i = 0 ; i < len.second ; ++i) s -= ndst1[i];

    std::cout << s << std::endl;

    // RawMemoryManager::freePinned(src);
    RawMemoryManager::freePinned(dst);
    RawMemoryManager::freePinned(ndst);
    // RawMemoryManager::freePinned(src1);
    RawMemoryManager::freePinned(dst1);
    RawMemoryManager::freePinned(ndst1);
}

TEST_F(MultiGPUTest, gpuDriverParallelOnGpu) {
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );

    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_datekey"             );
    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_year"                );

    const char *testLabel = "gpuSSBM_Q1_1_parallel_hash_on_gpu";
    const char *planPath  = "inputs/plans/ssbm_q1_1_parallel_gpu.json";
 
    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuDriverParallelOnGpuEarlyFilter) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }
    
    gpu_run(cudaSetDevice(0));

    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );

    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_datekey"             );
    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_year"                );

    __itt_resume();
    const char *testLabel = "gpuSSBM_Q1_1_parallel_hash_on_gpu_early_filter";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_parallel_gpu_earlyfilter.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (RawPipeline * p: pipelines) {
        nvtxRangePushA("pip");
        {
            time_block t("T: ");
            p->open();
            p->consume(0);
            p->close();
        }
        nvtxRangePop();
    }
    __itt_pause();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result
    // SF:100 => 2087240435 (32bits)

    StorageManager::unloadAll();
}

TEST_F(MultiGPUTest, gpuDriverParallelOnGpuFull) {
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"    );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice");

    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_datekey"            );
    StorageManager::loadToGpus("inputs/ssbm/date.csv.d_year"               );

    const char *testLabel = "gpuSSBM_Q1_1_parallel_hash_on_gpu_full";
    const char *planPath  = "inputs/plans/ssbm_q1_1_parallel_gpu_full.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q1_1_SF100_gpuResident) {
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_discount"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_quantity"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_orderdate"    );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_extendedprice");

    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_datekey"            );
    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_year"               );

    const char *testLabel = "gpuSSBM_Q1_1_SF100_gpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_1_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q1_1_SF100_cpuResident) {
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_discount"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_quantity"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_orderdate"    , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_extendedprice", 0, 1);

    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_datekey"            , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_year"               , 0, 1);

    const char *testLabel = "gpuSSBM_Q1_1_SF100_cpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_1_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q1_2_SF100_gpuResident) {
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_discount"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_quantity"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_orderdate"    );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_extendedprice");

    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_datekey"            );
    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_yearmonthnum"       );

    const char *testLabel = "gpuSSBM_Q1_2_SF100_gpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_2_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q1_2_SF100_cpuResident) {
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_discount"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_quantity"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_orderdate"    , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_extendedprice", 0, 1);

    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_datekey"            , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_yearmonthnum"       , 0, 1);

    const char *testLabel = "gpuSSBM_Q1_2_SF100_cpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_2_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q2_1_bare_SF100_gpuResident) {
    StorageManager::load("inputs/ssbm100/date.csv.d_datekey"                    , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/date.csv.d_year"                       , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/customer.csv.c_custkey"                , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/customer.csv.c_region"                 , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/supplier.csv.s_suppkey"                , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/supplier.csv.s_region"                 , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/part.csv.p_partkey"                    , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/part.csv.p_category"                   , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/part.csv.p_size"                       , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_revenue"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_orderdate"            , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_partkey"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_suppkey"              , PINNED);//GPU_RESIDENT);

    const char *testLabel = "gpuSSBM_Q2_1_bare_SF100_gpuResident";
    const char *planPath  = "inputs/plans/ssbm_q2_1_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q4_3_bare_SF100_gpuResident) {
    StorageManager::load("inputs/ssbm100/date.csv.d_datekey"                    , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/date.csv.d_year"                       , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/customer.csv.c_custkey"                , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/customer.csv.c_region"                 , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/supplier.csv.s_suppkey"                , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/supplier.csv.s_city"                   , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/supplier.csv.s_nation"                 , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/part.csv.p_partkey"                    , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/part.csv.p_category"                   , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/part.csv.p_brand1"                     , PINNED);//GPU_RESIDENT);

    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_custkey"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_partkey"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_suppkey"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_orderdate"            , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_revenue"              , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_supplycost"           , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_quantity"             , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_extendedprice"        , PINNED);//GPU_RESIDENT);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_discount"             , PINNED);//GPU_RESIDENT);

    const char *testLabel = "gpuSSBM_Q4_3_bare_SF100_gpuResident";
    const char *planPath  = "inputs/plans/ssbm_q4_3_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuPlanGen) {
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_discount"      , PINNED);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_quantity"      , PINNED);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_orderdate"     , PINNED);
    StorageManager::load("inputs/ssbm100/lineorder.csv.lo_extendedprice" , PINNED);

    StorageManager::load("inputs/ssbm100/date.csv.d_datekey"             , PINNED);
    StorageManager::load("inputs/ssbm100/date.csv.d_year"                , PINNED);

    const char *testLabel = "gpuPlanGen";
    const char *planPath  = "inputs/plans/plan_generation_q1_1.json";
 
    runAndVerify(testLabel, planPath);
}


TEST_F(MultiGPUTest, gpuSSBM_Q1_3_SF100_gpuResident) {
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_discount"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_quantity"     );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_orderdate"    );
    StorageManager::loadToGpus("inputs/ssbm100/lineorder.csv.lo_extendedprice");

    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_datekey"            );
    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_year"               );
    StorageManager::loadToGpus("inputs/ssbm100/date.csv.d_weeknuminyear"      );

    const char *testLabel = "gpuSSBM_Q1_3_SF100_gpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_3_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuSSBM_Q1_3_SF100_cpuResident) {
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_discount"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_quantity"     , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_orderdate"    , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/lineorder.csv.lo_extendedprice", 0, 1);

    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_datekey"            , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_year"               , 0, 1);
    StorageManager::loadEverywhere("inputs/ssbm100/date.csv.d_weeknuminyear"      , 0, 1);

    const char *testLabel = "gpuSSBM_Q1_3_SF100_cpuResident";
    const char *planPath  = "inputs/plans/ssbm_q1_3_gpu_sf100.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, gpuPingPong) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }
    
    gpu_run(cudaSetDevice(0));

    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );

    __itt_resume();
    const char *testLabel = "reduceScanPingPongMultigpu";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-ping-pong-multigpu.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (RawPipeline * p: pipelines) {
        nvtxRangePushA("pip");
        {
            time_block t("T: ");
            p->open();
            p->consume(0);
            p->close();
        }
        nvtxRangePop();
    }
    __itt_pause();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();
}

TEST_F(MultiGPUTest, gpuPingHashRearrangePong) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }
    
    gpu_run(cudaSetDevice(0));

    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );

    __itt_resume();
    const char *testLabel = "reduceScanPingHashRearrangePongMultigpu";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-ping-rearrange-pong-multigpu.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (RawPipeline * p: pipelines) {
        nvtxRangePushA("pip");
        {
            time_block t("T: ");
            p->open();
            p->consume(0);
            p->close();
        }
        nvtxRangePop();
    }
    __itt_pause();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();

}

TEST_F(MultiGPUTest, gpuStorageManager) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }
    
    gpu_run(cudaSetDevice(0));

    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );

    __itt_resume();
    const char *testLabel = "gpuStorageManager";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-gpu-storage-manager.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (RawPipeline * p: pipelines) {
        nvtxRangePushA("pip");
        {
            time_block t("T: ");
            p->open();
            p->consume(0);
            p->close();
        }
        nvtxRangePop();
    }
    __itt_pause();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();

}

TEST_F(MultiGPUTest, cpuSequential) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice" , PINNED);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"             , PINNED);
    StorageManager::load("inputs/ssbm/date.csv.d_year"                , PINNED);
    
    gpu_run(cudaSetDevice(0));

    __itt_resume();
    const char *testLabel = "cpuSequential";
    // GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_cpu.json";

    // std::vector<RawPipeline *> pipelines;
    {
        time_block t("T: ");

        // ctx                   = new RawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel);

        // return verifyTestResult(TEST_OUTPUTS "/tests-cpu-sequential/", testLabel);
        
        // ctx->compileAndLoad();

        // pipelines = ctx->getPipelines();
    }

    EXPECT_TRUE(verifyTestResult(TEST_OUTPUTS "/tests-multigpu-integration/", testLabel));

    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();
}
TEST_F(MultiGPUTest, cpuParallel) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice" , PINNED);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"             , PINNED);
    StorageManager::load("inputs/ssbm/date.csv.d_year"                , PINNED);
    
    gpu_run(cudaSetDevice(0));

    __itt_resume();
    const char *testLabel = "cpuParallel";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_multicore2.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("T: ");

        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);

        // return verifyTestResult(TEST_OUTPUTS "/tests-cpu-sequential/", testLabel);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    // EXPECT_TRUE(verifyTestResult(TEST_OUTPUTS "/tests-multigpu-integration/", testLabel));

    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();
}


TEST_F(MultiGPUTest, cpuScanReduce) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"      , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"     , PINNED);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice" , PINNED);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"             , PINNED);
    StorageManager::load("inputs/ssbm/date.csv.d_year"                , PINNED);
    
    gpu_run(cudaSetDevice(0));

    __itt_resume();
    const char *testLabel = "cpuScanReduce";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-cpu-storage-manager.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel, false);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (RawPipeline * p: pipelines) {
        nvtxRangePushA("pip");
        {
            time_block t("T: ");
            p->open();
            p->consume(0);
            p->close();
        }
        nvtxRangePop();
    }

    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }


    EXPECT_TRUE(verifyTestResult(TEST_OUTPUTS "/tests-multigpu-integration/", testLabel));
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();
}

TEST_F(MultiGPUTest, multicpuScanReduce) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_discount"     );
    StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_quantity"     );
    StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_orderdate"    );
    StorageManager::loadToCpus("inputs/ssbm/lineorder.csv.lo_extendedprice");

    StorageManager::loadToCpus("inputs/ssbm/date.csv.d_datekey"            );
    StorageManager::loadToCpus("inputs/ssbm/date.csv.d_year"               );
    
    gpu_run(cudaSetDevice(0));

    // __itt_resume();
    const char *testLabel = "multicpuScanReduce";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-multicpu-storage-manager.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel, false);
        CatalogParser catalog = CatalogParser(catalogJSON, ctx);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, testLabel, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    // for (size_t i = 0 ; i < 100; ++i){
        __itt_resume();
        for (RawPipeline * p: pipelines) {
            nvtxRangePushA("pip");
            {
                time_block t("T: ");
                p->open();
                p->consume(0);
                p->close();
            }
            nvtxRangePop();
        }
        __itt_pause();
    // }
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStop());
    }


    EXPECT_TRUE(verifyTestResult(TEST_OUTPUTS "/tests-multigpu-integration/", testLabel));
    // int32_t c_out;
    // gpu_run(cudaMemcpy(&c_out, aggr, sizeof(int32_t), cudaMemcpyDefault));
    // //for the current dataset, regenerating it may change the results
    // EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));
    // EXPECT_TRUE(0 && "How do I get the result now ?"); //FIXME: now it becomes too complex to get the result

    StorageManager::unloadAll();
}


TEST_F(MultiGPUTest, multigpuScanReduceWithTransfer) {
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_discount"      , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_quantity"      , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_orderdate"     , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_extendedprice" , 1, 0);

    const char *testLabel = "multicpuScanReduceWithTransfer";
    const char *planPath  = "inputs/plans/reduce-scan-multigpu.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, multicpuScanReduceWithTransfer) {
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_discount"      , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_quantity"      , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_orderdate"     , 1, 0);
    StorageManager::loadEverywhere("inputs/ssbm/lineorder.csv.lo_extendedprice" , 1, 0);

    const char *testLabel = "multicpuScanReduceWithTransfer";
    const char *planPath  = "inputs/plans/reduce-scan-multicpu-storage-manager-w-mem-copy.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q1_1_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_quantity");
    load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
    load("inputs/ssbm100/lineorder.csv.lo_discount");

    const char *testLabel = "q1_1_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q1_1.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q1_2_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_yearmonthnum");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_quantity");
    load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
    load("inputs/ssbm100/lineorder.csv.lo_discount");

    const char *testLabel = "q1_2_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q1_2.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q1_3_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_quantity");
    load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
    load("inputs/ssbm100/lineorder.csv.lo_discount");
    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/date.csv.d_weeknuminyear");

    const char *testLabel = "q1_3_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q1_3.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q2_1_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_category");
    load("inputs/ssbm100/part.csv.p_size");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q2_1_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q2_1.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q2_2_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_brand1");
    load("inputs/ssbm100/part.csv.p_size");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q2_2_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q2_2.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q2_3_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_brand1");
    load("inputs/ssbm100/part.csv.p_size");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q2_3_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q2_3.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q3_1_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_region");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q3_1_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q3_1.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q3_2_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_nation");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_nation");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q3_2_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q3_2.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q3_3_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_city");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_city");
    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q3_3_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q3_3.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q3_4_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_city");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_city");
    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/date.csv.d_yearmonth");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");

    const char *testLabel = "q3_4_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q3_4.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q4_1_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_mfgr");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_region");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");
    load("inputs/ssbm100/lineorder.csv.lo_supplycost");

    const char *testLabel = "q4_1_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q4_1.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q4_2_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_mfgr");
    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_region");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_region");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");
    load("inputs/ssbm100/lineorder.csv.lo_supplycost");

    const char *testLabel = "q4_2_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q4_2.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q4_3_sql_bare) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");
    load("inputs/ssbm100/supplier.csv.s_suppkey");
    load("inputs/ssbm100/supplier.csv.s_nation");
    load("inputs/ssbm100/customer.csv.c_custkey");
    load("inputs/ssbm100/customer.csv.c_region");
    load("inputs/ssbm100/part.csv.p_partkey");
    load("inputs/ssbm100/part.csv.p_category");
    load("inputs/ssbm100/lineorder.csv.lo_custkey");
    load("inputs/ssbm100/lineorder.csv.lo_partkey");
    load("inputs/ssbm100/lineorder.csv.lo_suppkey");
    load("inputs/ssbm100/lineorder.csv.lo_orderdate");
    load("inputs/ssbm100/lineorder.csv.lo_revenue");
    load("inputs/ssbm100/lineorder.csv.lo_supplycost");

    const char *testLabel = "q4_3_sql_bare";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q4_3.sql_bare.json";

    runAndVerify(testLabel, planPath);
}

TEST_F(MultiGPUTest, q2_1_sql_bare_w_groupby) {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };

    load("inputs/ssbm100/date.csv.d_datekey");
    load("inputs/ssbm100/date.csv.d_year");

    const char *testLabel = "q2_1_sql_bare_w_groupby";
    const char *planPath  = "inputs/plans/proteus_bare_plans/q2_1.sql_bare_w_groupby.json";

    runAndVerify(testLabel, planPath);
}