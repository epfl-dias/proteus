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

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "nvToolsExt.h"
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

// #include "common/gpu/gpu-common.hpp"
#include "common/common.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "util/raw-functions.hpp"
#include "util/raw-pipeline.hpp"
#include "plan/plan-parser.hpp"
#include "multigpu/buffer_manager.cuh"
#include "storage/raw-storage-manager.hpp"
#include "multigpu/numa_utils.cuh"
#include <cuda_profiler_api.h>

#include <vector>
#include <thread>

class MultiGPUTest : public ::testing::Test {
protected:
    virtual void SetUp();
    virtual void TearDown();

    void launch(void ** args, dim3 gridDim, dim3 blockDim);
    void launch(void ** args, dim3 gridDim);
    void launch(void ** args);

    bool flushResults = true;
    const char * testPath = TEST_OUTPUTS "/tests-output/";

    const char * catalogJSON = "inputs/plans/catalog.json";
public:
    CUdevice  *device ;
    CUcontext *context;
};

void thread_warm_up(){}

void MultiGPUTest::SetUp   (){
    setbuf(stdout, NULL);

    // int devCount;

    // gpu_run(cuInit(0));
    // gpu_run(cuDeviceGetCount(&devCount));

    // device  = new CUdevice [devCount];
    // context = new CUcontext[devCount];

    // for (int i = 0 ; i < devCount ; ++i){
    //     gpu_run(cuDeviceGet(device  + i, i));
    //     gpu_run(cuCtxCreate(context + i, 0, device[i]));
    // }

    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaFree(0));
    }
    
    // gpu_run(cudaSetDevice(0));

    gpu(cudaFree(0));

    // gpu(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

    // std::vector<std::thread> thrds;
    // for (int i = 0 ; i < 20 ; ++i) thrds.emplace_back(thread_warm_up);
    // for (auto &t: thrds) t.join();

    // srand(time(0));

    buffer_manager<int32_t>::init(100);
}

void MultiGPUTest::TearDown(){
    buffer_manager<int32_t>::destroy();
}

TEST_F(MultiGPUTest, gpuDriverSequential) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    gpu_run(cudaSetDevice(0));

    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );//, GPU_RESIDENT);

    // StorageManager::loadToGpus("inputs/ssbm/date.csv.d_datekey"             );//, GPU_RESIDENT);
    // StorageManager::loadToGpus("inputs/ssbm/date.csv.d_year"                );//, GPU_RESIDENT);

    StorageManager::load("inputs/ssbm/lineorder.csv.lo_discount"      , GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_quantity"      , GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_orderdate"     , GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/lineorder.csv.lo_extendedprice" , GPU_RESIDENT);

    StorageManager::load("inputs/ssbm/date.csv.d_datekey"             , GPU_RESIDENT);
    StorageManager::load("inputs/ssbm/date.csv.d_year"                , GPU_RESIDENT);

    gpu_run(cudaSetDevice(0));
    
    const char *testLabel = "gpuSSBM_Q1_1c";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1c.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaDeviceSynchronize());
    }

    {
    nvtxRangePushA("gen (hj build)");
    {
        time_block t("Tgen (hj build): ");
        pipelines[0]->open();
        pipelines[0]->consume(0);
        pipelines[0]->close();
    }
    nvtxRangePop();
    }

    // int32_t * aggr;
    {
    nvtxRangePushA("Tgen (hj probe)");
    {
        time_block t("Tgen (hj probe): ");
        pipelines[1]->open();
        pipelines[1]->consume(0);
        // aggr = pipelines[1]->getStateVar<int32_t **>(0)[0];
        pipelines[1]->close();
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

TEST_F(MultiGPUTest, gpuDriverMultiReduce) {
    int devices = get_num_of_gpus();
    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaProfilerStart());
    }

    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_discount"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_quantity"      );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_orderdate"     );
    StorageManager::loadToGpus("inputs/ssbm/lineorder.csv.lo_extendedprice" );
    
    gpu_run(cudaSetDevice(0));

    __itt_resume();
    const char *testLabel = "gpuDriverMultiReduce";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-multigpu.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    {
        time_block t("T: ");
        pipelines[0]->open();
        pipelines[0]->consume(0);
        pipelines[0]->close();
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

TEST_F(MultiGPUTest, gpuDriverParallel) {
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
    const char *testLabel = "gpuSSBM_Q1_1_parallel";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_parallel.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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

TEST_F(MultiGPUTest, gpuDriverParallelOnGpu) {
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
    const char *testLabel = "gpuSSBM_Q1_1_parallel_hash_on_gpu";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_parallel_gpu.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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

TEST_F(MultiGPUTest, gpuDriverParallelOnGpuFull) {
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
    const char *testLabel = "gpuSSBM_Q1_1_parallel_hash_on_gpu_full";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1_parallel_gpu_full.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
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
