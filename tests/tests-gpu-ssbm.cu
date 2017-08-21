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
#include "common/gpu/gpu-common.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "plugins/gpu-col-scan-plugin.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"

#include "common/common.hpp"
#include "util/raw-context.hpp"
#include "util/raw-functions.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"

#include <unordered_map>

#include "plan/plan-parser.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
// #include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <ctime>

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

class GpuSSBMTest3 : public ::testing::Test {
protected:
    virtual void SetUp();
    virtual void TearDown();

    bool flushResults = true;
    const char * testPath = TEST_OUTPUTS "/tests-output/";

    const char * catalogJSON = "inputs/plans/catalog.json";
    
private:
    RawCatalog * catalog;
    CachingService * caches;

    size_t    N;

public:
    int32_t * c;
    int32_t * h_c;
};

void GpuSSBMTest3::SetUp() {
    catalog = &RawCatalog::getInstance();
    caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();

    N = 1024*1024*256;
    
    gpu_run(cudaMalloc(&c, sizeof(int32_t)*16));
    gpu_run(cudaMallocHost(&h_c, sizeof(int32_t)*16));

    srand(time(NULL));
}

void GpuSSBMTest3::TearDown() {
    // gpu_run(cudaFree(a));
    // gpu_run(cudaFree(b));
    gpu_run(cudaFree(c));
    // gpu_run(cudaFree(d));

    // gpu_run(cudaFreeHost(h_a));
    // gpu_run(cudaFreeHost(h_b));
    gpu_run(cudaFreeHost(h_c));
    // gpu_run(cudaFreeHost(h_d));
    // gpu_run(cudaFreeHost(h_e));
}


struct mmap_file{
private:
    int    fd;

public:
    size_t filesize;
    void *     data;
    void * gpu_data;

    mmap_file(std::string name){
        filesize = getFileSize(name.c_str());
        fd       = open(name.c_str(), O_RDONLY, 0);

        //Execute mmap
        data     = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        assert(data != MAP_FAILED);
        
        gpu_run(cudaMalloc(&gpu_data,       filesize));
        gpu_run(cudaMemcpy( gpu_data, data, filesize, cudaMemcpyDefault));
    }

    ~mmap_file(){
        munmap(data, filesize);
        close (fd  );

        gpu_run(cudaFree(gpu_data));
    }
};

template<typename T>
struct mmap_file_splitable{
private:
    int    fd;

public:
    size_t filesize;
    T    * data    ;
    T    * gpu_data;
    size_t index   ;
    size_t step    ;

    mmap_file_splitable(std::string name, size_t step): index(0), step(step){
        filesize = getFileSize(name.c_str());
        fd       = open(name.c_str(), O_RDONLY, 0);

        //Execute mmap
        data     = (T *) mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        assert(data != MAP_FAILED);

        gpu_run(cudaMalloc(&gpu_data,       step * sizeof(T)                   ));
    }

    size_t next(){
        size_t l_step = min(step, remaining());
        // std::cout << index << "->";
        gpu_run(cudaMemcpy( gpu_data, data + index, l_step * sizeof(T), cudaMemcpyDefault));
        index += l_step;

        // std::cout << index << std::endl;

        return l_step;
    }

    size_t remaining(){
        return (filesize - index * sizeof(T))/sizeof(T);
    }

    ~mmap_file_splitable(){
        munmap(data, filesize);
        close (fd  );

        gpu_run(cudaFree(gpu_data));
    }
};


//FIXME: make configuration dynamic
constexpr uint32_t log_parts1 = 0;//9;
constexpr uint32_t log_parts2 = 0;//6;//8;

constexpr int32_t g_d        = log_parts1 + log_parts2;
constexpr int32_t p_d        = 5;

constexpr int32_t max_chain  = (32 - 1) * 2 - 1;

#define hj_d (5 + p_d + g_d)

constexpr uint32_t hj_mask = ((1 << hj_d) - 1);

constexpr int32_t partitions = 1 << p_d;
constexpr int32_t partitions_mask = partitions - 1;

constexpr int32_t grid_parts = 1 << g_d;
constexpr int32_t grid_parts_mask = grid_parts - 1;

struct alignas(alignof(int64_t)) hj_bucket{
    int32_t next;
    int32_t val ;

    constexpr __host__ __device__ hj_bucket(int32_t next, int32_t value): next(next), val(value){}
};

TEST_F(GpuSSBMTest3, gpuSSBM_Q1_1b) {
    const char *testLabel = "gpuSSBM_Q1_1b";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1b.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    gpu_run(cudaMemset(c, 0, sizeof(int32_t) * 16));

    mmap_file f_lo_discount     ("inputs/ssbm/lo_discount.bin"     );
    mmap_file f_lo_quantity     ("inputs/ssbm/lo_quantity.bin"     );
    mmap_file f_lo_orderdate    ("inputs/ssbm/lo_orderdate.bin"    );
    mmap_file f_lo_extendedprice("inputs/ssbm/lo_extendedprice.bin");

    mmap_file f_d_datekey       ("inputs/ssbm/d_datekey.bin"       );
    mmap_file f_d_year          ("inputs/ssbm/d_year.bin"          );

    size_t d_N  = (f_d_datekey  .filesize)/sizeof(int32_t);
    size_t lo_N = (f_lo_quantity.filesize)/sizeof(int32_t);

    {
        time_block t("Topen0: ");
        pipelines[0]->open();
    }
    {
        time_block t("Tgenerated (hj build): ");

        pipelines[0]->consume(d_N, f_d_datekey.gpu_data, f_d_year.gpu_data);
    }
    {
        time_block t("Tclose0: ");
        pipelines[0]->close();
    }

    {
        time_block t("Topen1: ");
        pipelines[1]->open();
    }
    {
        time_block t("Tgenerated (hj probe): ");

        // Kernel launch
        pipelines[1]->consume(lo_N, 
                                c, 
                                f_lo_discount.gpu_data, 
                                f_lo_quantity.gpu_data, 
                                f_lo_orderdate.gpu_data, 
                                f_lo_extendedprice.gpu_data);
    }
    {
        time_block t("Tclose1: ");
        pipelines[1]->close();
    }

    size_t c_out;
    // gpu_run(cudaMemcpy(&c_out, c3, sizeof(size_t), cudaMemcpyDefault));

    //this result is constant for SSBM Q1.1
    // EXPECT_EQ(365, c_out);

    gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));

    //for the current dataset, regenerating it may change the results
    EXPECT_TRUE(c_out == UINT64_C(4472807765583) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(4472807765583)));

    // gpu_run(cudaFree(first         ));
    // gpu_run(cudaFree(next_w_values ));

}

TEST_F(GpuSSBMTest3, gpuSSBM_Q1_2) {
    const char *testLabel = "gpuSSBM_Q1_2";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_2.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    gpu_run(cudaMemset(c, 0, sizeof(int32_t) * 16));

    mmap_file f_lo_discount     ("inputs/ssbm/lo_discount.bin"     );
    mmap_file f_lo_quantity     ("inputs/ssbm/lo_quantity.bin"     );
    mmap_file f_lo_orderdate    ("inputs/ssbm/lo_orderdate.bin"    );
    mmap_file f_lo_extendedprice("inputs/ssbm/lo_extendedprice.bin");

    mmap_file f_d_datekey       ("inputs/ssbm/d_datekey.bin"       );
    mmap_file f_d_yearmonthnum  ("inputs/ssbm/d_yearmonthnum.bin"  );

    size_t d_N  = (f_d_datekey.filesize)/sizeof(int32_t);
    size_t lo_N = (f_lo_quantity.filesize)/sizeof(int32_t);

    {
        time_block t("Topen0: ");
        pipelines[0]->open();
    }
    {
        time_block t("Tgenerated (hj build): ");

        pipelines[0]->consume(d_N, f_d_datekey.gpu_data, f_d_yearmonthnum.gpu_data);
    }
    {
        time_block t("Tclose0: ");
        pipelines[0]->close();
    }


    {
        time_block t("Topen1: ");
        pipelines[1]->open();
    }
    {
        time_block t("Tgenerated (hj probe): ");

        // Kernel launch
        pipelines[1]->consume(lo_N, 
                                c, 
                                f_lo_discount.gpu_data, 
                                f_lo_quantity.gpu_data, 
                                f_lo_orderdate.gpu_data, 
                                f_lo_extendedprice.gpu_data);
    }
    {
        time_block t("Tclose1: ");
        pipelines[1]->close();
    }


    size_t c_out;
    // gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));
    // std::cout << c_out << std::endl;

    // //this results is constant for SSBM Q1.1
    // EXPECT_EQ(c_out, 31);

    gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));
    std::cout << c_out << std::endl;

    //for the current dataset, regenerating it may change the results
    EXPECT_TRUE(c_out == UINT64_C(965049065847) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(965049065847)));

}


TEST_F(GpuSSBMTest3, gpuSSBM_Q1_3) {

    const char *testLabel = "gpuSSBM_Q1_3";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_3.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    gpu_run(cudaMemset(c, 0, sizeof(int32_t) * 16));

    mmap_file f_lo_discount     ("inputs/ssbm/lo_discount.bin"     );
    mmap_file f_lo_quantity     ("inputs/ssbm/lo_quantity.bin"     );
    mmap_file f_lo_orderdate    ("inputs/ssbm/lo_orderdate.bin"    );
    mmap_file f_lo_extendedprice("inputs/ssbm/lo_extendedprice.bin");

    mmap_file f_d_datekey       ("inputs/ssbm/d_datekey.bin"       );
    mmap_file f_d_weeknuminyear ("inputs/ssbm/d_weeknuminyear.bin" );
    mmap_file f_d_year          ("inputs/ssbm/d_year.bin"          );

    size_t d_N  = (f_d_datekey.filesize)/sizeof(int32_t);
    size_t lo_N = (f_lo_quantity.filesize)/sizeof(int32_t);

    {
        time_block t("Topen0: ");
        pipelines[0]->open();
    }
    {
        time_block t("Tgenerated (hj build): ");

        pipelines[0]->consume(d_N, f_d_datekey.gpu_data, f_d_weeknuminyear.gpu_data, f_d_year.gpu_data);
    }
    {
        time_block t("Tclose0: ");
        pipelines[0]->close();
    }


    {
        time_block t("Topen1: ");
        pipelines[1]->open();
    }
    {
        time_block t("Tgenerated (hj probe): ");

        // Kernel launch
        pipelines[1]->consume(lo_N, 
                                c, 
                                f_lo_discount.gpu_data, 
                                f_lo_quantity.gpu_data, 
                                f_lo_orderdate.gpu_data, 
                                f_lo_extendedprice.gpu_data);
    }
    {
        time_block t("Tclose1: ");
        pipelines[1]->close();
    }

    size_t c_out;
    // gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));
    // std::cout << c_out << std::endl;

    //this results is constant for SSBM Q1.1
    // EXPECT_EQ(c_out, 31);

    gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));
    std::cout << c_out << std::endl;

    //for the current dataset, regenerating it may change the results
    EXPECT_TRUE(c_out == UINT64_C(261356323969) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(261356323969)));
}


TEST_F(GpuSSBMTest3, gpuSSBM_Q1_1_100) {

    const char *testLabel = "gpuSSBM_Q1_1_100";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q1_1b.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }

    gpu_run(cudaMemset(c, 0, sizeof(int32_t) * 16));

    size_t step_size = 1024*1024*16;

    mmap_file_splitable<int32_t> f_lo_discount     ("inputs/ssbm100/lo_discount.bin"     , step_size);
    mmap_file_splitable<int32_t> f_lo_quantity     ("inputs/ssbm100/lo_quantity.bin"     , step_size);
    mmap_file_splitable<int32_t> f_lo_orderdate    ("inputs/ssbm100/lo_orderdate.bin"    , step_size);
    mmap_file_splitable<int32_t> f_lo_extendedprice("inputs/ssbm100/lo_extendedprice.bin", step_size);

    mmap_file f_d_datekey       ("inputs/ssbm/d_datekey.bin"          );
    mmap_file f_d_year          ("inputs/ssbm/d_year.bin"             );

    size_t d_N  = (f_d_datekey.filesize)/sizeof(int32_t);
    size_t lo_N = (f_lo_quantity.filesize)/sizeof(int32_t);

    {
        time_block t("Topen0: ");
        pipelines[0]->open();
    }
    {
        time_block t("Tgenerated (hj build): ");

        pipelines[0]->consume(d_N, f_d_datekey.gpu_data, f_d_year.gpu_data);
    }
    {
        time_block t("Tclose0: ");
        pipelines[0]->close();
    }

    {
        time_block t("Topen1: ");
        pipelines[1]->open();
    }

    {
        time_block t("Tgenerated (hj probe): ");
        size_t remaining;
        while((remaining = f_lo_discount.remaining()) > 0) {
            size_t l_step = f_lo_discount.next();
            f_lo_quantity.next();
            f_lo_orderdate.next();
            f_lo_extendedprice.next();

            {
                // time_block t("Tgenerated (hj probe, step): ");

                // Kernel launch
                pipelines[1]->consume(l_step, 
                                        c, 
                                        f_lo_discount.gpu_data, 
                                        f_lo_quantity.gpu_data, 
                                        f_lo_orderdate.gpu_data, 
                                        f_lo_extendedprice.gpu_data);
            }
        }
    }

    {
        time_block t("Tclose1: ");
        pipelines[1]->close();
    }

    size_t c_out;
    // gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));

    // //this results is constant for SSBM Q1.1
    // EXPECT_EQ(c_out, 365);

    gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));
    std::cout << c_out << std::endl;

    //for the current dataset, regenerating it may change the results
    EXPECT_TRUE((c_out == UINT64_C(-1) || ((uint32_t) c_out) == ((uint32_t) UINT64_C(-1))) && "TODO: compare with correct value"); //FIXME: compare with correct value
}

TEST_F(GpuSSBMTest3, gpuSSBM_Q2_1) {

    const char *testLabel = "gpuSSBM_Q2_1";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/ssbm_q2_1.json";

    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        pipelines = ctx->getPipelines();
    }
}
