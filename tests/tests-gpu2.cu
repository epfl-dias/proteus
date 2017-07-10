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

const dim3 defaultBlockDim(1024, 1, 1);
const dim3 defaultGridDim (1024, 1, 1);

class GPUOutputTest3 : public ::testing::Test {
protected:
    virtual void SetUp();
    virtual void TearDown();

    CSVPlugin * openCSV(RawContext* const context, string& fname,
        RecordType& rec, vector<RecordAttribute*> whichFields,
        char delimInner, int lineHint, int policy, bool stringBrackets = true)
    {
        CSVPlugin * plugin = new CSVPlugin(context, fname, rec, whichFields);
            catalog->registerPlugin(fname, plugin);
            return plugin;
    }


    void createKernel(GpuRawContext &ctx);
    void createKernel2(GpuRawContext &ctx);

    void launch_kernel(void ** args, dim3 gridDim, dim3 blockDim);
    void launch_kernel(void ** args, dim3 gridDim);
    void launch_kernel(void ** args);


    bool flushResults = true;
    const char * testPath = TEST_OUTPUTS "/tests-output/";

    const char * catalogJSON = "inputs/plans/catalog.json";
    
private:
    RawCatalog * catalog;
    CachingService * caches;

public:
    size_t    N;
    int32_t * a;
    double  * b;
    int32_t * c;
    int32_t * d;

    int32_t * h_a;
    double  * h_b;
    int32_t * h_c;
    int32_t * h_d;
    int32_t * h_e;

    CUdevice device;
    CUmodule cudaModule;
    CUcontext context;
    CUfunction function;
    CUlinkState linker;
    int devCount;
    int devMajor, devMinor;
};

void GPUOutputTest3::createKernel(GpuRawContext &ctx){
    // Module * mod = ctx.getModule();

    // Type * int32_type = Type::getInt32Ty(ctx.getLLVMContext());
    // Type * int1_type  = Type::getInt1Ty(ctx.getLLVMContext());

    // std::vector<Type *> inputs;
    // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 0));
    // inputs.push_back(PointerType::get(Type::getDoubleTy(ctx.getLLVMContext()), /* address space */ 0));
    // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 1)); // needs to be in device memory for atomic write
    // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 1)); // needs to be in device memory for atomic write
    // inputs.push_back(PointerType::get(int1_type                              , /* address space */ 1)); // needs to be in device memory for atomic write
    // inputs.push_back(PointerType::get(int1_type                              , /* address space */ 1)); // needs to be in device memory for atomic write

    // Type * size_type;
    // if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty(ctx.getLLVMContext());
    // else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty(ctx.getLLVMContext());
    // else                          assert(false);
    // inputs.push_back(size_type);

    // FunctionType *entry_point_type = FunctionType::get(Type::getVoidTy(ctx.getLLVMContext()), inputs, false);
    
    // Function *entry_point = Function::Create(entry_point_type, Function::ExternalLinkage, "jit_kernel", mod);

    // for (size_t i = 0 ; i < 2 ; ++i){
    //     entry_point->setOnlyReadsMemory(i + 1); //+1 because 0 is the return value
    //     entry_point->setDoesNotAlias(i + 1); //+1 because 0 is the return value
    // }
    // for (size_t i = 2 ; i < 6 ; ++i){
    //     entry_point->setDoesNotAlias(i + 1); //+1 because 0 is the return value
    // }

    // ctx.setGlobalFunction(entry_point);

    // //SCAN1
    string filename = string("inputs/sailors.csv");
    PrimitiveType * intType = new IntType();
    PrimitiveType* floatType = new FloatType();
    PrimitiveType* stringType = new StringType();
    RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
            intType);
    RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
            stringType);
    RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
            intType);
    RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
            floatType);

    list<RecordAttribute*> attrList;
    attrList.push_back(sid);
    attrList.push_back(sname);
    attrList.push_back(rating);
    attrList.push_back(age);

    RecordType rec1 = RecordType(attrList);

    vector<RecordAttribute*> whichFields;
    whichFields.push_back(sid);
    whichFields.push_back(age);


    GpuColScanPlugin * pg = new GpuColScanPlugin(&ctx, filename, rec1, whichFields);
    catalog->registerPlugin(filename, pg);
  
    Scan scan(&ctx, *pg);

  // /**
  //  * REDUCE
  //  */
  
  RecordAttribute projTuple = RecordAttribute(filename, activeLoop, new Int64Type());
  list<RecordAttribute> projections = list<RecordAttribute>();
  projections.push_back(projTuple);
  projections.push_back(*sid);
  projections.push_back(*age);

  expressions::Expression* arg = new expressions::InputArgument(&rec1, 0, projections);

  expressions::Expression* outputExpr = new expressions::RecordProjection(intType, arg, *sid);

  expressions::Expression* lhs = new expressions::RecordProjection(floatType, arg, *age);
  expressions::Expression* rhs = new expressions::FloatConstant(40.0);
  expressions::Expression* predicate = new expressions::GtExpression(new BoolType(), lhs, rhs);
  expressions::Expression* rhs2 = new expressions::IntConstant(60);
  expressions::Expression* predicateExpr = new expressions::LtExpression(new BoolType(), outputExpr, rhs2);

  vector<Monoid> accs;
  vector<expressions::Expression*> exprs;
  accs.push_back(SUM);
  accs.push_back(MAX);
  accs.push_back(OR);
  accs.push_back(AND);
  exprs.push_back(outputExpr);
  exprs.push_back(outputExpr);
  exprs.push_back(predicateExpr);
  exprs.push_back(predicateExpr);

  opt::GpuReduce reduce = opt::GpuReduce(accs, 
                                            exprs, 
                                            predicate, 
                                            &scan, 
                                            &ctx);

  scan.setParent(&reduce);

  reduce.produce();

  // ctx.getBuilder()->SetInsertPoint(ctx.getEndingBlock());

  //   ctx.getBuilder()->CreateRetVoid();

    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
#ifdef DEBUGCTX
//  getModule()->dump();
#endif
    // Validate the generated code, checking for consistency.
    verifyFunction(*ctx.getGlobalFunction());

    //Run function
    ctx.prepareFunction(ctx.getGlobalFunction());
}

void GPUOutputTest3::createKernel2(GpuRawContext &ctx){
    // //SCAN1
    string filename = string("inputs/sailors.csv");
    PrimitiveType * intType = new IntType();
    PrimitiveType* floatType = new FloatType();
    PrimitiveType* stringType = new StringType();
    RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
            intType);
    RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
            stringType);
    RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
            intType);
    RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
            floatType);

    list<RecordAttribute*> attrList;
    attrList.push_back(sid);
    attrList.push_back(sname);
    attrList.push_back(rating);
    attrList.push_back(age);

    RecordType rec1 = RecordType(attrList);

    vector<RecordAttribute*> whichFields;
    whichFields.push_back(sid);
    whichFields.push_back(age);


    GpuColScanPlugin * pg = new GpuColScanPlugin(&ctx, filename, rec1, whichFields);
    catalog->registerPlugin(filename, pg);
  
    Scan scan(&ctx, *pg);

    // /**
    //  * REDUCE
    //  */
  
    RecordAttribute projTuple = RecordAttribute(filename, activeLoop, new Int64Type());
    list<RecordAttribute> projections = list<RecordAttribute>();
    projections.push_back(projTuple);
    projections.push_back(*sid);
    projections.push_back(*age);

    expressions::Expression* arg = new expressions::InputArgument(&rec1, 0, projections);

    expressions::Expression* outputExpr = new expressions::RecordProjection(intType, arg, *sid);

    expressions::Expression* lhs = new expressions::RecordProjection(floatType, arg, *age);
    expressions::Expression* rhs = new expressions::FloatConstant(40.0);
    expressions::Expression* predicate = new expressions::GtExpression(new BoolType(), lhs, rhs);

    Select sel(predicate, &scan);
    scan.setParent(&sel);

    GpuExprMaterializer mat({GpuMatExpr{outputExpr, 0, 0}}, vector<size_t>{
            ((const PrimitiveType *) outputExpr->getExpressionType())->getLLVMType(ctx.getLLVMContext())->getPrimitiveSizeInBits()
        }, &sel, &ctx, "mat");
    sel.setParent(&mat);

    mat.produce();

    // ctx.getBuilder()->SetInsertPoint(ctx.getEndingBlock());

    // ctx.getBuilder()->CreateRetVoid();

    LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
#ifdef DEBUGCTX
//  getModule()->dump();
#endif
    // Validate the generated code, checking for consistency.
    verifyFunction(*ctx.getGlobalFunction());

    //Run function
    ctx.prepareFunction(ctx.getGlobalFunction());
}

// Dummy test use to wake up the device before running any other test.
// This makes timings more meaningful.
TEST_F(GPUOutputTest3, gpuWakeUp) { // DO NOT DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}


__global__ void kernel_gpuReduceNumeric(const int32_t * __restrict__ sid_ptr,
                                        const double  * __restrict__ age_ptr,
                                              int32_t * __restrict__ result_sum,
                                              int32_t * __restrict__ result_max,
                                              bool    * __restrict__ result_or,
                                              bool    * __restrict__ result_and,
                                              size_t cnt){
    const size_t tid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int    laneid = tid & 0x1F;
    int32_t   local_sum = 0;
    int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

    bool      local_and = true ;
    bool      local_or  = false;

    for (size_t i = tid ; i < cnt ; i += blockDim.x * gridDim.x){
        if (age_ptr[i] > 40){
            int32_t sid = sid_ptr[i];
            local_sum  += sid;
            local_max   = max(local_max, sid);
            local_and   = local_and && (sid < 60);
            local_or    = local_or  || (sid < 60);
        }
    }

    #pragma unroll
    for (int m = 32 >> 1; m > 0; m >>= 1){
        local_sum += __shfl_xor(local_sum, m);
        local_max  = max(local_max, __shfl_xor(local_max, m));
    }

    local_and = __all(local_and);
    local_or  = __any(local_or );

    if (laneid == 0){
        atomicAdd(result_sum, local_sum);
        atomicMax(result_max, local_max);
        if ( local_or ) *result_or  = true ;
        if (!local_and) *result_and = false;
        // atomicAnd((int *) result_and, (int) local_and);
    }
}

void cpu_gpuReduceNumeric(const int32_t * __restrict__ sid_ptr,
                            const double  * __restrict__ age_ptr,
                                  int32_t * __restrict__ result_sum,
                                  int32_t * __restrict__ result_max,
                                  bool    * __restrict__ result_or,
                                  bool    * __restrict__ result_and,
                                  size_t cnt){
    int32_t   local_sum = 0;
    int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

    bool      local_and = true ;
    bool      local_or  = false;

    for (size_t i = 0 ; i < cnt ; ++i) {
        if (age_ptr[i] > 40.0) {
            int32_t sid = sid_ptr[i];
            local_sum  += sid;
            local_max   = std::max(local_max, sid);
            local_and   = local_and && (sid < 60);
            local_or    = local_or  || (sid < 60);
        }
    }
    *result_sum = local_sum;
    *result_max = local_max;
    *result_and = local_and;
    *result_or  = local_or ;
}

TEST_F(GPUOutputTest3, gpuReduceNumeric) {
    auto start = std::chrono::system_clock::now();

    const char *testLabel = "gpuReduceNumeric";
    GpuRawContext * ctx;

    {
        auto start = std::chrono::system_clock::now();

    ctx = new GpuRawContext(testLabel);
    createKernel(*ctx);

    ctx->compileAndLoad();

    // Get kernel function
    function = ctx->getKernel();

        auto end   = std::chrono::system_clock::now();
        std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    // Create driver context
    // for (size_t i = 0 ; i < N ; ++i) h_c[i] = 0;

    h_c[0] =  0;
    h_c[1] =  0;
    h_c[2] =  0;
    h_c[3] = 0xFF;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));

    int32_t * c2 = c + 1;
    bool    * c3 = (bool *) (c + 2);
    bool    * c4 = (bool *) (c + 3);

    // Kernel parameters
    void *KernelParams[] = {&a, &b, &N, &c, &c2, &c3, &c4};


    {
        auto start = std::chrono::system_clock::now();
        // Kernel launch
        launch_kernel(KernelParams);

        gpu_run(cudaDeviceSynchronize());

        auto end   = std::chrono::system_clock::now();
        std::cout << "Tgenerated: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*4, cudaMemcpyDefault));

    for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_c[i] << " "; std::cout << std::endl;

    int32_t h_d[4];

    h_d[0] = 0;
    h_d[1] = 0;
    h_d[2] = 0;
    h_d[3] = 0xFF;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_d, sizeof(int32_t) * 4, cudaMemcpyDefault));

    {
        auto start = std::chrono::system_clock::now();

        kernel_gpuReduceNumeric<<<1024, 1024, 0, 0>>>(a, b, c, c2, c3, c4, N);

        gpu_run(cudaDeviceSynchronize());
        
        auto end   = std::chrono::system_clock::now();
        std::cout << "Thandwritten: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    gpu_run(cudaMemcpy(h_d, c, sizeof(int32_t)*4, cudaMemcpyDefault));

    for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_d[i] << " "; std::cout << std::endl;

    int32_t   local_sum = 0;
    int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

    bool      local_and = true ;
    bool      local_or  = false;
    {

        auto start = std::chrono::system_clock::now();

        cpu_gpuReduceNumeric(h_a, h_b, &local_sum, &local_max, &local_and, &local_or, N);

        auto end   = std::chrono::system_clock::now();
        std::cout << "Tcpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    std::cout << local_sum << " " << local_max << " " << local_and << " " << local_or << std::endl;
    EXPECT_EQ(local_sum, h_c[0]);
    EXPECT_EQ(local_max, h_c[1]);
    EXPECT_EQ(local_and, !!(h_c[2] & 0xFF));
    EXPECT_EQ(local_or , !!(h_c[3] & 0xFF));
    EXPECT_EQ(local_sum, h_d[0]);
    EXPECT_EQ(local_max, h_d[1]);
    EXPECT_EQ(local_and, !!(h_d[2] & 0xFF));
    EXPECT_EQ(local_or , !!(h_d[3] & 0xFF));
}


__global__ void kernel_select(  const int32_t * __restrict__ sid_ptr,
                                const double  * __restrict__ age_ptr,
                                      int32_t * __restrict__ out_ptr,
                                      int32_t * __restrict__ out_cnt,
                                      size_t                 cnt){
    const size_t tid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int    laneid = tid & 0x1F;

    for (size_t i = tid ; i < cnt ; i += blockDim.x * gridDim.x){
        if (age_ptr[i] > 40){
            // int32_t filter = __ballot(1);
            // int     lcnt   = __popc(filter);

            int32_t old_cnt = atomicAdd(out_cnt, 1);

            // int r = __popc(filter & ((1 << laneid) - 1));
            out_ptr[old_cnt] = sid_ptr[i];
        }
    }
}


void cpu_gpuSelectNumeric(const int32_t * __restrict__ sid_ptr,
                            const double  * __restrict__ age_ptr,
                                  int32_t * __restrict__ result_ptr,
                                  size_t cnt){
    int32_t   res = 0;

    for (size_t i = 0 ; i < cnt ; ++i) {
        if (age_ptr[i] > 40.0) result_ptr[res++] = sid_ptr[i];
    }
}

void GPUOutputTest3::launch_kernel(void ** args, dim3 gridDim, dim3 blockDim){
    gpu_run(cuLaunchKernel(function, gridDim.x, gridDim.y, gridDim.z,
                                 blockDim.x, blockDim.y, blockDim.z,
                                 0, NULL, args, NULL));
}

void GPUOutputTest3::launch_kernel(void ** args, dim3 gridDim){
    launch_kernel(args, gridDim, defaultBlockDim);
}

void GPUOutputTest3::launch_kernel(void ** args){
    launch_kernel(args, defaultGridDim, defaultBlockDim);
}

TEST_F(GPUOutputTest3, gpuSelectNumeric) {
    auto start = std::chrono::system_clock::now();

    const char *testLabel = "gpuSelectNumeric";
    GpuRawContext * ctx;

    {
        auto start = std::chrono::system_clock::now();

    ctx = new GpuRawContext(testLabel);
    createKernel2(*ctx);

    ctx->compileAndLoad();

    // Get kernel function
    function = ctx->getKernel();

        auto end   = std::chrono::system_clock::now();
        std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Create driver context
    
    h_c[0] =  0;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 1, cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));

    // Kernel parameters
    void *KernelParams[] = {&a, &b, &N, &d, &c};


    {
        auto start = std::chrono::system_clock::now();
        // Kernel launch
        // gpu_run(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
        //                              blockSizeX, blockSizeY, blockSizeZ,
        //                              0, NULL, KernelParams, NULL));
        launch_kernel(KernelParams);


        gpu_run(cudaDeviceSynchronize());

        auto end   = std::chrono::system_clock::now();
        std::cout << "Tgenerated: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*1, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(h_d, d, sizeof(int32_t)*N, cudaMemcpyDefault));

    h_c[1] =  0;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c + 1, sizeof(int32_t) * 1, cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));


    {
        time_block t("Thandwritten: ");

        kernel_select<<<1024, 1024, 0, 0>>>(a, b, d, c, N);

        gpu_run(cudaDeviceSynchronize());
    }
    gpu_run(cudaMemcpy(h_c + 1, c, sizeof(int32_t)*1, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(h_e, d, sizeof(int32_t)*N, cudaMemcpyDefault));

    EXPECT_EQ(h_c[0], h_c[1]);

    if (h_c[0] == h_c[1]){
        std::sort(h_e, h_e + h_c[1]);
        std::sort(h_d, h_d + h_c[0]);

        for (int i = 0 ; i < std::min(h_c[0], h_c[1]) ; ++i) EXPECT_EQ(h_d[i], h_e[i]);
    }

//     for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_c[i] << " "; std::cout << std::endl;

//     int32_t h_d[4];

//     h_d[0] = 0;
//     h_d[1] = 0;
//     h_d[2] = 0;
//     h_d[3] = 0xFF;

//     gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(c, h_d, sizeof(int32_t) * 4, cudaMemcpyDefault));

//     {
//         auto start = std::chrono::system_clock::now();

//         kernel_gpuReduceNumeric<<<1024, 1024, 0, 0>>>(a, b, c, c2, c3, c4, N);

//         gpu_run(cudaDeviceSynchronize());
        
//         auto end   = std::chrono::system_clock::now();
//         std::cout << "Thandwritten: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

//     }
//     gpu_run(cudaMemcpy(h_d, c, sizeof(int32_t)*4, cudaMemcpyDefault));

//     for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_d[i] << " "; std::cout << std::endl;

//     int32_t   local_sum = 0;
//     int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

//     bool      local_and = true ;
//     bool      local_or  = false;
    {

        auto start = std::chrono::system_clock::now();

        cpu_gpuSelectNumeric(h_a, h_b, h_d, N);

        auto end   = std::chrono::system_clock::now();
        std::cout << "Tcpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
//     std::cout << local_sum << " " << local_max << " " << local_and << " " << local_or << std::endl;
//     EXPECT_EQ(local_sum, h_c[0]);
//     EXPECT_EQ(local_max, h_c[1]);
//     EXPECT_EQ(local_and, !!(h_c[2] & 0xFF));
//     EXPECT_EQ(local_or , !!(h_c[3] & 0xFF));
//     EXPECT_EQ(local_sum, h_d[0]);
//     EXPECT_EQ(local_max, h_d[1]);
//     EXPECT_EQ(local_and, !!(h_d[2] & 0xFF));
//     EXPECT_EQ(local_or , !!(h_d[3] & 0xFF));
}

void GPUOutputTest3::SetUp() {
    // NVPTXTargetMachine64 TM(TheNVPTXTarget64, Triple("nvptx64-nvidia-cuda"), "sm_61", );
    // TODO: initialize only the required ones...
    // InitializeAllTargets();
    // InitializeAllTargetMCs();
    // InitializeAllAsmPrinters();
    // InitializeAllAsmParsers();

    catalog = &RawCatalog::getInstance();
    caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();

    // CUDA initialization
    // gpu_run(cuInit(0));
    // gpu_run(cuDeviceGetCount(&devCount));
    // gpu_run(cuDeviceGet(&device, 0));

    // char name[128];
    // gpu_run(cuDeviceGetName(name, 128, device));
    // std::cout << "Using CUDA Device [0]: " << name << "\n";

    // gpu_run(cuDeviceComputeCapability(&devMajor, &devMinor, device));
    // std::cout << "Device Compute Capability: " << devMajor << "." << devMinor << "\n";
    // if (devMajor < 2) {
    //     std::cout << "ERROR: Device 0 is not SM 2.0 or greater\n";
    //     EXPECT_TRUE(false);
    // }

    // gpu_run(cuCtxCreate(&context, 0, device));
    N = 1024*1024*256;
    
    gpu_run(cudaMalloc(&a, sizeof(int32_t)*N));
    gpu_run(cudaMalloc(&b, sizeof(double )*N));
    gpu_run(cudaMalloc(&c, sizeof(int32_t)*4));
    gpu_run(cudaMalloc(&d, sizeof(int32_t)*N));

    gpu_run(cudaMallocHost(&h_a, sizeof(int32_t)*N));
    gpu_run(cudaMallocHost(&h_b, sizeof(double )*N));
    gpu_run(cudaMallocHost(&h_c, sizeof(int32_t)*4));
    gpu_run(cudaMallocHost(&h_d, sizeof(int32_t)*N));
    gpu_run(cudaMallocHost(&h_e, sizeof(int32_t)*N));


    srand(time(NULL));

    for (size_t i = 0 ; i < N ; ++i) h_a[i] = rand();
    for (size_t i = 0 ; i < N ; ++i) h_b[i] = (((double) rand())/RAND_MAX) * 80;
}

void GPUOutputTest3::TearDown() {
    gpu_run(cudaFree(a));
    gpu_run(cudaFree(b));
    gpu_run(cudaFree(c));
    gpu_run(cudaFree(d));

    gpu_run(cudaFreeHost(h_a));
    gpu_run(cudaFreeHost(h_b));
    gpu_run(cudaFreeHost(h_c));
    gpu_run(cudaFreeHost(h_d));
    gpu_run(cudaFreeHost(h_e));
}



__global__ void kernel_gpuPlan(const int32_t * __restrict__ sid_ptr,
                                const double  * __restrict__ age_ptr,
                                      int32_t * __restrict__ result_cnt,
                                      int32_t * __restrict__ result_max,
                                      size_t cnt){
    const size_t tid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int    laneid = tid & 0x1F;

    int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

    for (size_t i = tid ; i < cnt ; i += blockDim.x * gridDim.x){
        // if (age_ptr[i] > 40)
        local_max   = max(local_max, sid_ptr[i]);
    }

    #pragma unroll
    for (int m = 32 >> 1; m > 0; m >>= 1){
        local_max  = max(local_max, __shfl_xor(local_max, m));
    }

    if (laneid == 0) atomicMax(result_max, local_max);

    if (blockIdx.x == 0 && threadIdx.x == 0) *result_cnt = cnt;
}

TEST_F(GPUOutputTest3, gpuPlan) {

    const char *testLabel = "gpuPlan";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-scan-gpu.json";

    {
        auto start = std::chrono::system_clock::now();

        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        // Get kernel function
        function = ctx->getKernel();

        auto end   = std::chrono::system_clock::now();
        std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    h_c[0] =  0;
    h_c[1] =  0;
    h_c[2] =  0;
    h_c[3] = 0xFF;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));

    int32_t * c2 = c + 1;

    // Kernel parameters
    void *KernelParams[] = {&a, &b, &N, &c, &c2};

    {
        time_block t("Tgenerated: ");
        // Kernel launch
        launch_kernel(KernelParams);

        gpu_run(cudaDeviceSynchronize());
    }

    int32_t h2_c[4];
    gpu_run(cudaMemcpy(h2_c, c, sizeof(int32_t)*4, cudaMemcpyDefault));

    h_c[0] =  0;
    h_c[1] =  0;
    h_c[2] =  0;
    h_c[3] = 0xFF;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));

    c2 = c + 1;

    {
        time_block t("Thandwritten: ");

        kernel_gpuPlan<<<1024, 1024, 0, 0>>>(a, b, c, c2, N);

        gpu_run(cudaDeviceSynchronize());
    }

    gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*4, cudaMemcpyDefault));

    for (size_t i = 0 ; i < 2 ; ++i) EXPECT_EQ(h_c[i], h2_c[i]);
}

TEST_F(GPUOutputTest3, gpuPlan2) {

    const char *testLabel = "gpuPlan2";
    GpuRawContext * ctx;

    const char* planPath = "inputs/plans/reduce-select-gpu.json";

    {
        auto start = std::chrono::system_clock::now();

        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        // Get kernel function
        function = ctx->getKernel();

        auto end   = std::chrono::system_clock::now();
        std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Create driver context
    
    h_c[0] =  0;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 1, cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));

    // Kernel parameters
    void *KernelParams[] = {&a, &b, &N, &d, &c};


    {
        auto start = std::chrono::system_clock::now();
        // Kernel launch
        // gpu_run(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
        //                              blockSizeX, blockSizeY, blockSizeZ,
        //                              0, NULL, KernelParams, NULL));
        launch_kernel(KernelParams);


        gpu_run(cudaDeviceSynchronize());

        auto end   = std::chrono::system_clock::now();
        std::cout << "Tgenerated: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*1, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(h_d, d, sizeof(int32_t)*N, cudaMemcpyDefault));

    h_c[1] =  0;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c + 1, sizeof(int32_t) * 1, cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));


    {
        time_block t("Thandwritten: ");

        kernel_select<<<1024, 1024, 0, 0>>>(a, b, d, c, N);

        gpu_run(cudaDeviceSynchronize());
    }
    gpu_run(cudaMemcpy(h_c + 1, c, sizeof(int32_t)*1, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(h_e, d, sizeof(int32_t)*N, cudaMemcpyDefault));

    EXPECT_EQ(h_c[0], h_c[1]);

    if (h_c[0] == h_c[1]){
        std::sort(h_e, h_e + h_c[1]);
        std::sort(h_d, h_d + h_c[0]);

        for (int i = 0 ; i < std::min(h_c[0], h_c[1]) ; ++i) EXPECT_EQ(h_d[i], h_e[i]);
    }
}