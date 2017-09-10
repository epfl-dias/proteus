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

    void launch(void ** args, dim3 gridDim, dim3 blockDim);
    void launch(void ** args, dim3 gridDim);
    void launch(void ** args);

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

// void GPUOutputTest3::createKernel(GpuRawContext &ctx){
//     // Module * mod = ctx.getModule();

//     // Type * int32_type = Type::getInt32Ty(ctx.getLLVMContext());
//     // Type * int1_type  = Type::getInt1Ty(ctx.getLLVMContext());

//     // std::vector<Type *> inputs;
//     // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 0));
//     // inputs.push_back(PointerType::get(Type::getDoubleTy(ctx.getLLVMContext()), /* address space */ 0));
//     // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 1)); // needs to be in device memory for atomic write
//     // inputs.push_back(PointerType::get(int32_type                             , /* address space */ 1)); // needs to be in device memory for atomic write
//     // inputs.push_back(PointerType::get(int1_type                              , /* address space */ 1)); // needs to be in device memory for atomic write
//     // inputs.push_back(PointerType::get(int1_type                              , /* address space */ 1)); // needs to be in device memory for atomic write

//     // Type * size_type;
//     // if      (sizeof(size_t) == 4) size_type = Type::getInt32Ty(ctx.getLLVMContext());
//     // else if (sizeof(size_t) == 8) size_type = Type::getInt64Ty(ctx.getLLVMContext());
//     // else                          assert(false);
//     // inputs.push_back(size_type);

//     // FunctionType *entry_point_type = FunctionType::get(Type::getVoidTy(ctx.getLLVMContext()), inputs, false);
    
//     // Function *entry_point = Function::Create(entry_point_type, Function::ExternalLinkage, "jit_kernel", mod);

//     // for (size_t i = 0 ; i < 2 ; ++i){
//     //     entry_point->setOnlyReadsMemory(i + 1); //+1 because 0 is the return value
//     //     entry_point->setDoesNotAlias(i + 1); //+1 because 0 is the return value
//     // }
//     // for (size_t i = 2 ; i < 6 ; ++i){
//     //     entry_point->setDoesNotAlias(i + 1); //+1 because 0 is the return value
//     // }

//     // ctx.setGlobalFunction(entry_point);

//     // //SCAN1
//     string filename = string("inputs/sailors.csv");
//     PrimitiveType * intType = new IntType();
//     PrimitiveType* floatType = new FloatType();
//     PrimitiveType* stringType = new StringType();
//     RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
//             intType);
//     RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
//             stringType);
//     RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
//             intType);
//     RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
//             floatType);

//     list<RecordAttribute*> attrList;
//     attrList.push_back(sid);
//     attrList.push_back(sname);
//     attrList.push_back(rating);
//     attrList.push_back(age);

//     RecordType rec1 = RecordType(attrList);

//     vector<RecordAttribute*> whichFields;
//     whichFields.push_back(sid);
//     whichFields.push_back(age);


//     GpuColScanPlugin * pg = new GpuColScanPlugin(&ctx, filename, rec1, whichFields);
//     catalog->registerPlugin(filename, pg);
  
//     Scan scan(&ctx, *pg);

//   // /**
//   //  * REDUCE
//   //  */
  
//   RecordAttribute projTuple = RecordAttribute(filename, activeLoop, new Int64Type());
//   list<RecordAttribute> projections = list<RecordAttribute>();
//   projections.push_back(projTuple);
//   projections.push_back(*sid);
//   projections.push_back(*age);

//   expressions::Expression* arg = new expressions::InputArgument(&rec1, 0, projections);

//   expressions::Expression* outputExpr = new expressions::RecordProjection(intType, arg, *sid);

//   expressions::Expression* lhs = new expressions::RecordProjection(floatType, arg, *age);
//   expressions::Expression* rhs = new expressions::FloatConstant(40.0);
//   expressions::Expression* predicate = new expressions::GtExpression(new BoolType(), lhs, rhs);
//   expressions::Expression* rhs2 = new expressions::IntConstant(60);
//   expressions::Expression* predicateExpr = new expressions::LtExpression(new BoolType(), outputExpr, rhs2);

//   vector<Monoid> accs;
//   vector<expressions::Expression*> exprs;
//   accs.push_back(SUM);
//   accs.push_back(MAX);
//   accs.push_back(OR);
//   accs.push_back(AND);
//   exprs.push_back(outputExpr);
//   exprs.push_back(outputExpr);
//   exprs.push_back(predicateExpr);
//   exprs.push_back(predicateExpr);

//   opt::GpuReduce reduce = opt::GpuReduce(accs, 
//                                             exprs, 
//                                             predicate, 
//                                             &scan, 
//                                             &ctx);

//   scan.setParent(&reduce);

//   reduce.produce();

//   // ctx.getBuilder()->SetInsertPoint(ctx.getEndingBlock());

//   //   ctx.getBuilder()->CreateRetVoid();

//     LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
// #ifdef DEBUGCTX
// //  getModule()->dump();
// #endif
//     // Validate the generated code, checking for consistency.
//     verifyFunction(*ctx.getGlobalFunction());

//     //Run function
//     ctx.prepareFunction(ctx.getGlobalFunction());
// }

// void GPUOutputTest3::createKernel2(GpuRawContext &ctx){
//     // //SCAN1
//     string filename = string("inputs/sailors.csv");
//     PrimitiveType * intType = new IntType();
//     PrimitiveType* floatType = new FloatType();
//     PrimitiveType* stringType = new StringType();
//     RecordAttribute* sid = new RecordAttribute(1, filename, string("sid"),
//             intType);
//     RecordAttribute* sname = new RecordAttribute(2, filename, string("sname"),
//             stringType);
//     RecordAttribute* rating = new RecordAttribute(3, filename, string("rating"),
//             intType);
//     RecordAttribute* age = new RecordAttribute(4, filename, string("age"),
//             floatType);

//     list<RecordAttribute*> attrList;
//     attrList.push_back(sid);
//     attrList.push_back(sname);
//     attrList.push_back(rating);
//     attrList.push_back(age);

//     RecordType rec1 = RecordType(attrList);

//     vector<RecordAttribute*> whichFields;
//     whichFields.push_back(sid);
//     whichFields.push_back(age);


//     GpuColScanPlugin * pg = new GpuColScanPlugin(&ctx, filename, rec1, whichFields);
//     catalog->registerPlugin(filename, pg);
  
//     Scan scan(&ctx, *pg);

//     // /**
//     //  * REDUCE
//     //  */
  
//     RecordAttribute projTuple = RecordAttribute(filename, activeLoop, new Int64Type());
//     list<RecordAttribute> projections = list<RecordAttribute>();
//     projections.push_back(projTuple);
//     projections.push_back(*sid);
//     projections.push_back(*age);

//     expressions::Expression* arg = new expressions::InputArgument(&rec1, 0, projections);

//     expressions::Expression* outputExpr = new expressions::RecordProjection(intType, arg, *sid);

//     expressions::Expression* lhs = new expressions::RecordProjection(floatType, arg, *age);
//     expressions::Expression* rhs = new expressions::FloatConstant(40.0);
//     expressions::Expression* predicate = new expressions::GtExpression(new BoolType(), lhs, rhs);

//     Select sel(predicate, &scan);
//     scan.setParent(&sel);

//     GpuExprMaterializer mat({GpuMatExpr{outputExpr, 0, 0}}, vector<size_t>{
//             ((const PrimitiveType *) outputExpr->getExpressionType())->getLLVMType(ctx.getLLVMContext())->getPrimitiveSizeInBits()
//         }, &sel, &ctx, "mat");
//     sel.setParent(&mat);

//     mat.produce();

//     // ctx.getBuilder()->SetInsertPoint(ctx.getEndingBlock());

//     // ctx.getBuilder()->CreateRetVoid();

//     LOG(INFO) << "[Prepare Function: ] Exit"; //and dump code so far";
// #ifdef DEBUGCTX
// //  getModule()->dump();
// #endif
//     // Validate the generated code, checking for consistency.
//     verifyFunction(*ctx.getGlobalFunction());

//     //Run function
//     ctx.prepareFunction(ctx.getGlobalFunction());
// }

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

// TEST_F(GPUOutputTest3, gpuReduceNumeric) {
//     auto start = std::chrono::system_clock::now();

//     const char *testLabel = "gpuReduceNumeric";
//     GpuRawContext * ctx;

//     {
//         auto start = std::chrono::system_clock::now();

//     ctx = new GpuRawContext(testLabel);
//     createKernel(*ctx);

//     ctx->compileAndLoad();

//     // Get kernel function
//     function = ctx->getKernel()[0];

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//     }
//     // Create driver context
//     // for (size_t i = 0 ; i < N ; ++i) h_c[i] = 0;

//     h_c[0] =  0;
//     h_c[1] =  0;
//     h_c[2] =  0;
//     h_c[3] = 0xFF;

//     gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));

//     int32_t * c2 = c + 1;
//     bool    * c3 = (bool *) (c + 2);
//     bool    * c4 = (bool *) (c + 3);

//     // Kernel parameters
//     // void *KernelParams[] = {&a, &b, &N, &c, &c2, &c3, &c4};
//     void *KernelParams[] = {&c, &c2, &c3, &c4, &a, &b, &N};

//     {
//         auto start = std::chrono::system_clock::now();
//         // Kernel launch
//         launch(KernelParams);

//         gpu_run(cudaDeviceSynchronize());

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "Tgenerated: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//     }
//     gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*4, cudaMemcpyDefault));

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
//     {

//         auto start = std::chrono::system_clock::now();

//         cpu_gpuReduceNumeric(h_a, h_b, &local_sum, &local_max, &local_and, &local_or, N);

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "Tcpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

//     }
//     std::cout << local_sum << " " << local_max << " " << local_and << " " << local_or << std::endl;
//     EXPECT_EQ(local_sum, h_c[0]);
//     EXPECT_EQ(local_max, h_c[1]);
//     EXPECT_EQ(local_and, !!(h_c[2] & 0xFF));
//     EXPECT_EQ(local_or , !!(h_c[3] & 0xFF));
//     EXPECT_EQ(local_sum, h_d[0]);
//     EXPECT_EQ(local_max, h_d[1]);
//     EXPECT_EQ(local_and, !!(h_d[2] & 0xFF));
//     EXPECT_EQ(local_or , !!(h_d[3] & 0xFF));
// }


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

void GPUOutputTest3::launch(void ** args, dim3 gridDim, dim3 blockDim){
    launch_kernel(function, args, gridDim, blockDim);
}

void GPUOutputTest3::launch(void ** args, dim3 gridDim){
    launch_kernel(function, args, gridDim);
}

void GPUOutputTest3::launch(void ** args){
    launch_kernel(function, args);
}

// TEST_F(GPUOutputTest3, gpuSelectNumeric) {
//     auto start = std::chrono::system_clock::now();

//     const char *testLabel = "gpuSelectNumeric";
//     GpuRawContext * ctx;

//     {
//         auto start = std::chrono::system_clock::now();

//     ctx = new GpuRawContext(testLabel);
//     createKernel2(*ctx);

//     ctx->compileAndLoad();

//     // Get kernel function
//     function = ctx->getKernel()[0];

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "codegen: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//     }

//     // Create driver context
    
//     h_c[0] =  0;

//     gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 1, cudaMemcpyDefault));
//     // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));

//     // Kernel parameters
//     // void *KernelParams[] = {&a, &b, &N, &d, &c};
//     void *KernelParams[] = {&d, &c, &a, &b, &N};


//     {
//         auto start = std::chrono::system_clock::now();
//         // Kernel launch
//         // gpu_run(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
//         //                              blockSizeX, blockSizeY, blockSizeZ,
//         //                              0, NULL, KernelParams, NULL));
//         launch(KernelParams);


//         gpu_run(cudaDeviceSynchronize());

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "Tgenerated: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//     }
//     gpu_run(cudaMemcpy(h_c, c, sizeof(int32_t)*1, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(h_d, d, sizeof(int32_t)*N, cudaMemcpyDefault));

//     h_c[1] =  0;

//     gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(c, h_c + 1, sizeof(int32_t) * 1, cudaMemcpyDefault));
//     // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));


//     {
//         time_block t("Thandwritten: ");

//         kernel_select<<<1024, 1024, 0, 0>>>(a, b, d, c, N);

//         gpu_run(cudaDeviceSynchronize());
//     }
//     gpu_run(cudaMemcpy(h_c + 1, c, sizeof(int32_t)*1, cudaMemcpyDefault));
//     gpu_run(cudaMemcpy(h_e, d, sizeof(int32_t)*N, cudaMemcpyDefault));

//     EXPECT_EQ(h_c[0], h_c[1]);

//     if (h_c[0] == h_c[1]){
//         std::sort(h_e, h_e + h_c[1]);
//         std::sort(h_d, h_d + h_c[0]);

//         for (int i = 0 ; i < std::min(h_c[0], h_c[1]) ; ++i) EXPECT_EQ(h_d[i], h_e[i]);
//     }

// //     for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_c[i] << " "; std::cout << std::endl;

// //     int32_t h_d[4];

// //     h_d[0] = 0;
// //     h_d[1] = 0;
// //     h_d[2] = 0;
// //     h_d[3] = 0xFF;

// //     gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
// //     gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
// //     gpu_run(cudaMemcpy(c, h_d, sizeof(int32_t) * 4, cudaMemcpyDefault));

// //     {
// //         auto start = std::chrono::system_clock::now();

// //         kernel_gpuReduceNumeric<<<1024, 1024, 0, 0>>>(a, b, c, c2, c3, c4, N);

// //         gpu_run(cudaDeviceSynchronize());
        
// //         auto end   = std::chrono::system_clock::now();
// //         std::cout << "Thandwritten: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

// //     }
// //     gpu_run(cudaMemcpy(h_d, c, sizeof(int32_t)*4, cudaMemcpyDefault));

// //     for (size_t i = 0 ; i < 4 ; ++i) std::cout << h_d[i] << " "; std::cout << std::endl;

// //     int32_t   local_sum = 0;
// //     int32_t   local_max = 0; //FIXME: should be MAX_NEG_INT, but codegen currently sets it to zero

// //     bool      local_and = true ;
// //     bool      local_or  = false;
//     {

//         auto start = std::chrono::system_clock::now();

//         cpu_gpuSelectNumeric(h_a, h_b, h_d, N);

//         auto end   = std::chrono::system_clock::now();
//         std::cout << "Tcpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

//     }
// //     std::cout << local_sum << " " << local_max << " " << local_and << " " << local_or << std::endl;
// //     EXPECT_EQ(local_sum, h_c[0]);
// //     EXPECT_EQ(local_max, h_c[1]);
// //     EXPECT_EQ(local_and, !!(h_c[2] & 0xFF));
// //     EXPECT_EQ(local_or , !!(h_c[3] & 0xFF));
// //     EXPECT_EQ(local_sum, h_d[0]);
// //     EXPECT_EQ(local_max, h_d[1]);
// //     EXPECT_EQ(local_and, !!(h_d[2] & 0xFF));
// //     EXPECT_EQ(local_or , !!(h_d[3] & 0xFF));
// }

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


    // for (int i = 0 ; i < devCount ; ++i){
    //     CUdevice device;
    //     CUcontext context;
    //     gpu_run(cuDeviceGet(&device, i));
    //     gpu_run(cuCtxCreate(&context, 0, device));
    // }

    // char name[128];
    // gpu_run(cuDeviceGetName(name, 128, device));
    // std::cout << "Using CUDA Device [0]: " << name << "\n";

    // gpu_run(cuDeviceComputeCapability(&devMajor, &devMinor, device));
    // std::cout << "Device Compute Capability: " << devMajor << "." << devMinor << "\n";
    // if (devMajor < 2) {
    //     std::cout << "ERROR: Device 0 is not SM 2.0 or greater\n";
    //     EXPECT_TRUE(false);
    // }

    N = 1024*1024*256;
    
    gpu_run(cudaMalloc(&a, sizeof(int32_t)*N));
    gpu_run(cudaMalloc(&b, sizeof(double )*N));
    gpu_run(cudaMalloc(&c, sizeof(int32_t)*16));
    gpu_run(cudaMalloc(&d, sizeof(int32_t)*N));

    gpu_run(cudaMallocHost(&h_a, sizeof(int32_t)*N));
    gpu_run(cudaMallocHost(&h_b, sizeof(double )*N));
    gpu_run(cudaMallocHost(&h_c, sizeof(int32_t)*16));
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
        function = ctx->getKernel()[0];

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
    // void *KernelParams[] = {&a, &b, &N, &c, &c2};
    void *KernelParams[] = {&c, &c2, &a, &b, &N};

    {
        time_block t("Tgenerated: ");
        // Kernel launch
        launch(KernelParams);

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
        time_block t("Tcodegen: ");

        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        // Get kernel function
        function = ctx->getKernel()[0];
    }

    // Create driver context
    
    h_c[0] =  0;

    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(double ) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 1, cudaMemcpyDefault));
    // gpu_run(cudaMemcpy(d, h_d, sizeof(int32_t) * N, cudaMemcpyDefault));

    // Kernel parameters
    // void *KernelParams[] = {&a, &b, &N, &d, &c};
    void *KernelParams[] = {&d, &c, &a, &b, &N};

    {
        auto start = std::chrono::system_clock::now();
        // Kernel launch
        // gpu_run(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
        //                              blockSizeX, blockSizeY, blockSizeZ,
        //                              0, NULL, KernelParams, NULL));
        launch(KernelParams);


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
        std::cout << index << "->";
        gpu_run(cudaMemcpy( gpu_data, data + index, l_step * sizeof(T), cudaMemcpyDefault));
        index += l_step;

        std::cout << index << std::endl;

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

extern __shared__ int     int_shared  [];
extern __shared__ int64_t int64_shared[];

__host__ __device__ __forceinline__ uint32_t hasht(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

__global__ void build_subpartitions(const int32_t  * __restrict__ S_partitioned,
                                    const int32_t * __restrict__ gpart_offset,
                                    const int32_t * __restrict__ gpart_cnt,
                                          uint32_t * __restrict__ hj_index,
                                          int32_t  * __restrict__ hj_data){
    const size_t base     = gpart_offset[blockIdx.x & grid_parts_mask];
    const size_t cnt      = base + gpart_cnt[blockIdx.x & grid_parts_mask];

    int32_t * cnts = int_shared;

    // const int32_t prevwrapmask  = (1 << get_laneid()) - 1;
    // const int32_t lane_leq_mask = (prevwrapmask << 1) | 1;

    for (int i = threadIdx.x; i < partitions ; i += blockDim.x){
        atomicExch(cnts + i, 0);
    }

    if (threadIdx.x == 0) atomicExch(cnts + partitions, base);

    __syncthreads();

    for (size_t i = base + threadIdx.x; i < cnt ; i += blockDim.x){ //FIXME: loop until ___all___ lanes have finished!
        // vec4 tmp_probe = *reinterpret_cast<const vec4 *>(S_partitioned + i);
        // #pragma unroll
        // for (int k = 0 ; k < 4 ; ++k){
        int32_t  sid_probe = S_partitioned[i];

        uint32_t current   = hasht(sid_probe) & hj_mask;
        int32_t  bbucket   = (current >> 5) & partitions_mask;

        atomicAdd(cnts + bbucket, 1);
    }

    __syncthreads(); //if you manage to remove this line, change the loop below!

    for (int i = threadIdx.x; i < partitions ; i += blockDim.x){
        // atomicExch(cnts + i, atomicAdd(cnts + partitions, atomicExch(cnts[i], 0)));
        int32_t old_cnt = cnts[i];
        // assert(old_cnt <= 30 * 32); //FIXME: remove limitation
        int32_t tmp = atomicAdd(cnts + partitions, old_cnt); //use above line if you remove syncthreads!!!!
        cnts[i] = tmp;

        hj_index[((blockIdx.x & grid_parts_mask) * partitions + i) * 32    ] = tmp;
        hj_index[((blockIdx.x & grid_parts_mask) * partitions + i) * 32 + 1] = old_cnt;
    }

    __syncthreads(); //if you manage to remove this line, change the loop above!

    for (size_t i = base + threadIdx.x; i < cnt ; i += blockDim.x){ //FIXME: loop until ___all___ lanes have finished!
        // vec4 tmp_probe = *reinterpret_cast<const vec4 *>(S_partitioned + i);
        // #pragma unroll
        // for (int k = 0 ; k < 4 ; ++k){
        int32_t  sid_probe = S_partitioned[i];
        uint32_t current   = hasht(sid_probe) & hj_mask;
        int32_t  bbucket   = (current >> 5) & partitions_mask;

        hj_data[atomicAdd(cnts + bbucket, 1)] = sid_probe;
    }
}


__launch_bounds__(32, 1024) __global__ void build_buckets(      uint32_t * __restrict__ hj_index   ,
                              const int32_t  * __restrict__ hj_data_in ,
                                    int32_t  * __restrict__ hj_data_out,
                                    uint32_t * __restrict__ hj_index_cnt){
    uint32_t tid_base = 32 * blockIdx.x;
    // assert(blockDim.x == 32);
    // const int    laneid   =  threadIdx.x &  0x1F;

    // volatile int32_t  * d_buff = int_shared;
    uint32_t * mark_p = (uint32_t *) int_shared;//(uint32_t *) (d_buff + 32 * 32);
    uint32_t * mark_c = mark_p + 32;
    volatile uint32_t * mark_b = mark_c + max_chain;
    // uint32_t * mark_c = (uint32_t *) (mark_p + 32);

    // const int32_t prevwrapmask  = (1 << get_laneid()) - 1;
    // const int32_t lane_leq_mask = (prevwrapmask << 1) | 1;

    atomicExch(mark_p + threadIdx.x, 0);
    for (int j = 0 ; j < max_chain ; j += 32){
        atomicExch(mark_c + threadIdx.x + j, 0);
    }

    uint32_t tmp    = hj_index[tid_base + get_laneid()];

    uint32_t base   = __shfl(tmp, 0);
    uint32_t icnt   = __shfl(tmp, 1);

    int32_t x[32];

    for (int j = 1 ; j < 32 ; ++j){
        if (32 * (j - 1) + get_laneid() >= icnt) continue;
        x[j] = hj_data_in[base + 32 * (j - 1) + get_laneid()];
    }
    
    // #pragma unroll
    for (int j = 1 ; j < 32 ; ++j){
        if (32 * (j - 1) + get_laneid() >= icnt) continue;
        int32_t h = hasht(x[j]) & hj_mask & 0x1F;

        int32_t old = atomicAdd(mark_p + h, 1);
        // assert(old < max_chain);
        atomicAdd(mark_c + old, 1);
    }

    __threadfence();

    int32_t pre_cnt = 0;

    for (int j = 0 ; j < max_chain ; j += 32){
        int32_t cnt = mark_c[threadIdx.x + j];
        #pragma unroll
        for (int i = 1; i < 32; i <<= 1){
            int32_t tmp = __shfl_up(cnt, i);
            if (i <= get_laneid()) cnt += tmp;
        }

        mark_c[threadIdx.x + j] = cnt + pre_cnt - mark_c[threadIdx.x + j];

        pre_cnt += __shfl(cnt, 31);
    }

    int32_t max_bucket = 0;

    int32_t pcnt    = mark_p[get_laneid()];
    for (int j = 0 ; j < max_chain ; ++j){
        uint32_t mask = __ballot(j < pcnt);
        if (get_laneid() == 0) mark_b[j] = mask;
        if (mask) max_bucket = j;
    }

    __threadfence();
    
    // {
        // int32_t x[32];

        // for (int j = 1 ; j < 32 ; ++j){
        //     if (32 * (j - 1) + get_laneid() >= icnt) continue;
        //     x[j] = hj_data_in[base + 32 * (j - 1) + get_laneid()];
        // }
        
        // #pragma unroll
        for (int j = 1 ; j < 32 ; ++j){
            if (32 * (j - 1) + get_laneid() >= icnt) continue;
            int32_t h = hasht(x[j]) & hj_mask & 0x1F;

        // // #pragma unroll
        // for (int j = 1 ; j < 32 ; ++j){
        //     if (32 * (j - 1) + get_laneid() >= icnt) continue;
        //     // int32_t loc_base = __shfl(base, j);

        //     int32_t x = hj_data_in[base + 32 * (j - 1) + get_laneid()];

            // int32_t h = hasht(x) & hj_mask & 0x1F;

            uint32_t r = atomicSub(mark_p + h, 1) - 1;

            uint32_t index = mark_c[r];
            index += __popc(mark_b[r] & ((1 << h) - 1));
            index += base;

            hj_data_out[index] = x[j];
        }
    // }

    if (get_laneid() > 1) hj_index[tid_base + get_laneid()] = mark_b[get_laneid() - 2];

    // assert(max_bucket < 30);

    // int32_t j = 30;
    // while (j < max_bucket){
    //     uint32_t next;
    //     if (get_laneid() == 0) {
    //         next = atomicAdd(hj_index_cnt, 32);
    //         hj_index[tid_base + 1] = next;
    //     }
    //     tid_base = __shfl(next, 0);
    //     if (get_laneid() != 0) hj_index[tid_base + get_laneid()] = mark_b[get_laneid() - 1 + j];
    //     j += 31;
    // }
    // if (get_laneid() == 0) hj_index[tid_base + 1] = 0;
}

struct alignas(int64_t) hjt {
    int32_t key;
    int32_t val;

    __device__ hjt(){}
    __device__ hjt(const hjt& h): key(h.key), val(h.val){}
    __device__ hjt(const volatile hjt& h): key(h.key), val(h.val){}

    __device__ volatile hjt &operator=(const hjt& h) volatile{
        key = h.key;
        val = h.val;
        return *this;
    }
};

__global__ void probe(const   hjt       * __restrict__ R_partitioned,
                        const int32_t   * __restrict__ gpart_offset,
                        const int32_t   * __restrict__ gpart_cnt,
                        const uint32_t  * __restrict__ hj_index,
                        const int32_t   * __restrict__ hj_data ,
                              size_t    * __restrict__ res){
    const size_t base     = gpart_offset[blockIdx.x & grid_parts_mask];
    const size_t cnt      = base + gpart_cnt[blockIdx.x & grid_parts_mask];

    const size_t tid_base = (threadIdx.x & ~0x1F) + blockDim.x * (blockIdx.x >> g_d);
    // const int    laneid   =  threadIdx.x &  0x1F;


    // volatile int32_t  * buffer = int_shared;
    volatile hjt * d_buff = (hjt *) int64_shared;//buffer + 32 * partitions;
    uint32_t * mark_e = (uint32_t *) (d_buff + 32 * partitions);
    uint32_t * mark_s = (uint32_t *) (mark_e + partitions     );


    // const int32_t prevwrapmask  = (1 << get_laneid()) - 1;
    // const int32_t lane_leq_mask = (prevwrapmask << 1) | 1;

    if (threadIdx.x < 32) {
        for (int i = 0 ; i < partitions ; i += 32){
            atomicExch(mark_e + threadIdx.x + i, 0);
            atomicExch(mark_s + threadIdx.x + i, 0);
        }
    }

    __syncthreads();

    int32_t s = 0;

    for (size_t i = tid_base + base; i < cnt ; i += blockDim.x * (gridDim.x >> g_d)){ //FIXME: loop until ___all___ lanes have finished!
        // vec4 tmp_probe = *reinterpret_cast<const vec4 *>(sid2_ptr + i);
        // #pragma unroll
        // for (int k = 0 ; k < 4 ; ++k){
            hjt  sid_probe;
            uint32_t current = (uint32_t) -1;
            int32_t  bbucket = -1;

            if (i + get_laneid() < cnt){
                sid_probe = R_partitioned[i + get_laneid()];
                current   = hasht(sid_probe.key) & hj_mask;
                bbucket   = (current >> 5) & partitions_mask;
                current  &= ~0x1F;
            }

            while (__any(bbucket >= 0)){
                int32_t olde = 0;
                if (bbucket >= 0){
                    int32_t old     = atomicAdd(mark_s + bbucket, 1); //FIXME: May overflow
                    // assert(old < 100000);
                    atomicMin(mark_s + bbucket, 64);

                    if (old < 32){
                        d_buff[((bbucket + old) & partitions_mask) + (old * partitions)] = sid_probe;
                        __threadfence_block();
                        olde = atomicInc(((uint32_t *) (mark_e + bbucket)), 31);
                        bbucket = -1;
                    }
                }

                int32_t ready = __ballot(olde == 31);
                while (ready){
                    //find pivot
                    int32_t loc    = __ffs(ready) - 1;
                    int32_t pivot2 = __shfl(current, loc);
                    ready         &= ready - 1;

                    uint32_t tmp = hj_index[pivot2 | get_laneid()];

                    int32_t pivot = (pivot2 >> 5) & partitions_mask;
                    hjt probe = d_buff[((pivot + get_laneid()) & partitions_mask) + (get_laneid() * partitions)];

                    if (get_laneid() == 0) atomicExch(mark_s + pivot, 0);

                    int32_t prb_mask = 1 << (hasht(probe.key) & 0x1F);
                    int32_t leq_mask = prb_mask - 1;

                    uint32_t base   = __shfl(tmp, 0);

                    #pragma unroll
                    for (int j = 2 ; j < 32 ; ++j){
                        int32_t loc_mask = __shfl(tmp, j);

                        //Accessing everything "just in case" and then shuffling
                        //permits the compiler to hide the latency.
                        //Modify with care!
                        int32_t x = hj_data[base + get_laneid()];
                        
                        base += __popc(loc_mask);

                        x = __shfl(x, __popc(loc_mask & leq_mask));

                        if (loc_mask & prb_mask){
                            if (x == probe.key) s += probe.val;

                            // {
                            //     int32_t tmp = hasht(x) & hj_mask;
                            //     assert(tmp == (hasht(probe) & hj_mask));
                            // }
            
                        }
                    }
                }
            // }
        }
    }


    __syncthreads();

    if (threadIdx.x < 32){
        for (int pivot = 0 ; pivot < partitions ; ++pivot){
            int max_lane = atomicCAS(mark_e+pivot, 55, 55);

            if (max_lane == 0) continue;

            hjt probe  = (hjt) d_buff[((pivot + get_laneid()) & partitions_mask) + (get_laneid() * partitions)];

            uint32_t prob_base = __shfl((hasht(probe.key) & hj_mask & ~0x1F), 0);

            uint32_t tmp = hj_index[prob_base + get_laneid()];

            int32_t prb_mask = 1 << (hasht(probe.key) & 0x1F);
            int32_t leq_mask = prb_mask - 1;

            uint32_t base = __shfl(tmp, 0);

            #pragma unroll
            for (int j = 2 ; j < 32 ; ++j){
                int32_t loc_mask = __shfl(tmp, j);

                //Accessing everything "just in case" and then shuffling
                //permits the compiler to hide the latency.
                //Modify with care!
                int32_t x = hj_data[base + get_laneid()];
                
                base += __popc(loc_mask);

                x = __shfl(x, __popc(loc_mask & leq_mask));

                if (get_laneid() < max_lane){
                    if (loc_mask & prb_mask){
                        if (x == probe.key) s += probe.val;

                        // {
                        //     int32_t tmp = hasht(x) & hj_mask;
                        //     assert(tmp == (hasht(probe) & hj_mask));
                        // }
                    }
                }
            }
        }
    }

    // printf("%d\n", s);
    atomicAdd((unsigned long long int *) res, (unsigned long long int) s);
}


struct alignas(alignof(int64_t)) hj_bucket{
    int32_t next;
    int32_t val ;

    constexpr __host__ __device__ hj_bucket(int32_t next, int32_t value): next(next), val(value){}
};

__global__ void probeBucketChainingDevicePacked(const hjt       * __restrict__ probeInput,
                                                      size_t    * __restrict__ output,
                                                const int32_t   * __restrict__ first,
                                                const hj_bucket * __restrict__ next_w_values,
                                                      int                      d,
                                                      int                      N) {
    size_t out = 0;

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N ; idx += blockDim.x * gridDim.x){
        hjt      probe = probeInput[idx]; //read once, hint to the compiler that it should not care about aliasing

        uint32_t bucket = hasht(probe.key) % (1 << d);

        int32_t current = first[bucket];

        while (current >= 0) {
            hj_bucket tmp = next_w_values[current];
            current = tmp.next;

            if (tmp.val == probe.key) out += probe.val;
        }
    }
    atomicAdd(output, out);
}

struct gb_bucket{
    int32_t next;
    int32_t key ;
    int32_t sum ;
    int32_t hold;
};

__global__ void hashGroupBy(const int32_t   * __restrict__ a,
                            const int32_t   * __restrict__ b,

                                  int32_t   * __restrict__ first,
                                  gb_bucket * __restrict__ next ,
                                  int                      d,
                                  size_t                   N) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N ; idx += blockDim.x * gridDim.x){
        int32_t key = a[idx]; //read once, hint to the compiler that it should not care about aliasing
        int32_t val = b[idx];

        uint32_t bucket = hasht(key) % (1 << d);

        int32_t current = first[bucket];

        bool written = false;

        if (current == ((uint32_t) -1)){
            if (!written){
                next[idx].sum  = val;
                next[idx].key  = key;
                next[idx].next =  -1;

                written = true;
                __threadfence();
            }
            current = atomicCAS(&(first[bucket]), -1, idx);
        }

        if (current != ((uint32_t) -1)){
            while (true) {
                int32_t   next_bucket = next[current].next;

                if (next[current].key == key) {
                    atomicAdd(&(next[current].sum), val);
                    if (written) next[idx].next = idx;
                    break;
                }

                if (next_bucket == ((uint32_t) -1)){
                    if (!written){
                        next[idx].sum  = val;
                        next[idx].key  = key;
                        next[idx].next =  -1;

                        written = true;
                        __threadfence();
                    }
                    next_bucket = atomicCAS(&(next[current].next), -1, idx);
                    if (next_bucket == ((uint32_t) -1)) break;
                }

                current = next_bucket;
            }
        }
    }
}



__global__ void hashGroupBy(const int32_t   * __restrict__ a,
                            const int32_t   * __restrict__ b,
                                  int32_t   * __restrict__ cnt,
                                  int32_t   * __restrict__ first,
                                  gb_bucket * __restrict__ next ,
                                  int                      d,
                                  size_t                   N) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N ; idx += blockDim.x * gridDim.x){
        int32_t key = a[idx]; //read once, hint to the compiler that it should not care about aliasing
        int32_t val = b[idx];

        uint32_t bucket = hasht(key) % (1 << d);

        int32_t current = first[bucket];

        bool written = false;
        int32_t old_cnt;

        if (current == ((uint32_t) -1)){
            if (!written){
                old_cnt            = atomicAdd(cnt, 1);
                next[old_cnt].sum  = val;
                next[old_cnt].key  = key;
                next[old_cnt].next =  -1;

                written = true;
                __threadfence();
            }
            current = atomicCAS(&(first[bucket]), -1, old_cnt);
        }

        if (current != ((uint32_t) -1)){
            while (true) {
                int32_t   next_bucket = next[current].next;

                if (next[current].key == key) {
                    atomicAdd(&(next[current].sum), val);
                    if (written) next[old_cnt].next = old_cnt;
                    break;
                }

                if (next_bucket == ((uint32_t) -1)){
                    if (!written){
                        old_cnt            = atomicAdd(cnt, 1);
                        next[old_cnt].sum  = val;
                        next[old_cnt].key  = key;
                        next[old_cnt].next =  -1;

                        written = true;
                        __threadfence();
                    }
                    next_bucket = atomicCAS(&(next[current].next), -1, old_cnt);
                    if (next_bucket == ((uint32_t) -1)) break;
                }

                current = next_bucket;
            }
        }
    }
}


TEST_F(GPUOutputTest3, gpuHashGroupBy) {
    const char *testLabel = "gpuHashGroupBy";
    const char* planPath = "inputs/plans/gpu-group-by.json";

    GpuRawContext * ctx;
    std::vector<RawPipeline *> pipelines;
    {
        time_block t("Tcodegen: ");
        
        ctx                   = new GpuRawContext(testLabel);
        CatalogParser catalog = CatalogParser(catalogJSON);
        PlanExecutor exec     = PlanExecutor(planPath, catalog, ctx);
        
        ctx->compileAndLoad();

        // Get kernel function
        pipelines = ctx->getPipelines();
    }

    h_c[0] =  0;
    h_c[1] =  0;
    h_c[2] =  0;
    h_c[3] =  0;

    gpu_run(cudaFree(b));
    gpu_run(cudaFree(d));
    d = NULL;

    gpu_run(cudaMalloc(&b, sizeof(int32_t) * N));


    for (size_t i = 0 ; i < N ; ++i) h_a[i] %= 100;
    for (size_t i = 0 ; i < N ; ++i) ((int32_t *) h_b)[i] = 1;//rand();

    gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(a, h_a, sizeof(int32_t) * N, cudaMemcpyDefault));
    gpu_run(cudaMemcpy(b, h_b, sizeof(int32_t) * N, cudaMemcpyDefault));

    int32_t   * first;
    gb_bucket * next ;

    int d = 10;//1 + ((int32_t) ceil(log2(N)));
    std::cout << "d = " << d << std::endl;

    {
        time_block t("Tmallocs (init): ");
        gpu_run(cudaMalloc(&first, sizeof(int32_t  ) * (1 << d)));
        gpu_run(cudaMalloc(&next , sizeof(gb_bucket) * N       ));
    }
    {
        time_block t("Tinit (hj init): ");
        gpu_run(cudaMemset(first, -1, (1 << d) * sizeof(int32_t)));
    }


    {
        time_block t("TgroupBy - handwritten: ");

        hashGroupBy<<<128, 1024>>>(a, (int32_t *) b, first, next, d, N);
        gpu_run(cudaDeviceSynchronize());
    }
    {
        time_block t("Tinit (hj init): ");
        gpu_run(cudaMemset(first, -1, (1 << d) * sizeof(int32_t)));
    }


    {
        time_block t("TgroupBy - handwritten: ");

        hashGroupBy<<<128, 1024>>>(a, (int32_t *) b, c, first, next, d, N);
        gpu_run(cudaDeviceSynchronize());
    }
    size_t c_out;
    gpu_run(cudaMemcpy(&c_out, c, sizeof(size_t), cudaMemcpyDefault));

    std::cout << c_out << std::endl;
    // gpu_run(cudaMemcpy(c, h_c, sizeof(int32_t) * 4, cudaMemcpyDefault));
    // {
    //     time_block t("Tinit (hj init): ");
    //     gpu_run(cudaMemset(first, -1, (1 << d) * sizeof(int32_t)));
    // }


    gpu_run(cudaFree(first));
    gpu_run(cudaFree(next ));
    gpu_run(cudaFree(c    ));

    {
        time_block t("Topen: ");

        pipelines[0]->open();
    }


    first = pipelines[0]->getStateVar<int32_t *>(ctx, 0);
    next  = (gb_bucket *) pipelines[0]->getStateVar<void    *>(ctx, 1);

    c     = pipelines[0]->getStateVar<int32_t *>(ctx, 2);

    // struct state_t{
    //     int32_t   * first;
    //     gb_bucket * next ;
    //     int32_t   * cnt  ;
    // };

    // state_t st{first, next, c};
    {
        time_block t("TgroupBy - generated: ");
        // *((state_t *) (pipelines[0]->state)) = st;

        pipelines[0]->consume(N, b, a);
    }

    int32_t   * h_first;
    gb_bucket * h_next ;

    gpu_run(cudaMallocHost(&h_first, sizeof(int32_t  ) * (1 << d)));
    gpu_run(cudaMallocHost(&h_next , sizeof(gb_bucket) * N       ));

    gpu_run(cudaMemcpy(h_first, first, sizeof(int32_t  ) * (1 << d), cudaMemcpyDefault));
    gpu_run(cudaMemcpy(h_next , next , sizeof(gb_bucket) * N       , cudaMemcpyDefault));

    // size_t c_out;
    gpu_run(cudaMemcpy(&c_out, c, sizeof(int32_t), cudaMemcpyDefault));

    std::cout << c_out << std::endl;

    {
        time_block t("Tclose: ");

        pipelines[0]->close();
    }

    first = NULL;
    next  = NULL;
    c     = NULL;

    std::unordered_map<int32_t, int32_t> groups;
    {
        time_block t("Tnaive-cpu: ");
        for (size_t i = 0 ; i < N ; ++i) {
            auto t = groups.emplace(h_a[i], ((int32_t *) h_b)[i]);
            if (!t.second) t.first->second += ((int32_t *) h_b)[i];
        }
    }

    std::vector<std::pair<int32_t, int32_t>> groups2(groups.begin(), groups.end());
    std::sort(groups2.begin(), groups2.end());

    // for (const auto &t: groups2){
    //     std::cout << t.first << " " << t.second << std::endl;
    // }

    std::vector<std::pair<int32_t, int32_t>> groups3;

    for (size_t i = 0 ; i < (1 << d); ++i){
        int32_t k = h_first[i];
        while (k != -1){
            gb_bucket tmp = h_next[k];
            groups3.emplace_back(tmp.key, tmp.sum);
            k = tmp.next;
        }
    }

    std::sort(groups3.begin(), groups3.end());

    // for (const auto &t: groups3){
    //     std::cout << t.first << " " << t.second << std::endl;
    // }

    EXPECT_EQ(groups2, groups3);
}
