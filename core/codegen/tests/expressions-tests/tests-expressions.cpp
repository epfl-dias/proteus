/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include <gtest/gtest.h>

#include <codegen/context/parallel-context.hpp>
#include <codegen/expressions/expressions.hpp>
#include <codegen/test/pipeline-environment.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

#include "codegen/test/context-test-utlis.hpp"
#include "codegen/test/expressions-test-utils.hpp"

using namespace codegen;

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new PipelineTestEnvironment);

/**
 * Simple fixture to show the usage of expressions with ordinary Context
 *
 * @see DummyTestContext
 */
class ExpressionsTestContext : public ::testing::Test {
 public:
  const std::string testModuleName = "test";

  // Flag to enable printing of the arithmetic expression in
  // TestExpressionVisitor
  const bool isPrintOperationsEnabled = false;

  std::unique_ptr<DummyTestContext> testContext;
  std::unique_ptr<ExprVisitor> visitor;

 protected:
  void SetUp() final {
    // Create a context
    DummyTestContext::InitJIT();
    testContext = std::make_unique<DummyTestContext>(testModuleName);

    // Create a visitor
    visitor = std::make_unique<TestExpressionVisitor>(testContext.get(),
                                                      isPrintOperationsEnabled);
  }
};

enum class ParallelContextRoot {
  kCPU,
  kGPU,
};

/**
 * Fixture to test basic usage of expression_t with ParallelContext with the
 * minimal functionality to create assertions inside generated code. Fixture is
 * parameterized with the execution mode (CPU or GPU), so the test will be
 * adapted to run both on CPU and GPU.
 *
 * @see SmokeTest to see the structure of the test
 */
class ExpressionsTestParallelContext
    : public ::testing::TestWithParam<ParallelContextRoot> {
 public:
  const std::string testModuleName = "test";

  // Flag to enable printing of the arithmetic expression in
  // TestExpressionVisitor
  const bool isPrintOperationsEnabled = false;

  std::unique_ptr<ParallelContext> parallelContext;
  std::unique_ptr<ExprVisitor> visitor;

  /**
   * Result is a mechanism to notify the host code about failed assertions in
   * the generated code. It works in the same way both on CPU and GPU.
   * After running the generated code, the result variable will contain 0 if no
   * asserts fail and 1 otherwise. The result variable can be accessed inside
   * the generated code via parallelContext->getArgument
   */
  using ResultT = int32_t;
  ResultT result{0};
  size_t resultId{0};

  /**
   * Assert that lhs equals to rhs
   * @param lhs llvm::Value containing integer value
   * @param rhs llvm::Value containing integer value
   * If lhs doesn't equal to rhs, then write to a result 1
   * @see result
   *
   * Important note: function is thread-safe on GPU as long as all
   * invocations are made in the same warp. The reason behind this is that
   * workerScopedAtomicXchg sync threads in one warp.
   */
  void LLVMAssertIEQ(llvm::Value *lhs, llvm::Value *rhs) const {
    auto comparisonResult =
        parallelContext->getBuilder()->CreateICmpNE(lhs, rhs);
    parallelContext->gen_if({comparisonResult, parallelContext->createFalse()})(
        [&] {
          llvm::Value *resultMem = parallelContext->getArgument(resultId);
          parallelContext->workerScopedAtomicXchg(
              resultMem, parallelContext->createInt32(1));
        });
  }

 protected:
  void SetUp() final {
    // Create a context
    parallelContext = std::unique_ptr<ParallelContext>(
        ParallelContext::prepareParallelContext(testModuleName, IsGpuRoot()));

    // Create a result variable
    resultId = parallelContext->appendParameter(
        llvm::Type::getInt32PtrTy(parallelContext->getLLVMContext()));

    // Create a visitor
    visitor = std::make_unique<TestExpressionVisitor>(parallelContext.get(),
                                                      isPrintOperationsEnabled);
  }

  void TearDown() final {
    // For each leaf PipelineGen, wait for the compiled function and set up the
    // pipeline
    auto pipelines = parallelContext->getPipelines();

    // Support only one leaf pipeline in the tests
    ASSERT_EQ(1, pipelines.size());
    auto &pipeline = pipelines.front();

    // Allocate a session for the pipeline
    auto *session = MemoryManager::mallocPinned(sizeof(size_t));

    // Initialize a memory for the result variable
    // The main reason to have the separate function is that
    // the initialization is non-trivial for GPU
    auto *resultMemory = InitializeResultMemory();

    pipeline->open(session);
    // Provide the result memory as a parameter of the pipeline
    pipeline->consume(resultMemory);
    pipeline->close();

    // Release the memory for the result variable
    ReleaseResultMemory(resultMemory);

    // Deallocate the session
    MemoryManager::freePinned(session);

    /**
     * Assert that all assertions in the generated code were successful
     * @see result
     */
    ASSERT_EQ(0, result);
  }

 private:
  /**
   * Initialize memory for the result variable to use inside the generated code
   *
   * For the CPU, the initialization is trivial, as we can use the result
   * field of the fixture class itself. For the GPU, we need to allocate an
   * memory for the variable on the device.
   * @return pointer to the ResultT memory that can be used in the generated
   * code
   * @see result
   */
  ResultT *InitializeResultMemory() {
    if (IsGpuRoot()) {
      void *gpuData = MemoryManager::mallocGpu(sizeof(ResultT));
      auto copyResult =
          cudaMemcpy(gpuData, &result, sizeof(ResultT), cudaMemcpyHostToDevice);
      assert(cudaSuccess == copyResult);
      return reinterpret_cast<ResultT *>(gpuData);
    } else {
      return &result;
    }
  }

  /**
   * Release memory of the result variable
   *
   * For the CPU, the function does nothing. For the GPU, we need to copy the
   * result value back to the class' field and deallocate memory on the device.
   * @param resultMemory pointer to the ResultT memory initialized in the
   * InitializeResultMemory
   * @see InitializeResultMemory
   */
  void ReleaseResultMemory(ResultT *resultMemory) {
    if (IsGpuRoot()) {
      auto copyResult = cudaMemcpy(&result, resultMemory, sizeof(ResultT),
                                   cudaMemcpyDeviceToHost);
      ASSERT_EQ(cudaSuccess, copyResult);
      MemoryManager::freeGpu(resultMemory);
    } else {
      // Do nothing, as resultMemory is the pointer to the class' field
    }
  }

  [[nodiscard]] static bool IsGpuRoot() {
    switch (GetParam()) {
      case ParallelContextRoot::kGPU:
        return true;
      case ParallelContextRoot::kCPU:
      default:
        return false;
    }
  }
};

// Generate a function that returns value generated from the constant expression
// using ordinary Context
TEST_F(ExpressionsTestContext, SmokeTest) {
  constexpr int32_t number = 5;

  auto *builder = testContext->getBuilder();
  testContext->prepareTestFunction("returnConstantFunc", testContext->i32_type,
                                   {});

  expression_t value = number;
  auto result = value.accept(*visitor);
  builder->CreateRet(result.value);

  testContext->finishTestFunction(/*dump=*/false);
  auto *returnConstantFunc =
      testContext->getCompiledTestFunction<int32_t (*)()>("returnConstantFunc");
  ASSERT_EQ(returnConstantFunc(), number);
}

void define_multiply(llvm::Module &module) {
  llvm::Type *int32Type = llvm::Type::getInt32Ty(module.getContext());
  llvm::FunctionType *funcType =
      llvm::FunctionType::get(int32Type, {int32Type, int32Type}, false);

  llvm::Function *mulFunction = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "multiply", module);

  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(module.getContext(), "entry", mulFunction);
  llvm::IRBuilder<> builder(entry);

  llvm::Function::arg_iterator argsValues = mulFunction->arg_begin();
  llvm::Value *a = argsValues++;
  llvm::Value *b = argsValues;

  llvm::Value *result = builder.CreateMul(a, b, "res");

  builder.CreateRet(result);
}

/* Calls a simple extern llvm function.
 *
 * This currently only runs for a plain Context (not for a ParallelContext).
 * This is because it doesn't work for GPU (see issue #118).
 */
TEST_F(ExpressionsTestContext, ExternExpressionMultiply) {
  auto *builder = testContext->getBuilder();
  testContext->prepareTestFunction("resultWrapper", testContext->i32_type, {});

  // register the function
  {
    auto &llvmContext = testContext->getLLVMContext();
    auto module = std::make_unique<llvm::Module>("externModule", llvmContext);
    define_multiply(*module.get());

    testContext->linkExternModule(std::move(module));
  }

  // call it
  auto expression = expressions::ExternExpression(
      "multiply", {expression_t{2}, expression_t{6}}, new IntType());
  auto number = expression.accept(*visitor);

  // set the wrapper
  builder->CreateRet(number.value);

  testContext->finishTestFunction(/*dump=*/false);
  auto *resultWrapper =
      testContext->getCompiledTestFunction<int32_t (*)()>("resultWrapper");

  // expect 2 * 6 == 12
  ASSERT_EQ(resultWrapper(), 12);
}

void define_divide(llvm::Module &module) {
  llvm::Type *int32Type = llvm::Type::getInt32Ty(module.getContext());
  llvm::FunctionType *funcType =
      llvm::FunctionType::get(int32Type, {int32Type, int32Type}, false);

  llvm::Function *mulFunction = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "divide", module);

  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(module.getContext(), "entry", mulFunction);
  llvm::IRBuilder<> builder(entry);

  llvm::Function::arg_iterator argsValues = mulFunction->arg_begin();
  llvm::Value *a = argsValues++;
  llvm::Value *b = argsValues;

  llvm::Value *result = builder.CreateSDiv(a, b, "res");

  builder.CreateRet(result);
}

/* Create a deep extern expression.
 *
 * This currently only runs for a plain Context (not for a ParallelContext).
 * This is because it doesn't work for GPU (see issue #118).
 */
TEST_F(ExpressionsTestContext, ExternExpressionComplex) {
  auto *builder = testContext->getBuilder();
  testContext->prepareTestFunction("resultWrapper", testContext->i32_type, {});

  // register the function
  {
    auto &llvmContext = testContext->getLLVMContext();
    auto module = std::make_unique<llvm::Module>("externModule", llvmContext);
    define_divide(*module.get());

    testContext->linkExternModule(std::move(module));
  }

  // 8 / -2 == -4
  auto expr0 = expressions::ExternExpression(
      "divide", {expression_t{8}, expression_t{-2}}, new IntType());
  // -4 / -1 == 4
  auto expr1 = expressions::ExternExpression(
      "divide", {expr0, expression_t{-1}}, new IntType());
  // (4 / 3) - 3 == -2
  auto expr2 = expressions::ExternExpression("divide", {expr1, expression_t{3}},
                                             new IntType()) -
               expression_t{3};

  auto number = expr2.accept(*visitor);

  // set the wrapper
  builder->CreateRet(number.value);

  testContext->finishTestFunction(/*dump=*/false);
  auto *resultWrapper =
      testContext->getCompiledTestFunction<int32_t (*)()>("resultWrapper");

  ASSERT_EQ(resultWrapper(), -2);
}

/*
 * All tests in the ExpressionsTestParallelContext fixture work the same way
 * After we set up a pipeline generation we generate expressions and assert that
 * the value after the visitor is the same as expected.
 *
 * The fixture's SetUp initialize the result value (see fixture's comments for
 * more details), then this value will be changed from 0 to 1 in case of failed
 * assertion. In the test TearDown() function there is an assert that the result
 * equals to zero.
 *
 * Important note: LLVMAssert function is thread-safe on GPU as long as all
 * invocations are made in the same warp. The reason behind this is that
 * workerScopedAtomicXchg sync threads in one warp.
 */
TEST_P(ExpressionsTestParallelContext, SmokeTest) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  expression_t value = 1;
  auto number = value.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(1));

  parallelContext->compileAndLoad();
}

// Generate a sum of two expressions
TEST_P(ExpressionsTestParallelContext, SumTest) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  expression_t lhs = 1;
  expression_t rhs = 2;
  auto expression = lhs + rhs;

  auto number = expression.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(3));

  parallelContext->compileAndLoad();
}

// Generate a subtraction of two expressions
TEST_P(ExpressionsTestParallelContext, SubTest) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  expression_t lhs = 3;
  expression_t rhs = 2;
  auto expression = lhs - rhs;

  auto number = expression.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(1));

  parallelContext->compileAndLoad();
}

// Generate an arithmetic expression
TEST_P(ExpressionsTestParallelContext, ComplexExpression1) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  auto expression = expression_t{20} - expression_t{5} * expression_t{3};

  auto number = expression.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(5));

  parallelContext->compileAndLoad();
}

// Generate an arithmetic expression with brackets
TEST_P(ExpressionsTestParallelContext, ComplexExpression2) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  auto expression = (expression_t{20} - expression_t{5}) * expression_t{3};

  auto number = expression.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(45));

  parallelContext->compileAndLoad();
}

// Generate an arithmetic expression with brackets
TEST_P(ExpressionsTestParallelContext, ComplexExpression3) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  auto expression = (expression_t{10} - expression_t{4}) / expression_t{3} +
                    expression_t{4} * expression_t{2};

  auto number = expression.accept(*visitor);
  LLVMAssertIEQ(number.value, parallelContext->createInt32(10));

  parallelContext->compileAndLoad();
}

/*
 * Generate a quadratic matrices multiplication of two matrices consisting of
 * expressions This test called dummy because it is running only on one thread
 * (it is important when running on GPU as the compiled pipeline will run on all
 * threads)
 */
TEST_P(ExpressionsTestParallelContext, MatrixMultiplicationDummy) {
  parallelContext->setGlobalFunction(/*leaf=*/true);

  // Initial matrices parameters
  constexpr size_t kQuadraticMatrixSize = 5;
  constexpr int32_t kMinValue = -100;
  constexpr int32_t kMaxValue = 100;

  // Generate two initial quadratic matrices
  auto matrixA = GenerateRandomQuadraticMatrix<int32_t>(kQuadraticMatrixSize,
                                                        kMinValue, kMaxValue);
  auto matrixB = GenerateRandomQuadraticMatrix<int32_t>(kQuadraticMatrixSize,
                                                        kMinValue, kMaxValue);
  // Multiply the matrices to get expected matrix
  auto matrixResult = MultiplyMatrices<int32_t>(matrixA, matrixB);

  // Convert initial matrices to the matrices of expressions
  auto expressionMatrixA = GetExpressionMatrixFromMatrix<int32_t>(matrixA);
  auto expressionMatrixB = GetExpressionMatrixFromMatrix<int32_t>(matrixB);

  // Multiply expressions matrices, so that we generate a matrix consisting of
  // the expression trees
  auto expressionMatrixResult =
      MultiplyMatrices<expression_t>(expressionMatrixA, expressionMatrixB);

  // Convert expected matrix to the expression matrix, so that we can compare
  // with it in the generated code
  auto targetExpressionMatrixResult =
      GetExpressionMatrixFromMatrix<int32_t>(matrixResult);

  // Generate multiplication on the first thread only, that allows us to execute
  // this code only on one GPU thread
  auto isFirstThread = parallelContext->getBuilder()->CreateICmpEQ(
      parallelContext->threadId(), parallelContext->createInt64(0));

  parallelContext->gen_if({isFirstThread, parallelContext->createFalse()})([&] {
    for (size_t rowIdx = 0; rowIdx < kQuadraticMatrixSize; ++rowIdx) {
      for (size_t columnIdx = 0; columnIdx < kQuadraticMatrixSize;
           ++columnIdx) {
        // Generate a multiplication
        auto result =
            expressionMatrixResult[rowIdx][columnIdx].accept(*visitor);
        auto expectedResult =
            targetExpressionMatrixResult[rowIdx][columnIdx].accept(*visitor);
        // Compare with the expected value
        LLVMAssertIEQ(result.value, expectedResult.value);
      }
    }
  });

  parallelContext->compileAndLoad();
}

INSTANTIATE_TEST_SUITE_P(ExpressionsTestParallelContextCpuGpu,
                         ExpressionsTestParallelContext,
                         testing::Values(ParallelContextRoot::kCPU,
                                         ParallelContextRoot::kGPU));

/*
 * That is the same test as MatrixMultiplicationDummy except that it runs only
 * on GPU in parallel (so that each GPU thread calculates its own result matrix
 * cell)
 * This test is inefficient and needed to show the functionality of expressions
 * on GPU.
 *
 * Content of the test is basically extracted parts of
 * ExpressionTestParallelContext SetUp and TearDown, the main difference in the
 * final matrix multiplication loop, so you can pay attention only to this.
 */
TEST(ParallelMatrixMultiplication, GpuRoot) {
  auto &topology = topology::getInstance();
  auto &gpus = topology.getGpus();
  auto execScope = gpus.at(0).set_on_scope();

  // Initialize variables to calculate the number of the equal cells in the
  // final matrix and expected matrix. gpuResult will be used in the generated
  // code.
  void *cpuResult = MemoryManager::mallocPinned(4);
  std::memset(cpuResult, 0, 4);
  void *gpuResult = MemoryManager::mallocGpu(4);
  auto copyResult = cudaMemcpy(gpuResult, cpuResult, 4, cudaMemcpyHostToDevice);
  ASSERT_EQ(cudaSuccess, copyResult);

  auto parallelContext = std::unique_ptr<ParallelContext>(
      ParallelContext::prepareParallelContext("test",
                                              /*gpuRoot=*/true));

  // Append a parameter to the pipeline, so we can provide pointer to the
  // gpuResult
  auto comparisonResultId = parallelContext->appendParameter(
      llvm::Type::getInt32PtrTy(parallelContext->getLLVMContext()));

  parallelContext->setGlobalFunction(/*leaf=*/true);

  constexpr size_t kQuadraticMatrixSize = 5;
  constexpr int32_t kMinValue = -100;
  constexpr int32_t kMaxValue = 100;

  // Checks that the number of cells in the matrix is less than the warp_size,
  // so that we can use workerScopeAtomicAdd with the gpuResult (all atomics
  // memory orders on GPU are relaxed, so we call warp_sync after each atomic
  // operation)
  ASSERT_LE(kQuadraticMatrixSize * kQuadraticMatrixSize, warp_size);

  auto matrixA = GenerateRandomQuadraticMatrix<int32_t>(kQuadraticMatrixSize,
                                                        kMinValue, kMaxValue);
  auto matrixB = GenerateRandomQuadraticMatrix<int32_t>(kQuadraticMatrixSize,
                                                        kMinValue, kMaxValue);
  auto matrixResult = MultiplyMatrices<int32_t>(matrixA, matrixB);

  auto expressionMatrixA = GetExpressionMatrixFromMatrix<int32_t>(matrixA);
  auto expressionMatrixB = GetExpressionMatrixFromMatrix<int32_t>(matrixB);
  auto expressionMatrixResult =
      MultiplyMatrices<expression_t>(expressionMatrixA, expressionMatrixB);

  auto targetExpressionMatrixResult =
      GetExpressionMatrixFromMatrix<int32_t>(matrixResult);

  llvm::Value *comparisonResultMem =
      parallelContext->getArgument(comparisonResultId);

  TestExpressionVisitor visitor(parallelContext.get(),
                                /*isPrintEnabled=*/false);
  for (size_t rowIdx = 0; rowIdx < kQuadraticMatrixSize; ++rowIdx) {
    for (size_t columnIdx = 0; columnIdx < kQuadraticMatrixSize; ++columnIdx) {
      // Distribute the cells according to the threadId
      size_t targetThreadId = rowIdx * kQuadraticMatrixSize + columnIdx;
      auto isTargetThread = parallelContext->getBuilder()->CreateICmpEQ(
          parallelContext->threadId(),
          parallelContext->createInt64(targetThreadId));

      // Process with the calculation if the id equals the target threadId
      // Note: this is very inefficient, as we basically generate a bunch of
      // ifs, so that every thread needs to go through all unnecessary
      // conditions, do not use this code anywhere. This is here just for the
      // sake of demonstration.
      parallelContext->gen_if(
          {isTargetThread, parallelContext->createFalse()})([&] {
        auto result = expressionMatrixResult[rowIdx][columnIdx].accept(visitor);
        auto expectedResult =
            targetExpressionMatrixResult[rowIdx][columnIdx].accept(visitor);
        auto equalResults = parallelContext->getBuilder()->CreateICmpEQ(
            result.value, expectedResult.value);

        // Increment atomic if the result equals the expected result
        parallelContext->gen_if({equalResults, parallelContext->createFalse()})(
            [&] {
              parallelContext->workerScopedAtomicAdd(
                  comparisonResultMem, parallelContext->createInt32(1));
            });
      });
    }
  }

  parallelContext->compileAndLoad();

  auto pipelines = parallelContext->getPipelines();
  ASSERT_EQ(1, pipelines.size());
  auto &pipeline = pipelines.front();

  auto *session = MemoryManager::mallocPinned(sizeof(size_t));

  pipeline->open(session);

  auto *comparisonResultGPU = reinterpret_cast<int32_t *>(gpuResult);
  pipeline->consume(comparisonResultGPU);

  pipeline->close();

  MemoryManager::freePinned(session);

  copyResult = cudaMemcpy(cpuResult, gpuResult, 4, cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaSuccess, copyResult);

  int32_t comparisonResult;
  std::memcpy(&comparisonResult, cpuResult, 4);

  MemoryManager::freeGpu(gpuResult);
  MemoryManager::freePinned(cpuResult);

  ASSERT_EQ(kQuadraticMatrixSize * kQuadraticMatrixSize, comparisonResult);
}
