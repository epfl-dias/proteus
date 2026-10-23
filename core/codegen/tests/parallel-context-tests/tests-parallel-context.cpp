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
#include <codegen/test/pipeline-environment.hpp>
#include <memory>
#include <platform/memory/memory-manager.hpp>

::testing::Environment* const pools_env =
    ::testing::AddGlobalTestEnvironment(new PipelineTestEnvironment);

class ParallelContextTest : public ::testing::Test {
 protected:
  void SetUp() final {
    // Create a context
    parallelContext = std::unique_ptr<ParallelContext>(
        ParallelContext::prepareParallelContext(testModuleName,
                                                /*gpuRoot=*/false));
  }

 public:
  const std::string testModuleName = "test";

  std::unique_ptr<ParallelContext> parallelContext;
};

// Use ParallelContext to generate and empty pipeline, compile and run it
TEST_F(ParallelContextTest, SmokeTest) {
  // During the construction of the ParallelContext, the first PipelineGen was
  // added as the generator of the pipelines in the ParallelContext.
  // ParallelContext::setGlobalFunction calls prepare on the current pipeline
  // generator, that set up the main pipeline functions.
  // See more in the pipeline tests
  parallelContext->setGlobalFunction(/*leaf=*/true);

  // Move the latest PipelineGen from the generators to the pipelines and call
  // PipelineGen::compileAndLoad. Also prepare the ParallelContext for the
  // further generations.
  // Note: we do not need to call Builder's CreateRetVoid, because it is done by
  // compileAndLoad
  parallelContext->compileAndLoad();

  // For each leaf PipelineGen, wait for the compiled function and set up the
  // pipeline
  auto pipelines = parallelContext->getPipelines();

  ASSERT_FALSE(pipelines.empty());
  auto& pipeline = pipelines.front();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Generate a pipeline using ParallelContext with an additional state variable
// with init/deinit functions. Generate printi function call with the specified
// state variable as an argument. Compile and run the pipeline.
TEST_F(ParallelContextTest, CallFunction) {
  // See PipelineTest::AppendAndAllocateStateVar
  // Note that you must appendStateVar before calling setGlobalFunction
  llvm::IntegerType* type =
      llvm::Type::getInt32Ty(parallelContext->getLLVMContext());
  auto stateVar = parallelContext->appendStateVar(
      llvm::PointerType::getUnqual(type),
      [=](llvm::Value* pip) -> llvm::Value* {
        // Be careful to use ParallelContext::allocateStateVar, but not
        // Context::allocateStateVar. Note that the
        // ParallelContext::allocateStateVar redirects calls to the {Cpu,
        // Gpu}PipelineGen, so it is safe to use it
        auto mem = parallelContext->allocateStateVar(type);
        parallelContext->getBuilder()->CreateStore(
            parallelContext->createInt32(100), mem);
        return mem;
      },
      [=](llvm::Value* pip, llvm::Value* state_var) {
        parallelContext->deallocateStateVar(state_var);
      });

  parallelContext->setGlobalFunction(/*leaf=*/true);

  // Fetch an argument using the id received from the appendParameter function
  llvm::Value* argumentMem = parallelContext->getStateVar(stateVar);

  // Load the payload of the argument (remember, the argument itself is a
  // pointer)
  llvm::Value* argumentPayload = parallelContext->getBuilder()->CreateLoad(
      argumentMem->getType()->getPointerElementType(), argumentMem);

  // Generate a print function call
  llvm::Function* printInt = parallelContext->getFunction("printi");
  parallelContext->gen_call(printInt, {argumentPayload});

  // Move the latest PipelineGen from the generators to the pipelines and call
  // PipelineGen::compileAndLoad. Also prepare the ParallelContext for the
  // further generations.
  // Note: we do not need to call Builder's CreateRetVoid, because it is done by
  // compileAndLoad
  parallelContext->compileAndLoad();

  // For each leaf PipelineGen, wait for the compiled function and set up the
  // pipeline
  auto pipelines = parallelContext->getPipelines();

  ASSERT_FALSE(pipelines.empty());
  auto& pipeline = pipelines.front();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Generate two pipelines, one is chained after another
TEST_F(ParallelContextTest, ChainedPipelines) {
  // Set up the second pipeline as a non-leaf pipeline
  // (this means that the pipeline will be excluded from the
  // ParallelContext::getPipelines() function)
  parallelContext->setGlobalFunction(/*leaf=*/false);

  // The second pipeline calls printi function with the argument = 2
  llvm::Value* second = parallelContext->createInt32(2);
  llvm::Function* printInt = parallelContext->getFunction("printi");
  parallelContext->gen_call(printInt, {second});

  // Move the latest PipelineGen from the generators to the pipelines and call
  // PipelineGen::compileAndLoad. Also prepare the ParallelContext for the
  // further generations.
  parallelContext->compileAndLoad();

  // Take the last pipeline and chain it after the current pipeline
  auto* second_pip = parallelContext->removeLatestPipeline();
  parallelContext->setChainedPipeline(second_pip);

  // Set up the first pipeline as a leaf pipeline (it will be returned from the
  // ParallelContext::getPipelines() function)
  parallelContext->setGlobalFunction(true);

  // The first pipeline calls printi function with the argument = 1
  llvm::Value* first = parallelContext->createInt32(1);
  llvm::Function* printIntSecond = parallelContext->getFunction("printi");
  parallelContext->gen_call(printIntSecond, {first});

  // Similarly move the first pipeline from the generators to the pipelines
  parallelContext->compileAndLoad();

  // For each leaf PipelineGen (in this case, the PipelineGen for the first
  // pipeline), wait for the compiled function and set up the pipeline
  auto pipelines = parallelContext->getPipelines();
  ASSERT_EQ(1, pipelines.size());
  auto& pipeline = pipelines.front();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Generate two pipelines, one is chained after another
// Create a variable in the first pipeline, print it, multiply it by 2 and pass
// to the next pipeline for printing
TEST_F(ParallelContextTest, ChainedPipelinesWithArguments) {
  llvm::IntegerType* type =
      llvm::Type::getInt32Ty(parallelContext->getLLVMContext());

  // Firstly, define the second pipeline in the chain
  // The reason for it is that we need to provide this second pipeline as an
  // copyStateFrom argument to the ParallelContext::pushPipeline to start the
  // first pipeline codegen
  size_t secondPipArgumentId = parallelContext->appendParameter(type);

  // This pipeline is non-leaf, as we will call it directly from the first
  // pipeline
  parallelContext->setGlobalFunction(/*leaf=*/false);

  // Get the first argument of the pipeline and print it
  llvm::Value* secondPipArgument =
      parallelContext->getArgument(secondPipArgumentId);
  llvm::Function* printInt = parallelContext->getFunction("printi");
  parallelContext->gen_call(printInt, {secondPipArgument});

  // Save the current PipelineGen which is used for the second pipeline
  // generation
  auto* secondPip = parallelContext->getCurrentPipeline();

  // Move the latest PipelineGen from the generators to the pipelines and call
  // PipelineGen::compileAndLoad
  parallelContext->popPipeline();

  // Prepare for the first pipeline generation. Pass second_pip as an
  // copyStateFrom argument, so ParallelContext will store the state of the
  // second pipeline as the state variable in the first pipeline
  parallelContext->pushPipeline(secondPip);

  // The first pipeline is leaf, so we can get it using
  // ParallelContext::getPipelines()
  parallelContext->setGlobalFunction(/*leaf=*/true);

  // Get the consume function of the subpipeline of the first pipeline which is
  // the second pipeline
  auto subPipelineConsumeFunction =
      parallelContext->getFunction("subpipeline_consume");
  auto subPipelineConsumeFunctionType =
      subPipelineConsumeFunction->getFunctionType();

  // The first pipeline generates the first value and print it
  llvm::Value* firstPipArgument = parallelContext->createInt32(500);
  llvm::Function* printIntSecond = parallelContext->getFunction("printi");
  parallelContext->gen_call(printIntSecond, {firstPipArgument});

  // Next, the first pipeline multiply it by 2 to pass it to the second pipeline
  llvm::Value* secondPipInput = parallelContext->getBuilder()->CreateMul(
      firstPipArgument, parallelContext->createInt32(2));

  // The first pipeline need to prepare arguments for the consume function. The
  // first argument will be our value, and the second will be the state of the
  // second pipeline
  std::vector<llvm::Value*> args{secondPipInput};
  auto subStateType = subPipelineConsumeFunctionType->getParamType(
      subPipelineConsumeFunctionType->getNumParams() - 1);
  llvm::Value* subStatePtr = parallelContext->getBuilder()->CreateBitCast(
      parallelContext->getSubStateVar(), subStateType);
  args.emplace_back(subStatePtr);

  // Call consume of the second pipeline from the first pipeline
  parallelContext->getBuilder()->CreateCall(subPipelineConsumeFunction, args);

  // Move the latest PipelineGen from the generators to the pipelines and call
  // PipelineGen::compileAndLoad. Also prepare the ParallelContext for the
  // further generations.
  parallelContext->compileAndLoad();

  // For each leaf PipelineGen (in this case, the PipelineGen for the first
  // pipeline), wait for the compiled function and set up the pipeline
  auto pipelines = parallelContext->getPipelines();
  ASSERT_EQ(1, pipelines.size());
  auto& pipeline = pipelines.front();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the chain of pipelines
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}
