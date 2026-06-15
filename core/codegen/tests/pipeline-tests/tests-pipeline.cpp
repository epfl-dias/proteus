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

#include <codegen/context/context.hpp>
#include <codegen/jit/cpu-pipeline.hpp>
#include <codegen/jit/pipeline.hpp>
#include <codegen/test/pipeline-environment.hpp>
#include <codegen/test/pipeline-test-utils.hpp>
#include <platform/memory/memory-manager.hpp>

using namespace codegen;

::testing::Environment* const pools_env =
    ::testing::AddGlobalTestEnvironment(new PipelineTestEnvironment);

class PipelineTest : public ::testing::Test {
 protected:
  using PipelineGenPtr = DummyPipelineContext::PipelineGenPtr;

  void SetUp() final {
    // Create a context
    testContext = std::make_unique<DummyPipelineContext>(testModuleName);

    // Create an instance of PipelineGenFactory
    auto& pipelineGenFactory = CpuPipelineGenFactory::getInstance();

    // Create a CpuPipelineGen
    auto testName =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    cpuPipelineGen = pipelineGenFactory.create(testContext.get(), testName);

    // Set the CpuPipelineGen as a PipelineGen in the context
    testContext->setPipelineGen(cpuPipelineGen);

    // Create a few helper types in advance
    auto& llvmContext = cpuPipelineGen->getModule()->getContext();
    voidType = llvm::Type::getVoidTy(llvmContext);
    i32Type = llvm::Type::getInt32Ty(llvmContext);
  }

 public:
  const std::string testModuleName = "test";

  std::unique_ptr<DummyPipelineContext> testContext;
  PipelineGenPtr cpuPipelineGen = nullptr;

  llvm::Type* voidType{nullptr};
  llvm::Type* i32Type{nullptr};
};

// Make and run an empty pipeline
TEST_F(PipelineTest, SmokeTest) {
  // Set up the main pipeline functions
  // Note: this is one-shot function, so you can't create more than one
  // pipeline using PipelineGen
  cpuPipelineGen->prepare();

  /**
   * Pipeline code generation
   */

  // Generate return from the main pipeline function
  cpuPipelineGen->getBuilder()->CreateRetVoid();

  /**
   * Pipeline compilation and execution
   */

  // Start compiling
  cpuPipelineGen->compileAndLoad();

  // Wait for the compiled function and set up the pipeline
  auto pipeline = cpuPipelineGen->getPipeline();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Make and run a pipeline which calls the printi function
TEST_F(PipelineTest, CallFunction) {
  // Set up main pipeline functions
  cpuPipelineGen->prepare();

  /**
   * Pipeline code generation
   */

  // Generate a print function argument
  llvm::Value* argument = testContext->createInt32(5);

  // Generate a print function call
  llvm::Function* printInt = cpuPipelineGen->getFunction("printi");
  testContext->gen_call(printInt, {argument});

  // Generate return from the main pipeline function
  cpuPipelineGen->getBuilder()->CreateRetVoid();

  /**
   * Pipeline compilation and execution
   */

  // Start compiling
  cpuPipelineGen->compileAndLoad();

  // Wait for the compiled function and set up the pipeline
  auto pipeline = cpuPipelineGen->getPipeline();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);
  pipeline->consume();
  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Make and run a pipeline with an additional input parameter which will be
// passed to the printi function. This is not a common way to pass a value to
// the Pipeline in the runtime, you can see an example of usage in
// mem-move-device.cpp
// More common way is in the next test case.
TEST_F(PipelineTest, AppendParameter) {
  // Append an additional argument to the main pipeline function
  // Note: the argument type should be a pointer type
  auto argumentId =
      cpuPipelineGen->appendParameter(llvm::PointerType::get(i32Type, 0));

  // Set up main pipeline functions
  cpuPipelineGen->prepare();

  /**
   * Pipeline code generation
   */

  // Fetch an argument using the id received from the appendParameter function
  llvm::Value* argument = cpuPipelineGen->getArgument(argumentId);

  // Load the payload of the argument (remember, the argument itself is a
  // pointer)
  llvm::Value* argumentPayload =
      cpuPipelineGen->getBuilder()->CreateLoad(i32Type, argument);

  // Generate a print function call
  llvm::Function* printInt = cpuPipelineGen->getFunction("printi");
  testContext->gen_call(printInt, {argumentPayload});

  // Generate return from the main pipeline function
  cpuPipelineGen->getBuilder()->CreateRetVoid();

  /**
   * Pipeline compilation and execution
   */

  // Start compiling
  cpuPipelineGen->compileAndLoad();

  // Wait for the compiled function and set up the pipeline
  auto pipeline = cpuPipelineGen->getPipeline();

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);

  // Create a payload for out main pipeline function and execute the function
  int32_t pipelinePayload = 10;
  pipeline->consume(&pipelinePayload);

  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Make and run a pipeline with an additional state variable which will be
// passed to the printi function. This is a common way to pass an argument to
// the pipeline in the runtime.
TEST_F(PipelineTest, AppendStateVar) {
  // Append an additional argument to the main pipeline function
  // appendStateVar without init/deinit functions are the simplest way to
  // initialize a state var, but you need to keep in mind that the state var is
  // just a pointer, so you will need to manually initialize memory for the
  // pointer and set it via the setStateVar function call.
  // Note: the argument type must be a pointer type
  auto stateVar =
      cpuPipelineGen->appendStateVar(llvm::PointerType::get(i32Type, 0));

  // Set up main pipeline functions
  cpuPipelineGen->prepare();

  /**
   * Pipeline code generation
   */

  // Fetch an argument using the id received from the appendParameter function
  llvm::Value* argument = cpuPipelineGen->getStateVar(stateVar);

  // Load the payload of the argument (remember, the argument itself is a
  // pointer)
  llvm::Value* argumentPayload =
      cpuPipelineGen->getBuilder()->CreateLoad(i32Type, argument);

  // Generate a print function call
  llvm::Function* printInt = cpuPipelineGen->getFunction("printi");
  testContext->gen_call(printInt, {argumentPayload});

  // Generate return from the main pipeline function
  cpuPipelineGen->getBuilder()->CreateRetVoid();

  /**
   * Pipeline compilation and execution
   */

  // Start compiling
  cpuPipelineGen->compileAndLoad();

  // Wait for the compiled function and set up the pipeline
  auto pipeline = cpuPipelineGen->getPipeline();

  // Initialize the actual memory for the state var
  int32_t pipelinePayload = 15;
  pipeline->setStateVar<int32_t*>(stateVar, &pipelinePayload);

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);

  // Create a payload for out main pipeline function and execute the function
  pipeline->consume();

  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}

// Make and run a pipeline with an additional state variable which will be
// passed to the printi function. Specify init and deinit functions for this
// variable. This is a common way to pass an argument to the pipeline in the
// runtime.
TEST_F(PipelineTest, AppendAndAllocateStateVar) {
  llvm::IntegerType* type =
      llvm::Type::getInt32Ty(testContext->getLLVMContext());
  // Append an additional argument to the main pipeline function
  // Note: the argument type must be a pointer type
  // The second argument is a function that will be called for the
  // initialization of the state variable The third argument is a function that
  // will be called at the end of the pipeline.
  // In this test case, we explicitly provide init/deinit functions where we
  // initialize a memory for the state var. This is useful in situations where
  // you want to initialize memory behind the state var inside the generated
  // code.
  auto stateVar = cpuPipelineGen->appendStateVar(
      llvm::PointerType::getUnqual(type),
      [=](llvm::Value* pip) -> llvm::Value* {
        // Be careful to use ParallelContext::allocateStateVar, but not
        // Context::allocateStateVar. Note that the
        // ParallelContext::allocateStateVar redirects calls to the {Cpu,
        // Gpu}PipelineGen, so it is safe to use it
        auto mem = cpuPipelineGen->allocateStateVar(type);
        testContext->getBuilder()->CreateStore(testContext->createInt32(100),
                                               mem);
        return mem;
      },
      [=](llvm::Value* pip, llvm::Value* state_var) {
        cpuPipelineGen->deallocateStateVar(state_var);
      });

  // Set up main pipeline functions
  cpuPipelineGen->prepare();

  /**
   * Pipeline code generation
   */

  // Fetch an argument using the id received from the appendParameter function
  llvm::Value* argument_mem = cpuPipelineGen->getStateVar(stateVar);

  // Load the payload of the argument (remember, the argument itself is a
  // pointer)
  llvm::Value* argumentPayload = cpuPipelineGen->getBuilder()->CreateLoad(
      argument_mem->getType()->getPointerElementType(), argument_mem);

  // Generate a print function call
  llvm::Function* printInt = cpuPipelineGen->getFunction("printi");
  testContext->gen_call(printInt, {argumentPayload});

  // Generate return from the main pipeline function
  cpuPipelineGen->getBuilder()->CreateRetVoid();

  /**
   * Pipeline compilation and execution
   */

  // Start compiling
  cpuPipelineGen->compileAndLoad();

  // Wait for the compiled function and set up the pipeline
  auto pipeline = cpuPipelineGen->getPipeline();

  //  int32_t pipelinePayload = 15;
  //  pipeline->setStateVar<int32_t*>(stateVar, &pipelinePayload);

  // Allocate a session for the pipeline
  auto* session = MemoryManager::mallocPinned(sizeof(size_t));

  // Actually execute the pipeline
  pipeline->open(session);

  // Create a payload for out main pipeline function and execute the function
  pipeline->consume();

  pipeline->close();

  // Deallocate the session
  MemoryManager::freePinned(session);
}
