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
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <memory>

#include "codegen/context/context.hpp"
#include "codegen/test/context-test-utlis.hpp"

using namespace codegen;

class ContextTest : public ::testing::Test {
 protected:
  void SetUp() final {
    DummyTestContext::InitJIT();
    testContext = std::make_unique<DummyTestContext>(testModuleName);
  }

 public:
  const std::string testModuleName = "test";

  std::unique_ptr<DummyTestContext> testContext;
};

// Just make an empty void function
TEST_F(ContextTest, SmokeTest) {
  auto* builder = testContext->getBuilder();
  testContext->prepareTestFunction("emptyFunc", testContext->voidType, {});

  builder->CreateRetVoid();

  testContext->finishTestFunction(/*dump=*/true);
  auto* justWorkingFunc =
      testContext->getCompiledTestFunction<void (*)()>("emptyFunc");
  justWorkingFunc();
}

// Make a function that returns a constant number
TEST_F(ContextTest, Constant) {
  auto* builder = testContext->getBuilder();
  testContext->prepareTestFunction("constantFunc", testContext->i32_type, {});

  constexpr int32_t number = 5;
  auto* value = testContext->createInt32(number);
  builder->CreateRet(value);

  testContext->finishTestFunction(/*dump=*/true);
  auto* constantFunc =
      testContext->getCompiledTestFunction<int32_t (*)()>("constantFunc");
  ASSERT_EQ(constantFunc(), number);
}

// Make a function that creates a struct similar to the ReferenceStruct and
// populate it with values from the reference, then returns the sum of all
// fields in the struct
struct ReferenceStruct {
  int32_t a;
  int32_t b;
  int32_t c;
};
TEST_F(ContextTest, ConstructStruct) {
  auto* builder = testContext->getBuilder();
  testContext->prepareTestFunction("constructStructFunc", testContext->i32_type,
                                   {});

  constexpr ReferenceStruct reference{.a = 1, .b = 2, .c = 3};
  constexpr int32_t referenceSum = reference.a + reference.b + reference.c;

  // Collect values which will be in a new struct
  std::vector<llvm::Value*> valuesList;
  valuesList.push_back(testContext->createInt32(reference.a));
  valuesList.push_back(testContext->createInt32(reference.b));
  valuesList.push_back(testContext->createInt32(reference.c));

  // Create a struct from the values and allocate the memory for it
  auto* llvmStruct =
      testContext->constructStruct(valuesList.begin(), valuesList.end());

  // Store the struct in the memory
  auto llvmStructValueMemory =
      testContext->toMem(llvmStruct, testContext->createFalse(), "struct");

  // Get elements from the struct (using the memory location of the struct)
  auto* a = testContext->getStructElem(llvmStructValueMemory.mem, 0);
  auto* b = testContext->getStructElem(llvmStructValueMemory.mem, 1);
  auto* c = testContext->getStructElem(llvmStructValueMemory.mem, 2);

  // Sum all elements
  llvm::Value* acc = testContext->createInt32(0);
  acc = builder->CreateAdd(acc, a);
  acc = builder->CreateAdd(acc, b);
  acc = builder->CreateAdd(acc, c);

  builder->CreateRet(acc);

  testContext->finishTestFunction(/*dump=*/true);
  auto* constructStructFunc =
      testContext->getCompiledTestFunction<int32_t (*)()>(
          "constructStructFunc");
  ASSERT_EQ(constructStructFunc(), referenceSum);
}

// Allocate an array of int32_t, populate it using values from the
// referenceArray, read a value on the referenceTestOffset index from the array
TEST_F(ContextTest, Array) {
  auto* builder = testContext->getBuilder();
  testContext->prepareTestFunction("arrayFunc", testContext->i32_type, {});

  const std::vector<int32_t> referenceArray{1, 2, 3, 4, 5};
  llvm::Value* arraySize = testContext->createSizeT(referenceArray.size());
  llvm::Value* arrayMem = testContext->CreateEntryBlockAlloca(
      "array", testContext->i32_type, arraySize);

  for (size_t idx = 0; idx < referenceArray.size(); ++idx) {
    llvm::Value* valueToInsert = testContext->createInt32(referenceArray[idx]);
    llvm::Value* offset = testContext->createSizeT(idx);
    llvm::Value* arrayElementPtr =
        testContext->getArrayElemMem(arrayMem, offset);
    builder->CreateStore(valueToInsert, arrayElementPtr);
  }

  const size_t referenceTestOffset = 2;
  llvm::Value* testOffset = testContext->createSizeT(referenceTestOffset);
  llvm::Value* arrayTestElement =
      testContext->getArrayElem(arrayMem, testOffset);

  builder->CreateRet(arrayTestElement);

  testContext->finishTestFunction(/*dump=*/true);
  auto* arrayFunc =
      testContext->getCompiledTestFunction<int32_t (*)()>("arrayFunc");
  ASSERT_EQ(arrayFunc(), referenceArray[referenceTestOffset]);
}

// Make a function that returns 1 if the input value is greater than 0, -1 if
// the input value is less than 0 and 0 if the value equals to 0
TEST_F(ContextTest, If) {
  auto* builder = testContext->getBuilder();
  auto* function = testContext->prepareTestFunction(
      "testZeroFunc", testContext->i32_type, {testContext->i32_type});

  llvm::Value* argument = function->arg_begin();
  llvm::Value* zero = testContext->createInt32(0);

  // ICmpSGT == Integer Compare Signed Greater Than
  llvm::Value* isGreaterThanZero = builder->CreateICmpSGT(argument, zero);

  // if (argument > zero)
  {
    auto ifGreaterZero = testContext->gen_if({isGreaterThanZero});

    auto thenGreaterZero = std::move(ifGreaterZero)([&]() {  // NOLINT
      llvm::Value* result = testContext->createInt32(1);
      builder->CreateRet(result);
    });
  }

  // ICmpSGT == Integer Compare Signed Less Than
  llvm::Value* isLessThanZero = builder->CreateICmpSLT(argument, zero);

  // if (argument < zero)
  {
    auto ifLessZero = testContext->gen_if({isLessThanZero});

    auto thenLessZero = std::move(ifLessZero)([&]() {  // NOLINT
      llvm::Value* result = testContext->createInt32(-1);
      builder->CreateRet(result);
    });
  }

  llvm::Value* result = testContext->createInt32(0);
  builder->CreateRet(result);

  testContext->finishTestFunction(/*dump=*/true);
  auto* greaterZero =
      testContext->getCompiledTestFunction<int32_t (*)(int32_t)>(
          "testZeroFunc");
  ASSERT_EQ(greaterZero(5), 1);
  ASSERT_EQ(greaterZero(-5), -1);
  ASSERT_EQ(greaterZero(0), 0);
}

// Make a function that makes a sum of elements from 1 to n where n is a
// function argument
TEST_F(ContextTest, While) {
  auto* builder = testContext->getBuilder();
  auto* function = testContext->prepareTestFunction(
      "sum", testContext->i32_type, {testContext->i32_type});

  // The following code is equivalent to:
  // auto* currentNumber = new int(1);
  // auto* accumulator = new int(0);
  // while (*currentNumber < n) {
  //  *accumulator += *currentNumber;
  //  ++(*currentNumber);
  // }
  // return *accumulator
  // The reason to store currentNumber and accumulator in the heap is that
  // we can access them in the condition and loop body without phi node
  // (https://llvm.org/docs/LangRef.html#phi-instruction)

  llvm::Value* n = function->arg_begin();

  // Allocate a memory for the currentNumber and store 1 in it
  llvm::Value* currentNumberPtr = testContext->CreateEntryBlockAlloca(
      "currentNumberPtr", testContext->i32_type);
  testContext->getBuilder()->CreateStore(testContext->createInt32(1),
                                         currentNumberPtr);

  // Create while block with the function that returns condition to continue the
  // loop
  auto whileLessOrEqualsN = testContext->gen_while([&]() -> ProteusValue {
    // Load currentNumber from the heap
    llvm::Value* currentNumber = builder->CreateLoad(
        currentNumberPtr->getType()->getPointerElementType(), currentNumberPtr,
        "currentNumber");

    // ICmpSLE == Integer Compare Signed Less or Equals
    llvm::Value* isLessOrEqualsN = builder->CreateICmpSLE(currentNumber, n);

    return {isLessOrEqualsN, testContext->createFalse()};
  });

  // Allocate a memory for the accumulator and store 0 in it
  llvm::Value* accumulatorPtr = testContext->CreateEntryBlockAlloca(
      "accumulatorPtr", testContext->i32_type);
  builder->CreateStore(testContext->createInt32(0), accumulatorPtr);

  // Call while block with a loop body as an argument
  std::move(whileLessOrEqualsN)([&](llvm::BranchInst*) {
    // Load currentNumber and accumulator from the heap
    llvm::Value* currentNumber = builder->CreateLoad(
        currentNumberPtr->getType()->getPointerElementType(), currentNumberPtr);
    llvm::Value* accumulator = builder->CreateLoad(
        accumulatorPtr->getType()->getPointerElementType(), accumulatorPtr);

    // Update currentNumber and accumulator
    accumulator = builder->CreateAdd(accumulator, currentNumber);
    currentNumber =
        builder->CreateAdd(currentNumber, testContext->createInt32(1));

    // Store updated currentNumber and accumulator in the heap
    builder->CreateStore(currentNumber, currentNumberPtr);
    builder->CreateStore(accumulator, accumulatorPtr);
  });

  // Load final accumulator value
  llvm::Value* accumulator = builder->CreateLoad(
      accumulatorPtr->getType()->getPointerElementType(), accumulatorPtr);

  builder->CreateRet(accumulator);

  testContext->finishTestFunction(/*dump=*/true);
  auto* sum = testContext->getCompiledTestFunction<int32_t (*)(int32_t)>("sum");
  ASSERT_EQ(sum(1), 1);
  ASSERT_EQ(sum(5), 15);
}
