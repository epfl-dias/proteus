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

#ifndef PROTEUS_CONTEXT_TEST_UTLIS_HPP
#define PROTEUS_CONTEXT_TEST_UTLIS_HPP

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "codegen/context/context.hpp"

namespace codegen {

class DummyCPUModule;

/**
 * @class DummyTestContext
 * is used as a helper class ONLY for Context unit tests
 *
 * It contains minimal toolchain to use Context functions for compilation and
 * running code without other codebase dependencies
 */
class DummyTestContext : public Context {
 public:
  /**
   * Construct the base class and DummyCPUModule
   * @param moduleName name of the LLVM module (ModuleID, source_filename)
   */
  explicit DummyTestContext(const std::string& moduleName);

  ~DummyTestContext() override;

  /**
   * Empty placeholder for the Context base function
   */
  void prepareFunction(llvm::Function*) final;

  [[nodiscard]] llvm::Module* getModule() const final;

  [[nodiscard]] llvm::IRBuilder<>* getBuilder() const final;

  /**
   * Initialization of minimum LLVM modules just to test basic functions
   * @see PipelineGen::init()
   */
  static void InitJIT();

  /**
   * Only for unit tests purposes, register the function and set up an insert
   * point in the builder
   * @param name the name of the function
   * @param result the type of the function output
   * @param arguments the vector of types of arguments
   * @return pointer to the prepared function
   */
  llvm::Function* prepareTestFunction(
      llvm::StringRef name, llvm::Type* result,
      const std::vector<llvm::Type*>& arguments);

  /**
   * Only for unit tests purposes. Check the module for errors and add it to the
   * compilation layer.
   * @param dump flag to enable printing the LLVM IR of the module, can be
   * useful to explore what the functions in the tests are generating
   */
  void finishTestFunction(bool dump);

  /**
   * Only for unit tests purposes. Get the compiled function. To call this
   * function you should firstly call prepareTestFunction and finishTestFunction
   * @tparam F type of the pointer to the target function, e. g. char
   * (*)(int32_t) -- type of the function that has an int32_t argument and char
   * return type
   * @param name the name of the function (must be passed in advance to
   * prepareTestFunction)
   * @return pointer to the target function
   */
  template <typename F>
  F getCompiledTestFunction(llvm::StringRef name) {
    return reinterpret_cast<F>(getCompiledRaw(name));
  }

 private:
  std::unique_ptr<DummyCPUModule> module;

 public:
  llvm::Type* voidType;
  llvm::Type* i32_type;

 private:
  uintptr_t getCompiledRaw(llvm::StringRef name);
};

}  // namespace codegen

#endif  // PROTEUS_CONTEXT_TEST_UTLIS_HPP
