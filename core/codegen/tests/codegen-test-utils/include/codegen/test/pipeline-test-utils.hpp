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
#ifndef PROTEUS_PIPELINE_TEST_UTILS_HPP
#define PROTEUS_PIPELINE_TEST_UTILS_HPP

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <codegen/context/context.hpp>
#include <codegen/jit/pipeline.hpp>
#include <string>

namespace codegen {

/**
 * @class DummyPipelineContext
 * is used as a helper class ONLY for Pipeline/PipelineGen unit tests
 *
 * It contains minimal toolchain to use PipelineGen functions for compilation.
 * It should be used in pair with PipelineGen similar to ParallelContext.
 * @see ParallelContext
 */
class DummyPipelineContext final : public Context {
 public:
  using PipelineGenPtr = PipelineGen*;

  /**
   * Construct the base class.
   * @param moduleName name of the LLVM module (ModuleID, source_filename)
   * @note Do not construct CPUModule as it is now the
   * responsibility of PipelineGen.
   */
  explicit DummyPipelineContext(const std::string& moduleName);

  ~DummyPipelineContext() final = default;

  /**
   * Empty placeholder for the Context base function
   */
  void prepareFunction(llvm::Function*) final {}

  [[nodiscard]] llvm::Module* getModule() const final;

  [[nodiscard]] llvm::IRBuilder<>* getBuilder() const final;

  /**
   * Set up a PipelineGen, so the Context will use its llvm::Module and
   * llvm::Builder
   *
   * This functionality is implemented in ParallelContext via generators class
   * field. Generators are filled from the PipelineGen factories. This function
   * is a simplified version of this functionality.
   * @param pipelineGen pointer to the PipelineGen that will be used in this
   * Context.
   *
   * @see ParallelContext::pushPipeline(PipelineGen*)
   */
  void setPipelineGen(PipelineGenPtr pipelineGen);

 private:
  PipelineGenPtr currentPipeline = nullptr;
};

}  // namespace codegen

#endif  // PROTEUS_PIPELINE_TEST_UTILS_HPP
