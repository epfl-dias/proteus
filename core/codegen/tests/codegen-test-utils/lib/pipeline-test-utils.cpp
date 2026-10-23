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
#include <cassert>
#include <codegen/test/pipeline-test-utils.hpp>

using namespace codegen;

DummyPipelineContext::DummyPipelineContext(const std::string& moduleName)
    : Context(moduleName) {}

llvm::Module* DummyPipelineContext::getModule() const {
  assert(currentPipeline);
  return currentPipeline->getModule();
}

llvm::IRBuilder<>* DummyPipelineContext::getBuilder() const {
  assert(currentPipeline);
  return currentPipeline->getBuilder();
}

void DummyPipelineContext::setPipelineGen(PipelineGenPtr pipelineGen) {
  currentPipeline = pipelineGen;
}
