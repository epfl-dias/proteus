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
#ifndef PROTEUS_CPU_PIPELINE_HPP
#define PROTEUS_CPU_PIPELINE_HPP

#include <codegen/jit/cpu-module.hpp>
#include <codegen/jit/cpu-pipeline.hpp>
#include <codegen/jit/pipeline.hpp>

class CpuPipelineGen : public PipelineGen {
 protected:
  std::unique_ptr<CpuModule> module;

 private:
  CpuPipelineGen(Context *context, std::string pipName = "pip",
                 PipelineGen *copyStateFrom = nullptr);

  friend class CpuPipelineGenFactory;

 public:
  void compileAndLoad() override;

  llvm::Module *getModule() const override { return module->getModule(); }
  const llvm::DataLayout &getDataLayout() const override;

 public:
  void *getCompiledFunction(llvm::Function *f) override;
};

class CpuPipelineGenFactory : public PipelineGenFactory {
 protected:
  CpuPipelineGenFactory() {}

  void registerFunctions(PipelineGen *pipelineGen) override;

 public:
  static PipelineGenFactory &getInstance() {
    static CpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) override;
};

#endif  // PROTEUS_CPU_PIPELINE_HPP
