/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef RAW_CPU_PIPELINE_HPP_
#define RAW_CPU_PIPELINE_HPP_

#include "pipeline.hpp"
#include "util/jit/cpu-module.hpp"

class CpuPipelineGen : public PipelineGen {
 protected:
  std::unique_ptr<CpuModule> module;

 private:
  CpuPipelineGen(Context *context, std::string pipName = "pip",
                 PipelineGen *copyStateFrom = nullptr);

  friend class CpuPipelineGenFactory;

 public:
  virtual void compileAndLoad();

  virtual llvm::Module *getModule() const { return module->getModule(); }
  virtual const llvm::DataLayout &getDataLayout() const;

 public:
  virtual void *getCompiledFunction(llvm::Function *f);
};

class CpuPipelineGenFactory : public PipelineGenFactory {
 protected:
  CpuPipelineGenFactory() {}

 public:
  static PipelineGenFactory &getInstance() {
    static CpuPipelineGenFactory instance;
    return instance;
  }

  PipelineGen *create(Context *context, std::string pipName,
                      PipelineGen *copyStateFrom) {
    return new CpuPipelineGen(context, pipName, copyStateFrom);
  }
};

#endif /* RAW_CPU_PIPELINE_HPP_ */
