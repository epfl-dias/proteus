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

#ifndef RAW_GPU_MODULE_HPP_
#define RAW_GPU_MODULE_HPP_

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

#include "common/gpu/gpu-common.hpp"

#include "util/jit/raw-module.hpp"

class RawGpuModule : public RawModule {
 protected:
  static llvm::LLVMTargetMachine *TheTargetMachine;
  static llvm::legacy::PassManager Passes;
  static llvm::PassManagerBuilder Builder;
  // static std::unique_ptr<llvm::legacy::FunctionPassManager>   FPasses ;

 protected:
  // llvm::ExecutionEngine                             * TheExecutionEngine  ;
  CUmodule *cudaModule;

 public:
  RawGpuModule(RawContext *context, std::string pipName = "pip");

  static void init();

  virtual void compileAndLoad();

  virtual void *getCompiledFunction(Function *f) const;

 protected:
  virtual void optimizeModule(Module *M);
};

#endif /* RAW_GPU_MODULE_HPP_ */