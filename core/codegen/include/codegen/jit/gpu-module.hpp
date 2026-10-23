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
#ifndef PROTEUS_GPU_MODULE_HPP
#define PROTEUS_GPU_MODULE_HPP

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/Function.h>
#include <llvm/Target/TargetMachine.h>

#include <codegen/jit/jit-module.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <string>

class GpuModule : public JITModule {
 protected:
  static llvm::LLVMTargetMachine *TheTargetMachine;
  llvm::orc::ResourceTrackerSP resourceTracker;

 public:
  GpuModule(Context *context, std::string pipName = "pip");
  ~GpuModule() override;

  static void init();

  void compileAndLoad() override;

  void *getCompiledFunction(llvm::Function *f) const override;

  virtual void markToAvoidInteralizeFunction(std::string func);

 protected:
  std::set<std::string> preserveFromInternalization;
};

#endif  // PROTEUS_GPU_MODULE_HPP
