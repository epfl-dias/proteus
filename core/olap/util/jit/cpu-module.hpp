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

#ifndef CPU_MODULE_HPP_
#define CPU_MODULE_HPP_

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"
#include "util/jit/jit-module.hpp"

class CpuModule : public JITModule {
 protected:
  llvm::ExecutionEngine *TheExecutionEngine;

 public:
  explicit CpuModule(Context *context, std::string pipName = "pip");

  void compileAndLoad() override;
  inline llvm::Module *getModule() const override {
    return JITModule::getModule();
  }

  [[nodiscard]] static const llvm::DataLayout &getDL();

  [[nodiscard]] const llvm::DataLayout &getDataLayout() const override {
    return CpuModule::getDL();
  }

  void *getCompiledFunction(llvm::Function *f) const override;
  void *getCompiledFunction(std::string str) const;

 protected:
  virtual void optimizeModule(llvm::Module *M);
};

#endif /* CPU_MODULE_HPP_ */
