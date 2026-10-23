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
#ifndef PROTEUS_JIT_MODULE_HPP
#define PROTEUS_JIT_MODULE_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <codegen/context/context.hpp>

// globals initialized in jit-module.cpp
// can be set by binaries/libraries that link to libcodegen.
// e.g. based on command line flags
extern bool print_generated_code;
extern bool insert_preopt_debug_info;
extern bool insert_postopt_debug_info;
extern bool dump_compiled_object_files;

class JITModule {
 protected:
  static llvm::IRBuilder<> *TheBuilder;

  llvm::Module *TheModule;
  const std::string pipName;
  const Context *context;

 public:
  JITModule(Context *context, std::string pipName = "pip");
  virtual ~JITModule() {}

  virtual void compileAndLoad() = 0;

  virtual llvm::Module *getModule() const;
  virtual const llvm::DataLayout &getDataLayout() const {
    return getModule()->getDataLayout();
  }

  virtual void *getCompiledFunction(llvm::Function *f) const = 0;

 protected:
  static void init(llvm::LLVMContext &llvmContext);
};

#endif  // PROTEUS_JIT_MODULE_HPP
