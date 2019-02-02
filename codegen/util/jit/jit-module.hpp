/*
    Proteus -- High-performance query processing on heterogeneous hardware.

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

#ifndef MODULE_HPP_
#define MODULE_HPP_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "util/context.hpp"

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

  llvm::Module *getModule() const;

  virtual void *getCompiledFunction(llvm::Function *f) const = 0;

 protected:
  static void init(llvm::LLVMContext &llvmContext);
};

#endif /* MODULE_HPP_ */
