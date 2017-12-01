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

#ifndef RAW_CPU_MODULE_HPP_
#define RAW_CPU_MODULE_HPP_

#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"

#include "util/jit/raw-module.hpp"

class RawCpuModule: public RawModule {
protected:
    static llvm::LLVMTargetMachine                    * TheTargetMachine    ;
    static llvm::legacy::PassManager                    Passes              ;
    static llvm::PassManagerBuilder                     Builder             ;
    // static std::unique_ptr<llvm::legacy::FunctionPassManager>   FPasses         ;

protected:
    llvm::ExecutionEngine                             * TheExecutionEngine  ;
public:
    RawCpuModule(RawContext * context, std::string pipName = "pip");

    static void init();

    virtual void compileAndLoad();

    virtual void * getCompiledFunction(Function * f) const;

protected:
    virtual void optimizeModule(Module * M);
};

#endif /* RAW_CPU_MODULE_HPP_ */