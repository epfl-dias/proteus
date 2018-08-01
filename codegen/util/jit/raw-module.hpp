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

#ifndef RAW_MODULE_HPP_
#define RAW_MODULE_HPP_

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"

#include "util/raw-context.hpp"

class RawModule {
protected:
    static llvm::IRBuilder<>  * TheBuilder  ;

    llvm::Module              * TheModule   ;
    const std::string           pipName     ;
    const RawContext          * context     ;
public:
    RawModule(RawContext * context, std::string pipName = "pip");
    virtual ~RawModule(){}

    virtual void compileAndLoad() = 0;

    Module * getModule() const;

    virtual void * getCompiledFunction(Function * f) const = 0;

protected:
    static void init(LLVMContext &llvmContext);
};

#endif /* RAW_MODULE_HPP_ */