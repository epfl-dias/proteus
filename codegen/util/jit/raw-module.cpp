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

#include "util/jit/raw-module.hpp"

using namespace llvm;

IRBuilder<> * RawModule::TheBuilder = nullptr;

RawModule::RawModule(RawContext * context, std::string pipName): 
    TheModule(new Module(pipName, context->getLLVMContext())),
    pipName(pipName), 
    context(context){

    if (TheBuilder == nullptr) init(context->getLLVMContext());
}

void RawModule::init(LLVMContext &llvmContext){
    assert(TheBuilder == nullptr && "Module already initialized");
    TheBuilder = new IRBuilder<>(llvmContext);
}

Module * RawModule::getModule() const{
    return TheModule;
}
