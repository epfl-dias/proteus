/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_PROJECT_RECORD_HPP
#define PROTEUS_PROJECT_RECORD_HPP

#include <llvm/IR/IRBuilder.h>

#include <olap/values/expressionTypes.hpp>

llvm::Value *projectArg(const RecordType *type, llvm::Value *record,
                        RecordAttribute *attr,
                        llvm::IRBuilder<> *const Builder);

#endif /* PROTEUS_PROJECT_RECORD_HPP */
