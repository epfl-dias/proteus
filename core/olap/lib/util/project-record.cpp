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

#include "project-record.hpp"

llvm::Value *projectArg(const RecordType *type, llvm::Value *record,
                        RecordAttribute *attr,
                        llvm::IRBuilder<> *const Builder) {
  if (!(record->getType()->isStructTy())) return nullptr;
  if (!(((llvm::StructType *)record->getType())
            ->isLayoutIdentical(
                (llvm::StructType *)type->getLLVMType(record->getContext()))))
    return nullptr;
  int index = type->getIndex(attr);
  if (index < 0) return nullptr;
  return Builder->CreateExtractValue(record, index);
}
