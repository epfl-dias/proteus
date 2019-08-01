/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "values/expressionTypes.hpp"

bool recordComparator(RecordAttribute *x, RecordAttribute *y) {
  return (x->getAttrNo() < y->getAttrNo());
}

llvm::Value *RecordType::projectArg(llvm::Value *record, RecordAttribute *attr,
                                    llvm::IRBuilder<> *const Builder) const {
  if (!(record->getType()->isStructTy())) return nullptr;
  if (!(((llvm::StructType *)record->getType())
            ->isLayoutIdentical(
                (llvm::StructType *)getLLVMType(record->getContext()))))
    return nullptr;
  int index = getIndex(attr);
  if (index < 0) return nullptr;
  return Builder->CreateExtractValue(record, index);
}

int RecordType::getIndex(RecordAttribute *x) const {
  int index = 0;
  for (const auto &attr : args) {
    if (x->getAttrName() == attr->getAttrName()) return index;
    ++index;
  }
  return -1;
}
