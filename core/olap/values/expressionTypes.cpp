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

#include <llvm/IR/IRBuilder.h>

bool recordComparator(RecordAttribute *x, RecordAttribute *y) {
  return (x->getAttrNo() < y->getAttrNo());
}

int RecordType::getIndex(RecordAttribute *x) const {
#ifndef NDEBUG
  size_t cnt = 0;
#endif
  int maybe_index = -1;
  int index = 0;
  for (const auto &attr : args) {
    if (x->getAttrName() == attr->getAttrName()) {
      if (x->getRelationName() == attr->getRelationName()) return index;
      maybe_index = index;
#ifndef NDEBUG
      ++cnt;
#endif
    }
    ++index;
  }
  assert(cnt <= 1 && "Multiple matches for attribute name, but none with rel");
  return maybe_index;
}

std::ostream &operator<<(std::ostream &o, const RecordAttribute &rec) {
  return (o << rec.getRelationName() << "." << rec.getAttrName() << ": "
            << rec.getOriginalType()->getType());
}

llvm::Type *BoolType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getInt1Ty(ctx);
}

llvm::Type *StringType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::StructType::get(
      ctx, {llvm::Type::getInt8PtrTy(ctx), llvm::Type::getInt32Ty(ctx)});
}

llvm::Type *DStringType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getInt32Ty(ctx);
}

llvm::Type *FloatType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getDoubleTy(ctx);
}

llvm::Type *IntType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getInt32Ty(ctx);
}

llvm::Type *Int64Type::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getInt64Ty(ctx);
}

llvm::Type *DateType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::Type::getInt64Ty(ctx);
}

llvm::Type *BlockType::getLLVMType(llvm::LLVMContext &ctx) const {
  return llvm::PointerType::get(getNestedType().getLLVMType(ctx), 0);
}

llvm::Type *RecordType::getLLVMType(llvm::LLVMContext &ctx) const {
  std::vector<llvm::Type *> body;
  for (const auto &attr : args) body.push_back(attr->getLLVMType(ctx));
  return llvm::StructType::get(ctx, body);
}
