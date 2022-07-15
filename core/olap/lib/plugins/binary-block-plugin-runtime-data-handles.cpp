/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#include <olap/plugins/binary-block-plugin-runtime-data-handles.hpp>

namespace proteus {
namespace olap_plugins {

namespace detail {
void **getDataPointerForFile_runtime(size_t i, const char *relName,
                                     const char *attrName, void *session,
                                     BinaryBlockPluginRuntimeDataHandles *pg) {
  return pg->getDataPointerForFile_runtime(i, relName, attrName, session);
}

void freeDataPointerForFile_runtime(void **inn,
                                    BinaryBlockPluginRuntimeDataHandles *pg) {
  pg->freeDataPointerForFile_runtime(inn);
}

void *getSession() { return nullptr; }
void releaseSession(void *session) {}

int64_t *getNumOfTuplesPerPartition_runtime(
    const char *relName, void *session,
    BinaryBlockPluginRuntimeDataHandles *pg) {
  return pg->getNumOfTuplesPerPartition_runtime(relName, session);
}

void freeNumOfTuplesPerPartition_runtime(
    int64_t *inn, BinaryBlockPluginRuntimeDataHandles *pg) {
  pg->freeNumOfTuplesPerPartition_runtime(inn);
}
}  // namespace detail

llvm::Value *BinaryBlockPluginRuntimeDataHandles::getSession(
    ParallelContext *context) const {
  return context->gen_call(
      ::proteus::olap_plugins::detail::getSession, {},
      ::llvm::Type::getInt8PtrTy(context->getLLVMContext()));
}

void BinaryBlockPluginRuntimeDataHandles::releaseSession(
    ParallelContext *context, llvm::Value *session_ptr) const {
  context->gen_call(::proteus::olap_plugins::detail::releaseSession,
                    {session_ptr});
}

llvm::Value *BinaryBlockPluginRuntimeDataHandles::getDataPointersForFile(
    ParallelContext *context, size_t i, llvm::Value *session_ptr) const {
  ::llvm::LLVMContext &llvmContext = context->getLLVMContext();

  ::llvm::Type *char8ptr = ::llvm::Type::getInt8PtrTy(llvmContext);

  ::llvm::Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this), char8ptr);

  ::llvm::Value *N_parts_ptr = context->gen_call(
      ::proteus::olap_plugins::detail::getDataPointerForFile_runtime,
      {context->createSizeT(i),
       context->CreateGlobalString(wantedFields[i]->getRelationName().c_str()),
       context->CreateGlobalString(wantedFields[i]->getAttrName().c_str()),
       session_ptr, this_ptr},
      ::llvm::PointerType::getUnqual(char8ptr));

  auto data_type = ::llvm::PointerType::getUnqual(::llvm::ArrayType::get(
      RecordAttribute{*(wantedFields[i]), true}.getLLVMType(
          context->getLLVMContext()),
      Nparts));

  return context->getBuilder()->CreatePointerCast(N_parts_ptr, data_type);
  // return BinaryBlockPlugin::getDataPointersForFile(i);
}

void BinaryBlockPluginRuntimeDataHandles::freeDataPointersForFile(
    ParallelContext *context, size_t i, ::llvm::Value *v) const {
  ::llvm::LLVMContext &llvmContext = context->getLLVMContext();
  auto data_type =
      ::llvm::PointerType::getUnqual(::llvm::Type::getInt8PtrTy(llvmContext));
  auto casted = context->getBuilder()->CreatePointerCast(v, data_type);
  ::llvm::Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      ::llvm::Type::getInt8PtrTy(llvmContext));
  context->gen_call(
      ::proteus::olap_plugins::detail::freeDataPointerForFile_runtime,
      {casted, this_ptr});
}

std::pair<llvm::Value *, llvm::Value *>
BinaryBlockPluginRuntimeDataHandles::getPartitionSizes(
    ParallelContext *context, llvm::Value *session_ptr) const {
  ::llvm::IRBuilder<> *Builder = context->getBuilder();

  ::llvm::IntegerType *sizeType = context->createSizeType();

  ::llvm::Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      ::llvm::Type::getInt8PtrTy(context->getLLVMContext()));

  ::llvm::Value *N_parts_ptr = Builder->CreatePointerCast(
      context->gen_call(
          proteus::olap_plugins::detail::getNumOfTuplesPerPartition_runtime,
          {context->CreateGlobalString(fnamePrefix.c_str()), session_ptr,
           this_ptr}),
      ::llvm::PointerType::getUnqual(::llvm::ArrayType::get(sizeType, Nparts)));

  ::llvm::Value *max_pack_size = ::llvm::ConstantInt::get(sizeType, 0);
  for (size_t i = 0; i < Nparts; ++i) {
    auto sv = Builder->CreateInBoundsGEP(
        N_parts_ptr->getType()->getNonOpaquePointerElementType(), N_parts_ptr,
        {context->createSizeT(0), context->createSizeT(i)});
    auto v = Builder->CreateLoad(sv->getType()->getPointerElementType(), sv);
    auto cond = Builder->CreateICmpUGT(max_pack_size, v);
    max_pack_size = Builder->CreateSelect(cond, max_pack_size, v);
  }

  return {N_parts_ptr, max_pack_size};
  // return BinaryBlockPlugin::getPartitionSizes();
}

void BinaryBlockPluginRuntimeDataHandles::freePartitionSizes(
    ParallelContext *context, ::llvm::Value *v) const {
  ::llvm::Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      ::llvm::Type::getInt8PtrTy(context->getLLVMContext()));
  context->gen_call(
      ::proteus::olap_plugins::detail::freeNumOfTuplesPerPartition_runtime,
      {v, this_ptr});
}

}  // namespace olap_plugins
}  // namespace proteus
