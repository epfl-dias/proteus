/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "vector-plugin.hpp"

#include <variant>

std::vector<RecordAttribute *> getAttributeVector(
    const std::vector<
        std::pair<RecordAttribute *, std::shared_ptr<proteus_any_vector>>>
        &whichFields) {
  std::vector<RecordAttribute *> attrs;
  attrs.reserve(whichFields.size());
  for (const auto &val : whichFields) {
    attrs.emplace_back(new RecordAttribute{*val.first});
  }
  return attrs;
}

RecordType recordTypeFromDataVector(
    const std::vector<
        std::pair<RecordAttribute *, std::shared_ptr<proteus_any_vector>>>
        &whichFields) {
  return getAttributeVector(whichFields);
}

auto getVectorSize(const proteus_any_vector &v) {
  return std::visit([](auto &d) { return d.size(); }, v);
}

auto getVectorData(const proteus_any_vector &v) {
  return std::visit([](auto &d) { return (void *)d.data(); }, v);
}

VectorPlugin::VectorPlugin(
    ParallelContext *context,
    const std::vector<
        std::pair<RecordAttribute *, std::shared_ptr<proteus_any_vector>>>
        &whichFields)
    : BinaryBlockPlugin(context, whichFields.front().first->getRelationName(),
                        recordTypeFromDataVector(whichFields),
                        getAttributeVector(whichFields), false) {
  Nparts = 1;
  data_ptrs_ptr = new void **[whichFields.size()];
  data_size_ptr = new size_t;
  for (auto &f : whichFields) {
    data_ptrs_ptr[data.size()] = new void *;
    *data_ptrs_ptr[data.size()] = getVectorData(*f.second);
    data.emplace_back(f.second);
    assert(getVectorSize(*data.front()) == getVectorSize(*data.back()));
  }
  *data_size_ptr = getVectorSize(*data.front());
}

VectorPlugin::~VectorPlugin() {
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    delete data_ptrs_ptr[i];
  }
  delete[] data_ptrs_ptr;
  delete data_size_ptr;
}

llvm::Value *VectorPlugin::getDataPointersForFile(
    ParallelContext *context, size_t i, llvm::Value *session_ptr) const {
  return context->CastPtrToLlvmPtr(
      llvm::PointerType::getUnqual(llvm::ArrayType::get(
          llvm::PointerType::getUnqual(
              wantedFields[i]->getLLVMType(context->getLLVMContext())),
          Nparts)),
      data_ptrs_ptr[i]);
}

void VectorPlugin::freeDataPointersForFile(ParallelContext *context, size_t i,
                                           llvm::Value *v) const {
  // vectors have the lifetime of the plugin, no freeing needed
}

std::pair<llvm::Value *, llvm::Value *> VectorPlugin::getPartitionSizes(
    ParallelContext *context, llvm::Value *session_ptr) const {
  return {context->CastPtrToLlvmPtr(
              llvm::PointerType::getUnqual(
                  llvm::ArrayType::get(context->createSizeType(), Nparts)),
              data_size_ptr),
          context->createSizeT(*data_size_ptr)};
}

void VectorPlugin::freePartitionSizes(ParallelContext *context,
                                      llvm::Value *v) const {}
