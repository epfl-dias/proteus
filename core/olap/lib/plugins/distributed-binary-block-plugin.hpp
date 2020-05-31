/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef DISTRIBUTED_BINARY_BLOCK_PLUGIN_HPP_
#define DISTRIBUTED_BINARY_BLOCK_PLUGIN_HPP_

#include <memory>

#include "olap/plugins/binary-block-plugin.hpp"
#include "storage/storage-manager.hpp"

class DistributedBinaryBlockPlugin : public BinaryBlockPlugin {
 public:
  DistributedBinaryBlockPlugin(ParallelContext *const context,
                               string fnamePrefix, RecordType rec,
                               vector<RecordAttribute *> &whichFields)
      : BinaryBlockPlugin(context, fnamePrefix, rec, whichFields, false) {
    auto &llvmContext = context->getLLVMContext();

    // std::vector<Type *> parts_array;
    for (const auto &in : wantedFields) {
      string fileName = fnamePrefix + "." + in->getAttrName();

      const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
      size_t type_size = context->getSizeOf(llvm_type);
      fieldSizes.emplace_back(type_size);

      wantedFieldsFiles.emplace_back(StorageManager::getInstance().request(
          fileName, type_size, DISTRIBUTED));
      // Show the intent to the storage manager
      wantedFieldsFiles.back().registerIntent();
      // wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
      // FIXME: consider if address space should be global memory rather than
      // generic
      // Type * t = PointerType::get(((const PrimitiveType *)
      // tin)->getLLVMType(llvmContext), /* address space */ 0);

      // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

      if (in->getOriginalType()->getTypeID() == DSTRING) {
        // fetch the dictionary
        void *dict = StorageManager::getInstance().getDictionaryOf(fileName);
        ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
      }
    }

    finalize_data(context);
  }
};

#endif /* DISTRIBUTED_BINARY_BLOCK_PLUGIN_HPP_ */
