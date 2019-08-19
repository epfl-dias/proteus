/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#include "plugins/aeolus-plugin.hpp"

#include "communication/comm-manager.hpp"
#include "expressions/expressions-hasher.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

using namespace llvm;

AeolusPlugin::AeolusPlugin(ParallelContext *const context, string fnamePrefix,
                           RecordType rec,
                           vector<RecordAttribute *> &whichFields,
                           string pgType)
    : ScanToBlockSMPlugin(context, fnamePrefix, rec, whichFields, false),
      pgType(pgType) {
  if (wantedFields.size() == 0) {
    string error_msg{"[ScanToBlockSMPlugin: ] Invalid number of fields"};
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  LLVMContext &llvmContext = context->getLLVMContext();

  size_t pos_path = fnamePrefix.find_last_of("/");
  size_t pos_suffix = fnamePrefix.find_last_of(".csv");
  if (pos_path == std::string::npos) pos_path = 0;

  std::string aeolus_rel_name = "tpcc_" + fnamePrefix.substr(pos_path + 1);
  aeolus_rel_name = aeolus_rel_name.substr(0, aeolus_rel_name.length() - 4);

  if (pgType.compare("block-cow") == 0) {
    time_block t("TpgBlockCOW: ");

    auto &txn_storage = storage::Schema::getInstance();
    storage::ColumnStore *tbl = nullptr;

    std::cout << "Finding Table: " << aeolus_rel_name << std::endl;

    for (auto &tb : txn_storage.getTables()) {
      std::cout << "TableI: " << tb->name << std::endl;
      if (aeolus_rel_name.compare(tb->name) == 0) {
        tbl = (storage::ColumnStore *)tb;
        std::cout << "TableMATCH: " << tb->name << std::endl;
        std::cout << "TableMATCH: " << tbl->name << std::endl;
        break;
      }
    }
    assert(tbl != nullptr);

    uint64_t num_records = tbl->getNumRecords();

    std::cout << "# Records " << num_records << std::endl;

    for (const auto &in : wantedFields) {
      const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
      size_t type_size = context->getSizeOf(llvm_type);

      // wantedFieldsFiles.emplace_back(StorageManager::getOrLoadFile(fileName,
      // type_size, PAGEABLE));

      for (auto &c : tbl->getColumns()) {
        if (c->name.compare(in->getAttrName()) == 0) {
          const std::vector<storage::mem_chunk *> d = c->get_data();

          std::vector<mem_file> mfiles{d.size()};

          for (size_t i = 0; i < mfiles.size(); ++i) {
            mfiles[i].data = d[i]->data;

            if (d[i]->size >= (c->elem_size * num_records)) {
              mfiles[i].size = d[i]->size;
            } else {
              // to be fixed but for now, Aeolus doesnt expand beyond single
              // chunk so it will be ok.
              assert(false && "FIX ME");
            }
          }
          wantedFieldsFiles.emplace_back(mfiles);
          break;
        }
      }

      // wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
      // FIXME: consider if address space should be global memory rather than
      // generic
      // Type * t = PointerType::get(((const PrimitiveType *)
      // tin)->getLLVMType(llvmContext), /* address space */ 0);

      // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

      if (in->getOriginalType()->getTypeID() == DSTRING) {
        // fetch the dictionary
        string fileName = fnamePrefix + "." + in->getAttrName();
        void *dict = StorageManager::getDictionaryOf(fileName);
        ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
      }
    }

    std::cout << "[AEOLUS PLUGIN DONE]" << std::endl;

  } else {
    assert(false && "Not Implemented");
  }

  finalize_data();
}

extern "C" {
Plugin *createBlockCowPlugin(ParallelContext *context, std::string fnamePrefix,
                             RecordType rec,
                             std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields, "block-cow");
}

Plugin *createBlockSnapshotPlugin(ParallelContext *context,
                                  std::string fnamePrefix, RecordType rec,
                                  std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields,
                          "block-snapshot");
}

Plugin *createBlockRemotePlugin(ParallelContext *context,
                                std::string fnamePrefix, RecordType rec,
                                std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields,
                          "block-remote");
}
}
