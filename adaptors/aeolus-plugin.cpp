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

#include "aeolus-plugin.hpp"

#include <string>

#include "communication/comm-manager.hpp"
#include "expressions/expressions-hasher.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"

using namespace llvm;

AeolusPlugin::AeolusPlugin(ParallelContext *const context, string fnamePrefix,
                           RecordType rec,
                           vector<RecordAttribute *> &whichFields,
                           string pgType)
    : BinaryBlockPlugin(context, fnamePrefix, rec, whichFields, false),
      pgType(pgType) {
  if (wantedFields.size() == 0) {
    string error_msg{"[BinaryBlockPlugin: ] Invalid number of fields"};
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  LLVMContext &llvmContext = context->getLLVMContext();

  size_t pos_path = fnamePrefix.find_last_of("/");
  if (pos_path == std::string::npos) pos_path = 0;
  std::string extra_prefix = "";
  std::string txn_schema =
      scheduler::WorkerPool::getInstance().get_benchmark_name();
  // convert to lower case
  std::transform(txn_schema.begin(), txn_schema.end(), txn_schema.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (txn_schema.compare("tpcc") == 0) {
    extra_prefix = "tpcc_";
  } else if (txn_schema.compare("ssbm") == 0) {
    extra_prefix = "ssbm_";
  }

  std::string aeolus_rel_name = extra_prefix + fnamePrefix.substr(pos_path + 1);
  aeolus_rel_name = aeolus_rel_name.substr(
      0, aeolus_rel_name.length() - 4);  // remove the trailing .csv

  time_block t("TpgBlockCOW: ");

  auto &txn_storage = storage::Schema::getInstance();
  storage::ColumnStore *tbl = nullptr;

  std::cout << "Finding Table: " << aeolus_rel_name << std::endl;

  for (auto &tb : txn_storage.getTables()) {
    if (aeolus_rel_name.compare(tb->name) == 0) {
      assert(tb->storage_layout == storage::COLUMN_STORE);
      tbl = (storage::ColumnStore *)tb;
      break;
    }
  }
  assert(tbl != nullptr);

  // uint64_t num_records = tbl->getNumRecords();

  // std::cout << "# Records " << num_records << std::endl;
  if (pgType.compare("block-cow") == 0 ||
      pgType.compare("block-snapshot") == 0) {
    for (const auto &in : wantedFields) {
      const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
      size_t type_size = context->getSizeOf(llvm_type);

      // wantedFieldsFiles.emplace_back(StorageManager::getOrLoadFile(fileName,
      // type_size, PAGEABLE));

      for (auto &c : tbl->getColumns()) {
        if (c->name.compare(in->getAttrName()) == 0) {
          // uint64_t num_records = c->snapshot_get_num_records();
          // std::cout << "NUM RECORDS:" << num_records << std::endl;

          auto d = c->snapshot_get_data();

          // std::vector<storage::mem_chunk> d = c->snapshot_get_data();
          std::vector<mem_file> mfiles{d.size()};

          for (size_t i = 0; i < mfiles.size(); ++i) {
            std::cout << "AEO name: " << c->name << std::endl;
            std::cout << "AEO #-records: " << (d[i].second) << std::endl;
            //*(ptr + i) = c->elem_size * d[i].second;

            mfiles[i].data = d[i].first.data;
            mfiles[i].size = c->elem_size * d[i].second;  // num_records
          }
          wantedFieldsFiles.emplace_back(mfiles);
          break;
        }
      }

      if (in->getOriginalType()->getTypeID() == DSTRING) {
        string fileName = fnamePrefix + "." + in->getAttrName();
        // fetch the dictionary
        void *dict = StorageManager::getDictionaryOf(fileName);
        ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
      }
    }

  } else {
    assert(false && "Not Implemented");
  }

  std::cout << "[AEOLUS PLUGIN DONE] " << fnamePrefix << std::endl;
  finalize_data();
}

extern "C" {

storage::ColumnStore *getRelation(std::string fnamePrefix) {
  size_t pos_path = fnamePrefix.find_last_of("/");
  if (pos_path == std::string::npos) pos_path = 0;
  std::string extra_prefix = "";
  std::string txn_schema =
      scheduler::WorkerPool::getInstance().get_benchmark_name();
  // convert to lower case
  std::transform(txn_schema.begin(), txn_schema.end(), txn_schema.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (txn_schema.compare("tpcc") == 0) {
    extra_prefix = "tpcc_";
  } else if (txn_schema.compare("ssbm") == 0) {
    extra_prefix = "ssbm_";
  }

  std::string aeolus_rel_name = extra_prefix + fnamePrefix.substr(pos_path + 1);
  aeolus_rel_name = aeolus_rel_name.substr(
      0, aeolus_rel_name.length() - 4);  // remove the trailing .csv

  for (auto &tb : storage::Schema::getInstance().getTables()) {
    if (aeolus_rel_name.compare(tb->name) == 0) {
      // assert(tb->storage_layout == storage::COLUMN_STORE);
      return (storage::ColumnStore *)tb;
    }
  }
  assert(false && "Relation not found.");
}

void **getDataPointerForFile(const char *relName, const char *attrName,
                             void *session) {
  const auto &tbl = getRelation({relName});

  for (auto &c : tbl->getColumns()) {
    if (strcmp(c->name.c_str(), attrName) == 0) {
      const auto &data_arenas = c->snapshot_get_data();
      void **arr = (void **)malloc(sizeof(void *) * data_arenas.size());
      for (uint i = 0; i < data_arenas.size(); i++) {
        arr[i] = data_arenas[i].first.data;
      }
      return arr;
    }
  }
  assert(false && "ERROR: getDataPointerForFile");
}
void freeDataPointerForFile(void **inn) { free(inn); }

int64_t *getNumOfTuplesPerPartition(const char *relName, void *session) {
  const auto &tbl = getRelation({relName});

  const auto &c = tbl->getColumns()[0];

  const auto &data_arenas = c->snapshot_get_data();
  int64_t *arr = (int64_t *)malloc(sizeof(int64_t *) * data_arenas.size());

  for (uint i = 0; i < data_arenas.size(); i++) {
    arr[i] = c->elem_size * data_arenas[i].second;
  }
  return arr;

  assert(false && "ERROR: getNumOfTuplesPerPartition");
}

void freeNumOfTuplesPerPartition(int64_t *inn) {
  free(inn);
  // TODO: bit-mast reset logic.
}

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
