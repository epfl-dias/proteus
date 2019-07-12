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

#include "plugins/aeolus-plugin.hpp"

#include "communication/comm-manager.hpp"
#include "expressions/expressions-hasher.hpp"

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

  // std::vector<Type *> parts_array;
  for (const auto &in : wantedFields) {
    string fileName = fnamePrefix + "." + in->getAttrName();

    const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
    size_t type_size = context->getSizeOf(llvm_type);

    std::cout << "[AUNN][Plugin-block-remote] type: " << in->getType()
              << std::endl;

    if (strcmp(pgType.c_str(), "block-remote") == 0) {
      time_block t("TpgBlockRemote: ");
      std::cout << "[AUNN][Plugin-block-remote] filename: " << fileName
                << std::endl;

      uint64_t snapshot_epoch;
      ushort master_ver;
      {
        time_block t("TcommSnapshot: ");
        if (communication::CommManager::getInstance().reqeust_snapshot(
                master_ver, snapshot_epoch)) {
          std::cout << "[scan-to-blocks-sm-plugin][block-remote] epoch: "
                    << snapshot_epoch << std::endl;
          std::cout << "[scan-to-blocks-sm-plugin][block-remote] master_ver: "
                    << master_ver << std::endl;

        } else {
          throw new std::runtime_error(
              "[scan-to-blocks-sm-plugin][block-remote] Snapshot request "
              "failed");
        }
      }

      fileName =
          "/dev/shm/" + std::to_string(master_ver) + "__" + in->getAttrName();
      std::cout << fileName << std::endl;
      wantedFieldsFiles.emplace_back(
          StorageManager::getOrLoadFile(fileName, type_size, PAGEABLE));

      // format filename here and then try proteus storage manager loader..

      // name format: master_version + __  + column.name

      // std::vector<mem_file> mfiles{1};

      // if(n_recs_fetch != 0 ){
      //     mfiles[0].size = sizeof(int) * n_recs_fetch;
      // } else {
      //     mfiles[0].size = getFileSize_t(fileName.c_str()); //8224;//
      // }

      // mfiles[0].data = getSHMPtr(fileName.c_str(), mfiles[0].size);
      // usleep(read_txn_wait_usec);

      //          wantedFieldsFiles.emplace_back(mfiles);

      std::cout << "[AUNN][Plugin-block-remote] filename: " << in->getAttrName()
                << std::endl;

    } else if (strcmp(pgType.c_str(), "block-snapshot") == 0) {
      time_block t("TpgBlockSnapshot: ");
      // In this mode, take and update snapshot of local storage. yes, we do
      // need to lock the storage too but as proteus is single-query, we dont
      // need to locl it for now.

      // communication::CommManager::getInstance().reqeust_snapshot();

      uint64_t snapshot_epoch;
      ushort master_ver;
      {
        time_block t("TcommSnapshot: ");
        if (communication::CommManager::getInstance().reqeust_snapshot(
                master_ver, snapshot_epoch)) {
          std::cout << "[scan-to-blocks-sm-plugin][block-remote] epoch: "
                    << snapshot_epoch << std::endl;
          std::cout << "[scan-to-blocks-sm-plugin][block-remote] master_ver: "
                    << master_ver << std::endl;

        } else {
          throw new std::runtime_error(
              "[scan-to-blocks-sm-plugin][block-remote] Snapshot request "
              "failed");
        }
      }

      // Update the snapshot

      std::string shmfileName =
          "/dev/shm/" + std::to_string(master_ver) + "__" + in->getAttrName();
      fileName = fnamePrefix + "." + in->getAttrName();

      std::vector<mem_file> proteus_file =
          StorageManager::getOrLoadFile(fileName, type_size, PINNED);

      std::vector<mem_file> snapshot_file =
          StorageManager::getOrLoadFile(shmfileName, type_size, PAGEABLE);

      // update the snapshot through bit mask.

      // Assumption here is snapshot is pre-allocated and non-extendible.
      // not for strings.
      // assert(proteus_file.size() == snapshot_file.size());

      // assert((in->getType().find("Int") != std::string::npos) &&
      // in->getType().c_str());

      // assert(proteus_file.size() == snapshot_file.size());

      vector<mem_file>::iterator pt, st;
      int cnt = 0;
      {
        time_block t("TpgBlockSnapshotUpd: ");
        std::cout << "here1" << std::endl;
        for (pt = proteus_file.begin(), st = snapshot_file.begin();
             pt != proteus_file.end(); pt++, st++) {
          // type_size
          // pt->size
          // pt->data
          // iterate over tuples

          //   uint8_t *p = ((uint8_t*) chunk->data)+i;
          // if(*p >> 7 == 1){
          //   counter++;
          // }

          // uint8_t
          // uint16_t
          // uint32_t
          // uint64_t
          std::cout << "here" << std::endl;
          for (uint i = 0; i < (pt->size) / 8; i++) {
            size_t offset = i * type_size;
            uint8_t *data_st = ((uint8_t *)st->data) + offset;
            // if(*data_st >> 7 == 1){
            // updated tuple
            uint8_t *data_pt = ((uint8_t *)pt->data) + offset;
            *data_pt = *data_st;
            // *data_st = *data_st & (01111111);

            cnt++;
            //}
          }
        }
      }
      // add the file to list
      wantedFieldsFiles.emplace_back(proteus_file);

      // unload the snapshot file
      // StorageManager::unloadFile(shmfileName);

      std::cout << "[scan-to-blocks-sm-plugin][block-snapshot] file: "
                << in->getAttrName() << " -- update counter: " << cnt
                << std::endl;

    } else {
      time_block t("TpgBlock: ");

      wantedFieldsFiles.emplace_back(
          StorageManager::getOrLoadFile(fileName, type_size, PAGEABLE));
    }
    // wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
    // FIXME: consider if address space should be global memory rather than
    // generic
    // Type * t = PointerType::get(((const PrimitiveType *)
    // tin)->getLLVMType(llvmContext), /* address space */ 0);

    // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

    if (in->getOriginalType()->getTypeID() == DSTRING) {
      // fetch the dictionary
      void *dict = StorageManager::getDictionaryOf(fileName);
      ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
    }
  }

  finalize_data();
}
