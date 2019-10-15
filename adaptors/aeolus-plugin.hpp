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

#ifndef AEOLUS_PLUGIN_HPP_
#define AEOLUS_PLUGIN_HPP_

#include "plugins/binary-block-plugin.hpp"

class AeolusPlugin : public BinaryBlockPlugin {
 public:
  AeolusPlugin(ParallelContext *const context, std::string fnamePrefix,
               RecordType rec, std::vector<RecordAttribute *> &whichFields,
               std::string pgType);

 protected:
  virtual llvm::Value *getSession() const;

  virtual llvm::Value *getDataPointersForFile(size_t i,
                                              llvm::Value *session_ptr) const;
  virtual void freeDataPointersForFile(size_t i, llvm::Value *v) const;
  virtual std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      llvm::Value *session_ptr) const;
  virtual void freePartitionSizes(llvm::Value *v) const;

  virtual void releaseSession(llvm::Value *) const;

 public:
  virtual void **getDataPointerForFile_runtime(const char *relName,
                                               const char *attrName,
                                               void *session);

  virtual void freeDataPointerForFile_runtime(void **inn);

  virtual int64_t *getNumOfTuplesPerPartition_runtime(const char *relName,
                                                      void *session);

  virtual void freeNumOfTuplesPerPartition_runtime(int64_t *inn);

 private:
  std::string pgType;
};

class AeolusCowPlugin : public AeolusPlugin {
 public:
  AeolusCowPlugin(ParallelContext *const context, std::string fnamePrefix,
                  RecordType rec, std::vector<RecordAttribute *> &whichFields)
      : AeolusPlugin(context, fnamePrefix, rec, whichFields, "block-cow") {}
};

extern "C" {
Plugin *createBlockCowPlugin(ParallelContext *context, std::string fnamePrefix,
                             RecordType rec,
                             std::vector<RecordAttribute *> &whichFields);

Plugin *createBlockRemotePlugin(ParallelContext *context,
                                std::string fnamePrefix, RecordType rec,
                                std::vector<RecordAttribute *> &whichFields);
Plugin *createBlockLocalPlugin(ParallelContext *context,
                               std::string fnamePrefix, RecordType rec,
                               std::vector<RecordAttribute *> &whichFields);
Plugin *createBlockElasticPlugin(ParallelContext *context,
                                 std::string fnamePrefix, RecordType rec,
                                 std::vector<RecordAttribute *> &whichFields);
}

#endif /* AEOLUS_PLUGIN_HPP_ */
