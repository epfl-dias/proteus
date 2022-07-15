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

#ifndef PROTEUS_BINARY_BLOCK_PLUGIN_RUNTIME_DATA_HANDLES_HPP
#define PROTEUS_BINARY_BLOCK_PLUGIN_RUNTIME_DATA_HANDLES_HPP

#include <olap/plugins/binary-block-plugin.hpp>

namespace proteus {
namespace olap_plugins {

class BinaryBlockPluginRuntimeDataHandles : public ::BinaryBlockPlugin {
 public:
  using ::BinaryBlockPlugin::BinaryBlockPlugin;

 protected:
  llvm::Value *getSession(ParallelContext *context) const override;

  llvm::Value *getDataPointersForFile(ParallelContext *context, size_t i,
                                      llvm::Value *session_ptr) const override;
  void freeDataPointersForFile(ParallelContext *context, size_t i,
                               llvm::Value *v) const override;
  std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      ParallelContext *context, llvm::Value *session_ptr) const override;
  void freePartitionSizes(ParallelContext *context,
                          llvm::Value *v) const override;

  void releaseSession(ParallelContext *context, llvm::Value *) const override;

 public:
  virtual void **getDataPointerForFile_runtime(size_t i, const char *relName,
                                               const char *attrName,
                                               void *session) = 0;

  virtual void freeDataPointerForFile_runtime(void **inn) = 0;

  virtual int64_t *getNumOfTuplesPerPartition_runtime(const char *relName,
                                                      void *session) = 0;

  virtual void freeNumOfTuplesPerPartition_runtime(int64_t *inn) = 0;
};

}  // namespace olap_plugins
}  // namespace proteus

#endif /* PROTEUS_BINARY_BLOCK_PLUGIN_RUNTIME_DATA_HANDLES_HPP */
