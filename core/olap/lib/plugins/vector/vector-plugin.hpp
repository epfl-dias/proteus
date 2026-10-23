/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_VECTOR_PLUGIN_HPP
#define PROTEUS_VECTOR_PLUGIN_HPP

#include <olap/plugins/binary-block-plugin.hpp>
#include <olap/values/types.hpp>
#include <variant>

class VectorPlugin : public BinaryBlockPlugin {
 public:
  VectorPlugin(OlapParallelContext *context,
               const std::vector<std::pair<RecordAttribute *,
                                           std::shared_ptr<proteus_any_vector>>>
                   &fields);

  ~VectorPlugin() override;

 protected:
  llvm::Value *getDataPointersForFile(OlapParallelContext *context, size_t i,
                                      llvm::Value *session_ptr) const override;

  void freeDataPointersForFile(OlapParallelContext *context, size_t i,
                               llvm::Value *v) const override;

  std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      OlapParallelContext *context, llvm::Value *session_ptr) const override;

  void freePartitionSizes(OlapParallelContext *context,
                          llvm::Value *v) const override;

 private:
  std::vector<std::shared_ptr<proteus_any_vector>> data;
  void ***data_ptrs_ptr;
  size_t *data_size_ptr;
};

#endif /* PROTEUS_VECTOR_PLUGIN_HPP */
