/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                Ecole Polytechnique Federale de Lausanne

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

#ifndef COMPRESSED_BLOCK_TO_TUPLES_HPP
#define COMPRESSED_BLOCK_TO_TUPLES_HPP

#include <platform/common/gpu/gpu-common.hpp>
#include "olap/util/parallel-context.hpp"

#include "lib/operators/operators.hpp"

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED = false,
          bool GLOBAL_STAGE = false>
class CompressedBlockToTuples : public experimental::UnaryOperator {
 public:
  CompressedBlockToTuples(Operator *const child, std::vector<expression_t> wantedFields,
                bool gpu = true, gran_t granularity = gran_t::GRID)
      : UnaryOperator(child),
        wantedFields(std::move(wantedFields)),
        granularity(granularity),
        gpu(gpu) {
    assert((gpu || granularity == gran_t::THREAD) &&
           "CPU can only have a THREAD-granularity block2tuples");
    assert(granularity != gran_t::BLOCK &&
           "BLOCK granurality is not supported yet");  // TODO: support BLOCK
                                                       // granularity
  }

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : wantedFields) {
      auto a = attr.getRegisteredAs();
      auto type = a.getOriginalType();
      if (dynamic_cast<const BlockType *>(type)) {
        type = &(dynamic_cast<const BlockType *>(type)->getNestedType());
      }
      auto rec =
          new RecordAttribute(a.getRelationName(), a.getAttrName(), type);
      attrs.emplace_back(rec);
    }
    return attrs;
  }

  [[nodiscard]] bool isPacked() const override {
    assert(getChild()->isPacked());
    return false;
  }

  void registerGpuFunctions(OlapParallelContext *context);

 private:
  ProteusBareValue step(OlapParallelContext *context, llvm::IntegerType *type);
  void nextEntry(llvm::Value *mem_itemCtr, OlapParallelContext *context);
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  std::vector<StateVar> old_buffs;
  bool release_input_buffers_{true};
  gran_t granularity;

  bool gpu;
};

using CompressedBlockToTuplesShared = CompressedBlockToTuples<false, false>;
using CompressedBlockToTuplesLegacyGlobalTileWindow =
    CompressedBlockToTuples<true, false>;
using CompressedBlockToTuplesRegisters = CompressedBlockToTuples<false, true>;
using CompressedBlockToTuplesGlobalStage =
    CompressedBlockToTuples<false, false, true>;

extern template class CompressedBlockToTuples<false, false>;
extern template class CompressedBlockToTuples<true, false>;
extern template class CompressedBlockToTuples<false, true>;
extern template class CompressedBlockToTuples<false, false, true>;

#endif //COMPRESSED_BLOCK_TO_TUPLES_HPP
