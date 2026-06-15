//
// Created by milossdjuric on 9/11/25.
//

#ifndef COMPRESSED_BLOCK_TO_TUPLES_HPP
#define COMPRESSED_BLOCK_TO_TUPLES_HPP

#include <platform/common/gpu/gpu-common.hpp>
#include "olap/util/parallel-context.hpp"

#include "lib/operators/operators.hpp"

template <bool INGPU_MATERIALIZE, bool RFOR_PREDECODE_REGISTERS = false>
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

using CompressedBlockToTuplesDeferred = CompressedBlockToTuples<false, false>;
using CompressedBlockToTuplesMaterialize = CompressedBlockToTuples<true, false>;
using CompressedBlockToTuplesReg = CompressedBlockToTuples<false, true>;

extern template class CompressedBlockToTuples<false, false>;
extern template class CompressedBlockToTuples<true, false>;
extern template class CompressedBlockToTuples<false, true>;

#endif //COMPRESSED_BLOCK_TO_TUPLES_HPP
