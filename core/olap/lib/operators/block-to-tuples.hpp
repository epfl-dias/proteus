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
#ifndef BLOCK_TO_TUPLES_HPP_
#define BLOCK_TO_TUPLES_HPP_

#include <platform/common/gpu/gpu-common.hpp>

#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class BlockToTuples : public experimental::UnaryOperator {
 public:
  BlockToTuples(Operator *const child, std::vector<expression_t> wantedFields,
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

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
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

 private:
  void nextEntry(llvm::Value *mem_itemCtr, ParallelContext *context);
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  std::vector<StateVar> old_buffs;
  gran_t granularity;

  bool gpu;
};

#endif /* BLOCK_TO_TUPLES_HPP_ */
