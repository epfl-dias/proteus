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

#include "engines/olap/util/parallel-context.hpp"
#include "operators/operators.hpp"

class BlockToTuples : public UnaryOperator {
 public:
  BlockToTuples(Operator *const child, ParallelContext *const context,
                const std::vector<expression_t> &wantedFields, bool gpu = true,
                gran_t granularity = gran_t::GRID)
      : UnaryOperator(child),
        wantedFields(wantedFields),
        context(context),
        granularity(granularity),
        gpu(gpu) {
    assert((gpu || granularity == gran_t::THREAD) &&
           "CPU can only have a THREAD-granularity block2tuples");
    assert(granularity != gran_t::BLOCK &&
           "BLOCK granurality is not supported yet");  // TODO: support BLOCK
                                                       // granularity
  }

  virtual ~BlockToTuples() { LOG(INFO) << "Collapsing BlockToTuples operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const {
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

 private:
  void nextEntry();
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  std::vector<size_t> old_buffs;
  ParallelContext *const context;
  llvm::AllocaInst *mem_itemCtr;
  gran_t granularity;

  bool gpu;
};

#endif /* BLOCK_TO_TUPLES_HPP_ */
