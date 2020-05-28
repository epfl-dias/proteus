/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_BLOOM_FILTER_REPACK_HPP
#define PROTEUS_BLOOM_FILTER_REPACK_HPP

#include <operators/operators.hpp>

#include "bloom-filter.hpp"

class BloomFilterRepack : public BloomFilter {
 public:
  BloomFilterRepack(Operator *const child, expression_t probe,
                    std::vector<expression_t> wantedFields, size_t filterSize,
                    uint64_t bloomId)
      : BloomFilter(child, std::move(probe), filterSize, bloomId),
        wantedFields(std::move(wantedFields)) {}

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] bool isFiltering() const override { return true; }

  [[nodiscard]] RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : wantedFields) {
      auto a = attr.getRegisteredAs();
      auto type = a.getOriginalType();
      auto rec =
          new RecordAttribute(a.getRelationName(), a.getAttrName(), type);
      attrs.emplace_back(rec);
    }
    return attrs;
  }

  void consumeVector(ParallelContext *context, const OperatorState &childState,
                     llvm::Value *filter, size_t vsize, llvm::Value *offset,
                     llvm::Value *cnt);

 protected:
  virtual void consume_flush(ParallelContext *context);
  virtual void open(Pipeline *pip);
  virtual void close(Pipeline *pip);

  std::vector<expression_t> wantedFields;
  std::vector<StateVar> old_buffs;
  std::vector<StateVar> out_buffs;
  StateVar cntVar_id;
  StateVar oidVar_id;
  PipelineGen *closingPip;
  llvm::Function *flushingFunc;
};

#endif /* PROTEUS_BLOOM_FILTER_REPACK_HPP */
