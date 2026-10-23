/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#ifndef PRE_DECOMPRESS_HPP_
#define PRE_DECOMPRESS_HPP_

#include <platform/common/gpu/gpu-common.hpp>

#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class PreDecompress : public experimental::UnaryOperator {
 public:
  PreDecompress(Operator *const child, std::vector<expression_t> wantedFields,
                bool gpu = true, gran_t granularity = gran_t::GRID)
      : UnaryOperator(child),
        wantedFields(std::move(wantedFields)),
        gpu(gpu) {
    assert((gpu || granularity == gran_t::THREAD) &&
           "CPU can only have a THREAD-granularity block2tuples");
    assert(granularity != gran_t::BLOCK &&
           "BLOCK granurality is not supported yet");
  }

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }
  [[nodiscard]] RecordType getRowType() const override {
    return getChild()->getRowType();
  }
  [[nodiscard]] bool isPacked() const override { return getChild()->isPacked(); }

 private:
  void open(Pipeline *pip);
  void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  bool gpu;
};

#endif  // PRE_DECOMPRESS_HPP_
