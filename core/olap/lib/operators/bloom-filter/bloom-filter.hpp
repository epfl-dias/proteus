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

#ifndef PROTEUS_BLOOM_FILTER_HPP
#define PROTEUS_BLOOM_FILTER_HPP

#include "lib/operators/operators.hpp"

class BloomFilter : public experimental::UnaryOperator {
 public:
  BloomFilter(Operator *child, expression_t e, size_t filterSize,
              uint64_t bloomId);

  [[nodiscard]] RecordType getRowType() const override {
    return getChild()->getRowType();
  }

 protected:
  llvm::Type *getFilterType(ParallelContext *context) const;
  expressions::RefExpression findInFilter(
      ParallelContext *context, const OperatorState &childState) const;

  const expression_t e;

  StateVar filter_ptr;

  const uint64_t bloomId;
  const size_t filterSize;
};

#endif /* PROTEUS_BLOOM_FILTER_HPP */
