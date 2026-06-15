/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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

#ifndef NONE_HPP_
#define NONE_HPP_

#include "operators.hpp"

/**
 * Minimal pass-through operator.
 *
 * It preserves the child state and row type verbatim, so it can sit between
 * block-oriented operators regardless of whether the blocks are compressed or
 * already decompressed.
 */
class NoneOperator : public experimental::UnaryOperator {
 public:
  explicit NoneOperator(Operator *const child) : UnaryOperator(child) {}

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;

  [[nodiscard]] bool isFiltering() const override {
    return getChild()->isFiltering();
  }

  [[nodiscard]] RecordType getRowType() const override {
    return getChild()->getRowType();
  }
};

#endif /* NONE_HPP_ */
