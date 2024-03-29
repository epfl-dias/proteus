/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef OUTER_UNNEST_HPP_
#define OUTER_UNNEST_HPP_

#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/path.hpp"
#include "operators.hpp"

class OuterUnnest : public UnaryOperator {
 public:
  OuterUnnest(expression_t pred, Path path, Operator *const child)
      : UnaryOperator(child), path(std::move(path)), pred(std::move(pred)) {}
  ~OuterUnnest() override { LOG(INFO) << "Collapsing Outer Unnest operator"; }
  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  bool isFiltering() const override { return true; }

  RecordType getRowType() const override {
    // FIXME: implement
    throw runtime_error("unimplemented");
  }

 private:
  void generate(Context *const context, const OperatorState &childState) const;
  expression_t pred;
  Path path;
};

#endif /* UNNEST_HPP_ */
