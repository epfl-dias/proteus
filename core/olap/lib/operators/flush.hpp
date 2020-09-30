/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2018
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

#ifndef FLUSH_HPP_
#define FLUSH_HPP_

#include "lib/expressions/expressions-flusher.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "olap/operators/monoids.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class Flush : public experimental::UnaryOperator {
 public:
  Flush(vector<expression_t> outputExprs, Operator *child, std::string outPath);
  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override {
    return getChild()->isFiltering();
  }

  [[nodiscard]] RecordType getRowType() const override {
    return {std::vector<RecordAttribute *>{new RecordAttribute(rowcount)}};
  }

  virtual std::string getOutputPath() { return outPath; }

 protected:
  StateVar result_cnt_id;

  expressions::RecordConstruction outputExpr;

  std::string outPath;
  std::string relName;

  RecordAttribute rowcount{outPath, "ROWCOUNT", new Int64Type()};
};

#endif /* FLUSH_HPP_ */
