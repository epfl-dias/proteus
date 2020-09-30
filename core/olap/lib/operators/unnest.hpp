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

#ifndef UNNEST_HPP_
#define UNNEST_HPP_

#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/path.hpp"
#include "lib/util/catalog.hpp"
#include "operators.hpp"

//#define DEBUGUNNEST
/**
 * XXX Paper comment: 'Very few ways of evaluating unnest operator -> lamdaDB
 * only provides a nested-loop variation'
 */
class Unnest : public UnaryOperator {
 public:
  Unnest(expression_t pred, Path path, Operator *const child)
      : UnaryOperator(child), path(path), pred(std::move(pred)) {
    Catalog &catalog = Catalog::getInstance();
    catalog.registerPlugin(path.toString(), path.getRelevantPlugin());
  }
  Unnest(expression_t pred, expression_t path, Operator *const child)
      : Unnest(pred,
               Path(path.getRegisteredRelName(),
                    dynamic_cast<const expressions::RecordProjection *>(
                        path.getUnderlyingExpression())),
               child) {}
  ~Unnest() override { LOG(INFO) << "Collapsing Unnest operator"; }
  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  bool isFiltering() const override { return true; }

  RecordType getRowType() const override;

 private:
  void generate(Context *const context, const OperatorState &childState) const;
  Path path;
  expression_t pred;
};

#endif /* UNNEST_HPP_ */
