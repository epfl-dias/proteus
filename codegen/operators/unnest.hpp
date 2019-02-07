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

#include "expressions/expressions-generator.hpp"
#include "expressions/path.hpp"
#include "operators/operators.hpp"
#include "util/catalog.hpp"

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
  virtual ~Unnest() { LOG(INFO) << "Collapsing Unnest operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return true; }

 private:
  void generate(Context *const context, const OperatorState &childState) const;
  expression_t pred;
  Path path;
};

#endif /* UNNEST_HPP_ */
