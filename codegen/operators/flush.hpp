/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2018
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "codegen/util/parallel-context.hpp"
#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"

class Flush : public UnaryOperator {
 public:
  Flush(vector<expression_t> outputExprs, Operator *const child,
        Context *context, const char *outPath = "out.json");
  virtual ~Flush() { LOG(INFO) << "Collapsing Flush operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return getChild()->isFiltering(); }

 protected:
  Context *context;
  size_t result_cnt_id;

  expression_t outputExpr;
  vector<expression_t> outputExprs_v;

  const char *outPath;
  std::string relName;

 private:
  void generate(Context *const context, const OperatorState &childState) const;
};

#endif /* FLUSH_HPP_ */
