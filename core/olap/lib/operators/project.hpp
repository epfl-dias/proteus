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

#ifndef PROJECT_HPP_
#define PROJECT_HPP_

#include "lib/expressions/expressions-flusher.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "olap/operators/monoids.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class Project : public UnaryOperator {
 public:
  Project(vector<expression_t> outputExprs, string relName,
          Operator *const child, Context *context);
  ~Project() override { LOG(INFO) << "Collapsing Project operator"; }
  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  bool isFiltering() const override { return getChild()->isFiltering(); }

  RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : outputExprs) {
      attrs.emplace_back(new RecordAttribute{attr.getRegisteredAs()});
    }
    return attrs;
  }

 protected:
  Context *context;
  StateVar oid_id;
  string relName;

  vector<expression_t> outputExprs;

  const char *outPath;

 private:
  void generate(Context *const context, const OperatorState &childState) const;
};

#endif /* PROJECT_HPP_ */
