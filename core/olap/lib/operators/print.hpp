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

#include "lib/expressions/expressions-generator.hpp"
#include "operators.hpp"

class Print : public UnaryOperator {
 public:
  Print(llvm::Function *debug, expressions::RecordProjection *arg,
        Operator *const child)
      : UnaryOperator(child), arg(arg), print(debug) {}
  ~Print() override { LOG(INFO) << "Collapsing print operator"; }

  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  bool isFiltering() const override { return getChild()->isFiltering(); }

 private:
  expressions::RecordProjection *arg;
  llvm::Function *print;

  OperatorState *generate(const OperatorState &childState);
};
