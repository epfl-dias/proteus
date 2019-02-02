/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "expressions/expressions-generator.hpp"
#include "operators/operators.hpp"

class Print : public UnaryOperator {
 public:
  Print(llvm::Function *debug, expressions::RecordProjection *arg,
        Operator *const child)
      : UnaryOperator(child), arg(arg), print(debug) {}
  virtual ~Print() { LOG(INFO) << "Collapsing print operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return getChild()->isFiltering(); }

 private:
  expressions::RecordProjection *arg;
  llvm::Function *print;

  OperatorState *generate(const OperatorState &childState);
};
