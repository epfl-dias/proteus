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

class Join : public BinaryOperator {
 public:
  Join(expressions::BinaryExpression *predicate, Operator *leftChild,
       Operator *rightChild, char *opLabel, Materializer &mat)
      : BinaryOperator(leftChild, rightChild),
        pred(predicate),
        htName(opLabel),
        mat(mat) {}
  virtual ~Join() { LOG(INFO) << "Collapsing Join operator"; }
  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);
  Materializer &getMaterializer() { return mat; }
  virtual bool isFiltering() const { return true; }

 private:
  char *htName;
  OperatorState *generate(Operator *op, OperatorState *childState);
  expressions::BinaryExpression *pred;
  Materializer &mat;
};
