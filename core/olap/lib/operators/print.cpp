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

#include "print.hpp"

void Print::produce_(ParallelContext *context) { getChild()->produce(context); }

void Print::consume(Context *const context, const OperatorState &childState) {
  // Generate condition
  ExpressionGeneratorVisitor exprGenerator{context, childState};
  ProteusValue toPrint = arg->accept(exprGenerator);

  // Call print
  context->getBuilder()->CreateCall(print, toPrint.value);

  // Trigger parent
  OperatorState newState{*this, childState};
  getParent()->consume(context, newState);
}
