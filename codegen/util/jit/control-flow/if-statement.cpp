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

#include "util/jit/control-flow/if-statement.hpp"

#include "expressions/expressions-generator.hpp"
#include "util/context.hpp"

static ProteusValue gen(const expression_t &expr, const OperatorState &state,
                        Context *context) {
  ExpressionGeneratorVisitor egv{context, state};
  return expr.accept(egv);
}

if_branch::if_branch(const expression_t &expr, const OperatorState &state,
                     Context *context, llvm::BasicBlock *afterBB)
    : if_branch(gen(expr, state, context), context, afterBB) {}

if_branch::if_branch(const expression_t &expr, const OperatorState &state,
                     Context *context)
    : if_branch(expr, state, context, nullptr) {}

llvm::IRBuilder<> *if_then::getBuilder(Context *context) {
  return context->getBuilder();
}

llvm::LLVMContext &if_then::getLLVMContext(Context *context) {
  return context->getLLVMContext();
}
