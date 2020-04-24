/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "bloom-filter-build.hpp"

#include <expressions/expressions-generator.hpp>
#include <values/indexed-seq.hpp>

void BloomFilterBuild::produce_(ParallelContext *context) {
  auto t = getFilterType(context);

  filter_ptr = context->appendStateVar(
      t,
      [=](llvm::Value *) -> llvm::Value * {
        auto tarr = t->getPointerElementType();
        auto mem = context->allocateStateVar(tarr);
        context->CodegenMemset(
            mem, context->createInt8(0),
            (tarr->getArrayElementType()->getPrimitiveSizeInBits()) *
                tarr->getArrayNumElements());
        return mem;
      },
      [=](llvm::Value *pip, llvm::Value *s) {
        context->gen_call("setBloomFilter",
                          {pip, s, context->createInt64(bloomId)}, t);
      });

  getChild()->produce(context);
}

void BloomFilterBuild::consume(ParallelContext *context,
                               const OperatorState &childState) {
  auto ref = findInFilter(context, childState);

  ExpressionGeneratorVisitor v{context, childState};
  ref.assign(true).accept(v);

  getParent()->consume(context, childState);
}
