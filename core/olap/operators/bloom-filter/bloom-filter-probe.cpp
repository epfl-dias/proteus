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

#include "bloom-filter-probe.hpp"

#include <expressions/expressions/ref-expression.hpp>
#include <values/indexed-seq.hpp>

void BloomFilterProbe::produce_(ParallelContext *context) {
  auto t = getFilterType(context);

  filter_ptr = context->appendStateVar(
      t,
      [=](llvm::Value *pip) -> llvm::Value * {
        return context->gen_call("getBloomFilter",
                                 {pip, context->createInt64(bloomId)}, t);
      },
      [=](llvm::Value *, llvm::Value *s) {
        //        context->deallocateStateVar(s);
      });

  getChild()->produce(context);
}

void BloomFilterProbe::consume(ParallelContext *context,
                               const OperatorState &childState) {
  auto ref = findInFilter(context, childState);

  context->gen_if(ref, childState)([&]() {
    // yield to parent
    getParent()->consume(context, childState);
  });
}
