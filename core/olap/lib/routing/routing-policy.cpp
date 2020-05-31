/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "olap/routing/routing-policy.hpp"

#include "lib/expressions/expressions-generator.hpp"
#include "lib/operators/operators.hpp"
#include "olap/routing/affinitizers.hpp"
#include "topology/topology.hpp"

extern "C" size_t random_local_cu(void *ptr, AffinityPolicy *aff) {
  return aff->getIndexOfRandLocalCU(ptr);
}

namespace routing {
::routing_target Random::evaluate(ParallelContext *const context,
                                  const OperatorState &childState) {
  if (fanout == 1) return {context->createInt64(0), false};
  auto Builder = context->getBuilder();

  auto crand = context->getFunction("rand");
  auto target = Builder->CreateCall(crand, {});
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), true};
}

::routing_target HashBased::evaluate(ParallelContext *const context,
                                     const OperatorState &childState) {
  auto Builder = context->getBuilder();

  ExpressionGeneratorVisitor exprGenerator{context, childState};
  auto target = e.accept(exprGenerator).value;
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), false};
}

::routing_target Local::evaluate(ParallelContext *const context,
                                 const OperatorState &childState) {
  if (fanout == 1) return {context->createInt64(0), false};

  auto Builder = context->getBuilder();
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto ptr = Builder->CreateLoad(childState[wantedField].mem);
  auto ptr8 = Builder->CreateBitCast(ptr, charPtrType);

  auto this_ptr = Builder->CreateIntToPtr(context->createInt64((uintptr_t)aff),
                                          charPtrType);

  auto target = context->gen_call("random_local_cu", {ptr8, this_ptr},
                                  context->createSizeType());

  return {target, true};
}

Local::Local(size_t fanout, const std::vector<RecordAttribute *> &wantedFields,
             const AffinityPolicy *aff)
    : fanout(fanout), wantedField(*wantedFields[0]), aff(aff) {}

}  // namespace routing
