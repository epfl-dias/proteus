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

#include "routing-policy.hpp"

#include "expressions/expressions-generator.hpp"

namespace routing {
::routing_target Random::evaluate(ParallelContext *const context,
                                  const OperatorState &childState) {
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
  auto Builder = context->getBuilder();
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto getdev = context->getFunction((dev == DeviceType::CPU)
                                         ? "rand_local_cpu"
                                         : "get_ptr_device_or_rand_for_host");

  auto ptr = Builder->CreateLoad(childState[wantedField].mem);
  auto ptr8 = Builder->CreateBitCast(ptr, charPtrType);

  auto target = Builder->CreateCall(getdev, {ptr8});
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);

  // FIXME: should we retry for GPUs as well? (for example for 2CPU+4GPU setups)
  return {Builder->CreateURem(target, fanoutV), dev == DeviceType::CPU};
}
}  // namespace routing
