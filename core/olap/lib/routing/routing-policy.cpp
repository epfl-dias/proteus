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

#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/topology/topology.hpp>

#include "lib/expressions/expressions-generator.hpp"
#include "lib/operators/operators.hpp"
#include "olap/routing/affinitizers.hpp"

extern "C" size_t random_local_cu(void *ptr, AffinityPolicy *aff) {
  return aff->getIndexOfRandLocalCU(ptr);
}

namespace routing {
::routing_target Random::evaluate(ParallelContext *const context,
                                  const OperatorState &childState,
                                  ProteusValueMemory retrycnt) {
  if (fanout == 1) return {context->createInt64(0), false};
  auto Builder = context->getBuilder();

  auto state = [&] {
    save_current_blocks_and_restore_at_exit_scope e{context};
    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    ExpressionGeneratorVisitor vis{context, childState};
    return context->toMem(expressions::rand().accept(vis));
  }();

  auto target = Builder->CreateLoad(state.mem);
  ExpressionGeneratorVisitor vis{context, childState};
  Builder->CreateStore(
      Builder->CreateZExtOrTrunc(
          expressions::HashExpression{
              expressions::ProteusValueExpression{
                  new IntType(), ProteusValue{target, state.isNull}}}
              .accept(vis)
              .value,
          target->getType()),
      state.mem);

  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), true};
}

::routing_target HashBased::evaluate(ParallelContext *const context,
                                     const OperatorState &childState,
                                     ProteusValueMemory retrycnt) {
  auto Builder = context->getBuilder();

  ExpressionGeneratorVisitor exprGenerator{context, childState};
  auto target = e.accept(exprGenerator).value;
  auto fanoutV =
      llvm::ConstantInt::get((llvm::IntegerType *)target->getType(), fanout);
  return {Builder->CreateURem(target, fanoutV), false};
}

::routing_target Local::evaluate(ParallelContext *const context,
                                 const OperatorState &childState,
                                 ProteusValueMemory retrycnt) {
  if (fanout == 1) return {context->createInt64(0), false};

  auto Builder = context->getBuilder();
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto ptr = Builder->CreateLoad(childState[wantedField].mem);
  auto ptr8 = Builder->CreateBitCast(ptr, charPtrType);

  auto this_ptr = Builder->CreateIntToPtr(context->createInt64((uintptr_t)aff),
                                          charPtrType);

  auto target = context->gen_call(random_local_cu, {ptr8, this_ptr});

  return {target, true};
}

Local::Local(size_t fanout, const std::vector<RecordAttribute *> &wantedFields,
             const AffinityPolicy *aff)
    : fanout(fanout), wantedField(*wantedFields[0]), aff(aff) {}

LocalServer::LocalServer(size_t fanout)
    : HashBased(fanout, (int)InfiniBandManager::server_id()) {}

PreferLocal::PreferLocal(size_t fanout,
                         const std::vector<RecordAttribute *> &wantedFields,
                         const AffinityPolicy *aff)
    : priority(fanout, wantedFields, aff), alternative(fanout) {}

routing_target PreferLocal::evaluate(ParallelContext *context,
                                     const OperatorState &childState,
                                     ProteusValueMemory retrycnt) {
  auto Builder = context->getBuilder();

  llvm::BasicBlock *b1;
  llvm::BasicBlock *b2;
  llvm::Value *p1;
  llvm::Value *p2;
  auto phi_type = llvm::IntegerType::getInt64Ty(context->getLLVMContext());

  context
      ->gen_if(lt(
                   expressions::ProteusValueExpression{
                       new IntType(),
                       {Builder->CreateLoad(retrycnt.mem), retrycnt.isNull}},
                   1),
               childState)([&]() {
        p1 = Builder->CreateZExt(
            priority.evaluate(context, childState, retrycnt).target, phi_type);
        b1 = Builder->GetInsertBlock();
      })
      .gen_else([&]() {
        p2 = Builder->CreateZExt(
            alternative.evaluate(context, childState, retrycnt).target,
            phi_type);
        b2 = Builder->GetInsertBlock();
      });

  auto phi = Builder->CreatePHI(phi_type, 2);
  phi->addIncoming(p1, b1);
  phi->addIncoming(p2, b2);

  return {phi, true};
}

PreferLocalServer::PreferLocalServer(size_t fanout)
    : priority(fanout), alternative(fanout) {}

routing_target PreferLocalServer::evaluate(ParallelContext *context,
                                           const OperatorState &childState,
                                           ProteusValueMemory retrycnt) {
  auto Builder = context->getBuilder();

  llvm::BasicBlock *b1;
  llvm::BasicBlock *b2;
  llvm::Value *p1;
  llvm::Value *p2;
  auto phi_type = llvm::IntegerType::getInt64Ty(context->getLLVMContext());

  context
      ->gen_if(lt(
                   expressions::ProteusValueExpression{
                       new IntType(),
                       {Builder->CreateLoad(retrycnt.mem), retrycnt.isNull}},
                   1),
               childState)([&]() {
        p1 = Builder->CreateZExt(
            priority.evaluate(context, childState, retrycnt).target, phi_type);
        b1 = Builder->GetInsertBlock();
      })
      .gen_else([&]() {
        p2 = Builder->CreateZExt(
            alternative.evaluate(context, childState, retrycnt).target,
            phi_type);
        b2 = Builder->GetInsertBlock();
      });

  auto phi = Builder->CreatePHI(phi_type, 2);
  phi->addIncoming(p1, b1);
  phi->addIncoming(p2, b2);

  return {phi, true};
}

}  // namespace routing
