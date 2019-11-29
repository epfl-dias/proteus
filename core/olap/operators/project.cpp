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

#include "operators/project.hpp"

#include "memory/memory-manager.hpp"
#include "util/parallel-context.hpp"

Project::Project(vector<expression_t> outputExprs, string relName,
                 Operator *const child, Context *context)
    : UnaryOperator(child),
      context(context),
      relName(relName),
      outputExprs(outputExprs) {}

void Project::produce() {
  auto t = llvm::Type::getInt32Ty(context->getLLVMContext());
  oid_id = context->appendStateVar(
      llvm::PointerType::getUnqual(t),
      [=](llvm::Value *) {
        auto mem_acc = context->allocateStateVar(t);

        // Builder->CreateStore(context->createInt32(0), mem_acc);

        return mem_acc;
      },

      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); });

  getChild()->produce();
}

void Project::consume(Context *const context, const OperatorState &childState) {
  generate(context, childState);
}

void Project::generate(Context *const context,
                       const OperatorState &childState) const {
  auto Builder = context->getBuilder();
  auto TheFunction = Builder->GetInsertBlock()->getParent();

  auto cBB = Builder->GetInsertBlock();
  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto state_mem_oid = context->getStateVar(oid_id);
  auto local_oid = Builder->CreateLoad(state_mem_oid);
  auto local_mem_oid =
      context->CreateEntryBlockAlloca("oid", local_oid->getType());
  auto local_mem_cnt =
      context->CreateEntryBlockAlloca("cnt", local_oid->getType());
  Builder->CreateStore(local_oid, local_mem_oid);
  Builder->CreateStore(context->createInt32(1), local_mem_cnt);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateStore(Builder->CreateLoad(local_mem_oid), state_mem_oid);

  Builder->SetInsertPoint(cBB);

  map<RecordAttribute, ProteusValueMemory> bindings;

  ProteusValueMemory oid_value;
  oid_value.mem = local_mem_oid;
  oid_value.isNull = context->createFalse();

  // store new_val to accumulator
  llvm::Value *next_oid = Builder->CreateAdd(Builder->CreateLoad(local_mem_oid),
                                             context->createInt32(0));
  Builder->CreateStore(next_oid, local_mem_oid);
  bindings[RecordAttribute(relName, activeLoop, new IntType())] = oid_value;

  ProteusValueMemory cnt_value;
  cnt_value.mem = local_mem_cnt;
  cnt_value.isNull = context->createFalse();
  bindings[RecordAttribute(relName, "activeCnt", new IntType())] = cnt_value;

  for (const auto &outputExpr : outputExprs) {
    ExpressionGeneratorVisitor exprGenerator{context, childState};

    ProteusValue val = outputExpr.accept(exprGenerator);
    auto mem = context->CreateEntryBlockAlloca(TheFunction, "proj",
                                               val.value->getType());
    Builder->CreateStore(val.value, mem);

    ProteusValueMemory mem_val{mem, val.isNull};
    bindings[outputExpr.getRegisteredAs()] = mem_val;
  }

  OperatorState state{*this, bindings};
  getParent()->consume(context, state);
}
