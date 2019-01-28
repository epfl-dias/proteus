/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2018
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

#include "operators/project.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "util/raw-memory-manager.hpp"

Project::Project(vector<expression_t> outputExprs, string relName,
                 RawOperator *const child, RawContext *context)
    : UnaryRawOperator(child),
      outputExprs(outputExprs),
      relName(relName),
      context(context) {}

void Project::produce() {
  IntegerType *t = Type::getInt32Ty(context->getLLVMContext());
  oid_id = context->appendStateVar(
      PointerType::getUnqual(t),
      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();

        Value *mem_acc = context->allocateStateVar(t);

        // Builder->CreateStore(context->createInt32(0), mem_acc);

        return mem_acc;
      },

      [=](llvm::Value *, llvm::Value *s) { context->deallocateStateVar(s); });

  getChild()->produce();
}

void Project::consume(RawContext *const context,
                      const OperatorState &childState) {
  generate(context, childState);
}

void Project::generate(RawContext *const context,
                       const OperatorState &childState) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  BasicBlock *cBB = Builder->GetInsertBlock();
  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  Value *state_mem_oid = context->getStateVar(oid_id);
  Value *local_oid = Builder->CreateLoad(state_mem_oid);
  AllocaInst *local_mem_oid =
      context->CreateEntryBlockAlloca("oid", local_oid->getType());
  AllocaInst *local_mem_cnt =
      context->CreateEntryBlockAlloca("cnt", local_oid->getType());
  Builder->CreateStore(local_oid, local_mem_oid);
  Builder->CreateStore(context->createInt32(1), local_mem_cnt);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateStore(Builder->CreateLoad(local_mem_oid), state_mem_oid);

  Builder->SetInsertPoint(cBB);

  map<RecordAttribute, RawValueMemory> bindings;

  RawValueMemory oid_value;
  oid_value.mem = local_mem_oid;
  oid_value.isNull = context->createFalse();

  // store new_val to accumulator
  Value *next_oid = Builder->CreateAdd(Builder->CreateLoad(local_mem_oid),
                                       context->createInt32(0));
  Builder->CreateStore(next_oid, local_mem_oid);
  bindings[RecordAttribute(relName, activeLoop, new IntType())] = oid_value;

  RawValueMemory cnt_value;
  cnt_value.mem = local_mem_cnt;
  cnt_value.isNull = context->createFalse();
  bindings[RecordAttribute(relName, "activeCnt", new IntType())] = cnt_value;

  for (const auto &outputExpr : outputExprs) {
    ExpressionGeneratorVisitor exprGenerator{context, childState};

    RawValue val = outputExpr.accept(exprGenerator);
    AllocaInst *mem = context->CreateEntryBlockAlloca(TheFunction, "proj",
                                                      val.value->getType());
    Builder->CreateStore(val.value, mem);

    RawValueMemory mem_val{mem, val.isNull};
    bindings[outputExpr.getRegisteredAs()] = mem_val;
  }

  OperatorState state{*this, bindings};
  getParent()->consume(context, state);
}

void Project::open(RawPipeline *pip) const {
  std::cout << "Project:open" << std::endl;
  Type *llvm_type = Type::getInt32Ty(context->getLLVMContext());

  size_t size_in_bytes = (llvm_type->getPrimitiveSizeInBits() + 7) / 8;

  void *oid = pip->getStateVar<void *>(oid_id);

  gpu_run(cudaMemset(oid, 0, size_in_bytes));
}

void Project::close(RawPipeline *pip) const {
  std::cout << "Project:close" << std::endl;
}
