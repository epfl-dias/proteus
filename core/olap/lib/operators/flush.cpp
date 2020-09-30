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

#include "flush.hpp"

#include <utility>

#include "olap/util/parallel-context.hpp"

using namespace llvm;

Flush::Flush(vector<expression_t> outputExprs_v, Operator *const child,
             std::string outPath)
    : UnaryOperator(child),
      outPath(std::move(outPath)),
      outputExpr(outputExprs_v),
      relName(outputExprs_v[0].getRegisteredRelName()) {}

void Flush::produce_(ParallelContext *context) {
  IntegerType *t = Type::getInt64Ty(context->getLLVMContext());
  result_cnt_id = context->appendStateVar(
      PointerType::getUnqual(t),
      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();

        Value *mem_acc = context->allocateStateVar(t);

        Builder->CreateStore(context->createInt64(0), mem_acc);

        OperatorState childState{*this,
                                 map<RecordAttribute, ProteusValueMemory>{}};
        ExpressionFlusherVisitor flusher{context, childState, outPath.c_str(),
                                         relName};
        flusher.beginList();

        return mem_acc;
      },

      [=](llvm::Value *, llvm::Value *s) {
        OperatorState childState{*this,
                                 map<RecordAttribute, ProteusValueMemory>{}};
        ExpressionFlusherVisitor flusher{context, childState, outPath.c_str(),
                                         relName};
        flusher.endList();
        flusher.flushOutput(outputExpr.getExpressionType());

        if (getParent()) {
          std::map<RecordAttribute, ProteusValueMemory> variableBindings;
          auto oidtype = Catalog::getInstance()
                             .getPlugin(rowcount.getRelationName())
                             ->getOIDType();
          RecordAttribute oid{rowcount.getRelationName(), activeLoop, oidtype};
          variableBindings[oid] = context->toMem(
              llvm::ConstantInt::get(
                  oidtype->getLLVMType(context->getLLVMContext()), 0),
              context->createFalse());
          variableBindings[rowcount] = context->toMem(
              context->getBuilder()->CreateLoad(s), context->createFalse());
          getParent()->consume(context, {*this, variableBindings});
        }
        context->deallocateStateVar(s);
      });

  getChild()->produce(context);
}

void Flush::consume(ParallelContext *context, const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();

  ExpressionFlusherVisitor flusher{context, childState, outPath.c_str(),
                                   relName};

  // results so far
  Value *mem_resultCtr = context->getStateVar(result_cnt_id);
  Value *resultCtr = Builder->CreateLoad(mem_resultCtr);

  // flushing out delimiter (IF NEEDED)
  flusher.flushDelim(resultCtr);

  outputExpr.accept(flusher);

  // increase result ctr
  Value *resultCtrInc = Builder->CreateAdd(resultCtr, context->createInt64(1));
  Builder->CreateStore(resultCtrInc, mem_resultCtr);
}
