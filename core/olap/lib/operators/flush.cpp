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

#include "memory/memory-manager.hpp"
#include "olap/util/parallel-context.hpp"

using namespace llvm;

expression_t buildOutputExpression(const vector<expression_t> &outputExprs) {
  list<expressions::AttributeConstruction> attrs;
  std::vector<RecordAttribute *> recattr;
  for (auto expr : outputExprs) {
    assert(expr.isRegistered() && "All output expressions must be registered!");
    expressions::AttributeConstruction *newAttr =
        new expressions::AttributeConstruction(expr.getRegisteredAttrName(),
                                               expr);
    attrs.push_back(*newAttr);
    recattr.push_back(new RecordAttribute{expr.getRegisteredAs()});
  }
  return expression_t::make<expressions::RecordConstruction>(
      new RecordType(recattr), attrs);
}

Flush::Flush(vector<expression_t> outputExprs_v, Operator *const child,
             Context *context, std::string outPath)
    : UnaryOperator(child),
      context(context),
      outPath(outPath),
      outputExpr(buildOutputExpression(outputExprs_v)),
      relName(outputExprs_v[0].getRegisteredRelName()),
      outputExprs_v(outputExprs_v) {}

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

void Flush::consume(Context *const context, const OperatorState &childState) {
  generate(context, childState);
}

void Flush::generate(Context *const context,
                     const OperatorState &childState) const {
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
