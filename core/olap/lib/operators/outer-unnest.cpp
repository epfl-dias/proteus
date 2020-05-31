/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "outer-unnest.hpp"

using namespace llvm;

void OuterUnnest::produce_(ParallelContext *context) {
  getChild()->produce(context);
}

void OuterUnnest::consume(Context *const context,
                          const OperatorState &childState) {
  generate(context, childState);
}

/**
 * { (v',w) | v <-     X,
 *               w <-    if and{ not p(v,w') | v != NULL, w' <- path(v)}
 *                       then [NULL]
 *                       else { w' | w' <- path(v), p(v,w') }
 */
void OuterUnnest::generate(Context *const context,
                           const OperatorState &childState) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Type *boolType = Type::getInt1Ty(llvmContext);
  Type *type_w = nullptr;
  vector<Value *> ArgsV;

  /**
   *  Preparing if(v != NULL) condition
   *  XXX This check is (sort-of) an 'early stopping' condition
   */
  BasicBlock *IfNotNull;
  BasicBlock *ElseMerge =
      BasicBlock::Create(llvmContext, "unnest_merge", TheFunction);
  {
    context->CreateIfBlock(TheFunction, "if_notNull", &IfNotNull, ElseMerge);

    // Retrieving v using the projection -> Keeping v out of v.w
    auto pathProj = path.get();
    expression_t expr_v = pathProj->getExpr();

    // Evaluating v
    ExpressionGeneratorVisitor vGenerator = ExpressionGeneratorVisitor(
        context, childState, pathProj->getOriginalRelationName());
    ProteusValue val_v = expr_v.accept(vGenerator);
    Value *val_v_isNull = val_v.isNull;

    Value *nullCond =
        Builder->CreateICmpNE(val_v_isNull, context->createTrue());
    Builder->CreateCondBr(nullCond, IfNotNull, ElseMerge);
  }

  /**
   * if( v != NULL )
   * FIXME After this point on, there is no reason to
   * (re-)check for v being NULL..
   */
  BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
  AllocaInst *mem_accumulating = nullptr;
  ProteusValueMemory nestedValueItem;
  Plugin *pg = path.getRelevantPlugin();
  {
    Builder->SetInsertPoint(IfNotNull);

    // Generate path. Value returned must be a collection
    ExpressionGeneratorVisitor pathExprGenerator =
        ExpressionGeneratorVisitor(context, childState);
    auto pathProj = path.get();

    ProteusValue nestedValueAll = pathProj->accept(pathExprGenerator);
#ifdef DEBUG  // pred condition printout
//    vector<Value*> ArgsV;
//    Function* debugBoolean = context->getFunction("printBoolean");
//    ArgsV.push_back(nestedValueAll.isNull);
//    Builder->CreateCall(debugBoolean, ArgsV);
#endif
    /**
     * and{ not p(v,w') | v != NULL, w' <- path(v)}
     *
     * aka:
     * foreach val in nestedValue:
     *         acc = acc AND not(if(condition))
     *             ...
     */
    mem_accumulating =
        context->CreateEntryBlockAlloca(TheFunction, "mem_acc", boolType);
    Value *val_zero = Builder->getInt1(1);
    Builder->CreateStore(val_zero, mem_accumulating);

    // w' <- path(v)
    context->CreateForLoop("outUnnestLoopCond", "outUnnestLoopBody",
                           "outUnnestLoopInc", "outUnnestLoopEnd", &loopCond,
                           &loopBody, &loopInc, &loopEnd);

    // ENTRY: init the vars used by the plugin
    ProteusValueMemory mem_currentObjId =
        pg->initCollectionUnnest(nestedValueAll);
    Builder->CreateBr(loopCond);

    Builder->SetInsertPoint(loopCond);
    Value *isNotNullCond = Builder->CreateNot(nestedValueAll.isNull);
    ProteusValue hasNext =
        pg->collectionHasNext(nestedValueAll, mem_currentObjId);
    // FIXME Can (..should?) break in two checks: if(!isNull) { if(hasNext) ...
    // }
    Value *endCond = Builder->CreateAnd(isNotNullCond, hasNext.value);
    Builder->CreateCondBr(endCond, loopBody, loopEnd);

    Builder->SetInsertPoint(loopBody);

    nestedValueItem = pg->collectionGetNext(mem_currentObjId);
    type_w = (nestedValueItem.mem)->getAllocatedType();

    map<RecordAttribute, ProteusValueMemory> *unnestBindings =
        new map<RecordAttribute, ProteusValueMemory>(childState.getBindings());
    Catalog &catalog = Catalog::getInstance();
    LOG(INFO) << "[Outer Unnest: ] Registering plugin of " << path.toString();
    catalog.registerPlugin(path.toString(), pg);

    // attrNo does not make a difference
    RecordAttribute unnestedAttr = RecordAttribute(
        2, path.toString(), activeLoop, pathProj->getExpressionType());

    (*unnestBindings)[unnestedAttr] = nestedValueItem;
    OperatorState *newState = new OperatorState(*this, *unnestBindings);

    // Predicate Evaluation: Generate condition
    ExpressionGeneratorVisitor predExprGenerator =
        ExpressionGeneratorVisitor(context, *newState);
    ProteusValue condition = pred.accept(predExprGenerator);
    Value *invert_condition = Builder->CreateNot(condition.value);
    Value *val_acc_current = Builder->CreateLoad(mem_accumulating);
    Value *val_acc_new = Builder->CreateAnd(invert_condition, val_acc_current);
    Builder->CreateStore(val_acc_new, mem_accumulating);

    // getParent()->consume(context, *newState);
    (*unnestBindings).erase(unnestedAttr);
    Builder->CreateBr(loopInc);
  }

  // Just branch to the INC part of unnest loop
  {
    Builder->SetInsertPoint(loopInc);
    Builder->CreateBr(loopCond);
    Builder->SetInsertPoint(loopEnd);
    // Builder->CreateBr(ElseIsNull);
  }

  /*
   * Time to generate "if-then-else"
   *
   * w <-    IF and{ not p(v,w') | v != NULL, w' <- path(v)}
   *         THEN [NULL]
   *         ELSE { w' | w' <- path(v), p(v,w') }
   *
   * FIXME Code as is will loop twice over path(v) -> Optimize
   */
  Builder->SetInsertPoint(loopEnd);
  BasicBlock *ifOuterNull, *elseOuterNotNull;

  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifOuterUnnestNull",
                              "elseOuterNotNull", &ifOuterNull,
                              &elseOuterNotNull);

  Value *ifCond = Builder->CreateLoad(mem_accumulating);
  Builder->CreateCondBr(ifCond, ifOuterNull, elseOuterNotNull);

  // attrNo does not make a difference
  RecordAttribute unnestedAttr = RecordAttribute(
      2, path.toString(), activeLoop, path.get()->getExpressionType());
  map<RecordAttribute, ProteusValueMemory> *unnestBindings =
      new map<RecordAttribute, ProteusValueMemory>(childState.getBindings());
  //'if' --> NULL
  {
    Builder->SetInsertPoint(ifOuterNull);
    Value *undefValue = Constant::getNullValue(type_w);
    ProteusValueMemory valWrapper;
    AllocaInst *mem_undefValue =
        context->CreateEntryBlockAlloca(TheFunction, "val_undef", type_w);
    Builder->CreateStore(undefValue, mem_undefValue);
    valWrapper.mem = mem_undefValue;
    valWrapper.isNull = context->createTrue();

    (*unnestBindings)[unnestedAttr] = valWrapper;
    //        cout << "In outer unnest: " <<
    //        unnestedAttr.getOriginalRelationName() << "_"
    //                << unnestedAttr.getName() << endl;
    OperatorState *newStateNull = new OperatorState(*this, *unnestBindings);

    // Triggering parent
    //    {
    //        Function* debugInt64 = context->getFunction("printi64");
    //        ArgsV.clear();
    //        ArgsV.push_back(Builder->getInt64(4444));
    //        Builder->CreateCall(debugInt64, ArgsV);
    //        ArgsV.clear();
    //    }
    getParent()->consume(context, *newStateNull);
    Builder->CreateBr(ElseMerge);
  }

  //'else - outerNotNull'
  {
    Builder->SetInsertPoint(elseOuterNotNull);
    // XXX Same as unnest!
    // foreach val in nestedValue: if(condition) ...
    BasicBlock *loopCond2, *loopBody2, *loopInc2, *loopEnd2;
    context->CreateForLoop("unnestChildLoopCond2", "unnestChildLoopBody2",
                           "unnestChildLoopInc2", "unnestChildLoopEnd2",
                           &loopCond2, &loopBody2, &loopInc2, &loopEnd2);
    /**
     * ENTRY:
     * init the vars used by the plugin
     */
    ExpressionGeneratorVisitor pathExprGenerator =
        ExpressionGeneratorVisitor(context, childState);
    ProteusValue nestedValueAll = path.get()->accept(pathExprGenerator);
    ProteusValueMemory mem_currentObjId2 =
        pg->initCollectionUnnest(nestedValueAll);
    Builder->CreateBr(loopCond2);

    Builder->SetInsertPoint(loopCond2);
    ProteusValue endCond =
        pg->collectionHasNext(nestedValueAll, mem_currentObjId2);
    Builder->CreateCondBr(endCond.value, loopBody2, loopEnd2);

    // body
    {
      Builder->SetInsertPoint(loopBody2);
      ProteusValueMemory nestedValueItem2 =
          pg->collectionGetNext(mem_currentObjId2);
      // Preparing call to parent
      map<RecordAttribute, ProteusValueMemory> *unnestBindings2 =
          new map<RecordAttribute, ProteusValueMemory>(
              childState.getBindings());

      (*unnestBindings2)[unnestedAttr] = nestedValueItem2;
      //            cout << "In outer unnest: " <<
      //            unnestedAttr.getOriginalRelationName()
      //                    << "_" << unnestedAttr.getName() << endl;

      OperatorState *newState2 = new OperatorState(*this, *unnestBindings2);

      /**
       * Predicate Evaluation:
       */
      BasicBlock *ifBlock, *elseBlock;
      context->CreateIfElseBlocks(context->getGlobalFunction(),
                                  "ifOuterUnnestCond", "elseOuterUnnestCond",
                                  &ifBlock, &elseBlock, loopInc2);

      // Generate condition
      ExpressionGeneratorVisitor predExprGenerator =
          ExpressionGeneratorVisitor(context, *newState2);
      ProteusValue condition2 = pred.accept(predExprGenerator);
      Builder->CreateCondBr(condition2.value, ifBlock, elseBlock);

      /*
       * IF BLOCK
       * CALL NEXT OPERATOR, ADDING nestedValueItem binding
       */

      {
        Builder->SetInsertPoint(ifBlock);
        //                    {
        //                        Function* debugInt64 =
        //                        context->getFunction("printi64");
        //                        ArgsV.clear();
        //                        ArgsV.push_back(Builder->getInt64(5555));
        //                        Builder->CreateCall(debugInt64, ArgsV);
        //                        ArgsV.clear();
        //                    }
        // Triggering parent
        getParent()->consume(context, *newState2);
        Builder->CreateBr(loopInc2);
      }

      /**
       * ELSE BLOCK
       * Just branch to the INC part of unnest loop
       */
      {
        Builder->SetInsertPoint(elseBlock);
        Builder->CreateBr(loopInc2);
      }

      {
        Builder->SetInsertPoint(loopInc2);
        Builder->CreateBr(loopCond2);
      }
    }

    {
      Builder->SetInsertPoint(loopEnd2);
      Builder->CreateBr(ElseMerge);
    }
  }

  /**
   * else -> What to do?
   *         -> Nothing; just keep the insert point there
   */
  Builder->SetInsertPoint(ElseMerge);
}
