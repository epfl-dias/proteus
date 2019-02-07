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

#include "operators/unnest.hpp"

void Unnest::produce() { getChild()->produce(); }

void Unnest::consume(Context *const context, const OperatorState &childState) {
  generate(context, childState);
}

void Unnest::generate(Context *const context,
                      const OperatorState &childState) const {
  auto Builder = context->getBuilder();

  // Generate path. Value returned must be a collection
  ExpressionGeneratorVisitor pathExprGenerator =
      ExpressionGeneratorVisitor(context, childState);
  const expressions::RecordProjection *pathProj = path.get();
  ProteusValue nestedValueAll = pathProj->accept(pathExprGenerator);

  /**
   * foreach val in nestedValue:
   *         if(condition)
   *             ...
   */
  llvm::BasicBlock *loopCond, *loopBody, *loopInc, *loopEnd;
  context->CreateForLoop("unnestChildLoopCond", "unnestChildLoopBody",
                         "unnestChildLoopInc", "unnestChildLoopEnd", &loopCond,
                         &loopBody, &loopInc, &loopEnd);
  /**
   * ENTRY:
   * init the vars used by the plugin
   */
  Plugin *pg = path.getRelevantPlugin();
  ProteusValueMemory mem_currentObjId =
      pg->initCollectionUnnest(nestedValueAll);
  Builder->CreateBr(loopCond);

  Builder->SetInsertPoint(loopCond);
  ProteusValue endCond =
      pg->collectionHasNext(nestedValueAll, mem_currentObjId);
  auto val_hasNext = endCond.value;
#ifdef DEBUGUNNEST
  {
    // Printing the active token that will be forwarded
    llvm::Function *debugBoolean = context->getFunction("printBoolean");
    Builder->CreateCall(debugBoolean, {val_hasNext});
  }
#endif
  Builder->CreateCondBr(val_hasNext, loopBody, loopEnd);

  Builder->SetInsertPoint(loopBody);
#ifdef DEBUGUNNEST
//    {
//    //Printing the active token that will be forwarded
//    vector<Value*> ArgsV;
//    ArgsV.clear();
//    ArgsV.push_back(context->createInt64(111));
//    Function* debugInt = context->getFunction("printi64");
//    Builder->CreateCall(debugInt, ArgsV);
//    }
#endif
  ProteusValueMemory nestedValueItem = pg->collectionGetNext(mem_currentObjId);
  // #ifdef DEBUGUNNEST
  {
    auto debugInt = context->getFunction("printi64");

    auto val_offset = context->getStructElem(nestedValueItem.mem, 0);
    auto val_rowId = context->getStructElem(nestedValueItem.mem, 1);
    auto val_currentTokenNo = context->getStructElem(nestedValueItem.mem, 2);
    // Printing the active token that will be forwarded
    Builder->CreateCall(debugInt, {val_offset});

    Builder->CreateCall(debugInt, {val_rowId});

    Builder->CreateCall(debugInt, {val_currentTokenNo});
  }
  // #endif

  // Preparing call to parent
  std::map<RecordAttribute, ProteusValueMemory> unnestBindings{
      childState.getBindings()};
  Catalog &catalog = Catalog::getInstance();
  LOG(INFO) << "[Unnest: ] Registering plugin of " << path.toString();
  std::cout << "[Unnest: ] Registering plugin of " << path.toString()
            << std::endl;
  catalog.registerPlugin(path.toString(), pg);

  // attrNo does not make a difference
  RecordAttribute unnestedAttr = RecordAttribute(2, path.toString(), activeLoop,
                                                 pathProj->getExpressionType());

  unnestBindings[unnestedAttr] = nestedValueItem;
  OperatorState newState{*this, unnestBindings};

  /**
   * Predicate Evaluation:
   */
  llvm::BasicBlock *ifBlock, *elseBlock;
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifUnnestCond",
                              "elseUnnestCond", &ifBlock, &elseBlock, loopInc);

  // Generate condition
  ExpressionGeneratorVisitor predExprGenerator =
      ExpressionGeneratorVisitor(context, newState);
  ProteusValue condition = pred.accept(predExprGenerator);
  Builder->CreateCondBr(condition.value, ifBlock, elseBlock);

  /*
   * IF BLOCK
   * CALL NEXT OPERATOR, ADDING nestedValueItem binding
   */
  Builder->SetInsertPoint(ifBlock);
  // Triggering parent
  getParent()->consume(context, newState);
  Builder->CreateBr(loopInc);

  /**
   * ELSE BLOCK
   * Just branch to the INC part of unnest loop
   */
  Builder->SetInsertPoint(elseBlock);
  Builder->CreateBr(loopInc);

  Builder->SetInsertPoint(loopInc);
  Builder->CreateBr(loopCond);

  Builder->SetInsertPoint(loopEnd);
#ifdef DEBUGUNNEST
  {
    // Printing the active token that will be forwarded
    llvm::Function *debugInt = context->getFunction("printi64");
    Builder->CreateCall(debugInt, {context->createInt64(222)});
  }
#endif
}
