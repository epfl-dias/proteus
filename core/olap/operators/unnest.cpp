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

void Unnest::produce_(ParallelContext *context) {
  getChild()->produce(context);
}

void Unnest::consume(Context *const context, const OperatorState &childState) {
  generate(context, childState);
}

void Unnest::generate(Context *const context,
                      const OperatorState &childState) const {
  auto Builder = context->getBuilder();

  // Generate path. Value returned must be a collection
  ExpressionGeneratorVisitor pathExprGenerator =
      ExpressionGeneratorVisitor(context, childState);
  auto pathProj = path.get();
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

  // Preparing call to parent
  std::map<RecordAttribute, ProteusValueMemory> unnestBindings{
      childState.getBindings()};
  Catalog &catalog = Catalog::getInstance();
  LOG(INFO) << "[Unnest: ] Registering plugin of " << path.toString();
  std::cout << "[Unnest: ] Registering plugin of " << path.toString()
            << std::endl;
  catalog.registerPlugin(path.toString(), pg);

  // attrNo does not make a difference
  RecordAttribute unnestedAttr(2, path.toString(), activeLoop,
                               pathProj->getExpressionType());

  unnestBindings[unnestedAttr] = nestedValueItem;
  OperatorState newState{*this, unnestBindings};

  /**
   * Predicate Evaluation:
   */
  context->gen_if(pred, newState)([&]() {
    // Triggering parent
    getParent()->consume(context, newState);
  });

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

RecordType Unnest::getRowType() const {
  RecordType rec{getChild()->getRowType()};
  auto *unnestedAttr =
      new RecordAttribute(rec.getArgs().size(), path.toString(), activeLoop,
                          path.get()->getExpressionType());
  rec.appendAttribute(unnestedAttr);
  return rec;
}
