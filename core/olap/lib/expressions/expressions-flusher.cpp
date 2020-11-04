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

#include "expressions-flusher.hpp"

#include <lib/util/project-record.hpp>
#include <olap/expressions/expressions/ref-expression.hpp>

#include "lib/operators/operators.hpp"

using namespace llvm;

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::IntConstant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushInt = context->getFunction("flushInt");
  vector<Value *> ArgsV;
  Value *val_int =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
  ArgsV.push_back(val_int);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushInt, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::Int64Constant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushInt = context->getFunction("flushInt64");
  vector<Value *> ArgsV;
  Value *val_int64 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e->getVal()));
  ArgsV.push_back(val_int64);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushInt, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::DateConstant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushInt = context->getFunction("flushDate");
  vector<Value *> ArgsV;
  Value *val_int64 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e->getVal()));
  ArgsV.push_back(val_int64);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushInt, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::FloatConstant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushDouble = context->getFunction("flushDouble");
  vector<Value *> ArgsV;
  Value *val_double =
      ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
  ArgsV.push_back(val_double);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushDouble, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::BoolConstant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushBoolean = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  Value *val_boolean =
      ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
  ArgsV.push_back(val_boolean);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushBoolean, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::DStringConstant *e) {
  string error_msg = string(
      "[Expression Flusher: ] No support for flushing DString constants");
  LOG(ERROR) << error_msg;
  throw runtime_error(error_msg);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::StringConstant *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushStringC = context->getFunction("flushStringC");

  string toFlush = e->getVal();
  size_t start = 0;
  size_t end = toFlush.length();
  const char *str = toFlush.c_str();
  Value *strLLVM = context->CreateGlobalString(str);

  vector<Value *> ArgsV;
  ArgsV.push_back(strLLVM);
  ArgsV.push_back(context->createInt64(start));
  ArgsV.push_back(context->createInt64(end));
  ArgsV.push_back(outputFileLLVM);

  context->getBuilder()->CreateCall(flushStringC, ArgsV);
  return placeholder;
}

/**
 * Always flush the full entry in this case!
 * Reason: This visitor does not actually recurse -
 * for any recursion, the ExpressionGeneratorVisitor is used
 */
ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::InputArgument *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Catalog &catalog = Catalog::getInstance();

  std::vector<Value *> ArgsV;

  const map<RecordAttribute, ProteusValueMemory> &activeVars =
      currState.getBindings();
  list<RecordAttribute> projections = e->getProjections();
  list<RecordAttribute>::iterator it = projections.begin();

  // Is there a case that I am doing extra work?
  // Example: Am I flushing a value that is nested
  // in some value that I have already hashed?
  for (; it != projections.end(); it++) {
    if (it->getAttrName() == activeLoop) {
      map<RecordAttribute, ProteusValueMemory>::const_iterator itBindings;
      for (itBindings = activeVars.begin(); itBindings != activeVars.end();
           itBindings++) {
        RecordAttribute currAttr = itBindings->first;
        if (currAttr.getRelationName() == it->getRelationName() &&
            currAttr.getAttrName() == activeLoop) {
          // Flush value now
          ProteusValueMemory mem_activeTuple = itBindings->second;

          Plugin *plugin = catalog.getPlugin(currAttr.getRelationName());
          if (plugin == nullptr) {
            string error_msg =
                string("[Expression Flusher: ] No plugin provided");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
          }

          plugin->flushTuple(mem_activeTuple, outputFileLLVM);
        }
      }
    }
  }
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::ProteusValueExpression *e) {
  Catalog &catalog = Catalog::getInstance();

  Plugin *plugin = catalog.getPlugin(activeRelation);

  // Resetting activeRelation here would break nested-record-projections
  // activeRelation = "";
  if (plugin == nullptr) {
    string error_msg = string("[Expression Generator: ] No plugin provided");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  } else {
    plugin->flushValueEager(context, e->getValue(), e->getExpressionType(),
                            outputFile);
  }
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::RecordProjection *e) {
  Catalog &catalog = Catalog::getInstance();
  Plugin *plugin = catalog.getPlugin(activeRelation);

  if (activeRelation != e->getOriginalRelationName()) {
    ExpressionGeneratorVisitor exprGenerator(context, currState);
    ProteusValue val = e->accept(exprGenerator);

    plugin->flushValueEager(context, val, e->getExpressionType(), outputFile);
    return placeholder;
  }
  /**
   *  Missing connection apparently ('activeRelation')
   */
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState, activeRelation);

  ProteusValue record = e->getExpr().accept(exprGenerator);
  {
    // if (plugin->getPluginType() != PGBINARY) {
    /* Cache Logic */
    /* XXX Apply in other visitors too! */
    CachingService &cache = CachingService::getInstance();
    //            cout << "LOOKING FOR STUFF FROM "<<e->getRelationName() << "."
    //                    << e->getAttribute().getAttrName()<< endl;
    CacheInfo info = cache.getCache(e);
    /* Must also make sure that no explicit binding exists => No duplicate work
     */
    auto it = currState.getBindings().find(e->getAttribute());
    if (info.structFieldNo != -1 && it == currState.getBindings().end()) {
#ifdef DEBUGCACHING
      cout << "[Flusher: ] Expression found for "
           << e->getOriginalRelationName() << "."
           << e->getAttribute().getAttrName() << "!" << endl;
#endif
      if (!cache.getCacheIsFull(e)) {
#ifdef DEBUGCACHING
        cout << "...but is not useable " << endl;
#endif
      } else {
        assert(dynamic_cast<ParallelContext *>(context));
        ProteusValue tmpWrapper = plugin->readCachedValue(
            info, currState, dynamic_cast<ParallelContext *>(context));
        //                Value *tmp = tmpWrapper.value;
        //                AllocaInst *mem_tmp =
        //                context->CreateEntryBlockAlloca(F,
        //                "mem_cachedToFlush", tmp->getType());
        //                TheBuilder->CreateStore(tmp,mem_tmp);
        //                ProteusValueMemory mem_tmpWrapper = { mem_tmp,
        //                tmpWrapper.isNull };
        //                plugin->flushValue(mem_tmpWrapper,
        //                e->getExpressionType(),outputFileLLVM);
        plugin->flushValueEager(context, tmpWrapper, e->getExpressionType(),
                                outputFile);
        return placeholder;
      }
    } else {
#ifdef DEBUGCACHING
      cout << "[Flusher: ] No cache found for " << e->getOriginalRelationName()
           << "." << e->getAttribute().getAttrName() << "!" << endl;
#endif
    }
    //}
  }
  // Resetting activeRelation here would break nested-record-projections
  // activeRelation = "";
  if (plugin == nullptr) {
    string error_msg = string("[Expression Generator: ] No plugin provided");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  } else {
    Bindings bindings = {&currState, record};
    ProteusValueMemory mem_path;
    // cout << "Active Relation: " << e->getProjectionName() << endl;
    if (e->getProjectionName() != activeLoop) {
      const RecordType *exprType =
          dynamic_cast<const RecordType *>(e->getExpr().getExpressionType());
      if (exprType) {
        RecordAttribute attr = e->getAttribute();
        Value *val =
            projectArg(exprType, record.value, &attr, context->getBuilder());
        if (val) {
          ProteusValue valWrapper;
          valWrapper.value = val;
          valWrapper.isNull =
              record.isNull;  // FIXME: what if only one attribute is nullptr?

          plugin->flushValueEager(context, valWrapper, e->getExpressionType(),
                                  outputFile);
          return placeholder;
        }
      }
      // Path involves a projection / an object
      assert(dynamic_cast<ParallelContext *>(context));
      mem_path = plugin->readPath(
          activeRelation, bindings, e->getProjectionName().c_str(),
          e->getAttribute(), dynamic_cast<ParallelContext *>(context));
    } else {
      // Path involves a primitive datatype
      //(e.g., the result of unnesting a list of primitives)
      Plugin *pg = catalog.getPlugin(activeRelation);
      RecordAttribute tupleIdentifier(activeRelation, activeLoop,
                                      pg->getOIDType());
      mem_path = currState[tupleIdentifier];
    }
    plugin->flushValue(context, mem_path, e->getExpressionType(), outputFile);
  }
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(const expressions::IfThenElse *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  IRBuilder<> *const TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = TheBuilder->GetInsertBlock()->getParent();

  // Need to evaluate, not hash!
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue ifCond = e->getIfCond().accept(exprGenerator);

  // Prepare blocks
  BasicBlock *ThenBB;
  BasicBlock *ElseBB;
  BasicBlock *MergeBB = BasicBlock::Create(llvmContext, "ifExprCont", F);
  context->CreateIfElseBlocks(F, "ifExprThen", "ifExprElse", &ThenBB, &ElseBB,
                              MergeBB);

  // if
  TheBuilder->CreateCondBr(ifCond.value, ThenBB, ElseBB);

  // then
  TheBuilder->SetInsertPoint(ThenBB);
  e->getIfResult().accept(*this);
  TheBuilder->CreateBr(MergeBB);

  // else
  TheBuilder->SetInsertPoint(ElseBB);
  e->getElseResult().accept(*this);

  TheBuilder->CreateBr(MergeBB);

  // cont.
  TheBuilder->SetInsertPoint(MergeBB);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::EqExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::NeExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::GeExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::GtExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::LeExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::LtExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::AndExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::OrExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);

  ProteusValue exprResult = e->accept(exprGenerator);
  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::AddExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::SubExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::MultExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::DivExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::ModExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of binary "
             "expression can only be primitive"));
}

// #include "plugins/json-plugin.hpp"
#include "lib/plugins/csv-plugin-pm.hpp"

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::RecordConstruction *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  // char delim = ',';

  // Function *flushStr = context->getFunction("flushStringCv2");
  // Function *flushFunc = context->getFunction("flushChar");
  // vector<Value*> ArgsV;

  // const list<expressions::AttributeConstruction> atts = e->getAtts();
  // list<expressions::AttributeConstruction>::const_iterator it;
  // //Start 'record'
  // ArgsV.push_back(context->createInt8('{'));
  // ArgsV.push_back(outputFileLLVM);
  // context->getBuilder()->CreateCall(flushFunc, ArgsV);
  // for (it = atts.begin(); it != atts.end();)
  // {
  //     //attrName
  //     Value* val_attr = context->CreateGlobalString(
  //             it->getBindingName().c_str());
  //     ArgsV.clear();
  //     ArgsV.push_back(val_attr);
  //     ArgsV.push_back(outputFileLLVM);
  //     context->getBuilder()->CreateCall(flushStr, ArgsV);

  //     //:
  //     ArgsV.clear();
  //     ArgsV.push_back(context->createInt8(':'));
  //     ArgsV.push_back(outputFileLLVM);
  //     context->getBuilder()->CreateCall(flushFunc, ArgsV);

  //     //value
  //     expressions::Expression* expr = (*it).getExpression();
  //     ProteusValue partialFlush = expr->accept(*this);

  //     //comma, if needed
  //     it++;
  //     if (it != atts.end())
  //     {
  //         ArgsV.clear();
  //         ArgsV.push_back(context->createInt8(delim));
  //         ArgsV.push_back(outputFileLLVM);
  //         context->getBuilder()->CreateCall(flushFunc, ArgsV);
  //     }
  // }
  // ArgsV.clear();
  // ArgsV.push_back(context->createInt8('}'));
  // ArgsV.push_back(outputFileLLVM);
  // context->getBuilder()->CreateCall(flushFunc, ArgsV);

  // FIXME: this may break JSON plugin!!!! can we materialize lists and objects
  // like that ?! NEED TO TEST
  RecordType recType = *((const RecordType *)e->getExpressionType());

  // FIXME: this is a very quick and dirty hack in order to propagate all
  // the necessary information to the JSON plugin in order to allow it to
  // print the Value. We should find either the correct way to do it, or
  // update the interfaces to also accept the bindings (or a visitor)
  struct tmpstruct {
    const expressions::Expression *e;
    ExpressionFlusherVisitor *v;
  };

  ExpressionGeneratorVisitor exprGen(context, currState);
  ProteusValue recValue = e->accept(exprGen);
  recValue.isNull = (llvm::Value *)new tmpstruct{e, this};
  pg->flushValueEager(context, recValue, e->getExpressionType(), outputFile);

  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::MaxExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::MinExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::RandExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::HintExpression *e) {
  return e->getExpr().accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::HashExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::ShiftLeftExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::LogicalShiftRightExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::ArithmeticShiftRightExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::XORExpression *e) {
  ExpressionGeneratorVisitor v(context, currState);
  ProteusValue rv = e->accept(v);
  expressions::ProteusValueExpression rve(e->getExpressionType(), rv);
  return rve.accept(*this);
}

/* Code almost identical to CSVPlugin::flushValue */
void ExpressionFlusherVisitor::flushValue(Value *val, typeID val_type) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  Function *flushFunc;
  switch (val_type) {
    case BOOL: {
      flushFunc = context->getFunction("flushBoolean");
      vector<Value *> ArgsV;
      ArgsV.push_back(val);
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case STRING: {
      /* Untested */
      flushFunc = context->getFunction("flushStringObj");
      vector<Value *> ArgsV;
      ArgsV.push_back(val);
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case FLOAT: {
      flushFunc = context->getFunction("flushDouble");
      vector<Value *> ArgsV;
      ArgsV.push_back(val);
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case INT64: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushInt64");
      ArgsV.push_back(val);
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case INT: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushInt");
      ArgsV.push_back(val);
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case BAG:
    case LIST:
    case SET:
      LOG(ERROR) << "[ExpressionFlusherVisitor: ] This method is meant for "
                    "primitives!";
      throw runtime_error(string(
          "[ExpressionFlusherVisitor: ] This method is meant for primitives!"));
    case RECORD:
      LOG(ERROR) << "[ExpressionFlusherVisitor: ] This method is meant for "
                    "primitives!";
      throw runtime_error(string(
          "[ExpressionFlusherVisitor: ] This method is meant for primitives!"));
    default:
      LOG(ERROR) << "[ExpressionFlusherVisitor: ] Unknown datatype";
      throw runtime_error(
          string("[ExpressionFlusherVisitor: ] Unknown datatype"));
  }
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::NegExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getExpr().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of negate "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::ExtractExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator{context, currState};
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *type = e->getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (type->isPrimitive()) {
    typeID id = type->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of negate "
             "expression can only be primitive"));
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::TestNullExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator{context, currState};
  ProteusValue exprResult = e->accept(exprGenerator);

  Function *flushFunc = context->getFunction("flushBoolean");
  vector<Value *> ArgsV;
  ArgsV.push_back(exprResult.value);
  ArgsV.push_back(outputFileLLVM);
  context->getBuilder()->CreateCall(flushFunc, ArgsV);
  return placeholder;
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::CastExpression *e) {
  outputFileLLVM = context->CreateGlobalString(this->outputFile);
  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  ProteusValue exprResult = e->accept(exprGenerator);

  const ExpressionType *childType = e->getExpr().getExpressionType();
  Function *flushFunc = nullptr;
  string instructionLabel;

  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
        instructionLabel = string("flushInt64");
        break;
      case INT:
        instructionLabel = string("flushInt");
        break;
      case FLOAT:
        instructionLabel = string("flushDouble");
        break;
      case BOOL:
        instructionLabel = string("flushBoolean");
        break;
      case STRING:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(string(
            "[ExpressionFlusherVisitor]: string operations not supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionFlusherVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionFlusherVisitor]: Unknown Input"));
    }
    flushFunc = context->getFunction(instructionLabel);
    vector<Value *> ArgsV;
    ArgsV.push_back(exprResult.value);
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
    return placeholder;
  }
  throw runtime_error(
      string("[ExpressionFlusherVisitor]: input of cast "
             "expression can only be primitive"));
}
ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::RefExpression *e) {
  return e->getExpr().accept(*this);
}

ProteusValue ExpressionFlusherVisitor::visit(
    const expressions::AssignExpression *) {
  throw std::runtime_error(
      "[ExpressionFlusherVisitor]: unsupported flushing of assignment "
      "operation");
}
