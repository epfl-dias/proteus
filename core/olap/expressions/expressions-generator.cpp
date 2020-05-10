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

#include "expressions/expressions-generator.hpp"

#include <expressions/expressions/ref-expression.hpp>
#include <util/project-record.hpp>

#include "expressions/expressions-hasher.hpp"
#include "operators/operators.hpp"

using namespace llvm;

#pragma push_macro("DEBUG")  // FIXME: REMOVE!!! used to disable prints, as they
                             // are currently undefined for the gpu side
#undef DEBUG  // FIXME: REMOVE!!! used to disable prints, as they are currently
              // undefined for the gpu side

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::IntConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::Int64Constant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::DateConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::FloatConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantFP::get(context->getLLVMContext(), APFloat(e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::BoolConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantInt::get(context->getLLVMContext(), APInt(1, e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::DStringConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e->getVal()));
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::StringConstant *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();

  char *str = new char[e->getVal().length() + 1];
  strcpy(str, e->getVal().c_str());
  Value *globalStr = context->CreateGlobalString(str);

  StructType *strObjType = context->CreateStringStruct();
  Function *F = context->getGlobalFunction();
  AllocaInst *mem_strObj =
      context->CreateEntryBlockAlloca(F, e->getVal(), strObjType);

  Value *val_0 = context->createInt32(0);
  Value *val_1 = context->createInt32(1);

  vector<Value *> idxList = vector<Value *>();
  idxList.push_back(val_0);
  idxList.push_back(val_0);
  Value *structPtr = TheBuilder->CreateGEP(mem_strObj, idxList);
  TheBuilder->CreateStore(globalStr, structPtr);

  idxList.clear();
  idxList.push_back(val_0);
  idxList.push_back(val_1);
  structPtr = TheBuilder->CreateGEP(mem_strObj, idxList);
  TheBuilder->CreateStore(context->createInt32(e->getVal().length()),
                          structPtr);

  Value *val_strObj = TheBuilder->CreateLoad(mem_strObj);
  ProteusValue valWrapper;
  valWrapper.value = val_strObj;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::InputArgument *e) {
  IRBuilder<> *const Builder = context->getBuilder();
  Catalog &catalog = Catalog::getInstance();
  AllocaInst *argMem = nullptr;
  Value *isNull;

  /* No caching logic here, because InputArgument generally
   * does not result in materializing / converting / etc. actions! */
  {
    const map<RecordAttribute, ProteusValueMemory> &activeVars =
        currState.getBindings();
    map<RecordAttribute, ProteusValueMemory>::const_iterator it;

    // A previous visitor has indicated which relation is relevant
    if (activeRelation != "") {
      Plugin *pg = catalog.getPlugin(activeRelation);
      RecordAttribute relevantAttr =
          RecordAttribute(activeRelation, activeLoop, pg->getOIDType());
      it = activeVars.find(relevantAttr);
      if (it == activeVars.end()) {
        string error_msg = string(
                               "[Expression Generator: ] Could not find "
                               "tuple information for ") +
                           activeRelation + "." +
                           e->getProjections().front().getName();
        cout << activeVars.size() << endl;
        map<RecordAttribute, ProteusValueMemory>::const_iterator it =
            activeVars.begin();
        for (; it != activeVars.end(); it++) {
          cout << it->first.getRelationName() << "-" << it->first.getAttrName()
               << "-" << it->first.getOriginalRelationName() << endl;
        }
        // cout << endl;
        // cout << activeRelation << endl;
        // cout << relevantAttr.getRelationName() << "-" <<
        // relevantAttr.getAttrName()
        //             << "-" << relevantAttr.getOriginalRelationName() << endl;
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      } else {
        argMem = (it->second).mem;
        isNull = (it->second).isNull;
      }
    } else {
      list<RecordAttribute> projections = e->getProjections();
      list<RecordAttribute>::iterator it = projections.begin();

      for (; it != projections.end(); it++) {
        /* OID info */
        if (it->getAttrName() == activeLoop) {
          map<RecordAttribute, ProteusValueMemory>::const_iterator itBindings;
          for (itBindings = activeVars.begin(); itBindings != activeVars.end();
               itBindings++) {
            RecordAttribute currAttr = itBindings->first;
            if (currAttr.getRelationName() == it->getRelationName() &&
                currAttr.getAttrName() == activeLoop) {
              /* Found info needed! */
              //                            cout << "Binding found!!!" << endl;
              //                            cout << currAttr.getRelationName()
              //                            << "_" << currAttr.getAttrName() <<
              //                            endl;
              ProteusValueMemory mem_activeTuple = itBindings->second;
              argMem = mem_activeTuple.mem;
              isNull = mem_activeTuple.isNull;
            }
          }
        }
      }
    }
    /*else    {
        LOG(WARNING) << "[Expression Generator: ] No active relation found -
    Non-record case (OR e IS A TOPMOST EXPR.!)"; Have seen this occurring in
    Nest int relationsCount = 0; for(it = activeVars.begin(); it !=
    activeVars.end(); it++)    { RecordAttribute currAttr = it->first; cout <<
    currAttr.getRelationName() <<" and "<< currAttr.getAttrName() << endl;
            //XXX Does 1st part of check ever get satisfied? activeRelation is
    empty here if(currAttr.getRelationName() == activeRelation &&
    currAttr.getAttrName() == activeLoop)    {

                argMem = (it->second).mem;
                isNull = (it->second).isNull;
                relationsCount++;
            }
        }
        if (!relationsCount) {
            string error_msg = string("[Expression Generator: ] Could not find
    tuple information"); LOG(ERROR)<< error_msg; throw runtime_error(error_msg);
        } else if (relationsCount > 1) {
            string error_msg =
                    string("[Expression Generator: ] Could not distinguish
    appropriate bindings"); LOG(ERROR)<< error_msg; throw
    runtime_error(error_msg);
        }
    }*/
  }

  assert(argMem && "Argument not found");

  ProteusValue valWrapper;
  /* XXX Should this load be removed?? Will an optimization pass realize it is
   * not needed? */
  valWrapper.value = Builder->CreateLoad(argMem);
  valWrapper.isNull = isNull;  // context->createFalse();

  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::RecordProjection *e) {
  Catalog &catalog = Catalog::getInstance();
  IRBuilder<> *const Builder = context->getBuilder();
  activeRelation = e->getOriginalRelationName();
  string projName = e->getProjectionName();

  Plugin *plugin = catalog.getPlugin(activeRelation);

  if (plugin != nullptr) {
    /* Cache Logic */
    /* XXX Apply in other visitors too! */
    CachingService &cache = CachingService::getInstance();
    CacheInfo info = cache.getCache(e);

    /* Must also make sure that no explicit binding exists => No duplicate work
     */
    map<RecordAttribute, ProteusValueMemory>::const_iterator it =
        currState.getBindings().find(e->getAttribute());
#ifdef DEBUGCACHING
    if (it != currState.getBindings().end()) {
      cout << "Even if cached, binding's already there!" << endl;
    }
#endif
    if (info.structFieldNo != -1 && it == currState.getBindings().end()) {
#ifdef DEBUGCACHING
      cout << "[Generator: ] Expression found for "
           << e->getOriginalRelationName() << "."
           << e->getAttribute().getAttrName() << "!" << endl;
#endif
      if (!cache.getCacheIsFull(e)) {
#ifdef DEBUGCACHING
        cout << "...but is not useable " << endl;
#endif
      } else {
        assert(dynamic_cast<ParallelContext *>(context));
        return plugin->readCachedValue(
            info, currState, dynamic_cast<ParallelContext *>(context));
      }
    } else {
#ifdef DEBUGCACHING
      cout << "[Generator: ] No cache found for "
           << e->getOriginalRelationName() << "."
           << e->getAttribute().getAttrName() << "!" << endl;
#endif
    }
    //}
  }

  ProteusValue record = e->getExpr().accept(*this);

  // Resetting activeRelation here would break nested-record-projections
  if (plugin == nullptr) {
    string error_msg = string("[Expression Generator: ] No plugin provided");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  } else {
    Bindings bindings = {&currState, record};
    ProteusValueMemory mem_path;
    ProteusValueMemory mem_val;
    // cout << "Active RelationProj: " << activeRelation << "_" <<
    // e->getProjectionName() << endl;
    if (e->getProjectionName() != activeLoop) {
      const RecordType *exprType =
          dynamic_cast<const RecordType *>(e->getExpr().getExpressionType());
      if (exprType) {
        RecordAttribute attr = e->getAttribute();
        Value *val = projectArg(exprType, record.value, &attr, Builder);
        if (val) {
          ProteusValue valWrapper;
          valWrapper.value = val;
          valWrapper.isNull =
              record.isNull;  // FIXME: what if only one attribute is nullptr?
          return valWrapper;
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
      // cout << "PROJ: " << activeRelation << endl;
      Plugin *pg = catalog.getPlugin(activeRelation);
      RecordAttribute tupleIdentifier(activeRelation, activeLoop,
                                      pg->getOIDType());

      mem_path = currState[tupleIdentifier];
    }
    assert(dynamic_cast<ParallelContext *>(context));
    mem_val = plugin->readValue(mem_path, e->getExpressionType(),
                                dynamic_cast<ParallelContext *>(context));
    Value *val = Builder->CreateLoad(mem_val.mem);
    ProteusValue valWrapper;
    valWrapper.value = val;
    valWrapper.isNull = mem_val.isNull;
    return valWrapper;
  }
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::RecordConstruction *e) {
  IRBuilder<> *const Builder = context->getBuilder();

  std::vector<llvm::Value *> vals;
  llvm::Value *isNull = context->createFalse();
  for (const auto &attr : e->getAtts()) {
    auto v = attr.getExpression().accept(*this);
    vals.emplace_back(v.value);
    isNull = Builder->CreateOr(isNull, v.isNull);
  }

  llvm::Value *v = context->constructStruct(vals.begin(), vals.end());

  return {v, isNull};
}

// ProteusValue ExpressionGeneratorVisitor::visit(const
// expressions::RecordProjection *e) {
//    Catalog& catalog             = Catalog::getInstance();
//    IRBuilder<>* const Builder        = context->getBuilder();
//    activeRelation                     = e->getOriginalRelationName();
//    string projName                 = e->getProjectionName();
//
//    Plugin* plugin                     = catalog.getPlugin(activeRelation);
//
//    {
//        //if (plugin->getPluginType() != PGBINARY) {
//        /* Cache Logic */
//        /* XXX Apply in other visitors too! */
//        CachingService& cache = CachingService::getInstance();
//        //cout << "LOOKING FOR STUFF FROM "<<e->getRelationName() << "."
//        //<< e->getAttribute().getAttrName()<< endl;
//        CacheInfo info = cache.getCache(e);
//        if (info.structFieldNo != -1) {
//#ifdef DEBUGCACHING
//            cout << "[Generator: ] Expression found for "
//                    << e->getOriginalRelationName() << "."
//                    << e->getAttribute().getAttrName() << "!" << endl;
//#endif
//            if (!cache.getCacheIsFull(e)) {
//#ifdef DEBUGCACHING
//                cout << "...but is not useable " << endl;
//#endif
//            } else {
//                return plugin->readCachedValue(info, currState);
//            }
//        } else {
//#ifdef DEBUGCACHING
//            cout << "[Generator: ] No cache found for "
//                    << e->getOriginalRelationName() << "."
//                    << e->getAttribute().getAttrName() << "!" << endl;
//#endif
//        }
//        //}
//    }
//
//    ProteusValue record                    = e->getExpr()->accept(*this);
//    //Resetting activeRelation here would break nested-record-projections
//    //activeRelation = "";
//    if(plugin == nullptr)    {
//        string error_msg = string("[Expression Generator: ] No plugin
//        provided"); LOG(ERROR) << error_msg; throw runtime_error(error_msg);
//    }    else    {
//        Bindings bindings = { &currState, record };
//        ProteusValueMemory mem_path;
//        ProteusValueMemory mem_val;
//        //cout << "Active RelationProj: " << activeRelation << "_" <<
//        e->getProjectionName() << endl; if (e->getProjectionName() !=
//        activeLoop) {
//            //Path involves a projection / an object
//            mem_path = plugin->readPath(activeRelation, bindings,
//                    e->getProjectionName().c_str(),e->getAttribute());
//        } else {
//            //Path involves a primitive datatype
//            //(e.g., the result of unnesting a list of primitives)
//            //cout << "PROJ: " << activeRelation << endl;
//            Plugin* pg = catalog.getPlugin(activeRelation);
//            RecordAttribute tupleIdentifier = RecordAttribute(activeRelation,
//                    activeLoop,pg->getOIDType());
//            map<RecordAttribute, ProteusValueMemory>::const_iterator it =
//                    currState.getBindings().find(tupleIdentifier);
//            if (it == currState.getBindings().end()) {
//                string error_msg =
//                        "[Expression Generator: ] Current tuple binding not
//                        found";
//                LOG(ERROR)<< error_msg;
//                throw runtime_error(error_msg);
//            }
//            mem_path = it->second;
//        }
//        mem_val = plugin->readValue(mem_path, e->getExpressionType());
//        Value *val = Builder->CreateLoad(mem_val.mem);
//#ifdef DEBUG
//        {
//            /* Printing the pos. to be marked */
////            if(e->getProjectionName() == "age") {
////                cout << "AGE! " << endl;
////            if (e->getProjectionName() == "c2") {
////                cout << "C2! " << endl;
//            if (e->getProjectionName() == "dim") {
//                cout << "dim! " << endl;
//                map<RecordAttribute, ProteusValueMemory>::const_iterator it =
//                        currState.getBindings().begin();
//                for (; it != currState.getBindings().end(); it++) {
//                    string relname = it->first.getRelationName();
//                    string attrname = it->first.getAttrName();
//                    cout << "Available: " << relname << "_" << attrname <<
//                    endl;
//                }
//
//                /* Radix treats this as int64 (!) */
////                val->getType()->dump();
//                Function* debugInt = context->getFunction("printi");
//                Function* debugInt64 = context->getFunction("printi64");
//                vector<Value*> ArgsV;
//                ArgsV.clear();
//                ArgsV.push_back(val);
//                Builder->CreateCall(debugInt, ArgsV);
//            } else {
//                cout << "Other projection - " << e->getProjectionName() <<
//                endl;
//            }
//        }
//#endif
//        ProteusValue valWrapper;
//        valWrapper.value = val;
//        valWrapper.isNull = mem_val.isNull;
//        return valWrapper;
//    }
//}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::IfThenElse *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = TheBuilder->GetInsertBlock()->getParent();

  ProteusValue ifCond = e->getIfCond().accept(*this);

  // Prepare result
  //    AllocaInst* mem_result = context->CreateEntryBlockAlloca(F,
  //    "ifElseResult", (ifResult.value)->getType()); AllocaInst*
  //    mem_result_isNull = context->CreateEntryBlockAlloca(F,
  //    "ifElseResultIsNull", (ifResult.isNull)->getType());
  AllocaInst *mem_result = nullptr;
  AllocaInst *mem_result_isNull = nullptr;

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
  ProteusValue ifResult = e->getIfResult().accept(*this);
  mem_result = context->CreateEntryBlockAlloca(F, "ifElseResult",
                                               (ifResult.value)->getType());
  mem_result_isNull = context->CreateEntryBlockAlloca(
      F, "ifElseResultIsNull", (ifResult.isNull)->getType());

  TheBuilder->CreateStore(ifResult.value, mem_result);
  TheBuilder->CreateStore(ifResult.isNull, mem_result_isNull);
  TheBuilder->CreateBr(MergeBB);

  // else
  TheBuilder->SetInsertPoint(ElseBB);
  ProteusValue elseResult = e->getElseResult().accept(*this);
  TheBuilder->CreateStore(elseResult.value, mem_result);
  TheBuilder->CreateStore(elseResult.isNull, mem_result_isNull);
  TheBuilder->CreateBr(MergeBB);

  // cont.
  TheBuilder->SetInsertPoint(MergeBB);
  ProteusValue valWrapper;
  valWrapper.value = TheBuilder->CreateLoad(mem_result);
  valWrapper.isNull = TheBuilder->CreateLoad(mem_result_isNull);

  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::EqExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();

  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();

  typeID id = childType->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  switch (id) {
    case DSTRING:
    case DATE:
    case INT64:
    case INT: {
      valWrapper.value = TheBuilder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    }
    case FLOAT:
      valWrapper.value = TheBuilder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = TheBuilder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    // XXX Does this code work if we are iterating over a json primitive array?
    // Example: ["alpha","beta","gamma"]
    case STRING: {
      Function *stringEquality = context->getFunction("equalStringObjs");
      valWrapper.value = TheBuilder->CreateCall(
          stringEquality, {left.value, right.value}, "equalStringObjsCall");
      return valWrapper;
    }
    case RECORD: {
      Value *res = context->createTrue();
      const auto &al = dynamic_cast<const RecordType *>(
                           e->getLeftOperand().getExpressionType())
                           ->getArgs();
      const auto &ar = dynamic_cast<const RecordType *>(
                           e->getRightOperand().getExpressionType())
                           ->getArgs();
      auto it_al = al.begin();
      auto it_ar = ar.begin();
      assert(al.size() == ar.size() && al.size() > 0);
      for (size_t i = 0; i < al.size(); ++i, ++it_al, ++it_ar) {
        expressions::ProteusValueExpression e_l{
            (*it_al)->getOriginalType(),
            ProteusValue{TheBuilder->CreateExtractValue(left.value, i),
                         context->createFalse()}};
        expressions::ProteusValueExpression e_r{
            (*it_ar)->getOriginalType(),
            ProteusValue{TheBuilder->CreateExtractValue(right.value, i),
                         context->createFalse()}};
        auto r = eq(e_l, e_r).accept(*this);
        res = TheBuilder->CreateAnd(res, r.value);
      }

      return ProteusValue{res, context->createFalse()};
    }
      //        case STRING: {
      //            //{ i8*, i32 }
      //            StructType *stringObjType = context->CreateStringStruct();
      //            AllocaInst *mem_str_obj_left =
      //            context->CreateEntryBlockAlloca(F,
      //                                string("revertedStringObjLeft"),
      //                                stringObjType);
      //            AllocaInst *mem_str_obj_right =
      //            context->CreateEntryBlockAlloca(F,
      //                                            string("revertedStringObjRight"),
      //                                            stringObjType);
      //            TheBuilder->CreateStore(left.value,mem_str_obj_left);
      //            TheBuilder->CreateStore(right.value,mem_str_obj_right);
      //
      //            Value *ptr_s1 = context->getStructElem(mem_str_obj_left,0);
      //            Value *ptr_s2 = context->getStructElem(mem_str_obj_right,0);
      //            Value *len1 = context->getStructElem(mem_str_obj_left,1);
      //            Value *len2 = context->getStructElem(mem_str_obj_right,1);
      //
      //            Value *toCmp = context->createInt32(0);
      //            valWrapper.value =
      //            TheBuilder->CreateICmpEQ(mystrncmp(ptr_s1,ptr_s2,len2).value,toCmp);
      //            return valWrapper;
      //        }
    /*case STRING: {
        //{ i8*, i32 }
        StructType *stringObjType = context->CreateStringStruct();
        AllocaInst *mem_str_obj_left = context->CreateEntryBlockAlloca(F,
                string("revertedStringObjLeft"), stringObjType);
        AllocaInst *mem_str_obj_right = context->CreateEntryBlockAlloca(F,
                string("revertedStringObjRight"), stringObjType);
        TheBuilder->CreateStore(left.value, mem_str_obj_left);
        TheBuilder->CreateStore(right.value, mem_str_obj_right);

        Value *ptr_s1 = context->getStructElem(mem_str_obj_left, 0);
        Value *ptr_s2 = context->getStructElem(mem_str_obj_right, 0);
        Value *len1 = context->getStructElem(mem_str_obj_left, 1);
        Value *len2 = context->getStructElem(mem_str_obj_right, 1);


        this->declareLLVMFunc();
        IRBuilder<>* Builder = context->getBuilder();
        LLVMContext& llvmContext = context->getLLVMContext();
        Function *F = Builder->GetInsertBlock()->getParent();
        Module *mod = context->getModule();

        std::vector<Value*> int32_call_params;
        int32_call_params.push_back(ptr_s1);
        int32_call_params.push_back(ptr_s2);
        int32_call_params.push_back(len1);
        int32_call_params.push_back(len2);

        Function* func_mystrncmpllvm =
       context->getModule()->getFunction("mystrncmpllvm"); CallInst* int32_call
       = CallInst::Create(func_mystrncmpllvm, int32_call_params,
       "callStrncmp",Builder->GetInsertBlock());
        int32_call->setCallingConv(CallingConv::C);
        int32_call->setTailCall(false);
        cout << "Not Inlined?? " << int32_call->isNoInline() << endl;
        AttributeSet int32_call_PAL;
        int32_call->setAttributes(int32_call_PAL);


        Value *toCmp = context->createInt32(0);
        valWrapper.value = TheBuilder->CreateICmpEQ(
                int32_call, toCmp);
        return valWrapper;
            }*/
    case BAG:
    case LIST:
    case SET:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: invalid expression type"));
    default:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown Input"));
  }
}

int mystrncmp(const char *s1, const char *s2, size_t n) {
  for (; n > 0; s1++, s2++, --n)
    if (*s1 != *s2)
      return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
    else if (*s1 == '\0')
      return 0;
  return 0;
}

void ExpressionGeneratorVisitor::declareLLVMFunc() {
  Module *mod = context->getModule();

  // Type Definitions
  std::vector<Type *> FuncTy_0_args;
  IntegerType *IntegerTy_1 = IntegerType::get(mod->getContext(), 8);
  PointerType *PointerTy_1 = PointerType::get(IntegerTy_1, 0);

  IntegerType *IntegerTy_2 = IntegerType::get(mod->getContext(), 32);
  IntegerType *int32Ty = IntegerTy_2;

  FuncTy_0_args.push_back(PointerTy_1);
  FuncTy_0_args.push_back(PointerTy_1);
  FuncTy_0_args.push_back(IntegerTy_2);
  FuncTy_0_args.push_back(IntegerTy_2);
  FunctionType *FuncTy_0 = FunctionType::get(
      /*Result=*/IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_0_args,
      /*isVarArg=*/false);

  // Function Declarations
  Function *func_mystrncmpllvm = mod->getFunction("mystrncmpllvm");
  if (!func_mystrncmpllvm) {
    func_mystrncmpllvm = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"mystrncmpllvm", mod);
    func_mystrncmpllvm->setCallingConv(CallingConv::C);
  }

  AttributeList func_mystrncmpllvm_PAL;
  {
    AttrBuilder B;
    B.addAttribute(Attribute::NoUnwind);
    B.addAttribute(Attribute::AlwaysInline);

    func_mystrncmpllvm_PAL = AttributeList::get(
        mod->getContext(), AttributeList::AttrIndex::FunctionIndex, B);
  }
  func_mystrncmpllvm->setAttributes(func_mystrncmpllvm_PAL);
  func_mystrncmpllvm->addFnAttr(llvm::Attribute::AlwaysInline);

  // Global Variable Declarations

  // Constant Definitions
  ConstantInt *const_int32_4 =
      ConstantInt::get(mod->getContext(), APInt(32, StringRef("1"), 10));
  ConstantInt *const_int32_5 =
      ConstantInt::get(mod->getContext(), APInt(32, StringRef("-1"), 10));
  ConstantInt *const_int32_6 =
      ConstantInt::get(mod->getContext(), APInt(32, StringRef("0"), 10));

  // Global Variable Definitions

  // Function Definitions

  // Function: mystrncmpllvm (func_mystrncmpllvm)
  {
    Function::arg_iterator args = func_mystrncmpllvm->arg_begin();
    Value *ptr_s1 = &(*args);
    ptr_s1->setName("s1");
    Value *ptr_s2 = &(*++args);
    ptr_s2->setName("s2");
    Value *int32_n1 = &(*++args);
    int32_n1->setName("n1");
    Value *int32_n2 = &(*++args);
    int32_n2->setName("n2");

    BasicBlock *label_entry = BasicBlock::Create(mod->getContext(), "entry",
                                                 func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_then = BasicBlock::Create(mod->getContext(), "if.then",
                                                   func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_end = BasicBlock::Create(mod->getContext(), "if.end",
                                                  func_mystrncmpllvm, nullptr);
    BasicBlock *label_for_cond = BasicBlock::Create(
        mod->getContext(), "for.cond", func_mystrncmpllvm, nullptr);
    BasicBlock *label_for_body = BasicBlock::Create(
        mod->getContext(), "for.body", func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_then5 = BasicBlock::Create(
        mod->getContext(), "if.then5", func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_else = BasicBlock::Create(mod->getContext(), "if.else",
                                                   func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_then13 = BasicBlock::Create(
        mod->getContext(), "if.then13", func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_end14 = BasicBlock::Create(
        mod->getContext(), "if.end14", func_mystrncmpllvm, nullptr);
    BasicBlock *label_if_end15 = BasicBlock::Create(
        mod->getContext(), "if.end15", func_mystrncmpllvm, nullptr);
    BasicBlock *label_for_inc = BasicBlock::Create(mod->getContext(), "for.inc",
                                                   func_mystrncmpllvm, nullptr);
    BasicBlock *label_for_end = BasicBlock::Create(mod->getContext(), "for.end",
                                                   func_mystrncmpllvm, nullptr);
    BasicBlock *label_return = BasicBlock::Create(mod->getContext(), "return",
                                                  func_mystrncmpllvm, nullptr);

    // Block entry (label_entry)
    AllocaInst *ptr_retval =
        context->createAlloca(label_entry, "retval", int32Ty);
    ptr_retval->setAlignment(llvm::MaybeAlign(4));
    AllocaInst *ptr_s1_addr =
        context->createAlloca(label_entry, "s1.addr", PointerTy_1);
    ptr_s1_addr->setAlignment(llvm::MaybeAlign(8));
    AllocaInst *ptr_s2_addr =
        context->createAlloca(label_entry, "s2.addr", PointerTy_1);
    ptr_s2_addr->setAlignment(llvm::MaybeAlign(8));
    AllocaInst *ptr_n1_addr =
        context->createAlloca(label_entry, "n1.addr", int32Ty);
    ptr_n1_addr->setAlignment(llvm::MaybeAlign(4));
    AllocaInst *ptr_n2_addr =
        context->createAlloca(label_entry, "n2.addr", int32Ty);
    ptr_n2_addr->setAlignment(llvm::MaybeAlign(4));
    StoreInst *void_7 = new StoreInst(ptr_s1, ptr_s1_addr, false, label_entry);
    void_7->setAlignment(llvm::MaybeAlign(8));
    StoreInst *void_8 = new StoreInst(ptr_s2, ptr_s2_addr, false, label_entry);
    void_8->setAlignment(llvm::MaybeAlign(8));
    StoreInst *void_9 =
        new StoreInst(int32_n1, ptr_n1_addr, false, label_entry);
    void_9->setAlignment(llvm::MaybeAlign(4));
    StoreInst *void_10 =
        new StoreInst(int32_n2, ptr_n2_addr, false, label_entry);
    void_10->setAlignment(llvm::MaybeAlign(4));
    LoadInst *int32_11 = new LoadInst(ptr_n1_addr, "", false, label_entry);
    int32_11->setAlignment(llvm::MaybeAlign(4));
    LoadInst *int32_12 = new LoadInst(ptr_n2_addr, "", false, label_entry);
    int32_12->setAlignment(llvm::MaybeAlign(4));
    ICmpInst *int1_cmp = new ICmpInst(*label_entry, ICmpInst::ICMP_NE, int32_11,
                                      int32_12, "cmp");
    BranchInst::Create(label_if_then, label_if_end, int1_cmp, label_entry);

    // Block if.then (label_if_then)
    StoreInst *void_14 =
        new StoreInst(const_int32_5, ptr_retval, false, label_if_then);
    void_14->setAlignment(llvm::MaybeAlign(4));
    BranchInst::Create(label_return, label_if_then);

    // Block if.end (label_if_end)
    BranchInst::Create(label_for_cond, label_if_end);

    // Block for.cond (label_for_cond)
    LoadInst *int32_17 = new LoadInst(ptr_n1_addr, "", false, label_for_cond);
    int32_17->setAlignment(llvm::MaybeAlign(4));
    ICmpInst *int1_cmp1 = new ICmpInst(*label_for_cond, ICmpInst::ICMP_SGT,
                                       int32_17, const_int32_6, "cmp1");
    BranchInst::Create(label_for_body, label_for_end, int1_cmp1,
                       label_for_cond);

    // Block for.body (label_for_body)
    LoadInst *ptr_19 = new LoadInst(ptr_s1_addr, "", false, label_for_body);
    ptr_19->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int8_20 = new LoadInst(ptr_19, "", false, label_for_body);
    int8_20->setAlignment(llvm::MaybeAlign(1));
    CastInst *int32_conv =
        new SExtInst(int8_20, IntegerType::get(mod->getContext(), 32), "conv",
                     label_for_body);
    LoadInst *ptr_21 = new LoadInst(ptr_s2_addr, "", false, label_for_body);
    ptr_21->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int8_22 = new LoadInst(ptr_21, "", false, label_for_body);
    int8_22->setAlignment(llvm::MaybeAlign(1));
    CastInst *int32_conv2 =
        new SExtInst(int8_22, IntegerType::get(mod->getContext(), 32), "conv2",
                     label_for_body);
    ICmpInst *int1_cmp3 = new ICmpInst(*label_for_body, ICmpInst::ICMP_NE,
                                       int32_conv, int32_conv2, "cmp3");
    BranchInst::Create(label_if_then5, label_if_else, int1_cmp3,
                       label_for_body);

    // Block if.then5 (label_if_then5)
    LoadInst *ptr_24 = new LoadInst(ptr_s1_addr, "", false, label_if_then5);
    ptr_24->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int8_25 = new LoadInst(ptr_24, "", false, label_if_then5);
    int8_25->setAlignment(llvm::MaybeAlign(1));
    CastInst *int32_conv6 =
        new ZExtInst(int8_25, IntegerType::get(mod->getContext(), 32), "conv6",
                     label_if_then5);
    LoadInst *ptr_26 = new LoadInst(ptr_s2_addr, "", false, label_if_then5);
    ptr_26->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int8_27 = new LoadInst(ptr_26, "", false, label_if_then5);
    int8_27->setAlignment(llvm::MaybeAlign(1));
    CastInst *int32_conv7 =
        new ZExtInst(int8_27, IntegerType::get(mod->getContext(), 32), "conv7",
                     label_if_then5);
    ICmpInst *int1_cmp8 = new ICmpInst(*label_if_then5, ICmpInst::ICMP_SLT,
                                       int32_conv6, int32_conv7, "cmp8");
    SelectInst *int32_cond = SelectInst::Create(
        int1_cmp8, const_int32_5, const_int32_4, "cond", label_if_then5);
    StoreInst *void_28 =
        new StoreInst(int32_cond, ptr_retval, false, label_if_then5);
    void_28->setAlignment(llvm::MaybeAlign(4));
    BranchInst::Create(label_return, label_if_then5);

    // Block if.else (label_if_else)
    LoadInst *ptr_30 = new LoadInst(ptr_s1_addr, "", false, label_if_else);
    ptr_30->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int8_31 = new LoadInst(ptr_30, "", false, label_if_else);
    int8_31->setAlignment(llvm::MaybeAlign(1));
    CastInst *int32_conv10 =
        new SExtInst(int8_31, IntegerType::get(mod->getContext(), 32), "conv10",
                     label_if_else);
    ICmpInst *int1_cmp11 = new ICmpInst(*label_if_else, ICmpInst::ICMP_EQ,
                                        int32_conv10, const_int32_6, "cmp11");
    BranchInst::Create(label_if_then13, label_if_end14, int1_cmp11,
                       label_if_else);

    // Block if.then13 (label_if_then13)
    StoreInst *void_33 =
        new StoreInst(const_int32_6, ptr_retval, false, label_if_then13);
    void_33->setAlignment(llvm::MaybeAlign(4));
    BranchInst::Create(label_return, label_if_then13);

    // Block if.end14 (label_if_end14)
    BranchInst::Create(label_if_end15, label_if_end14);

    // Block if.end15 (label_if_end15)
    BranchInst::Create(label_for_inc, label_if_end15);

    // Block for.inc (label_for_inc)
    LoadInst *ptr_37 = new LoadInst(ptr_s1_addr, "", false, label_for_inc);
    ptr_37->setAlignment(llvm::MaybeAlign(8));
    GetElementPtrInst *ptr_incdec_ptr = GetElementPtrInst::Create(
        IntegerTy_1, ptr_37, const_int32_4, "incdec.ptr", label_for_inc);
    StoreInst *void_38 =
        new StoreInst(ptr_incdec_ptr, ptr_s1_addr, false, label_for_inc);
    void_38->setAlignment(llvm::MaybeAlign(8));
    LoadInst *ptr_39 = new LoadInst(ptr_s2_addr, "", false, label_for_inc);
    ptr_39->setAlignment(llvm::MaybeAlign(8));
    GetElementPtrInst *ptr_incdec_ptr16 = GetElementPtrInst::Create(
        IntegerTy_1, ptr_39, const_int32_4, "incdec.ptr16", label_for_inc);
    StoreInst *void_40 =
        new StoreInst(ptr_incdec_ptr16, ptr_s2_addr, false, label_for_inc);
    void_40->setAlignment(llvm::MaybeAlign(8));
    LoadInst *int32_41 = new LoadInst(ptr_n1_addr, "", false, label_for_inc);
    int32_41->setAlignment(llvm::MaybeAlign(4));
    auto int32_dec = llvm::BinaryOperator::Create(
        Instruction::Add, int32_41, const_int32_5, "dec", label_for_inc);
    StoreInst *void_42 =
        new StoreInst(int32_dec, ptr_n1_addr, false, label_for_inc);
    void_42->setAlignment(llvm::MaybeAlign(4));
    BranchInst::Create(label_for_cond, label_for_inc);

    // Block for.end (label_for_end)
    StoreInst *void_44 =
        new StoreInst(const_int32_6, ptr_retval, false, label_for_end);
    void_44->setAlignment(llvm::MaybeAlign(4));
    BranchInst::Create(label_return, label_for_end);

    // Block return (label_return)
    LoadInst *int32_46 = new LoadInst(ptr_retval, "", false, label_return);
    int32_46->setAlignment(llvm::MaybeAlign(4));
    ReturnInst::Create(mod->getContext(), int32_46, label_return);
  }

  /* Calling convention */
  //      AllocaInst* ptr_xx = new AllocaInst(PointerTy_1, "xx",
  //      label_entry_51); ptr_xx->setAlignment(llvm::MaybeAlign(8));
  //      AllocaInst* ptr_yy = new AllocaInst(PointerTy_1, "yy",
  //      label_entry_51); ptr_yy->setAlignment(llvm::MaybeAlign(8)); LoadInst*
  //      ptr_52 = new LoadInst(ptr_xx, "", false, label_entry_51);
  //      ptr_52->setAlignment(llvm::MaybeAlign(8));
  //      LoadInst* ptr_53 = new LoadInst(ptr_yy, "", false, label_entry_51);
  //      ptr_53->setAlignment(llvm::MaybeAlign(8));
  //      std::vector<Value*> int32_call_params;
  //      int32_call_params.push_back(ptr_52);
  //      int32_call_params.push_back(ptr_53);
  //      int32_call_params.push_back(const_int32_9);
  //      int32_call_params.push_back(const_int32_9);
  //      CallInst* int32_call = CallInst::Create(func_mystrncmpllvm,
  //      int32_call_params, "call", label_entry_51);
  //      int32_call->setCallingConv(CallingConv::C);
  //      int32_call->setTailCall(false);
  //      AttributeSet int32_call_PAL;
  //      int32_call->setAttributes(int32_call_PAL);
}

/* returns 0/1/-1 */
ProteusValue ExpressionGeneratorVisitor::mystrncmp(Value *ptr_s1, Value *ptr_s2,
                                                   Value *int32_n) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = Builder->GetInsertBlock()->getParent();
  PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);

  IntegerType *int32Ty = IntegerType::get(llvmContext, 32);

  // Constant Definitions
  ConstantInt *const_int32_4 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("1"), 10));
  ConstantInt *const_int32_5 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("-2"), 10));
  ConstantInt *const_int32_6 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("0"), 10));
  ConstantInt *const_int32_7 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("-1"), 10));

  BasicBlock *label_entry =
      BasicBlock::Create(llvmContext, "entry", F, nullptr);
  BasicBlock *label_for_cond =
      BasicBlock::Create(llvmContext, "for.cond", F, nullptr);
  BasicBlock *label_for_body =
      BasicBlock::Create(llvmContext, "for.body", F, nullptr);
  BasicBlock *label_if_then =
      BasicBlock::Create(llvmContext, "if.then", F, nullptr);
  BasicBlock *label_if_else =
      BasicBlock::Create(llvmContext, "if.else", F, nullptr);
  BasicBlock *label_if_then11 =
      BasicBlock::Create(llvmContext, "if.then11", F, nullptr);
  BasicBlock *label_if_end =
      BasicBlock::Create(llvmContext, "if.end", F, nullptr);
  BasicBlock *label_if_end12 =
      BasicBlock::Create(llvmContext, "if.end12", F, nullptr);
  BasicBlock *label_for_inc =
      BasicBlock::Create(llvmContext, "for.inc", F, nullptr);
  BasicBlock *label_for_end =
      BasicBlock::Create(llvmContext, "for.end", F, nullptr);
  BasicBlock *label_if_then16 =
      BasicBlock::Create(llvmContext, "if.then16", F, nullptr);
  BasicBlock *label_if_end17 =
      BasicBlock::Create(llvmContext, "if.end17", F, nullptr);

  /* Connect w. previous */
  Builder->CreateBr(label_entry);

  // Block entry (label_entry)
  AllocaInst *ptr_s1_addr =
      context->createAlloca(label_entry, "s1.addr", charPtrType);
  ptr_s1_addr->setAlignment(llvm::MaybeAlign(8));
  AllocaInst *ptr_s2_addr =
      context->createAlloca(label_entry, "s2.addr", charPtrType);
  ptr_s2_addr->setAlignment(llvm::MaybeAlign(8));
  AllocaInst *ptr_n_addr =
      context->createAlloca(label_entry, "n.addr", int32Ty);
  ptr_n_addr->setAlignment(llvm::MaybeAlign(4));
  AllocaInst *ptr_result =
      context->createAlloca(label_entry, "result", int32Ty);
  ptr_result->setAlignment(llvm::MaybeAlign(4));
  StoreInst *void_8 = new StoreInst(ptr_s1, ptr_s1_addr, false, label_entry);
  void_8->setAlignment(llvm::MaybeAlign(8));
  StoreInst *void_9 = new StoreInst(ptr_s2, ptr_s2_addr, false, label_entry);
  void_9->setAlignment(llvm::MaybeAlign(8));
  StoreInst *void_10 = new StoreInst(int32_n, ptr_n_addr, false, label_entry);
  void_10->setAlignment(llvm::MaybeAlign(4));
  StoreInst *void_11 =
      new StoreInst(const_int32_5, ptr_result, false, label_entry);
  void_11->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_cond, label_entry);

  // Block for.cond (label_for_cond)
  LoadInst *int32_13 = new LoadInst(ptr_n_addr, "", false, label_for_cond);
  int32_13->setAlignment(llvm::MaybeAlign(4));
  ICmpInst *int1_cmp = new ICmpInst(*label_for_cond, ICmpInst::ICMP_SGT,
                                    int32_13, const_int32_6, "cmp");
  BranchInst::Create(label_for_body, label_for_end, int1_cmp, label_for_cond);

  // Block for.body (label_for_body)
  LoadInst *ptr_15 = new LoadInst(ptr_s1_addr, "", false, label_for_body);
  ptr_15->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_16 = new LoadInst(ptr_15, "", false, label_for_body);
  int8_16->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv = new SExtInst(
      int8_16, IntegerType::get(llvmContext, 32), "conv", label_for_body);
  LoadInst *ptr_17 = new LoadInst(ptr_s2_addr, "", false, label_for_body);
  ptr_17->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_18 = new LoadInst(ptr_17, "", false, label_for_body);
  int8_18->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv1 = new SExtInst(
      int8_18, IntegerType::get(llvmContext, 32), "conv1", label_for_body);
  ICmpInst *int1_cmp2 = new ICmpInst(*label_for_body, ICmpInst::ICMP_NE,
                                     int32_conv, int32_conv1, "cmp2");
  BranchInst::Create(label_if_then, label_if_else, int1_cmp2, label_for_body);

  // Block if.then (label_if_then)
  LoadInst *ptr_20 = new LoadInst(ptr_s1_addr, "", false, label_if_then);
  ptr_20->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_21 = new LoadInst(ptr_20, "", false, label_if_then);
  int8_21->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv4 = new ZExtInst(
      int8_21, IntegerType::get(llvmContext, 32), "conv4", label_if_then);
  LoadInst *ptr_22 = new LoadInst(ptr_s2_addr, "", false, label_if_then);
  ptr_22->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_23 = new LoadInst(ptr_22, "", false, label_if_then);
  int8_23->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv5 = new ZExtInst(
      int8_23, IntegerType::get(llvmContext, 32), "conv5", label_if_then);
  ICmpInst *int1_cmp6 = new ICmpInst(*label_if_then, ICmpInst::ICMP_SLT,
                                     int32_conv4, int32_conv5, "cmp6");
  SelectInst *int32_cond = SelectInst::Create(
      int1_cmp6, const_int32_7, const_int32_4, "cond", label_if_then);
  StoreInst *void_24 =
      new StoreInst(int32_cond, ptr_result, false, label_if_then);
  void_24->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_end, label_if_then);

  // Block if.else (label_if_else)
  LoadInst *ptr_26 = new LoadInst(ptr_s1_addr, "", false, label_if_else);
  ptr_26->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_27 = new LoadInst(ptr_26, "", false, label_if_else);
  int8_27->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv8 = new SExtInst(
      int8_27, IntegerType::get(llvmContext, 32), "conv8", label_if_else);
  ICmpInst *int1_cmp9 = new ICmpInst(*label_if_else, ICmpInst::ICMP_EQ,
                                     int32_conv8, const_int32_6, "cmp9");
  BranchInst::Create(label_if_then11, label_if_end, int1_cmp9, label_if_else);

  // Block if.then11 (label_if_then11)
  StoreInst *void_29 =
      new StoreInst(const_int32_6, ptr_result, false, label_if_then11);
  void_29->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_end, label_if_then11);

  // Block if.end (label_if_end)
  BranchInst::Create(label_if_end12, label_if_end);

  // Block if.end12 (label_if_end12)
  BranchInst::Create(label_for_inc, label_if_end12);

  // Block for.inc (label_for_inc)
  LoadInst *ptr_33 = new LoadInst(ptr_s1_addr, "", false, label_for_inc);
  ptr_33->setAlignment(llvm::MaybeAlign(8));
  GetElementPtrInst *ptr_incdec_ptr =
      GetElementPtrInst::Create(charPtrType->getPointerElementType(), ptr_33,
                                const_int32_4, "incdec.ptr", label_for_inc);
  StoreInst *void_34 =
      new StoreInst(ptr_incdec_ptr, ptr_s1_addr, false, label_for_inc);
  void_34->setAlignment(llvm::MaybeAlign(8));
  LoadInst *ptr_35 = new LoadInst(ptr_s2_addr, "", false, label_for_inc);
  ptr_35->setAlignment(llvm::MaybeAlign(8));
  GetElementPtrInst *ptr_incdec_ptr13 =
      GetElementPtrInst::Create(charPtrType->getPointerElementType(), ptr_35,
                                const_int32_4, "incdec.ptr13", label_for_inc);
  StoreInst *void_36 =
      new StoreInst(ptr_incdec_ptr13, ptr_s2_addr, false, label_for_inc);
  void_36->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int32_37 = new LoadInst(ptr_n_addr, "", false, label_for_inc);
  int32_37->setAlignment(llvm::MaybeAlign(4));
  auto int32_dec = llvm::BinaryOperator::Create(
      Instruction::Add, int32_37, const_int32_7, "dec", label_for_inc);
  StoreInst *void_38 =
      new StoreInst(int32_dec, ptr_n_addr, false, label_for_inc);
  void_38->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_cond, label_for_inc);

  // Block for.end (label_for_end)
  LoadInst *int32_40 = new LoadInst(ptr_result, "", false, label_for_end);
  int32_40->setAlignment(llvm::MaybeAlign(4));
  ICmpInst *int1_cmp14 = new ICmpInst(*label_for_end, ICmpInst::ICMP_EQ,
                                      int32_40, const_int32_5, "cmp14");
  BranchInst::Create(label_if_then16, label_if_end17, int1_cmp14,
                     label_for_end);

  // Block if.then16 (label_if_then16)
  StoreInst *void_42 =
      new StoreInst(const_int32_6, ptr_result, false, label_if_then16);
  void_42->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_if_end17, label_if_then16);

  // Block if.end17 (label_if_end17)
  LoadInst *int32_44 = new LoadInst(ptr_result, "", false, label_if_end17);
  int32_44->setAlignment(llvm::MaybeAlign(4));

  /* OUTPUT */
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = int32_44;

  Builder->SetInsertPoint(label_if_end17);

  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::mystrncmp(Value *ptr_s1, Value *ptr_s2,
                                                   Value *int32_n1,
                                                   Value *int32_n2) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = Builder->GetInsertBlock()->getParent();
  PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);
  IntegerType *int32Ty = IntegerType::get(llvmContext, 32);

  // Constant Definitions
  ConstantInt *const_int32_4 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("1"), 10));
  ConstantInt *const_int32_5 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("-2"), 10));
  ConstantInt *const_int32_6 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("0"), 10));
  ConstantInt *const_int32_7 =
      ConstantInt::get(llvmContext, APInt(32, StringRef("-1"), 10));

  //    BasicBlock *label_size_entry = BasicBlock::Create(llvmContext,
  //    "sizeEntry", F, 0);
  BasicBlock *label_size_cond =
      BasicBlock::Create(llvmContext, "sizeCond", F, nullptr);
  BasicBlock *label_size_then =
      BasicBlock::Create(llvmContext, "sizeThen", F, nullptr);
  BasicBlock *label_size_else =
      BasicBlock::Create(llvmContext, "sizeElse", F, nullptr);
  BasicBlock *label_size_merge =
      BasicBlock::Create(llvmContext, "sizeMerge", F, nullptr);

  /* Connect w. previous */
  Builder->CreateBr(label_size_cond);
  AllocaInst *ptr_result =
      context->createAlloca(label_size_cond, "result", int32Ty);
  Value *size_cond = Builder->CreateICmpNE(int32_n1, int32_n2);

  Builder->CreateCondBr(size_cond, label_size_then, label_size_else);

  Builder->SetInsertPoint(label_size_then);
  Builder->CreateStore(const_int32_7, ptr_result);
  Builder->CreateBr(label_size_merge);

  Builder->SetInsertPoint(label_size_else);

  BasicBlock *label_entry =
      BasicBlock::Create(llvmContext, "entry", F, nullptr);
  BasicBlock *label_for_cond =
      BasicBlock::Create(llvmContext, "for.cond", F, nullptr);
  BasicBlock *label_for_body =
      BasicBlock::Create(llvmContext, "for.body", F, nullptr);
  BasicBlock *label_if_then =
      BasicBlock::Create(llvmContext, "if.then", F, nullptr);
  BasicBlock *label_if_else =
      BasicBlock::Create(llvmContext, "if.else", F, nullptr);
  BasicBlock *label_if_then11 =
      BasicBlock::Create(llvmContext, "if.then11", F, nullptr);
  BasicBlock *label_if_end =
      BasicBlock::Create(llvmContext, "if.end", F, nullptr);
  BasicBlock *label_if_end12 =
      BasicBlock::Create(llvmContext, "if.end12", F, nullptr);
  BasicBlock *label_for_inc =
      BasicBlock::Create(llvmContext, "for.inc", F, nullptr);
  BasicBlock *label_for_end =
      BasicBlock::Create(llvmContext, "for.end", F, nullptr);
  BasicBlock *label_if_then16 =
      BasicBlock::Create(llvmContext, "if.then16", F, nullptr);
  BasicBlock *label_if_end17 =
      BasicBlock::Create(llvmContext, "if.end17", F, nullptr);

  /* Connect w. previous */
  Builder->CreateBr(label_entry);

  // Block entry (label_entry)
  AllocaInst *ptr_s1_addr =
      context->createAlloca(label_entry, "s1.addr", charPtrType);
  ptr_s1_addr->setAlignment(llvm::MaybeAlign(8));
  AllocaInst *ptr_s2_addr =
      context->createAlloca(label_entry, "s2.addr", charPtrType);
  ptr_s2_addr->setAlignment(llvm::MaybeAlign(8));
  AllocaInst *ptr_n_addr =
      context->createAlloca(label_entry, "n.addr", int32Ty);
  ptr_n_addr->setAlignment(llvm::MaybeAlign(4));

  ptr_result->setAlignment(llvm::MaybeAlign(4));
  StoreInst *void_8 = new StoreInst(ptr_s1, ptr_s1_addr, false, label_entry);
  void_8->setAlignment(llvm::MaybeAlign(8));
  StoreInst *void_9 = new StoreInst(ptr_s2, ptr_s2_addr, false, label_entry);
  void_9->setAlignment(llvm::MaybeAlign(8));
  StoreInst *void_10 = new StoreInst(int32_n2, ptr_n_addr, false, label_entry);
  void_10->setAlignment(llvm::MaybeAlign(4));
  StoreInst *void_11 =
      new StoreInst(const_int32_5, ptr_result, false, label_entry);
  void_11->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_cond, label_entry);

  // Block for.cond (label_for_cond)
  LoadInst *int32_13 = new LoadInst(ptr_n_addr, "", false, label_for_cond);
  int32_13->setAlignment(llvm::MaybeAlign(4));
  ICmpInst *int1_cmp = new ICmpInst(*label_for_cond, ICmpInst::ICMP_SGT,
                                    int32_13, const_int32_6, "cmp");
  BranchInst::Create(label_for_body, label_for_end, int1_cmp, label_for_cond);

  // Block for.body (label_for_body)
  LoadInst *ptr_15 = new LoadInst(ptr_s1_addr, "", false, label_for_body);
  ptr_15->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_16 = new LoadInst(ptr_15, "", false, label_for_body);
  int8_16->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv = new SExtInst(
      int8_16, IntegerType::get(llvmContext, 32), "conv", label_for_body);
  LoadInst *ptr_17 = new LoadInst(ptr_s2_addr, "", false, label_for_body);
  ptr_17->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_18 = new LoadInst(ptr_17, "", false, label_for_body);
  int8_18->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv1 = new SExtInst(
      int8_18, IntegerType::get(llvmContext, 32), "conv1", label_for_body);
  ICmpInst *int1_cmp2 = new ICmpInst(*label_for_body, ICmpInst::ICMP_NE,
                                     int32_conv, int32_conv1, "cmp2");
  BranchInst::Create(label_if_then, label_if_else, int1_cmp2, label_for_body);

  // Block if.then (label_if_then)
  LoadInst *ptr_20 = new LoadInst(ptr_s1_addr, "", false, label_if_then);
  ptr_20->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_21 = new LoadInst(ptr_20, "", false, label_if_then);
  int8_21->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv4 = new ZExtInst(
      int8_21, IntegerType::get(llvmContext, 32), "conv4", label_if_then);
  LoadInst *ptr_22 = new LoadInst(ptr_s2_addr, "", false, label_if_then);
  ptr_22->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_23 = new LoadInst(ptr_22, "", false, label_if_then);
  int8_23->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv5 = new ZExtInst(
      int8_23, IntegerType::get(llvmContext, 32), "conv5", label_if_then);
  ICmpInst *int1_cmp6 = new ICmpInst(*label_if_then, ICmpInst::ICMP_SLT,
                                     int32_conv4, int32_conv5, "cmp6");
  SelectInst *int32_cond = SelectInst::Create(
      int1_cmp6, const_int32_7, const_int32_4, "cond", label_if_then);
  StoreInst *void_24 =
      new StoreInst(int32_cond, ptr_result, false, label_if_then);
  void_24->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_end, label_if_then);

  // Block if.else (label_if_else)
  LoadInst *ptr_26 = new LoadInst(ptr_s1_addr, "", false, label_if_else);
  ptr_26->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int8_27 = new LoadInst(ptr_26, "", false, label_if_else);
  int8_27->setAlignment(llvm::MaybeAlign(1));
  CastInst *int32_conv8 = new SExtInst(
      int8_27, IntegerType::get(llvmContext, 32), "conv8", label_if_else);
  ICmpInst *int1_cmp9 = new ICmpInst(*label_if_else, ICmpInst::ICMP_EQ,
                                     int32_conv8, const_int32_6, "cmp9");
  BranchInst::Create(label_if_then11, label_if_end, int1_cmp9, label_if_else);

  // Block if.then11 (label_if_then11)
  StoreInst *void_29 =
      new StoreInst(const_int32_6, ptr_result, false, label_if_then11);
  void_29->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_end, label_if_then11);

  // Block if.end (label_if_end)
  BranchInst::Create(label_if_end12, label_if_end);

  // Block if.end12 (label_if_end12)
  BranchInst::Create(label_for_inc, label_if_end12);

  // Block for.inc (label_for_inc)
  LoadInst *ptr_33 = new LoadInst(ptr_s1_addr, "", false, label_for_inc);
  ptr_33->setAlignment(llvm::MaybeAlign(8));
  GetElementPtrInst *ptr_incdec_ptr = GetElementPtrInst::Create(
      ptr_s1->getType(), ptr_33, const_int32_4, "incdec.ptr", label_for_inc);
  StoreInst *void_34 =
      new StoreInst(ptr_incdec_ptr, ptr_s1_addr, false, label_for_inc);
  void_34->setAlignment(llvm::MaybeAlign(8));
  LoadInst *ptr_35 = new LoadInst(ptr_s2_addr, "", false, label_for_inc);
  ptr_35->setAlignment(llvm::MaybeAlign(8));
  GetElementPtrInst *ptr_incdec_ptr13 = GetElementPtrInst::Create(
      ptr_s2->getType(), ptr_35, const_int32_4, "incdec.ptr13", label_for_inc);
  StoreInst *void_36 =
      new StoreInst(ptr_incdec_ptr13, ptr_s2_addr, false, label_for_inc);
  void_36->setAlignment(llvm::MaybeAlign(8));
  LoadInst *int32_37 = new LoadInst(ptr_n_addr, "", false, label_for_inc);
  int32_37->setAlignment(llvm::MaybeAlign(4));
  llvm::BinaryOperator *int32_dec = llvm::BinaryOperator::Create(
      Instruction::Add, int32_37, const_int32_7, "dec", label_for_inc);
  StoreInst *void_38 =
      new StoreInst(int32_dec, ptr_n_addr, false, label_for_inc);
  void_38->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_for_cond, label_for_inc);

  // Block for.end (label_for_end)
  LoadInst *int32_40 = new LoadInst(ptr_result, "", false, label_for_end);
  int32_40->setAlignment(llvm::MaybeAlign(4));
  ICmpInst *int1_cmp14 = new ICmpInst(*label_for_end, ICmpInst::ICMP_EQ,
                                      int32_40, const_int32_5, "cmp14");
  BranchInst::Create(label_if_then16, label_if_end17, int1_cmp14,
                     label_for_end);

  // Block if.then16 (label_if_then16)
  StoreInst *void_42 =
      new StoreInst(const_int32_6, ptr_result, false, label_if_then16);
  void_42->setAlignment(llvm::MaybeAlign(4));
  BranchInst::Create(label_if_end17, label_if_then16);

  /* FINAL MERGE */
  Builder->CreateBr(label_size_merge);

  LoadInst *int32_44 = new LoadInst(ptr_result, "", false, label_size_merge);
  int32_44->setAlignment(llvm::MaybeAlign(4));

  /* OUTPUT */
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = int32_44;

  Builder->SetInsertPoint(label_size_merge);

  return valWrapper;
}

/* returns true / false!!*/

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::NeExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  left.value->dump();
  right.value->dump();
  left.value->getType()->dump();
  right.value->getType()->dump();
  LOG(INFO) << *e->getLeftOperand().getExpressionType();
  LOG(INFO) << *e->getRightOperand().getExpressionType();

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case DSTRING:
      case INT64:
      case DATE:
      case INT:
        valWrapper.value = TheBuilder->CreateICmpNE(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFCmpONE(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateICmpNE(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::GeExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case DSTRING:
      case INT64:
      case DATE:
      case INT:
        valWrapper.value = TheBuilder->CreateICmpSGE(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFCmpOGE(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateICmpSGE(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::GtExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case DSTRING:
      case INT64:
      case DATE:
      case INT:
        valWrapper.value = TheBuilder->CreateICmpSGT(left.value, right.value);
        return valWrapper;
      case FLOAT:
#ifdef DEBUG
      {
        vector<Value *> ArgsV;
        ArgsV.clear();
        ArgsV.push_back(left.value);
        Function *debugF = context->getFunction("printFloat");
        TheBuilder->CreateCall(debugF, ArgsV);
      }
#endif
        valWrapper.value = TheBuilder->CreateFCmpOGT(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateICmpSGT(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::LeExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case DSTRING:
      case INT64:
      case DATE:
      case INT:
        valWrapper.value = TheBuilder->CreateICmpSLE(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFCmpOLE(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateICmpSLE(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::LtExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case DSTRING:
      case INT64:
      case DATE:
      case INT:
#ifdef DEBUG
      {
        vector<Value *> ArgsV;
        ArgsV.clear();
        ArgsV.push_back(left.value);
        Function *debugInt = context->getFunction("printi");
        TheBuilder->CreateCall(debugInt, ArgsV);
      }
#endif
        valWrapper.value = TheBuilder->CreateICmpSLT(left.value, right.value);
        return valWrapper;
      case FLOAT:
#ifdef DEBUG
      {
        vector<Value *> ArgsV;
        ArgsV.clear();
        ArgsV.push_back(left.value);
        Function *debugF = context->getFunction("printFloat");
        TheBuilder->CreateCall(debugF, ArgsV);
      }
#endif
        valWrapper.value = TheBuilder->CreateFCmpOLT(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateICmpSLT(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::AddExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  typeID id = childType->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT64:
    case DATE:
    case INT:
      valWrapper.value = TheBuilder->CreateAdd(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = TheBuilder->CreateFAdd(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = TheBuilder->CreateAdd(left.value, right.value);
      return valWrapper;
    case INDEXEDSEQ:
      valWrapper.value = TheBuilder->CreateGEP(
          left.value, {context->createInt64(0), right.value});
      return valWrapper;
    case STRING:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                    "supported yet";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: string operations not "
                 "supported yet"));
    case BAG:
    case LIST:
    case SET:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: invalid expression type"));
    case RECORD:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: invalid expression type"));
    default:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown Input"));
  }
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::SubExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
      case DATE:
      case INT:
        valWrapper.value = TheBuilder->CreateSub(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFSub(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateSub(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::MultExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
      case INT:
        valWrapper.value = TheBuilder->CreateMul(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFMul(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateMul(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::DivExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
      case INT:
        valWrapper.value = TheBuilder->CreateSDiv(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFDiv(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = TheBuilder->CreateSDiv(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::ModExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  if (childType->isPrimitive()) {
    typeID id = childType->getTypeID();
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();

    switch (id) {
      case INT64:
      case INT:
      case BOOL:
        valWrapper.value = TheBuilder->CreateSRem(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = TheBuilder->CreateFRem(left.value, right.value);
        return valWrapper;
      case STRING:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: string operations not "
                      "supported yet";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: string operations not "
                   "supported yet"));
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      case RECORD:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: invalid expression type";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: invalid expression type"));
      default:
        LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
        throw runtime_error(
            string("[ExpressionGeneratorVisitor]: Unknown Input"));
    }
  }
  throw runtime_error(
      string("[ExpressionGeneratorVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::AndExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateAnd(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::OrExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateOr(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::ShiftLeftExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateShl(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::ArithmeticShiftRightExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateAShr(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::LogicalShiftRightExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateLShr(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::XORExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  ProteusValue left = e->getLeftOperand().accept(*this);
  ProteusValue right = e->getRightOperand().accept(*this);
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  valWrapper.value = TheBuilder->CreateXor(left.value, right.value);
  return valWrapper;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::ProteusValueExpression *e) {
  return e->getValue();
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::MaxExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::MinExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::HashExpression *e) {
  ExpressionHasherVisitor h(context, currState);
  return e->getExpr().accept(h);
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::NegExpression *e) {
  ProteusValue v = e->getExpr().accept(*this);
  if (v.value->getType()->isFloatingPointTy()) {
    v.value = context->getBuilder()->CreateFNeg(v.value);
  } else {
    v.value = context->getBuilder()->CreateNeg(v.value);
  }
  return v;
}

int getYearDiv(expressions::extract_unit unit) {
  switch (unit) {
    case expressions::extract_unit::YEAR:
      return 1;
    case expressions::extract_unit::DECADE:
      return 10;
    case expressions::extract_unit::CENTURY:
      return 100;
    case expressions::extract_unit::MILLENNIUM:
      return 1000;
    default: {
      assert(false);
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown extract unit";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown extract unit"));
    }
  }
}

int getYearAbsOffset(expressions::extract_unit unit) {
  switch (unit) {
    case expressions::extract_unit::YEAR:
      return 0;
    case expressions::extract_unit::DECADE:
      return 0;
    case expressions::extract_unit::CENTURY:
      return 99;
    case expressions::extract_unit::MILLENNIUM:
      return 999;
    default: {
      assert(false);
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown extract unit";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown extract unit"));
    }
  }
}

/**
 * Mimic Calcite's julianExtract
 *
 * based on:
 * https://github.com/apache/calcite-avatica/blob/d86f3722885ea4fb8a13ef1f86976c0941576630/core/src/main/java/org/apache/calcite/avatica/util/DateTimeUtils.java#L746
 *
 * (
 * https://github.com/apache/calcite-avatica/blob/master/core/src/main/java/org/apache/calcite/avatica/util/DateTimeUtils.java
 * )
 */

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::ExtractExpression *e) {
  /**
   * Mimic Calcite's julianExtract
   *
   * based on:
   * https://github.com/apache/calcite-avatica/blob/d86f3722885ea4fb8a13ef1f86976c0941576630/core/src/main/java/org/apache/calcite/avatica/util/DateTimeUtils.java#L746
   *
   * (
   * https://github.com/apache/calcite-avatica/blob/master/core/src/main/java/org/apache/calcite/avatica/util/DateTimeUtils.java
   * )
   */
  IRBuilder<> *const TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();

  auto floorDiv = [=](Value *x, Value *y) {
    // java: long r = x / y;
    auto r = TheBuilder->CreateSDiv(x, y);

    // if ((x ^ y) < 0 && (r * y != x)) {
    //   r--;
    // }
    // return r;

    // ((x ^ y) < 0 && (r * y != x)) ? r - 1 : r
    return TheBuilder->CreateSelect(
        TheBuilder->CreateAnd(
            TheBuilder->CreateICmpSLT(TheBuilder->CreateXor(x, y),
                                      context->createInt64(0)),
            TheBuilder->CreateICmpNE(TheBuilder->CreateMul(r, y), x)),
        TheBuilder->CreateSub(r, context->createInt64(1)), r);
  };

  auto floorMod = [=](Value *x, Value *y) {
    // java: x - floorDiv(x, y) * y;
    return TheBuilder->CreateSub(x, TheBuilder->CreateMul(floorDiv(x, y), y));
  };

  auto v = e->getExpr().accept(*this);
  auto u = e->getExtractUnit();

  constexpr int32_t MS_PER_S{1000};
  constexpr int32_t MS_PER_M{60 * MS_PER_S};
  constexpr int32_t MS_PER_H{60 * MS_PER_M};
  constexpr int64_t MS_PER_DAY{24 * MS_PER_H};
  constexpr int64_t EPOCH_JULIAN{2440588};  // days

  auto julian = TheBuilder->CreateAdd(
      TheBuilder->CreateUDiv(v.value, context->createInt64(MS_PER_DAY)),
      context->createInt64(EPOCH_JULIAN));
  // java: int j = julian + 32044;
  auto j = TheBuilder->CreateAdd(julian, context->createInt64(32044));
  // java: int g = j / 146097;
  auto g = TheBuilder->CreateSDiv(j, context->createInt64(146097));
  // java: int dg = j % 146097;
  auto dg = TheBuilder->CreateSRem(j, context->createInt64(146097));
  // java: int c = (dg / 36524 + 1) * 3 / 4;
  auto c = TheBuilder->CreateSDiv(
      TheBuilder->CreateMul(
          TheBuilder->CreateAdd(
              TheBuilder->CreateSDiv(dg, context->createInt64(36524)),
              context->createInt64(1)),
          context->createInt64(3)),
      context->createInt64(4));
  // java: int dc = dg - c * 36524;
  auto dc = TheBuilder->CreateSub(
      dg, TheBuilder->CreateMul(c, context->createInt64(36524)));
  // java: int b = dc / 1461;
  auto b = TheBuilder->CreateSDiv(dc, context->createInt64(1461));
  // java: int db = dc % 1461;
  auto db = TheBuilder->CreateSRem(dc, context->createInt64(1461));
  // java: int a = (db / 365 + 1) * 3 / 4;
  auto a = TheBuilder->CreateSDiv(
      TheBuilder->CreateMul(
          TheBuilder->CreateAdd(
              TheBuilder->CreateSDiv(db, context->createInt64(365)),
              context->createInt64(1)),
          context->createInt64(3)),
      context->createInt64(4));
  // java: int da = db - a * 365;
  auto da = TheBuilder->CreateSub(
      db, TheBuilder->CreateMul(a, context->createInt64(365)));

  // integer number of full years elapsed since March 1, 4801 BC
  // java: int y = g * 400 + c * 100 + b * 4 + a;
  auto y = TheBuilder->CreateAdd(
      TheBuilder->CreateAdd(  // g * 400 + c * 100
          TheBuilder->CreateMul(g, context->createInt64(400)),
          TheBuilder->CreateMul(c, context->createInt64(100))),
      TheBuilder->CreateAdd(  // b * 4 + a
          TheBuilder->CreateMul(b, context->createInt64(4)), a));

  // integer number of full months elapsed since the last March 1
  // java: int m = (da * 5 + 308) / 153 - 2;
  auto m = TheBuilder->CreateSub(
      TheBuilder->CreateSDiv(
          TheBuilder->CreateAdd(
              TheBuilder->CreateMul(da, context->createInt64(5)),
              context->createInt64(308)),
          context->createInt64(153)),
      context->createInt64(2));

  // number of days elapsed since day 1 of the month
  // java: int d = da - (m + 4) * 153 / 5 + 122;
  auto d = TheBuilder->CreateAdd(
      TheBuilder->CreateSub(
          da, TheBuilder->CreateSDiv(
                  TheBuilder->CreateMul(
                      TheBuilder->CreateAdd(m, context->createInt64(4)),
                      context->createInt64(153)),
                  context->createInt64(5))),
      context->createInt64(122));
  // java: int year = y - 4800 + (m + 2) / 12;
  auto year = TheBuilder->CreateAdd(
      TheBuilder->CreateSub(y, context->createInt64(4800)),
      TheBuilder->CreateSDiv(TheBuilder->CreateAdd(m, context->createInt64(2)),
                             context->createInt64(12)));
  // java: int month = (m + 2) % 12 + 1;
  auto month = TheBuilder->CreateAdd(
      TheBuilder->CreateSRem(TheBuilder->CreateAdd(m, context->createInt64(2)),
                             context->createInt64(12)),
      context->createInt64(1));
  // java: int day = d + 1;
  auto day = TheBuilder->CreateAdd(d, context->createInt64(1));

  auto vtime = TheBuilder->CreateTrunc(
      floorMod(v.value, context->createInt64(MS_PER_DAY)),
      Type::getInt32Ty(llvmContext));

  switch (u) {
    case expressions::extract_unit::MILLISECOND: {
      v.value = TheBuilder->CreateURem(vtime, context->createInt32(MS_PER_S));
      break;
    }
    case expressions::extract_unit::SECOND: {
      v.value = TheBuilder->CreateURem(
          TheBuilder->CreateUDiv(vtime, context->createInt32(MS_PER_S)),
          context->createInt32(60));
      break;
    }
    case expressions::extract_unit::MINUTE: {
      v.value = TheBuilder->CreateURem(
          TheBuilder->CreateUDiv(vtime, context->createInt32(MS_PER_M)),
          context->createInt32(60));
      break;
    }
    case expressions::extract_unit::HOUR: {
      v.value = TheBuilder->CreateUDiv(vtime, context->createInt32(MS_PER_H));
      break;
    }
    case expressions::extract_unit::DAYOFWEEK: {  // sun = 1, sat = 7
      v.value = TheBuilder->CreateAdd(
          floorMod(TheBuilder->CreateAdd(julian, context->createInt64(1)),
                   context->createInt64(7)),
          context->createInt64(1));
      break;
    }
    case expressions::extract_unit::ISO_DAYOFWEEK: {  // mon = 1, sun = 7
      v.value = TheBuilder->CreateAdd(floorMod(julian, context->createInt64(7)),
                                      context->createInt64(1));
      break;
    }
    case expressions::extract_unit::DAYOFMONTH: {  // DAY
      v.value = day;
      break;
    }
    case expressions::extract_unit::DAYOFYEAR: {
      //    int a = (14 - month) / 12;
      //    int y = year + 4800 - a;
      //    int m = month + 12 * a - 3;
      //    return day + (153 * m + 2) / 5
      //        + 365 * y
      //        + y / 4
      //        - y / 100
      //        + y / 400
      //        - 32045;

      //    int a = (14 - 1) / 12;
      //    int y = year + 4800 - a;
      //    int m = 1 + 12 * a - 3;
      //    return 1 + (153 * m + 2) / 5
      //        + 365 * y
      //        + y / 4
      //        - y / 100
      //        + y / 400
      //        - 32045;

      //    int y = year + 4799;
      //    return 365 * y
      //        + y / 4
      //        - y / 100
      //        + y / 400
      //        - 31738;
      auto y = TheBuilder->CreateAdd(year, context->createInt64(4799));
      v.value = TheBuilder->CreateAdd(
          TheBuilder->CreateSub(
              TheBuilder->CreateAdd(
                  TheBuilder->CreateMul(context->createInt64(365), y),
                  TheBuilder->CreateSDiv(y, context->createInt64(4))),
              TheBuilder->CreateSDiv(y, context->createInt64(100))),
          TheBuilder->CreateSub(  // y / 400 - 31738
              TheBuilder->CreateSDiv(y, context->createInt64(400)),
              context->createInt64(31738)));
      break;
    }
    case expressions::extract_unit::WEEK: {
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown extract unit";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown extract unit"));
    }
    case expressions::extract_unit::MONTH: {
      v.value = month;
      break;
    }
    case expressions::extract_unit::QUARTER: {
      v.value = TheBuilder->CreateUDiv(
          TheBuilder->CreateAdd(month, context->createInt64(2)),
          context->createInt64(3));
      break;
    }
    case expressions::extract_unit::ISO_YEAR: {
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown extract unit";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown extract unit"));
    }
    case expressions::extract_unit::YEAR:
    case expressions::extract_unit::DECADE:
    case expressions::extract_unit::CENTURY:
    case expressions::extract_unit::MILLENNIUM: {
      int aoff = getYearAbsOffset(u);
      Value *soff = TheBuilder->CreateSelect(
          TheBuilder->CreateICmpSGT(year, context->createInt64(0)),
          context->createInt64(+aoff), context->createInt64(-aoff));
      Value *fxty = TheBuilder->CreateAdd(year, soff);
      v.value =
          TheBuilder->CreateSDiv(fxty, context->createInt64(getYearDiv(u)));
      break;
    }
    default: {
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown extract unit";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown extract unit"));
    }
  }

  v.value = TheBuilder->CreateSExtOrTrunc(
      v.value, e->getExpressionType()->getLLVMType(llvmContext));
  return v;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::TestNullExpression *e) {
  ProteusValue v = e->getExpr().accept(*this);
  Value *result;
  if (e->isNullTest())
    result = v.isNull;
  else
    result = context->getBuilder()->CreateNot(v.isNull);
  return ProteusValue{result, context->createFalse()};
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::CastExpression *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();

  ProteusValue v = e->getExpr().accept(*this);

  const ExpressionType *etype_to = e->getExpressionType();
  llvm::Type *ltype_to = etype_to->getLLVMType(llvmContext);

  const ExpressionType *etype_from = e->getExpr().getExpressionType();
  llvm::Type *ltype_from = etype_from->getLLVMType(llvmContext);

  bool fromFP = ltype_from->isIntegerTy();
  bool toFP = ltype_to->isIntegerTy();

  if (toFP && fromFP) {
    v.value = TheBuilder->CreateSExtOrTrunc(v.value, ltype_to);
  } else if (!toFP && fromFP) {
    v.value = TheBuilder->CreateSIToFP(v.value, ltype_to);
  } else if (toFP && !fromFP) {
    v.value = TheBuilder->CreateFPToSI(v.value, ltype_to);
  } else {
    v.value = TheBuilder->CreateFPCast(v.value, ltype_to);
  }

  return v;
}

ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::RefExpression *e) {
  auto ptr = e->getExpr().accept(*this);

  return {context->getBuilder()->CreateLoad(ptr.value), ptr.isNull};
}
ProteusValue ExpressionGeneratorVisitor::visit(
    const expressions::AssignExpression *e) {
  auto ptr = e->getRef().getExpr().accept(*this);
  auto val = e->getExpr().accept(*this);

  context->getBuilder()->CreateStore(val.value, ptr.value);

  return ptr;
}

#pragma pop_macro("DEBUG")  // FIXME: REMOVE!!! used to disable prints, as they
                            // are currently undefined for the gpu side
