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

#include "expressions/expressions-hasher.hpp"

#include <expressions/expressions/ref-expression.hpp>
#include <util/project-record.hpp>

#include "operators/operators.hpp"

using namespace llvm;

ProteusValue ExpressionHasherVisitor::hashInt32(
    const expressions::Expression *e) {
  ExpressionGeneratorVisitor exprGenerator{context, currState};
  return ::hashInt32(e->accept(exprGenerator), context);
}

ProteusValue hashInt32(ProteusValue v, Context *context) {
  IRBuilder<> *Builder = context->getBuilder();

  // FIXME: hash-function (single step murmur) is designed for 32bit inputs

  assert(v.value->getType()->isIntegerTy(32));

  Type *int64Type = Type::getInt64Ty(context->getLLVMContext());

  Value *key = v.value;
  Value *hash = Builder->CreateSExt(key, int64Type);

  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));
  hash = Builder->CreateMul(hash, context->createInt64(0x85ebca6b));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 13));
  hash = Builder->CreateMul(hash, context->createInt64(0xc2b2ae35));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));

  return ProteusValue{hash, v.isNull};
}

ProteusValue ExpressionHasherVisitor::hashInt64(
    const expressions::Expression *e) {
  ExpressionGeneratorVisitor exprGenerator{context, currState};
  return ::hashInt64(e->accept(exprGenerator), context);
}

ProteusValue hashInt64(ProteusValue v, Context *context) {
  IRBuilder<> *Builder = context->getBuilder();

  // FIXME: hash-function (single step murmur) is designed for 32bit inputs

  assert(v.value->getType()->isIntegerTy(64));

  Type *int64Type = Type::getInt64Ty(context->getLLVMContext());

  Value *key = v.value;
  Value *hash = Builder->CreateSExt(key, int64Type);

  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 33));
  hash = Builder->CreateMul(hash, context->createInt64(0xff51afd7ed558ccd));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 33));
  hash = Builder->CreateMul(hash, context->createInt64(0xc4ceb9fe1a85ec53));
  hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 33));

  return ProteusValue{hash, v.isNull};
}

ProteusValue ExpressionHasherVisitor::hashPrimitive(
    const expressions::Expression *e) {
  auto type = e->getExpressionType();

  assert(type->isPrimitive() &&
         "Non-primitive types should be handled by the "
         "corresponding visit function!");

  ExpressionGeneratorVisitor exprGenerator{context, currState};
  return ::hashPrimitive(e->accept(exprGenerator), type->getTypeID(), context);
}

ProteusValue hashPrimitive(ProteusValue v, typeID type, Context *context) {
  std::string instructionLabel;

  switch (type) {
    case INT:
    case DSTRING: {
      return hashInt32(v, context);
    }
    case INT64:
    case DATE:
      return hashInt64(v, context);
      break;
    case FLOAT:
      instructionLabel = "hashDouble";
      break;
    case BOOL:
      instructionLabel = "hashBoolean";
      break;
    case STRING: {
      IRBuilder<> *Builder = context->getBuilder();
      Function *hashStringObj = context->getFunction("hashStringObject");

      Value *hashResult = Builder->CreateCall(hashStringObj, v.value);

      return ProteusValue{hashResult, context->createFalse()};
    }
    case BAG:
    case LIST:
    case SET:
    case COMPOSITE:
    case RECORD:
    case BLOCK:
      LOG(ERROR) << "[ExpressionHasherVisitor]: invalid expression type";
      throw runtime_error(
          string("[ExpressionHasherVisitor]: invalid expression type"));
    default:
      LOG(ERROR) << "[ExpressionHasherVisitor]: Unknown Input";
      throw runtime_error(string("[ExpressionHasherVisitor]: Unknown Input"));
  }

  IRBuilder<> *const TheBuilder = context->getBuilder();
  Function *hashFunc = context->getFunction(instructionLabel);
  Value *hashResult = TheBuilder->CreateCall(hashFunc, v.value);

  return ProteusValue{hashResult, v.isNull};
}

ProteusValue ExpressionHasherVisitor::visit(const expressions::IntConstant *e) {
  return hashInt32(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::DStringConstant *e) {
  return hashInt32(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::Int64Constant *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::DateConstant *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::FloatConstant *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::BoolConstant *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::StringConstant *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  size_t hashResultC = hashString(e->getVal());
  Value *hashResult =
      ConstantInt::get(context->getLLVMContext(), APInt(64, hashResultC));

  ProteusValue valWrapper;
  valWrapper.value = hashResult;
  valWrapper.isNull = context->createFalse();
#ifdef DEBUG_HASH
  vector<Value *> argsV;
  argsV.clear();
  argsV.push_back(hashResult);
  Function *debugInt64 = context->getFunction("printi64");
  TheBuilder->CreateCall(debugInt64, argsV);
#endif
  return valWrapper;
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::InputArgument *e) {
  Catalog &catalog = Catalog::getInstance();
  Function *const F = context->getGlobalFunction();
  IRBuilder<> *const TheBuilder = context->getBuilder();
  Type *int64Type = Type::getInt64Ty(context->getLLVMContext());

  Function *hashCombine = context->getFunction("combineHashes");
  Value *hashedValue = context->createInt64(0);
  vector<Value *> ArgsV;

  const map<RecordAttribute, ProteusValueMemory> &activeVars =
      currState.getBindings();

  list<RecordAttribute> projections = e->getProjections();
  list<RecordAttribute>::iterator it = projections.begin();

  // Initializing resulting hashed value
  AllocaInst *mem_hashedValue =
      context->CreateEntryBlockAlloca(F, string("hashValue"), int64Type);
  TheBuilder->CreateStore(hashedValue, mem_hashedValue);

  // Is there a case that I am doing extra work?
  // Example: Am I hashing a value that is nested
  // in some value that I have already hashed?

  for (; it != projections.end(); it++) {
    /* Explicitly looking for OID!!! */
    if (it->getAttrName() == activeLoop) {
      map<RecordAttribute, ProteusValueMemory>::const_iterator itBindings;
      for (itBindings = activeVars.begin(); itBindings != activeVars.end();
           itBindings++) {
        RecordAttribute currAttr = itBindings->first;
        if (currAttr.getRelationName() == it->getRelationName() &&
            currAttr.getAttrName() == activeLoop) {
          // Hash value now
          ProteusValueMemory mem_activeTuple = itBindings->second;

          Plugin *plugin = catalog.getPlugin(currAttr.getRelationName());
          if (plugin == nullptr) {
            string error_msg =
                string("[Expression Hasher: ] No plugin provided");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
          }

          // Combine with previous hashed values (if applicable)
          // Does order matter?
          // Does result of retrieving tuple1->tuple2 differ from
          // tuple2->tuple1??? (Probably should)
          ProteusValue partialHash = plugin->hashValue(
              mem_activeTuple, e->getExpressionType(), context);
          ArgsV.clear();
          ArgsV.push_back(hashedValue);
          ArgsV.push_back(partialHash.value);

          hashedValue =
              TheBuilder->CreateCall(hashCombine, ArgsV, "combineHashesRes");
          TheBuilder->CreateStore(hashedValue, mem_hashedValue);
          break;
        }
      }
    }
  }

  ProteusValue hashValWrapper;
  hashValWrapper.value = TheBuilder->CreateLoad(mem_hashedValue);
  hashValWrapper.isNull = context->createFalse();
  return hashValWrapper;
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::ProteusValueExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);

  Catalog &catalog = Catalog::getInstance();

  Plugin *plugin = catalog.getPlugin(activeRelation);

  // Resetting activeRelation here would break nested-record-projections
  // activeRelation = "";
  if (plugin == nullptr) {
    string error_msg = string("[Expression Generator: ] No plugin provided");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  } else {
    return plugin->hashValueEager(e->getValue(), e->getExpressionType(),
                                  context);
  }
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::RecordProjection *e) {
  Catalog &catalog = Catalog::getInstance();
  activeRelation = e->getOriginalRelationName();

  ExpressionGeneratorVisitor exprGenerator =
      ExpressionGeneratorVisitor(context, currState);
  /* Need this 'hint' before launching generator,
   * otherwise (potential) InputArg visitor will crash */
  exprGenerator.setActiveRelation(activeRelation);
  ProteusValue record = e->getExpr().accept(exprGenerator);
  Plugin *plugin = catalog.getPlugin(activeRelation);

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
    map<RecordAttribute, ProteusValueMemory>::const_iterator it =
        currState.getBindings().find(e->getAttribute());
    if (info.structFieldNo != -1 && it == currState.getBindings().end()) {
#ifdef DEBUGCACHING
      cout << "[Hasher: ] Expression found for " << e->getOriginalRelationName()
           << "." << e->getAttribute().getAttrName() << "!" << endl;
#endif
      if (!cache.getCacheIsFull(e)) {
#ifdef DEBUGCACHING
        cout << "...but is not useable " << endl;
#endif
      } else {
        ProteusValue tmpWrapper = plugin->readCachedValue(info, currState);
        //                Value *tmp = tmpWrapper.value;
        //                AllocaInst *mem_tmp =
        //                context->CreateEntryBlockAlloca(F, "mem_cachedToHash",
        //                tmp->getType()); TheBuilder->CreateStore(tmp,mem_tmp);
        //                ProteusValueMemory mem_tmpWrapper = { mem_tmp,
        //                tmpWrapper.isNull }; ProteusValue mem_val =
        //                plugin->hashValue(mem_tmpWrapper,
        //                e->getExpressionType());
        ProteusValue mem_val =
            plugin->hashValueEager(tmpWrapper, e->getExpressionType(), context);
        return mem_val;
      }
    } else {
#ifdef DEBUGCACHING
      cout << "[Hasher: ] No cache found for " << e->getOriginalRelationName()
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

          return plugin->hashValueEager(valWrapper, attr.getOriginalType(),
                                        context);
        }
      }

      // Path involves a projection / an object
      mem_path =
          plugin->readPath(activeRelation, bindings,
                           e->getProjectionName().c_str(), e->getAttribute());
    } else {
      // Path involves a primitive datatype
      //(e.g., the result of unnesting a list of primitives)
      Plugin *pg = catalog.getPlugin(activeRelation);
      RecordAttribute tupleIdentifier =
          RecordAttribute(activeRelation, activeLoop, pg->getOIDType());
      map<RecordAttribute, ProteusValueMemory>::const_iterator it =
          currState.getBindings().find(tupleIdentifier);
      if (it == currState.getBindings().end()) {
        string error_msg =
            "[Expression Generator: ] Current tuple binding not found";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
      mem_path = it->second;
    }
    return plugin->hashValue(mem_path, e->getExpressionType(), context);
  }
}

ProteusValue ExpressionHasherVisitor::visit(const expressions::IfThenElse *e) {
  IRBuilder<> *const TheBuilder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = TheBuilder->GetInsertBlock()->getParent();
  Type *int64Type = Type::getInt64Ty(llvmContext);

  ProteusValue hashResult;
  AllocaInst *mem_hashResult =
      context->CreateEntryBlockAlloca(F, "hashResult", int64Type);

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
  hashResult = e->getIfResult().accept(*this);
  TheBuilder->CreateStore(hashResult.value, mem_hashResult);
  TheBuilder->CreateBr(MergeBB);

  // else
  TheBuilder->SetInsertPoint(ElseBB);
  hashResult = e->getElseResult().accept(*this);
  TheBuilder->CreateStore(hashResult.value, mem_hashResult);

  TheBuilder->CreateBr(MergeBB);

  // cont.
  TheBuilder->SetInsertPoint(MergeBB);
  ProteusValue valWrapper;
  valWrapper.value = TheBuilder->CreateLoad(mem_hashResult);
  valWrapper.isNull = context->createFalse();

  return valWrapper;
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::EqExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::NeExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::GeExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::GtExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::LeExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::LtExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::AndExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::OrExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::AddExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::SubExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::MultExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::DivExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::ModExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of binary "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::RecordConstruction *e) {
  IRBuilder<> *const Builder = context->getBuilder();
  Type *int64Type = Type::getInt64Ty(context->getLLVMContext());

  Value *seed = context->createInt64(0);
  Value *comb = ConstantInt::get(int64Type, 0x9e3779b9);

  for (const auto &attr : e->getAtts()) {
    expression_t expr = attr.getExpression();
    Value *hv = expr.accept(*this).value;

    hv = Builder->CreateAdd(hv, comb);
    hv = Builder->CreateAdd(hv, Builder->CreateShl(seed, 6));
    hv = Builder->CreateAdd(hv, Builder->CreateLShr(seed, 2));
    hv = Builder->CreateXor(hv, seed);

    seed = hv;
  }

  return ProteusValue{seed, context->createFalse()};
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::MaxExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::MinExpression *e) {
  return e->getCond()->accept(*this);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::HashExpression *e) {
  assert(false && "This does not really make sense... Why to hash a hash?");
  return e->getExpr().accept(*this);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::NegExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of negate "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::ExtractExpression *e) {
  if (e->getExpressionType()->isPrimitive()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: input of extract "
             "expression can only be primitive"));
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::TestNullExpression *e) {
  return hashPrimitive(e);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::CastExpression *e) {
  // do not _just_ cast the inner expression, as hash may be different
  // between the casted and non-casted expressions
  // instead, hash the casted expression
  if (e->getExpressionType()) return hashPrimitive(e);
  throw runtime_error(
      string("[ExpressionHasherVisitor]: output of cast "
             "expression can only be primitive"));
}
ProteusValue ExpressionHasherVisitor::visit(
    const expressions::RefExpression *e) {
  return e->getExpr().accept(*this);
}

ProteusValue ExpressionHasherVisitor::visit(
    const expressions::AssignExpression *) {
  throw std::runtime_error(
      "[ExpressionHasherVisitor]: unsupported hashing of assignment operation");
}
