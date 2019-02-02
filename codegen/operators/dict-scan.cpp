/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "operators/dict-scan.hpp"
#include "llvm/IR/TypeBuilder.h"
#include "storage/storage-manager.hpp"

#include <regex>

using namespace llvm;

class DictMatchIter {
  typedef std::map<int, std::string> dict_t;

  typedef ptrdiff_t difference_type;
  typedef const int32_t value_type;
  typedef const int32_t &reference;
  typedef const int32_t *pointer;
  typedef std::input_iterator_tag iterator_category;

  dict_t *dict;
  dict_t::iterator curr;
  dict_t::iterator end;

  int32_t val;
  const std::regex rex;

 public:
  DictMatchIter(const DictScan &dictscan, std::string r) : rex(r) {
    const auto &attr = dictscan.getAttr();
    const auto &path = attr.getRelationName() + "." + attr.getAttrName();
    dict = (dict_t *)StorageManager::getDictionaryOf(path);
    assert(dict);

    curr = dict->begin();
    end = dict->end();

    while (curr != end && !std::regex_match(curr->second, rex)) ++curr;
    if (curr != end) val = curr->first;
  }

  /* end of dict */
  DictMatchIter(const DictScan &dictscan) : rex("") {
    const auto &attr = dictscan.getAttr();
    const auto &path = attr.getRelationName() + "." + attr.getAttrName();
    dict = (dict_t *)StorageManager::getDictionaryOf(path);
    assert(dict);

    curr = end = dict->end();
  }

  DictMatchIter operator++() {
    // be carefull with the order of evaluation! it matters!
    while (++curr != end && !std::regex_match(curr->second, rex))
      ;
    if (curr != end) val = curr->first;
    return *this;
  }

  value_type operator*() const { return val; }

  pointer operator->() const { return &val; }

  bool operator==(const DictMatchIter &other) { return curr == other.curr; }

  bool operator!=(const DictMatchIter &other) { return curr != other.curr; }
};

DictMatchIter DictScan::begin() const { return DictMatchIter{*this, regex}; }

DictMatchIter DictScan::end() const { return DictMatchIter{*this}; }

extern "C" void initDictScan(const DictScan *dict, DictMatchIter *curr,
                             DictMatchIter *end) {
  new (curr) DictMatchIter(*dict, dict->getRegex());
  new (end) DictMatchIter(*dict);
}

extern "C" bool hasNextDictScan(DictMatchIter *curr, DictMatchIter *end) {
  return *curr != *end;
}

extern "C" int32_t getDictScan(DictMatchIter *begin) { return **begin; }

extern "C" void nextDictScan(DictMatchIter *begin) { ++(*begin); }

void DictScan::produce() {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  size_t iter_size = sizeof(DictMatchIter) * 8;

  context->setGlobalFunction(true);
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  Function *F = context->getGlobalFunction();

  Type *void_type = Type::getVoidTy(llvmContext);
  Type *iter_type = Type::getIntNTy(llvmContext, iter_size);
  Type *iterPtrType = PointerType::getUnqual(iter_type);
  Type *i32_type = Type::getInt32Ty(llvmContext);
  Type *bool_type = Type::getIntNTy(llvmContext, sizeof(bool) * 8);
  PointerType *charPtrType = Type::getInt8PtrTy(llvmContext);

  auto initDictScan = FunctionType::get(
      void_type, {charPtrType, iterPtrType, iterPtrType}, false);
  auto finitDictScan = Function::Create(initDictScan, Function::ExternalLinkage,
                                        "initDictScan", context->getModule());

  auto hasNextDictScan =
      FunctionType::get(bool_type, {iterPtrType, iterPtrType}, false);
  auto fhasNextDictScan =
      Function::Create(hasNextDictScan, Function::ExternalLinkage,
                       "hasNextDictScan", context->getModule());

  auto getDictScan = FunctionType::get(i32_type, {iterPtrType}, false);
  auto fgetDictScan = Function::Create(getDictScan, Function::ExternalLinkage,
                                       "getDictScan", context->getModule());

  auto nextDictScan = FunctionType::get(void_type, {iterPtrType}, false);
  auto fnextDictScan = Function::Create(nextDictScan, Function::ExternalLinkage,
                                        "nextDictScan", context->getModule());

  auto t = context->CastPtrToLlvmPtr(charPtrType, this);

  Plugin *pg = Catalog::getInstance().getPlugin(regAs.getRelationName());
  Type *oid_type = pg->getOIDType()->getLLVMType(llvmContext);

  AllocaInst *curr = context->CreateEntryBlockAlloca(F, "curr", iter_type);
  AllocaInst *end = context->CreateEntryBlockAlloca(F, "end", iter_type);
  AllocaInst *oid = context->CreateEntryBlockAlloca(F, activeLoop, oid_type);

  RecordAttribute oidRA{regAs.getRelationName(), activeLoop, pg->getOIDType()};

  Builder->CreateCall(finitDictScan, {t, curr, end});

  Builder->CreateStore(ConstantInt::get(oid_type, 0), oid);

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "Cond", F);
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "Loop", F);
  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "After", F);
  BasicBlock *EndBB = BasicBlock::Create(llvmContext, "End", F);
  context->setEndingBlock(EndBB);

  Builder->SetInsertPoint(CondBB);

  auto hasNext = Builder->CreateCall(fhasNextDictScan, {curr, end});

  auto cond = Builder->CreateICmpNE(hasNext, ConstantInt::get(bool_type, 0));

  // if (cond.value) {
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  Builder->SetInsertPoint(LoopBB);

  auto v = Builder->CreateCall(fgetDictScan, curr);
  AllocaInst *enc = context->CreateEntryBlockAlloca(F, "enc", v->getType());
  Builder->CreateStore(v, enc);

  {
    std::map<RecordAttribute, ProteusValueMemory> vars;
    vars[oidRA] = ProteusValueMemory{oid, context->createFalse()};
    vars[regAs] = ProteusValueMemory{enc, context->createFalse()};
    OperatorState state{*this, vars};
    getParent()->consume(context, state);
  }

  Builder->CreateCall(fnextDictScan, curr);
  auto old_oid = Builder->CreateLoad(oid);
  auto new_oid = Builder->CreateAdd(old_oid, ConstantInt::get(oid_type, 1));
  Builder->CreateStore(new_oid, oid);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(AfterBB);
  Builder->CreateBr(EndBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
}

void DictScan::consume(ParallelContext *const context,
                       const OperatorState &childState) {
  throw runtime_error(
      string("unexpected call to DistScan::consume (DictScan "
             "operator has no children)"));
}
