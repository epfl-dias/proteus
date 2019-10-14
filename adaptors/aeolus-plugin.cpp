/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#include "aeolus-plugin.hpp"

#include <string>

#include "communication/comm-manager.hpp"
#include "expressions/expressions-hasher.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"

using namespace llvm;

extern "C" {
storage::ColumnStore *getRelation(std::string fnamePrefix) {
  for (auto &tb : storage::Schema::getInstance().getTables()) {
    if (fnamePrefix.compare(tb->name) == 0) {
      // assert(tb->storage_layout == storage::COLUMN_STORE);
      return (storage::ColumnStore *)tb;
    }
  }
  assert(false && "Relation not found.");
}

void **getDataPointerForFile(const char *relName, const char *attrName,
                             void *session) {
  const auto &tbl = getRelation({relName});

  for (auto &c : tbl->getColumns()) {
    if (strcmp(c->name.c_str(), attrName) == 0) {
      const auto &data_arenas = c->snapshot_get_data();
      void **arr = (void **)malloc(sizeof(void *) * data_arenas.size());
      for (uint i = 0; i < data_arenas.size(); i++) {
        arr[i] = data_arenas[i].first.data;
      }
      return arr;
    }
  }
  assert(false && "ERROR: getDataPointerForFile");
}

void freeDataPointerForFile(void **inn) { free(inn); }

void *getSession() { return nullptr; }
void releaseSession(void *session) {}

int64_t *getNumOfTuplesPerPartition(const char *relName, void *session) {
  const auto &tbl = getRelation({relName});

  const auto &c = tbl->getColumns()[0];

  const auto &data_arenas = c->snapshot_get_data();
  int64_t *arr = (int64_t *)malloc(sizeof(int64_t *) * data_arenas.size());

  for (uint i = 0; i < data_arenas.size(); i++) {
    arr[i] = data_arenas[i].second;
  }

  return arr;
}

void freeNumOfTuplesPerPartition(int64_t *inn) {
  free(inn);
  // TODO: bit-mast reset logic.
}

Plugin *createBlockCowPlugin(ParallelContext *context, std::string fnamePrefix,
                             RecordType rec,
                             std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields, "block-cow");
}

Plugin *createBlockSnapshotPlugin(ParallelContext *context,
                                  std::string fnamePrefix, RecordType rec,
                                  std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields,
                          "block-snapshot");
}

Plugin *createBlockRemotePlugin(ParallelContext *context,
                                std::string fnamePrefix, RecordType rec,
                                std::vector<RecordAttribute *> &whichFields) {
  return new AeolusPlugin(context, fnamePrefix, rec, whichFields,
                          "block-remote");
}
}

AeolusPlugin::AeolusPlugin(ParallelContext *const context, string fnamePrefix,
                           RecordType rec,
                           vector<RecordAttribute *> &whichFields,
                           string pgType)
    : BinaryBlockPlugin(context, fnamePrefix, rec, whichFields, false),
      pgType(pgType) {
  Nparts =
      getRelation(fnamePrefix)->getColumns()[0]->snapshot_get_data().size();
}

llvm::Value *createCall(std::string func,
                        std::initializer_list<llvm::Value *> args,
                        Context *context, llvm::Type *ret) {
  Function *f;
  try {
    f = context->getFunction(func);
    assert(ret == f->getReturnType());
  } catch (std::runtime_error &) {
    std::vector<llvm::Type *> v;
    v.reserve(args.size());
    for (const auto &arg : args) v.emplace_back(arg->getType());
    FunctionType *FTfunc = llvm::FunctionType::get(ret, v, false);

    f = Function::Create(FTfunc, Function::ExternalLinkage, func,
                         context->getModule());

    context->registerFunction((new std::string{func})->c_str(), f);
  }

  return context->getBuilder()->CreateCall(f, args);
}

void createCall2(std::string func, std::initializer_list<llvm::Value *> args,
                 Context *context) {
  createCall(func, args, context, Type::getVoidTy(context->getLLVMContext()));
}

llvm::Value *AeolusPlugin::getSession() const {
  return createCall("getSession", {}, context,
                    Type::getInt8PtrTy(context->getLLVMContext()));
}

void AeolusPlugin::releaseSession(llvm::Value *session_ptr) const {
  createCall2("releaseSession", {session_ptr}, context);
}

Value *AeolusPlugin::getDataPointersForFile(size_t i,
                                            llvm::Value *session_ptr) const {
  LLVMContext &llvmContext = context->getLLVMContext();

  Type *char8ptr = Type::getInt8PtrTy(llvmContext);

  Value *N_parts_ptr = createCall(
      "getDataPointerForFile",
      {context->CreateGlobalString(wantedFields[i]->getRelationName().c_str()),
       context->CreateGlobalString(wantedFields[i]->getAttrName().c_str()),
       session_ptr},
      context, PointerType::getUnqual(char8ptr));

  auto data_type = PointerType::getUnqual(
      ArrayType::get(RecordAttribute{*(wantedFields[i]), true}.getLLVMType(
                         context->getLLVMContext()),
                     Nparts));

  return context->getBuilder()->CreatePointerCast(N_parts_ptr, data_type);
  // return BinaryBlockPlugin::getDataPointersForFile(i);
}

void AeolusPlugin::freeDataPointersForFile(size_t i, Value *v) const {
  LLVMContext &llvmContext = context->getLLVMContext();
  auto data_type = PointerType::getUnqual(Type::getInt8PtrTy(llvmContext));
  auto casted = context->getBuilder()->CreatePointerCast(v, data_type);
  createCall2("freeDataPointerForFile", {casted}, context);
}

std::pair<Value *, Value *> AeolusPlugin::getPartitionSizes(
    llvm::Value *session_ptr) const {
  IRBuilder<> *Builder = context->getBuilder();

  IntegerType *sizeType = context->createSizeType();

  Value *N_parts_ptr = createCall(
      "getNumOfTuplesPerPartition",
      {context->CreateGlobalString(fnamePrefix.c_str()), session_ptr}, context,
      PointerType::getUnqual(ArrayType::get(sizeType, Nparts)));

  Value *max_pack_size = ConstantInt::get(sizeType, 0);
  for (size_t i = 0; i < Nparts; ++i) {
    auto v = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        N_parts_ptr, {context->createSizeT(0), context->createSizeT(i)}));
    v->dump();
    auto cond = Builder->CreateICmpUGT(max_pack_size, v);
    max_pack_size = Builder->CreateSelect(cond, max_pack_size, v);
  }

  return {N_parts_ptr, max_pack_size};
  // return BinaryBlockPlugin::getPartitionSizes();
}

void AeolusPlugin::freePartitionSizes(Value *v) const {
  createCall2("freeNumOfTuplesPerPartition", {v}, context);
}
