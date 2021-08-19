/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "olap/plugins/binary-block-plugin.hpp"

#include <lib/expressions/expressions-flusher.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <utility>

#include "lib/expressions/expressions-hasher.hpp"
#include "lib/operators/operators.hpp"
#include "olap/util/parallel-context.hpp"

using namespace llvm;

BinaryBlockPlugin::BinaryBlockPlugin(
    ParallelContext *context, const string &fnamePrefix, RecordType rec,
    const std::vector<RecordAttribute *> &whichFields)
    : BinaryBlockPlugin(context, fnamePrefix, std::move(rec), whichFields,
                        !whichFields.empty()) {}

std::vector<RecordAttribute *> ensureRelName(
    std::vector<RecordAttribute *> whichFields, const std::string &relName) {
  for (auto &fields : whichFields)
    fields = new RecordAttribute(relName, fields->getAttrName(),
                                 fields->getOriginalType());
  return whichFields;
}

BinaryBlockPlugin::BinaryBlockPlugin(
    ParallelContext *context, const string &fnamePrefix, RecordType rec,
    const std::vector<RecordAttribute *> &whichFields, bool load)
    : fnamePrefix(fnamePrefix),
      rec(std::move(rec)),
      wantedFields(ensureRelName(whichFields, fnamePrefix)),
      Nparts(0) {
  if (load) {
    loadData(context, ALLSOCKETS);
    finalize_data(context);
  }
}

void BinaryBlockPlugin::loadData(ParallelContext *context, data_loc loc) {
  LLVMContext &llvmContext = context->getLLVMContext();
  if (wantedFields.empty()) {
    string error_msg{"[BinaryBlockPlugin: ] Invalid number of fields"};
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  // std::vector<Type *> parts_array;
  for (const auto &in : wantedFields) {
    string fileName = fnamePrefix + "." + in->getAttrName();

    const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
    size_t type_size = context->getSizeOf(llvm_type);
    fieldSizes.emplace_back(type_size);

    wantedFieldsFiles.emplace_back(
        StorageManager::getInstance().request(fileName, type_size, loc));
    // Show the intent to the storage manager
    wantedFieldsFiles.back().registerIntent();

    // wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
    // FIXME: consider if address space should be global memory rather than
    // generic
    // Type * t = PointerType::get(((const PrimitiveType *)
    // tin)->getLLVMType(llvmContext), /* address space */ 0);

    // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

    if (in->getOriginalType()->getTypeID() == DSTRING) {
      // fetch the dictionary
      void *dict = StorageManager::getInstance().getDictionaryOf(fileName);
      ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
    }
  }

  finalize_data(context);
}

void BinaryBlockPlugin::finalize_data(ParallelContext *context) {
  Nparts = wantedFieldsFiles[0].getSegmentCount();
}

void BinaryBlockPlugin::init() {}

void BinaryBlockPlugin::generate(const ::Operator &producer,
                                 ParallelContext *context) {
  return scan(producer, context);
}

ProteusValueMemory BinaryBlockPlugin::readProteusValue(
    ProteusValueMemory val, const ExpressionType *type,
    ParallelContext *context) {
  try {
    if (val.mem->getType()->getPointerElementType()->isPointerTy() &&
        val.mem->getType()
                ->getPointerElementType()
                ->getPointerElementType()
                ->getTypeID() ==
            type->getLLVMType(context->getLLVMContext())->getTypeID()) {
      auto v = context->getBuilder()->CreateLoad(val.mem);
      auto ld = context->getBuilder()->CreateLoad(v);

      {
        llvm::Metadata *Args2[] = {nullptr};
        MDNode *n2 = MDNode::get(context->getLLVMContext(), Args2);
        n2->replaceOperandWith(0, n2);

        llvm::Metadata *Args[] = {nullptr, n2};
        MDNode *n = MDNode::get(context->getLLVMContext(), Args);
        n->replaceOperandWith(0, n);
        ld->setMetadata(LLVMContext::MD_alias_scope, n);
      }

      {  // Loaded value will be the same in all the places it will be loaded
        //! invariant.load !{i32 1}
        llvm::Metadata *Args[] = {
            llvm::ValueAsMetadata::get(context->createInt32(1))};
        MDNode *n = MDNode::get(context->getLLVMContext(), Args);
        ld->setMetadata(LLVMContext::MD_invariant_load, n);
      }

      //    {  // Loaded value will be the same in all the places it will be
      //    loaded
      //      //! invariant.load !{i32 1}
      //      llvm::Metadata *Args[] = {
      //          llvm::ValueAsMetadata::get(context->createInt32(1))};
      //      MDNode *n = MDNode::get(context->getLLVMContext(), Args);
      //      ld->setMetadata(LLVMContext::MD_nontemporal, n);
      //    }

      val = context->toMem(ld, val.isNull);
    }
  } catch (...) {
  }
  return val;
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
ProteusValueMemory BinaryBlockPlugin::readPath(string activeRelation,
                                               Bindings bindings,
                                               const char *pathVar,
                                               RecordAttribute attr,
                                               ParallelContext *context) {
  // XXX Make sure that using fnamePrefix in this search does not cause issues
  RecordAttribute tmpKey{fnamePrefix, pathVar, this->getOIDType()};
  return readProteusValue((*(bindings.state))[tmpKey], attr.getOriginalType(),
                          context);
}

/* FIXME Differentiate between operations that need the code and the ones
 * needing the materialized string */
ProteusValueMemory BinaryBlockPlugin::readValue(ProteusValueMemory mem_value,
                                                const ExpressionType *type,
                                                ParallelContext *context) {
  return mem_value;
}

ProteusValue BinaryBlockPlugin::readCachedValue(CacheInfo info,
                                                const OperatorState &currState,
                                                ParallelContext *context) {
  IRBuilder<> *const Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();

  /* Need OID to retrieve corresponding value from bin. cache */
  RecordAttribute tupleIdentifier{fnamePrefix, activeLoop, getOIDType()};
  ProteusValueMemory mem_oidWrapper = currState[tupleIdentifier];
  /* OID is a plain integer - starts from 1!!! */
  Value *val_oid = Builder->CreateLoad(mem_oidWrapper.mem);
  val_oid = Builder->CreateSub(val_oid, context->createInt64(1));

  /* Need to find appropriate position in cache now -- should be OK(?) */

  StructType *cacheType = context->ReproduceCustomStruct(info.objectTypes);
  // Value *typeSize = ConstantExpr::getSizeOf(cacheType);
  char *rawPtr = *(info.payloadPtr);
  int posInStruct = info.structFieldNo;

  /* Cast to appr. type */
  PointerType *ptr_cacheType = PointerType::get(cacheType, 0);
  Value *val_cachePtr = context->CastPtrToLlvmPtr(ptr_cacheType, rawPtr);

  Value *val_cacheShiftedPtr = context->getArrayElemMem(val_cachePtr, val_oid);
  //  val_cacheShiftedPtr->getType()->dump();
  //  cout << "Pos in struct? " << posInStruct << endl;
  //  Value *val_cachedField = context->getStructElem(val_cacheShiftedPtr,
  //          posInStruct - 1);

  // XXX
  // -1 because bin files has no OID (?)
  Value *val_cachedField =
      context->getStructElem(val_cacheShiftedPtr, posInStruct /* - 1*/);
  Type *fieldType = val_cachedField->getType();

  /* This Alloca should not appear in optimized code */
  AllocaInst *mem_cachedField =
      context->CreateEntryBlockAlloca(F, "tmpCachedField", fieldType);
  Builder->CreateStore(val_cachedField, mem_cachedField);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateLoad(mem_cachedField);
  valWrapper.isNull = context->createFalse();
#ifdef DEBUG
  {
    vector<Value *> ArgsV;

    Function *debugSth = context->getFunction("printi64");
    ArgsV.push_back(val_oid);
    Builder->CreateCall(debugSth, ArgsV);
  }
#endif
  return valWrapper;
}

ProteusValue BinaryBlockPlugin::hashValue(ProteusValueMemory mem_value,
                                          const ExpressionType *type,
                                          Context *context) {
  IRBuilder<> *Builder = context->getBuilder();
  assert(dynamic_cast<ParallelContext *>(context));
  auto mem = readProteusValue(mem_value, type,
                              dynamic_cast<ParallelContext *>(context));
  ProteusValue v{Builder->CreateLoad(mem.mem), mem.isNull};
  return hashPrimitive(v, type->getTypeID(), context);
}

ProteusValue BinaryBlockPlugin::hashValueEager(ProteusValue valWrapper,
                                               const ExpressionType *type,
                                               Context *context) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();
  Value *tmp = valWrapper.value;
  AllocaInst *mem_tmp =
      context->CreateEntryBlockAlloca(F, "mem_cachedToHash", tmp->getType());
  Builder->CreateStore(tmp, mem_tmp);
  ProteusValueMemory mem_tmpWrapper{mem_tmp, valWrapper.isNull};
  return hashValue(mem_tmpWrapper, type, context);
}

void BinaryBlockPlugin::finish() {
  LOG(INFO) << "[BinaryBlockPlugin] Finish";
  int cnt = 0;
  for (const auto &attr : wantedFields) {
    if (attr->getOriginalType()->getTypeID() == STRING) {
      int dictionaryFd = dictionaries[cnt];
      close(dictionaryFd);
      munmap(dictionariesBuf[cnt], dictionaryFilesizes[cnt]);
    }
    cnt++;
  }
}

llvm::Value *BinaryBlockPlugin::getValueSize(ProteusValueMemory mem_value,
                                             const ExpressionType *type,
                                             ParallelContext *context) {
  switch (type->getTypeID()) {
    case BOOL:
    case INT:
    case FLOAT: {
      Type *explicitType = (mem_value.mem)->getAllocatedType();
      return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
    }
    case STRING: {
      string error_msg =
          string("[Binary Col Plugin]: Strings not supported yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    case BAG:
    case LIST:
    case SET: {
      string error_msg =
          string("[Binary Col Plugin]: Cannot contain collections");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    case RECORD: {
      string error_msg = string("[Binary Col Plugin]: Cannot contain records");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    default: {
      string error_msg = string("[Binary Col Plugin]: Unknown datatype");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
}

void BinaryBlockPlugin::skipLLVM(ParallelContext *context,
                                 RecordAttribute attName, Value *offset) {
  // Prepare
  IRBuilder<> *Builder = context->getBuilder();

  // Fetch values from symbol table
  string currPosVar = std::string{posVar} + "." + attName.getAttrName();
  auto mem_pos = NamedValuesBinaryCol.at(currPosVar);

  // Increment and store back
  Value *val_curr_pos = Builder->CreateLoad(mem_pos);
  Value *val_new_pos = Builder->CreateAdd(val_curr_pos, offset);
  Builder->CreateStore(val_new_pos, mem_pos);
}

void BinaryBlockPlugin::nextEntry(ParallelContext *context,
                                  llvm::Value *blockSize) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();

  // Necessary because it's the itemCtr that affects the scan loop
  auto mem_itemCtr = NamedValuesBinaryCol.at(itemCtrVar);

  // Necessary because it's the itemCtr that affects the scan loop
  auto part_i_ptr = NamedValuesBinaryCol.at("part_i_ptr");

  // Necessary because it's the itemCtr that affects the scan loop
  auto block_i_ptr = NamedValuesBinaryCol.at("block_i_ptr");

  // Increment and store back
  BasicBlock *wrapBB = BasicBlock::Create(llvmContext, "incWrap", F);
  BasicBlock *stepBB = BasicBlock::Create(llvmContext, "incStep", F);
  BasicBlock *afterBB = BasicBlock::Create(llvmContext, "incAfter", F);

  Value *part_i = Builder->CreateLoad(part_i_ptr, "part_i");

  auto *size_type = (IntegerType *)part_i->getType();

  Value *part_N = ConstantInt::get(size_type, Nparts - 1);

  Value *cond = Builder->CreateICmpULT(part_i, part_N);
  Builder->CreateCondBr(cond, stepBB, wrapBB);

  Builder->SetInsertPoint(stepBB);
  Builder->CreateStore(
      Builder->CreateAdd(part_i, ConstantInt::get(size_type, 1)), part_i_ptr);

  Builder->CreateBr(afterBB);

  Builder->SetInsertPoint(wrapBB);

  Builder->CreateStore(ConstantInt::get(size_type, 0), part_i_ptr);

  Value *block_i = Builder->CreateLoad(block_i_ptr, "block_i");
  Builder->CreateStore(Builder->CreateAdd(block_i, blockSize), block_i_ptr);

  Builder->CreateBr(afterBB);

  Builder->SetInsertPoint(afterBB);

  // itemCtr = block_i_ptr * Nparts + part_i_ptr * blockSize
  Value *itemCtr = Builder->CreateAdd(
      Builder->CreateMul(Builder->CreateLoad(block_i_ptr),
                         ConstantInt::get(size_type, Nparts)),
      Builder->CreateMul(Builder->CreateLoad(part_i_ptr), blockSize));

  Builder->CreateStore(itemCtr, mem_itemCtr);
}

/* Operates over int*! */
void BinaryBlockPlugin::readAsIntLLVM(
    ParallelContext *context, RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(context, std::move(attName), variables);
}

/* Operates over char*! */
void BinaryBlockPlugin::readAsLLVM(
    ParallelContext *context, RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  // Prepare
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  string posVarStr = string(posVar);
  string currPosVar = posVarStr + "." + attName.getAttrName();
  string bufVarStr = string(bufVar);
  string currBufVar = bufVarStr + "." + attName.getAttrName();

  // Fetch values from symbol table
  auto mem_pos = NamedValuesBinaryCol.at(currPosVar);
  Value *val_pos = Builder->CreateLoad(mem_pos);

  auto buf = NamedValuesBinaryCol.at(currBufVar);
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_pos);
  Value *parsedInt = Builder->CreateLoad(bufShiftedPtr);

  AllocaInst *mem_currResult = context->CreateEntryBlockAlloca(
      TheFunction, "currResult", parsedInt->getType());
  Builder->CreateStore(parsedInt, mem_currResult);

  ProteusValueMemory mem_valWrapper{mem_currResult, context->createFalse()};
  variables[attName] = mem_valWrapper;
}

void BinaryBlockPlugin::readAsBooleanLLVM(
    ParallelContext *context, RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(context, std::move(attName), variables);
}

void BinaryBlockPlugin::readAsFloatLLVM(
    ParallelContext *context, RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(context, std::move(attName), variables);
}

const void **getDataForField(size_t i, BinaryBlockPlugin *pg) {
  return pg->getDataForField(i);
}

const void **BinaryBlockPlugin::getDataForField(size_t i) {
  auto fieldPtr =
      (const void **)MemoryManager::mallocPinned(Nparts * sizeof(void *));

  if (!wantedFieldsFiles[i].isPinned()) wantedFieldsFiles[i].pin();

  size_t j = 0;
  for (const auto &t : wantedFieldsFiles[i].getSegments())
    fieldPtr[j++] = t.data;
  if (j != Nparts) {
    string error_msg{"Columns do not have the same number of partitions"};
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  return fieldPtr;
}

void freeDataForField(size_t j, const void **d, BinaryBlockPlugin *pg) {
  return pg->freeDataForField(j, d);
}

void BinaryBlockPlugin::freeDataForField(size_t j, const void **d) {
  assert(wantedFieldsFiles[j].isPinned());
  wantedFieldsFiles[j].unpin();
  MemoryManager::freePinned(d);
}

int64_t *getTuplesPerPartition(BinaryBlockPlugin *pg) {
  return pg->getTuplesPerPartition();
}

void freeTuplesPerPartition(int64_t *p, BinaryBlockPlugin *pg) {
  pg->freeTuplesPerPartition(p);
}

int64_t *BinaryBlockPlugin::getTuplesPerPartition() {
  assert(!wantedFields.empty() && "Unimplemented zero column scan");
  assert(!wantedFieldsFiles.empty() && "Unimplemented zero column scan");
  auto N_parts_init =
      (int64_t *)MemoryManager::mallocPinned(Nparts * sizeof(void *));

#ifndef NDEBUG
  std::vector<size_t> N_parts_init_sizes;
#endif
  if (!wantedFieldsFiles[0].isPinned()) wantedFieldsFiles[0].pin();

  size_t max_pack_size = 0;
  size_t k = 0;

  assert(wantedFieldsFiles[0].getSegmentCount() == Nparts);
  for (const auto &t : wantedFieldsFiles[0].getSegments()) {
    //    assert((t.size % context->getSizeOf(
    //        wantedFields[0]->getLLVMType(llvmContext))) == 0);
    size_t pack_N = t.size / fieldSizes[0];
#ifndef NDEBUG
    N_parts_init_sizes.push_back(pack_N);
#endif
    N_parts_init[k++] = pack_N;
    max_pack_size = std::max(pack_N, max_pack_size);
  }

  // Following check is too expensive for codegen-time
#ifndef NDEBUG
  for (size_t j = 0; j < wantedFields.size(); ++j) {
    if (!wantedFieldsFiles[j].isPinned()) wantedFieldsFiles[j].pin();
    assert(wantedFieldsFiles[j].getSegmentCount() == Nparts);
    const auto &files = wantedFieldsFiles[j].getSegments();
    const size_t size = fieldSizes[j];
    for (size_t i = 0; i < files.size(); ++i) {
      const auto &t = files[i];
      assert((t.size % size) == 0);
      size_t pack_N = t.size / size;
      assert(pack_N == N_parts_init_sizes[i]);
    }
  }
#endif
  return N_parts_init;
}

void BinaryBlockPlugin::freeTuplesPerPartition(int64_t *p) {
  MemoryManager::freePinned(p);
}

llvm::Value *BinaryBlockPlugin::getDataPointersForFile(ParallelContext *context,
                                                       size_t i,
                                                       llvm::Value *) const {
  LLVMContext &llvmContext = context->getLLVMContext();

  Function *F = context->getGlobalFunction();
  //  IRBuilder<> *Builder = context->getBuilder();

  //  std::vector<Constant *> part_ptrs;

  Type *col_type = wantedFields[i]->getLLVMType(llvmContext);
  //  Type *col_ptr_type = PointerType::getUnqual(col_type);

  //  for (const auto &t : wantedFieldsFiles[i].get()) {
  //    Constant *constaddr = context->createInt64((int64_t)(t.data));
  //    Constant *constptr = ConstantExpr::getIntToPtr(constaddr, col_ptr_type);
  //
  //    part_ptrs.emplace_back(constptr);
  //  }
  //  if (part_ptrs.size() != Nparts) {
  //    string error_msg{
  //        "[BinaryBlockPlugin: ] Columns do not have the same "
  //        "number of partitions"};
  //    LOG(ERROR) << error_msg;
  //    throw runtime_error(error_msg);
  //  }

  Type *char8ptr = Type::getInt8PtrTy(llvmContext);

  Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this), char8ptr);

  Value *N_parts_ptr = context->gen_call(&::getDataForField,
                                         {context->createSizeT(i), this_ptr});

  //  // file_parts_init.emplace_back(
  //  Value *const_parts = ConstantArray::get(
  //      ArrayType::get(RecordAttribute{*(wantedFields[i]), true}.getLLVMType(
  //                         context->getLLVMContext()),
  //                     Nparts),
  //      part_ptrs);

  auto data_type = PointerType::getUnqual(
      ArrayType::get(RecordAttribute{*(wantedFields[i]), true}.getLLVMType(
                         context->getLLVMContext()),
                     Nparts));

  auto ptr = context->getBuilder()->CreatePointerCast(N_parts_ptr, data_type);

  //  auto ptr = context->CreateEntryBlockAlloca(
  //      F, wantedFields[i]->getAttrName() + "_parts_ptr", data_type);
  //
  ////  N_parts_ptr
  //  context->getBuilder()->CreateStore(N_parts_ptrD, ptr);

  return ptr;
}

void BinaryBlockPlugin::freeDataPointersForFile(ParallelContext *context,
                                                size_t i, Value *v) const {
  LLVMContext &llvmContext = context->getLLVMContext();
  auto data_type = PointerType::getUnqual(Type::getInt8PtrTy(llvmContext));
  auto casted = context->getBuilder()->CreatePointerCast(v, data_type);
  Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this), Type::getInt8PtrTy(llvmContext));
  context->gen_call(&::freeDataForField,
                    {context->createSizeT(i), casted, this_ptr});
}

std::pair<llvm::Value *, llvm::Value *> BinaryBlockPlugin::getPartitionSizes(
    ParallelContext *context, llvm::Value *) const {
  IRBuilder<> *Builder = context->getBuilder();

  IntegerType *sizeType = context->createSizeType();

  Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      Type::getInt8PtrTy(context->getLLVMContext()));

  Value *N_parts_ptr = Builder->CreatePointerCast(
      context->gen_call(&::getTuplesPerPartition, {this_ptr}),
      PointerType::getUnqual(ArrayType::get(sizeType, Nparts)));

  Value *max_pack_size = ConstantInt::get(sizeType, 0);
  for (size_t i = 0; i < Nparts; ++i) {
    auto v = Builder->CreateLoad(Builder->CreateInBoundsGEP(
        N_parts_ptr, {context->createSizeT(0), context->createSizeT(i)}));
    auto cond = Builder->CreateICmpUGT(max_pack_size, v);
    max_pack_size = Builder->CreateSelect(cond, max_pack_size, v);
  }

  return {N_parts_ptr, max_pack_size};
}

void BinaryBlockPlugin::freePartitionSizes(ParallelContext *context,
                                           Value *v) const {
  Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      Type::getInt8PtrTy(context->getLLVMContext()));
  context->gen_call(&::freeTuplesPerPartition, {v, this_ptr});
}

void BinaryBlockPlugin::scan(const ::Operator &producer,
                             ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  context->setGlobalFunction(true);

  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  // Prepare
  IntegerType *size_type = context->createSizeType();

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> variableBindings;

  // Get the ENTRY BLOCK
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  llvm::Value *session = getSession(context);

  std::vector<Value *> parts_ptrs;

  // std::vector<Constant *> file_parts_init;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    parts_ptrs.emplace_back(getDataPointersForFile(context, i, session));
  }

  // Constant * file_parts = ConstantStruct::get(parts_arrays_type,
  // file_parts_init); Builder->CreateStore(file_parts, file_parts_ptr);

  auto partsizes = getPartitionSizes(context, session);
  Value *N_parts_ptr = partsizes.first;
  Value *maxPackCnt = partsizes.second;
  maxPackCnt->setName("maxPackCnt");

  size_t max_field_size = 0;
  for (const auto &f : wantedFields) {
    size_t field_size = context->getSizeOf(f->getLLVMType(llvmContext));
    max_field_size = std::max(field_size, max_field_size);
  }

  ConstantInt *zero_idx = ConstantInt::get(size_type, 0);

  AllocaInst *part_i_ptr =
      context->CreateEntryBlockAlloca(F, "part_i_ptr", size_type);
  Builder->CreateStore(zero_idx, part_i_ptr);
  NamedValuesBinaryCol["part_i_ptr"] = part_i_ptr;

  AllocaInst *block_i_ptr =
      context->CreateEntryBlockAlloca(F, "block_i_ptr", size_type);
  Builder->CreateStore(zero_idx, block_i_ptr);
  NamedValuesBinaryCol["block_i_ptr"] = block_i_ptr;

  AllocaInst *mem_itemCtr =
      context->CreateEntryBlockAlloca(F, itemCtrVar, size_type);
  Builder->CreateStore(zero_idx, mem_itemCtr);
  NamedValuesBinaryCol[itemCtrVar] = mem_itemCtr;

  auto blockSize =
      ConstantInt::get(size_type, BlockManager::block_size / max_field_size);

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

  // // Start insertion in CondBB.
  // Builder->SetInsertPoint(CondBB);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);
  BasicBlock *MainBB = BasicBlock::Create(llvmContext, "scanMain", F);

  // Make the new basic block for the increment, inserting after current
  // block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

  // Create the "AFTER LOOP" block and insert it.
  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
  context->setEndingBlock(AfterBB);

  Builder->SetInsertPoint(CondBB);

  // /**
  //  * Equivalent:
  //  * while(block_i < max(partsize))
  //  */

  Value *block_i = Builder->CreateLoad(block_i_ptr, "block_i");

  Value *cond = Builder->CreateICmpULT(block_i, maxPackCnt);

  // Insert the conditional branch into the end of CondBB.
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  Value *part_i_loc = Builder->CreateLoad(part_i_ptr, "part_i");
  Value *block_i_loc = Builder->CreateLoad(block_i_ptr, "block_i");

  Value *tupleCnt = Builder->CreateLoad(Builder->CreateInBoundsGEP(
      N_parts_ptr, std::vector<Value *>{context->createInt64(0), part_i_loc}));

  Value *part_unfinished = Builder->CreateICmpULT(block_i_loc, tupleCnt);

  Builder->CreateCondBr(part_unfinished, MainBB, IncBB);

  Builder->SetInsertPoint(MainBB);

  // Get the 'oid' of each record and pass it along.
  // More general/lazy plugins will only perform this action,
  // instead of eagerly 'converting' fields
  // FIXME This action corresponds to materializing the oid. Do we want this?
  RecordAttribute tupleIdentifier{
      fnamePrefix, activeLoop,
      this->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_posWrapper{mem_itemCtr, context->createFalse()};
  variableBindings[tupleIdentifier] = mem_posWrapper;

  // Actual Work (Loop through attributes etc.)
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute attr(*(wantedFields[i]));
    RecordAttribute block_attr(attr, true);

    Type *ptr_t =
        PointerType::get(attr.getLLVMType(context->getLLVMContext()), 0);

    Value *base_ptr = Builder->CreateInBoundsGEP(
        parts_ptrs[i],
        std::vector<Value *>{context->createInt64(0), part_i_loc});
    Value *val_bufPtr =
        Builder->CreateInBoundsGEP(Builder->CreateLoad(base_ptr), block_i_loc);
    val_bufPtr->setName(attr.getAttrName() + "_ptr");

    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attr.getAttrName() + "_ptr";

    AllocaInst *mem_currResult =
        context->CreateEntryBlockAlloca(F, currBufVar, ptr_t);
    Builder->CreateStore(val_bufPtr, mem_currResult);

    ProteusValueMemory mem_valWrapper{mem_currResult, context->createFalse()};
    variableBindings[block_attr] = mem_valWrapper;
  }

  AllocaInst *blockN_ptr =
      context->CreateEntryBlockAlloca(F, "blockN", tupleCnt->getType());

  Value *remaining = Builder->CreateSub(tupleCnt, block_i_loc);
  Value *blockN = Builder->CreateSelect(
      Builder->CreateICmpULT(blockSize, remaining), blockSize, remaining);
  Builder->CreateStore(blockN, blockN_ptr);

  RecordAttribute tupCnt =
      RecordAttribute(fnamePrefix, "activeCnt",
                      this->getOIDType());  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper;
  mem_cntWrapper.mem = blockN_ptr;
  mem_cntWrapper.isNull = context->createFalse();
  variableBindings[tupCnt] = mem_cntWrapper;

  // // Start insertion in IncBB.
  Builder->SetInsertPoint(IncBB);
  nextEntry(context, blockSize);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(MainBB);

  OperatorState state{producer, variableBindings};
  producer.getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);

  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);
  // Builder->CreateRetVoid();

  //  Finish up with end (the AfterLoop)
  //  Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());

  freePartitionSizes(context, N_parts_ptr);
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    freeDataPointersForFile(context, i, parts_ptrs[i]);
  }

  releaseSession(context, session);
  // Builder->SetInsertPoint(AfterBB);
}

RecordType BinaryBlockPlugin::getRowType() const {
  std::vector<RecordAttribute *> rec;
  for (const auto &attr : this->wantedFields) {
    auto ptr = attr;
    // This plugin outputs blocks, so let's convert any attribute to a
    // BlockType
    if (!dynamic_cast<const BlockType *>(ptr->getOriginalType())) {
      ptr = new RecordAttribute(*attr, true);
    }
    assert(!dynamic_cast<const BlockType *>(attr->getOriginalType()));
    rec.emplace_back(ptr);
  }
  return rec;
}

void BinaryBlockPlugin::flushValueEager(Context *context,
                                        ProteusValue mem_value,
                                        const ExpressionType *type,
                                        std::string fileName) {
  return flushValue(context, context->toMem(mem_value), type, fileName);
}

void BinaryBlockPlugin::flushValue(Context *context,
                                   ProteusValueMemory mem_value,
                                   const ExpressionType *type,
                                   std::string fileName) {
  return flushValueInternal(context, mem_value, type,
                            fileName + "/" + fileName);
}

auto *getSerializer(const char *file) {
  return &Catalog::getInstance().getSerializer(file);
}

void BinaryBlockPlugin::flushValueInternal(Context *context,
                                           ProteusValueMemory mem_value,
                                           const ExpressionType *type,
                                           std::string fileName) {
  IRBuilder<> *Builder = context->getBuilder();
  Value *val_attr = Builder->CreateLoad(mem_value.mem);
  switch (type->getTypeID()) {
    case RECORD: {
      auto attrs = ((const RecordType *)type)->getArgs();

      size_t i = 0;
      for (const auto &attr : attrs) {
        // value
        auto partialFlush = context->toMem(
            Builder->CreateExtractValue(val_attr, i), context->createFalse());

        auto concat = fileName + "." + attr->getAttrName();

        flushValueInternal(context, partialFlush, attr->getOriginalType(),
                           concat);

        ++i;
      }

      return;
    }
      //    case STRING:
    case SET:
    case COMPOSITE:
    case BLOCK:
    case LIST:
    case BAG: {
      LOG(ERROR) << "Unsupported datatype: " << *type;
      throw runtime_error("Unsupported datatype");
    }
    default: {
      auto BB = Builder->GetInsertBlock();

      // Find serializer, in the entry block
      Builder->SetInsertPoint(context->getCurrentEntryBlock());
      auto ser = context->gen_call(
          getSerializer,
          {context->CastPtrToLlvmPtr(
              llvm::Type::getInt8PtrTy(context->getLLVMContext()),
              (new std::string{fileName})->c_str())});
      Builder->SetInsertPoint(BB);

      // Back to normal flow, find cached serializer value
      auto flushFunc =
          dynamic_cast<ParallelContext *>(context)->getFunctionNameOverload(
              "flushBinary", val_attr->getType());

      context->gen_call(flushFunc, {val_attr, ser},
                        Type::getVoidTy(context->getLLVMContext()));
      return;
    }
  }
}

void BinaryBlockPlugin::flushBeginList(llvm::Value *) {}
void BinaryBlockPlugin::flushEndList(llvm::Value *) {}
void BinaryBlockPlugin::flushDelim(llvm::Value *, int) {}
void BinaryBlockPlugin::flushDelim(llvm::Value *, llvm::Value *, int) {}

void BinaryBlockPlugin::flushOutputInternal(Context *context,
                                            std::string fileName,
                                            const ExpressionType *type) {
  if (auto r = dynamic_cast<const RecordType *>(type)) {
    for (const auto &attr : r->getArgs()) {
      flushOutputInternal(context, fileName + "." + attr->getAttrName(),
                          attr->getOriginalType());
    }
  } else {
    // Find serializer, in the entry block
    auto f = context->CastPtrToLlvmPtr(
        llvm::Type::getInt8PtrTy(context->getLLVMContext()),
        (new std::string{fileName})->c_str());

    auto ser = context->gen_call(getSerializer, {f});

    context->gen_call(flushBinaryOutput, {f, ser});
  }
}

void BinaryBlockPlugin::flushOutput(Context *context, std::string fileName,
                                    const ExpressionType *type) {
  flushOutputInternal(context, fileName + "/" + fileName, type);
}

extern "C" Plugin *createBlockPlugin(
    ParallelContext *context, std::string fnamePrefix, RecordType rec,
    const std::vector<RecordAttribute *> &whichFields) {
  return new BinaryBlockPlugin(context, fnamePrefix, rec, whichFields);
}
