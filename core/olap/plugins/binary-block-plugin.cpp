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

#include "plugins/binary-block-plugin.hpp"

#include "expressions/expressions-hasher.hpp"
#include "memory/block-manager.hpp"
#include "operators/operators.hpp"
#include "util/parallel-context.hpp"

using namespace llvm;

BinaryBlockPlugin::BinaryBlockPlugin(ParallelContext *const context,
                                     string fnamePrefix, RecordType rec,
                                     vector<RecordAttribute *> &whichFields)
    : BinaryBlockPlugin(context, fnamePrefix, rec, whichFields, true) {}

BinaryBlockPlugin::BinaryBlockPlugin(ParallelContext *const context,
                                     string fnamePrefix, RecordType rec,
                                     vector<RecordAttribute *> &whichFields,
                                     bool load)
    : fnamePrefix(fnamePrefix),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      itemCtrVar("itemCtr") {
  if (wantedFields.size() == 0) {
    string error_msg{"[BinaryBlockPlugin: ] Invalid number of fields"};
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  LLVMContext &llvmContext = context->getLLVMContext();

  if (load) {
    // std::vector<Type *> parts_array;
    for (const auto &in : wantedFields) {
      string fileName = fnamePrefix + "." + in->getAttrName();

      const auto llvm_type = in->getOriginalType()->getLLVMType(llvmContext);
      size_t type_size = context->getSizeOf(llvm_type);

      wantedFieldsFiles.emplace_back(
          StorageManager::getOrLoadFile(fileName, type_size, ALLSOCKETS));
      // wantedFieldsFiles.emplace_back(StorageManager::getFile(fileName));
      // FIXME: consider if address space should be global memory rather than
      // generic
      // Type * t = PointerType::get(((const PrimitiveType *)
      // tin)->getLLVMType(llvmContext), /* address space */ 0);

      // wantedFieldsArg_id.push_back(context->appendParameter(t, true, true));

      if (in->getOriginalType()->getTypeID() == DSTRING) {
        // fetch the dictionary
        void *dict = StorageManager::getDictionaryOf(fileName);
        ((DStringType *)(in->getOriginalType()))->setDictionary(dict);
      }
    }

    finalize_data();
  }
}

void BinaryBlockPlugin::finalize_data() {
  LLVMContext &llvmContext = context->getLLVMContext();
  Nparts = wantedFieldsFiles[0].size();
  for (size_t i = 1; i < wantedFields.size(); ++i) {
    size_t Nparts_loc = wantedFieldsFiles[i].size();
    if (Nparts_loc != Nparts) {
      string error_msg{
          "[BinaryBlockPlugin: ] Columns do not have the same "
          "number of partitions"};
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  // for (const auto &in : wantedFields) {
  //   RecordAttribute bin(*in, true);
  //   parts_array.emplace_back(
  //       ArrayType::get(bin.getLLVMType(llvmContext), Nparts));
  // }

  // parts_arrays_type = StructType::get(llvmContext, parts_array);
}

BinaryBlockPlugin::BinaryBlockPlugin(ParallelContext *const context,
                                     string fnamePrefix, RecordType rec)
    : fnamePrefix(fnamePrefix),
      rec(rec),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      itemCtrVar("itemCtr") {
  Nparts = 0;
}

BinaryBlockPlugin::~BinaryBlockPlugin() {
  std::cout << "freeing plugin..." << std::endl;
}

void BinaryBlockPlugin::init() {}

void BinaryBlockPlugin::generate(const ::Operator &producer) {
  return scan(producer);
}

ProteusValueMemory BinaryBlockPlugin::readProteusValue(
    ProteusValueMemory val, const ExpressionType *type) {
  if (val.mem->getType()->getPointerElementType()->isPointerTy() &&
      val.mem->getType()
              ->getPointerElementType()
              ->getPointerElementType()
              ->getTypeID() ==
          type->getLLVMType(context->getLLVMContext())->getTypeID()) {
    auto v = context->getBuilder()->CreateLoad(val.mem);
    val = context->toMem(context->getBuilder()->CreateLoad(v), val.isNull);
  }
  return val;
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
ProteusValueMemory BinaryBlockPlugin::readPath(string activeRelation,
                                               Bindings bindings,
                                               const char *pathVar,
                                               RecordAttribute attr) {
  // XXX Make sure that using fnamePrefix in this search does not cause issues
  RecordAttribute tmpKey{fnamePrefix, pathVar, this->getOIDType()};
  return readProteusValue((*(bindings.state))[tmpKey], attr.getOriginalType());
}

/* FIXME Differentiate between operations that need the code and the ones
 * needing the materialized string */
ProteusValueMemory BinaryBlockPlugin::readValue(ProteusValueMemory mem_value,
                                                const ExpressionType *type) {
  return mem_value;
}

ProteusValue BinaryBlockPlugin::readCachedValue(
    CacheInfo info, const OperatorState &currState) {
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
                                          const ExpressionType *type) {
  IRBuilder<> *Builder = context->getBuilder();
  auto mem = readProteusValue(mem_value, type);
  ProteusValue v{Builder->CreateLoad(mem.mem), mem.isNull};
  return hashPrimitive(v, type->getTypeID(), context);
}

ProteusValue BinaryBlockPlugin::hashValueEager(ProteusValue valWrapper,
                                               const ExpressionType *type) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();
  Value *tmp = valWrapper.value;
  AllocaInst *mem_tmp =
      context->CreateEntryBlockAlloca(F, "mem_cachedToHash", tmp->getType());
  Builder->CreateStore(tmp, mem_tmp);
  ProteusValueMemory mem_tmpWrapper{mem_tmp, valWrapper.isNull};
  return hashValue(mem_tmpWrapper, type);
}

void BinaryBlockPlugin::finish() {
  LOG(INFO) << "[BinaryBlockPlugin] Finish";
  int cnt = 0;
  for (const auto &attr : wantedFields) {
    close(fd[cnt]);
    munmap(buf[cnt], colFilesize[cnt]);

    if (attr->getOriginalType()->getTypeID() == STRING) {
      int dictionaryFd = dictionaries[cnt];
      close(dictionaryFd);
      munmap(dictionariesBuf[cnt], dictionaryFilesizes[cnt]);
    }
    cnt++;
  }
}

Value *BinaryBlockPlugin::getValueSize(ProteusValueMemory mem_value,
                                       const ExpressionType *type) {
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

void BinaryBlockPlugin::skipLLVM(RecordAttribute attName, Value *offset) {
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

void BinaryBlockPlugin::nextEntry() {
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

  IntegerType *size_type = (IntegerType *)part_i->getType();

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
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(attName, variables);
}

/* Operates over char*! */
void BinaryBlockPlugin::readAsLLVM(
    RecordAttribute attName,
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

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_currResult;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

/* Operates over char*! */
void BinaryBlockPlugin::readAsInt64LLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(attName, variables);
}

/*
 * FIXME Needs to be aware of dictionary (?).
 * Probably readValue() is the appropriate place for this.
 * I think forwarding the dict. code (int32) is sufficient here.
 */
void BinaryBlockPlugin::readAsStringLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsIntLLVM(attName, variables);
}

void BinaryBlockPlugin::readAsBooleanLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(attName, variables);
}

void BinaryBlockPlugin::readAsFloatLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  readAsLLVM(attName, variables);
}

void BinaryBlockPlugin::prepareArray(RecordAttribute attName) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();

  string posVarStr = string(posVar);
  string currPosVar = posVarStr + "." + attName.getAttrName();
  string bufVarStr = string(bufVar);
  string currBufVar = bufVarStr + "." + attName.getAttrName();

  /* Code equivalent to skip(size_t) */
  Value *val_offset = context->createInt64(sizeof(size_t));
  auto mem_pos = NamedValuesBinaryCol.at(currPosVar);

  // Increment and store back
  Value *val_curr_pos = Builder->CreateLoad(mem_pos);
  Value *val_new_pos = Builder->CreateAdd(val_curr_pos, val_offset);
  /* Not storing this 'offset' - we want the cast buffer to
   * conceptually start from 0 */
  //  Builder->CreateStore(val_new_pos,mem_pos);

  /* Get relevant char* rawBuf */
  auto buf = NamedValuesBinaryCol.at(currBufVar);
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, val_new_pos);

  auto type = PointerType::getUnqual(attName.getLLVMType(llvmContext));
  auto mem_bufPtr = context->CreateEntryBlockAlloca(F, "mem_bufPtr", type);
  Value *val_bufPtr = Builder->CreateBitCast(bufShiftedPtr, type);
  Builder->CreateStore(val_bufPtr, mem_bufPtr);
  NamedValuesBinaryCol[currBufVar] = mem_bufPtr;
}

Value *BinaryBlockPlugin::getDataPointersForFile(size_t i,
                                                 llvm::Value *) const {
  LLVMContext &llvmContext = context->getLLVMContext();

  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  std::vector<Constant *> part_ptrs;

  Type *col_type = wantedFields[i]->getLLVMType(llvmContext);
  Type *col_ptr_type = PointerType::getUnqual(col_type);

  for (const auto &t : wantedFieldsFiles[i]) {
    Constant *constaddr = context->createInt64((int64_t)(t.data));
    Constant *constptr = ConstantExpr::getIntToPtr(constaddr, col_ptr_type);

    part_ptrs.emplace_back(constptr);
  }

  // file_parts_init.emplace_back(
  Value *const_parts = ConstantArray::get(
      ArrayType::get(RecordAttribute{*(wantedFields[i]), true}.getLLVMType(
                         context->getLLVMContext()),
                     Nparts),
      part_ptrs);

  auto ptr = context->CreateEntryBlockAlloca(
      F, wantedFields[i]->getAttrName() + "_parts_ptr", const_parts->getType());

  Builder->CreateStore(const_parts, ptr);

  return ptr;
}

void BinaryBlockPlugin::freeDataPointersForFile(size_t i, Value *v) const {}

std::pair<Value *, Value *> BinaryBlockPlugin::getPartitionSizes(
    llvm::Value *session) const {
  LLVMContext &llvmContext = context->getLLVMContext();

  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  IntegerType *sizeType = context->createSizeType();

  std::vector<Constant *> N_parts_init;
#ifndef NDEBUG
  std::vector<size_t> N_parts_init_sizes;
#endif

  size_t max_pack_size = 0;
  for (const auto &t : wantedFieldsFiles[0]) {
    assert((t.size % context->getSizeOf(
                         wantedFields[0]->getLLVMType(llvmContext))) == 0);
    size_t pack_N =
        t.size / context->getSizeOf(wantedFields[0]->getLLVMType(llvmContext));
#ifndef NDEBUG
    N_parts_init_sizes.push_back(pack_N);
#endif
    N_parts_init.push_back(context->createInt64(pack_N));
    max_pack_size = std::max(pack_N, max_pack_size);
  }

#ifndef NDEBUG
  for (size_t j = 0; j < wantedFields.size(); ++j) {
    const auto &files = wantedFieldsFiles[j];
    const size_t size =
        context->getSizeOf(wantedFields[j]->getLLVMType(llvmContext));
    for (size_t i = 0; i < files.size(); ++i) {
      const auto &t = files[i];
      assert((t.size % size) == 0);
      size_t pack_N = t.size / size;
      assert(pack_N == N_parts_init_sizes[i]);
    }
  }
#endif

  ArrayType *arr_type = ArrayType::get(sizeType, Nparts);
  Value *N_parts_ptr =
      context->CreateEntryBlockAlloca(F, "N_parts_ptr", arr_type);
  Builder->CreateStore(ConstantArray::get(arr_type, N_parts_init), N_parts_ptr);

  return {N_parts_ptr, ConstantInt::get(sizeType, max_pack_size)};
}

void BinaryBlockPlugin::freePartitionSizes(Value *) const {}

void BinaryBlockPlugin::scan(const ::Operator &producer) {
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

  llvm::Value *session = getSession();

  std::vector<Value *> parts_ptrs;

  // std::vector<Constant *> file_parts_init;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    parts_ptrs.emplace_back(getDataPointersForFile(i, session));
  }

  // Constant * file_parts = ConstantStruct::get(parts_arrays_type,
  // file_parts_init); Builder->CreateStore(file_parts, file_parts_ptr);

  auto partsizes = getPartitionSizes(session);
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

  blockSize =
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
  RecordAttribute tupleIdentifier =
      RecordAttribute(fnamePrefix, activeLoop,
                      this->getOIDType());  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_posWrapper;
  mem_posWrapper.mem = mem_itemCtr;
  mem_posWrapper.isNull = context->createFalse();
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

    ProteusValueMemory mem_valWrapper;
    mem_valWrapper.mem = mem_currResult;
    mem_valWrapper.isNull = context->createFalse();
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
  nextEntry();

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

  freePartitionSizes(N_parts_ptr);
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    freeDataPointersForFile(i, parts_ptrs[i]);
  }

  releaseSession(session);
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
