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

#include "plugins/csv-plugin-pm.hpp"
#include "expressions/expressions-hasher.hpp"

using namespace llvm;

namespace pm {

CSVPlugin::CSVPlugin(Context *const context, string &fname, RecordType &rec,
                     vector<RecordAttribute *> whichFields, int lineHint,
                     int policy, bool stringBrackets)
    : fname(fname),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      fsizeVar("fileSize"),
      lines(lineHint),
      policy(policy),
      stringBrackets(stringBrackets) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  fd = -1;
  buf = NULL;
  std::sort(wantedFields.begin(), wantedFields.end());

  LOG(INFO) << "[CSVPlugin: ] " << fname;

  struct stat statbuf;
  const char *name_c = fname.c_str();
  stat(name_c, &statbuf);
  fsize = statbuf.st_size;

  fd = open(name_c, O_RDONLY);
  if (fd == -1 && whichFields.size() > 0) {
    throw runtime_error(string("csv.open"));
  }

  this->delimInner = ';';
  this->delimEnd = '\n';

  /* PM */
  CachingService &cache = CachingService::getInstance();

  char *pmCast = cache.getPM(fname);
  if (pmCast == NULL) {
    // cout << "NEW (CSV) PM" << endl;
    hasPM = false;
    newlines = (size_t *)malloc(lines * sizeof(size_t));
    if (newlines == NULL) {
      string error_msg = "[CSVPlugin: ] Malloc Failure";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }

    if (fd != -1) {
      /* -1 bc field 0 does not have to be indexed */
      int pmFields = (rec.getArgsNo() / policy) - 1;
      if (pmFields < 0) {
        std::cout << pmFields << " " << rec.getArgsNo() << " " << policy
                  << std::endl;
        string error_msg = "[CSVPlugin: ] Erroneous PM policy";
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
      LOG(INFO) << "PM will have " << pmFields << " field(s)";
      pm = (short **)malloc(lines * sizeof(short *));
      short *pm_ = (short *)malloc(lines * pmFields * sizeof(short));
      for (size_t i = 0; i < lines; i++) {
        pm[i] = (pm_ + i * pmFields);
      }

      /* Store PM in cache */
      /* To be used by subsequent queries */
      pmCSV *pmStruct = new pmCSV();
      pmStruct->newlines = newlines;
      pmStruct->offsets = pm;
      pmCast = (char *)pmStruct;
      cache.registerPM(fname, pmCast);
    }
  } else {
    // cout << "(CSV) PM REUSE" << endl;
    hasPM = true;
    pmCSV *pmStruct = (pmCSV *)pmCast;

    this->newlines = pmStruct->newlines;
    this->pm = pmStruct->offsets;
  }
}

CSVPlugin::CSVPlugin(Context *const context, string &fname, RecordType &rec,
                     vector<RecordAttribute *> whichFields, char delimInner,
                     int lineHint, int policy, bool stringBrackets)
    : fname(fname),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      fsizeVar("fileSize"),
      lines(lineHint),
      policy(policy),
      stringBrackets(stringBrackets) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  fd = -1;
  buf = NULL;

  struct {
    bool operator()(RecordAttribute *a, RecordAttribute *b) {
      return a->getAttrNo() < b->getAttrNo();
    }
  } orderByNumber;
  std::sort(wantedFields.begin(), wantedFields.end(), orderByNumber);

  LOG(INFO) << "[CSVPlugin: ] " << fname;
  struct stat statbuf;
  const char *name_c = fname.c_str();
  stat(name_c, &statbuf);
  fsize = statbuf.st_size;

  fd = open(name_c, O_RDONLY);
  if (fd == -1 && whichFields.size() > 0) {
    throw runtime_error(string("csv.open"));
  }

  this->delimInner = delimInner;
  this->delimEnd = '\n';

  /* PM */
  CachingService &cache = CachingService::getInstance();

  char *pmCast = cache.getPM(fname);
  if (pmCast == NULL) {
    cout << "NEW (CSV) PM" << endl;
    hasPM = false;
    newlines = (size_t *)malloc(lines * sizeof(size_t));
    if (newlines == NULL) {
      string error_msg = "[CSVPlugin: ] Malloc Failure";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }

    /* -1 bc field 0 does not have to be indexed */
    int pmFields = (rec.getArgsNo() / policy);  // - 1;
    if (pmFields < 0) {
      string error_msg = "[CSVPlugin: ] Erroneous PM policy";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    LOG(INFO) << "PM will have " << pmFields << " field(s)";
    pm = (short **)malloc(lines * sizeof(short *));
    short *pm_ = (short *)malloc(lines * pmFields * sizeof(short));
    for (size_t i = 0; i < lines; i++) {
      pm[i] = (pm_ + i * pmFields);
    }

    /* Store PM in cache */
    /* To be used by subsequent queries */
    pmCSV *pmStruct = new pmCSV();
    pmStruct->newlines = newlines;
    pmStruct->offsets = pm;
    pmCast = (char *)pmStruct;
    cache.registerPM(fname, pmCast);
  } else {
    cout << "(CSV) PM REUSE" << endl;
    hasPM = true;
    pmCSV *pmStruct = (pmCSV *)pmCast;

    this->newlines = pmStruct->newlines;
    this->pm = pmStruct->offsets;
  }
}

CSVPlugin::CSVPlugin(Context *const context, string &fname, RecordType &rec,
                     vector<RecordAttribute *> whichFields, int lineHint,
                     int policy, size_t *newlines, short **offsets,
                     bool stringBrackets)
    : fname(fname),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      fsizeVar("fileSize"),
      lines(lineHint),
      policy(policy),
      stringBrackets(stringBrackets) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  fd = -1;
  buf = NULL;
  std::sort(wantedFields.begin(), wantedFields.end());

  LOG(INFO) << "[CSVPlugin: ] " << fname;
  struct stat statbuf;
  const char *name_c = fname.c_str();
  stat(name_c, &statbuf);
  fsize = statbuf.st_size;

  fd = open(name_c, O_RDONLY);
  if (fd == -1 && whichFields.size() > 0) {
    throw runtime_error(string("csv.open"));
  }

  /* PM */
  hasPM = true;
  this->newlines = newlines;
  this->pm = offsets;
  this->delimInner = ';';
  this->delimEnd = '\n';
}

CSVPlugin::CSVPlugin(Context *const context, string &fname, RecordType &rec,
                     vector<RecordAttribute *> whichFields, char delimInner,
                     int lineHint, int policy, size_t *newlines,
                     short **offsets, bool stringBrackets)
    : fname(fname),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      fsizeVar("fileSize"),
      lines(lineHint),
      policy(policy),
      stringBrackets(stringBrackets) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  fd = -1;
  buf = NULL;
  std::sort(wantedFields.begin(), wantedFields.end());

  LOG(INFO) << "[CSVPlugin: ] " << fname;
  struct stat statbuf;
  const char *name_c = fname.c_str();
  stat(name_c, &statbuf);
  fsize = statbuf.st_size;

  fd = open(name_c, O_RDONLY);
  if (fd == -1 && whichFields.size() > 0) {
    throw runtime_error(string("csv.open"));
  }

  /* PM */
  hasPM = true;
  this->newlines = newlines;
  this->pm = offsets;
  this->delimInner = delimInner;
  this->delimEnd = '\n';
}

CSVPlugin::~CSVPlugin() {}

void CSVPlugin::init() {
  context->setGlobalFunction(true);

  /* Preparing the codegen part */
  Function *F = context->getGlobalFunction();
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  /* PM - LLVM LAND */
  PointerType *size_tPtrType = Type::getInt64PtrTy(llvmContext);
  mem_newlines =
      context->CreateEntryBlockAlloca(F, string("mem_newlines"), size_tPtrType);
  Value *val_newlines =
      context->CastPtrToLlvmPtr(size_tPtrType, (char *)newlines);
  Builder->CreateStore(val_newlines, mem_newlines);

  Type *int16PtrType = Type::getInt16PtrTy(llvmContext);
  PointerType *int162DPtrType = PointerType::get(int16PtrType, 0);
  mem_pm = context->CreateEntryBlockAlloca(F, string("mem_pm"), int162DPtrType);
  Value *val_pm = context->CastPtrToLlvmPtr(int162DPtrType, (char *)pm);
  Builder->CreateStore(val_pm, mem_pm);

  Type *int32Type = Type::getInt32Ty(llvmContext);
  // int32Type->getTypeID();
  mem_lineCtr =
      context->CreateEntryBlockAlloca(F, string("mem_lineCtr"), int32Type);
  Value *val_zero = context->createInt32(0);
  Builder->CreateStore(val_zero, mem_lineCtr);

  /* Pages may be read AND WRITTEN (to compute hashes in-place when needed) */
  if (fd != -1) {
    buf = (char *)mmap(NULL, fsize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE /*| MAP_POPULATE*/, fd, 0);
    if (buf == MAP_FAILED) {
      throw runtime_error(string("csv.mmap"));
    }
  } else {
    buf = NULL;
  }

  // Allocating memory
  AllocaInst *offsetMem = context->CreateEntryBlockAlloca(
      F, string(posVar), Type::getInt64Ty(llvmContext));
  AllocaInst *bufMem =
      context->CreateEntryBlockAlloca(F, string(bufVar), charPtrType);
  AllocaInst *fsizeMem = context->CreateEntryBlockAlloca(
      F, string(fsizeVar), Type::getInt64Ty(llvmContext));
  Value *offsetVal = Builder->getInt64(0);
  Builder->CreateStore(offsetVal, offsetMem);
  NamedValuesCSV[posVar] = offsetMem;

  Value *fsizeVal = Builder->getInt64(fsize);
  Builder->CreateStore(fsizeVal, fsizeMem);
  NamedValuesCSV[fsizeVar] = fsizeMem;

  // Typical way to pass a pointer via the LLVM API
  AllocaInst *AllocaPtr =
      context->CreateEntryBlockAlloca(F, string("charPtr"), charPtrType);
  Value *ptrVal = ConstantInt::get(llvmContext, APInt(64, ((uint64_t)buf)));
  // i8*
  Value *unshiftedPtr = Builder->CreateIntToPtr(ptrVal, charPtrType);
  Builder->CreateStore(unshiftedPtr, bufMem);
  NamedValuesCSV[bufVar] = bufMem;
};

void CSVPlugin::generate(const ::Operator &producer) {
  if (!hasPM) {
    return scanAndPopulatePM(producer);
  } else {
    return scanPM(producer);
  }
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
ProteusValueMemory CSVPlugin::readPath(string activeRelation, Bindings bindings,
                                       const char *pathVar,
                                       RecordAttribute attr) {
  ProteusValueMemory mem_valWrapper;
  {
    const OperatorState *state = bindings.state;
    const map<RecordAttribute, ProteusValueMemory> &csvProjections =
        state->getBindings();
    RecordAttribute tmpKey =
        RecordAttribute(fname, pathVar, this->getOIDType());
    map<RecordAttribute, ProteusValueMemory>::const_iterator it;
    it = csvProjections.find(tmpKey);
    if (it == csvProjections.end()) {
      string error_msg =
          string("[CSV plugin - readPath ]: Unknown variable name ") + pathVar;
      cout << "Nothing found for " << fname << "_" << pathVar << " in "
           << csvProjections.size() << " bindings" << endl;
      for (it = csvProjections.begin(); it != csvProjections.end(); it++) {
        RecordAttribute attr = it->first;
        cout << attr.getRelationName() << "_" << attr.getAttrName() << endl;
      }
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    mem_valWrapper = it->second;
  }

  return mem_valWrapper;
}

ProteusValueMemory CSVPlugin::readValue(ProteusValueMemory mem_value,
                                        const ExpressionType *type) {
  return mem_value;
}

ProteusValue CSVPlugin::readCachedValue(CacheInfo info,
                                        const OperatorState &currState) {
  return readCachedValue(info, currState.getBindings());
}

ProteusValue CSVPlugin::readCachedValue(
    CacheInfo info, const map<RecordAttribute, ProteusValueMemory> &bindings) {
  IRBuilder<> *const Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();

  /* Need OID to retrieve corresponding value from bin. cache */
  RecordAttribute tupleIdentifier =
      RecordAttribute(fname, activeLoop, getOIDType());
  map<RecordAttribute, ProteusValueMemory>::const_iterator it =
      bindings.find(tupleIdentifier);
  if (it == bindings.end()) {
    string error_msg =
        "[Expression Generator: ] Current tuple binding / OID not found";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory mem_oidWrapper = it->second;
  /* OID is a plain integer - starts from 0!!*/
  Value *val_oid = Builder->CreateLoad(mem_oidWrapper.mem);

  StructType *cacheType = context->ReproduceCustomStruct(info.objectTypes);
  // Value *typeSize = ConstantExpr::getSizeOf(cacheType);
  char *rawPtr = *(info.payloadPtr);
  int posInStruct = info.structFieldNo;

  /* Cast to appr. type */
  PointerType *ptr_cacheType = PointerType::get(cacheType, 0);
  Value *val_cachePtr = context->CastPtrToLlvmPtr(ptr_cacheType, rawPtr);

  Value *val_cacheShiftedPtr = context->getArrayElemMem(val_cachePtr, val_oid);
  Value *val_cachedField =
      context->getStructElem(val_cacheShiftedPtr, posInStruct);
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
    /* Obviously only works to peek integer fields */
    vector<Value *> ArgsV;

    Function *debugSth = context->getFunction("printi");
    ArgsV.push_back(val_oid);
    Builder->CreateCall(debugSth, ArgsV);
  }
#endif
  return valWrapper;
}

ProteusValue CSVPlugin::hashValue(ProteusValueMemory mem_value,
                                  const ExpressionType *type) {
  IRBuilder<> *Builder = context->getBuilder();
  ProteusValue v{Builder->CreateLoad(mem_value.mem), mem_value.isNull};
  return hashPrimitive(v, type->getTypeID(), context);
}

ProteusValue CSVPlugin::hashValueEager(ProteusValue valWrapper,
                                       const ExpressionType *type) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();
  Value *tmp = valWrapper.value;
  AllocaInst *mem_tmp =
      context->CreateEntryBlockAlloca(F, "mem_cachedToHash", tmp->getType());
  Builder->CreateStore(tmp, mem_tmp);
  ProteusValueMemory mem_tmpWrapper = {mem_tmp, valWrapper.isNull};
  return hashValue(mem_tmpWrapper, type);
}

void CSVPlugin::flushValue(ProteusValueMemory mem_value,
                           const ExpressionType *type, Value *fileName) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *flushFunc;
  Value *val_attr = Builder->CreateLoad(mem_value.mem);
  switch (type->getTypeID()) {
    case BOOL: {
      flushFunc = context->getFunction("flushBoolean");
      vector<Value *> ArgsV;
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case STRING: {
      /* Untested */
      flushFunc = context->getFunction("flushStringObj");
      vector<Value *> ArgsV;
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case FLOAT: {
      flushFunc = context->getFunction("flushDouble");
      vector<Value *> ArgsV;
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case INT: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushInt");
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case INT64: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushInt64");
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case DATE: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushDate");
      ArgsV.push_back(val_attr);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case DSTRING: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushDString");
      void *dict = ((const DStringType *)type)->getDictionary();
      LLVMContext &llvmContext = context->getLLVMContext();
      Value *llvmDict = context->CastPtrToLlvmPtr(
          PointerType::getInt8PtrTy(llvmContext), dict);
      ArgsV.push_back(val_attr);
      ArgsV.push_back(llvmDict);
      ArgsV.push_back(fileName);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
      return;
    }
    case BAG:
    case LIST:
    case SET:
      LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain collections";
      throw runtime_error(
          string("[CSV PLUGIN: ] CSV files do not contain collections"));
    case RECORD: {
      char delim = ',';

      Function *flushStr = context->getFunction("flushStringCv2");
      Function *flushFunc = context->getFunction("flushChar");
      vector<Value *> ArgsV{context->createInt8(delim), fileName};

      const list<RecordAttribute *> &attrs =
          ((const RecordType *)type)->getArgs();
      list<expressions::AttributeConstruction>::const_iterator it;

      size_t i = 0;
      for (const auto &attr : attrs) {
        // value
        ProteusValue partialFlush;
        partialFlush.value = Builder->CreateExtractValue(val_attr, i);
        partialFlush.isNull = mem_value.isNull;
        flushValueEager(partialFlush, attr->getOriginalType(), fileName);

        // comma, if needed
        ++i;
        if (i != attrs.size()) {
          context->getBuilder()->CreateCall(flushFunc, ArgsV);
        }
      }

      return;
    }
    default:
      LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
      throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
  }
}

void CSVPlugin::flushValueEager(ProteusValue valWrapper,
                                const ExpressionType *type, Value *fileName) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();
  Value *tmp = valWrapper.value;
  AllocaInst *mem_tmp =
      context->CreateEntryBlockAlloca(F, "mem_cachedToFlush", tmp->getType());
  Builder->CreateStore(tmp, mem_tmp);
  ProteusValueMemory mem_tmpWrapper = {mem_tmp, valWrapper.isNull};
  return flushValue(mem_tmpWrapper, type, fileName);
}

void CSVPlugin::finish() {
  close(fd);
  munmap(buf, fsize);
}

Value *CSVPlugin::getValueSize(ProteusValueMemory mem_value,
                               const ExpressionType *type) {
  switch (type->getTypeID()) {
    case BOOL:
    case INT:
    case FLOAT:
    case STRING: {
      Type *explicitType = (mem_value.mem)->getAllocatedType();
      return context->createInt64(context->getSizeOf(explicitType));
    }
    case BAG:
    case LIST:
    case SET: {
      string error_msg = string("[CSV Plugin]: Cannot contain collections");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    case RECORD: {
      string error_msg = string("[CSV Plugin]: Cannot contain records");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    default: {
      string error_msg = string("[CSV Plugin]: Unknown datatype");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
}

void CSVPlugin::skipDelimLLVM(Value *delim, Function *debugChar,
                              Function *debugInt) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca =
      context->CreateEntryBlockAlloca(TheFunction, "cur_pos", int64Type);
  // Store the value into the alloca.
  Builder->CreateStore(Builder->CreateLoad(pos, "start_pos"), Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB =
      BasicBlock::Create(llvmContext, "skipDelimLoop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Emit the body of the loop.
  // Here we essentially have no body; we only need to take care of the 'step'
  // Emit the step value. (+1)
  Value *StepVal = Builder->getInt64(1);

  // Compute the end condition.
  // Involves pointer arithmetics
  Value *index = Builder->CreateLoad(Alloca);
  Value *lhsPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
  Value *lhs = Builder->CreateLoad(lhsShiftedPtr, "bufVal");
  Value *rhs = delim;
  Value *EndCond = Builder->CreateICmpNE(lhs, rhs);

  // Reload, increment, and restore the alloca.
  // This handles the case where the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateLoad(Alloca);
  Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
  Builder->CreateStore(NextVar, Alloca);

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(llvmContext, "afterSkipDelimLoop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);
  ////////////////////////////////

  //'return' pos value
  Value *finalVar = Builder->CreateLoad(Alloca);
  Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);
}

void CSVPlugin::skipDelimLLVM(Value *delim) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca =
      context->CreateEntryBlockAlloca(TheFunction, "cur_pos", int64Type);
  // Store the value into the alloca.
  Builder->CreateStore(Builder->CreateLoad(pos, "start_pos"), Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB =
      BasicBlock::Create(llvmContext, "skipDelimLoop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Emit the body of the loop.
  // Here we essentially have no body; we only need to take care of the 'step'
  // Emit the step value. (+1)
  Value *StepVal = Builder->getInt64(1);

  // Compute the end condition.
  // Involves pointer arithmetics
  Value *index = Builder->CreateLoad(Alloca);
  Value *lhsPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
  Value *lhs = Builder->CreateLoad(lhsShiftedPtr, "bufVal");
  Value *rhs = delim;
  Value *EndCond = Builder->CreateICmpNE(lhs, rhs);

  // Reload, increment, and restore the alloca.
  // This handles the case where the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateLoad(Alloca);
  Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
  Builder->CreateStore(NextVar, Alloca);

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(llvmContext, "afterSkipDelimLoop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);

  //'return' pos value
  Value *finalVar = Builder->CreateLoad(Alloca);
  Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);
}

void CSVPlugin::skipDelimBackwardsLLVM(Value *delim) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Function *F = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst *mem_cur_pos =
      context->CreateEntryBlockAlloca(F, "cur_pos", int64Type);

  // Goto ending position of previous field
  Value *start_pos = Builder->CreateLoad(pos, "start_pos");
#ifdef DEBUGPM
  {
    vector<Value *> ArgsV;
    Function *debugInt64 = context->getFunction("printi64");
    ArgsV.push_back(start_pos);
    Builder->CreateCall(debugInt64, ArgsV);
  }
#endif
  start_pos = Builder->CreateSub(start_pos, context->createInt64(2));
  Builder->CreateStore(start_pos, mem_cur_pos);
  Value *val_step = Builder->getInt64(1);

  BasicBlock *skipCond, *skipBody, *skipInc, *skipEnd;
  context->CreateForLoop("skipBwdCond", "skipBwdBody", "skipBwdInc",
                         "skipBwdEnd", &skipCond, &skipBody, &skipInc,
                         &skipEnd);

  /* Condition: buf[pos] != 'delim' */
  Builder->CreateBr(skipCond);
  Builder->SetInsertPoint(skipCond);

  Value *val_index = Builder->CreateLoad(mem_cur_pos);
  Value *lhsPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, val_index);
  Value *lhs = Builder->CreateLoad(lhsShiftedPtr, "bufVal");
  Value *rhs = delim;
  Value *val_cond = Builder->CreateICmpNE(lhs, rhs);

  Builder->CreateCondBr(val_cond, skipBody, skipEnd);
  /* Body: No actual work done */
  Builder->SetInsertPoint(skipBody);
  Builder->CreateBr(skipInc);
  /* Inc: pos-- */
  Builder->SetInsertPoint(skipInc);
  val_index = Builder->CreateSub(val_index, val_step);
  Builder->CreateStore(val_index, mem_cur_pos);
  Builder->CreateBr(skipCond);

  /* End */
  Builder->SetInsertPoint(skipEnd);
  Value *val_finalPos = Builder->CreateLoad(mem_cur_pos);
  val_finalPos = Builder->CreateAdd(val_finalPos, val_step);
  Builder->CreateStore(val_finalPos, NamedValuesCSV[posVar]);
#ifdef DEBUGPM
  {
    vector<Value *> ArgsV;
    Function *debugInt64 = context->getFunction("printi64");
    ArgsV.push_back(val_finalPos);
    Builder->CreateCall(debugInt64, ArgsV);
  }
#endif
}

void CSVPlugin::skipLLVM() {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, this->delimInner));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, this->delimEnd));

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }
  AllocaInst *fsizePtr;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(fsizeVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + fsizeVar);
    }
    fsizePtr = it->second;
  }
  // Since we are the ones dictating what is flushed, file size should never
  // have to be used in a check

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca =
      context->CreateEntryBlockAlloca(TheFunction, "cur_pos", int64Type);
  Value *fsizeVal = Builder->CreateLoad(fsizePtr, "file_size");
  // Store the current pos value into the alloca, so that loop starts from
  // appropriate point. Redundant store / loads will be simplified by opt.pass
  Value *toInit = Builder->CreateLoad(pos, "start_pos");
  Builder->CreateStore(toInit, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "skipLoop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Emit the body of the loop.

  // Here we essentially have no body; we only need to take care of the 'step'
  // Emit the step value. (+1)
  Value *StepVal = Builder->getInt64(1);

  // Compute the end condition. More complex in this scenario (3 ands)
  Value *index = Builder->CreateLoad(Alloca);
  Value *lhsPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
  // equivalent to buf[pos]
  Value *lhs_ = Builder->CreateLoad(lhsShiftedPtr, "bufVal");
  // Only difference between skip() and skipDelim()!!!
  Value *rhs1 = delimInner;
  Value *rhs2 = delimEnd;
  Value *EndCond1 = Builder->CreateICmpNE(lhs_, rhs1);
  Value *EndCond2 = Builder->CreateICmpNE(lhs_, rhs2);
  Value *EndCond3 =
      Builder->CreateICmpSLT(Builder->CreateLoad(Alloca), fsizeVal);
  Value *EndCond_ = Builder->CreateAnd(EndCond1, EndCond2);
  Value *EndCond = Builder->CreateAnd(EndCond_, EndCond3);

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateLoad(Alloca);
  Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
  Builder->CreateStore(NextVar, Alloca);

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(llvmContext, "afterSkipLoop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);

  //'return' pos value
  Value *finalVar = Builder->CreateLoad(Alloca);
  Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);
}

void CSVPlugin::getFieldEndLLVM() {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, this->delimInner));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, this->delimEnd));

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }
  AllocaInst *fsizePtr;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(fsizeVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + fsizeVar);
    }
    fsizePtr = it->second;
  }
  // Since we are the ones dictating what is flushed, file size should never
  // have to be used in a check

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca =
      context->CreateEntryBlockAlloca(TheFunction, "cur_pos", int64Type);
  Value *fsizeVal = Builder->CreateLoad(fsizePtr, "file_size");
  // Store the current pos value into the alloca, so that loop starts from
  // appropriate point. Redundant store / loads will be simplified by opt.pass
  Value *toInit = Builder->CreateLoad(pos, "start_pos");
  Builder->CreateStore(toInit, Alloca);

  BasicBlock *fieldCond, *fieldBody, *fieldInc, *fieldEnd;
  context->CreateForLoop("fieldCond", "fieldBody", "fieldInc", "fieldEnd",
                         &fieldCond, &fieldBody, &fieldInc, &fieldEnd);

  Builder->CreateBr(fieldCond);
  Builder->SetInsertPoint(fieldCond);
  Value *StepVal = Builder->getInt64(1);
  // Compute the end condition. More complex in this scenario (3 ands)
  Value *index = Builder->CreateLoad(Alloca);
  Value *lhsPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *lhsShiftedPtr = Builder->CreateInBoundsGEP(lhsPtr, index);
  // equivalent to buf[pos]
  Value *lhs_ = Builder->CreateLoad(lhsShiftedPtr, "bufVal");
  // Only difference between skip() and skipDelim()!!!
  Value *rhs1 = delimInner;
  Value *rhs2 = delimEnd;
  Value *EndCond1 = Builder->CreateICmpNE(lhs_, rhs1);
  Value *EndCond2 = Builder->CreateICmpNE(lhs_, rhs2);
  Value *EndCond3 =
      Builder->CreateICmpSLT(Builder->CreateLoad(Alloca), fsizeVal);
  Value *EndCond_ = Builder->CreateAnd(EndCond1, EndCond2);
  Value *EndCond = Builder->CreateAnd(EndCond_, EndCond3);
  Builder->CreateCondBr(EndCond, fieldBody, fieldEnd);

  Builder->SetInsertPoint(fieldBody);
  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateLoad(Alloca);
  Value *NextVar = Builder->CreateAdd(CurVar, StepVal, "new_pos");
  Builder->CreateStore(NextVar, Alloca);
  Builder->CreateBr(fieldInc);

  Builder->SetInsertPoint(fieldInc);
  Builder->CreateBr(fieldCond);

  Builder->SetInsertPoint(fieldEnd);
  //'return' pos value
  Value *finalVar = Builder->CreateLoad(Alloca);
  Builder->CreateStore(finalVar, NamedValuesCSV[posVar]);
}

void CSVPlugin::readField(typeID id, RecordAttribute attName,
                          map<RecordAttribute, ProteusValueMemory> &variables) {
  cout << "READ (RAW) FIELD " << attName.getAttrName() << endl;
  switch (id) {
    case BOOL: {
      readAsBooleanLLVM(attName, variables);
      break;
    }
    case STRING: {
      readAsStringLLVM(attName, variables);
      break;
    }
    case FLOAT: {
      readAsFloatLLVM(attName, variables);
      break;
    }
    case INT: {
      readAsIntLLVM(attName, variables);
      break;
    }
    case BAG:
    case LIST:
    case SET: {
      string msg = "[CSV PLUGIN: ] CSV files do not contain collections";
      LOG(ERROR) << msg;
      throw runtime_error(msg);
    }
    case RECORD: {
      string msg =
          "[CSV PLUGIN: ] CSV files do not contain record-valued attributes";
      LOG(ERROR) << msg;
      throw runtime_error(msg);
    }
    default: {
      string msg = "[CSV PLUGIN: ] Unknown datatype";
      LOG(ERROR) << msg;
      throw runtime_error(msg);
    }
  }
}

void CSVPlugin::readAsIntLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables, Function *atoi_,
    Function *debugChar, Function *debugInt) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Value *start = Builder->CreateLoad(pos, "start_pos_atoi");
  getFieldEndLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(pos, "end_pos_atoi");
  // Must increase offset by 1 now
  //(uniform behavior with other skip methods)
  Value *val_1 = Builder->getInt64(1);
  Value *pos_inc = Builder->CreateAdd(index, val_1);
  Builder->CreateStore(pos_inc, pos);
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  Value *len = Builder->CreateSub(index, start);
  Value *len_32 = Builder->CreateTrunc(len, int32Type);

  vector<Value *> ArgsV;
  ArgsV.clear();
  ArgsV.push_back(bufShiftedPtr);
  ArgsV.push_back(len_32);

  Function *atois = context->getFunction("atois");
  Value *parsedInt = Builder->CreateCall(atois, ArgsV, "atois");
  AllocaInst *currResult =
      context->CreateEntryBlockAlloca(TheFunction, "currResult", int32Type);
  Builder->CreateStore(parsedInt, currResult);

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = currResult;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

void CSVPlugin::readAsIntLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Fetch values from symbol table
  AllocaInst *mem_pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    mem_pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Value *start = Builder->CreateLoad(mem_pos, "start_pos_atoi");
  getFieldEndLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(mem_pos, "end_pos_atoi");
  // Must increase offset by 1 now
  //(uniform behavior with other skip methods)
  Value *val_1 = Builder->getInt64(1);
  Value *pos_inc = Builder->CreateAdd(index, val_1);
  Builder->CreateStore(pos_inc, mem_pos);
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  Value *len = Builder->CreateSub(index, start);
  Value *len_32 = Builder->CreateTrunc(len, int32Type);

  AllocaInst *mem_result = context->CreateEntryBlockAlloca(
      TheFunction, "mem_currIntResult", int32Type);
  atois(bufShiftedPtr, len_32, mem_result, context);

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_result;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
#ifdef DEBUGPM
  {
    Function *debugInt = context->getFunction("printi");
    vector<Value *> ArgsV;

    ArgsV.push_back(Builder->CreateLoad(mem_result));
    Builder->CreateCall(debugInt, ArgsV, "printi");
    ArgsV.clear();
    ArgsV.push_back(context->createInt32(1001));
    Builder->CreateCall(debugInt, ArgsV, "printi");
  }
#endif

  // cout << "[CSV_PM: ] Stored " << attName.getAttrName() << endl;
}

void CSVPlugin::readAsStringLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();

  // Fetch values from symbol table
  AllocaInst *mem_pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    mem_pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Value *start = Builder->CreateLoad(mem_pos, "start_pos_str");
  Value *val_1 = Builder->getInt64(1);
  // Also skipping opening bracket!!
  if (stringBrackets) {
    start = Builder->CreateAdd(start, val_1);
  }
  getFieldEndLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(mem_pos, "end_pos_str");
  // Must increase offset by 1 now
  //(uniform behavior with other skip methods)

  Value *pos_inc = Builder->CreateAdd(index, val_1);
  Builder->CreateStore(pos_inc, mem_pos);
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  Value *len = Builder->CreateSub(index, start);
  // Also removing (size of) ending bracket!!
  if (stringBrackets) {
    len = Builder->CreateSub(len, val_1);
  }
  Value *len_32 = Builder->CreateTrunc(len, int32Type);

  StructType *stringObjType = context->CreateStringStruct();
  AllocaInst *mem_convertedValue = context->CreateEntryBlockAlloca(
      F, string("ingestedString"), stringObjType);
  Value *mem_str = context->getStructElemMem(mem_convertedValue, 0);
  Builder->CreateStore(bufShiftedPtr, mem_str);

  Value *mem_len = context->getStructElemMem(mem_convertedValue, 1);
  Builder->CreateStore(len_32, mem_len);

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_convertedValue;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;

  // cout << "[CSV_PM: ] Stored " << attName.getAttrName() << endl;
}

void CSVPlugin::readAsBooleanLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Type *int1Type = Type::getInt1Ty(llvmContext);

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *start = Builder->CreateLoad(pos, "start_pos_atob");
  getFieldEndLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(pos, "end_pos_atob");
  // Must increase offset by 1 now
  //(uniform behavior with other skip methods)
  Value *val_1 = Builder->getInt64(1);
  Value *pos_inc = Builder->CreateAdd(index, val_1);
  Builder->CreateStore(pos_inc, pos);

  vector<Value *> ArgsV;
  ArgsV.push_back(bufPtr);
  ArgsV.push_back(start);
  ArgsV.push_back(index);
  Function *conversionFunc = context->getFunction("convertBoolean64");

  Value *convertedValue =
      Builder->CreateCall(conversionFunc, ArgsV, "convertBoolean64");
  AllocaInst *mem_convertedValue =
      context->CreateEntryBlockAlloca(TheFunction, "currResult", int1Type);
  Builder->CreateStore(convertedValue, mem_convertedValue);

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_convertedValue;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

void CSVPlugin::readAsFloatLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables, Function *atof_,
    Function *debugChar, Function *debugFloat) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Fetch values from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Value *start = Builder->CreateLoad(pos, "start_pos_atoi");
  skipLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(pos, "end_pos_atoi");
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  vector<Value *> ArgsV;
  ArgsV.clear();
  ArgsV.push_back(bufShiftedPtr);
  Value *parsedFloat = Builder->CreateCall(atof_, ArgsV, "atof");
  AllocaInst *currResult =
      context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
  Builder->CreateStore(parsedFloat, currResult);

#ifdef DEBUGPM
  {
    ArgsV.clear();
    ArgsV.push_back(parsedFloat);
    Builder->CreateCall(debugFloat, ArgsV, "printf");
  }
#endif
  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = currResult;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

void CSVPlugin::readAsFloatLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  Function *atof_ = context->getFunction("atof");

  // Fetch values from symbol table
  AllocaInst *mem_pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    mem_pos = it->second;
  }
  AllocaInst *buf;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(bufVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + bufVar);
    }
    buf = it->second;
  }

  Value *start = Builder->CreateLoad(mem_pos, "start_pos_atof");
#ifdef DEBUGPM
  {
    vector<Value *> ArgsV;
    ArgsV.clear();
    ArgsV.push_back(start);
    Function *debugInt = context->getFunction("printi64");
    Function *debugFloat = context->getFunction("printFloat");
    Builder->CreateCall(debugInt, ArgsV, "printf");
  }
#endif
  skipLLVM();
  // index must be different than start!
  Value *index = Builder->CreateLoad(mem_pos, "end_pos_atof");
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  vector<Value *> ArgsV;
  ArgsV.clear();
  ArgsV.push_back(bufShiftedPtr);
  Value *parsedFloat = Builder->CreateCall(atof_, ArgsV, "atof");
  AllocaInst *currResult =
      context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
  Builder->CreateStore(parsedFloat, currResult);

#ifdef DEBUGPM
  {
    Function *debugFloat = context->getFunction("printFloat");
    ArgsV.clear();
    ArgsV.push_back(parsedFloat);
    Builder->CreateCall(debugFloat, ArgsV, "printf");
  }
#endif
  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = currResult;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

/* Scans Input File and Populates PM */
void CSVPlugin::scanAndPopulatePM(const ::Operator &producer) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int16Type = Type::getInt16Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> *variableBindings =
      new map<RecordAttribute, ProteusValueMemory>();

  // Fetch value from symbol table
  AllocaInst *pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    pos = it->second;
  }
  AllocaInst *fsizePtr;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(fsizeVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + fsizeVar);
    }
    fsizePtr = it->second;
  }

  //  BYTECODE
  //    entry:
  //    %pos.addr = alloca i32, align 4
  //    %fsize.addr = alloca i32, align 4
  //    store i32 %pos, i32* %pos.addr, align 4
  //    store i32 %fsize, i32* %fsize.addr, align 4
  //    br label %for.cond

  //  API equivalent: Only the branch is needed. The allocas were taken care of
  //  before

  // Get the ENTRY BLOCK
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  BasicBlock *CondBB =
      BasicBlock::Create(llvmContext, "csvScanCond", TheFunction);

  // Start insertion in CondBB.
  Builder->SetInsertPoint(CondBB);
  Value *lhs = Builder->CreateLoad(pos);
  Value *rhs = Builder->CreateLoad(fsizePtr);
  Value *cond = Builder->CreateICmpSLT(lhs, rhs);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB =
      BasicBlock::Create(llvmContext, "csvScanBody", TheFunction);

  // Create the "AFTER LOOP" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(llvmContext, "csvScanEnd", TheFunction);
  context->setEndingBlock(AfterBB);

  // Insert the conditional branch into the end of CondBB.
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Get the line number and pass it along.
  // More general/lazy CSV plugins will only perform this action,
  // instead of eagerly converting fields
  ExpressionType *oidType = new IntType();
  RecordAttribute tupleIdentifier = RecordAttribute(fname, activeLoop, oidType);

  ProteusValueMemory mem_posWrapper;
  // mem_posWrapper.mem = pos;
  mem_posWrapper.mem = mem_lineCtr;
  mem_posWrapper.isNull = context->createFalse();
  (*variableBindings)[tupleIdentifier] = mem_posWrapper;

  /* Actual Work (Loop through attributes etc.) */
  int cur_col = 0;
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, this->delimInner));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, this->delimEnd));
  Function *atoi_ = context->getFunction("atoi");
  Function *atof_ = context->getFunction("atof");
  Function *debugChar = context->getFunction("printc");
  Function *debugInt = context->getFunction("printi");
  Function *debugFloat = context->getFunction("printFloat");

  if (atoi_ == 0 || atof_ == 0 || debugChar == 0 || debugInt == 0 ||
      debugFloat == 0) {
    LOG(ERROR) << "One of the functions needed not found!";
    throw runtime_error(string("One of the functions needed not found!"));
  }

  /* Store (current) newline position */
  AllocaInst *mem_pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    mem_pos = it->second;
  }
  Value *val_posNewline = Builder->CreateLoad(mem_pos);
  Value *val_newlines = Builder->CreateLoad(mem_newlines);
  Value *val_lineCtr = Builder->CreateLoad(mem_lineCtr);
  Value *newlinesShifted =
      Builder->CreateInBoundsGEP(val_newlines, val_lineCtr);
  Builder->CreateStore(val_posNewline, newlinesShifted);

  /* Loop through fields and gather info (values needed and PM offsets) */
  int pmFields = rec.getArgsNo() / policy;
  vector<int> pmFieldsNo;
  for (int i = 1; i <= pmFields; i++) {
    pmFieldsNo.push_back(i * policy);
  }

  // Using it to know if a final skip is needed
  int lastFieldNo = 0;
  int pmCtr = 0;
  for (int i = 1; i <= rec.getArgsNo(); i++) {
    LOG(INFO) << "Considering field " << i;
    if (pmFieldsNo.empty() && wantedFields.empty() &&
        lastFieldNo < rec.getArgsNo()) {
      skipDelimLLVM(delimEnd, debugChar, debugInt);
      LOG(INFO) << "(All empty) Skip field " << i;
      break;
    }

    /* Check usefulness for PM */
    /* Field Numbers in Record expected to start from 1!! */
    vector<int>::iterator pmIt;
    pmIt = find(pmFieldsNo.begin(), pmFieldsNo.end(), i - 1);
    if (pmIt != pmFieldsNo.end()) {
      //            cout << "Place field " << i << ") in PM"<<endl;

      /* What to store: offset */
      AllocaInst *mem_pos;
      {
        map<string, AllocaInst *>::iterator it;
        it = NamedValuesCSV.find(posVar);
        if (it == NamedValuesCSV.end()) {
          string msg = string("Unknown variable name: ") + posVar;
          throw runtime_error(msg);
        }
        mem_pos = it->second;
      }
      Value *val_pos = Builder->CreateLoad(mem_pos);
      Value *val_offset = Builder->CreateSub(val_pos, val_posNewline);
      Value *val_offset16 = Builder->CreateTrunc(val_offset, int16Type);

      Value *val_pm = Builder->CreateLoad(mem_pm);
      Value *mem_pmRow = Builder->CreateInBoundsGEP(val_pm, val_lineCtr);
      Value *val_pmRow = Builder->CreateLoad(mem_pmRow);
      Value *val_pmColIdx = context->createInt32(pmCtr);
      Value *val_pmCol = Builder->CreateInBoundsGEP(val_pmRow, val_pmColIdx);
      Builder->CreateStore(val_offset16, val_pmCol);

#ifdef DEBUGPM
      {
        //                vector<Value*> ArgsV;
        //                Function* debugInt32 = context->getFunction("printi");
        //                ArgsV.push_back(val_pmColIdx);
        //                Builder->CreateCall(debugInt32, ArgsV);
        //                ArgsV.clear();
        //                Function* debugInt64 =
        //                context->getFunction("printi64");
        //                ArgsV.push_back(Builder->CreateSExt(val_offset16,int64Type));
        //                Builder->CreateCall(debugInt64, ArgsV);
      }
#endif

      /* Control Logic */
      pmCtr++;
      pmFieldsNo.erase(pmIt);
    }

    /* Check usefulness for query */
    vector<RecordAttribute *>::iterator it;
    bool fieldNeeded = false;
    for (it = wantedFields.begin(); it != wantedFields.end(); it++) {
      int neededAttr = (*it)->getAttrNo();
      if (i == neededAttr) {
        string attrName = (*it)->getName();
        // cout << "Parsing "<< attrName << "!!" << endl;
        RecordAttribute attr = *(*it);

        /* Codegen: Convert Field */
        typeID id = (*it)->getOriginalType()->getTypeID();
        readField(id, attr, *variableBindings);

        /* Control Logic */
        fieldNeeded = true;
        wantedFields.erase(it);
        lastFieldNo = (*it)->getAttrNo();
        break;
      }
    }
    if (!fieldNeeded) {
      if (i != rec.getArgsNo()) {
        skipDelimLLVM(delimInner, debugChar, debugInt);
        LOG(INFO) << "Skip internal field " << i;
      } else {
        skipDelimLLVM(delimEnd, debugChar, debugInt);
        LOG(INFO) << "Skip final field " << i;
      }
    }
  }

  // Make the new basic block for the increment, inserting after current block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);
  // Start insertion in IncBB.
  Builder->SetInsertPoint(IncBB);

  // Triggering parent
  OperatorState *state = new OperatorState(producer, *variableBindings);
  ::Operator *const opParent = producer.getParent();
  //    cout << "Forwarding " << (*variableBindings).size() << endl;
  opParent->consume(context, *state);
  Value *val_1 = context->createInt32(1);
  val_lineCtr = Builder->CreateAdd(val_lineCtr, val_1);
  Builder->CreateStore(val_lineCtr, mem_lineCtr);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(CondBB);

  //    Finish up with end (the AfterLoop)
  //     Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());
}
void CSVPlugin::scanPM(const ::Operator &producer) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int16Type = Type::getInt16Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  // Util.
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, this->delimInner));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, this->delimEnd));

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> *variableBindings =
      new map<RecordAttribute, ProteusValueMemory>();

  // Fetch value from symbol table
  AllocaInst *mem_pos;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(posVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + posVar);
    }
    mem_pos = it->second;
  }
  AllocaInst *fsizePtr;
  {
    map<string, AllocaInst *>::iterator it;
    it = NamedValuesCSV.find(fsizeVar);
    if (it == NamedValuesCSV.end()) {
      throw runtime_error(string("Unknown variable name: ") + fsizeVar);
    }
    fsizePtr = it->second;
  }

  /* Materialized OID */
  RecordAttribute tupleIdentifier =
      RecordAttribute(fname, activeLoop, this->getOIDType());

  ProteusValueMemory mem_posWrapper;
  mem_posWrapper.mem = mem_lineCtr;
  mem_posWrapper.isNull = context->createFalse();
  (*variableBindings)[tupleIdentifier] = mem_posWrapper;

  /**
   * LOOP BLOCKS
   */
  BasicBlock *pmScanCond, *pmScanBody, *pmScanInc, *pmScanEnd;
  context->CreateForLoop("pmScanCond", "pmScanBody", "pmScanInc", "pmScanEnd",
                         &pmScanCond, &pmScanBody, &pmScanInc, &pmScanEnd);
  context->setEndingBlock(pmScanEnd);

  Value *val_lines = context->createInt32(lines);
  // Builder->CreateBr(pmScanCond);

  /* Condition: currLine != lines */
  Builder->SetInsertPoint(pmScanCond);
  Value *val_lineCtr = Builder->CreateLoad(mem_lineCtr);
  Value *val_cond = Builder->CreateICmpSLT(val_lineCtr, val_lines);
  Builder->CreateCondBr(val_cond, pmScanBody, pmScanEnd);

  /* Body */
  Builder->SetInsertPoint(pmScanBody);

  Value *val_newlines = Builder->CreateLoad(mem_newlines);
  /* Get curr. line start */
  Value *mem_newline = Builder->CreateGEP(val_newlines, val_lineCtr);
  Value *val_newline = Builder->CreateLoad(mem_newline);
  Builder->CreateStore(val_newline, mem_pos);
  int currAttr = 0;

  /* Preparing Cache Attempt */
  /* XXX Very silly conversion */
  list<RecordAttribute *>::iterator attrIter = rec.getArgs().begin();
  list<RecordAttribute> attrList;
  RecordAttribute projTuple =
      RecordAttribute(fname, activeLoop, this->getOIDType());
  attrList.push_back(projTuple);

  for (vector<RecordAttribute *>::iterator it = wantedFields.begin();
       it != wantedFields.end(); it++) {
    attrList.push_back(*(*it));
  }

  /* Actual Work */
  for (vector<RecordAttribute *>::iterator it = wantedFields.begin();
       it != wantedFields.end(); it++) {
#ifdef DEBUGPM
    cout << "[CSV_PM: ] Need Field " << (*it)->getOriginalRelationName() << "."
         << (*it)->getAttrName() << endl;
#endif
    /* Create search key for caches  */
    bool found = false;
    {
      expressions::InputArgument arg =
          expressions::InputArgument(&rec, 0, attrList);
      const ExpressionType *fieldType = (*it)->getOriginalType();
      const RecordAttribute &thisAttr = *(*it);
      expressions::Expression *thisField =
          new expressions::RecordProjection(fieldType, &arg, thisAttr);
      CachingService &cache = CachingService::getInstance();
      CacheInfo info = cache.getCache(thisField);
      //            CacheInfo info; info.structFieldNo = -1;
      if (info.structFieldNo != -1) {
#ifdef DEBUGCACHING
        cout << "[CSV_PM: ] Field " << (*it)->getOriginalRelationName() << "."
             << (*it)->getAttrName() << " found!" << endl;
#endif
        if (!cache.getCacheIsFull(thisField)) {
#ifdef DEBUGCACHING
          cout << "...but is not useable " << endl;
#endif
        } else {
          int posInStruct = info.structFieldNo;
          //                    cout << posInStruct <<" pos. out of "<<
          //                    info.objectTypes.size() << " in total" << endl;

          /* We already got the OID -
           * No need for extra work.
           */
          if (posInStruct != 0) {
            found = true;
            Type *int32Type = Type::getInt32Ty(llvmContext);

            StructType *cacheType =
                context->ReproduceCustomStruct(info.objectTypes);
            // Value *typeSize = ConstantExpr::getSizeOf(cacheType);
            char *rawPtr = *(info.payloadPtr);
            Value *val_cacheIdx = Builder->CreateLoad(mem_lineCtr);
#ifdef DEBUG
            {
              vector<Value *> ArgsV;

              Function *debugSth = context->getFunction("printi");
              ArgsV.push_back(val_cacheIdx);
              Builder->CreateCall(debugSth, ArgsV);
            }
#endif
            /* Cast to appr. type */
            PointerType *ptr_cacheType = PointerType::get(cacheType, 0);
            Value *val_cachePtr =
                context->CastPtrToLlvmPtr(ptr_cacheType, rawPtr);

            Value *val_cacheShiftedPtr =
                context->getArrayElemMem(val_cachePtr, val_cacheIdx);
            Value *val_cachedField =
                context->getStructElem(val_cacheShiftedPtr, posInStruct);
            Type *fieldType = val_cachedField->getType();
            /* This Alloca should not appear in optimized code */
            AllocaInst *mem_cachedField =
                context->CreateEntryBlockAlloca(F, "tmpCachedField", fieldType);
            Builder->CreateStore(val_cachedField, mem_cachedField);

            ProteusValueMemory mem_valWrapper;
            mem_valWrapper.mem = mem_cachedField;
            mem_valWrapper.isNull = context->createFalse();
            RecordAttribute attr = *(*it);
            (*variableBindings)[attr] = mem_valWrapper;
#ifdef DEBUG
            {
              vector<Value *> ArgsV;

              Function *debugSth = context->getFunction("printi");
              ArgsV.push_back(val_cachedField);
              Builder->CreateCall(debugSth, ArgsV);
            }
#endif
          }
        }
      } else {
        // cout << "No match found for " << fname << endl;
      }
    }

    if (!found) {
      /* Determine whether to scan forwards or backwards */
      int neededAttr = (*it)->getAttrNo() - 1;
      int pmDistanceBefore = neededAttr % policy;
      int pmDistanceAfter = policy - neededAttr % policy;
      int distanceFromCurr = neededAttr - currAttr;

      /* Parse from current field */
      if (distanceFromCurr <= pmDistanceBefore &&
          distanceFromCurr <= pmDistanceAfter) {
#ifdef DEBUGPM
        cout << "To get field " << (*it)->getAttrNo()
             << ", scan from current pos " << distanceFromCurr << " fields"
             << endl;
#endif
        /* How many fields to skip */
        for (int i = 0; i < distanceFromCurr; i++) {
          skipDelimLLVM(delimInner);
        }
        /* Codegen: Convert Field */
        RecordAttribute attr = *(*it);
        // cout << "1. Also " << attr.getAttrName() << endl;
        typeID id = (*it)->getOriginalType()->getTypeID();
        readField(id, attr, *variableBindings);
      }
      /* Parse forwards */
      else if (pmDistanceBefore <= pmDistanceAfter) {
#ifdef DEBUGPM
        cout << "To get field " << (*it)->getAttrNo() << ", scan forward "
             << pmDistanceBefore << " fields" << endl;
#endif
        int nearbyPM = neededAttr / policy;
        //            cout << "Array Field In PM: " << nearbyPM - 1 << endl;

        /* Get offset from PM */
        Value *val_pm = Builder->CreateLoad(mem_pm);
        Value *mem_pmRow = Builder->CreateInBoundsGEP(val_pm, val_lineCtr);
        Value *val_pmRow = Builder->CreateLoad(mem_pmRow);
        Value *val_pmColIdx = context->createInt32(nearbyPM - 1);
        Value *val_pmCol = Builder->CreateInBoundsGEP(val_pmRow, val_pmColIdx);
        Value *val_pmOffset16 = Builder->CreateLoad(val_pmCol);

        Value *val_pmOffset64 = Builder->CreateSExt(val_pmOffset16, int64Type);
        Value *val_offset = Builder->CreateAdd(val_newline, val_pmOffset64);

        /* Set position to curr_new_line + pm_offset */
        Builder->CreateStore(val_offset, mem_pos);
        //#ifdef DEBUGPM
        //                {
        //                    vector<Value*> ArgsV;
        //                    ArgsV.clear();
        //                    ArgsV.push_back(val_pmOffset16);
        //                    Function* debugInt =
        //                    context->getFunction("printShort");
        //                    Builder->CreateCall(debugInt, ArgsV, "printf");
        //                }
        //#endif
        /* How many fields to skip */
        for (int i = 0; i < pmDistanceBefore; i++) {
          skipDelimLLVM(delimInner);
        }
        /* Codegen: Convert Field */
        RecordAttribute attr = *(*it);
        typeID id = (*it)->getOriginalType()->getTypeID();
        // cout << "2. Also " << attr.getAttrName() << endl;
        readField(id, attr, *variableBindings);

      }
      /* Parse backwards */
      else {
#ifdef DEBUGPM
        cout << "To get field " << (*it)->getAttrNo() << ", scan backward "
             << pmDistanceAfter << " fields" << endl;
#endif
        int nearbyPM = (neededAttr / policy) + 1;
        // cout << "Array Field in PM: " << nearbyPM - 1 << endl;

        /* Get offset from PM */
        Value *val_pm = Builder->CreateLoad(mem_pm);
        Value *mem_pmRow = Builder->CreateInBoundsGEP(val_pm, val_lineCtr);
        Value *val_pmRow = Builder->CreateLoad(mem_pmRow);
        Value *val_pmColIdx = context->createInt32(nearbyPM - 1);
        Value *val_pmCol = Builder->CreateInBoundsGEP(val_pmRow, val_pmColIdx);
        Value *val_pmOffset16 = Builder->CreateLoad(val_pmCol);

        Value *val_pmOffset64 = Builder->CreateSExt(val_pmOffset16, int64Type);
        Value *val_offset = Builder->CreateAdd(val_newline, val_pmOffset64);
#ifdef DEBUG
        {
          vector<Value *> ArgsV;
          ArgsV.clear();
          ArgsV.push_back(val_pmOffset16);
          Function *debugInt = context->getFunction("printShort");
          Builder->CreateCall(debugInt, ArgsV, "printf");
        }
#endif
        /* Set position to curr_new_line + pm_offset */
        Builder->CreateStore(val_offset, mem_pos);

        /* How many fields to skip BACKWARDS*/
        /* XXX Double Work: Need methods accepting start + end pos.
         * to exploit the fact that I found ending pos.
         * before even knowing the starting one */
        //            Value *end_pos;
        for (int i = 0; i < pmDistanceAfter; i++) {
          skipDelimBackwardsLLVM(delimInner);
          //                if(i == pmDistanceAfter - 2)    {
          //                    end_pos = Builder->CreateLoad(mem_pos);
          //                }
        }
        /* Codegen: Convert Field */
        RecordAttribute attr = *(*it);
        typeID id = (*it)->getOriginalType()->getTypeID();
        // cout << "3. Also " << attr.getAttrName() << endl;
        readField(id, attr, *variableBindings);
      }

      /* All conversion functions actually advance field counter */
      currAttr = neededAttr + 1;
    }
  }

  Builder->CreateBr(pmScanInc);

  /* Inc: callParent; lineCtr++ */
  Builder->SetInsertPoint(pmScanInc);
  //    cout << "Forwarding " << (*variableBindings).size() << endl;
  OperatorState *state = new OperatorState(producer, *variableBindings);
  ::Operator *const opParent = producer.getParent();
  opParent->consume(context, *state);
  Value *val_1 = context->createInt32(1);
  val_lineCtr = Builder->CreateLoad(mem_lineCtr);
  val_lineCtr = Builder->CreateAdd(val_lineCtr, val_1);
  Builder->CreateStore(val_lineCtr, mem_lineCtr);
  // #ifdef DEBUG
  //     {
  //         vector<Value*> ArgsV;
  //         Function* debugInt = context->getFunction("printi");
  //         ArgsV.push_back(context->createInt32(100001));
  //         Builder->CreateCall(debugInt, ArgsV);
  //     }
  // #endif
  Builder->CreateBr(pmScanCond);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(pmScanCond);

  //    Finish up with end (the AfterLoop)
  //     Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());
}
}  // namespace pm
