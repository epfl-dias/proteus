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

#include "plugins/csv-plugin.hpp"

#include "operators/operators.hpp"

using namespace llvm;

CSVPlugin::CSVPlugin(Context *const context, string &fname, RecordType &rec,
                     vector<RecordAttribute *> &whichFields)
    : fname(fname),
      rec(rec),
      wantedFields(whichFields),
      context(context),
      posVar("offset"),
      bufVar("buf"),
      fsizeVar("fileSize") {
  pos = 0;
  fd = -1;
  buf = nullptr;
  fsize = 0;

  LOG(INFO) << "[CSVPlugin: ] " << fname;
  if (whichFields.size() > 0) {
    struct stat statbuf;
    const char *name_c = fname.c_str();
    stat(name_c, &statbuf);
    fsize = statbuf.st_size;

    fd = open(name_c, O_RDONLY);
    if (fd == -1) {
      throw runtime_error(string("csv.open"));
    }
  }
}

CSVPlugin::~CSVPlugin() {}

void CSVPlugin::init() {
  buf = (char *)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd,
                     0);
  if (buf == MAP_FAILED) {
    throw runtime_error(string("csv.mmap"));
  }

  // Preparing the codegen part

  //(Can probably wrap some of these calls in one function)
  Function *F = context->getGlobalFunction();
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();

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
  Value *ptrVal = ConstantInt::get(llvmContext, APInt(64, ((uint64_t)buf)));
  // i8*
  Value *unshiftedPtr = Builder->CreateIntToPtr(ptrVal, charPtrType);
  Builder->CreateStore(unshiftedPtr, bufMem);
  NamedValuesCSV[bufVar] = bufMem;
};

void CSVPlugin::generate(const ::Operator &producer, ParallelContext *context) {
  return scanCSV(producer, context->getGlobalFunction());
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
ProteusValueMemory CSVPlugin::readPath(string activeRelation, Bindings bindings,
                                       const char *pathVar,
                                       RecordAttribute attr,
                                       ParallelContext *context) {
  const OperatorState &state = *(bindings.state);
  RecordAttribute tmpKey(fname, pathVar, this->getOIDType());
  return state[tmpKey];
}

ProteusValueMemory CSVPlugin::readValue(ProteusValueMemory mem_value,
                                        const ExpressionType *type,
                                        ParallelContext *context) {
  return mem_value;
}

ProteusValue CSVPlugin::hashValue(ProteusValueMemory mem_value,
                                  const ExpressionType *type,
                                  Context *context) {
  IRBuilder<> *Builder = context->getBuilder();
  switch (type->getTypeID()) {
    case BOOL: {
      Function *hashBoolean = context->getFunction("hashBoolean");
      vector<Value *> ArgsV;
      ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
      Value *hashResult =
          context->getBuilder()->CreateCall(hashBoolean, ArgsV, "hashBoolean");

      ProteusValue valWrapper;
      valWrapper.value = hashResult;
      valWrapper.isNull = context->createFalse();
      return valWrapper;
    }
    case STRING: {
      LOG(ERROR) << "[CSV PLUGIN: ] String datatypes not supported yet";
      throw runtime_error(
          string("[CSV PLUGIN: ] String datatypes not supported yet"));
    }
    case FLOAT: {
      Function *hashDouble = context->getFunction("hashDouble");
      vector<Value *> ArgsV;
      ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
      Value *hashResult =
          context->getBuilder()->CreateCall(hashDouble, ArgsV, "hashDouble");

      ProteusValue valWrapper;
      valWrapper.value = hashResult;
      valWrapper.isNull = context->createFalse();
      return valWrapper;
    }
    case INT: {
      Function *hashInt = context->getFunction("hashInt");
      vector<Value *> ArgsV;
      ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
      Value *hashResult =
          context->getBuilder()->CreateCall(hashInt, ArgsV, "hashInt");

      ProteusValue valWrapper;
      valWrapper.value = hashResult;
      valWrapper.isNull = context->createFalse();
      return valWrapper;
    }
    case BAG:
    case LIST:
    case SET:
      LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain collections";
      throw runtime_error(
          string("[CSV PLUGIN: ] CSV files do not contain collections"));
    case RECORD:
      LOG(ERROR)
          << "[CSV PLUGIN: ] CSV files do not contain record-valued attributes";
      throw runtime_error(string(
          "[CSV PLUGIN: ] CSV files do not contain record-valued attributes"));
    default:
      LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
      throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
  }
}

ProteusValue CSVPlugin::hashValueEager(ProteusValue valWrapper,
                                       const ExpressionType *type,
                                       Context *context) {
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();
  Value *tmp = valWrapper.value;
  AllocaInst *mem_tmp =
      context->CreateEntryBlockAlloca(F, "mem_cachedToHash", tmp->getType());
  Builder->CreateStore(tmp, mem_tmp);
  ProteusValueMemory mem_tmpWrapper = {mem_tmp, valWrapper.isNull};
  return hashValue(mem_tmpWrapper, type, context);
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
      LOG(ERROR) << "[CSV PLUGIN: ] String datatypes not supported yet";
      throw runtime_error(
          string("[CSV PLUGIN: ] String datatypes not supported yet"));
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
    case DSTRING: {
      vector<Value *> ArgsV;
      flushFunc = context->getFunction("flushDString");
      void *dict = ((DStringType *)type)->getDictionary();
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
    case RECORD:
      LOG(ERROR)
          << "[CSV PLUGIN: ] CSV files do not contain record-valued attributes";
      throw runtime_error(string(
          "[CSV PLUGIN: ] CSV files do not contain record-valued attributes"));
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

llvm::Value *CSVPlugin::getValueSize(ProteusValueMemory mem_value,
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
      string error_msg = string("[CSV Plugin]: Strings not supported yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
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

// Private Functions

void CSVPlugin::skip() {
  while (pos < fsize && buf[pos] != ';' && buf[pos] != '\n') {
    pos++;
  }
  pos++;
}

inline size_t CSVPlugin::skipDelim(size_t pos, char *buf, char delim) {
  while (buf[pos] != delim) {
    pos++;
  }
  pos++;
  return pos;
}

// Gist of the code generated:
// Output this as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:
void CSVPlugin::skipDelimLLVM(Value *delim, Function *debugChar,
                              Function *debugInt) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
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

void CSVPlugin::skipLLVM() {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, ';'));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, '\n'));

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
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, ';'));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, '\n'));

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

void CSVPlugin::readAsIntLLVM(
    RecordAttribute attName,
    map<RecordAttribute, ProteusValueMemory> &variables, Function *atoi_,
    Function *debugChar, Function *debugInt) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int32Type = Type::getInt32Ty(llvmContext);
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
  LOG(INFO) << "[READ INT: ] Atoi Successful";

  ArgsV.clear();
  ArgsV.push_back(parsedInt);

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
  Type *int32Type = Type::getInt32Ty(llvmContext);
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

  //    vector<Value*> ArgsV;
  //    ArgsV.clear();
  //    ArgsV.push_back(bufShiftedPtr);
  //    ArgsV.push_back(len_32);

  //    Function* atois = context->getFunction("atois");
  //    Value* parsedInt = Builder->CreateCall(atois, ArgsV, "atois");

  AllocaInst *mem_result = context->CreateEntryBlockAlloca(
      TheFunction, "nen_currIntResult", int32Type);
  atois(bufShiftedPtr, len_32, mem_result, context);
  LOG(INFO) << "[READ INT: ] Atoi Successful";

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_result;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
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
  Value *bufPtr = Builder->CreateLoad(buf, "bufPtr");
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, start);
  vector<Value *> ArgsV;
  ArgsV.clear();
  ArgsV.push_back(bufShiftedPtr);
  Value *parsedFloat = Builder->CreateCall(atof_, ArgsV, "atof");
  AllocaInst *currResult =
      context->CreateEntryBlockAlloca(TheFunction, "currResult", doubleType);
  Builder->CreateStore(parsedFloat, currResult);
  LOG(INFO) << "[READ FLOAT: ] Atof Successful";

#ifdef DEBUG
//    ArgsV.clear();
//    ArgsV.push_back(parsedFloat);
//    Builder->CreateCall(debugFloat, ArgsV, "printf");
#endif
  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = currResult;
  mem_valWrapper.isNull = context->createFalse();
  variables[attName] = mem_valWrapper;
}

void CSVPlugin::scanCSV(const ::Operator &producer, Function *debug) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> variableBindings;

  // Fetch value from symbol table
  auto pos = NamedValuesCSV.at(posVar);
  auto fsizePtr = NamedValuesCSV.at(fsizeVar);

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

  //    BYTECODE
  //    for.cond:                                         ; preds = %for.inc,
  //    %entry
  //      %0 = load i32* %pos.addr, align 4
  //      %1 = load i32* %fsize.addr, align 4
  //      %cmp = icmp slt i32 %0, %1
  //      br i1 %cmp, label %for.body, label %for.end

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

  // Get the starting position of each record and pass it along.
  // More general/lazy CSV plugins will only perform this action,
  // instead of eagerly converting fields
  ExpressionType *oidType = new Int64Type();
  RecordAttribute tupleIdentifier = RecordAttribute(fname, activeLoop, oidType);

  ProteusValueMemory mem_posWrapper;
  mem_posWrapper.mem = pos;
  mem_posWrapper.isNull = context->createFalse();
  variableBindings[tupleIdentifier] = mem_posWrapper;

  //    BYTECODE
  //    for.body:                                         ; preds = %for.cond
  //      br label %for.inc

  // Actual Work (Loop through attributes etc.)
  int cur_col = 0;
  Value *delimInner = ConstantInt::get(llvmContext, APInt(8, ';'));
  Value *delimEnd = ConstantInt::get(llvmContext, APInt(8, '\n'));
  int lastFieldNo = -1;
  Function *atoi_ = context->getFunction("atoi");
  Function *atof_ = context->getFunction("atof");
  Function *debugChar = context->getFunction("printc");
  Function *debugInt = context->getFunction("printi");
  Function *debugFloat = context->getFunction("printFloat");

  if (atoi_ == nullptr || atof_ == nullptr || debugChar == nullptr ||
      debugInt == nullptr || debugFloat == nullptr) {
    LOG(ERROR) << "One of the functions needed not found!";
    throw runtime_error(string("One of the functions needed not found!"));
  }

  for (vector<RecordAttribute *>::iterator it = wantedFields.begin();
       it != wantedFields.end(); it++) {
    int neededAttr = (*it)->getAttrNo() - 1;
    for (; cur_col < neededAttr; cur_col++) {
      skipDelimLLVM(delimInner, debugChar, debugInt);
    }

    string attrName = (*it)->getName();
    RecordAttribute attr = *(*it);
    switch ((*it)->getOriginalType()->getTypeID()) {
      case BOOL:
        readAsBooleanLLVM(attr, variableBindings);
        break;
      case STRING:
        LOG(ERROR) << "[CSV PLUGIN: ] String datatypes not supported yet";
        throw runtime_error(
            string("[CSV PLUGIN: ] String datatypes not supported yet"));
      case FLOAT:
        readAsFloatLLVM(attr, variableBindings, atof_, debugChar, debugFloat);
        break;
      case INT:
        readAsIntLLVM(attr, variableBindings);
        break;
      case BAG:
      case LIST:
      case SET:
        LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain collections";
        throw runtime_error(
            string("[CSV PLUGIN: ] CSV files do not contain collections"));
      case RECORD:
        LOG(ERROR) << "[CSV PLUGIN: ] CSV files do not contain record-valued "
                      "attributes";
        throw runtime_error(
            string("[CSV PLUGIN: ] CSV files do not contain record-valued "
                   "attributes"));
      default:
        LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
        throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
    }

    // Using it to know if a final skip is needed
    lastFieldNo = neededAttr + 1;
    cur_col++;
  }

  if (lastFieldNo < rec.getArgsNo()) {
    // Skip rest of line
    skipDelimLLVM(delimEnd, debugChar, debugInt);
  }

  // Make the new basic block for the increment, inserting after current block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", TheFunction);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);
  // Start insertion in IncBB.
  Builder->SetInsertPoint(IncBB);

  // Triggering parent
  OperatorState *state = new OperatorState(producer, variableBindings);
  ::Operator *const opParent = producer.getParent();
  opParent->consume(context, *state);

  //    BYTECODE
  //    for.inc:                                          ; preds = %for.body
  //      %2 = load i32* %pos.addr, align 4
  //      %inc = add nsw i32 %2, 1
  //      store i32 %inc, i32* %pos.addr, align 4
  //      br label %for.cond

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(CondBB);

  //    Finish up with end (the AfterLoop)
  //     Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());
}

int CSVPlugin::readAsInt() {
  int start = pos;
  skip();
  return std::atoi(buf + start);
}

int CSVPlugin::eof() { return (pos >= fsize); }
