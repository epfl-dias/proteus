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

#include "plugins/json-jsmn-plugin.hpp"

// Definitely not enough as a solution
//#define MAXTOKENS 1000

using namespace llvm;

namespace jsmn {

#define TOKEN_PRINT(t)                                                   \
  printf("start: %d, end: %d, type: %d, size: %d\n", (t).start, (t).end, \
         (t).type, (t).size)

#define TOKEN_STRING(js, t, s)                             \
  (strncmp(js + (t).start, s, (t).end - (t).start) == 0 && \
   strlen(s) == (size_t)((t).end - (t).start))

JSONPlugin::JSONPlugin(Context *const context, string &fname,
                       ExpressionType *schema)
    : context(context),
      fname(fname),
      schema(schema),
      var_buf("bufPtr"),
      var_tokenPtr("tokens"),
      var_tokenOffset("tokenOffset"),
      var_tokenOffsetHash("tokenOffsetHash") {
  // Memory mapping etc
  LOG(INFO) << "[JSONPlugin - jsmn: ] " << fname;
  struct stat statbuf;
  const char *name_c = fname.c_str();
  stat(name_c, &statbuf);
  fsize = statbuf.st_size;
  fd = open(name_c, O_RDONLY);
  if (fd == -1) {
    throw runtime_error(string("json.open"));
  }
  buf = (const char *)mmap(nullptr, fsize, PROT_READ | PROT_WRITE, MAP_PRIVATE,
                           fd, 0);
  if (buf == MAP_FAILED) {
    throw runtime_error(string("json.mmap"));
  }

  // Retrieving schema - not needed yet
  Catalog &catalog = Catalog::getInstance();
  catalog.registerFileJSON(fname, schema);

  // Tokenizing
  int error_code;
  jsmn_parser p;

  // Populating our json 'positional index'
  tokens = new jsmntok_t[MAXTOKENS];
  if (tokens == nullptr) {
    throw runtime_error(string("new() of tokens failed"));
  }
  for (int i = 0; i < MAXTOKENS; i++) {
    tokens[i].start = 0;
    tokens[i].end = 0;
    tokens[i].size = 0;
  }

  jsmn_init(&p);
  error_code = jsmn_parse(&p, buf, strlen(buf), &tokens, MAXTOKENS);
  if (error_code < 0) {
    string msg = "Json (JSMN) plugin failure: ";
    LOG(ERROR) << msg << error_code;
    throw runtime_error(msg);
  }

  // Preparing structures and variables for codegen part
  Function *F = context->getGlobalFunction();
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);

  // Buffer holding the JSON document
  AllocaInst *mem_buf =
      context->CreateEntryBlockAlloca(F, std::string("charPtr"), charPtrType);
  Value *val_buf_i64 =
      ConstantInt::get(llvmContext, APInt(64, ((uint64_t)buf)));
  // i8*
  Value *val_buf = Builder->CreateIntToPtr(val_buf_i64, charPtrType);
  Builder->CreateStore(val_buf, mem_buf);
  NamedValuesJSON[var_buf] = mem_buf;

  // The array of tokens
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
  AllocaInst *mem_tokenPtr = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), ptr_jsmnStructType);
  Value *ptrVal = ConstantInt::get(llvmContext, APInt(64, ((uint64_t)tokens)));
  // i8*
  Value *unshiftedPtr = Builder->CreateIntToPtr(ptrVal, ptr_jsmnStructType);
  Builder->CreateStore(unshiftedPtr, mem_tokenPtr);
  NamedValuesJSON[var_tokenPtr] = mem_tokenPtr;

  AllocaInst *mem_tokenOffset = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenOffset), int64Type);
  // 0 represents the entire JSON document
  Value *offsetVal = Builder->getInt64(1);
  Builder->CreateStore(offsetVal, mem_tokenOffset);
  NamedValuesJSON[var_tokenOffset] = mem_tokenOffset;
}

ProteusValueMemory JSONPlugin::initCollectionUnnest(
    ProteusValue val_parentTokenNo) {
  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *int64Type = Type::getInt64Ty(llvmContext);

  AllocaInst *mem_currentToken = context->CreateEntryBlockAlloca(
      F, string("currentTokenUnnested"), int64Type);
  Value *val_1 = Builder->getInt64(1);
  Value *val_currentToken = Builder->CreateAdd(val_parentTokenNo.value, val_1);
  Builder->CreateStore(val_currentToken, mem_currentToken);
#ifdef DEBUG
//    std::vector<Value*> ArgsV;
//    Function* debugInt = context->getFunction("printi64");
//
//    ArgsV.push_back(val_parentTokenNo);
//    Builder->CreateCall(debugInt, ArgsV);
//    ArgsV.clear();
//    ArgsV.push_back(val_currentToken);
//    Builder->CreateCall(debugInt, ArgsV);
#endif
  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_currentToken;
  mem_valWrapper.isNull = val_parentTokenNo.isNull;
  return mem_valWrapper;
}

/**
 * tokens[i].end <= tokens[parentToken].end && tokens[i].end != 0
 */
ProteusValue JSONPlugin::collectionHasNext(
    ProteusValue val_parentTokenNo, ProteusValueMemory mem_currentTokenNo) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int8Type = Type::getInt8Ty(llvmContext);
  llvm::Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *parentToken =
      context->getArrayElem(mem_tokens, val_parentTokenNo.value);
  Builder->CreateStore(parentToken, mem_tokens_shifted);
  Value *parent_token_end = context->getStructElem(mem_tokens_shifted, 2);

  Value *val_currentTokenNo = Builder->CreateLoad(mem_currentTokenNo.mem);
  Value *currentToken = context->getArrayElem(mem_tokens, val_currentTokenNo);
  Builder->CreateStore(currentToken, mem_tokens_shifted);
  Value *current_token_end = context->getStructElem(mem_tokens_shifted, 2);

  Value *val_0 = Builder->getInt32(0);
  Value *endCond1 = Builder->CreateICmpSLE(current_token_end, parent_token_end);
  Value *endCond2 = Builder->CreateICmpNE(current_token_end, val_0);
  Value *endCond = Builder->CreateAnd(endCond1, endCond2);
  Value *endCond_isNull = context->createFalse();

  ProteusValue valWrapper;
  valWrapper.value = endCond;
#ifdef DEBUG
  vector<Value *> ArgsV;
  ArgsV.clear();
  ArgsV.push_back(endCond);
  Function *debugBoolean = context->getFunction("printBoolean");
  Builder->CreateCall(debugBoolean, ArgsV);
#endif
  valWrapper.isNull = endCond_isNull;
  return valWrapper;
}

ProteusValueMemory JSONPlugin::collectionGetNext(
    ProteusValueMemory mem_currentToken) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Type *int64Type = Type::getInt64Ty(llvmContext);
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
  Function *F = context->getGlobalFunction();
  vector<Value *> ArgsV;
  Function *debugInt = context->getFunction("printi");
  Function *debugInt64 = context->getFunction("printi64");

  Value *currentTokenNo = Builder->CreateLoad(mem_currentToken.mem);
  /**
   * Reason for this:
   * Need to return 'i', but also need to increment it before returning
   */
  AllocaInst *mem_tokenToReturn = context->CreateEntryBlockAlloca(
      F, std::string("tokenToUnnest"), int64Type);
  Value *tokenToReturn_isNull = context->createFalse();
  Builder->CreateStore(currentTokenNo, mem_tokenToReturn);

#ifdef DEBUG
  ////Printing the active token that will be forwarded
  //    std::vector<Value*> ArgsV;
  //    ArgsV.clear();
  //    ArgsV.push_back(currentTokenNo);
  //    Function* debugInt = context->getFunction("printi64");
  //    Builder->CreateCall(debugInt, ArgsV);
#endif

  /**
   * int i_contents = i+1;
   * while(tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start
   * != 0)    { i_contents++;
   *    }
   *    i = i_contents;
   */
  BasicBlock *skipContentsCond, *skipContentsBody, *skipContentsInc,
      *skipContentsEnd;
  context->CreateForLoop("skipContentsCond", "skipContentsBody",
                         "skipContentsInc", "skipContentsEnd",
                         &skipContentsCond, &skipContentsBody, &skipContentsInc,
                         &skipContentsEnd);
  /**
   * Entry Block:
   * int i_contents = i+1;
   */
  Value *val_1 = Builder->getInt64(1);
  AllocaInst *mem_i_contents =
      context->CreateEntryBlockAlloca(F, string("i_contents"), int64Type);
  Value *val_i_contents = Builder->CreateAdd(currentTokenNo, val_1);
  Builder->CreateStore(val_i_contents, mem_i_contents);
  Builder->CreateBr(skipContentsCond);

  /**
   * tokens[i_contents].start <= tokens[i].end && tokens[i_contents].start != 0
   */
  Builder->SetInsertPoint(skipContentsCond);
  // Prepare tokens[i_contents].start
  Value *val_0 = Builder->getInt32(0);
  val_i_contents = Builder->CreateLoad(mem_i_contents);
  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_i_contents_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i_contents = context->getArrayElem(mem_tokens, val_i_contents);
  Builder->CreateStore(token_i_contents, mem_tokens_i_contents_shifted);
  Value *token_i_contents_start =
      context->getStructElem(mem_tokens_i_contents_shifted, 1);

  // Prepare tokens[i].end
  AllocaInst *mem_tokens_i_shifted = context->CreateEntryBlockAlloca(
      F, string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i = context->getArrayElem(mem_tokens, currentTokenNo);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);
  Value *token_i_end = context->getStructElem(mem_tokens_i_shifted, 2);

  // Prepare condition
  Value *endCond1 = Builder->CreateICmpSLE(token_i_contents_start, token_i_end);
  Value *endCond2 = Builder->CreateICmpNE(token_i_contents_start, val_0);
  Value *endCond = Builder->CreateAnd(endCond1, endCond2);
  BranchInst::Create(skipContentsBody, skipContentsEnd, endCond,
                     skipContentsCond);

  /**
   * BODY:
   * i_contents++;
   */
  Builder->SetInsertPoint(skipContentsBody);
  Value *val_i_contents_1 = Builder->CreateAdd(val_i_contents, val_1);
  Builder->CreateStore(val_i_contents_1, mem_i_contents);
  val_i_contents = Builder->CreateLoad(mem_i_contents);
  Builder->CreateBr(skipContentsInc);

  /**
   * INC:
   * Nothing to do
   */
  Builder->SetInsertPoint(skipContentsInc);
  Builder->CreateBr(skipContentsCond);

  /**
   * END:
   * i = i_contents;
   */
  Builder->SetInsertPoint(skipContentsEnd);
  val_i_contents = Builder->CreateLoad(mem_i_contents);
  Builder->CreateStore(val_i_contents, mem_currentToken.mem);

  ProteusValueMemory mem_wrapperVal;
  mem_wrapperVal.mem = mem_tokenToReturn;
  mem_wrapperVal.isNull = tokenToReturn_isNull;
#ifdef DEBUG
  {
    vector<Value *> ArgsV;
    ArgsV.clear();
    ArgsV.push_back(val_i_contents);
    Function *debugInt = context->getFunction("printi64");
    Builder->CreateCall(debugInt, ArgsV);
  }
#endif
  return mem_wrapperVal;
}

void JSONPlugin::scanObjects(const ::Operator &producer, Function *debug) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();

  Function *debugInt64 = context->getFunction("printi64");
  vector<Value *> ArgsV;

  // Get the entry block
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> *variableBindings =
      new map<RecordAttribute, ProteusValueMemory>();

  /**
   * Loop through results (if any)
   * for (int i = 1; tokens[i].start != 0; )
   */
  BasicBlock *jsonScanCond, *jsonScanBody, *jsonScanInc, *jsonScanEnd;
  context->CreateForLoop("jsonScanCond", "jsonScanBody", "jsonScanInc",
                         "jsonScanEnd", &jsonScanCond, &jsonScanBody,
                         &jsonScanInc, &jsonScanEnd);
  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  context->setEndingBlock(jsonScanEnd);
  /**
   * Entry Block: Simply jumping to condition part
   */
  // Builder->CreateBr(jsonScanCond);

  /**
   * Condition: tokens[i].start != 0
   */
  Builder->SetInsertPoint(jsonScanCond);

  // Prepare left-hand side
  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
  Value *val_offset = Builder->CreateLoad(mem_tokenOffset);

  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
  AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i = context->getArrayElem(mem_tokens, val_offset);
  Builder->CreateStore(token_i, mem_tokens_shifted);
  // 0: jsmntype_t type;
  // 1: int start;
  // 2: int end;
  // 3: int size;
  Value *token_i_start = context->getStructElem(mem_tokens_shifted, 1);

#ifdef DEBUG
//        std::vector<Value*> ArgsV;
//        ArgsV.clear();
//        ArgsV.push_back(token_i_start);
//        Function* debugInt = context->getFunction("printi");
//        Builder->CreateCall(debugInt, ArgsV);
#endif

  // Prepare right-hand side
  Value *rhs = context->createInt32(0);

  ICmpInst *endCond = new ICmpInst(*jsonScanCond, ICmpInst::ICMP_NE,
                                   token_i_start, rhs, "cmpJSMNEnd");

  BranchInst::Create(jsonScanBody, jsonScanEnd, endCond, jsonScanCond);

  /**
   *    BODY
   */
  Builder->SetInsertPoint(jsonScanBody);

  // Triggering parent
  RecordAttribute tupleIdentifier =
      RecordAttribute(fname, activeLoop, this->getOIDType());
  ProteusValueMemory mem_tokenWrapper;
  mem_tokenWrapper.mem = mem_tokenOffset;
  mem_tokenWrapper.isNull = context->createFalse();
  (*variableBindings)[tupleIdentifier] = mem_tokenWrapper;
#ifdef DEBUG
//    ArgsV.clear();
//    ArgsV.push_back(Builder->CreateLoad(mem_tokenOffset));
//    Builder->CreateCall(debugInt64, ArgsV);
//    Builder->CreateCall(debugInt64, ArgsV);
//    ArgsV.clear();
#endif
  OperatorState *state = new OperatorState(producer, *variableBindings);
  ::Operator *const opParent = producer.getParent();
  opParent->consume(context, *state);

  // readPath(val_offset,"b");

  skipToEnd();

  Builder->CreateBr(jsonScanInc);
  /**
   * INC:
   * Nothing to do - skipToEnd() takes care of inc
   */
  Builder->SetInsertPoint(jsonScanInc);
  Builder->CreateBr(jsonScanCond);

  // Builder->SetInsertPoint(jsonScanEnd);
  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(jsonScanCond);

  /**
   * END:
   */
  //    Finish up with end (the jsonScanEnd)
  //     Any new code will be inserted in jsonScanEnd.
  Builder->SetInsertPoint(context->getEndingBlock());

  LOG(INFO) << "[Scan - JSON: ] End of scanObjects()";
}

/**
 *  while(tokens[i].start < tokens[curr].end && tokens[i].start != 0)    i++;
 */
void JSONPlugin::skipToEnd() {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
  Value *val_offset = Builder->CreateLoad(mem_tokenOffset);

  Value *val_curr = Builder->CreateLoad(mem_tokenOffset);
  AllocaInst *mem_tokens_curr_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_curr = context->getArrayElem(mem_tokens, val_curr);
  Builder->CreateStore(token_curr, mem_tokens_curr_shifted);
  Value *token_curr_end = context->getStructElem(mem_tokens_curr_shifted, 2);

  /**
   * LOOP BLOCKS
   */
  BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
  context->CreateForLoop("jsTokenSkipCond", "jsTokenSkipBody", "jsTokenSkipInc",
                         "jsTokenSkipEnd", &tokenSkipCond, &tokenSkipBody,
                         &tokenSkipInc, &tokenSkipEnd);

  /**
   * Entry Block: Simply jumping to condition part
   */
  Builder->CreateBr(tokenSkipCond);

  /**
   * Condition: tokens[i].start < tokens[curr].end && tokens[i].start != 0
   */
  Builder->SetInsertPoint(tokenSkipCond);
  val_offset = Builder->CreateLoad(mem_tokenOffset);
  AllocaInst *mem_tokens_i_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i = context->getArrayElem(mem_tokens, val_offset);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);

  // 0: jsmntype_t type;
  // 1: int start;
  // 2: int end;
  // 3: int size;
  Value *token_i_start = context->getStructElem(mem_tokens_i_shifted, 1);
  Value *rhs = context->createInt32(0);

  Value *endCond1 = Builder->CreateICmpSLT(token_i_start, token_curr_end);
  Value *endCond2 = Builder->CreateICmpNE(token_i_start, rhs);
  Value *endCond = Builder->CreateAnd(endCond1, endCond2);

  BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

  /**
   * BODY:
   * i++
   */
  Builder->SetInsertPoint(tokenSkipBody);

  val_offset = Builder->CreateLoad(mem_tokenOffset);
  Value *val_step = Builder->getInt64(1);
  // CastInst* token_i_start64 = new SExtInst(token_i_start, int64Type, "i_64",
  // tokenSkipBody); cout<<int64_conv->getType()->getTypeID()<< " vs " <<
  // val_step->getType()->getTypeID();

  // Builder->CreateCall(debugInt, ArgsV, "printi");
  Value *token_i_inc = Builder->CreateAdd(val_offset, val_step, "i_inc");
  Builder->CreateStore(token_i_inc, mem_tokenOffset);

#ifdef DEBUG
  // std::vector<Value*> ArgsV;
  // ArgsV.clear();
  // ArgsV.push_back(token_i_start);
  // Function* debugInt = context->getFunction("printi");
  // Builder->CreateCall(debugInt, ArgsV);
#endif

  Builder->CreateBr(tokenSkipInc);

  /**
   * INC:
   * Nothing to do
   * (in principle, job done in body could be done here)
   */
  Builder->SetInsertPoint(tokenSkipInc);
  Builder->CreateBr(tokenSkipCond);

  /**
   * END:
   */
  Builder->SetInsertPoint(tokenSkipEnd);
  LOG(INFO) << "[Scan - JSON: ] End of skiptoEnd()";
}

ProteusValueMemory JSONPlugin::readPath(string activeRelation,
                                        Bindings wrappedBindings,
                                        const char *path,
                                        RecordAttribute attr) {
  /**
   * FIXME Add an extra (generated) check here
   * Only objects are relevant to path expressions
   * These types of validation should be applied as high as possible
   * Still, probably unavoidable here
   *    if(tokens[parentTokenNo].type != JSMN_OBJECT)    {
   *        string msg = string("[JSON Plugin - jsmn: ]: Path traversal is only
   * applicable to objects"); LOG(ERROR) << msg; throw runtime_error(msg);
   *    }
   */

  const OperatorState &state = *(wrappedBindings.state);
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
  vector<Value *> argsV;

  // Get relevant token number
  ExpressionType *oidType = new IntType();
  RecordAttribute tupleIdentifier =
      RecordAttribute(activeRelation, activeLoop, oidType);
  //    RecordAttribute tupleIdentifier = RecordAttribute(fname,activeLoop);
  const map<RecordAttribute, ProteusValueMemory> &bindings =
      state.getBindings();
  map<RecordAttribute, ProteusValueMemory>::const_iterator it =
      bindings.find(tupleIdentifier);
  if (it == bindings.end()) {
    string error_msg = "[JSONPlugin - jsmn: ] Current tuple binding not found";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory mem_parentTokenNo = it->second;
  Value *parentTokenNo =
      Builder->CreateLoad(mem_parentTokenNo.mem, "parentTokenNo");

  // Preparing default return value (i.e., path not found)
  AllocaInst *mem_return =
      context->CreateEntryBlockAlloca(F, std::string("pathReturn"), int64Type);
  Value *minus_1 = context->createInt64(-1);
  Builder->CreateStore(minus_1, mem_return);

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_parent_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_parent = context->getArrayElem(mem_tokens, parentTokenNo);
  Builder->CreateStore(token_parent, mem_tokens_parent_shifted);
  Value *token_parent_end =
      context->getStructElem(mem_tokens_parent_shifted, 2);

  /**
   * LOOP BLOCKS
   */
  BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
  context->CreateForLoop("path_tokenSkipCond", "path_tokenSkipBody",
                         "path_tokenSkipInc", "path_tokenSkipEnd",
                         &tokenSkipCond, &tokenSkipBody, &tokenSkipInc,
                         &tokenSkipEnd);

  /**
   * Entry Block:
   */

  Value *val_1 = Builder->getInt64(1);
  Value *val_i = Builder->CreateAdd(parentTokenNo, val_1);
  AllocaInst *mem_i =
      context->CreateEntryBlockAlloca(F, std::string("tmp_i"), int64Type);
  Builder->CreateStore(val_i, mem_i);
  Builder->CreateBr(tokenSkipCond);

  /**
   * tokens[i].end <= tokens[parentToken].end
   */

  Builder->SetInsertPoint(tokenSkipCond);
  val_i = Builder->CreateLoad(mem_i);
  AllocaInst *mem_tokens_i_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i = context->getArrayElem(mem_tokens, val_i);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);

  // 0: jsmntype_t type;
  // 1: int start;
  // 2: int end;
  // 3: int size;
  Value *token_i_end = context->getStructElem(mem_tokens_i_shifted, 2);
  Value *endCond = Builder->CreateICmpSLE(token_i_end, token_parent_end);
  BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

  /**
   * BODY:
   */
  Builder->SetInsertPoint(tokenSkipBody);

  /**
   * IF-ELSE inside body:
   * if(TOKEN_STRING(buf,tokens[i],key.c_str()))
   */
  BasicBlock *ifBlock, *elseBlock;
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifTokenEq",
                              "elseTokenEq", &ifBlock, &elseBlock,
                              tokenSkipInc);

  Value *token_i_start = context->getStructElem(mem_tokens_i_shifted, 1);

  int len = strlen(path) + 1;
  char *pathCopy = (char *)malloc(len * sizeof(char));
  strcpy(pathCopy, path);
  pathCopy[len - 1] = '\0';
  Value *globalStr = context->CreateGlobalString(pathCopy);
  Value *buf = Builder->CreateLoad(NamedValuesJSON[var_buf]);
  // Preparing custom 'strcmp'
  //    vector<Value*> argsV;
  argsV.push_back(buf);
  argsV.push_back(token_i_start);
  argsV.push_back(token_i_end);
  argsV.push_back(globalStr);
  Function *tokenCmp = context->getFunction("compareTokenString");
  Value *tokenEq = Builder->CreateCall(tokenCmp, argsV);
  Value *rhs = context->createInt32(1);
  Value *cond = Builder->CreateICmpEQ(tokenEq, rhs);
  Builder->CreateCondBr(cond, ifBlock, elseBlock);

  /**
   * IF BLOCK
   * TOKEN_PRINT(tokens[i+1]);
   */
  Builder->SetInsertPoint(ifBlock);

  Value *val_i_1 = Builder->CreateAdd(val_i, val_1);
  AllocaInst *mem_tokens_i_1_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i_1 = context->getArrayElem(mem_tokens, val_i_1);
  Builder->CreateStore(token_i_1, mem_tokens_i_1_shifted);
  Value *token_i_1_start = context->getStructElem(mem_tokens_i_1_shifted, 1);

  // Storing return value (i+1)
  Builder->CreateStore(val_i_1, mem_return);

#ifdef DEBUG
  //    Function* debugInt = context->getFunction("printi");
  //    argsV.clear();
  //    Value *tmp = context->createInt32(100);
  //    argsV.push_back(tmp);
  //    Builder->CreateCall(debugInt, argsV);
  //
  //    argsV.clear();
  //    argsV.push_back(token_i_1_start);
  //    Builder->CreateCall(debugInt, argsV);
#endif

  Builder->CreateBr(tokenSkipEnd);

  /**
   * ELSE BLOCK
   */
  Builder->SetInsertPoint(elseBlock);
  Builder->CreateBr(tokenSkipInc);

  /**
   * (Back to LOOP)
   * INC:
   * i += 2
   */
  Builder->SetInsertPoint(tokenSkipInc);
  val_i = Builder->CreateLoad(mem_i);
  Value *val_2 = Builder->getInt64(2);
  Value *val_i_2 = Builder->CreateAdd(val_i, val_2);
  Builder->CreateStore(val_i_2, mem_i);

  //    argsV.clear();
  //    argsV.push_back(val_i_2);
  //    Function* debugInt64 = context->getFunction("printi64");
  //    Builder->CreateCall(debugInt64, argsV);

  token_i = context->getArrayElem(mem_tokens, val_i_2);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);

  Builder->CreateBr(tokenSkipCond);

  /**
   * END:
   */
  Builder->SetInsertPoint(tokenSkipEnd);
  LOG(INFO) << "[Scan - JSON: ] End of readPath()";

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_return;
  mem_valWrapper.isNull = context->createFalse();
  return mem_valWrapper;
}

ProteusValueMemory JSONPlugin::readPathInternal(
    ProteusValueMemory mem_parentTokenNo, const char *path) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
  vector<Value *> argsV;
  Function *debugInt = context->getFunction("printi64");
  Function *debugInt32 = context->getFunction("printi");

  Value *parentTokenNo = Builder->CreateLoad(mem_parentTokenNo.mem);

  // Preparing default return value (i.e., path not found)
  AllocaInst *mem_return =
      context->CreateEntryBlockAlloca(F, string("pathReturn"), int64Type);
  Value *minus_1 = context->createInt64(-1);
  Builder->CreateStore(minus_1, mem_return);

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_parent_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_parent = context->getArrayElem(mem_tokens, parentTokenNo);
  Builder->CreateStore(token_parent, mem_tokens_parent_shifted);
  Value *token_parent_end =
      context->getStructElem(mem_tokens_parent_shifted, 2);

  /**
   * LOOP BLOCKS
   */
  BasicBlock *tokenSkipCond, *tokenSkipBody, *tokenSkipInc, *tokenSkipEnd;
  context->CreateForLoop(
      "path_tokenSkipCondInternal", "path_tokenSkipBodyInternal",
      "path_tokenSkipIncInternal", "path_tokenSkipEndInternal", &tokenSkipCond,
      &tokenSkipBody, &tokenSkipInc, &tokenSkipEnd);

  /**
   * Entry Block:
   */

  Value *val_1 = Builder->getInt64(1);
  Value *val_i = Builder->CreateAdd(parentTokenNo, val_1);
  AllocaInst *mem_i =
      context->CreateEntryBlockAlloca(F, std::string("tmp_i"), int64Type);
  Builder->CreateStore(val_i, mem_i);
  Builder->CreateBr(tokenSkipCond);

  /**
   * tokens[i].end <= tokens[parentToken].end
   */

  Builder->SetInsertPoint(tokenSkipCond);
  val_i = Builder->CreateLoad(mem_i);
  AllocaInst *mem_tokens_i_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i = context->getArrayElem(mem_tokens, val_i);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);

  // 0: jsmntype_t type;
  // 1: int start;
  // 2: int end;
  // 3: int size;
  Value *token_i_end = context->getStructElem(mem_tokens_i_shifted, 2);
#ifdef DEBUG
//    argsV.clear();
//    argsV.push_back(token_i_end);
//    Builder->CreateCall(debugInt32, argsV);
//    argsV.clear();
//
//    argsV.push_back(token_parent_end);
//    Builder->CreateCall(debugInt32, argsV);
//    argsV.clear();
#endif
  Value *endCond = Builder->CreateICmpSLE(token_i_end, token_parent_end);
  BranchInst::Create(tokenSkipBody, tokenSkipEnd, endCond, tokenSkipCond);

  /**
   * BODY:
   */
  Builder->SetInsertPoint(tokenSkipBody);
#ifdef DEBUG
//    argsV.clear();
//    argsV.push_back(context->createInt64(666));
//    Builder->CreateCall(debugInt, argsV);
//    argsV.clear();
#endif

  /**
   * IF-ELSE inside body:
   * if(TOKEN_STRING(buf,tokens[i],key.c_str()))
   */
  BasicBlock *ifBlock, *elseBlock;
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifTokenEq",
                              "elseTokenEq", &ifBlock, &elseBlock,
                              tokenSkipInc);

  Value *token_i_start = context->getStructElem(mem_tokens_i_shifted, 1);

  int len = strlen(path) + 1;
  char *pathCopy = (char *)malloc(len * sizeof(char));
  strcpy(pathCopy, path);
  pathCopy[len] = '\0';
  Value *globalStr = context->CreateGlobalString(pathCopy);
  Value *buf = Builder->CreateLoad(NamedValuesJSON[var_buf]);
  // Preparing custom 'strcmp'
  argsV.push_back(buf);
  argsV.push_back(token_i_start);
  argsV.push_back(token_i_end);
  argsV.push_back(globalStr);
  Function *tokenCmp = context->getFunction("compareTokenString");
  Value *tokenEq = Builder->CreateCall(tokenCmp, argsV);
  Value *rhs = context->createInt32(1);
  Value *cond = Builder->CreateICmpEQ(tokenEq, rhs);
  Builder->CreateCondBr(cond, ifBlock, elseBlock);

  /**
   * IF BLOCK
   * TOKEN_PRINT(tokens[i+1]);
   */
  Builder->SetInsertPoint(ifBlock);

  Value *val_i_1 = Builder->CreateAdd(val_i, val_1);
  AllocaInst *mem_tokens_i_1_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token_i_1 = context->getArrayElem(mem_tokens, val_i_1);
  Builder->CreateStore(token_i_1, mem_tokens_i_1_shifted);
  Value *token_i_1_start = context->getStructElem(mem_tokens_i_1_shifted, 1);

  // Storing return value (i+1)
  Builder->CreateStore(val_i_1, mem_return);

  Builder->CreateBr(tokenSkipEnd);

  /**
   * ELSE BLOCK
   */
  Builder->SetInsertPoint(elseBlock);
  Builder->CreateBr(tokenSkipInc);

  /**
   * (Back to LOOP)
   * INC:
   * i += 2
   */
  Builder->SetInsertPoint(tokenSkipInc);
  val_i = Builder->CreateLoad(mem_i);
  Value *val_2 = Builder->getInt64(2);
  Value *val_i_2 = Builder->CreateAdd(val_i, val_2);
  Builder->CreateStore(val_i_2, mem_i);

  token_i = context->getArrayElem(mem_tokens, val_i_2);
  Builder->CreateStore(token_i, mem_tokens_i_shifted);

  Builder->CreateBr(tokenSkipCond);

  /**
   * END:
   */
  Builder->SetInsertPoint(tokenSkipEnd);
  LOG(INFO) << "[Scan - JSON: ] End of readPathInternal()";

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_return;
  mem_valWrapper.isNull = context->createFalse();
  return mem_valWrapper;
}

ProteusValueMemory JSONPlugin::readValue(ProteusValueMemory mem_value,
                                         const ExpressionType *type) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int8Type = Type::getInt8Ty(llvmContext);
  Type *int1Type = Type::getInt1Ty(llvmContext);
  llvm::Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();

  vector<Value *> ArgsV;
  Value *tokenNo = Builder->CreateLoad(mem_value.mem);

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
      F, string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token = context->getArrayElem(mem_tokens, tokenNo);
  Builder->CreateStore(token, mem_tokens_shifted);
  Value *token_start = context->getStructElem(mem_tokens_shifted, 1);
  Value *token_end = context->getStructElem(mem_tokens_shifted, 2);

  Value *bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, token_start);

  Function *conversionFunc = nullptr;

  AllocaInst *mem_convertedValue = nullptr;
  AllocaInst *mem_convertedValue_isNull = nullptr;
  Value *convertedValue = nullptr;

  mem_convertedValue_isNull =
      context->CreateEntryBlockAlloca(F, string("value_isNull"), int1Type);
  Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
  string error_msg;
  switch (type->getTypeID()) {
      //    case STRING:    cout << "Dealing with (and initializing) String" <<
      //    endl;
    case RECORD:
    case LIST: {
      mem_convertedValue = context->CreateEntryBlockAlloca(
          F, string("existingObject"), (mem_value.mem)->getAllocatedType());
      break;
    }
    case SET: {
      error_msg = string("[JSON Plugin - jsmn: ]: SET datatype cannot occur");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }
    case BAG: {
      error_msg = string("[JSON Plugin - jsmn: ]: BAG datatype cannot occur");
      LOG(ERROR) << error_msg;
    }
      throw runtime_error(string(error_msg));
    case BOOL: {
      mem_convertedValue =
          context->CreateEntryBlockAlloca(F, string("convertedBool"), int8Type);
      break;
    }
    case FLOAT: {
      mem_convertedValue = context->CreateEntryBlockAlloca(
          F, string("convertedFloat"), doubleType);
      break;
    }
    case INT: {
      mem_convertedValue =
          context->CreateEntryBlockAlloca(F, string("convertedInt"), int32Type);
#ifdef DEBUGJSMN
//        {
//        vector<Value*> ArgsV;
//        Function* debugInt = context->getFunction("printi");
//        ArgsV.push_back(Builder->CreateLoad(mem_convertedValue));
//        Builder->CreateCall(debugInt, ArgsV);
//        ArgsV.clear();
//        Value *tmp = context->createInt32(1001);
//        ArgsV.push_back(tmp);
//        Builder->CreateCall(debugInt, ArgsV);
//        }
#endif
      break;
    }
    case STRING: {
      StructType *stringObjType = context->CreateStringStruct();
      mem_convertedValue = context->CreateEntryBlockAlloca(
          F, string("ingestedString"), stringObjType);
      break;
    }
    default: {
      error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }
  }

  /**
   * Return (nil) for cases path was not found
   */
  BasicBlock *ifBlock, *elseBlock, *endBlock;
  endBlock = BasicBlock::Create(llvmContext, "afterReadValue", F);
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifPath",
                              "elsePathNullEq", &ifBlock, &elseBlock, endBlock);
  Value *minus_1 = context->createInt64(-1);
  Value *cond = Builder->CreateICmpNE(tokenNo, minus_1);
  Builder->CreateCondBr(cond, ifBlock, elseBlock);

  /**
   * IF BLOCK (tokenNo != -1)
   */
  Builder->SetInsertPoint(ifBlock);

  switch (type->getTypeID()) {
      //    case STRING: cout << "Reading String" << endl;
    case RECORD:
      // Object
    case LIST: {
      // Array
      Builder->CreateStore(tokenNo, mem_convertedValue);
      break;
    }
    case BOOL: {
      ArgsV.push_back(bufPtr);
      ArgsV.push_back(token_start);
      ArgsV.push_back(token_end);
      conversionFunc = context->getFunction("convertBoolean");
      convertedValue =
          Builder->CreateCall(conversionFunc, ArgsV, "convertBoolean");
      Builder->CreateStore(convertedValue, mem_convertedValue);
      Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
      break;
    }
    case FLOAT: {
      conversionFunc = context->getFunction("atof");
      ArgsV.push_back(bufShiftedPtr);
      convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atof");
      Builder->CreateStore(convertedValue, mem_convertedValue);
      Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
      break;
    }
    case INT: {
      conversionFunc = context->getFunction("atoi");
      ArgsV.push_back(bufShiftedPtr);
      convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atoi");
      Builder->CreateStore(convertedValue, mem_convertedValue);
      Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
#ifdef DEBUGJSMN
//        {
//        vector<Value*> ArgsV;
//        Function* debugInt = context->getFunction("printi");
//        ArgsV.push_back(convertedValue);
//        Builder->CreateCall(debugInt, ArgsV);
//        ArgsV.clear();
//        Value *tmp = context->createInt32(1002);
//        ArgsV.push_back(tmp);
//        Builder->CreateCall(debugInt, ArgsV);
//        }
#endif
      break;
    }
    case STRING: {
      Value *val_token = context->getArrayElem(mem_tokens, tokenNo);
      AllocaInst *mem_token = context->CreateEntryBlockAlloca(
          F, string(var_tokenPtr), context->CreateJSMNStruct());
      Builder->CreateStore(val_token, mem_token);

      Value *token_i_start = context->getStructElem(mem_token, 1);
      Value *token_i_end = context->getStructElem(mem_token, 2);
      Value *val_length = Builder->CreateSub(token_i_end, token_i_start);
      Value *val_length64 = Builder->CreateSExt(val_length, int64Type);
      /* Start and end offsets have taken care of op. and closing brackets*/

      /* Allocate Memory */
      //        Function *func_getMemory =
      //        context->getFunction("getMemoryChunk"); vector<Value*> ArgsV;
      //        ArgsV.push_back(val_length64);
      //        Value *convertedValue =
      //        Builder->CreateCall(func_getMemory,ArgsV);
      //        context->CodegenMemcpy(convertedValue,bufShiftedPtr,val_length);
      //        Builder->CreateStore(convertedValue, mem_convertedValue);

      StructType *stringObjType = context->CreateStringStruct();
      Value *mem_str = context->getStructElemMem(mem_convertedValue, 0);
      Builder->CreateStore(bufShiftedPtr, mem_str);
      // context->CodegenMemcpy(mem_str,bufShiftedPtr,val_length);
      Value *mem_len = context->getStructElemMem(mem_convertedValue, 1);
      Builder->CreateStore(val_length, mem_len);

      Builder->CreateStore(context->createFalse(), mem_convertedValue_isNull);
      break;
    }
    default: {
      error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }
  }
  Builder->CreateBr(endBlock);

  /**
   * ELSE BLOCK
   * return "(nullptr)"
   */
  Builder->SetInsertPoint(elseBlock);
#ifdef DEBUG  // Invalid / nullptr!
//    ArgsV.clear();
//    Function* debugInt = context->getFunction("printi");
//    Value *tmp = context->createInt32(-111);
//    ArgsV.push_back(tmp);
//    Builder->CreateCall(debugInt, ArgsV);
#endif
  Value *undefValue =
      Constant::getNullValue(mem_convertedValue->getAllocatedType());
  Builder->CreateStore(undefValue, mem_convertedValue);
  Builder->CreateStore(context->createTrue(), mem_convertedValue_isNull);
  Builder->CreateBr(endBlock);

  Builder->SetInsertPoint(endBlock);

  ProteusValueMemory mem_valWrapper;
  mem_valWrapper.mem = mem_convertedValue;
  mem_valWrapper.isNull = Builder->CreateLoad(mem_convertedValue_isNull);
#ifdef DEBUG  // Invalid / nullptr!
//    ArgsV.clear();
//    Function* debugBoolean = context->getFunction("printBoolean");
//    ArgsV.push_back(mem_valWrapper.isNull);
//    Builder->CreateCall(debugBoolean, ArgsV);
#endif
  return mem_valWrapper;
}

ProteusValue JSONPlugin::hashValue(ProteusValueMemory mem_value,
                                   const ExpressionType *type) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int8Type = Type::getInt8Ty(llvmContext);
  Type *int1Type = Type::getInt1Ty(llvmContext);
  llvm::Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();

  Function *debugInt64 = context->getFunction("printi64");
  Function *debugInt = context->getFunction("printi");
  vector<Value *> ArgsV;
  Value *tokenNo = Builder->CreateLoad(mem_value.mem);

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
      F, string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token = context->getArrayElem(mem_tokens, tokenNo);
  Builder->CreateStore(token, mem_tokens_shifted);
  Value *token_start = context->getStructElem(mem_tokens_shifted, 1);
  Value *token_end = context->getStructElem(mem_tokens_shifted, 2);

  Value *bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);
  Value *bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, token_start);

  Function *conversionFunc = nullptr;
  Function *hashFunc = nullptr;
  AllocaInst *mem_hashedValue = nullptr;
  AllocaInst *mem_hashedValue_isNull = nullptr;
  Value *hashedValue = nullptr;
  Value *convertedValue = nullptr;

  // Preparing hasher state
  NamedValuesJSON[var_tokenOffsetHash] = mem_value.mem;
  mem_hashedValue =
      context->CreateEntryBlockAlloca(F, std::string("hashValue"), int64Type);
  mem_hashedValue_isNull = context->CreateEntryBlockAlloca(
      F, std::string("hashvalue_isNull"), int1Type);
  Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
  Builder->CreateStore(context->createInt64(0), mem_hashedValue);
  string error_msg;

  /**
   * Return (nil) for cases path was not found
   */
  BasicBlock *ifBlock, *elseBlock, *endBlock;
  endBlock = BasicBlock::Create(llvmContext, "afterReadValueHash", F);
  context->CreateIfElseBlocks(context->getGlobalFunction(), "ifPathHash",
                              "elsePathNullEqHash", &ifBlock, &elseBlock,
                              endBlock);
  Value *minus_1 = context->createInt64(-1);
  Value *cond = Builder->CreateICmpNE(tokenNo, minus_1);

  Builder->CreateCondBr(cond, ifBlock, elseBlock);

  /**
   * IF BLOCK (tokenNo != -1)
   */
  Builder->SetInsertPoint(ifBlock);
#ifdef DEBUG
//    ArgsV.clear();
//    ArgsV.push_back(context->createInt32(-34));
//    Builder->CreateCall(debugInt, ArgsV);
//    ArgsV.clear();
//    ArgsV.push_back(tokenNo);
//    Builder->CreateCall(debugInt64, ArgsV);
//    ArgsV.clear();
#endif
  switch (type->getTypeID()) {
    case STRING: {
      hashFunc = context->getFunction("hashStringC");
      ArgsV.clear();
      ArgsV.push_back(bufPtr);
      // FIXME tmp! - must make token_start and token_end datatype size_t
      Type *int64Type = Type::getInt64Ty(llvmContext);
      Value *token_start64 = Builder->CreateSExt(token_start, int64Type);
      Value *token_end64 = Builder->CreateSExt(token_end, int64Type);
      ArgsV.push_back(token_start64);
      ArgsV.push_back(token_end64);

      hashedValue = Builder->CreateCall(hashFunc, ArgsV, "hashStringC");
      Builder->CreateStore(hashedValue, mem_hashedValue);
      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

      Value *plus_1 = context->createInt64(1);
      Value *tokenNo_inc = Builder->CreateAdd(tokenNo, plus_1);
      Builder->CreateStore(tokenNo_inc, NamedValuesJSON[var_tokenOffsetHash]);
      break;
    }
    case RECORD: {
      // Object
      AllocaInst *mem_seed = context->CreateEntryBlockAlloca(
          F, std::string("hashSeed"), int64Type);
      Builder->CreateStore(context->createInt64(0), mem_seed);

      ProteusValueMemory recordElem;
      recordElem.mem = mem_value.mem;
      recordElem.isNull = context->createFalse();

      Function *hashCombine = context->getFunction("combineHashes");
      hashedValue = context->createInt64(0);

      list<RecordAttribute *> args = ((RecordType *)type)->getArgs();
      list<RecordAttribute *>::iterator it = args.begin();

      // Not efficient -> duplicate work performed since fields are not visited
      // incrementally BUT: Attributes might not be in sequence, so they need to
      // be visited in a generic way
      for (; it != args.end(); it++) {
        RecordAttribute *attr = *it;
        ProteusValueMemory mem_path =
            readPathInternal(recordElem, attr->getAttrName().c_str());
        NamedValuesJSON[var_tokenOffsetHash] = mem_path.mem;

        // CAREFUL: It's generated code that has to be stitched
        hashedValue = Builder->CreateLoad(mem_hashedValue);
        ProteusValue partialHash = hashValue(mem_path, attr->getOriginalType());
        ArgsV.clear();
        ArgsV.push_back(hashedValue);
        ArgsV.push_back(partialHash.value);
        // XXX Why loading now?
        // hashedValue = Builder->CreateLoad(mem_hashedValue);
        hashedValue =
            Builder->CreateCall(hashCombine, ArgsV, "combineHashesResJSON");
        Builder->CreateStore(hashedValue, mem_hashedValue);
      }

      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
      break;
    }
    case LIST: {
      // Array
      const ExpressionType &nestedType =
          ((CollectionType *)type)->getNestedType();
      AllocaInst *mem_seed = context->CreateEntryBlockAlloca(
          F, std::string("hashSeed"), int64Type);
      Builder->CreateStore(context->createInt64(0), mem_seed);

      // Initializing: i = tokenNo + 1;
      AllocaInst *mem_tokenCnt = context->CreateEntryBlockAlloca(
          F, std::string("tokenCnt"), int64Type);
      Value *val_tokenCnt =
          Builder->CreateAdd(context->createInt64(1), tokenNo);
      Builder->CreateStore(val_tokenCnt, mem_tokenCnt);

      // while (tokens[i].start < tokens[tokenNo].end && tokens[i].start != 0)
      Function *hashCombine = context->getFunction("combineHashes");
      hashedValue = context->createInt64(0);
      Builder->CreateStore(hashedValue, mem_hashedValue);
      BasicBlock *unrollCond, *unrollBody, *unrollInc, *unrollEnd;
      context->CreateForLoop("hashListCond", "hashListBody", "hashListInc",
                             "hashListEnd", &unrollCond, &unrollBody,
                             &unrollInc, &unrollEnd);

      Builder->CreateBr(unrollCond);
      Builder->SetInsertPoint(unrollCond);
      AllocaInst *mem_tokens_i_shifted = context->CreateEntryBlockAlloca(
          F, string(var_tokenPtr), context->CreateJSMNStruct());
      val_tokenCnt = Builder->CreateLoad(mem_tokenCnt);
      Value *token_i = context->getArrayElem(mem_tokens, val_tokenCnt);
      Builder->CreateStore(token_i, mem_tokens_i_shifted);

      // 0: jsmntype_t type;
      // 1: int start;
      // 2: int end;
      // 3: int size;
      Value *token_i_start = context->getStructElem(mem_tokens_i_shifted, 1);
      Value *rhs = context->createInt32(0);

      Value *endCond1 = Builder->CreateICmpSLT(token_i_start, token_end);
      Value *endCond2 = Builder->CreateICmpNE(token_i_start, rhs);
      Value *endCond = Builder->CreateAnd(endCond1, endCond2);

      Builder->CreateCondBr(endCond, unrollBody, unrollEnd);
      /**
       * BODY:
       * readValueEagerInterpreted(i, &nestedType);
       * i++
       */
      Builder->SetInsertPoint(unrollBody);

      ProteusValueMemory listElem;
      listElem.mem = mem_tokenCnt;
      listElem.isNull = context->createFalse();

      // CAREFUL: It's generated code that has to be stitched
      // XXX in the general case, nested type may vary between elements..
      ProteusValue partialHash = hashValue(listElem, &nestedType);

      ArgsV.clear();
      ArgsV.push_back(hashedValue);
      ArgsV.push_back(partialHash.value);
      hashedValue = Builder->CreateLoad(mem_hashedValue);
      hashedValue =
          Builder->CreateCall(hashCombine, ArgsV, "combineHashesResJSON");
      Builder->CreateStore(hashedValue, mem_hashedValue);

      // new offset due to recursive application of hashedValue
      Value *newTokenOffset =
          Builder->CreateLoad(NamedValuesJSON[var_tokenOffsetHash]);
      Builder->CreateStore(newTokenOffset, mem_tokenCnt);

      // Performing just a +1 addition would cause a lot of extra work
      // and defeat the purpose of recursion
      //        val_tokenCnt =
      //        Builder->CreateAdd(Builder->CreateLoad(mem_tokenCnt),
      //                context->createInt64(1));
      //        Builder->CreateStore(val_tokenCnt, mem_tokenCnt);

      Builder->CreateBr(unrollInc);

      /**
       * INC:
       * Nothing to do
       * (in principle, job done in body could be done here)
       */
      Builder->SetInsertPoint(unrollInc);
      Builder->CreateBr(unrollCond);

      /**
       * END:
       */
      Builder->SetInsertPoint(unrollEnd);

      // Not needed to store again -> body took care of it
      // Builder->CreateStore(hashedValue, mem_hashedValue);
      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
      break;
    }
    case BOOL: {
      conversionFunc = context->getFunction("convertBoolean");
      ArgsV.push_back(bufPtr);
      ArgsV.push_back(token_start);
      ArgsV.push_back(token_end);
      convertedValue =
          Builder->CreateCall(conversionFunc, ArgsV, "convertBoolean");

      hashFunc = context->getFunction("hashBoolean");
      ArgsV.clear();
      ArgsV.push_back(convertedValue);
      hashedValue = Builder->CreateCall(conversionFunc, ArgsV, "hashBoolean");
      Builder->CreateStore(hashedValue, mem_hashedValue);
      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

      Value *plus_1 = context->createInt64(1);
      Value *tokenNo_inc = Builder->CreateAdd(tokenNo, plus_1);
      Builder->CreateStore(tokenNo_inc, NamedValuesJSON[var_tokenOffsetHash]);
      break;
    }
    case FLOAT: {
      conversionFunc = context->getFunction("atof");
      ArgsV.push_back(bufShiftedPtr);
      convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atof");

      hashFunc = context->getFunction("hashDouble");
      ArgsV.clear();
      ArgsV.push_back(convertedValue);
      hashedValue = Builder->CreateCall(conversionFunc, ArgsV, "hashDouble");
      Builder->CreateStore(hashedValue, mem_hashedValue);
      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

      Value *plus_1 = context->createInt64(1);
      Value *tokenNo_inc = Builder->CreateAdd(tokenNo, plus_1);
      Builder->CreateStore(tokenNo_inc, NamedValuesJSON[var_tokenOffsetHash]);
      break;
    }
    case INT: {
      conversionFunc = context->getFunction("atoi");
      ArgsV.push_back(bufShiftedPtr);
      convertedValue = Builder->CreateCall(conversionFunc, ArgsV, "atoi");
#ifdef DEBUG
      //            ArgsV.clear();
      //            ArgsV.push_back(context->createInt32(-60));
      //            Builder->CreateCall(debugInt, ArgsV);
      //            ArgsV.clear();
      //            ArgsV.push_back(convertedValue);
      //            Builder->CreateCall(debugInt, ArgsV);
      //            ArgsV.clear();
#endif
      hashFunc = context->getFunction("hashInt");
      ArgsV.clear();
      ArgsV.push_back(convertedValue);
      hashedValue = Builder->CreateCall(hashFunc, ArgsV, "hashInt");
      Builder->CreateStore(hashedValue, mem_hashedValue);
      Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);

      Value *plus_1 = context->createInt64(1);
      Value *tokenNo_inc = Builder->CreateAdd(tokenNo, plus_1);
      Builder->CreateStore(tokenNo_inc, NamedValuesJSON[var_tokenOffsetHash]);
      break;
    }
    default: {
      error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }
  }
  Builder->CreateBr(endBlock);

  /**
   * ELSE BLOCK
   * "nullptr" case
   * TODO What should the behavior be in this case?
   *
   * c-p from Sybase manual: If the grouping column contains a null value,
   * that row becomes its own group in the results.
   * If the grouping column contains more than one null value,
   * the null values form a single group.
   */
  Builder->SetInsertPoint(elseBlock);

  Value *undefValue =
      Constant::getNullValue(mem_hashedValue->getAllocatedType());
  Builder->CreateStore(undefValue, mem_hashedValue);
  Builder->CreateStore(context->createTrue(), mem_hashedValue_isNull);
  Builder->CreateBr(endBlock);

  Builder->SetInsertPoint(endBlock);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateLoad(mem_hashedValue);
  valWrapper.isNull = Builder->CreateLoad(mem_hashedValue_isNull);
  return valWrapper;
}

/**
 * FIXME Not as general-purpose as possible!
 * Is there a reason to go on a case-by-case scenario based on datatype?
 * Isn't hashing the string enough?
 * Answer: In the general case, NO! FIELDS MIGHT BE PERMUTED
 */
// ProteusValue JSONPlugin::hashValue(ProteusValueMemory mem_value, const
// ExpressionType* type)    {
//    LLVMContext& llvmContext = context->getLLVMContext();
//    Type* charPtrType = Type::getInt8PtrTy(llvmContext);
//    Type* int64Type = Type::getInt64Ty(llvmContext);
//    Type* int32Type = Type::getInt32Ty(llvmContext);
//    Type* int8Type = Type::getInt8Ty(llvmContext);
//    Type* int1Type = Type::getInt1Ty(llvmContext);
//    llvm::Type* doubleType = Type::getDoubleTy(llvmContext);
//    IRBuilder<>* Builder = context->getBuilder();
//    Function* F = context->getGlobalFunction();
//    PointerType* ptr_jsmnStructType = context->CreateJSMNStructPtr();
//
//    Function* debugInt64 = context->getFunction("printi64");
//    Function* debugInt = context->getFunction("printi");
//    vector<Value*> ArgsV;
//    Value* tokenNo = Builder->CreateLoad(mem_value.mem);
//
//    AllocaInst* mem_tokens = NamedValuesJSON[var_tokenPtr];
//    AllocaInst* mem_tokenOffset = NamedValuesJSON[var_tokenOffset];
//
//    AllocaInst* mem_tokens_shifted = context->CreateEntryBlockAlloca(F,
//            std::string(var_tokenPtr), context->CreateJSMNStruct());
//    Value* token = context->getArrayElem(mem_tokens, tokenNo);
//    Builder->CreateStore(token,mem_tokens_shifted);
//    Value* token_start = context->getStructElem(mem_tokens_shifted,1);
//    Value* token_end = context->getStructElem(mem_tokens_shifted,2);
//
//    Value* bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);
//    Value* bufShiftedPtr = Builder->CreateInBoundsGEP(bufPtr, token_start);
//
//    Function* conversionFunc = nullptr;
//    Function* hashFunc = nullptr;
//
//    AllocaInst* mem_hashedValue = nullptr;
//    AllocaInst* mem_hashedValue_isNull = nullptr;
//    Value* hashedValue = nullptr;
//    Value* convertedValue = nullptr;
//
//    mem_hashedValue = context->CreateEntryBlockAlloca(F,
//                    std::string("hashValue"), int64Type);
//    mem_hashedValue_isNull = context->CreateEntryBlockAlloca(F,
//                    std::string("hashvalue_isNull"), int1Type);
//    Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
//    Builder->CreateStore(context->createInt64(0), mem_hashedValue);
//    string error_msg;
//
//    /**
//     * Return (nil) for cases path was not found
//     */
//    BasicBlock *ifBlock, *elseBlock, *endBlock;
//    endBlock = BasicBlock::Create(llvmContext, "afterReadValueHash", F);
//    context->CreateIfElseBlocks(context->getGlobalFunction(), "ifPathHash",
//    "elsePathNullEqHash",
//                                    &ifBlock, &elseBlock,endBlock);
//    Value* minus_1 = context->createInt64(-1);
//    Value *cond = Builder->CreateICmpNE(tokenNo,minus_1);
//#ifdef DEBUG
////            ArgsV.clear();
////            ArgsV.push_back(context->createInt32(-34));
////            Builder->CreateCall(debugInt, ArgsV);
////            ArgsV.clear();
////            ArgsV.push_back(tokenNo);
////            Builder->CreateCall(debugInt64, ArgsV);
////            ArgsV.clear();
//#endif
//    Builder->CreateCondBr(cond,ifBlock,elseBlock);
//
//    /**
//     * IF BLOCK (tokenNo != -1)
//     */
//    Builder->SetInsertPoint(ifBlock);
//
//    hashFunc = context->getFunction("hashStringC");
//    ArgsV.clear();
//    ArgsV.push_back(bufPtr);
//    //FIXME tmp! - must make token_start and token_end datatype size_t
//    Value* token_start64 = Builder->CreateSExt(token_start,int64Type);
//    Value* token_end64 = Builder->CreateSExt(token_end,int64Type);
//    ArgsV.push_back(token_start64);
//    ArgsV.push_back(token_end64);
//
//    hashedValue = Builder->CreateCall(hashFunc, ArgsV, "hashStringC");
//    Builder->CreateStore(hashedValue, mem_hashedValue);
//    Builder->CreateStore(context->createFalse(), mem_hashedValue_isNull);
//
//    Builder->CreateBr(endBlock);
//
//    /**
//     * ELSE BLOCK
//     * "nullptr" case
//     * TODO What should the behavior be in this case?
//     *
//     * c-p from Sybase manual: If the grouping column contains a null value,
//     * that row becomes its own group in the results.
//     * If the grouping column contains more than one null value,
//     * the null values form a single group.
//     */
//    Builder->SetInsertPoint(elseBlock);
//
//    Value* undefValue =
//    Constant::getNullValue(mem_hashedValue->getAllocatedType());
//    Builder->CreateStore(undefValue, mem_hashedValue);
//    Builder->CreateStore(context->createTrue(), mem_hashedValue_isNull);
//    Builder->CreateBr(endBlock);
//
//    Builder->SetInsertPoint(endBlock);
//
//    ProteusValue valWrapper;
//    valWrapper.value = Builder->CreateLoad(mem_hashedValue);
//    valWrapper.isNull = Builder->CreateLoad(mem_hashedValue_isNull);
//    return valWrapper;
//}
void JSONPlugin::flushChunk(ProteusValueMemory mem_value, Value *fileName) {
  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *int64Type = Type::getInt64Ty(llvmContext);
  Type *int32Type = Type::getInt32Ty(llvmContext);
  Type *int8Type = Type::getInt8Ty(llvmContext);
  Type *int1Type = Type::getInt1Ty(llvmContext);
  llvm::Type *doubleType = Type::getDoubleTy(llvmContext);
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = context->getGlobalFunction();
  PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();

  // Preparing arguments
  std::vector<Value *> ArgsV;
  // buffer
  Value *bufPtr = Builder->CreateLoad(NamedValuesJSON[var_buf]);
  // start + end
  Value *tokenNo = Builder->CreateLoad(mem_value.mem);

#ifdef DEBUG
//    Function* debugInt = context->getFunction("printi64");
#endif
#ifdef DEBUG
//        //Which token is the appropriate one?
//        Function* debugInt = context->getFunction("printi64");
//        ArgsV.push_back(tokenNo);
//        Builder->CreateCall(debugInt, ArgsV);
//        ArgsV.clear();
#endif

  AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
  AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
      F, std::string(var_tokenPtr), context->CreateJSMNStruct());
  Value *token = context->getArrayElem(mem_tokens, tokenNo);
  Builder->CreateStore(token, mem_tokens_shifted);
  Value *token_start = context->getStructElem(mem_tokens_shifted, 1);
  Value *token_end = context->getStructElem(mem_tokens_shifted, 2);
  // where to flush? -> got it ready

#ifdef DEBUG
//        ArgsV.clear();
//        debugInt = context->getFunction("printi");
//        ArgsV.push_back(token_start);
//        Builder->CreateCall(debugInt, ArgsV);
//        ArgsV.clear();
//
//        ArgsV.push_back(token_end);
//        Builder->CreateCall(debugInt, ArgsV);
//        ArgsV.clear();
#endif

  Value *token_start64 = Builder->CreateSExt(token_start, int64Type);
  Value *token_end64 = Builder->CreateSExt(token_end, int64Type);
  ArgsV.push_back(bufPtr);
  ArgsV.push_back(token_start64);
  ArgsV.push_back(token_end64);
  ArgsV.push_back(fileName);
  Function *flushFunc = context->getFunction("flushStringC");
  Builder->CreateCall(flushFunc, ArgsV);
}

void JSONPlugin::generate(const ::Operator &producer) {
  return scanObjects(producer, context->getGlobalFunction());
}

void JSONPlugin::finish() {
  close(fd);
  munmap((void *)buf, fsize);
}

Value *JSONPlugin::getValueSize(ProteusValueMemory mem_value,
                                const ExpressionType *type) {
  switch (type->getTypeID()) {
    case BOOL:
    case INT:
    case FLOAT: {
      Type *explicitType = (mem_value.mem)->getAllocatedType();
      return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
    }
    case STRING:
    case BAG:
    case LIST:
    case SET:
    case RECORD: {
      IRBuilder<> *Builder = context->getBuilder();
      Function *F = context->getGlobalFunction();
      /* mem_value contains the address of a tokenNo */
      /* Must return tokens[tokenNo].end - tokens[tokenNo].start */

      AllocaInst *mem_tokens = NamedValuesJSON[var_tokenPtr];
      Value *val_offset = Builder->CreateLoad(mem_value.mem);

      PointerType *ptr_jsmnStructType = context->CreateJSMNStructPtr();
      AllocaInst *mem_tokens_shifted = context->CreateEntryBlockAlloca(
          F, string(var_tokenPtr), context->CreateJSMNStruct());
      Value *token_i = context->getArrayElem(mem_tokens, val_offset);
      Builder->CreateStore(token_i, mem_tokens_shifted);
      // 0: jsmntype_t type;
      // 1: int start;
      // 2: int end;
      // 3: int size;
      Value *token_i_start = context->getStructElem(mem_tokens_shifted, 1);
      Value *token_i_end = context->getStructElem(mem_tokens_shifted, 2);
      return Builder->CreateSub(token_i_end, token_i_start);
    }
    default: {
      string error_msg = string("[JSON Plugin]: Unknown datatype");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
}

JSONPlugin::~JSONPlugin() { delete[] tokens; }

/**
 * NON-CODEGEN'D FUNCTIONALITY, USED FOR TESTING
 */

/*
 * Generic INTERPRETED Implementation: Simply identifies "tuples"
 * Assumption: Handles data in the form [ obj, obj, ..., obj ]
 */
void JSONPlugin::scanObjectsInterpreted(list<string> path,
                                        list<ExpressionType *> types) {
  // Token[0] contains info about entire document
  for (int i = 1; tokens[i].start != 0;) {
    // We want token i to be one of the 'outermost' objects
    int curr = i;

    // Work done for every 'tuple'
    // TOKEN_PRINT(tokens[i]);
    int neededToken = readPathInterpreted(i, path);
    // Actually, must return (nil) in this case!!!
    if (neededToken <= 0) {
      printf("(nil)\n");
    }
    readValueInterpreted(neededToken, types.back());

    //'skipToEnd'
    while (tokens[i].start < tokens[curr].end && tokens[i].start != 0) {
      i++;
    }
  }
}

int JSONPlugin::readPathInterpreted(int parentToken, list<string> path) {
  // Only objects are relevant to path expressions
  if (tokens[parentToken].type != JSMN_OBJECT) {
    string msg = string(
        "[JSON Plugin - jsmn: ]: Path traversal is only applicable to objects");
    LOG(ERROR) << msg;
    throw runtime_error(msg);
  }
  if (path.size() == 0) {
    string error_msg = "[JSONPlugin - jsmn: ] Path length cannot be 0";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  string key = path.front();
  for (int i = parentToken + 1; tokens[i].end <= tokens[parentToken].end;
       i += 2) {
    if (TOKEN_STRING(buf, tokens[i], key.c_str())) {
      // next one is the one I need
      // printf("Found ");
      // TOKEN_PRINT(tokens[i+1]);
      if (path.size() == 1) {
        return i + 1;
      } else {
        path.pop_front();
        return readPathInterpreted(i + 1, path);
      }
    }
  }
  return -1;  //(nil)
}

void JSONPlugin::readValueInterpreted(int tokenNo, const ExpressionType *type) {
  string error_msg;
  TOKEN_PRINT(tokens[tokenNo]);
  switch (type->getTypeID()) {
    case RECORD:
      // object
    case LIST:
      // array
      printf("Passing object along\n");
      break;
    case SET:
      error_msg = string("[JSON Plugin - jsmn: ]: SET datatype cannot occur");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    case BAG:
      error_msg = string("[JSON Plugin - jsmn: ]: BAG datatype cannot occur");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    case BOOL:
      if (TOKEN_STRING(buf, tokens[tokenNo], "true")) {
        printf("Value: True!\n");
      } else if (TOKEN_STRING(buf, tokens[tokenNo], "true")) {
        printf("Value: False!\n");
      } else {
        error_msg =
            string("[JSON Plugin - jsmn: ]: Error when parsing boolean");
        LOG(ERROR) << error_msg;
        throw runtime_error(string(error_msg));
      }
    case STRING:
      printf("Passing object along\n");
      break;
    case FLOAT:
      printf("Double value read: %f\n", atof(buf + tokens[tokenNo].start));
      break;
    case INT:
      // Must be careful with trailing whitespaces for some item
      // printf("Int value read: %d\n",atois(buf + tokens[tokenNo].start,
      // tokens[tokenNo].end - tokens[tokenNo].start));
      printf("Int value read: %d\n", atoi(buf + tokens[tokenNo].start));
      break;
    default:
      error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
  }
}

void JSONPlugin::scanObjectsEagerInterpreted(list<string> path,
                                             list<ExpressionType *> types) {
  // Token[0] contains info about entire document
  for (int i = 1; tokens[i].start != 0;) {
    // We want token i to be one of the 'outermost' objects
    int curr = i;

    // Work done for every 'tuple'
    // TOKEN_PRINT(tokens[i]);
    int neededToken = readPathInterpreted(i, path);
    // Actually, must return (nil) in this case!!!
    if (neededToken <= 0) {
      printf("(nil)\n");
    }
    readValueEagerInterpreted(neededToken, types.back());

    //'skipToEnd'
    while (tokens[i].start < tokens[curr].end && tokens[i].start != 0) {
      i++;
    }
  }
}

void JSONPlugin::readValueEagerInterpreted(int tokenNo,
                                           const ExpressionType *type) {
  string error_msg;
  TOKEN_PRINT(tokens[tokenNo]);
  switch (type->getTypeID()) {
    case RECORD: {
      list<RecordAttribute *> args = ((RecordType *)type)->getArgs();
      // XXX BUT ATTRIBUTES MIGHT NOT BE IN SEQUENCE!!
      //        int i = tokenNo+2;
      //        int sizeCnt = 0;
      //        list<RecordAttribute*>::iterator it = args.begin();
      //        for(; it != args.end(); it++)    {
      //            RecordAttribute *attr = *it;
      //            sizeCnt += 2;
      //            readValueEagerInterpreted(tokenNo +
      //            sizeCnt,attr->getOriginalType());
      //            //TOKEN_PRINT(tokens[tokenNo + sizeCnt]);
      //        }
      int i = tokenNo + 1;
      map<string, RecordAttribute *> &argTypes =
          ((RecordType *)type)->getArgsMap();
      while (tokens[i].start < tokens[tokenNo].end && tokens[i].start != 0) {
        string attrName =
            string(buf + tokens[i].start, tokens[i].end - tokens[i].start);
        const ExpressionType *attrType =
            (argTypes[attrName])->getOriginalType();
        i++;
        readValueEagerInterpreted(i, attrType);
        i++;
      }
      break;
    }
    case LIST: {
      // array
      printf("Passing list along\n");
      const ExpressionType &nestedType =
          ((CollectionType *)type)->getNestedType();
      int i = tokenNo + 1;
      while (tokens[i].start < tokens[tokenNo].end && tokens[i].start != 0) {
        readValueEagerInterpreted(i, &nestedType);
        i++;
      }
      break;
    }
    case SET:
      error_msg = string("[JSON Plugin - jsmn: ]: SET datatype cannot occur");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    case BAG:
      error_msg = string("[JSON Plugin - jsmn: ]: BAG datatype cannot occur");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    case BOOL:
      if (TOKEN_STRING(buf, tokens[tokenNo], "true")) {
        printf("Value: True!\n");
      } else if (TOKEN_STRING(buf, tokens[tokenNo], "true")) {
        printf("Value: False!\n");
      } else {
        error_msg =
            string("[JSON Plugin - jsmn: ]: Error when parsing boolean");
        LOG(ERROR) << error_msg;
        throw runtime_error(string(error_msg));
      }
    case STRING:
      printf("Passing object along\n");
      break;
    case FLOAT:
      printf("Double value read: %f\n", atof(buf + tokens[tokenNo].start));
      break;
    case INT:
      // Must be careful with trailing whitespaces for some item
      // printf("Int value read: %d\n",atois(buf + tokens[tokenNo].start,
      // tokens[tokenNo].end - tokens[tokenNo].start));
      printf("Int value read: %d\n", atoi(buf + tokens[tokenNo].start));
      break;
    default:
      error_msg = string("[JSON Plugin - jsmn: ]: Unknown expression type");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
  }
}

void JSONPlugin::unnestObjectsInterpreted(list<string> path) {
  // Token[0] contains info about entire document
  for (int i = 1; tokens[i].start != 0;) {
    // We want token i to be one of the 'outermost' objects
    int curr = i;

    cout << "PARENT: " << i << endl;
    TOKEN_PRINT(tokens[i]);
    cout << "KIDS: " << endl;
    // Work done for every 'tuple'
    // TOKEN_PRINT(tokens[i]);
    int neededToken = readPathInterpreted(i, path);
    // Actually, must return (nil) in this case!!!
    if (neededToken <= 0) {
      printf("(nil)\n");
    }

    TOKEN_PRINT(tokens[neededToken]);
    if (tokens[neededToken].type != 2) {
      string error_msg =
          string("[JSON Plugin - jsmn: ]: Can only unnest collections!");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    } else {
      unnestObjectInterpreted(neededToken);
    }
    //'skipToEnd'
    while (tokens[i].start < tokens[curr].end && tokens[i].start != 0) {
      i++;
    }
  }
}

void JSONPlugin::unnestObjectInterpreted(int parentToken) {
  int i = parentToken + 1;
  while (tokens[i].end <= tokens[parentToken].end && tokens[i].end != 0) {
    cout << i << ":     ";
    TOKEN_PRINT(tokens[i]);
    // skip all its children
    int i_contents = i + 1;
    while (tokens[i_contents].start <= tokens[i].end &&
           tokens[i_contents].start != 0) {
      i_contents++;
    }
    i = i_contents;
  }
}
}  // namespace jsmn
