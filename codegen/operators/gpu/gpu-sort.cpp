/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

#include "operators/gpu/gpu-sort.hpp"
#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"

expressions::RecordConstruction buildSortOutputExpression(
    GpuRawContext *const context, const vector<expression_t> &orderByFields,
    const vector<direction> &dirs) {
  size_t i = 0;

  list<expressions::AttributeConstruction> attrs;
  for (auto expr : orderByFields) {
    assert(expr.isRegistered() && "All expressions must be registered!");

    auto e = expr;
    if (dirs[i++] == DESC) e = -e;

    attrs.emplace_back(expr.getRegisteredAttrName(), e);
    // recattr.push_back(new RecordAttribute{expr->getRegisteredAs()});
  }

  return {attrs};
}

std::string computeSuffix(GpuRawContext *const context,
                          const vector<expression_t> &orderByFields) {
  std::string suffix = "";
  for (auto expr : orderByFields) {
    size_t size = context->getSizeOf(
        expr.getExpressionType()->getLLVMType(context->getLLVMContext()));

    if (size == 32 / 8) {
      suffix += "i";
    } else if (size == 64 / 8) {
      suffix += "l";
    } else {
      assert(false &&
             "GPU-sorting by attributes with size different than "
             "32/64-bits is not supported yet");
    }
  }
  return suffix;
}

GpuSort::GpuSort(RawOperator *const child, GpuRawContext *const context,
                 const vector<expression_t> &orderByFields,
                 const vector<direction> &dirs, gran_t granularity)
    : UnaryRawOperator(child),
      context(context),
      orderByFields(orderByFields),
      dirs(dirs),
      granularity(granularity),
      outputExpr(buildSortOutputExpression(context, orderByFields, dirs)),
      relName(orderByFields[0].getRegisteredRelName()),
      suffix(computeSuffix(context, orderByFields)) {
  assert(granularity == gran_t::GRID || granularity == gran_t::THREAD);
}

void GpuSort::produce() {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg = RawCatalog::getInstance().getPlugin(relName);
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);
  // Type   * cnt_type   = PointerType::getUnqual(ArrayType::get(oid_type,
  // numOfBuckets)); cntVar_id           = context->appendStateVar(cnt_type);

  // oidVar_id           =
  // context->appendStateVar(PointerType::getUnqual(oid_type));

  // std::vector<Type *> block_types;
  // for (size_t i = 0 ; i < orderByFields.size() ; ++i){
  //     block_types.emplace_back(RecordAttribute(orderByFields[i]->getRegisteredAs(),
  //     true).getLLVMType(llvmContext));
  //     wfSizes.emplace_back(context->getSizeOf(orderByFields[i]->getExpressionType()->getLLVMType(llvmContext)));
  // }

  // Type * block_stuct = StructType::get(llvmContext, block_types);

  // blkVar_id           =
  // context->appendStateVar(PointerType::getUnqual(ArrayType::get(block_stuct,
  // numOfBuckets)));

  Type *elemPointer = outputExpr.getExpressionType()->getLLVMType(llvmContext);
  mem_type = ArrayType::get(elemPointer, h_vector_size * sizeof(int32_t) /
                                             context->getSizeOf(elemPointer));

  flush_sorted();

  context->popPipeline();

  auto flush_pip = context->removeLatestPipeline();
  // flush_fun = flush_pip->getKernel();

  context->pushPipeline(flush_pip);

  memVar_id = context->appendStateVar(
      PointerType::getUnqual(mem_type),
      [=](llvm::Value *) {
        Function *gb = context->getFunction("get_dev_buffer");
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem = Builder->CreateCall(gb);
        mem = Builder->CreateBitCast(mem, PointerType::getUnqual(mem_type));
        return mem;  // context->allocateStateVar(mem_type);
      },

      [=](llvm::Value *, llvm::Value *s) {
        // context->deallocateStateVar(s);
      });

  cntVar_id = context->appendStateVar(
      PointerType::getUnqual(oid_type),
      [=](llvm::Value *) {
        IRBuilder<> *Builder = context->getBuilder();

        Value *mem_acc = context->allocateStateVar(oid_type);

        // Builder->CreateStore(ConstantInt::get(oid_type, 0), mem_acc);
        context->CodegenMemset(mem_acc, ConstantInt::get(oid_type, 0),
                               context->getSizeOf(oid_type));

        // OperatorState childState{*this, map<RecordAttribute,
        // RawValueMemory>{}}; ExpressionFlusherVisitor flusher{context,
        // childState, outPath.c_str(), relName}; flusher.beginList();

        return mem_acc;
      },

      [=](llvm::Value *, llvm::Value *s) {
        IRBuilder<> *Builder = context->getBuilder();

        Type *charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

        Function *f = context->getFunction("subpipeline_consume");
        FunctionType *f_t = f->getFunctionType();

        Type *substate_t = f_t->getParamType(f_t->getNumParams() - 1);

        Type *size_type = s->getType()->getPointerElementType();

        Function *F = Builder->GetInsertBlock()->getParent();
        Value *size_mem =
            context->CreateEntryBlockAlloca(F, "size_mem", size_type);
        context->CodegenMemcpy(size_mem, s, context->getSizeOf(size_type));
        Value *size = Builder->CreateLoad(size_mem);
        vector<Value *> args{context->getStateVar(memVar_id), size};

        this->call_sort(args[0], args[1]);

        Value *substate =
            Builder->CreateBitCast(context->getSubStateVar(), substate_t);
        args.emplace_back(substate);

        Builder->CreateCall(f, args);

        // OperatorState childState{*this, map<RecordAttribute,
        // RawValueMemory>{}}; ExpressionFlusherVisitor flusher{context,
        // childState, outPath.c_str(), relName}; flusher.endList();
        // flusher.flushOutput();
        context->deallocateStateVar(s);
      });

  getChild()->produce();
}

void GpuSort::consume(RawContext *const context,
                      const OperatorState &childState) {
  GpuRawContext *const ctx = dynamic_cast<GpuRawContext *const>(context);
  assert(ctx);
  consume(ctx, childState);
}

void GpuSort::consume(GpuRawContext *const context,
                      const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  map<RecordAttribute, RawValueMemory> bindings{childState.getBindings()};

  Plugin *pg = RawCatalog::getInstance().getPlugin(relName);
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  // size_t max_width = 0;
  // for (const auto &e: orderByFields){
  //     max_width = std::max(max_width,
  //     context->getSizeOf(e->getExpressionType()->getLLVMType(llvmContext)));
  // }

  // cap                   = blockSize / width;
  // Value * capacity      = ConstantInt::get(oid_type, cap);
  // Value * last_index    = ConstantInt::get(oid_type, cap - 1);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Value *s_cnt_mem = context->getStateVar(cntVar_id);
  // AllocaInst * ready_cnt_mem = context->CreateEntryBlockAlloca(F, "readyN",
  // oid_type); Builder->CreateStore(Builder->CreateLoad(s_cnt_mem),
  // ready_cnt_mem);
  Value *mem_ptr = context->getStateVar(memVar_id);

  // Builder->SetInsertPoint(context->getEndingBlock      ());
  // Builder->CreateStore(Builder->CreateLoad(ready_cnt_mem), s_cnt_mem);

  Builder->SetInsertPoint(insBB);

  map<RecordAttribute, RawValueMemory> variableBindings;

  // vector<Type *> members;
  // for (size_t i = 0 ; i < orderByFields.size() ; ++i){
  //     RecordAttribute tblock{orderByFields[i]->getRegisteredAs(), true};
  //     members.push_back(tblock.getLLVMType(llvmContext));
  // }

  // StructType * partition = StructType::get(llvmContext, members);

  Value *indx = Builder->CreateAtomicRMW(AtomicRMWInst::BinOp::Add, s_cnt_mem,
                                         ConstantInt::get(oid_type, 1),
                                         AtomicOrdering::Monotonic);

  // Value * indx = Builder->CreateLoad(ready_cnt_mem);

  Value *el_ptr = Builder->CreateInBoundsGEP(
      mem_ptr, std::vector<Value *>{context->createInt64(0), indx});

  ExpressionGeneratorVisitor exprGenerator(context, childState);
  RawValue valWrapper = outputExpr.accept(exprGenerator);
  Value *el = valWrapper.value;

  Builder->CreateStore(el, el_ptr);

  // Value * next_indx = Builder->CreateAdd(indx, ConstantInt::get((IntegerType
  // *) indx->getType(), 1)); Builder->CreateStore(next_indx, ready_cnt_mem);

  // RecordAttribute tupCnt  = RecordAttribute(relName, "activeCnt",
  // pg->getOIDType()); //FIXME: OID type for blocks ?

  // RawValueMemory mem_cntWrapper;
  // mem_cntWrapper.mem      = blockN_ptr;
  // mem_cntWrapper.isNull   = context->createFalse();
  // variableBindings[tupCnt] = mem_cntWrapper;

  // ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline *
  // pip){this->open (pip);});
  // ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline *
  // pip){this->close(pip);});
}

void GpuSort::flush_sorted() {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg = RawCatalog::getInstance().getPlugin(relName);
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  // flushingFunc = (*context)->createHelperFunction("flush", std::vector<Type
  // *>{mem, oid_type}, std::vector<bool>{true, true}, std::vector<bool>{true,
  // false}); closingPip   = (context->operator->()); IRBuilder<> * Builder =
  // context->getBuilder    (); BasicBlock  * insBB         =
  // Builder->GetInsertBlock(); Function    * F             =
  // insBB->getParent();
  // //Get the ENTRY BLOCK
  // context->setCurrentEntryBlock(Builder->GetInsertBlock());

  vector<size_t> params;
  params.emplace_back(
      context->appendParameter(PointerType::getUnqual(mem_type), true, true));
  params.emplace_back(context->appendParameter(oid_type, false, false));

  context->setGlobalFunction();

  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "end", F);
  // BasicBlock * MainBB  = BasicBlock::Create(llvmContext, "main", F);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  context->setEndingBlock(AfterBB);

  // std::vector<Type *> args{context->getStateVars()};

  // context->pushNewCpuPipeline();

  // for (Type * t: args) context->appendStateVar(t);

  // LLVMContext & llvmContext   = context->getLLVMContext();

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  // context->setGlobalFunction();

  // IRBuilder<> * Builder       = context->getBuilder    ();
  // BasicBlock  * insBB         = Builder->GetInsertBlock();
  // Function    * F             = insBB->getParent();
  // Get the ENTRY BLOCK
  // context->setCurrentEntryBlock(Builder->GetInsertBlock());

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "flushCond", F);

  // // Start insertion in CondBB.
  // Builder->SetInsertPoint(CondBB);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "flushBody", F);

  // Make the new basic block for the increment, inserting after current block.
  // BasicBlock *IncBB = BasicBlock::Create(llvmContext, "flushInc", F);

  // // Create the "AFTER LOOP" block and insert it.
  // BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "flushEnd", F);
  // context->setEndingBlock(AfterBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Value * cnt         = Builder->CreateLoad(context->getStateVar(cntVar_id));
  Value *mem_ptr = context->getArgument(params[0]);
  Value *cnt = context->getArgument(params[1]);

  Builder->SetInsertPoint(CondBB);

  Value *indx = context->threadId();

  Value *cond;
  if (granularity == gran_t::THREAD)
    cond = context->createTrue();
  else
    cond = Builder->CreateICmpEQ(indx, ConstantInt::get(indx->getType(), 0));
  // Insert the conditional branch into the end of CondBB.
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  map<RecordAttribute, RawValueMemory> variableBindings;

  // Value * rec_ptr = Builder->CreateInBoundsGEP(mem_ptr, std::vector<Value
  // *>{context->createInt64(0), indx});

  // Value * rec     = Builder->CreateLoad(rec_ptr);
  // RawValue v{rec, context->createFalse()};
  // expressions::RawValueExpression r{outputExpr.getExpressionType(), v};

  // OperatorState state{*this, variableBindings};
  // ExpressionFlusherVisitor flusher{context, state, outPath.c_str(), relName};

  // //flushing out delimiter (IF NEEDED)
  // flusher.flushDelim(Builder->CreateZExtOrBitCast(indx, int64_type));

  // r.accept(flusher);

  // RecordType *t = (RecordType *) outputExpr.getExpressionType();
  // size_t i = 0;
  // for (const auto &e: orderByFields){
  //     RecordAttribute attr = e->getRegisteredAs();
  //     Value         * p    = t->projectArg(rec, &attr, Builder);

  //     RawValueMemory mem;
  //     mem.mem    = context->CreateEntryBlockAlloca(F, attr.getAttrName(),
  //     p->getType()); mem.isNull = context->createFalse();

  //     if (dirs[i++] == DESC) p = Builder->CreateNeg(p);

  //     Builder->CreateStore(p, mem.mem);

  //     variableBindings[attr] = mem;
  // }

  RecordAttribute recs{relName, "__sorted", outputExpr.getExpressionType()};
  RecordAttribute brec{recs, true};
  mem_ptr = Builder->CreateBitCast(mem_ptr, brec.getLLVMType(llvmContext));
  AllocaInst *mem_mem_ptr =
      context->CreateEntryBlockAlloca(F, "__sorted_mem", mem_ptr->getType());
  Builder->CreateStore(mem_ptr, mem_mem_ptr);

  // Function * p = context->getFunction("printptr");
  // Builder->CreateCall(p, Builder->CreateBitCast(mem_ptr,
  // Type::getInt8PtrTy(llvmContext)));

  variableBindings[brec] = RawValueMemory{mem_mem_ptr, context->createFalse()};

  AllocaInst *cnt_mem =
      context->CreateEntryBlockAlloca(F, "activeCnt_mem", oid_type);
  RecordAttribute bcnt{relName, "activeCnt", pg->getOIDType()};
  Builder->CreateStore(cnt, cnt_mem);
  variableBindings[bcnt] = RawValueMemory{cnt_mem, context->createFalse()};
  // Function * p = context->getFunction("printptr");
  // cnt->getType()->dump();
  // Builder->CreateCall(p, Builder->CreateBitCast(mem_ptr,
  // Type::getInt8PtrTy(llvmContext))); p = context->getFunction("printi");
  // cnt->getType()->dump();
  // Builder->CreateCall(p, cnt);

  AllocaInst *blockN_ptr = context->CreateEntryBlockAlloca(F, "i", oid_type);
  RecordAttribute oid{relName, activeLoop, pg->getOIDType()};
  Builder->CreateStore(ConstantInt::get(oid_type, 0), blockN_ptr);
  variableBindings[oid] = RawValueMemory{blockN_ptr, context->createFalse()};

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(AfterBB);

  // Builder->SetInsertPoint(IncBB);

  // Value * next = Builder->CreateAdd(indx, ConstantInt::get(oid_type, 1));
  // Builder->CreateStore(next, blockN_ptr);

  // Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
}

// void GpuSort::open (RawPipeline * pip){
//     size_t * cnts = (size_t *) malloc(sizeof(size_t) * (numOfBuckets + 1));
//     //FIXME: is it always size_t the correct type ?

//     for (int i = 0 ; i < numOfBuckets + 1; ++i) cnts[i] = 0;

//     pip->setStateVar<size_t *>(cntVar_id, cnts);

//     void ** blocks = (void **) malloc(sizeof(void *) * numOfBuckets *
//     orderByFields.size());

//     for (int i = 0 ; i < numOfBuckets ; ++i){
//         for (size_t j = 0 ; j < orderByFields.size() ; ++j){
//             blocks[i * orderByFields.size() + j] = get_buffer(wfSizes[j] *
//             cap);
//         }
//     }

//     pip->setStateVar<void   *>(blkVar_id, (void *) blocks);

//     pip->setStateVar<size_t *>(oidVar_id, cnts + numOfBuckets);
// }

// void GpuSort::close(RawPipeline * pip){
//     // ((void (*)(void *)) this->flushFunc)(pip->getState());
//     ((void (*)(void *))
//     closingPip->getCompiledFunction(flushingFunc))(pip->getState());

//     free(pip->getStateVar<size_t *>(cntVar_id));
//     free(pip->getStateVar<void   *>(blkVar_id)); //FIXME: release buffers
//     before freeing memory!
//     // oidVar is part of cntVar, so they are freed together
// }

void GpuSort::call_sort(Value *mem, Value *N) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type = context->createSizeType();
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  Value *count = Builder->CreateZExtOrBitCast(N, size_type);

  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Value *mem_char_ptr = Builder->CreateBitCast(mem, charPtrType);

  Function *qsort = context->getFunction("qsort_" + suffix);
  Builder->CreateCall(qsort, {mem_char_ptr, count});
}
