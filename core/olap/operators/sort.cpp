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

#include "operators/sort.hpp"

#include <util/project-record.hpp>

#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "memory/block-manager.hpp"

using namespace llvm;

expressions::RecordConstruction buildSortOutputExpression(
    ParallelContext *const context, const vector<expression_t> &orderByFields) {
  size_t i = 0;

  list<expressions::AttributeConstruction> attrs;
  for (auto expr : orderByFields) {
    assert(expr.isRegistered() && "All expressions must be registered!");

    attrs.emplace_back(expr.getRegisteredAttrName(), expr);
    // recattr.push_back(new RecordAttribute{expr->getRegisteredAs()});
  }

  return {attrs};
}

Sort::Sort(Operator *const child, ParallelContext *const context,
           const vector<expression_t> &orderByFields,
           const vector<direction> &dirs)
    : UnaryOperator(child),
      context(context),
      orderByFields(orderByFields),
      dirs(dirs),
      outputExpr(buildSortOutputExpression(context, orderByFields)),
      relName(orderByFields[0].getRegisteredRelName()) {}

void Sort::produce_(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg = Catalog::getInstance().getPlugin(relName);
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
  mem_type = ArrayType::get(elemPointer, 1024 * 1024);

  flush_sorted();

  context->popPipeline();

  auto flush_pip = context->removeLatestPipeline();
  // flush_fun = flush_pip->getKernel();

  context->pushPipeline(flush_pip);

  memVar_id = context->appendStateVar(
      PointerType::getUnqual(mem_type),
      // [=](llvm::Value *){
      //     return context->allocateStateVar(mem_type);
      // },

      // [=](llvm::Value *, llvm::Value * s){
      //     context->deallocateStateVar(s);
      // }
      [=](llvm::Value *) {
        Function *gb = context->getFunction("get_buffer");
        IRBuilder<> *Builder = context->getBuilder();
        Value *mem = Builder->CreateCall(
            gb, context->createSizeT(BlockManager::block_size));
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

        Builder->CreateStore(ConstantInt::get(oid_type, 0), mem_acc);

        // OperatorState childState{*this, map<RecordAttribute,
        // ProteusValueMemory>{}}; ExpressionFlusherVisitor flusher{context,
        // childState, outPath.c_str(), relName}; flusher.beginList();

        return mem_acc;
      },

      [=](llvm::Value *, llvm::Value *s) {
        IRBuilder<> *Builder = context->getBuilder();

        Type *charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

        Function *f = context->getFunction("subpipeline_consume");
        FunctionType *f_t = f->getFunctionType();

        Type *substate_t = f_t->getParamType(f_t->getNumParams() - 1);

        vector<Value *> args{context->getStateVar(memVar_id),
                             Builder->CreateLoad(s)};

        this->call_sort(args[0], args[1]);

        Value *substate =
            Builder->CreateBitCast(context->getSubStateVar(), substate_t);
        args.emplace_back(substate);

        Builder->CreateCall(f, args);

        // OperatorState childState{*this, map<RecordAttribute,
        // ProteusValueMemory>{}}; ExpressionFlusherVisitor flusher{context,
        // childState, outPath.c_str(), relName}; flusher.endList();
        // flusher.flushOutput();
        context->deallocateStateVar(s);
      });

  getChild()->produce(context);
}

void Sort::consume(Context *const context, const OperatorState &childState) {
  ParallelContext *const ctx = dynamic_cast<ParallelContext *const>(context);
  assert(ctx);
  consume(ctx, childState);
}

void Sort::consume(ParallelContext *const context,
                   const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  Plugin *pg = Catalog::getInstance().getPlugin(relName);
  auto *oid_type = (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  AllocaInst *ready_cnt_mem =
      context->CreateEntryBlockAlloca(F, "readyN", oid_type);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Value *s_cnt_mem = context->getStateVar(cntVar_id);
  Builder->CreateStore(Builder->CreateLoad(s_cnt_mem), ready_cnt_mem);
  Value *mem_ptr = context->getStateVar(memVar_id);

  Builder->SetInsertPoint(context->getEndingBlock());
  Builder->CreateStore(Builder->CreateLoad(ready_cnt_mem), s_cnt_mem);

  Builder->SetInsertPoint(insBB);

  Value *indx = Builder->CreateLoad(ready_cnt_mem);

  Value *el_ptr = Builder->CreateInBoundsGEP(
      mem_ptr, std::vector<Value *>{context->createInt64(0), indx});

  ExpressionGeneratorVisitor exprGenerator(context, childState);
  ProteusValue valWrapper = outputExpr.accept(exprGenerator);
  Value *el = valWrapper.value;

  Builder->CreateStore(el, el_ptr);

  Value *next_indx = Builder->CreateAdd(
      indx, ConstantInt::get((IntegerType *)indx->getType(), 1));
  Builder->CreateStore(next_indx, ready_cnt_mem);
}

void Sort::flush_sorted() {
  LLVMContext &llvmContext = context->getLLVMContext();

  Plugin *pg = Catalog::getInstance().getPlugin(relName);
  auto *oid_type = (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);

  vector<size_t> params;
  params.emplace_back(
      context->appendParameter(PointerType::getUnqual(mem_type), true, true));
  params.emplace_back(context->appendParameter(oid_type, false, false));

  context->setGlobalFunction();

  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "end", F);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  context->setEndingBlock(AfterBB);

  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "flushCond", F);

  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "flushBody", F);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Value * cnt         = Builder->CreateLoad(context->getStateVar(cntVar_id));
  Value *mem_ptr = context->getArgument(params[0]);
  Value *cnt = context->getArgument(params[1]);

  Builder->SetInsertPoint(CondBB);

  Value *indx = context->threadId();

  Value *cond =
      Builder->CreateICmpEQ(indx, ConstantInt::get(indx->getType(), 0));
  // Insert the conditional branch into the end of CondBB.
  Builder->CreateCondBr(cond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  RecordAttribute recs{relName, "__sorted", outputExpr.getExpressionType()};
  RecordAttribute brec{recs, true};
  mem_ptr = Builder->CreateBitCast(mem_ptr, brec.getLLVMType(llvmContext));
  AllocaInst *mem_mem_ptr =
      context->CreateEntryBlockAlloca(F, "__sorted_mem", mem_ptr->getType());
  Builder->CreateStore(mem_ptr, mem_mem_ptr);

  variableBindings[brec] =
      ProteusValueMemory{mem_mem_ptr, context->createFalse()};

  AllocaInst *cnt_mem =
      context->CreateEntryBlockAlloca(F, "activeCnt_mem", oid_type);
  RecordAttribute bcnt{relName, "activeCnt", pg->getOIDType()};
  Builder->CreateStore(cnt, cnt_mem);
  variableBindings[bcnt] = ProteusValueMemory{cnt_mem, context->createFalse()};

  AllocaInst *blockN_ptr = context->CreateEntryBlockAlloca(F, "i", oid_type);
  RecordAttribute oid{relName, activeLoop, pg->getOIDType()};
  Builder->CreateStore(ConstantInt::get(oid_type, 0), blockN_ptr);
  variableBindings[oid] =
      ProteusValueMemory{blockN_ptr, context->createFalse()};

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(AfterBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(context->getEndingBlock());
}

void Sort::call_sort(Value *mem, Value *N) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  IntegerType *int32_type = Type::getInt32Ty(llvmContext);
  IntegerType *int64_type = Type::getInt64Ty(llvmContext);

  IntegerType *size_type;
  if (sizeof(size_t) == 4)
    size_type = int32_type;
  else if (sizeof(size_t) == 8)
    size_type = int64_type;
  else
    assert(false);

  Value *count = Builder->CreateZExtOrBitCast(N, size_type);

  Type *entry = outputExpr.getExpressionType()->getLLVMType(llvmContext);
  Value *size = context->createSizeT(context->getSizeOf(entry));

  Type *entry_pointer = PointerType::get(entry, 0);

  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Function *cmp;
  {
    save_current_blocks_and_restore_at_exit_scope save{context};

    FunctionType *ftype = FunctionType::get(
        int32_type, std::vector<Type *>{charPtrType, charPtrType}, false);
    // use f_num to overcome an llvm bu with keeping dots in function names when
    // generating PTX (which is invalid in PTX)
    cmp = Function::Create(ftype, Function::ExternalLinkage,
                           "cmp" + std::to_string((uintptr_t)this),
                           context->getModule());

    {
      Attribute readOnly = Attribute::get(context->getLLVMContext(),
                                          Attribute::AttrKind::ReadOnly);
      Attribute noAlias = Attribute::get(context->getLLVMContext(),
                                         Attribute::AttrKind::NoAlias);

      std::vector<std::pair<unsigned, Attribute>> attrs;
      for (size_t i = 1; i <= 2; ++i) {  //+1 because 0 is the return value
        attrs.emplace_back(i, readOnly);
        attrs.emplace_back(i, noAlias);
      }

      cmp->setAttributes(AttributeList::get(context->getLLVMContext(), attrs));
    }

    BasicBlock *insBB =
        BasicBlock::Create(context->getLLVMContext(), "entry", cmp);
    Builder->SetInsertPoint(insBB);
    Function *F = cmp;
    // Get the ENTRY BLOCK
    context->setCurrentEntryBlock(insBB);
    BasicBlock *endBB = BasicBlock::Create(llvmContext, "end", F);
    context->setEndingBlock(endBB);

    auto args = cmp->args().begin();

    map<RecordAttribute, ProteusValueMemory> bindings[2];
    Value *recs[2]{
        Builder->CreateLoad(Builder->CreateBitCast(args++, entry_pointer)),
        Builder->CreateLoad(Builder->CreateBitCast(args++, entry_pointer))};

    for (size_t i = 0; i < 2; ++i) {
      RecordType *t = (RecordType *)outputExpr.getExpressionType();
      for (const auto &e : orderByFields) {
        RecordAttribute attr = e.getRegisteredAs();
        Value *p = projectArg(t, recs[i], &attr, Builder);
        assert(p);

        ProteusValueMemory mem;
        mem.mem = context->CreateEntryBlockAlloca(F, attr.getAttrName(),
                                                  p->getType());
        mem.isNull = context->createFalse();

        Builder->CreateStore(p, mem.mem);

        bindings[i][attr] = mem;
      }
    }

    BasicBlock *mainBB = BasicBlock::Create(llvmContext, "main", F);

    BasicBlock *greaterBB = BasicBlock::Create(llvmContext, "greater", F);
    Builder->SetInsertPoint(greaterBB);
    Builder->CreateRet(context->createInt32(1));

    BasicBlock *lessBB = BasicBlock::Create(llvmContext, "less", F);
    Builder->SetInsertPoint(lessBB);
    Builder->CreateRet(context->createInt32(-1));

    Builder->SetInsertPoint(mainBB);

    size_t i = 0;
    for (const auto &e : orderByFields) {
      const auto &d = dirs[i++];
      if (d == direction::NONE) continue;
      RecordAttribute attr = e.getRegisteredAs();

      // FIXME: replace with expressions
      Value *arg0 = Builder->CreateLoad(bindings[0][attr].mem);
      Value *arg1 = Builder->CreateLoad(bindings[1][attr].mem);

      if (d == direction::DESC) std::swap(arg0, arg1);

      BasicBlock *eqPreBB = BasicBlock::Create(llvmContext, "eqPre", F);
      BasicBlock *notGTBB = BasicBlock::Create(llvmContext, "notGT", F);

      Value *condG = (arg0->getType()->isIntegerTy())
                         ? Builder->CreateICmpSGT(arg0, arg1)
                         : Builder->CreateFCmpOGT(arg0, arg1);
      Builder->CreateCondBr(condG, greaterBB, notGTBB);

      Builder->SetInsertPoint(notGTBB);
      Value *condL = (arg0->getType()->isIntegerTy())
                         ? Builder->CreateICmpSLT(arg0, arg1)
                         : Builder->CreateFCmpOLT(arg0, arg1);
      Builder->CreateCondBr(condL, lessBB, eqPreBB);

      Builder->SetInsertPoint(eqPreBB);
    }

    Builder->CreateBr(endBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(mainBB);

    Builder->SetInsertPoint(context->getEndingBlock());
    Builder->CreateRet(context->createInt32(0));
  }

  Value *mem_char_ptr = Builder->CreateBitCast(mem, charPtrType);
  std::vector<Value *> args{mem_char_ptr, count, size, cmp};

  Function *qsort = context->getFunction("qsort");
  Builder->CreateCall(qsort, args);
}
