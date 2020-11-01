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

#include "reduce-opt.hpp"

#include <platform/memory/memory-manager.hpp>

#include "lib/util/jit/pipeline.hpp"
#include "olap/util/parallel-context.hpp"

using namespace llvm;

namespace opt {
// Reduce::Reduce(std::vector<Monoid> accs, std::vector<expression_t>
// outputExprs,
//               expression_t pred, Operator *const child, Context *context,
//               bool flushResults, const char *outPath)
//    : UnaryOperator(child),
//      accs(std::move(accs)),
//      outputExprs(std::move(outputExprs)),
//      pred(std::move(pred)),
//      context(context),
//      flushResults(flushResults),
//      outPath(outPath) {
//  if (this->accs.size() != this->outputExprs.size()) {
//    string error_msg = string("[REDUCE: ] Erroneous constructor args");
//    LOG(ERROR) << error_msg;
//    throw runtime_error(error_msg);
//  }
//}

void Reduce::produce_(ParallelContext *context) {
  generate_flush(context);

  context->popPipeline();

  auto flush_pip = context->removeLatestPipeline();
  context->pushPipeline(flush_pip);

  assert(mem_accumulators.empty());
  if (mem_accumulators.empty()) {
    int aggsNo = aggs.size();
    /* Prepare accumulator FOREACH outputExpr */
    for (const auto &agg : aggs) {
      auto aggs_i = mem_accumulators.size();
      auto acc = agg.getMonoid();
      bool is_first = (aggs_i == 0);
      bool is_last = (aggs_i == aggsNo - 1);
      bool flushDelim = (aggsNo > 1) && !is_last;
      auto mem_accumulator = resetAccumulator(agg, is_first, is_last, context);
      mem_accumulators.emplace_back(mem_accumulator);
    }
  }

  getChild()->produce(context);
}

void Reduce::consume(ParallelContext *context,
                     const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Generate condition
  ExpressionGeneratorVisitor predExprGenerator{context, childState};
  ProteusValue condition = pred.accept(predExprGenerator);
  /**
   * Predicate Evaluation:
   */
  BasicBlock *entryBlock = Builder->GetInsertBlock();
  BasicBlock *endBlock =
      BasicBlock::Create(llvmContext, "reduceCondEnd", TheFunction);
  BasicBlock *ifBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "reduceIfCond", &ifBlock,
                         endBlock);

  /**
   * IF(pred) Block
   */
  Builder->SetInsertPoint(entryBlock);

  Builder->CreateCondBr(condition.value, ifBlock, endBlock);

  Builder->SetInsertPoint(ifBlock);

  /* Time to Compute Aggs */
  auto itAcc = aggs.begin();
  auto itMem = mem_accumulators.begin();

  for (; itAcc != aggs.end(); itAcc++, itMem++) {
    auto agg = *itAcc;
    Value *mem_accumulating = nullptr;

    switch (agg.getMonoid()) {
      case SUM:
      case MULTIPLY:
      case MAX:
      case OR:
      case AND: {
        BasicBlock *cBB = Builder->GetInsertBlock();
        Builder->SetInsertPoint(context->getCurrentEntryBlock());

        mem_accumulating = context->getStateVar(*itMem);
        Value *acc_init = Builder->CreateLoad(mem_accumulating);
        Value *acc_mem =
            context->CreateEntryBlockAlloca("acc", acc_init->getType());
        Builder->CreateStore(acc_init, acc_mem);

        Builder->SetInsertPoint(context->getEndingBlock());
        Builder->CreateStore(Builder->CreateLoad(acc_mem), mem_accumulating);

        Builder->SetInsertPoint(cBB);

        ExpressionGeneratorVisitor outputExprGenerator{context, childState};

        // Load accumulator -> acc_value
        ProteusValue acc_value{Builder->CreateLoad(acc_mem),
                               context->createFalse()};

        // new_value = acc_value op outputExpr
        expressions::ProteusValueExpression val{agg.getExpressionType(),
                                                acc_value};
        auto upd = agg.toReduceExpression(val);
        ProteusValue new_val = upd.accept(outputExprGenerator);

        // store new_val to accumulator
        Builder->CreateStore(new_val.value, acc_mem);
        break;
      }
      case BAGUNION:
        generateBagUnion(agg.getExpression(), context, childState,
                         context->getStateVar(*itMem));
        break;
      case APPEND:
        //        generateAppend(context, childState);
        //        break;
      case UNION:
      default: {
        string error_msg =
            string("[Reduce: ] Unknown / Still Unsupported accumulator");
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
    }
  }

  Builder->CreateBr(endBlock);

  /**
   * END Block
   */
  Builder->SetInsertPoint(endBlock);
}

// Flush out whatever you received
// FIXME Need 'output plugin' / 'serializer'
void Reduce::generateBagUnion(const expression_t &outputExpr,
                              ParallelContext *context,
                              const OperatorState &state,
                              Value *cnt_mem) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();

  ExpressionFlusherVisitor flusher{context, state,
                                   outputExpr.getRegisteredRelName().c_str(),
                                   outputExpr.getRegisteredRelName()};

  // Backing up insertion block
  BasicBlock *currBlock = Builder->GetInsertBlock();

  // Preparing collection output (e.g., flushing out '{' in the case of JSON)
  BasicBlock *loopEntryBlock = context->getCurrentEntryBlock();

  Builder->SetInsertPoint(loopEntryBlock);
  flusher.beginList();

  // Restoring
  Builder->SetInsertPoint(currBlock);

  // results so far
  Value *resultCtr = Builder->CreateLoad(cnt_mem);

  // flushing out delimiter (IF NEEDED)
  flusher.flushDelim(resultCtr);

  outputExpr.accept(flusher);

  // increase result ctr
  Value *resultCtrInc = Builder->CreateAdd(resultCtr, context->createInt64(1));
  Builder->CreateStore(resultCtrInc, cnt_mem);

  // Backing up insertion block
  currBlock = Builder->GetInsertBlock();

  // Prepare final result output (e.g., flushing out '}' in the case of JSON)
  Builder->SetInsertPoint(context->getEndingBlock());
  flusher.endList();
  flusher.flushOutput();

  /**
   * END Block
   */
  Builder->SetInsertPoint(currBlock);
}

void Reduce::generate_flush(ParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  (*context)->setMaxWorkerSize(1, 1);

  vector<size_t> params;

  for (const auto &agg : aggs) {
    switch (agg.getMonoid()) {
      case SUM:
      case MULTIPLY:
      case MAX:
      case OR:
      case AND: {
        params.emplace_back(context->appendParameter(
            PointerType::getUnqual(
                agg.getExpressionType()->getLLVMType(llvmContext)),
            true, true));
        break;
      }
      case UNION:
      case BAGUNION:
      case APPEND: {
        params.emplace_back(~((size_t)0));
        break;
      }
      default: {
        string error_msg = string("[Reduce: ] Unknown accumulator");
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
    }
  }

  context->setGlobalFunction();

  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "end", F);
  BasicBlock *MainBB = BasicBlock::Create(llvmContext, "main", F);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  context->setEndingBlock(AfterBB);

  Builder->SetInsertPoint(MainBB);

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  std::string rel_name;
  bool found = false;
  for (const auto &t : aggs) {
    if (t.getExpression().isRegistered()) {
      rel_name = t.getRegisteredAs().getRelationName();
      found = true;
      break;
    }
  }

  if (found) {
    Plugin *pg = Catalog::getInstance().getPlugin(rel_name);

    {
      RecordAttribute tupleOID(
          rel_name, activeLoop,
          pg->getOIDType());  // FIXME: OID type for blocks ?

      Value *oid =
          ConstantInt::get(pg->getOIDType()->getLLVMType(llvmContext), 0);
      oid->setName("oid");

      variableBindings[tupleOID] = context->toMem(oid, context->createFalse());
    }

    {
      RecordAttribute tupleCnt(
          rel_name, "activeCnt",
          pg->getOIDType());  // FIXME: OID type for blocks ?
      Value *N =
          ConstantInt::get(pg->getOIDType()->getLLVMType(llvmContext), 1);
      N->setName("cnt");

      variableBindings[tupleCnt] = context->toMem(N, context->createFalse());
    }
  }

  auto itAcc = aggs.begin();
  auto itMem = params.begin();

  for (; itAcc != aggs.end(); itAcc++, itMem++) {
    auto acc = itAcc->getMonoid();
    auto outputExpr = itAcc->getExpression();
    Value *mem_accumulating = nullptr;

    if (*itMem == ~((size_t)0) || acc == BAGUNION) {
      string error_msg = string("[Reduce: ] Not implemented yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }

    if (!outputExpr.isRegistered()) {
      string error_msg = string(
          "[Reduce: ] All expressions must be "
          "registered to forward them to the parent");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }

    Value *val_mem = context->getArgument(*itMem);
    val_mem->setName(outputExpr.getRegisteredAttrName() + "_ptr");
    Value *val_acc = Builder->CreateLoad(val_mem);
    AllocaInst *acc_alloca = context->CreateEntryBlockAlloca(
        outputExpr.getRegisteredAttrName(), val_acc->getType());

    context->getBuilder()->CreateStore(val_acc, acc_alloca);

    ProteusValueMemory acc_mem{acc_alloca, context->createFalse()};
    variableBindings[outputExpr.getRegisteredAs()] = acc_mem;
  }

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to AfterBB.
  Builder->CreateBr(AfterBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(MainBB);

  //  Finish up with end (the AfterLoop)
  //  Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());
}

StateVar Reduce::resetAccumulator(const agg_t &agg, bool is_first, bool is_last,
                                  ParallelContext *context) const {
  // Deal with 'memory allocations' as per monoid type requested
  switch (agg.getMonoid()) {
    case SUM:
    case MULTIPLY:
    case MAX:
    case OR:
    case AND: {
      Type *t = agg.getExpressionType()->getLLVMType(context->getLLVMContext());

      return context->appendStateVar(
          PointerType::getUnqual(t),

          [=](llvm::Value *) {
            IRBuilder<> *Builder = context->getBuilder();

            Value *mem_acc = context->allocateStateVar(t);

            Constant *val_id = getIdentityElementIfSimple(
                agg.getMonoid(), agg.getExpressionType(), context);

            // FIXME: Assumes that we val_id is a byte to be repeated, not so
            // general... needs a memset to store...
            // Builder->CreateStore(val_id, mem_acc);

            // Even for floating points, 00000000 = 0.0, so cast to integer type
            // of same length to avoid problems with initialization of floats
            Value *val = Builder->CreateBitCast(
                val_id, Type::getIntNTy(context->getLLVMContext(),
                                        context->getSizeOf(val_id) * 8));
            context->CodegenMemset(mem_acc, val,
                                   (t->getPrimitiveSizeInBits() + 7) / 8);

            return mem_acc;
          },

          [=](llvm::Value *, llvm::Value *s) {
            if (is_first) {
              auto itAcc = aggs.begin();
              auto itMem = mem_accumulators.begin();

              vector<Value *> args;
              for (; itAcc != aggs.end(); itAcc++, itMem++) {
                auto acc = *itAcc;

                if (*itMem == StateVar{} || acc.getMonoid() == BAGUNION)
                  continue;

                args.emplace_back(context->getStateVar(*itMem));
              }

              IRBuilder<> *Builder = context->getBuilder();

              Function *f = context->getFunction("subpipeline_consume");
              FunctionType *f_t = f->getFunctionType();

              Type *substate_t = f_t->getParamType(f_t->getNumParams() - 1);

              Value *substate =
                  Builder->CreateBitCast(context->getSubStateVar(), substate_t);
              args.emplace_back(substate);

              Builder->CreateCall(f, args);
            }

            context->deallocateStateVar(s);
          });
    }
    case UNION: {
      string error_msg = string("[Reduce: ] Not implemented yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    case BAGUNION: {
      Type *t = Type::getInt64Ty(context->getLLVMContext());

      return context->appendStateVar(
          PointerType::getUnqual(t),

          [=](llvm::Value *) {
            Value *m = context->allocateStateVar(t);
            IRBuilder<> *Builder = context->getBuilder();
            Builder->CreateStore(context->createInt64(0), m);
            return m;
          },

          [=](llvm::Value *, llvm::Value *s) {
            context->deallocateStateVar(s);
          });
    }
    case APPEND: {
      /*XXX Bags and Lists can be processed in streaming fashion -> No
       * accumulator needed */
      break;
    }
    default: {
      string error_msg = string("[Reduce: ] Unknown accumulator");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }

  return {};
}

}  // namespace opt
