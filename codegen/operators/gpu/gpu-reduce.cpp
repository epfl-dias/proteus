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

#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gmonoids.hpp"

namespace opt {

GpuReduce::GpuReduce(vector<Monoid>             accs,
            vector<expressions::Expression*>    outputExprs,
            expressions::Expression*            pred, 
            RawOperator* const                  child,
            GpuRawContext*                      context)
        : Reduce(accs, outputExprs, pred, child, context, false) {
    for (const auto &expr: outputExprs){
        if (!expr->getExpressionType()->isPrimitive()){
            string error_msg("[GpuReduce: ] Currently only supports primitive types");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
        }
    }
}

void GpuReduce::produce() {
    flushResults = flushResults && !getParent(); //TODO: is this the best way to do it ?
    generate_flush();

    ((GpuRawContext *) context)->popPipeline();

    auto flush_pip = ((GpuRawContext *) context)->removeLatestPipeline();
    flush_fun = flush_pip->getKernel();

    ((GpuRawContext *) context)->pushPipeline(flush_pip);

    assert(mem_accumulators.empty());
    if (mem_accumulators.empty()){
        vector<Monoid>::const_iterator itAcc;
        vector<expressions::Expression*>::const_iterator itExpr;
        itAcc = accs.begin();
        itExpr = outputExprs.begin();

        int aggsNo = accs.size();
        /* Prepare accumulator FOREACH outputExpr */
        for (; itAcc != accs.end(); itAcc++, itExpr++) {
            Monoid acc = *itAcc;
            expressions::Expression *outputExpr = *itExpr;
            bool flushDelim = (aggsNo > 1) && (itAcc != accs.end() - 1);
            bool is_first   = (itAcc == accs.begin()  );
            bool is_last    = (itAcc == accs.end() - 1);
            size_t mem_accumulator = resetAccumulator(outputExpr, acc, flushDelim, is_first, is_last);
            mem_accumulators.push_back(mem_accumulator);
        }
    }

    getChild()->produce();
}

void GpuReduce::consume(RawContext* const context, const OperatorState& childState) {
    consume((GpuRawContext *) context, childState);
}

void GpuReduce::consume(GpuRawContext* const context, const OperatorState& childState) {
    IRBuilder<>* Builder = context->getBuilder();
    LLVMContext& llvmContext = context->getLLVMContext();
    Function *TheFunction = Builder->GetInsertBlock()->getParent();

    int aggsNo = accs.size();

    //Generate condition
    ExpressionGeneratorVisitor predExprGenerator{context, childState};
    RawValue condition = pred->accept(predExprGenerator);
    /**
     * Predicate Evaluation:
     */
    BasicBlock* entryBlock = Builder->GetInsertBlock();
    BasicBlock *endBlock = BasicBlock::Create(llvmContext, "reduceCondEnd",
            TheFunction);
    BasicBlock *ifBlock;
    context->CreateIfBlock(context->getGlobalFunction(), "reduceIfCond",
            &ifBlock, endBlock);

    /**
     * IF(pred) Block
     */
    RawValue val_output;
    Builder->SetInsertPoint(entryBlock);

    Builder->CreateCondBr(condition.value, ifBlock, endBlock);

    Builder->SetInsertPoint(ifBlock);

    vector<Monoid                   >::const_iterator itAcc ;
    vector<expressions::Expression *>::const_iterator itExpr;
    vector<size_t                   >::const_iterator itMem ;
    /* Time to Compute Aggs */
    itAcc  = accs.begin();
    itExpr = outputExprs.begin();
    itMem  = mem_accumulators.begin();

    for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
        Monoid                      acc                 = *itAcc    ;
        expressions::Expression   * outputExpr          = *itExpr   ;
        Value                     * mem_accumulating    = NULL      ;

        switch (acc) {
        case SUM:
        case MULTIPLY:
        case MAX:
        case OR:
        case AND:{
            BasicBlock *cBB = Builder->GetInsertBlock();
            Builder->SetInsertPoint(context->getCurrentEntryBlock());

            mem_accumulating = context->getStateVar(*itMem);

            Constant * acc_init = getIdentityElementIfSimple(
                acc,
                outputExpr->getExpressionType(),
                context
            );
            Value * acc_mem  = context->CreateEntryBlockAlloca("acc", acc_init->getType());
            Builder->CreateStore(acc_init, acc_mem);

            Builder->SetInsertPoint(context->getEndingBlock());
            generate(acc, outputExpr, context, childState, acc_mem, mem_accumulating);

            Builder->SetInsertPoint(cBB);

            ExpressionGeneratorVisitor outputExprGenerator{context, childState};

            // Load accumulator -> acc_value
            RawValue acc_value;
            acc_value.value  = Builder->CreateLoad(acc_mem);
            acc_value.isNull = context->createFalse();

            // new_value = acc_value op outputExpr
            expressions::Expression * val = new expressions::RawValueExpression(outputExpr->getExpressionType(), acc_value);
            expressions::Expression * upd = toExpression(acc, val, outputExpr);
            assert(upd && "Monoid is not convertible to expression!");
            RawValue new_val = upd->accept(outputExprGenerator);

            // store new_val to accumulator
            Builder->CreateStore(new_val.value, acc_mem);
            break;
        }
        case BAGUNION:
        case APPEND:
            //      generateAppend(context, childState);
            //      break;
        case UNION:
        default: {
            string error_msg = string(
                    "[Reduce: ] Unknown / Still Unsupported accumulator");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        }
    }

    Builder->CreateBr(endBlock);

    /**
     * END Block
     */
    Builder->SetInsertPoint(endBlock);


    // ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    // ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});

}





// void GpuReduce::consume(RawContext* const context, const OperatorState& childState) {
//     Reduce::consume(context, childState);
//     generate(context, childState);
// }

// void GpuReduce::generate(RawContext* const context, const OperatorState& childState) const{
//     vector<Monoid                   >::const_iterator itAcc    ;
//     vector<expressions::Expression *>::const_iterator itExpr   ;
//     vector<size_t                   >::const_iterator itMem    ;
//     vector<int                      >::const_iterator itID;
//     /* Time to Compute Aggs */
//     itAcc       = accs.begin();
//     itExpr      = outputExprs.begin();
//     itMem       = mem_accumulators.begin();
//     itID        = out_ids.begin();

//     IRBuilder<>* Builder        = context->getBuilder();
    
//     for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++, ++itID) {
//         Monoid                      acc                     = *itAcc ;
//         expressions::Expression *   outputExpr              = *itExpr;
//         Value                   *   mem_accumulating        = NULL   ;

//         BasicBlock* insBlock = Builder->GetInsertBlock();
        
//         BasicBlock* entryBlock = context->getCurrentEntryBlock();
//         Builder->SetInsertPoint(entryBlock);

//         Value * global_acc_ptr = ((const GpuRawContext *) context)->getStateVar(*itID);

//         Builder->SetInsertPoint(insBlock);

//         switch (acc) {
//         case MAX:
//         case SUM:
//         case OR:
//         case AND:
//             mem_accumulating = context->getStateVar(*itMem);
//             generate(acc, outputExpr, (GpuRawContext * const) context, childState, mem_accumulating, global_acc_ptr);
//             break;
//         case MULTIPLY:
//         case BAGUNION:
//         case APPEND:
//         case UNION:
//         default: {
//             string error_msg = string(
//                     "[GpuReduce: ] Unknown / Still Unsupported accumulator");
//             LOG(ERROR)<< error_msg;
//             throw runtime_error(error_msg);
//         }
//         }
//     }

//     ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
//     ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
// }

void GpuReduce::generate(const Monoid &m, expressions::Expression* outputExpr,
        GpuRawContext* const context, const OperatorState& state,
        Value * mem_accumulating, Value *global_accumulator_ptr) const {

    IRBuilder<>* Builder        = context->getBuilder();
    LLVMContext& llvmContext    = context->getLLVMContext();
    Function *TheFunction       = Builder->GetInsertBlock()->getParent();

    gpu::Monoid * gm = gpu::Monoid::get(m);



    global_accumulator_ptr->setName("reduce_" + std::to_string(*gm) + "_ptr");

    BasicBlock* entryBlock = Builder->GetInsertBlock();
    
    BasicBlock* endBlock = context->getEndingBlock();
    Builder->SetInsertPoint(endBlock);

    Value* val_accumulating = Builder->CreateLoad(mem_accumulating);

    // Warp aggregate
    Value * aggr = gm->createWarpAggregateTo0(context, val_accumulating);

    // Store warp-aggregated final result (available to all threads of each warp)
    Builder->CreateStore(aggr, mem_accumulating);

    // Write to global accumulator only from a single thread per warp

    BasicBlock * laneendBlock   = BasicBlock::Create(llvmContext, "reduceWriteEnd", TheFunction);
    BasicBlock * laneifBlock    = context->CreateIfBlock(TheFunction,
                                                            "reduceWriteIf",
                                                            laneendBlock);

    Value * laneid              = context->laneId();
    Builder->CreateCondBr(Builder->CreateICmpEQ(laneid, ConstantInt::get(laneid->getType(), 0), "is_pivot"), laneifBlock, laneendBlock);

    Builder->SetInsertPoint(laneifBlock);

    gm->createAtomicUpdate(context, global_accumulator_ptr, aggr, llvm::AtomicOrdering::Monotonic);

    Builder->CreateBr(laneendBlock);
    context->setEndingBlock(laneendBlock);

    Builder->SetInsertPoint(entryBlock);
}

// void GpuReduce::open(RawPipeline * pip) const{
//     std::cout << "GpuReduce:open" << std::endl;
//     // cudaStream_t strm;
//     // gpu_run(cudaStreamCreate(&strm));
//     // for (size_t i = 0 ; i < mem_accumulators.size() ; ++i){
//     //     Type * llvm_type = ((const PrimitiveType *) outputExprs[i]->getExpressionType())->getLLVMType(context->getLLVMContext());

//     //     size_t size_in_bytes = (llvm_type->getPrimitiveSizeInBits() + 7)/8;

//     //     void * acc = pip->getStateVar<void *>(mem_accumulators[i]);
//     //     gpu_run(cudaMemsetAsync( acc, 0, size_in_bytes, strm)); //FIXME: reset every type of (data, monoid)

//     //     // pip->setStateVar(mem_accumulators[i], acc);s
//     // }
//     // gpu_run(cudaStreamSynchronize(strm));
//     // gpu_run(cudaStreamDestroy(strm));
// }

// void GpuReduce::close(RawPipeline * pip) const{
//     std::cout << "GpuReduce:close" << std::endl;
//     // for (size_t i = 0 ; i < mem_accumulators.size() ; ++i){
//     //     gpu_run(cudaFree(pip->getStateVar<uint32_t *>(context, mem_accumulators[i])));
//     // }

//     //create stream
//     //call consume
//     //sync

//     // for (size_t i = 0 ; i < mem_accumulators.size() ; ++i){
//     //     int32_t r; //NOTE: here we are assuming 32bits unsigned integer output, change for correct display!
//     //     gpu_run(cudaMemcpy(&r, pip->getStateVar<void *>(mem_accumulators[i]), sizeof(uint32_t), cudaMemcpyDefault));
//     //     std::cout << r << " " << pip->getStateVar<void *>(mem_accumulators[i]) << std::endl;
//     // }
// }

size_t GpuReduce::resetAccumulator(expressions::Expression* outputExpr,
        Monoid acc, bool flushDelim, bool is_first, bool is_last) const {
    size_t mem_accum_id = ~((size_t) 0);

    //Deal with 'memory allocations' as per monoid type requested
    switch (acc) {
        case SUM:
        case MULTIPLY:
        case MAX:
        case OR:
        case AND: {
            Type * t = outputExpr->getExpressionType()
                                    ->getLLVMType(context->getLLVMContext());

            mem_accum_id = context->appendStateVar(
                PointerType::getUnqual(t),

                [=](llvm::Value *){
                    IRBuilder<> * Builder = context->getBuilder();

                    Value * mem_acc = context->allocateStateVar(t);

                    Constant * val_id = getIdentityElementIfSimple(
                        acc,
                        outputExpr->getExpressionType(),
                        context
                    );

                    // FIXME: Assumes that we val_id is a byte to be repeated, not so general...
                    // needs a memset to store...
                    // Builder->CreateStore(val_id, mem_acc);

                    // Even for floating points, 00000000 = 0.0, so cast to integer type of same length to avoid problems with initialization of floats
                    Value * val = Builder->CreateBitCast(val_id, Type::getIntNTy(context->getLLVMContext(), context->getSizeOf(val_id) * 8));
                    context->CodegenMemset(mem_acc, val, (t->getPrimitiveSizeInBits() + 7) / 8);

                    return mem_acc;
                },

                [=](llvm::Value *, llvm::Value * s){
                    // if (flushResults && is_first && accs.size() > 1) flusher->beginList();

                    // Value* val_acc =  context->getBuilder()->CreateLoad(s);

                    // if (outputExpr->isRegistered()){
                    //  map<RecordAttribute, RawValueMemory> binding{};
                    //  AllocaInst * acc_alloca = context->CreateEntryBlockAlloca(outputExpr->getRegisteredAttrName(), val_acc->getType());
                    //  context->getBuilder()->CreateStore(val_acc, acc_alloca);
                    //  RawValueMemory acc_mem{acc_alloca, context->createFalse()};
                    //  binding[outputExpr->getRegisteredAs()] = acc_mem;
                    // }

                    if (is_first){  
                        vector<Monoid                   >::const_iterator itAcc  = accs.begin();
                        vector<expressions::Expression *>::const_iterator itExpr = outputExprs.begin();
                        vector<size_t                   >::const_iterator itMem  = mem_accumulators.begin();

                        vector<Value *> args;
                        for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++) {
                            Monoid                      acc                 = *itAcc    ;
                            expressions::Expression   * outputExpr          = *itExpr   ;
                            Value                     * mem_accumulating    = NULL      ;

                            if (*itMem == ~((size_t) 0)) continue;
                            
                            args.emplace_back(context->getStateVar(*itMem));
                        }

                        IRBuilder<> * Builder = context->getBuilder();

                        Type  * charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

                        Function * f        = context->getFunction("subpipeline_consume");
                        FunctionType * f_t  = f->getFunctionType();

                        Type  * substate_t  = f_t->getParamType(f_t->getNumParams()-1);

                        Value * substate    = Builder->CreateBitCast(((GpuRawContext *) context)->getSubStateVar(), substate_t);
                        args.emplace_back(substate);

                        Builder->CreateCall(f, args);
                    }

                    // if (flushResults){
                    //  flusher->flushValue(val_acc, outputExpr->getExpressionType()->getTypeID());
                    //  if (flushDelim) flusher->flushDelim();
                    // }

                    context->deallocateStateVar(s);

                    // if (flushResults && is_last  && accs.size() > 1) flusher->endList();

                    // if (flushResults && is_last  ) flusher->flushOutput();
                }
            );
            break;
        }
        case UNION: {
            string error_msg = string("[Reduce: ] Not implemented yet");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        case BAGUNION:
        case APPEND: {
            /*XXX Bags and Lists can be processed in streaming fashion -> No accumulator needed */
            break;
        }
        default: {
            string error_msg = string("[Reduce: ] Unknown accumulator");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
    }

    return mem_accum_id;
}




}


