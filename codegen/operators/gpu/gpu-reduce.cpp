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
}

void GpuReduce::produce() {
    for (const auto &expr: outputExprs){
        if (!expr->getExpressionType()->isPrimitive()){
            string error_msg("[GpuReduce: ] Currently only supports primitive types");
            LOG(ERROR) << error_msg;
            throw runtime_error(error_msg);
        }

        Type * t = PointerType::get(((const PrimitiveType *) expr->getExpressionType())->getLLVMType(context->getLLVMContext()), /* address space */ 1);
        out_ids.push_back(((GpuRawContext *) context)->appendStateVar(t));//, true, false));
    }

    getChild()->produce();
}

void GpuReduce::consume(RawContext* const context, const OperatorState& childState) {
    Reduce::consume(context, childState);
    generate(context, childState);
}

void GpuReduce::generate(RawContext* const context, const OperatorState& childState) const{
    vector<Monoid                   >::const_iterator itAcc    ;
    vector<expressions::Expression *>::const_iterator itExpr   ;
    vector<AllocaInst *             >::const_iterator itMem    ;
    vector<int                      >::const_iterator itID;
    /* Time to Compute Aggs */
    itAcc       = accs.begin();
    itExpr      = outputExprs.begin();
    itMem       = mem_accumulators.begin();
    itID        = out_ids.begin();

    IRBuilder<>* Builder        = context->getBuilder();
    
    for (; itAcc != accs.end(); itAcc++, itExpr++, itMem++, ++itID) {
        Monoid                      acc                     = *itAcc;
        expressions::Expression *   outputExpr              = *itExpr;
        AllocaInst *                mem_accumulating        = *itMem;

        BasicBlock* insBlock = Builder->GetInsertBlock();
        
        BasicBlock* entryBlock = context->getCurrentEntryBlock();
        Builder->SetInsertPoint(entryBlock);

        Value * global_acc_ptr = ((const GpuRawContext *) context)->getStateVar(*itID);

        Builder->SetInsertPoint(insBlock);

        switch (acc) {
        case MAX:
        case SUM:
        case OR:
        case AND:
            generate(acc, outputExpr, (GpuRawContext * const) context, childState, mem_accumulating, global_acc_ptr);
            break;
        case MULTIPLY:
        case BAGUNION:
        case APPEND:
        case UNION:
        default: {
            string error_msg = string(
                    "[GpuReduce: ] Unknown / Still Unsupported accumulator");
            LOG(ERROR)<< error_msg;
            throw runtime_error(error_msg);
        }
        }
    }

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
}

void GpuReduce::generate(const Monoid &m, expressions::Expression* outputExpr,
        GpuRawContext* const context, const OperatorState& state,
        AllocaInst *mem_accumulating, Value *global_accumulator_ptr) const {

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

void GpuReduce::open(RawPipeline * pip) const{
    for (size_t i = 0 ; i < out_ids.size() ; ++i){
        Type * llvm_type = ((const PrimitiveType *) outputExprs[i]->getExpressionType())->getLLVMType(context->getLLVMContext());

        size_t size_in_bytes = (llvm_type->getPrimitiveSizeInBits() + 7)/8;

        void * acc;
        gpu_run(cudaMalloc(&acc,    size_in_bytes));

        gpu_run(cudaMemset( acc, 0, size_in_bytes)); //FIXME: reset every type of (data, monoid)

        pip->setStateVar(out_ids[i], acc);
    }
}

void GpuReduce::close(RawPipeline * pip) const{
    // for (size_t i = 0 ; i < out_ids.size() ; ++i){
    //     gpu_run(cudaFree(pip->getStateVar<uint32_t *>(context, out_ids[i])));
    // }
    for (size_t i = 0 ; i < out_ids.size() ; ++i){
        uint32_t r;
        gpu_run(cudaMemcpy(&r, pip->getStateVar<uint32_t *>(out_ids[i]), sizeof(uint32_t), cudaMemcpyDefault));
        std::cout << r << std::endl;
    }
}



}


