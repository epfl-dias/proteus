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

#include "operators/mem-broadcast-device.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include "multigpu/buffer_manager.cuh"
#include "threadpool/threadpool.hpp"

struct buff_pair_brdcst{
    char * new_buff;
    char * to_release;
};

extern "C"{
void step_mmc_mem_move_broadcast_device(MemBroadcastDevice::MemMoveConf * mmc){
    if (!mmc->to_cpu) return;
    for (size_t i = 0 ; i < 16 ; ++i) mmc->targetbuffer[i] = NULL; //FIXME: can be much much more simple and optimal if codegen'ed
}

buff_pair_brdcst make_mem_move_broadcast_device(char * src, size_t bytes, int target_device, MemBroadcastDevice::MemMoveConf * mmc, bool disable_noop){
    const auto &topo = topology::getInstance();
    if (!(mmc->to_cpu)){
        int dev = topo.getGpuAddressed(src)->id;

        // assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to provide blocks of arbitary size
        if (!disable_noop && dev == target_device) return buff_pair_brdcst{src, NULL}; // block already in correct device
        // set_device_on_scope d(dev);

        // std::cout << target_device << std::endl;

        // if (dev >= 0) set_affinity_local_to_gpu(dev);
        assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to provide blocks of arbitary size
        char * buff = (char *) buffer_manager<int32_t>::h_get_buffer(target_device);

        assert(target_device >= 0);
        if (bytes > 0) buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm[target_device], false);

        return buff_pair_brdcst{buff, src};
    } else {
        const auto &gpus = topo.getGpus();
        uint32_t numa_id = gpus[target_device % gpus.size()].local_cpu;
        if (topo.getGpuAddressed(src)){
            char * buff = (char *) buffer_manager<int32_t>::get_buffer_numa(numa_id);
            assert(target_device >= 0);
            if (bytes > 0) buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm[target_device], false);
            
            return buff_pair_brdcst{buff, src};
        } else {
            int node = topo.getCpuNumaNodeAddressed(src)->id;

            int target_node = mmc->always_share ? 0 : numa_id;
            if (mmc->always_share || node == target_node) {
                if (!disable_noop) {
                    mmc->targetbuffer[target_node] = src;
                    return buff_pair_brdcst{src, NULL};
                }
                if (buffer_manager<int32_t>::share_host_buffer((int32_t *) src)) {
                    mmc->targetbuffer[target_node] = src;
                    return buff_pair_brdcst{src, src};
                }
            } else {
                char * dst = (char *) mmc->targetbuffer[target_node];
                if (dst) {
                    if (buffer_manager<int32_t>::share_host_buffer((int32_t *) dst)) {
                        mmc->targetbuffer[target_node] = dst;
                        return buff_pair_brdcst{dst, dst};
                    }
                }
            }

            char * buff = (char *) buffer_manager<int32_t>::get_buffer_numa(target_node);
            assert(target_device >= 0);
            if (bytes > 0) buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm[target_device], false);

            mmc->targetbuffer[target_node] = buff;
            return buff_pair_brdcst{buff, src};
        }
    }
}
}

void MemBroadcastDevice::produce() {
    LLVMContext & llvmContext   = context->getLLVMContext();
    Type * bool_type    = Type::getInt1Ty   (context->getLLVMContext());
    Type * int32_type   = Type::getInt32Ty  (context->getLLVMContext());
    Type * charPtrType  = Type::getInt8PtrTy(context->getLLVMContext());


    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    Type  * oidType             = pg->getOIDType()->getLLVMType(llvmContext);


    std::vector<Type *> tr_types;
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        tr_types.push_back(wantedFields[i]->getLLVMType(llvmContext));
        tr_types.push_back(wantedFields[i]->getLLVMType(llvmContext)); // old buffer, to be released
    }
    tr_types.push_back(int32_type);
    tr_types.push_back(oidType); //cnt
    tr_types.push_back(oidType); //oid

    data_type   = StructType::get(llvmContext, tr_types);

    RecordAttribute tupleCnt       {wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()}; //FIXME: OID type for blocks ?
    RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),  activeLoop, pg->getOIDType()}; 
    RecordAttribute tupleTarget    {wantedFields[0]->getRelationName(), "__broadcastTarget", new IntType()}; 

    // Generate catch code
    int p = context->appendParameter(PointerType::get(data_type, 0), true, true);
    context->setGlobalFunction();

    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * entryBB       = Builder->GetInsertBlock();
    Function    * F             = entryBB->getParent();

    BasicBlock *mainBB = BasicBlock::Create(llvmContext, "main", F);

    BasicBlock *endBB = BasicBlock::Create(llvmContext, "end", F);
    context->setEndingBlock(endBB);

    Builder->SetInsertPoint(entryBB);

    Value * params = Builder->CreateLoad(context->getArgument(p));
    
    map<RecordAttribute, RawValueMemory> variableBindings;

    Function * release = context->getFunction("release_buffer");

    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RawValueMemory mem_valWrapper;

        mem_valWrapper.mem    = context->CreateEntryBlockAlloca(F, wantedFields[i]->getAttrName() + "_ptr", wantedFields[i]->getOriginalType()->getLLVMType(llvmContext));
        mem_valWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

        Value * param = Builder->CreateExtractValue(params, 2 * i    );

        Value * src   = Builder->CreateExtractValue(params, 2 * i + 1);

        // BasicBlock *relBB = BasicBlock::Create(llvmContext, "rel", F);
        // BasicBlock *merBB = BasicBlock::Create(llvmContext, "mer", F);

        // Value * do_rel = Builder->CreateICmpEQ(param, src);
        // Builder->CreateCondBr(do_rel, merBB, relBB);

        // Builder->SetInsertPoint(relBB);

        Builder->CreateCall(release, Builder->CreateBitCast(src, charPtrType));

        // Builder->CreateBr(merBB);

        // Builder->SetInsertPoint(merBB);

        Builder->CreateStore(param, mem_valWrapper.mem);

        variableBindings[*(wantedFields[i])] = mem_valWrapper;
    }

    RawValueMemory mem_targetWrapper;
    mem_targetWrapper.mem    = context->CreateEntryBlockAlloca(F, "__broadcastTarget", tupleTarget.getLLVMType(llvmContext));
    mem_targetWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

    Value * target = Builder->CreateExtractValue(params, 2 * wantedFields.size() );
    Builder->CreateStore(target, mem_targetWrapper.mem);

    variableBindings[tupleTarget] = mem_targetWrapper;

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem    = context->CreateEntryBlockAlloca(F, "activeCnt", oidType);
    mem_cntWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

    Value * cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);
    Builder->CreateStore(cnt, mem_cntWrapper.mem);

    variableBindings[tupleCnt] = mem_cntWrapper;

    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem    = context->CreateEntryBlockAlloca(F, activeLoop, oidType);
    mem_oidWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

    Value * oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 2);
    Builder->CreateStore(oid, mem_oidWrapper.mem);

    variableBindings[tupleIdentifier] = mem_oidWrapper;

    context->setCurrentEntryBlock(Builder->GetInsertBlock());

    Builder->SetInsertPoint(mainBB);

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    Builder->CreateBr(endBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    // Insert an explicit fall through from the current (entry) block to the CondBB.
    Builder->CreateBr(mainBB);


    Builder->SetInsertPoint(context->getEndingBlock());
    // Builder->CreateRetVoid();


    context->popPipeline();

    catch_pip = context->removeLatestPipeline();

    //push new pipeline for the throw part
    context->pushPipeline();

    device_id_var       = context->appendStateVar(int32_type );
    // cu_stream_var       = context->appendStateVar(charPtrType);
    memmvconf_var       = context->appendStateVar(charPtrType);

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip);});
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip);});
    getChild()->produce();
}

void MemBroadcastDevice::consume(RawContext* const context, const OperatorState& childState) {
    //Prepare
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    Type * charPtrType          = Type::getInt8PtrTy(llvmContext);

    Type * cpp_bool_type        = Type::getInt8Ty(llvmContext);
    static_assert(sizeof(bool) == 1, "Fix type above");

    Type * workunit_type        = StructType::get(llvmContext, std::vector<Type *>{charPtrType, charPtrType});

    map<RecordAttribute, RawValueMemory> old_bindings{childState.getBindings()};
    
    // Find block size
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = old_bindings.find(tupleCnt);
    assert(it != old_bindings.end());

    RawValueMemory mem_cntWrapper = it->second;

    Function * make_mem_move = context->getFunction("make_mem_move_broadcast_device");
    
    Builder->SetInsertPoint(context->getCurrentEntryBlock());

    Value * device_id       = ((GpuRawContext *) context)->getStateVar(device_id_var);
    // Value * cu_stream       = ((GpuRawContext *) context)->getStateVar(cu_stream_var);

    Builder->SetInsertPoint(insBB);
    Value * N               = Builder->CreateLoad(mem_cntWrapper.mem);



    RecordAttribute tupleIdentifier = RecordAttribute(wantedFields[0]->getRelationName(),  activeLoop, pg->getOIDType()); 
    it = old_bindings.find(tupleIdentifier);
    assert(it != old_bindings.end());
    RawValueMemory mem_oidWrapper = it->second;
    Value * oid             = Builder->CreateLoad(mem_oidWrapper.mem);
    
    Value * memmv           = ((GpuRawContext *) context)->getStateVar(memmvconf_var);
    
    std::vector<std::vector<Value *>> pushed;

    for (const auto &t: targets) pushed.emplace_back();

    Value * null_ptr     = ConstantPointerNull::get((PointerType *) charPtrType);
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute block_attr(*(wantedFields[i]), true);

        auto it = old_bindings.find(block_attr);
        assert(it != old_bindings.end());
        RawValueMemory mem_valWrapper = it->second;
        
        Value * block           = Builder->CreateLoad(mem_valWrapper.mem);
        Value * mv              = Builder->CreateBitCast(
                                                            block,
                                                            charPtrType
                                                        );

        Type  * block_type      = mem_valWrapper.mem->getType()->getPointerElementType();
        Type  * mv_block_type   = block_type->getPointerElementType();

        Value * size            = ConstantInt::get(llvmContext, APInt(64, context->getSizeOf(mv_block_type)));
        Value * Nloc            = Builder->CreateZExtOrBitCast(N, size->getType());
        size                    = Builder->CreateMul(size, Nloc);

        Function *step_mmc = context->getFunction("step_mmc_mem_move_broadcast_device");
        Builder->CreateCall(step_mmc, std::vector<Value *>{memmv});

        Value * any_noop        = context->createFalse();
        for (size_t t_i = 0 ; t_i < targets.size() ; ++t_i){
            Value * target_id = context->createInt32(targets[t_i]);

            vector<Value *> mv_args{mv, size, target_id, memmv, any_noop}; //Builder->CreateZExtOrBitCast(any_noop, cpp_bool_type)};

            // Do actual mem move
            // Builder->CreateZExtOrBitCast(any_noop, cpp_bool_type)->getType()->dump();
            Value * moved_buffpair  = Builder->CreateCall(make_mem_move, mv_args);
            Value * moved           = Builder->CreateExtractValue(moved_buffpair, 0);
            Value * to_release      = Builder->CreateExtractValue(moved_buffpair, 1);

            pushed[t_i].push_back(Builder->CreateBitCast(moved, block_type));

            any_noop = Builder->CreateOr(any_noop, Builder->CreateICmpEQ(to_release, null_ptr));

            Value * null_block = ConstantPointerNull::get((PointerType *) block_type);

            if (t_i == targets.size() - 1){
                Value * rel = Builder->CreateSelect(any_noop, null_block, block);

                pushed[t_i].push_back(rel);
            } else {
                pushed[t_i].push_back(null_block);
            }
        }

    }
    
    for (size_t t_i = 0 ; t_i < targets.size() ; ++t_i){
        pushed[t_i].push_back(context->createInt32(t_i));
        pushed[t_i].push_back(N);
        pushed[t_i].push_back(oid);
    }

    for (size_t t_i = 0 ; t_i < targets.size() ; ++t_i){
        Value * d           = UndefValue::get(data_type);
        for (size_t i = 0 ; i < pushed[t_i].size() ; ++i){
            d               = Builder->CreateInsertValue(d, pushed[t_i][i], i);
        }

        Function * acquire      = context->getFunction("acquireWorkUnitBroadcast");

        // acquire->getFunctionType()->dump();
        Value * workunit_ptr8   = Builder->CreateCall(acquire, memmv);
        Value * workunit_ptr    = Builder->CreateBitCast(workunit_ptr8, PointerType::getUnqual(workunit_type));

        Value * workunit_dat    = Builder->CreateLoad(workunit_ptr);
        Value * d_ptr           = Builder->CreateExtractValue(workunit_dat, 0);
        d_ptr                   = Builder->CreateBitCast(d_ptr       , PointerType::getUnqual(data_type    ));
        Builder->CreateStore(d, d_ptr);

        Value * target_id = context->createInt32(targets[t_i]);

        Function * propagate = context->getFunction("propagateWorkUnitBroadcast");
        // propagate->getFunctionType()->dump();
        Builder->CreateCall(propagate, std::vector<Value *>{memmv, workunit_ptr8, target_id});
    }
}

void MemBroadcastDevice::open (RawPipeline * pip){
    std::cout << "MemBroadcastDevice:open" << std::endl;
    nvtxRangePushA("memmove::open");
    // cudaStream_t strm2;
    // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));
    
    MemMoveConf * mmc   = new MemMoveConf;
#ifndef NCUDA
    for (const auto &t: targets) {
        cudaStream_t strm;
        gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
        mmc->strm[t] = strm;
    }
    // mmc->strm2          = strm2;
#endif

    mmc->num_of_targets = targets.size();
    mmc->to_cpu         = to_cpu;
    mmc->always_share   = always_share;
    // mmc->slack          = slack;
    // mmc->next_e         = 0;
    // // mmc->events         = new cudaEvent_t[slack];
    // mmc->old_buffs      = new void      *[slack];

    workunit * wu = new workunit[slack];
    size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t) 0xF);
    // void * data_buff = malloc(data_size * slack);
    nvtxRangePushA("memmove::open2");
    for (size_t i = 0 ; i < slack ; ++i){
        wu[i].data  = malloc(data_size);//((void *) (((char *) data_buff) + i * data_size));
        gpu_run(cudaEventCreateWithFlags(&(wu[i].event), cudaEventDisableTiming  | cudaEventBlockingSync));
// // gpu_run(cudaEventCreateWithFlags(&(wu[i].event), cudaEventDisableTiming));//  | cudaEventBlockingSync));
//         gpu_run(cudaEventCreate(&(wu[i].event)));
//         gpu_run(cudaStreamCreate(&(wu[i].strm)));

        mmc->idle.push(wu + i);

        // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming | cudaEventBlockingSync));
        // gpu_run(cudaEventCreate(mmc->events + i));
        // mmc->old_buffs[i] = NULL;
    }
    nvtxRangePop();

    mmc->worker = ThreadPool::getInstance().enqueue(&MemBroadcastDevice::catcher, this, mmc, pip->getGroup(), exec_location{});

    int device = -1;
    if (!to_cpu) device = topology::getInstance().getActiveGpu().id;
    pip->setStateVar<int         >(device_id_var, device);

    // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
    pip->setStateVar<void      * >(memmvconf_var, mmc   );
    nvtxRangePop();
}

void MemBroadcastDevice::close(RawPipeline * pip){
    std::cout << "MemBroadcastDevice:close" << std::endl;
    // int device = get_device();
    // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
    MemMoveConf * mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

    mmc->tran.close();
    std::cout << "MemBroadcastDevice:close3" << std::endl;

    nvtxRangePop();
    mmc->worker.get();

    // gpu_run(cudaStreamSynchronize(g_strm));

    // int32_t h_s;
    // gpu_run(cudaMemcpy(&h_s, s, sizeof(int32_t), cudaMemcpyDefault));
    // std::cout << "rrr" << h_s << std::endl;

    // RawMemoryManager::freeGpu(s);
    std::cout << "MemBroadcastDevice:close4" << std::endl;

    if (!always_share){
        for (const auto &t: targets) {
            gpu_run(cudaStreamSynchronize(mmc->strm[t]));
            gpu_run(cudaStreamDestroy    (mmc->strm[t]));
        }
    } else {
        gpu_run(cudaStreamSynchronize(mmc->strm[0]));
    }

    std::cout << "MemBroadcastDevice:close2" << std::endl;
    // gpu_run(cudaStreamSynchronize(mmc->strm2));
    // gpu_run(cudaStreamDestroy    (mmc->strm2));

    nvtxRangePushA("MemMoveDev_running2");
    nvtxRangePushA("MemMoveDev_running");

    nvtxRangePushA("MemMoveDev_release");
    workunit * start_wu;
    // void     * start_wu_data;
    for (size_t i = 0 ; i < slack ; ++i){
        workunit * wu = mmc->idle.pop_unsafe();

        // if (mmc->old_buffs[i]) buffer_manager<int32_t>::release_buffer((int32_t *) mmc->old_buffs[i]);

        gpu_run(cudaEventDestroy(wu->event     ));
        // gpu_run(cudaEventDestroy(mmc->events[i]));
        free(wu->data);

        if (i == 0 || wu       < start_wu     ) start_wu      = wu;
        // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
    }
    nvtxRangePop();
    nvtxRangePop();
    // assert(mmc->tran.empty_unsafe());
    // assert(mmc->idle.empty_unsafe());
    // free(start_wu_data);
    delete[] start_wu;
    // delete[] mmc->events   ;
    // delete[] mmc->old_buffs;

    mmc->idle.close();

    // delete mmc->worker;
    delete mmc;
}


extern "C"{
    MemBroadcastDevice::workunit * acquireWorkUnitBroadcast(MemBroadcastDevice::MemMoveConf * mmc){
        MemBroadcastDevice::workunit * ret = nullptr;
#ifndef NDEBUG
        bool popres = 
#endif
        mmc->idle.pop(ret);
        assert(popres);
        return ret;
    }

    void propagateWorkUnitBroadcast(MemBroadcastDevice::MemMoveConf * mmc, MemBroadcastDevice::workunit * buff, int target_device){
        gpu_run(cudaEventRecord(buff->event, mmc->strm[target_device]));

        mmc->tran.push(buff);
    }


    bool acquirePendingWorkUnitBroadcast(MemBroadcastDevice::MemMoveConf * mmc, MemBroadcastDevice::workunit ** ret){
        if (!mmc->tran.pop(*ret)) return false;
        gpu_run(cudaEventSynchronize((*ret)->event));
        return true;
    }

    void releaseWorkUnitBroadcast(MemBroadcastDevice::MemMoveConf * mmc, MemBroadcastDevice::workunit * buff){
        mmc->idle.push(buff);
    }
}

void MemBroadcastDevice::catcher(MemMoveConf * mmc, int group_id, const exec_location &target_dev){
    set_exec_location_on_scope d(target_dev);

    nvtxRangePushA("memmove::catch");

    RawPipeline * pip = catch_pip->getPipeline(group_id);

    nvtxRangePushA("memmove::catch_open");
    pip->open();
    nvtxRangePop();
    {
        do {
            MemBroadcastDevice::workunit * p = nullptr;
            if (!acquirePendingWorkUnitBroadcast(mmc, &p)) break;

            nvtxRangePushA("memmove::catch_cons");
            pip->consume(0, p->data);
            nvtxRangePop();

            releaseWorkUnitBroadcast(mmc, p); //FIXME: move this inside the generated code
        } while (true);
    }

    nvtxRangePushA("memmove::catch_close");
    pip->close();
    nvtxRangePop();

    nvtxRangePop();
}
