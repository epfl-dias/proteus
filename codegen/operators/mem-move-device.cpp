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

#include "operators/mem-move-device.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include "multigpu/buffer_manager.cuh"
#include "util/raw-memory-manager.hpp"
#include "threadpool/threadpool.hpp"

struct buff_pair{
    char * new_buff;
    char * old_buff;
};

extern "C"{
buff_pair make_mem_move_device(char * src, size_t bytes, int target_device, MemMoveDevice::MemMoveConf * mmc){
    const auto *d = topology::getInstance().getGpuAddressed(src);
    int dev = d ? d->id : -1;

    if (dev == target_device) return buff_pair{src, src}; // block already in correct device

    // set_device_on_scope d(dev);

    // if (dev >= 0) set_affinity_local_to_gpu(dev);

    assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to provide blocks of arbitary size
    // std::cout << "MemMoveTarget: " << target_device << std::endl;
    char * buff = (char *) buffer_manager<int32_t>::h_get_buffer(target_device);
    
    // int numa_target = numa_node_of_gpu(target_device);
    // if (dev >= 0 && (numa_node_of_gpu(dev) != numa_target)){
    //     set_device_on_scope d(dev);

    //     if (dev >= 0) set_affinity_local_to_gpu(dev);

    //     size_t curr_e  = mmc->next_e;
    //     cudaEvent_t e  = mmc->events   [curr_e];
    //     void * old_ptr = mmc->old_buffs[curr_e];
    //     // mmc->old_buffs[curr_e] = NULL;

    //     if (old_ptr) buffer_manager<int32_t>::release_buffer((int32_t *) old_ptr); //FIXME: cannot release it yet!
    //     gpu_run(cudaEventSynchronize(e));
    //     mmc->next_e = (curr_e + 1) % mmc->slack;

    //     char * interbuff = (char *) buffer_manager<int32_t>::get_buffer_numa(numa_target);
    //     mmc->old_buffs[curr_e] = src;

    //     buffer_manager<int32_t>::overwrite_bytes(interbuff, src, bytes, mmc->strm2, false);
    //     gpu_run(cudaEventRecord(e, mmc->strm2));
    //     src = interbuff;
    
    //     gpu_run(cudaStreamWaitEvent(mmc->strm, e, 0));
    // }

    if (bytes > 0) buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm, false);
    // assert(bytes == sizeof(int32_t) * h_vector_size);
    // std::cout << bytes << " " << sizeof(int32_t) * h_vector_size << std::endl;
            // cudaStream_t strm;
            // gpu_run(cudaStreamCreate(&(wu->strm)));
            // gpu_run(cudaMemcpyAsync(buff, src, bytes, cudaMemcpyDefault, wu->strm));
            // gpu_run(cudaMemcpyAsync(buff2, buff, bytes, cudaMemcpyDefault, wu->strm));
            // std::cout << "alloc" << (void *) buff2 << std::endl;

    // gpu_run(cudaMemcpy(buff, src, bytes, cudaMemcpyDefault));
    // buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, wu->strm, false);
    // buffer_manager<int32_t>::overwrite_bytes(buff2, buff, bytes, wu->strm, false);
    // gpu_run(cudaStreamSynchronize(mmc->strm));
    // gpu_run(cudaStreamSynchronize(wu->strm));
    // buffer_manager<int32_t>::release_buffer ((int32_t *) src                             );

    return buff_pair{buff, src};
}
}

void MemMoveDevice::produce() {
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
    tr_types.push_back(oidType); //cnt
    tr_types.push_back(oidType); //oid

    data_type   = StructType::get(llvmContext, tr_types);

    RecordAttribute tupleCnt        = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?
    RecordAttribute tupleIdentifier = RecordAttribute(wantedFields[0]->getRelationName(),  activeLoop, pg->getOIDType()); 

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

        BasicBlock *relBB = BasicBlock::Create(llvmContext, "rel", F);
        BasicBlock *merBB = BasicBlock::Create(llvmContext, "mer", F);

        Value * do_rel = Builder->CreateICmpEQ(param, src);
        Builder->CreateCondBr(do_rel, merBB, relBB);

        Builder->SetInsertPoint(relBB);

        Builder->CreateCall(release, Builder->CreateBitCast(src, charPtrType));

        Builder->CreateBr(merBB);

        Builder->SetInsertPoint(merBB);

        Builder->CreateStore(param, mem_valWrapper.mem);

        variableBindings[*(wantedFields[i])] = mem_valWrapper;
    }

    RawValueMemory mem_cntWrapper;
    mem_cntWrapper.mem    = context->CreateEntryBlockAlloca(F, "activeCnt", oidType);
    mem_cntWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

    Value * cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size()    );
    Builder->CreateStore(cnt, mem_cntWrapper.mem);

    variableBindings[tupleCnt] = mem_cntWrapper;


    RawValueMemory mem_oidWrapper;
    mem_oidWrapper.mem    = context->CreateEntryBlockAlloca(F, activeLoop, oidType);
    mem_oidWrapper.isNull = context->createFalse(); //FIMXE: should we alse transfer this information ?

    Value * oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);
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

void MemMoveDevice::consume(RawContext* const context, const OperatorState& childState) {
    //Prepare
    LLVMContext & llvmContext   = context->getLLVMContext();
    IRBuilder<> * Builder       = context->getBuilder    ();
    BasicBlock  * insBB         = Builder->GetInsertBlock();
    Function    * F             = insBB->getParent();

    Type * charPtrType          = Type::getInt8PtrTy(context->getLLVMContext());

    Type  * workunit_type       = StructType::get(llvmContext, std::vector<Type *>{charPtrType, charPtrType});

    map<RecordAttribute, RawValueMemory> old_bindings{childState.getBindings()};
    
    // Find block size
    Plugin* pg = RawCatalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt", pg->getOIDType()); //FIXME: OID type for blocks ?

    auto it = old_bindings.find(tupleCnt);
    assert(it != old_bindings.end());

    RawValueMemory mem_cntWrapper = it->second;

    Function * make_mem_move = context->getFunction("make_mem_move_device");
    
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
    
    std::vector<Value *> pushed;
    Value * is_noop     = context->createTrue();
    for (size_t i = 0 ; i < wantedFields.size() ; ++i){
        RecordAttribute block_attr  (*(wantedFields[i]), true);

        auto it = old_bindings.find(block_attr);
        assert(it != old_bindings.end());
        RawValueMemory mem_valWrapper = it->second;
        
        Value * mv              = Builder->CreateBitCast(
                                                            Builder->CreateLoad(mem_valWrapper.mem), 
                                                            charPtrType
                                                        );

        Type  * mv_block_type   = mem_valWrapper.mem->getType()->getPointerElementType()->getPointerElementType();

        Value * size            = ConstantInt::get(llvmContext, APInt(64, context->getSizeOf(mv_block_type)));
        Value * Nloc            = Builder->CreateZExtOrBitCast(N, size->getType());
        size                    = Builder->CreateMul(size, Nloc);

        vector<Value *> mv_args{mv, size, device_id, memmv};

        // Do actual mem move
        Value * moved_buffpair  = Builder->CreateCall(make_mem_move, mv_args);
        Value * moved           = Builder->CreateExtractValue(moved_buffpair, 0);
        Value * to_release      = Builder->CreateExtractValue(moved_buffpair, 1);

        pushed.push_back(Builder->CreateBitCast(moved     , mem_valWrapper.mem->getType()->getPointerElementType()));
        pushed.push_back(Builder->CreateBitCast(to_release, mem_valWrapper.mem->getType()->getPointerElementType()));

        is_noop = Builder->CreateAnd(is_noop, Builder->CreateICmpEQ(moved, to_release));
    }
    pushed.push_back(N);
    pushed.push_back(oid);

    Value * d           = UndefValue::get(data_type);
    for (size_t i = 0 ; i < pushed.size() ; ++i){
        d               = Builder->CreateInsertValue(d, pushed[i], i);
    }

    Function * acquire      = context->getFunction("acquireWorkUnit");

    Value * workunit_ptr8   = Builder->CreateCall(acquire, memmv);
    Value * workunit_ptr    = Builder->CreateBitCast(workunit_ptr8, PointerType::getUnqual(workunit_type));

    Value * workunit_dat    = Builder->CreateLoad(workunit_ptr);
    Value * d_ptr           = Builder->CreateExtractValue(workunit_dat, 0);
    d_ptr                   = Builder->CreateBitCast(d_ptr       , PointerType::getUnqual(data_type    ));
    Builder->CreateStore(d, d_ptr);

    Function * propagate = context->getFunction("propagateWorkUnit");
    Builder->CreateCall(propagate, std::vector<Value *>{memmv, workunit_ptr8, is_noop});
}

void MemMoveDevice::open (RawPipeline * pip){
    std::cout << "MemMoveDevice:open" << std::endl;
    workunit * wu = (workunit *) RawMemoryManager::mallocPinned(sizeof(workunit) * slack);

    // nvtxRangePushA("memmove::open");
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm , cudaStreamNonBlocking));

    // cudaStream_t strm2;
    // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));
    
    rawlogger.log(this, log_op::MEMMOVE_OPEN_START);
    size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t) 0xF);

    void        * pmmc  = RawMemoryManager::mallocPinned(sizeof(MemMoveConf));
    MemMoveConf * mmc   = new (pmmc) MemMoveConf;

    rawlogger.log(this, log_op::MEMMOVE_OPEN_END);
#ifndef NCUDA
    mmc->strm           = strm;
    // mmc->strm2          = strm2;
#endif
    mmc->slack          = slack;
    mmc->next_e         = 0;
    // mmc->events         = new cudaEvent_t[slack];
    // mmc->old_buffs      = new void      *[slack];
    mmc->data_buffs     = RawMemoryManager::mallocPinned(data_size * slack);
    char * data_buff = (char *) mmc->data_buffs;
    rawlogger.log(this, log_op::MEMMOVE_OPEN_START);
    for (size_t i = 0 ; i < slack ; ++i){
        wu[i].data  = ((void *) (data_buff + i * data_size));
// // gpu_run(cudaEventCreateWithFlags(&(wu[i].event), cudaEventDisableTiming));//  | cudaEventBlockingSync));
//         gpu_run(cudaEventCreate(&(wu[i].event)));
//         gpu_run(cudaStreamCreate(&(wu[i].strm)));

        mmc->idle.push(wu + i);

        // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming | cudaEventBlockingSync));
        // gpu_run(cudaEventCreate(mmc->events + i));
        // mmc->old_buffs[i] = NULL;
    }
    rawlogger.log(this, log_op::MEMMOVE_OPEN_END);
    // nvtxRangePushA("memmove::open2");
    for (size_t i = 0 ; i < slack ; ++i){
        gpu_run(cudaEventCreateWithFlags(&(wu[i].event), cudaEventDisableTiming  | cudaEventBlockingSync));
// // gpu_run(cudaEventCreateWithFlags(&(wu[i].event), cudaEventDisableTiming));//  | cudaEventBlockingSync));
//         gpu_run(cudaEventCreate(&(wu[i].event)));
//         gpu_run(cudaStreamCreate(&(wu[i].strm)));

        // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming | cudaEventBlockingSync));
        // gpu_run(cudaEventCreate(mmc->events + i));
        // mmc->old_buffs[i] = NULL;
    }
    // nvtxRangePop();

    rawlogger.log(this, log_op::MEMMOVE_OPEN_START);
    mmc->worker = ThreadPool::getInstance().enqueue(&MemMoveDevice::catcher, this, mmc, pip->getGroup(), exec_location{});
    // mmc->worker = new thread(&MemMoveDevice::catcher, this, mmc, pip->getGroup(), exec_location{});
    rawlogger.log(this, log_op::MEMMOVE_OPEN_END);

    int device = -1;
    if (!to_cpu) device = topology::getInstance().getActiveGpu().id;
    pip->setStateVar<int         >(device_id_var, device);

    // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
    pip->setStateVar<void      * >(memmvconf_var, mmc   );
    // nvtxRangePop();
}

void MemMoveDevice::close(RawPipeline * pip){
    rawlogger.log(this, log_op::MEMMOVE_CLOSE_START);
    std::cout << "MemMoveDevice:close" << std::endl;
    // int device = get_device();
    // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
    MemMoveConf * mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

    mmc->tran.close();
    std::cout << "MemMoveDevice:close3" << std::endl;

    nvtxRangePop();
    mmc->worker.get();
    // mmc->worker->join();

    rawlogger.log(this, log_op::MEMMOVE_CLOSE_END);

    rawlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_START);
    // gpu_run(cudaStreamSynchronize(g_strm));

    // int32_t h_s;
    // gpu_run(cudaMemcpy(&h_s, s, sizeof(int32_t), cudaMemcpyDefault));
    // std::cout << "rrr" << h_s << std::endl;

    // RawMemoryManager::freeGpu(s);
    std::cout << "MemMoveDevice:close4" << std::endl;

    gpu_run(cudaStreamSynchronize(mmc->strm ));
    std::cout << "MemMoveDevice:close2" << std::endl;
    gpu_run(cudaStreamDestroy    (mmc->strm ));
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
        // free(wu->data);

        if (i == 0 || wu       < start_wu     ) start_wu      = wu;
        // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
    }
    nvtxRangePop();
    nvtxRangePop();

    RawMemoryManager::freePinned(mmc->data_buffs);
    // assert(mmc->tran.empty_unsafe());
    // assert(mmc->idle.empty_unsafe());
    // free(start_wu_data);
    // delete[] start_wu;
    RawMemoryManager::freePinned(start_wu);
    // delete[] mmc->events   ;
    // delete[] mmc->old_buffs;

    mmc->idle.close();//false);

    // delete mmc->worker;
    // delete mmc;
    mmc->~MemMoveConf();
    RawMemoryManager::freePinned(mmc);
    rawlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_END);
}


extern "C"{
    MemMoveDevice::workunit * acquireWorkUnit  (MemMoveDevice::MemMoveConf * mmc){
        MemMoveDevice::workunit * ret = nullptr;
#ifndef NDEBUG
        bool popres = 
#endif
        mmc->idle.pop(ret);
        assert(popres);
        return ret;
    }

    void propagateWorkUnit(MemMoveDevice::MemMoveConf * mmc, MemMoveDevice::workunit * buff, bool is_noop){
        // if (!is_noop)
        // gpu_run(cudaEventRecord(buff->event, mmc->strm));
        // gpu_run(cudaEventDestroy(buff->event));
        // gpu_run(cudaEventCreate(&(buff->event)));
        // gpu_run(cudaEventRecord(buff->event, mmc->strm));
        // gpu_run(cudaStreamSynchronize(mmc->strm));
            // std::cout << (void *) buff->event << " " << (void *) mmc->strm << std::endl;
        // std::cout << "rec" << (void *) buff->event << std::endl;

        // gpu_run(cudaEventSynchronize(buff->event));
        // gpu_run(cudaEventDestroy(buff->event));
        // gpu_run(cudaEventRecord(buff->event, mmc->strm));
        // gpu_run(cudaEventSynchronize(buff->event));
        // gpu_run(cudaEventSynchronize(buff->event));
        // gpu_run(cudaEventRecord(buff->event, mmc->strm));
        // gpu_run(cudaStreamWaitEvent(buff->strm, buff->event, 0));
        // std::cout << "asdasdasD" << __rdtsc() << " " << (void *) buff->event << std::endl;
        // gpu_run(cudaStreamSynchronize(buff->strm));
        // gpu_run(cudaStreamSynchronize(buff->strm));
            // gpu_run(cudaStreamSynchronize(buff->strm));
            // gpu_run(cudaStreamDestroy(buff->strm));
        if (!is_noop) gpu_run(cudaEventRecord(buff->event, mmc->strm));

        mmc->tran.push(buff);
    }


    bool acquirePendingWorkUnit(MemMoveDevice::MemMoveConf * mmc, MemMoveDevice::workunit ** ret){
        if (!mmc->tran.pop(*ret)) return false;
        gpu_run(cudaEventSynchronize((*ret)->event));
            // gpu_run(cudaStreamSynchronize((*ret)->strm));
            // gpu_run(cudaStreamDestroy((*ret)->strm));
        // gpu_run(cudaStreamSynchronize((*ret)->strm));
        // gpu_run(cudaEventSynchronize((*ret)->event));
        // gpu_run(cudaEventDestroy((*ret)->event));
        // gpu_run(cudaStreamSynchronize((*ret)->strm));
        // gpu_run(cudaStreamSynchronize((*ret)->strm));
        // gpu_run(cudaEventRecord((*ret)->event, mmc->strm));
        // gpu_run(cudaStreamWaitEvent((*ret)->strm, (*ret)->event, 0));
        // gpu_run(cudaStreamSynchronize((*ret)->strm));
        // std::cout << "asdasdasD" << __rdtsc() << " " << (void *) (*ret)->event << std::endl;
        // gpu_run(cudaStreamSynchronize(mmc->strm));
        // std::cout << "asdasdasD" << __rdtsc() << " " << (void *) (*ret)->event << std::endl;
        return true;
    }

    void releaseWorkUnit  (MemMoveDevice::MemMoveConf * mmc, MemMoveDevice::workunit * buff){
        mmc->idle.push(buff);
    }
}

void MemMoveDevice::catcher(MemMoveConf * mmc, int group_id, exec_location target_dev){
    // std::cout << target_dev. << std::endl;
    set_exec_location_on_scope d(target_dev);
    std::this_thread::yield();

    nvtxRangePushA("memmove::catch");

    RawPipeline * pip = catch_pip->getPipeline(group_id);

    nvtxRangePushA("memmove::catch_open");
    pip->open();
    nvtxRangePop();
    
    {
        do {
            MemMoveDevice::workunit * p = nullptr;
            // rawlogger.log(this, log_op::MEMMOVE_CONSUME_WAIT_START);
            if (!acquirePendingWorkUnit(mmc, &p)) break;
            // rawlogger.log(this, log_op::MEMMOVE_CONSUME_WAIT_END  );
            // ++cnt;
            // std::cout << (void *) p->event << " " << (void *) mmc->strm << std::endl;
            // gpu_run(cudaStreamSynchronize(mmc->strm));
            nvtxRangePushA("memmove::catch_cons");
            // N += ((int64_t *) p->data)[2];
            // std::cout << *((void **) p->data) << " " << get_device(*((void **) p->data)) << " Started.............................." << std::endl;
            // size_t x = ((int64_t *) p->data)[2];
            // int32_t k = 0;
            // for (size_t i = 0 ; i < x ; ++i){
            //     k += ((int32_t **) p->data)[1][i];
            // }
            // sum2 += k;
            // std::cout << "s" << ((int32_t **) p->data)[0] << " " << k << std::endl;
            rawlogger.log(this, log_op::MEMMOVE_CONSUME_START);
            pip->consume(0, p->data);
            rawlogger.log(this, log_op::MEMMOVE_CONSUME_END  );
            // // size_t x = ((int64_t *) p->data)[2];
            // for (size_t i = 0 ; i < x ; ++i){
            //     sum += ((int32_t **) p->data)[1][i];
            // }
            // std::cout << *((void **) p->data) << " " << get_device(*((void **) p->data)) << " Finished............................." << std::endl;
            nvtxRangePop();

            releaseWorkUnit(mmc, p); //FIXME: move this inside the generated code
        } while (true);
    }

    nvtxRangePushA("memmove::catch_close");
    pip->close();
    nvtxRangePop();

    nvtxRangePop();
}
