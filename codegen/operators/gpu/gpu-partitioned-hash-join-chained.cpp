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

#include "operators/gpu/gpu-partitioned-hash-join-chained.hpp"
#include "operators/gpu/gmonoids.hpp"
#include "util/raw-memory-manager.hpp"
#include "util/gpu/gpu-intrinsics.hpp"

#include "cuda.h"
#include "cuda_runtime.h"  

#define COMPACT_OFFSETS_
//#define PARTITION_PAYLOAD 

#define SHMEM_SIZE 4096
#define HT_LOGSIZE 10

extern __shared__ int int_shared[];

const size_t log2_bucket_size = 12;
const size_t bucket_size = 1 << log2_bucket_size;
const size_t bucket_size_mask = bucket_size - 1;

__device__ int hashd (int val) {
    /*val = (val >> 16) ^ val;
    val *= 0x85ebca6b;
    val = (val >> 13) ^ val;
    val *= 0xc2b2ae35;
    val = (val >> 16) ^ val;*/
    return val;
}

union vec4{
    int4    vec ;
    int32_t i[4];
};

__global__ void init_first    (    int32_t  * payload,
                                   int32_t  * cnt_ptr,
                                   uint32_t  * chains,
                                   uint32_t  * buckets_used) {
    uint32_t cnt = *cnt_ptr;

#ifndef PARTITION_PAYLOAD
    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < cnt; i += blockDim.x*gridDim.x)
        payload[i] = i;
#endif
    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < (cnt+bucket_size-1)/bucket_size - 1; i += blockDim.x*gridDim.x)
        chains[i] = i+1;

    if (threadIdx.x + blockIdx.x*blockDim.x == 0)
        chains[(cnt+bucket_size-1)/bucket_size - 1] = 0;

    *buckets_used = (cnt+bucket_size-1)/bucket_size;
}


__global__ void init_metadata (     uint64_t * heads,
                                    uint32_t * chains,
                                    int32_t * out_cnts,
                                    uint32_t * buckets_used,
                                    uint32_t   parts,
                                    uint32_t   buckets_num) {

    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < buckets_num; i += blockDim.x*gridDim.x)
        chains[i] = 0;

    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < parts; i += blockDim.x*gridDim.x) {
        out_cnts[i] = 0;
        heads[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);
    }

    if (threadIdx.x + blockIdx.x*blockDim.x == 0)
        *buckets_used = parts;

}

__global__ void compute_bucket_info (uint32_t* bucket_info, uint32_t* chains, int32_t* out_cnts, uint32_t log_parts) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t cnt = out_cnts[p];

        while (cnt > 0) {
            uint32_t local_cnt = (cnt >= bucket_size)? bucket_size : cnt;
            uint32_t val = (p << 15) + local_cnt;
            
            uint32_t next = chains[cur];
            bucket_info[cur] = val;

            cur = next;
            cnt -= bucket_size;
        }
    }
}

__global__ void verify_decomposition (uint32_t* bucket_info, uint32_t* buckets_used) {
    int cnt = *buckets_used;
    int sum = 0;

    for (int i = 0; i < cnt; i++) {
        if (bucket_info[i] != 0) {
            sum += bucket_info[i] & ((1 << 15) - 1);
        }
    }

    printf("%d\n", sum);
}

__global__ void verify_partitions (int* S_in, int* P_in, uint32_t* chains, int* out_cnts, int* S_out, int* P_out, int log_parts, int* error_cnt) {
    for (int p = blockIdx.x; p < (1 << log_parts); p += gridDim.x) {
        int current_bucket = p;
        int remaining = out_cnts[p];

        while (remaining > 0) {
            int cnt = (remaining > bucket_size) ? bucket_size : remaining;
            
            for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
                int offset = current_bucket*bucket_size + i;
                int payload = P_out[offset];


                if (S_out[offset] != S_in[payload]) {
                    printf("Errooooor %d %d %d %d %d %d!\n", S_out[offset], payload, S_in[payload], P_in[payload], cnt, current_bucket);
                    atomicAdd(error_cnt, 1);
                }
            }

            current_bucket = chains[p];
            remaining -= 4096;    
        }
    }
}
__global__ void printpart (int* S_in, uint32_t* chains, int* out_cnts, int p) {
    int bucket = p;
    int cnt = out_cnts[p];

    while (cnt > 0) {
        int local_cnt = (cnt < bucket_size) ? cnt : bucket_size;

        for (int i = 0; i < local_cnt; i++)
            printf("%d\n", S_in[bucket*bucket_size + i]);

        bucket = chains[bucket];
        cnt -= bucket_size;
    }
}

__global__ void decompose_chains (uint32_t* bucket_info, uint32_t* chains, int32_t* out_cnts, uint32_t log_parts, int threshold) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t  cnt = out_cnts[p];
        uint32_t first_cnt = (cnt >= threshold)? threshold : cnt;
        int32_t  cutoff = 0;

        while (cnt > 0) {
            cutoff += bucket_size;
            cnt -= bucket_size;

            uint32_t next = chains[cur];
            
            if (cutoff >= threshold && cnt > 0) {
                uint32_t local_cnt = (cnt >= threshold)? threshold : cnt;
                bucket_info[next] = (p << 15) + local_cnt;
                //printf("%d!!\n", next);
                chains[cur] = 0;
                cutoff = 0;
            } else if (next != 0) {
                bucket_info[next] = 0;
            }


            cur = next;
        }

        bucket_info[p] = (p << 15) + first_cnt;
    }
}



__global__ void build_partitions (
                                    const int32_t   * __restrict__ S,
                                    const int32_t   * __restrict__ P,
                                    const uint32_t  * __restrict__ bucket_info,
                                          uint32_t  * __restrict__ buckets_used,
                                          uint64_t  *              heads,
                                          uint32_t  * __restrict__ chains,
                                          int32_t   * __restrict__ out_cnts,
                                          int32_t   * __restrict__ output_S,
                                          int32_t   * __restrict__ output_P,
                                          uint32_t                 S_log_parts,
                                          uint32_t                 log_parts,
                                          uint32_t                 first_bit,
                                          uint32_t  *              bucket_num_ptr) {
    assert((((size_t) bucket_size) + ((size_t) blockDim.x) * gridDim.x) < (((size_t) 1) << 32));
    // assert((parts & (parts - 1)) == 0);
    const uint32_t S_parts   = 1 << S_log_parts;
    const uint32_t parts     = 1 << log_parts;
    const int32_t parts_mask = parts - 1;

    uint32_t buckets_num = *bucket_num_ptr;

    //get shared memory pointer

    uint32_t * router = (uint32_t *) int_shared; //[1024*4 + parts];

    //initialize shared memory

    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x ) 
        router[1024*4 + parts + j] = 0;
    
    if (threadIdx.x == 0) 
        router[0] = 0;

    __syncthreads();

    //loop over the blocks

    for (size_t i = blockIdx.x; i < buckets_num; i += gridDim.x) {
        uint32_t info = bucket_info[i];
        uint32_t cnt = info & ((1 << 15) - 1);
        uint32_t pid = info >> 15;

        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(S + bucket_size * i + 4*threadIdx.x));

        uint32_t thread_keys[4];

        //compute local histogram
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt){
                uint32_t partition = (hashd(thread_vals.i[k]) >> first_bit) & parts_mask;

                atomicAdd(router + (1024 * 4 + parts + partition), 1);
                
                thread_keys[k] = partition;
            } else {
                thread_keys[k] = 0;
            }
        }

        __syncthreads();


        //update bucket chain

        for (size_t j = threadIdx.x; j < parts ; j += blockDim.x ) {
            uint32_t cnt = router[1024 * 4 + parts + j];

            if (cnt > 0){
                atomicAdd(out_cnts + (pid << log_parts) + j, cnt); //Is this needed ?
                
                uint32_t pcnt     ;
                uint32_t bucket   ;
                uint32_t next_buck;

                // assert(cnt <= bucket_size);

                bool repeat = true;

                while (__any(repeat)){ //without the "repeat" variable, the compiler 
                                        //probably moves the "if(pcnt < bucket_size)"
                                        //block out of the loop, which creates a deadlock
                                        //using the repeat variable, it should 
                                        //convince the compiler that it should not
                    if (repeat){
                        uint64_t old_heads = __atomic_fetch_add(heads + (pid << log_parts) + j, ((uint64_t) cnt) << 32, __ATOMIC_SEQ_CST); //atomicAdd(heads + (pid << log_parts) + j, ((uint64_t) cnt) << 32);
    
                        atomicMin(heads + (pid << log_parts) + j, ((uint64_t) (2*bucket_size)) << 32);

                        pcnt       = ((uint32_t) (old_heads >> 32));
                        bucket     =  (uint32_t)  old_heads        ;

                        //now there are two cases:
                        // 2) old_heads.cnt >  bucket_size ( => locked => retry)
                        // if (pcnt       >= bucket_size) continue;
    
                        if (pcnt < bucket_size){
                            // 1) old_heads.cnt <= bucket_size

                            //check if the bucket was filled
                            if (pcnt + cnt >= bucket_size){ //&& pcnt <  bucket_size
                                // assert(pcnt + cnt < 2*bucket_size);
                                //must replace bucket!
                                    
                                if (bucket < (1 << 18)) {
                                    next_buck = atomicAdd(buckets_used, 1);                                
                                    chains[bucket]     = next_buck;
                                } else {
                                    next_buck = (pid << log_parts) + j;
                                }


                                // assert(next_buck >= parts);
    
                                // assert(pcnt + cnt - bucket_size >= 0);
                                // assert(pcnt + cnt - bucket_size <  bucket_size);
    
                                uint64_t tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);
    
                                // assert(((uint32_t) (tmp >> 32)) < bucket_size);
    
                                //atomicExch(heads + (pid << log_parts) + j, tmp); //also zeroes the cnt!
                                __atomic_exchange_n(heads + (pid << log_parts) + j, tmp, __ATOMIC_SEQ_CST);
                            } else {
                                next_buck = bucket;
                            }
    
                            repeat = false;
                        }
                    }
                }
    
                //NOTE shared memory requirements can be relaxed, but when moving the two last "rows" one up, we get a 10% performance penalty! This needs a little bit more investigation
                router[1024 * 4             + j] = atomicAdd(router, cnt);
                router[1024 * 4 +     parts + j] = 0;//cnt;//pcnt     ;
                router[1024 * 4 + 2 * parts + j] = (bucket    << log2_bucket_size) + pcnt;
                router[1024 * 4 + 3 * parts + j] =  next_buck << log2_bucket_size        ;
            }
        }

        __syncthreads();
    
    
        uint32_t total_cnt = router[0];
    
        __syncthreads();
    
        //calculate target positions for block-wise shuffle
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt)
                thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);
        }
    
        //perform the shuffle
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) 
            if (4*threadIdx.x + k < cnt)
                router[thread_keys[k]] = thread_vals.i[k];
    
        __syncthreads();
    
        int32_t thread_parts[4];

        //write out partition
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t  val       = router[threadIdx.x + 1024 * k];
                uint32_t partition = (hashd(val) >> first_bit) & parts_mask;

                uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];
                // uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                    
                bucket += cnt;
            
                output_S[bucket] = val;

                thread_parts[k] = partition;
            }
        }

        __syncthreads();

        thread_vals = *(reinterpret_cast<const vec4 *>(P + i*bucket_size + 4*threadIdx.x));

        //perform the shuffle
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) 
            if (4*threadIdx.x + k < cnt) {
                router[thread_keys[k]] = thread_vals.i[k];
            }

        __syncthreads();

        //write out payload  
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t  val       = router[threadIdx.x + 1024 * k];

                int32_t partition = thread_parts[k];

                uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];
                // uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                bucket += cnt;
           
                output_P[bucket] = val;
            }
        }

        //re-init
        if (threadIdx.x == 0) router[0] = 0;
    }
}


#define BUILD_SIZE 128000000
#define PROBE_SIZE 128000000

HashPartitioner::HashPartitioner (
            RecordAttribute*                   targetAttr,
            const std::vector<GpuMatExpr>      &parts_mat_exprs, 
            const std::vector<size_t>          &parts_packet_widths,
            expressions::Expression *           parts_keyexpr,
            RawOperator * const                 parts_child,
            GpuRawContext *                     context,
            int                                 log_parts,
            string                              opLabel) :
                    targetAttr(targetAttr),
                    parts_mat_exprs(parts_mat_exprs),
                    parts_packet_widths(parts_packet_widths),
                    parts_keyexpr(parts_keyexpr),
                    UnaryRawOperator(parts_child),
                    context(context),
                    log_parts(log_parts),
                    opLabel(opLabel) 
{
    vector<expressions::Expression*> expr;
    for (size_t i = 0; i < parts_mat_exprs.size(); i++)
        expr.push_back(parts_mat_exprs[i].expr);
    mat = new Materializer(expr);

    pg_out = new OutputPlugin(context, *mat, NULL);

    payloadType = pg_out->getPayloadType();

    //log_parts = 15;
    log_parts2 = log_parts / 2;
    log_parts1 = log_parts - log_parts2;
} 

void HashPartitioner::produce () {
    parts_mat_exprs.emplace_back(parts_keyexpr                  , 0, 32);

    std::sort(parts_mat_exprs.begin(), parts_mat_exprs.end(), [](const GpuMatExpr& a, const GpuMatExpr& b){
        if (a.packet == b.packet) return a.bitoffset < b.bitoffset;
        return a.packet < b.packet;
    });

    matFormat();
    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip); });
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close(pip); });     
    getChild()->produce();
}

void HashPartitioner::matFormat(){
    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    //assumes than build has already run
    
    const ExpressionType * out_type_key = parts_keyexpr->getExpressionType();
    Type * llvm_type_key = ((const PrimitiveType *) out_type_key)->getLLVMType(context->getLLVMContext());
    Type * t_ptr_key = PointerType::get(llvm_type_key, 1);
    param_pipe_ids.push_back(context->appendStateVar(t_ptr_key));

    Type * t_ptr_payload = PointerType::get(payloadType, 1);
    param_pipe_ids.push_back(context->appendStateVar(t_ptr_payload));

    Type * t_cnt = PointerType::get(int32_type, 1);
    cnt_pipe = context->appendStateVar(t_cnt);
    //std::cout << cnt_right_param_id << std::endl;
}

void HashPartitioner::consume(RawContext* const context, const OperatorState& childState) {
    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int64_type = Type::getInt64Ty(context->getLLVMContext());

    Value * out_cnt = ((const GpuRawContext *) context)->getStateVar(cnt_pipe);

    Value * old_cnt  = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Add, 
                                                out_cnt,
                                                ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
                                                llvm::AtomicOrdering::Monotonic);
    old_cnt->setName("index");

    const map<RecordAttribute, RawValueMemory>& bindings = childState.getBindings();
    pg_out->setBindings(&childState.getBindings());
    Value *val_payloadSize  = ConstantInt::get((IntegerType *) int64_type, context->getSizeOf(payloadType));
    vector<Type*> *materializedTypes = pg_out->getMaterializedTypes();
    PointerType *ptr_payloadType = PointerType::get(payloadType, 1);


    ExpressionGeneratorVisitor exprGenerator(context, childState);
    RawValue valWrapper = parts_keyexpr->accept(exprGenerator);
    Value * key_ptr = ((const GpuRawContext *) context)->getStateVar(param_pipe_ids[0]);
    key_ptr->setName(opLabel + "_key");
    Value* key_ptr_offset = Builder->CreateInBoundsGEP(key_ptr, old_cnt);
    Builder->CreateStore(valWrapper.value, key_ptr_offset);

    int offsetInWanted = 0;
    int offsetInStruct = 0;

    for (const auto &we : mat->getWantedExpressions()) {
        Value* valToMaterialize = NULL;

            ExpressionGeneratorVisitor exprGen{context, childState};
            RawValue currVal = we->accept(exprGen);
            /* FIX THE NECESSARY CONVERSIONS HERE */
            valToMaterialize = currVal.value;
        //}

        vector<Value*> idxList = vector<Value*>();
        idxList.push_back(context->createInt32(0));
        idxList.push_back(context->createInt32(offsetInStruct));

        //Shift in struct ptr
        Value* arena =  ((const GpuRawContext *) context)->getStateVar(param_pipe_ids[1]);
        arena->setName(opLabel + "_payload");
        Value* arenaShifted = Builder->CreateInBoundsGEP(arena, old_cnt);
        Value* structPtr = Builder->CreateGEP(arenaShifted, idxList);

        Builder->CreateStore(valToMaterialize, structPtr);
        offsetInStruct++;
        offsetInWanted++;
    }

    //BasicBlock* bb = Builder->GetInsertBlock();
    //Builder->SetInsertPoint(context->getEndingBlock());
    /*
    map<RecordAttribute, RawValueMemory>* new_bindings = new map<RecordAttribute, RawValueMemory>();
    
    AllocaInst* mem_arg = context->CreateEntryBlockAlloca(TheFunction, "mem_target_N", int32_type);
    Builder->CreateStore(Builder->CreateTrunc(context->createInt32(0), int32_type), mem_arg);
    RawValueMemory mem_valWrapper;
    mem_valWrapper.mem    = mem_arg;
    mem_valWrapper.isNull = context->createFalse();
    (*new_bindings)[*targetAttr] = mem_valWrapper;

    Plugin* pg = RawCatalog::getInstance().getPlugin(targetAttr->getRelationName());
    RecordAttribute tupleCnt = RecordAttribute(targetAttr->getRelationName(), "activeCnt", pg->getOIDType());
    AllocaInst* mem_arg2 = context->CreateEntryBlockAlloca(TheFunction, "mem_cnt_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg2);
    RawValueMemory mem_valWrapper2;
    mem_valWrapper2.mem    = mem_arg2;
    mem_valWrapper2.isNull = context->createFalse();
    (*new_bindings)[tupleCnt] = mem_valWrapper2;

    RecordAttribute tupleOID = RecordAttribute(targetAttr->getRelationName(), activeLoop, pg->getOIDType());
    AllocaInst* mem_arg3 = context->CreateEntryBlockAlloca(TheFunction, "mem_oid_N", pg->getOIDType()->getLLVMType(llvmContext));
    Builder->CreateStore(context->createInt32(0), mem_arg3);
    RawValueMemory mem_valWrapper3;
    mem_valWrapper3.mem    = mem_arg3;
    mem_valWrapper3.isNull = context->createFalse();
    (*new_bindings)[tupleOID] = mem_valWrapper3;*/

    /*vector<Value*> ArgsV; 
    Value* value = current_partition;
    ArgsV.push_back(value);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);*/

    /*OperatorState* newState = new OperatorState(*this, *new_bindings);
    getParent()->consume(context, *newState);*/

   // Builder->SetInsertPoint(bb);
}


void HashPartitioner::open(RawPipeline * pip){
    std::cout << "GpuOptJoin::open::probe_" <<  std::endl;

    vector<void*> param_ptr = state.cols[pip->getGroup()];

    int device;
    cudaGetDevice(&device);
    std::cout << state.cols[pip->getGroup()][0] << " " << device << std::endl;

    int32_t* cnt_ptr = (int32_t*) RawMemoryManager::mallocGpu(sizeof(int32_t));

    uint32_t   parts2 = 1 << (log_parts1 + log_parts2);
    size_t     buckets_num_max = (((PROBE_SIZE + parts2 - 1)/parts2 + bucket_size - 1)/bucket_size)*parts2;
    size_t     buffer_size = buckets_num_max * bucket_size;
    size_t     alloca_size = (buckets_num_max + parts2 + 4) * sizeof(int32_t) + parts2 * sizeof(uint64_t);

    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    gpu_run(cudaMemsetAsync( cnt_ptr,  0, sizeof( int32_t)             , strm));

    pip->setStateVar(cnt_pipe, cnt_ptr);
    for (size_t i = 0; i < parts_mat_exprs.size(); i++) {
        pip->setStateVar(param_pipe_ids[i], param_ptr[i]);
    }

    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));

    
    cnts_ptr[pip->getGroup()] = cnt_ptr;
    
    //std::cout << "GpuOptJoin::open::probe2" << std::endl;
}


void HashPartitioner::close (RawPipeline * pip){
    std::cout << "GpuOptJoin::close::probe_" << pip->getGroup() << std::endl;

    std::cout << parts_mat_exprs[0].expr->getRegisteredAttrName() << std::endl;

    int32_t h_cnt;
    gpu_run(cudaMemcpy(&h_cnt, cnts_ptr[pip->getGroup()], sizeof(int32_t), cudaMemcpyDefault));
    std::cout << "Probe cnt " << std::dec << h_cnt << std::endl;

    uint32_t parts2 = 1 << (log_parts1 + log_parts2);

    size_t     buckets_num_max = (((PROBE_SIZE + parts2 - 1)/parts2 + bucket_size - 1)/bucket_size)*parts2;
    size_t     buffer_size = buckets_num_max * bucket_size;

    char*    alloca = state.allocas[pip->getGroup()];
    int32_t* cnt_ptr = cnts_ptr[pip->getGroup()];

    size_t     alloca_size = (buckets_num_max + parts2 + 4) * sizeof(int32_t) + parts2 * sizeof(uint64_t);

    PartitionMetadata pmeta = state.meta[pip->getGroup()];

    int32_t  * keys_init = pmeta.keys;
    int32_t  * payload_init = pmeta.payload;
    uint32_t * chains_init  = pmeta.chains;
    int32_t  * out_cnts_init = pmeta.out_cnts;
    uint32_t * buckets_used_init = pmeta.buckets_used;
    uint64_t * heads_init = pmeta.heads;

    std::cout << payload_init << std::endl;

    char     * alloca1  = (char*) RawMemoryManager::mallocGpu(alloca_size);

    int32_t  * keys1 = (int32_t*) RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t));
    int32_t  * payload1 = (int32_t*) RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t));
    uint32_t * chains1  = (uint32_t*) (alloca1);
    int32_t  * out_cnts1 = (int32_t*) (alloca1 + (buckets_num_max)*sizeof(int32_t));
    uint32_t * buckets_used1 = (uint32_t*) (alloca1 + (buckets_num_max + parts2)*sizeof(int32_t));
    uint64_t * heads1 = (uint64_t*) (alloca1 + (buckets_num_max + parts2 + 4)*sizeof(int32_t));

    int32_t  * keys2 = keys_init;
    int32_t  * payload2 = payload_init;
    uint32_t * chains2  = chains_init;
    int32_t  * out_cnts2 = out_cnts_init;
    uint32_t * buckets_used2 = buckets_used_init;
    uint64_t * heads2 = heads_init;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //int32_t  * error_cnt = (int32_t*) RawMemoryManager::mallocGpu(sizeof(int32_t));
    //cudaMemset(error_cnt, 0, sizeof(int));

    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    gpu_run(cudaStreamSynchronize(strm));
    cudaEventRecord(start, strm);

    init_first  <<<64, 1024, 0, strm>>> (payload_init, cnts_ptr[pip->getGroup()], chains_init, buckets_used_init);
    
    init_metadata  <<<64, 1024, 0, strm>>> (heads1, chains1, out_cnts1, buckets_used1, 1 << log_parts1, buckets_num_max);
    
    compute_bucket_info  <<<64, 1024, 0, strm>>> (chains_init, chains_init, cnts_ptr[pip->getGroup()], 0);

    gpu_run(cudaStreamSynchronize(strm));

    build_partitions  <<<64, 1024, (1024*4 + 4*(1 << log_parts1)) * sizeof(int32_t) + ((2 * (1 << log_parts1) + 1)* sizeof(int32_t)), strm>>> (
                    keys_init, payload_init, chains_init, 
                    buckets_used1, heads1, chains1, out_cnts1, keys1, payload1,
                        0, log_parts1, log_parts2, buckets_used_init);

    //gpu_run(cudaStreamSynchronize(strm));
    //verify_partitions <<<64, 1024, 0, strm>>> (keys_init, payload_init, chains1, out_cnts1, keys1, payload1, log_parts1, error_cnt);
    

    //gpu_run(cudaMemcpyAsync(&h_cnt, error_cnt, sizeof(int32_t), cudaMemcpyDefault, strm));
    //std::cout << "Probe cnt " << h_cnt << std::endl;

    gpu_run(cudaStreamSynchronize(strm));

    init_metadata  <<<64, 1024, 0, strm>>> (heads2, chains2, out_cnts2, buckets_used2, 1 << (log_parts1 + log_parts2), buckets_num_max);
    compute_bucket_info  <<<64, 1024, 0, strm>>> (chains1, chains1, out_cnts1, log_parts1);

    gpu_run(cudaStreamSynchronize(strm));
    build_partitions  <<<64, 1024, (1024*4 + 4*(1 << log_parts2)) * sizeof(int32_t) + ((2 * (1 << log_parts2) + 1)* sizeof(int32_t)), strm>>> (
                    keys1, payload1, chains1, 
                    buckets_used2, heads2, chains2, out_cnts2, keys2, payload2,
                        log_parts1, log_parts2, 0, buckets_used1);
    gpu_run(cudaStreamSynchronize(strm));

    decompose_chains  <<<64, 1024, 0, strm>>> (pmeta.bucket_info, chains2, out_cnts2, log_parts1+log_parts2, 2*bucket_size);
    //compute_bucket_info  <<<64, 1024, 0, strm>>> (pmeta.bucket_info, chains2, out_cnts2, log_parts);
    //verify_decomposition  <<<64, 1024, 0, strm>>> (pmeta.bucket_info, buckets_used2);

    gpu_run(cudaStreamSynchronize(strm));

    cudaEventRecord(stop, strm);

    gpu_run(cudaStreamDestroy(strm));

    //gpu_run(cudaMemcpy(&h_cnt, buckets_used1, sizeof(int32_t), cudaMemcpyDefault));
    //std::cout << "Probe cnt " << h_cnt << std::endl;

    /*
    int sum = 0;

    for (int i = 0; i < (1 << log_parts); i++) {
        gpu_run(cudaMemcpy(&h_cnt, &out_cnts2[i], sizeof(int32_t), cudaMemcpyDefault));
        //std::cout << "woohooooooooooooooo " << h_cnt << std::endl;
        sum += h_cnt;
    }

    std::cout << "woohooooooooooooooo " << sum << std::endl;*/

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time " << milliseconds << std::endl;


    RawMemoryManager::freeGpu(keys1);
    RawMemoryManager::freeGpu(payload1);
    RawMemoryManager::freeGpu(alloca1);

    RawMemoryManager::freeGpu(cnt_ptr);

    std::cout << "partition::close" << std::endl;
}


GpuPartitionedHashJoinChained::GpuPartitionedHashJoinChained(
            const std::vector<GpuMatExpr>      &build_mat_exprs, 
            const std::vector<size_t>          &build_packet_widths,
            expressions::Expression *           build_keyexpr,
            HashPartitioner * const                 build_child,

            const std::vector<GpuMatExpr>      &probe_mat_exprs, 
            const std::vector<size_t>          &probe_mat_packet_widths,
            expressions::Expression *           probe_keyexpr,
            HashPartitioner * const                 probe_child,

            PartitionState&                     state_left,
            PartitionState&                     state_right,

            int                                 log_parts,
            GpuRawContext *                     context,
            string                              opLabel,
            RawPipelineGen**                    caller,
            RawOperator * const                      unionop): 
                build_mat_exprs(build_mat_exprs),
                probe_mat_exprs(probe_mat_exprs),
                build_packet_widths(build_packet_widths),
                build_keyexpr(build_keyexpr),
                probe_keyexpr(probe_keyexpr),
                hash_bits(hash_bits),
                BinaryRawOperator(build_child, probe_child),
                log_parts(log_parts), 
                context(context),
                opLabel(opLabel),
                caller(caller),
                state_left(state_left),
                state_right(state_right),
                unionop(unionop) {

    std::cout << "USING OPTIMIZED JOIN ALGORITHM " <<  maxBuildInputSize << std::endl;

    

    payloadType_left = build_child->getPayloadType();
    payloadType_right = probe_child->getPayloadType();

    //log_parts = 15;
    log_parts2 = log_parts / 2;
    log_parts1 = log_parts - log_parts2;

	this->hash_bits = HT_LOGSIZE;
}



__global__ void print_gpu () {
    printf ("Hello world\n");
}

void GpuPartitionedHashJoinChained::produce() {
    probe_mat_exprs.emplace_back(probe_keyexpr                  , 0, 32);

    std::sort(probe_mat_exprs.begin(), probe_mat_exprs.end(), [](const GpuMatExpr& a, const GpuMatExpr& b){
        if (a.packet == b.packet) return a.bitoffset < b.bitoffset;
        return a.packet < b.packet;
    });

    build_mat_exprs.emplace_back(build_keyexpr                  , 0, 32);

    std::sort(build_mat_exprs.begin(), build_mat_exprs.end(), [](const GpuMatExpr& a, const GpuMatExpr& b){
        if (a.packet == b.packet) return a.bitoffset < b.bitoffset;
        return a.packet < b.packet;
    });

    //context->getCurrentPipeline()->set_max_worker_size(1024, 64);

    buildHashTableFormat();
    probeHashTableFormat();

    if (caller != NULL) {
        ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->allocate (pip); });
    }

    ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->open (pip); });
    ((GpuRawContext *) context)->registerClose(this, [this](RawPipeline * pip){this->close (pip);});
    generate_joinloop(context);

    ((GpuRawContext *) context)->popPipeline();

    auto flush_pip = ((GpuRawContext *) context)->removeLatestPipeline();

    context->pushPipeline();

    if (caller == NULL) {     
        ((GpuRawContext *) context)->registerOpen (this, [this](RawPipeline * pip){this->allocate (pip); });
        
        getLeftChild()->produce();
        context->popPipeline();


        context->pushPipeline();
        context->setChainedPipeline(flush_pip);
        getRightChild()->produce();
    } else {
        *caller = flush_pip;
        getLeftChild()->produce();
        context->popPipeline();


        context->pushPipeline();
        //context->setChainedPipeline(flush_pip);
        getRightChild()->produce();
        //unionop->produce();
    }
    //context->getModule()->dump();
}


void GpuPartitionedHashJoinChained::consume(RawContext* const context, const OperatorState& childState) {
    
}

void GpuPartitionedHashJoinChained::probeHashTableFormat(){
    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    //assumes than build has already run

    const ExpressionType * out_type_key = probe_keyexpr->getExpressionType();
    Type * llvm_type_key = ((const PrimitiveType *) out_type_key)->getLLVMType(context->getLLVMContext());
    Type * t_ptr_key = PointerType::get(llvm_type_key, 1);
    probe_param_join_ids.push_back(context->appendStateVar(t_ptr_key));

    Type * t_ptr_payload = PointerType::get(payloadType_right, 1);
    probe_param_join_ids.push_back(context->appendStateVar(t_ptr_payload));

    Type * t_cnt = PointerType::get(int32_type, 1);
    cnt_right_join = context->appendStateVar(t_cnt);

    chains_right_join = context->appendStateVar(t_cnt);

    keys_partitioned_probe_id = context->appendStateVar(t_cnt);
    idxs_partitioned_probe_id = context->appendStateVar(t_cnt);
    //std::cout << cnt_right_param_id << std::endl;

    buckets_used_id = context->appendStateVar(t_cnt);
    bucket_info_id = context->appendStateVar(t_cnt);
    buffer_id = context->appendStateVar(t_cnt);
}




void GpuPartitionedHashJoinChained::buildHashTableFormat(){
    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *t_head_ptr = PointerType::get(int32_type, 1);
    //head_id = context->appendStateVar(t_head_ptr);//, true, false);

    const ExpressionType * out_type_key = build_keyexpr->getExpressionType();
    Type * llvm_type_key = ((const PrimitiveType *) out_type_key)->getLLVMType(context->getLLVMContext());
    Type * t_ptr_key = PointerType::get(llvm_type_key, 1);
    build_param_join_ids.push_back(context->appendStateVar(t_ptr_key));

    Type * t_ptr_payload = PointerType::get(payloadType_left, 1);
    build_param_join_ids.push_back(context->appendStateVar(t_ptr_payload));

    Type * t_cnt = PointerType::get(int32_type,  1);
    cnt_left_join = context->appendStateVar(t_cnt);

    chains_left_join = context->appendStateVar(t_cnt);

    keys_partitioned_build_id = context->appendStateVar(t_cnt);
    idxs_partitioned_build_id = context->appendStateVar(t_cnt);

    keys_cache_id = context->appendStateVar(t_cnt);
    idxs_cache_id = context->appendStateVar(t_cnt);
    next_cache_id = context->appendStateVar(t_cnt);
    //std::cout << cnt_left_param_id << std::endl;
}


Value * GpuPartitionedHashJoinChained::hash(Value * key){
    IRBuilder<>    *Builder     = context->getBuilder();

    Value * hash = key;

    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));
    hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0x85ebca6b));
    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 13));
    hash = Builder->CreateMul(hash, ConstantInt::get(key->getType(), 0xc2b2ae35));
    hash = Builder->CreateXor(hash, Builder->CreateLShr(hash, 16));
	hash = Builder->CreateLShr(hash, log_parts);
    hash = Builder->CreateAnd(hash, ConstantInt::get(hash->getType(), ((size_t(1)) << hash_bits) - 1));

    return hash;
}


void GpuPartitionedHashJoinChained::generate_joinloop(RawContext* const context) {
    context->setGlobalFunction();

    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    map<string, Value*> kernelBindings;

    Type *int32_type = Type::getInt32Ty(context->getLLVMContext());
    Type *int16_type = Type::getInt16Ty(context->getLLVMContext());

    BasicBlock *MainBB  = BasicBlock::Create(llvmContext, "Main", TheFunction);
    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "After", TheFunction);

    context->setCurrentEntryBlock(Builder->GetInsertBlock());
    Builder->SetInsertPoint(MainBB);    
    context->setEndingBlock(AfterBB);

    Value* mem_buffer = ((GpuRawContext*) context)->getStateVar(buffer_id);
    Value* buffer_offset = Builder->CreateMul(Builder->CreateTrunc(gpu_context->blockId(), int32_type), ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 8*SHMEM_SIZE));
    mem_buffer = Builder->CreateInBoundsGEP(mem_buffer, buffer_offset);

    //Value* shared_keys = mem_buffer;

    //Value* shared_idxs = Builder->CreateInBoundsGEP(mem_buffer, ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), SHMEM_SIZE));

    //Value* shared_next = Builder->CreatePointerCast(Builder->CreateInBoundsGEP(mem_buffer, ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 2*SHMEM_SIZE)), Type::getInt16PtrTy(context->getLLVMContext()));

    //Value* shared_head = Builder->CreateInBoundsGEP(mem_buffer, ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 3*SHMEM_SIZE));
  
    Value *shared_keys = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(int32_type , SHMEM_SIZE),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(int32_type, SHMEM_SIZE)
                                        ),
                                        "keys_shared",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );

    Value *shared_idxs = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(int32_type , SHMEM_SIZE),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(int32_type, SHMEM_SIZE)
                                        ),
                                        "idxs_shared",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );
#ifdef COMPACT_OFFSETS_
    Value *shared_next = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(int16_type , SHMEM_SIZE),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(int16_type, SHMEM_SIZE)
                                        ),
                                        "next_shared",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );
#else
    Value *shared_next = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(int32_type , SHMEM_SIZE),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(int32_type, SHMEM_SIZE)
                                        ),
                                        "next_shared",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );
#endif
	std::cout << (1 << hash_bits) << std::endl;

    Value *shared_head = new GlobalVariable(  *(context->getModule()),
                                        ArrayType::get(int32_type , 1 << hash_bits),
                                        false,
                                        GlobalVariable::LinkageTypes::InternalLinkage,
                                        ConstantAggregateZero::get(
                                            ArrayType::get(int32_type, 1 << hash_bits)
                                        ),
                                        "head_shared",
                                        nullptr,
                                        GlobalVariable::ThreadLocalMode::NotThreadLocal,
                                        3,
                                        false
                                        );

    Value* out_cnts = gpu_context->getStateVar(cnt_right_join);

    kernelBindings["keys_shared"] = shared_keys;
    kernelBindings["idxs_shared"] = shared_idxs;
    kernelBindings["next_shared"] = shared_next;
    kernelBindings["head_shared"] = shared_head;

    Value* mem_buckets_used = ((GpuRawContext*) context)->getStateVar(buckets_used_id);
    Value* mem_bucket_info = ((GpuRawContext*) context)->getStateVar(bucket_info_id);

    Value * buckets_used_val = Builder->CreateLoad(mem_buckets_used);

    BasicBlock *CondBB  = BasicBlock::Create(llvmContext, "PartCond", TheFunction);
    BasicBlock *LoopBB  = BasicBlock::Create(llvmContext, "PartLoop", TheFunction);
    BasicBlock *IncBB  = BasicBlock::Create(llvmContext,  "PartInc", TheFunction);
    BasicBlock *MergeBB  = BasicBlock::Create(llvmContext, "PartAfter", TheFunction);

    Value* bucket_size_val = ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), bucket_size);
    bucket_size_val->setName(opLabel + "_bucket_size");
    kernelBindings["bucket_size"] = bucket_size_val;
    Value* partition_cnt = ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 1 << log_parts);
    partition_cnt->setName(opLabel + "_partition_cnt");
    AllocaInst * mem_partition = context->CreateEntryBlockAlloca(TheFunction, "mem_partition", int32_type);
    Builder->CreateStore(Builder->CreateTrunc(gpu_context->blockId(), int32_type), mem_partition);
    AllocaInst * mem_bucket = context->CreateEntryBlockAlloca(TheFunction, "mem_bucket", int32_type);
    Builder->CreateStore(Builder->CreateTrunc(gpu_context->blockId(), int32_type), mem_bucket);

    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(CondBB);

    Value* current_bucket = Builder->CreateLoad(mem_bucket);
    current_bucket->setName(opLabel + "_current_partition");
    kernelBindings["current_bucket"] = current_bucket;
    Value* bucket_cond = Builder->CreateICmpSLT(current_bucket, buckets_used_val);
    Builder->CreateCondBr(bucket_cond, LoopBB, MergeBB);

    Builder->SetInsertPoint(IncBB);
    Value* next_bucket = Builder->CreateAdd(current_bucket, Builder->CreateTrunc(gpu_context->gridDim(), int32_type));
    next_bucket->setName(opLabel + "_next_partition");
    Builder->CreateStore(next_bucket, mem_bucket);
    Builder->CreateBr(CondBB);

    Builder->SetInsertPoint(LoopBB);

    BasicBlock *IfBB  = BasicBlock::Create(llvmContext, "InitIf", TheFunction);

    Value * current_info = Builder->CreateLoad(Builder->CreateInBoundsGEP(mem_bucket_info, current_bucket)); 
    Builder->CreateCondBr(Builder->CreateICmpNE(
        current_info, 
        ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 0)), 
        IfBB, IncBB);

    Builder->SetInsertPoint(IfBB);

    
    //current_partition = Builder->CreateSub(current_partition, ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), -1));
    kernelBindings["current_info"] = current_info;

    Value * current_partition = Builder->CreateLShr(current_info, 15);

    kernelBindings["current_partition"] = current_partition;

    BasicBlock *InitCondBB  = BasicBlock::Create(llvmContext, "InitCond", TheFunction);
    BasicBlock *InitLoopBB  = BasicBlock::Create(llvmContext, "InitLoop", TheFunction);
    BasicBlock *InitIncBB  = BasicBlock::Create(llvmContext,  "InitInc", TheFunction);
    BasicBlock *InitMergeBB  = BasicBlock::Create(llvmContext, "InitAfter", TheFunction);


    Value * head_ptr = kernelBindings["head_shared"];
    head_ptr->setName(opLabel + "_head_ptr");

    AllocaInst * mem_it = context->CreateEntryBlockAlloca(TheFunction, "mem_it", int32_type);
    Builder->CreateStore(Builder->CreateTrunc(gpu_context->threadIdInBlock(), int32_type), mem_it);
    Builder->CreateBr(InitCondBB);

    Builder->SetInsertPoint(InitCondBB);
    Value * val_it = Builder->CreateLoad(mem_it);
    Value* init_cond = Builder->CreateICmpSLT(val_it, ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 1 << hash_bits));
    Builder->CreateCondBr(init_cond, InitLoopBB, InitMergeBB);

    Builder->SetInsertPoint(InitIncBB);
    Value * next_it = Builder->CreateAdd(val_it, Builder->CreateTrunc(gpu_context->blockDim(), int32_type));
    Builder->CreateStore(next_it, mem_it);
    Builder->CreateBr(InitCondBB);

    Builder->SetInsertPoint(InitLoopBB);
    std::vector<Value *> val_it_v {context->createInt32(0), val_it};
    Builder->CreateStore(ConstantInt::get(int32_type, ~((size_t) 0)), Builder->CreateInBoundsGEP(head_ptr, val_it_v));
    //Builder->CreateStore(ConstantInt::get(int32_type, ~((size_t) 0)), Builder->CreateInBoundsGEP(head_ptr, val_it));     
    Builder->CreateBr(InitIncBB);    

    Builder->SetInsertPoint(InitMergeBB);
    Function * syncthreads = context->getFunction("syncthreads");
    Builder->CreateCall(syncthreads);


    generate_build(context, kernelBindings);
    generate_probe(context, kernelBindings);
    Builder->CreateBr(IncBB);

    Builder->SetInsertPoint(MergeBB);
    Builder->CreateBr(AfterBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(MainBB);

    Builder->SetInsertPoint(context->getEndingBlock());
}

void GpuPartitionedHashJoinChained::generate_build(RawContext* const context, map<string, Value*>& kernelBindings) {
    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();

    GpuRawContext*  const gpu_context = (GpuRawContext *  const) context;

    Type *int16_type = Type::getInt16Ty(context->getLLVMContext());

    Value* bucket_size_val = kernelBindings["bucket_size"];
    Value* current_partition = kernelBindings["current_partition"];
    Value* out_cnts = gpu_context->getStateVar(cnt_left_join);
    Value* chains = gpu_context->getStateVar(chains_left_join);
    Value * head_ptr = kernelBindings["head_shared"];
    //head_ptr->setName(opLabel + "_head_ptr");

    Value* current_cnt_ptr = Builder->CreateInBoundsGEP(out_cnts, current_partition);
    Value* current_cnt = Builder->CreateLoad(current_cnt_ptr);
    current_cnt->setName(opLabel + "_current_cnt_build");

    Value* mem_keys = gpu_context->getStateVar(keys_partitioned_build_id);
    mem_keys->setName(opLabel + "_mem_keys_build");
    Value* mem_idxs = gpu_context->getStateVar(idxs_partitioned_build_id);
    mem_keys->setName(opLabel + "_mem_idxs_build");

    Value* keys_cache = kernelBindings["keys_shared"];
    Value* idxs_cache = kernelBindings["idxs_shared"];
    Value* next_cache = kernelBindings["next_shared"];

    AllocaInst *mem_offset = context->CreateEntryBlockAlloca(TheFunction, "mem_offset_build", current_cnt->getType());
    //Builder->CreateStore(Builder->CreateTrunc(gpu_context->threadIdInBlock(), current_cnt->getType()), mem_offset);
    AllocaInst *mem_base = context->CreateEntryBlockAlloca(TheFunction, "mem_base_build", current_cnt->getType());
    Builder->CreateStore(ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 0), mem_base);
    AllocaInst *mem_bucket = context->CreateEntryBlockAlloca(TheFunction, "mem_bucket_build", current_partition->getType());
    Builder->CreateStore(current_partition, mem_bucket);

    BasicBlock *BlockCondBB  = BasicBlock::Create(llvmContext, "BlockCond", TheFunction);
    BasicBlock *BlockLoopBB  = BasicBlock::Create(llvmContext, "BlockLoop", TheFunction);
    BasicBlock *BlockAfterBB  = BasicBlock::Create(llvmContext, "BlockAfter", TheFunction);
    BasicBlock *BlockIncBB  = BasicBlock::Create(llvmContext, "BlockInc", TheFunction);

    Builder->CreateBr(BlockCondBB);

    Builder->SetInsertPoint(BlockCondBB);
    Value* base = Builder->CreateLoad(mem_base);
    Value* bucket = Builder->CreateLoad(mem_bucket);
    base->setName(opLabel + "_base_offset_build");
    bucket->setName(opLabel + "_bucket_build");
    Value* block_cond = Builder->CreateICmpSLT(base, current_cnt);
    Builder->CreateCondBr(block_cond, BlockLoopBB, BlockAfterBB);


    Builder->SetInsertPoint(BlockIncBB);
    Value* next_base = Builder->CreateAdd(base, bucket_size_val);
    Value* next_bucket = Builder->CreateLoad(Builder->CreateInBoundsGEP(chains, bucket));
    next_base->setName(opLabel + "_next_base_offset_build");
    next_bucket->setName(opLabel + "_next_bucket_build");
    Builder->CreateStore(next_base, mem_base);
    Builder->CreateStore(next_bucket, mem_bucket);
    Builder->CreateBr(BlockCondBB);

    Builder->SetInsertPoint(BlockLoopBB);

    Builder->CreateStore(Builder->CreateTrunc(gpu_context->threadIdInBlock(), current_cnt->getType()), mem_offset);
    Value* block_offset = Builder->CreateMul(bucket_size_val, bucket);
    Value* remaining = Builder->CreateSub(current_cnt, base);
    Value* this_block_size = Builder->CreateSelect(Builder->CreateICmpSLT(remaining, bucket_size_val), remaining, bucket_size_val);

    BasicBlock *InBlockCondBB  = BasicBlock::Create(llvmContext, "InBlockCond", TheFunction);
    BasicBlock *InBlockLoopBB  = BasicBlock::Create(llvmContext, "InBlockLoop", TheFunction);
    BasicBlock *InBlockIncBB  = BasicBlock::Create(llvmContext, "InBlockInc", TheFunction);

    Builder->CreateBr(InBlockCondBB);

    Builder->SetInsertPoint(InBlockCondBB);
    Value* offset = Builder->CreateLoad(mem_offset);
    offset->setName(opLabel + "_inblock_offset_build");
    Value* inblock_cond = Builder->CreateICmpSLT(offset, this_block_size);
    Builder->CreateCondBr(inblock_cond, InBlockLoopBB, BlockIncBB);

    Builder->SetInsertPoint(InBlockIncBB);
    Value* next_offset = Builder->CreateAdd(offset, Builder->CreateTrunc(gpu_context->blockDim(), current_cnt->getType()));
    next_offset->setName(opLabel + "_next_offset_build");

    /*vector<Value*> ArgsV; 
    Value* value = Builder->CreateTrunc(gpu_context->gridDim(), current_cnt->getType());
    ArgsV.push_back(value);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);*/

    Builder->CreateStore(next_offset, mem_offset);
    Builder->CreateBr(InBlockCondBB);

    Builder->SetInsertPoint(InBlockLoopBB);

    Value* absolute_offset = Builder->CreateAdd(block_offset, offset);
    Value* keys_ptr = Builder->CreateInBoundsGEP(mem_keys, absolute_offset);
    Value* idxs_ptr = Builder->CreateInBoundsGEP(mem_idxs, absolute_offset);
    
    Value* wr_offset = Builder->CreateAdd(offset, base);
    std::vector<Value *> wr_offset_v {context->createInt32(0), wr_offset};

    Value* keys_out_ptr = Builder->CreateInBoundsGEP(keys_cache, wr_offset_v);
    Value* next_out_ptr = Builder->CreateInBoundsGEP(next_cache, wr_offset_v);
    Value* idxs_out_ptr = Builder->CreateInBoundsGEP(idxs_cache, wr_offset_v);

    //Value* keys_out_ptr = Builder->CreateInBoundsGEP(keys_cache, wr_offset);
    //Value* next_out_ptr = Builder->CreateInBoundsGEP(next_cache, wr_offset);
    //Value* idxs_out_ptr = Builder->CreateInBoundsGEP(idxs_cache, wr_offset);

    Value* key_val = Builder->CreateLoad(keys_ptr);
    //Value* key_val = gpu_intrinsic::load_cs((GpuRawContext*) context, keys_ptr);
    Value* idx_val = Builder->CreateLoad(idxs_ptr);
    //Value* idx_val = gpu_intrinsic::load_cs((GpuRawContext*) context, idxs_ptr);
    Value* hash_val = GpuPartitionedHashJoinChained::hash(key_val);

    std::vector<Value *> hash_val_v {context->createInt32(0), hash_val};
    Value *old_head = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, 
                                                Builder->CreateInBoundsGEP(head_ptr, hash_val_v),
                                                wr_offset,
                                                llvm::AtomicOrdering::Monotonic);

    //Value *old_head = Builder->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp::Xchg, 
    //                                            Builder->CreateInBoundsGEP(head_ptr, hash_val),
    //                                            wr_offset,
    //                                            llvm::AtomicOrdering::Monotonic);

    /*vector<Value*> ArgsV; 
    Value* value = current_partition;
    ArgsV.push_back(value);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);*/

    Builder->CreateStore(key_val, keys_out_ptr);
    //gpu_intrinsic::store_wb32((GpuRawContext*) context, keys_out_ptr, key_val);
#ifdef COMPACT_OFFSETS_
    Builder->CreateStore(Builder->CreateTrunc(old_head, int16_type), next_out_ptr);
    //gpu_intrinsic::store_wb16((GpuRawContext*) context, next_out_ptr, Builder->CreateTrunc(old_head, int16_type));
#else
    Builder->CreateStore(old_head, next_out_ptr);
    //gpu_intrinsic::store_wb16((GpuRawContext*) context, next_out_ptr, old_head);
#endif
    Builder->CreateStore(idx_val, idxs_out_ptr);
    //gpu_intrinsic::store_wb32((GpuRawContext*) context, idxs_out_ptr, idx_val);

    Builder->CreateBr(InBlockIncBB);


    Builder->SetInsertPoint(BlockAfterBB);

    Function * syncthreads = context->getFunction("syncthreads");
    Builder->CreateCall(syncthreads);
}

void GpuPartitionedHashJoinChained::generate_probe(RawContext* const context,  map<string, Value*>& kernelBindings) {
    IRBuilder<>    *Builder     = context->getBuilder();
    LLVMContext    &llvmContext = context->getLLVMContext();
    Function       *TheFunction = Builder->GetInsertBlock()->getParent();
    GpuRawContext*  const gpu_context = (GpuRawContext * const) context;

    Type *int16_type = Type::getInt16Ty(context->getLLVMContext());

	Value* bucket_size_val = kernelBindings["bucket_size"];
    Value* current_partition = kernelBindings["current_partition"];
    Value* out_cnts = gpu_context->getStateVar(cnt_right_join);
	Value* chains = gpu_context->getStateVar(chains_right_join);
    Value * head_ptr = kernelBindings["head_shared"];
    //head_ptr->setName(opLabel + "_head_ptr_probe");

    Value* current_cnt_ptr = Builder->CreateInBoundsGEP(out_cnts, current_partition);
    Value* current_cnt = Builder->CreateLoad(current_cnt_ptr);
    //current_cnt->setName(opLabel + "_current_cnt_probe");

    Value* mem_keys = gpu_context->getStateVar(keys_partitioned_probe_id);
    mem_keys->setName(opLabel + "_mem_keys_probe");
    Value* mem_idxs = gpu_context->getStateVar(idxs_partitioned_probe_id);
    mem_idxs->setName(opLabel + "_mem_idxs_probe");

    Value* keys_cache = kernelBindings["keys_shared"];
    Value* idxs_cache = kernelBindings["idxs_shared"];
    Value* next_cache = kernelBindings["next_shared"];

    AllocaInst *mem_offset = context->CreateEntryBlockAlloca(TheFunction, "mem_offset_probe", current_cnt->getType());
    AllocaInst *mem_base = context->CreateEntryBlockAlloca(TheFunction, "mem_base_probe", current_cnt->getType());
    Builder->CreateStore(ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 0), mem_base);
    AllocaInst *mem_bucket = context->CreateEntryBlockAlloca(TheFunction, "mem_bucket_probe", current_partition->getType());
    Builder->CreateStore(kernelBindings["current_bucket"], mem_bucket);

    Value * current_info = kernelBindings["current_info"];
    current_cnt = Builder->CreateAnd(current_info, ConstantInt::get(current_cnt->getType(), (1 << 15) - 1));

    BasicBlock *BlockCondBB  = BasicBlock::Create(llvmContext, "BlockCondProbe", TheFunction);
    BasicBlock *BlockLoopBB  = BasicBlock::Create(llvmContext, "BlockLoopProbe", TheFunction);
    BasicBlock *BlockAfterBB  = BasicBlock::Create(llvmContext, "BlockAfterProbe", TheFunction);
    BasicBlock *BlockIncBB  = BasicBlock::Create(llvmContext, "BlockIncProbe", TheFunction);

    Builder->CreateBr(BlockCondBB);

    Builder->SetInsertPoint(BlockCondBB);


    Value* base = Builder->CreateLoad(mem_base);
    Value* bucket = Builder->CreateLoad(mem_bucket);
    base->setName(opLabel + "_base_offset_probe");
    bucket->setName(opLabel + "_bucket_probe");
    Value* block_cond = Builder->CreateICmpSLT(base, current_cnt);
    Builder->CreateCondBr(block_cond, BlockLoopBB, BlockAfterBB);

    Builder->SetInsertPoint(BlockIncBB);
    Value* next_base = Builder->CreateAdd(base, bucket_size_val);
    Value* next_bucket = Builder->CreateLoad(Builder->CreateInBoundsGEP(chains, bucket));
    next_base->setName(opLabel + "_next_base_offset_probe");
    next_bucket->setName(opLabel + "_next_bucket_probe");
    Builder->CreateStore(next_base, mem_base);
    Builder->CreateStore(next_bucket, mem_bucket);

    /*BasicBlock *BlockSingleBB  = BasicBlock::Create(llvmContext, "BlockSingle", TheFunction);
    BasicBlock *BlockAllBB  = BasicBlock::Create(llvmContext, "BlockAll", TheFunction);

    Value* print_cond = Builder->CreateICmpEQ(
                            Builder->CreateTrunc(gpu_context->threadIdInBlock(), current_cnt->getType()), 

                            ConstantInt::get(Type::getInt32Ty(context->getLLVMContext()), 0)
                            );    
    Builder->CreateCondBr(print_cond, BlockSingleBB, BlockAllBB);

    Builder->SetInsertPoint(BlockSingleBB);

    vector<Value*> ArgsV; 
    Value* value = next_bucket;
    ArgsV.push_back(value);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);
    Builder->CreateBr(BlockAllBB);

    Builder->SetInsertPoint(BlockAllBB);*/

    Builder->CreateBr(BlockCondBB);

    Builder->SetInsertPoint(BlockLoopBB);

    Builder->CreateStore(Builder->CreateTrunc(gpu_context->threadIdInBlock(), current_cnt->getType()), mem_offset);
    Value* block_offset = Builder->CreateMul(bucket_size_val, bucket);
    Value* remaining = Builder->CreateSub(current_cnt, base);
    Value* this_block_size = Builder->CreateSelect(Builder->CreateICmpSLT(remaining, bucket_size_val), remaining, bucket_size_val);

    BasicBlock *InBlockCondBB  = BasicBlock::Create(llvmContext, "InBlockCondProbe", TheFunction);
    BasicBlock *InBlockLoopBB  = BasicBlock::Create(llvmContext, "InBlockLoopProbe", TheFunction);
    BasicBlock *InBlockIncBB  = BasicBlock::Create(llvmContext, "InBlockIncProbe", TheFunction);

    Builder->CreateBr(InBlockCondBB);

    Builder->SetInsertPoint(InBlockCondBB);
    Value* offset = Builder->CreateLoad(mem_offset);
    offset->setName(opLabel + "_inblock_offset_probe");

    /*vector<Value*> ArgsV; 
    Value* value = offset;
    ArgsV.push_back(value);
    Function* debugInt = context->getFunction("printi");
    Builder->CreateCall(debugInt, ArgsV);*/

    Value* inblock_cond = Builder->CreateICmpSLT(offset, this_block_size);
    Builder->CreateCondBr(inblock_cond, InBlockLoopBB, BlockIncBB);

    Builder->SetInsertPoint(InBlockIncBB);
    Value* next_offset = Builder->CreateAdd(offset, Builder->CreateTrunc(gpu_context->blockDim(), current_cnt->getType()));
    next_offset->setName(opLabel + "_next_offset_probe");
    Builder->CreateStore(next_offset, mem_offset);
    Builder->CreateBr(InBlockCondBB);

    Builder->SetInsertPoint(InBlockLoopBB);



    Value* absolute_offset = Builder->CreateAdd(block_offset, offset);
    Value* keys_ptr = Builder->CreateInBoundsGEP(mem_keys, absolute_offset);
    Value* idxs_ptr = Builder->CreateInBoundsGEP(mem_idxs, absolute_offset);

    Value* key_val = Builder->CreateLoad(keys_ptr);
    Value* idx_val = Builder->CreateLoad(idxs_ptr);
    //Value* key_val = gpu_intrinsic::load_cs((GpuRawContext*) context, keys_ptr);
    //Value* idx_val = gpu_intrinsic::load_cs((GpuRawContext*) context, idxs_ptr);

    Value* hash_val = GpuPartitionedHashJoinChained::hash(key_val);
    std::vector<Value *> hash_val_v {context->createInt32(0), hash_val};
    Value* first_ptr = Builder->CreateInBoundsGEP(head_ptr, hash_val_v);
    //Value* first_ptr = Builder->CreateInBoundsGEP(head_ptr, hash_val);
    Value* first = Builder->CreateLoad(first_ptr);
    //Value* first = gpu_intrinsic::load_ca((GpuRawContext*) context, first_ptr);

#ifdef COMPACT_OFFSETS_
    first = Builder->CreateTrunc(first, int16_type);
#endif

    AllocaInst *mem_current = context->CreateEntryBlockAlloca(TheFunction, "mem_current_probe", first->getType());
    Builder->CreateStore(first, mem_current);


    BasicBlock *ChainCondBB  = BasicBlock::Create(llvmContext, "chainFollowCond", TheFunction);
    BasicBlock *ChainThenBB  = BasicBlock::Create(llvmContext, "chainFollowThen", TheFunction);
    BasicBlock *ChainMatchBB     = BasicBlock::Create(llvmContext, "matchFollow", TheFunction);
    BasicBlock *ChainIncBB = BasicBlock::Create(llvmContext, "chainFollowInc"   , TheFunction);
    BasicBlock *ChainMergeBB = BasicBlock::Create(llvmContext, "chainFollowCont", TheFunction);

    Builder->CreateBr(ChainCondBB);

    Builder->SetInsertPoint(ChainCondBB);

    Value* current = Builder->CreateLoad(mem_current);
    std::vector<Value *> current_v {context->createInt32(0), current};
    Value * condition_chain = Builder->CreateICmpNE(current, ConstantInt::get(current->getType(), ~((size_t) 0)));
    Builder->CreateCondBr(condition_chain, ChainThenBB, ChainMergeBB);

    Builder->SetInsertPoint(ChainIncBB);
    Value* build_next = Builder->CreateLoad(Builder->CreateInBoundsGEP(next_cache, current_v));
    //Value* build_next = Builder->CreateLoad(Builder->CreateInBoundsGEP(next_cache, current));
    //Value* build_next = gpu_intrinsic::load_ca((GpuRawContext*) context, Builder->CreateInBoundsGEP(next_cache, current));
    Builder->CreateStore(build_next, mem_current);
    Builder->CreateBr(ChainCondBB);

    Builder->SetInsertPoint(ChainThenBB);

    Value* build_key = Builder->CreateLoad(Builder->CreateInBoundsGEP(keys_cache, current_v));
    //Value* build_key = Builder->CreateLoad(Builder->CreateInBoundsGEP(keys_cache, current));
    //Value* build_key = gpu_intrinsic::load_ca((GpuRawContext*)context, Builder->CreateInBoundsGEP(keys_cache, current));
    Value* condition_match = Builder->CreateICmpEQ(key_val, build_key);
    Builder->CreateCondBr(condition_match, ChainMatchBB, ChainIncBB);
    Builder->SetInsertPoint(ChainMatchBB);

    Value* build_idx = Builder->CreateLoad(Builder->CreateInBoundsGEP(idxs_cache, current_v));
    //Value* build_idx = Builder->CreateLoad(Builder->CreateInBoundsGEP(idxs_cache, current));
    //Value* build_idx = gpu_intrinsic::load_ca((GpuRawContext*) context, Builder->CreateInBoundsGEP(idxs_cache, current));

    std::cout << "hi0" << std::endl;


    map<RecordAttribute, RawValueMemory>* allJoinBindings = new map<RecordAttribute, RawValueMemory>();

    if (probe_keyexpr->isRegistered()){
		{ //NOTE: Is there a better way ?
            RawCatalog& catalog             = RawCatalog::getInstance();
            string probeRel                 = probe_keyexpr->getRegisteredRelName();
            Plugin* pg                      = catalog.getPlugin(probeRel);
            assert(pg);
            RecordAttribute * probe_oid     = new RecordAttribute(probeRel, activeLoop, pg->getOIDType());

            PrimitiveType * pr_oid_type = dynamic_cast<PrimitiveType *>(pg->getOIDType());
            if (!pr_oid_type){
                string error_msg("[GpuOptJoin: ] Only primitive OIDs are supported.");
                LOG(ERROR)<< error_msg;
                throw runtime_error(error_msg);
            }

            llvm::Type * llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

            AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                    "mem_" + probeRel + "_" + activeLoop,
                                    llvm_oid_type);

            Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

            RawValueMemory mem_valWrapper;
            mem_valWrapper.mem    = mem_arg;
            mem_valWrapper.isNull = context->createFalse();

            if (allJoinBindings->count(*probe_oid) == 0){
                (*allJoinBindings)[*probe_oid] = mem_valWrapper;
            }
        }

        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                "mem_" +  probe_keyexpr->getRegisteredAttrName(),
                                key_val->getType());

        Builder->CreateStore(key_val, mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();
        (*allJoinBindings)[probe_keyexpr->getRegisteredAs()] = mem_valWrapper;
    }

    if (build_keyexpr->isRegistered()){
		// set activeLoop for build rel if not set (may be multiple ones!)
        { //NOTE: Is there a better way ?
            RawCatalog& catalog             = RawCatalog::getInstance();
            string buildRel                 = build_keyexpr->getRegisteredRelName();
            Plugin* pg                      = catalog.getPlugin(buildRel);
            assert(pg);
            RecordAttribute * build_oid     = new RecordAttribute(buildRel, activeLoop, pg->getOIDType());

            PrimitiveType * pr_oid_type = dynamic_cast<PrimitiveType *>(pg->getOIDType());
            if (!pr_oid_type){
                string error_msg("[GpuOptJoin: ] Only primitive OIDs are supported.");
                LOG(ERROR)<< error_msg;
                throw runtime_error(error_msg);
            }

            llvm::Type * llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

            AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                    "mem_" + buildRel + "_" + activeLoop,
                                    llvm_oid_type);

            Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

            RawValueMemory mem_valWrapper;
            mem_valWrapper.mem    = mem_arg;
            mem_valWrapper.isNull = context->createFalse();

            if (allJoinBindings->count(*build_oid) == 0){
                (*allJoinBindings)[*build_oid] = mem_valWrapper;
            }
        }

        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                "mem_" +  build_keyexpr->getRegisteredAttrName(),
                                build_key->getType());

        Builder->CreateStore(build_key, mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse(); //FIMXE: is this correct ?
        (*allJoinBindings)[build_keyexpr->getRegisteredAs()] = mem_valWrapper;
    }

    //std::cout << "Flag001" << std::endl;

    Value* payload_right_ptr = Builder->CreateInBoundsGEP(gpu_context->getStateVar(probe_param_join_ids[1]), idx_val);
    Value* payload_left_ptr = Builder->CreateInBoundsGEP(gpu_context->getStateVar(build_param_join_ids[1]), build_idx);

    std::cout << "hi1" << std::endl;

    //from probe side
    for (size_t i = 1 ; i < probe_mat_exprs.size(); i++) {
        GpuMatExpr &mexpr = probe_mat_exprs[i];

        { //NOTE: Is there a better way ?
            RawCatalog& catalog             = RawCatalog::getInstance();
            string probeRel                 = mexpr.expr->getRegisteredRelName();
            Plugin* pg                      = catalog.getPlugin(probeRel);
            assert(pg);
            RecordAttribute * probe_oid     = new RecordAttribute(probeRel, activeLoop, pg->getOIDType());

            PrimitiveType * pr_oid_type = dynamic_cast<PrimitiveType *>(pg->getOIDType());
            if (!pr_oid_type){
                string error_msg("[GpuOptJoin: ] Only primitive OIDs are supported.");
                LOG(ERROR)<< error_msg;
                throw runtime_error(error_msg);
            }

            llvm::Type * llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

            AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                    "mem_" + probeRel + "_" + activeLoop,
                                    llvm_oid_type);

            Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

            RawValueMemory mem_valWrapper;
            mem_valWrapper.mem    = mem_arg;
            mem_valWrapper.isNull = context->createFalse();

            if (allJoinBindings->count(*probe_oid) == 0){
                (*allJoinBindings)[*probe_oid] = mem_valWrapper;
            }
        }

        std::cout << mexpr.expr->getRegisteredAttrName() << std::endl;

        /*Value * in_ptr = gpu_context->getStateVar(probe_param_join_ids[i]);

        Value* value_ptr = Builder->CreateInBoundsGEP(in_ptr, idx_val);*/

#ifndef PARTITION_PAYLOAD
        vector<Value*> idxList = vector<Value*>();
        idxList.push_back(context->createInt32(0));
        idxList.push_back(context->createInt32(i-1));

        Value* value_ptr = Builder->CreateInBoundsGEP(payload_right_ptr, idxList);
        Value* value_val = Builder->CreateLoad(value_ptr);
#else
        Value * value_val = idx_val;
#endif


        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                "mem_" +  mexpr.expr->getRegisteredAttrName(),
                                value_val->getType());

        Builder->CreateStore(value_val, mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();

        (*allJoinBindings)[mexpr.expr->getRegisteredAs()] = mem_valWrapper;
    }

    std::cout << "hi3" << std::endl;

    //std::cout << "Flag002" << std::endl;

    //from build side
    for (size_t i = 1 ; i < build_mat_exprs.size(); i++) {
        GpuMatExpr &mexpr = build_mat_exprs[i];

        // set activeLoop for build rel if not set (may be multiple ones!)
        { //NOTE: Is there a better way ?
            RawCatalog& catalog             = RawCatalog::getInstance();
            string buildRel                 = mexpr.expr->getRegisteredRelName();
            Plugin* pg                      = catalog.getPlugin(buildRel);
            assert(pg);
            RecordAttribute * build_oid     = new RecordAttribute(buildRel, activeLoop, pg->getOIDType());

            PrimitiveType * pr_oid_type = dynamic_cast<PrimitiveType *>(pg->getOIDType());
            if (!pr_oid_type){
                string error_msg("[GpuOptJoin: ] Only primitive OIDs are supported.");
                LOG(ERROR)<< error_msg;
                throw runtime_error(error_msg);
            }

            llvm::Type * llvm_oid_type = pr_oid_type->getLLVMType(llvmContext);

            AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                    "mem_" + buildRel + "_" + activeLoop,
                                    llvm_oid_type);

            Builder->CreateStore(UndefValue::get(llvm_oid_type), mem_arg);

            RawValueMemory mem_valWrapper;
            mem_valWrapper.mem    = mem_arg;
            mem_valWrapper.isNull = context->createFalse();

            if (allJoinBindings->count(*build_oid) == 0){
                (*allJoinBindings)[*build_oid] = mem_valWrapper;
            }
        }

        // ExpressionGeneratorVisitor exprGenerator(context, childState);

        /*Value * out_ptr = gpu_context->getStateVar(build_param_join_ids[i]);

        Value* val = Builder->CreateLoad(Builder->CreateInBoundsGEP(out_ptr, build_idx));*/

        std::cout << mexpr.expr->getRegisteredAttrName() << std::endl;

#ifndef PARTITION_PAYLOAD
        vector<Value*> idxList = vector<Value*>();
        idxList.push_back(context->createInt32(0));
        idxList.push_back(context->createInt32(i-1));

        Value* value_ptr = Builder->CreateInBoundsGEP(payload_left_ptr, idxList);
        Value* val = Builder->CreateLoad(value_ptr);
#else
        Value * val = build_idx;
#endif

        AllocaInst * mem_arg = context->CreateEntryBlockAlloca(TheFunction,
                                "mem_" +  mexpr.expr->getRegisteredAttrName(),
                                val->getType());

        Builder->CreateStore(val, mem_arg);

        RawValueMemory mem_valWrapper;
        mem_valWrapper.mem    = mem_arg;
        mem_valWrapper.isNull = context->createFalse();

        (*allJoinBindings)[mexpr.expr->getRegisteredAs()] = mem_valWrapper;
    }

    std::cout << "hi5" << std::endl;    

    OperatorState* newState = new OperatorState(*this, *allJoinBindings);
    getParent()->consume(context, *newState);

    Builder->CreateBr(ChainIncBB);

    Builder->SetInsertPoint(ChainMergeBB);
    
    Builder->CreateBr(InBlockIncBB);

    Builder->SetInsertPoint(BlockAfterBB);

    Function * syncthreads = context->getFunction("syncthreads");
    Builder->CreateCall(syncthreads);
}

void GpuPartitionedHashJoinChained::allocate (RawPipeline * pip) {
    std::cout << "GpuOptJoin::open " << pip->getGroup() << std::endl;

    vector<void*> probe_param_ptr;
    vector<void*> build_param_ptr;
    

    uint32_t   parts2 = 1 << (log_parts1 + log_parts2);
    size_t     buckets_num_max = (((PROBE_SIZE + parts2 - 1)/parts2 + bucket_size - 1)/bucket_size)*parts2;
    size_t     alloca_size = (2 * buckets_num_max + parts2 + 4) * sizeof(int32_t) + parts2 * sizeof(uint64_t);
    size_t     buffer_size = buckets_num_max * bucket_size;

    
    char* alloca_build = (char*) RawMemoryManager::mallocGpu(alloca_size);
    state_left.allocas[pip->getGroup()] = alloca_build;

    char* alloca_probe = (char*) RawMemoryManager::mallocGpu(alloca_size);
    state_right.allocas[pip->getGroup()] = alloca_probe;

    for (size_t i = 0; i < build_mat_exprs.size(); i++) {
        GpuMatExpr& w = build_mat_exprs[i];
        const ExpressionType * out_type = w.expr->getExpressionType();
        Type * llvm_type = ((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext());

        //if (i == 0)
        //    std::cout << "TypeID=" << llvm_type->getPrimitiveSizeInBits() << std::endl;

#ifndef PARTITION_PAYLOAD
        if (i == 0)
            build_param_ptr.emplace_back(RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t)));
        else
            build_param_ptr.emplace_back(RawMemoryManager::mallocGpu(((llvm_type->getPrimitiveSizeInBits()+7)/8) * BUILD_SIZE));
#else
        build_param_ptr.emplace_back(RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t)));
#endif
    }

    for (size_t i = 0; i < probe_mat_exprs.size(); i++) {
        GpuMatExpr& w = probe_mat_exprs[i];
        const ExpressionType * out_type = w.expr->getExpressionType();
        Type * llvm_type = ((const PrimitiveType *) out_type)->getLLVMType(context->getLLVMContext());

        //if (i == 0)
        //    std::cout << "TypeID=" << llvm_type->getPrimitiveSizeInBits() << std::endl;
#ifndef PARTITION_PAYLOAD
        if (i == 0)
            probe_param_ptr.emplace_back(RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t)));
        else
            probe_param_ptr.emplace_back(RawMemoryManager::mallocGpu(((llvm_type->getPrimitiveSizeInBits()+7)/8) * PROBE_SIZE));
#else
        probe_param_ptr.emplace_back(RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t)));
#endif    
    }

    std::cout << build_mat_exprs.size() << std::endl;
    std::cout << probe_mat_exprs.size() << std::endl;

    state_left.cols[pip->getGroup()] = build_param_ptr;
    state_right.cols[pip->getGroup()] = probe_param_ptr;

    PartitionMetadata pmeta;
    pmeta.keys = (int32_t*) state_left.cols[pip->getGroup()][0];
#ifndef PARTITION_PAYLOAD
    pmeta.payload = (int32_t*) RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t));
#else
    pmeta.payload = (int32_t*) state_left.cols[pip->getGroup()][1];
#endif
    pmeta.chains = (uint32_t*) (alloca_build);
    pmeta.bucket_info = (uint32_t*) (alloca_build + (buckets_num_max)*sizeof(int32_t));
    pmeta.out_cnts = (int32_t*) (alloca_build + (2 * buckets_num_max)*sizeof(int32_t));
    pmeta.heads = (uint64_t*) (alloca_build + (2 * buckets_num_max + parts2 + 4)*sizeof(int32_t));
    pmeta.buckets_used = (uint32_t*) (alloca_build + (2 * buckets_num_max + parts2)*sizeof(int32_t));

    state_left.meta[pip->getGroup()] = pmeta;

    std::cout << pmeta.keys << std::endl;

    pmeta.keys = (int32_t*) state_right.cols[pip->getGroup()][0];
#ifndef PARTITION_PAYLOAD
    pmeta.payload = (int32_t*) RawMemoryManager::mallocGpu(buffer_size*sizeof(int32_t));
#else
    pmeta.payload = (int32_t*) state_right.cols[pip->getGroup()][1];
#endif
    pmeta.chains = (uint32_t*) (alloca_probe);
    pmeta.bucket_info = (uint32_t*) (alloca_probe + (buckets_num_max)*sizeof(int32_t));
    pmeta.out_cnts = (int32_t*) (alloca_probe + (2 * buckets_num_max)*sizeof(int32_t));
    pmeta.heads = (uint64_t*) (alloca_probe+ (2 * buckets_num_max + parts2 + 4)*sizeof(int32_t));
    pmeta.buckets_used = (uint32_t*) (alloca_probe + (2 * buckets_num_max + parts2)*sizeof(int32_t));

    state_right.meta[pip->getGroup()] = pmeta;

    std::cout << pmeta.keys << std::endl;
}

void GpuPartitionedHashJoinChained::open (RawPipeline * pip) {
    PartitionMetadata pdata_probe = state_right.meta[pip->getGroup()];
    PartitionMetadata pdata_build = state_left.meta[pip->getGroup()];

    vector<void*> build_param_ptr = state_left.cols[pip->getGroup()];
    vector<void*> probe_param_ptr = state_right.cols[pip->getGroup()];

    buffer[pip->getGroup()] = (int32_t*) RawMemoryManager::mallocGpu(8*SHMEM_SIZE*64*sizeof(int32_t));

    cudaEventCreate(&jstart[pip->getGroup()]);
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    //gpu_run(cudaMemsetAsync(head_ptr, -1, sizeof(uint32_t) * (1 << hash_bits), strm));

    cudaEventRecord(jstart[pip->getGroup()], strm);
    
    //pip->setStateVar(head_id, head_ptr);

    pip->setStateVar(buffer_id, buffer[pip->getGroup()]);

    pip->setStateVar(cnt_left_join, pdata_build.out_cnts);
    pip->setStateVar(cnt_right_join, pdata_probe.out_cnts);

    pip->setStateVar(chains_left_join, pdata_build.chains);
    pip->setStateVar(chains_right_join, pdata_probe.chains);

    pip->setStateVar(keys_partitioned_probe_id, pdata_probe.keys);
    pip->setStateVar(idxs_partitioned_probe_id, pdata_probe.payload);

    pip->setStateVar(keys_partitioned_build_id, pdata_build.keys);
    pip->setStateVar(idxs_partitioned_build_id, pdata_build.payload);

    pip->setStateVar(buckets_used_id, pdata_probe.buckets_used);
    pip->setStateVar(bucket_info_id, pdata_probe.bucket_info);

    for (size_t i = 0; i < build_mat_exprs.size(); i++) {
        pip->setStateVar(build_param_join_ids[i], build_param_ptr[i]);
    }

    for (size_t i = 0; i < probe_mat_exprs.size(); i++) {
        pip->setStateVar(probe_param_join_ids[i], probe_param_ptr[i]);
    }

    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));

    std::cout << "GpuOptJoin::open2" <<  std::endl;
    //sleep(1);
}

void GpuPartitionedHashJoinChained::close (RawPipeline * pip){
    //std::cout << "GpuOptJoin::close" <<  std::endl;

    vector<void*> build_param_ptr = state_left.cols[pip->getGroup()];
    vector<void*> probe_param_ptr = state_right.cols[pip->getGroup()];
    
    RawMemoryManager::freeGpu(buffer[pip->getGroup()]);

    cudaEventCreate(&jstop[pip->getGroup()]);
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    cudaEventRecord(jstop[pip->getGroup()], strm);
    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, jstart[pip->getGroup()], jstop[pip->getGroup()]);
    std::cout << pip->getGroup() << "Timez " << std::dec << milliseconds << std::endl;

    for (size_t i = 1; i < build_mat_exprs.size(); i++){
        RawMemoryManager::freeGpu(build_param_ptr[i]);
    }

    for (size_t i = 1; i < probe_mat_exprs.size(); i++) {
        RawMemoryManager::freeGpu(probe_param_ptr[i]);
    }

    PartitionMetadata pdata_probe = state_right.meta[pip->getGroup()];
    PartitionMetadata pdata_build = state_left.meta[pip->getGroup()];

    RawMemoryManager::freeGpu(pdata_build.keys);
#ifndef PARTITION_PAYLOAD
    RawMemoryManager::freeGpu(pdata_build.payload);
#endif
    RawMemoryManager::freeGpu(pdata_build.chains);

    RawMemoryManager::freeGpu(pdata_probe.keys);
#ifndef PARTITION_PAYLOAD    
    RawMemoryManager::freeGpu(pdata_probe.payload);
#endif
    RawMemoryManager::freeGpu(pdata_probe.chains);
}

 