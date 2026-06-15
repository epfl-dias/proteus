/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "cpu-to-gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include <gflags/gflags.h>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/tracing.hpp>

#include "lib/util/catalog.hpp"
#include "lib/util/jit/olap-pipeline.hpp"

DECLARE_bool(decom_op);
DECLARE_bool(in_gpu);
DECLARE_bool(one_kernel);
DECLARE_bool(pre_decom);
DECLARE_int32(num_threads);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);
DECLARE_string(compression_type);

using namespace llvm;

// Maximum number of storage blocks buffered in batch mode.
static constexpr int kBatchMaxBlocks = 4096;
// Round descriptor stride up to 4 pointers (32B on 64-bit) per block.
static constexpr int kBatchDescPad = 4;
static constexpr bool print_execution_configuration = false;

namespace {
inline bool use_batch_mode() {
  // One-kernel path is opt-in and only meaningful for in-GPU execution.
  return FLAGS_in_gpu && FLAGS_one_kernel;
}

inline bool use_none_like_query_launch_policy() {
  // In --pre_decom mode the decode kernel already ran in PreDecompress::open(),
  // so the subsequent query kernel should launch like the uncompressed path.
  return FLAGS_compression_type == "NONE" || FLAGS_pre_decom;
}
}  // namespace

size_t CpuToGpu::batchDescriptorStride() const {
  // Per block we store one pointer per projected field plus:
  //   - oid pointer
  //   - tuple-count pointer
  const size_t raw_stride = wantedFields.size() + 2;  // + oid + cnt
  // Keep descriptors aligned to a small pointer multiple to simplify
  // downstream indexing and improve copy/coalescing behavior.
  return ((raw_stride + kBatchDescPad - 1) / kBatchDescPad) * kBatchDescPad;
}

void CpuToGpu::ensureBatchDeviceBuffer(const void *pipeline_key,
                                       size_t needed_blocks) {
  const size_t stride = batchDescriptorStride();

  std::lock_guard<std::mutex> lock(batchDeviceBuffersMutex_);
  // Cache one reusable GPU descriptor buffer per pipeline state object.
  // The pipeline pointer is stable across execute() calls for a prepared query.
  auto &entry = batchDeviceBuffers_[pipeline_key];

  if (entry.batch_descs_gpu != nullptr && entry.capacity_blocks >= needed_blocks)
    return;

  int current_device = -1;
  gpu_run(cudaGetDevice(&current_device));

  if (entry.batch_descs_gpu == nullptr) {
    entry.device_id = current_device;
  } else if (entry.device_id != current_device) {
    // Descriptor buffers are CUDA-device specific allocations.
    // Reusing them from another device would be unsafe.
    throw std::runtime_error(
        "CpuToGpu batch descriptor buffer reused from a different CUDA device");
  }

  // Grow geometrically so larger batches amortize cudaMalloc/cudaFree cost.
  // Keep at least kBatchMaxBlocks to match normal batch-mode expectations.
  size_t new_capacity = std::max<size_t>(
      kBatchMaxBlocks,
      std::max<size_t>(needed_blocks, entry.capacity_blocks ? entry.capacity_blocks : 1));
  while (new_capacity < needed_blocks) new_capacity *= 2;

  void *new_buffer_raw = nullptr;
  const size_t new_bytes = new_capacity * stride * sizeof(void *);
  gpu_run(cudaMalloc(&new_buffer_raw, new_bytes));

  if (entry.batch_descs_gpu != nullptr) {
    gpu_run(cudaFree(entry.batch_descs_gpu));
  }

  entry.batch_descs_gpu = static_cast<void **>(new_buffer_raw);
  entry.capacity_blocks = new_capacity;
}

CpuToGpu::BatchDeviceBufferCacheEntry CpuToGpu::getBatchDeviceBuffer(
    const void *pipeline_key) const {
  std::lock_guard<std::mutex> lock(batchDeviceBuffersMutex_);
  auto it = batchDeviceBuffers_.find(pipeline_key);
  if (it == batchDeviceBuffers_.end() || it->second.batch_descs_gpu == nullptr) {
    // Invariant: open() preallocates; close() only fetches an initialized buffer.
    throw std::runtime_error("CpuToGpu batch descriptor buffer not initialized");
  }
  return it->second;
}

CpuToGpu::~CpuToGpu() {
  std::lock_guard<std::mutex> lock(batchDeviceBuffersMutex_);
  for (auto &[pipeline_key, entry] : batchDeviceBuffers_) {
    (void)pipeline_key;
    if (entry.batch_descs_gpu == nullptr) continue;

    int previous_device = -1;
    cudaError_t get_dev_err = cudaGetDevice(&previous_device);
    if (get_dev_err != cudaSuccess) previous_device = -1;

    if (entry.device_id >= 0 && previous_device != entry.device_id) {
      // cudaFree must run in the owning device context for this allocation.
      cudaError_t set_dev_err = cudaSetDevice(entry.device_id);
      if (set_dev_err != cudaSuccess) {
        LOG(WARNING) << "CpuToGpu failed to set CUDA device "
                     << entry.device_id << " while freeing cached descriptor "
                     << "buffer: " << cudaGetErrorString(set_dev_err);
        continue;
      }
    }

    cudaError_t free_err = cudaFree(entry.batch_descs_gpu);
    if (free_err != cudaSuccess) {
      LOG(WARNING) << "CpuToGpu failed to free cached descriptor buffer: "
                   << cudaGetErrorString(free_err);
    }

    if (previous_device >= 0 && entry.device_id >= 0 &&
        previous_device != entry.device_id) {
      // Restore caller's device so operator teardown does not leak device state.
      cudaError_t restore_dev_err = cudaSetDevice(previous_device);
      if (restore_dev_err != cudaSuccess) {
        LOG(WARNING) << "CpuToGpu failed to restore CUDA device "
                     << previous_device << ": "
                     << cudaGetErrorString(restore_dev_err);
      }
    }

    entry.batch_descs_gpu = nullptr;
    entry.capacity_blocks = 0;
  }
  batchDeviceBuffers_.clear();
  LOG(INFO) << "Collapsing CpuToGpu operator";
}

void CpuToGpu::produce_(OlapParallelContext *context) {
  generateGpuSide(context);

  context->popPipeline();

  gpu_pip = context->removeLatestPipeline();

  context->pushDeviceProvider<OlapCpuPipelineGenFactory>();
  context->pushPipeline(gpu_pip);

  const bool batch_mode = use_batch_mode();

  context->registerOpen(this, [this, batch_mode](Pipeline *pip) {
    event_range<range_log_op::CPU2GPU_OPEN> er{id, gpu_pip->getUUID(),
                                               pip->getGroup()};
    auto strm = createNonBlockingStream();
    pip->setStateVar<void *>(this->childVar_id, gpu_pip->getKernel());
    pip->setStateVar<decltype(strm)>(this->strmVar_id, strm);

    if (batch_mode) {
      const size_t stride = this->batchDescriptorStride();
      void **buf =
          static_cast<void **>(malloc(kBatchMaxBlocks * stride * sizeof(void *)));
      pip->setStateVar<void *>(this->batchBufVar_id, buf);

      auto *count = new int64_t(0);
      pip->setStateVar<int64_t *>(this->batchCountVar_id, count);

      auto *ss_cache = new void *(nullptr);
      pip->setStateVar<void **>(this->subStateCacheVar_id, ss_cache);

      // Allocate once per pipeline and reuse across execute() calls.
      this->ensureBatchDeviceBuffer(pip, kBatchMaxBlocks);
    }
  });

  context->registerClose(this, [this, batch_mode](Pipeline *pip) {
    event_range<range_log_op::CPU2GPU_CLOSE> er{id, gpu_pip->getUUID(),
                                                pip->getGroup()};
    if (batch_mode) {
      void **batch_descs_cpu = pip->getStateVar<void **>(this->batchBufVar_id);
      int64_t num_blocks = *pip->getStateVar<int64_t *>(this->batchCountVar_id);
      void *sub_state = *pip->getStateVar<void **>(this->subStateCacheVar_id);
      cudaStream_t strm = pip->getStateVar<cudaStream_t>(this->strmVar_id);
      CUfunction entry = reinterpret_cast<CUfunction>(
          pip->getStateVar<void *>(this->childVar_id));

      if (num_blocks > 0) {
        // Ensure cached GPU descriptor storage can hold this close() batch.
        this->ensureBatchDeviceBuffer(pip, static_cast<size_t>(num_blocks));
        auto batch_cache = this->getBatchDeviceBuffer(pip);

        int current_device = -1;
        gpu_run(cudaGetDevice(&current_device));
        if (batch_cache.device_id >= 0 && batch_cache.device_id != current_device) {
          // Protect against using a cached buffer from a different active device.
          throw std::runtime_error(
              "CpuToGpu batch descriptor buffer used from a different CUDA "
              "device");
        }

        const size_t stride = this->batchDescriptorStride();
        const size_t buf_bytes =
            static_cast<size_t>(num_blocks) * stride * sizeof(void *);
        void **batch_descs_gpu = batch_cache.batch_descs_gpu;

        // Only descriptor metadata is copied each run; the GPU buffer itself is reused.
        gpu_run(cudaMemcpyAsync(batch_descs_gpu, batch_descs_cpu, buf_bytes,
                                cudaMemcpyHostToDevice, strm));

        void *kernel_args[2] = {&batch_descs_gpu, sub_state};

        const unsigned int pip_block_x = gpu_pip->getMaxBlockSize();

        const bool use_none_like_launch_policy =
            use_none_like_query_launch_policy();
        // Generic launch policy:
        //   NONE / pre_decom query kernel -> 1024 threads
        //   compressed late-decode kernel -> trust pipeline worker shape
        //   fallback -> keep existing default/override behavior
        unsigned int block_x = 0u;
        if (use_none_like_launch_policy) {
          block_x = 1024u;
        } else if (pip_block_x > 0u) {
          block_x = pip_block_x;
        } else {
          block_x = (FLAGS_num_threads > 0)
                        ? static_cast<unsigned>(FLAGS_num_threads)
                        : defaultBlockDim.x;
        }

        // Compressed paths honor pipeline-selected worker caps.
        // Uncompressed execution keeps the historical 1024-thread launch.
        if (!use_none_like_launch_policy && pip_block_x > 0u) {
          block_x = std::min(block_x, pip_block_x);
        }

        int kernel_max_threads = 0;
        gpu_run(cuFuncGetAttribute(&kernel_max_threads,
                                   CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                   entry));
        if (kernel_max_threads > 0) {
          block_x = std::min(block_x,
                             static_cast<unsigned int>(kernel_max_threads));
        }

        const unsigned int pip_grid_x = gpu_pip->getMaxGridSize();
        unsigned int grid_x = defaultGridDim.x;
        if (use_none_like_launch_policy) {
          // NONE keeps CTA width at 1024 threads, but grid width follows
          // topology defaults to avoid over-launching CTAs. --pre_decom uses
          // the same launch policy for the query kernel because decompression
          // is already paid by the separate PreDecompress kernel.
          grid_x = defaultGridDim.x;
        } else if (pip_grid_x > 0u) {
          // Compressed one-kernel launches use the pipeline-selected grid cap
          // directly so reduced CTA widths can scale launch parallelism.
          grid_x = pip_grid_x;
        }
        if (grid_x == 0u) {
          grid_x = 1u;
        }

        const dim3 blockDim(block_x, 1, 1);
        const dim3 gridDim2D(grid_x, static_cast<unsigned>(num_blocks), 1);
        if (print_execution_configuration) {
          // Query ptxas/JIT-resolved kernel resources only when explicitly
          // printing the launch configuration to avoid adding hot-path cost.
          int kernel_num_regs = 0;
          int kernel_static_smem_bytes = 0;
          int kernel_local_bytes = 0;
          gpu_run(cuFuncGetAttribute(&kernel_num_regs,
                                     CU_FUNC_ATTRIBUTE_NUM_REGS, entry));
          gpu_run(cuFuncGetAttribute(&kernel_static_smem_bytes,
                                     CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                     entry));
          gpu_run(cuFuncGetAttribute(&kernel_local_bytes,
                                     CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                     entry));

          int occ_blocks_per_sm = 0;
          gpu_run(cuOccupancyMaxActiveBlocksPerMultiprocessor(
              &occ_blocks_per_sm, entry, static_cast<int>(block_x),
              /*dynamicSMemSize=*/0));
          const int occ_warps_per_sm =
              occ_blocks_per_sm * static_cast<int>((block_x + 31u) / 32u);
          int max_threads_per_sm = 0;
          int warp_size = 32;
          gpu_run(cudaDeviceGetAttribute(
              &max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
              current_device));
          gpu_run(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize,
                                         current_device));
          const int max_warps_per_sm =
              (max_threads_per_sm > 0 && warp_size > 0)
                  ? (max_threads_per_sm / warp_size)
                  : 0;
          const double occ_theoretical_pct =
              (max_warps_per_sm > 0)
                  ? (100.0 * static_cast<double>(occ_warps_per_sm) /
                     static_cast<double>(max_warps_per_sm))
                  : 0.0;

          LOG(INFO) << "[CpuToGpu] one-kernel launch "
                    << "compression=" << FLAGS_compression_type
                    << " pre_decom=" << FLAGS_pre_decom
                    << " query_launch_mode="
                    << (use_none_like_launch_policy ? "none_like"
                                                    : "compressed")
                    << " block_x=" << block_x
                    << " grid_x=" << gridDim2D.x
                    << " grid_y(num_blocks)=" << gridDim2D.y
                    << " jit_compiled_kernel_regs=" << kernel_num_regs
                    << " kernel_static_smem_bytes="
                    << kernel_static_smem_bytes
                    << " kernel_local_bytes=" << kernel_local_bytes
                    << " occ_blocks_per_sm=" << occ_blocks_per_sm
                    << " occ_warps_per_sm=" << occ_warps_per_sm
                    << " occ_theoretical_pct=" << occ_theoretical_pct
                    << " threads_per_block_override="
                    << FLAGS_threads_per_block
                    << " items_per_thread_override="
                    << FLAGS_items_per_thread;
        }
        launch_kernel(entry, reinterpret_cast<void **>(kernel_args), gridDim2D,
                      blockDim, strm);

        syncAndDestroyStream(strm);
      } else {
        syncAndDestroyStream(strm);
      }

      free(batch_descs_cpu);
      delete pip->getStateVar<int64_t *>(this->batchCountVar_id);
      delete pip->getStateVar<void **>(this->subStateCacheVar_id);
    } else {
      syncAndDestroyStream(pip->getStateVar<cudaStream_t>(this->strmVar_id));
    }
  });

  LLVMContext &llvmContext = context->getLLVMContext();
  Type *charPtrType = Type::getInt8PtrTy(llvmContext);

  childVar_id = context->appendStateVar(charPtrType);
  strmVar_id = context->appendStateVar(charPtrType);

  if (batch_mode) {
    batchBufVar_id = context->appendStateVar(charPtrType);
    batchCountVar_id = context->appendStateVar(
        PointerType::get(Type::getInt64Ty(llvmContext), 0));
    subStateCacheVar_id =
        context->appendStateVar(PointerType::get(charPtrType, 0));
  }

  getChild()->produce(context);

  context->popDeviceProvider();
}

void CpuToGpu::generateGpuSide(OlapParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  const bool batch_mode = use_batch_mode();
  if (batch_mode) {
    Type *charPtrType = Type::getInt8PtrTy(llvmContext);
    Type *i64_ty = Type::getInt64Ty(llvmContext);
    Type *i8pp_ty = PointerType::get(charPtrType, 0);

    size_t batchDescsArgId =
        context->appendParameter(i8pp_ty, /*noalias=*/false, /*readonly=*/false);

    context->setGlobalFunction();

    IRBuilder<> *Builder = context->getBuilder();
    BasicBlock *insBB = Builder->GetInsertBlock();
    Function *F = insBB->getParent();

    BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "end", F);
    BasicBlock *MainBB = BasicBlock::Create(llvmContext, "main", F);

    context->setCurrentEntryBlock(insBB);
    context->setEndingBlock(AfterBB);

    Plugin *pg =
        Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
    IntegerType *oid_type =
        static_cast<IntegerType *>(pg->getOIDType()->getLLVMType(llvmContext));
    const std::string relName = wantedFields[0]->getRelationName();
    const int raw_stride = static_cast<int>(wantedFields.size()) + 2;
    const int64_t stride =
        (raw_stride + kBatchDescPad - 1) / kBatchDescPad * kBatchDescPad;

    Value *batch_descs = context->getArgument(batchDescsArgId);
    batch_descs->setName("batch_descs");

    Builder->SetInsertPoint(MainBB);

    Function *ctaid_y_fn =
        context->getFunction("llvm.nvvm.read.ptx.sreg.ctaid.y");
    Value *block_id = Builder->CreateZExt(
        Builder->CreateCall(ctaid_y_fn, {}, "ctaid_y"), i64_ty, "block_id");
    Value *base_off =
        Builder->CreateMul(block_id, ConstantInt::get(i64_ty, stride));

    map<RecordAttribute, ProteusValueMemory> variableBindings;

    for (size_t i = 0; i < wantedFields.size(); ++i) {
      Value *off =
          Builder->CreateAdd(base_off, ConstantInt::get(i64_ty, (int64_t)i));
      Value *slot_ptr = Builder->CreateGEP(charPtrType, batch_descs, off);
      Value *col_raw = Builder->CreateLoad(charPtrType, slot_ptr);

      Type *field_type =
          wantedFields[i]->getOriginalType()->getLLVMType(llvmContext);
      Value *col_ptr = Builder->CreateBitCast(col_raw, field_type);

      AllocaInst *mem = context->CreateEntryBlockAlloca(
          wantedFields[i]->getAttrName() + "_ptr", col_ptr->getType());
      Builder->CreateStore(col_ptr, mem);

      ProteusValueMemory tmp;
      tmp.mem = mem;
      tmp.isNull = context->createFalse();
      variableBindings[*(wantedFields[i])] = tmp;
    }

    {
      Value *off = Builder->CreateAdd(
          base_off, ConstantInt::get(i64_ty, (int64_t)wantedFields.size()));
      Value *slot_ptr = Builder->CreateGEP(charPtrType, batch_descs, off);
      Value *oid_raw = Builder->CreateLoad(charPtrType, slot_ptr);
      Value *oid = Builder->CreatePtrToInt(oid_raw, oid_type);

      AllocaInst *oid_mem =
          context->CreateEntryBlockAlloca("batch_oid_ptr", oid_type);
      Builder->CreateStore(oid, oid_mem);

      RecordAttribute tupleOID(relName, activeLoop, pg->getOIDType());
      ProteusValueMemory tmp;
      tmp.mem = oid_mem;
      tmp.isNull = context->createFalse();
      variableBindings[tupleOID] = tmp;
    }

    {
      Value *off = Builder->CreateAdd(
          base_off,
          ConstantInt::get(i64_ty, (int64_t)(wantedFields.size() + 1)));
      Value *slot_ptr = Builder->CreateGEP(charPtrType, batch_descs, off);
      Value *cnt_raw = Builder->CreateLoad(charPtrType, slot_ptr);
      Value *cnt = Builder->CreatePtrToInt(cnt_raw, oid_type);

      AllocaInst *cnt_mem =
          context->CreateEntryBlockAlloca("batch_cnt_ptr", oid_type);
      Builder->CreateStore(cnt, cnt_mem);

      RecordAttribute tupleCnt(relName, "activeCnt", pg->getOIDType());
      ProteusValueMemory tmp;
      tmp.mem = cnt_mem;
      tmp.isNull = context->createFalse();
      variableBindings[tupleCnt] = tmp;
    }

    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    Builder->CreateBr(AfterBB);

    Builder->SetInsertPoint(context->getCurrentEntryBlock());
    Builder->CreateBr(MainBB);

    Builder->SetInsertPoint(context->getEndingBlock());
    return;
  }

  std::vector<size_t> wantedFieldsArg_id;
  for (const auto &tin : wantedFields) {
    Type *t = tin->getOriginalType()->getLLVMType(llvmContext);

    wantedFieldsArg_id.emplace_back(context->appendParameter(t, true, true));
  }

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  IntegerType *oid_type =
      (IntegerType *)pg->getOIDType()->getLLVMType(llvmContext);
  size_t tupleOIDArg_id = context->appendParameter(oid_type, false, false);
  size_t tupleCntArg_id = context->appendParameter(oid_type, false, false);

  context->setGlobalFunction();

  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "end", F);
  BasicBlock *MainBB = BasicBlock::Create(llvmContext, "main", F);

  context->setCurrentEntryBlock(Builder->GetInsertBlock());
  context->setEndingBlock(AfterBB);

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    Value *arg = context->getArgument(wantedFieldsArg_id[i]);
    arg->setName(wantedFields[i]->getAttrName());
    AllocaInst *mem = context->CreateEntryBlockAlloca(
        F, wantedFields[i]->getAttrName() + "_ptr", arg->getType());
    Builder->CreateStore(arg, mem);

    ProteusValueMemory tmp;
    tmp.mem = mem;
    tmp.isNull = context->createFalse();

    variableBindings[*(wantedFields[i])] = tmp;
  }

  {
    RecordAttribute tupleOID =
        RecordAttribute(wantedFields[0]->getRelationName(), activeLoop,
                        pg->getOIDType());  // FIXME: OID type for blocks ?

    Value *oid = context->getArgument(tupleOIDArg_id);
    oid->setName("oid");
    AllocaInst *mem =
        context->CreateEntryBlockAlloca(F, "activeLoop_ptr", oid->getType());
    Builder->CreateStore(oid, mem);

    ProteusValueMemory tmp;
    tmp.mem = mem;
    tmp.isNull = context->createFalse();

    variableBindings[tupleOID] = tmp;
  }

  {
    RecordAttribute tupleCnt =
        RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                        pg->getOIDType());  // FIXME: OID type for blocks ?

    Value *N = context->getArgument(tupleCntArg_id);
    N->setName("cnt");
    AllocaInst *mem =
        context->CreateEntryBlockAlloca(F, "activeCnt_ptr", N->getType());
    Builder->CreateStore(N, mem);

    // Function * printi = context->getFunction("printi64");
    // Builder->CreateCall(printi, std::vector<Value *>{N});

    ProteusValueMemory tmp;
    tmp.mem = mem;
    tmp.isNull = context->createFalse();

    variableBindings[tupleCnt] = tmp;
  }

  Builder->SetInsertPoint(MainBB);
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

void CpuToGpu::consume(OlapParallelContext *const context,
                       const OperatorState &childState) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  BasicBlock *insBB = Builder->GetInsertBlock();
  Function *F = insBB->getParent();

  Type *charPtrType = Type::getInt8PtrTy(llvmContext);
  Type *i64_ty = Type::getInt64Ty(llvmContext);

  const auto &activeVars = childState.getBindings();

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());

  RecordAttribute tupleOID(wantedFields[0]->getRelationName(), activeLoop,
                           pg->getOIDType());
  ProteusValueMemory mem_oidWrapper = activeVars.at(tupleOID);

  RecordAttribute tupleCnt(wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType());
  ProteusValueMemory mem_cntWrapper = activeVars.at(tupleCnt);

  if (use_batch_mode()) {
    Builder->SetInsertPoint(insBB);

    Value *buf_raw = context->getStateVar(batchBufVar_id);
    Value *buf = Builder->CreateBitCast(buf_raw,
                                        PointerType::get(charPtrType, 0));

    Value *count_ptr = context->getStateVar(batchCountVar_id);
    Value *count = Builder->CreateLoad(i64_ty, count_ptr, "batch_count");

    const int raw_stride = static_cast<int>(wantedFields.size()) + 2;
    const int64_t stride =
        (raw_stride + kBatchDescPad - 1) / kBatchDescPad * kBatchDescPad;
    Value *base_off =
        Builder->CreateMul(count, ConstantInt::get(i64_ty, stride));

    for (size_t i = 0; i < wantedFields.size(); ++i) {
      auto it = activeVars.find(*(wantedFields[i]));
      assert(it != activeVars.end());
      ProteusValueMemory mem_val = it->second;

      Value *col_val = Builder->CreateLoad(
          mem_val.mem->getType()->getNonOpaquePointerElementType(), mem_val.mem);
      Value *off =
          Builder->CreateAdd(base_off, ConstantInt::get(i64_ty, (int64_t)i));
      Value *slot = Builder->CreateGEP(charPtrType, buf, off);
      Builder->CreateStore(Builder->CreateBitCast(col_val, charPtrType), slot);
    }

    {
      Value *oid_val = Builder->CreateLoad(
          mem_oidWrapper.mem->getType()->getNonOpaquePointerElementType(),
          mem_oidWrapper.mem);
      Value *off = Builder->CreateAdd(
          base_off, ConstantInt::get(i64_ty, (int64_t)wantedFields.size()));
      Value *slot = Builder->CreateGEP(charPtrType, buf, off);
      Builder->CreateStore(Builder->CreateIntToPtr(oid_val, charPtrType), slot);
    }

    {
      Value *cnt_val = Builder->CreateLoad(
          mem_cntWrapper.mem->getType()->getNonOpaquePointerElementType(),
          mem_cntWrapper.mem);
      Value *off = Builder->CreateAdd(
          base_off,
          ConstantInt::get(i64_ty, (int64_t)(wantedFields.size() + 1)));
      Value *slot = Builder->CreateGEP(charPtrType, buf, off);
      Builder->CreateStore(Builder->CreateIntToPtr(cnt_val, charPtrType), slot);
    }

    {
      Value *ss_cache = context->getStateVar(subStateCacheVar_id);
      Builder->CreateStore(context->getSubStateVar(), ss_cache);
    }

    Value *new_count = Builder->CreateAdd(count, ConstantInt::get(i64_ty, 1));
    Builder->CreateStore(new_count, count_ptr);
    return;
  }

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  Type *ptr_t = PointerType::get(charPtrType, 0);

  Type *kernel_params_type = ArrayType::get(
      charPtrType, wantedFields.size() + 3);  // input + N + oid + state

  Value *kernel_params = UndefValue::get(kernel_params_type);
  Value *kernel_params_addr =
      context->CreateEntryBlockAlloca(F, "gpu_params", kernel_params_type);

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    auto it = activeVars.find(*(wantedFields[i]));
    assert(it != activeVars.end());
    ProteusValueMemory mem_valWrapper = it->second;

    kernel_params = Builder->CreateInsertValue(
        kernel_params, Builder->CreateBitCast(mem_valWrapper.mem, charPtrType),
        i);
  }

  kernel_params = Builder->CreateInsertValue(
      kernel_params, Builder->CreateBitCast(mem_oidWrapper.mem, charPtrType),
      wantedFields.size());

  kernel_params = Builder->CreateInsertValue(
      kernel_params, Builder->CreateBitCast(mem_cntWrapper.mem, charPtrType),
      wantedFields.size() + 1);

  Value *subState = context->getSubStateVar();

  kernel_params = Builder->CreateInsertValue(kernel_params, subState,
                                             wantedFields.size() + 2);

  Builder->CreateStore(kernel_params, kernel_params_addr);

  Value *entry = context->getStateVar(childVar_id);
  Value *strm = context->getStateVar(strmVar_id);

  Builder->SetInsertPoint(insBB);

  // Launch GPU kernel.
  // Uncompressed and pre-decompressed query execution keep the historical
  // 1024-thread CTA launch. In --pre_decom mode, CLI launch-shape overrides
  // apply to the separate decompression kernel, not to the query kernel.
  const bool use_none_like_launch_policy = use_none_like_query_launch_policy();
  unsigned int launch_block_dim =
      use_none_like_launch_policy ? 1024u : gpu_pip->getMaxBlockSize();
  unsigned int launch_grid_dim =
      use_none_like_launch_policy ? defaultGridDim.x : gpu_pip->getMaxGridSize();
  if (launch_grid_dim == 0u) launch_grid_dim = defaultGridDim.x;

  auto block_dim = ConstantInt::get(Type::getInt32Ty(llvmContext),
                                    launch_block_dim);
  auto grid_dim = ConstantInt::get(Type::getInt32Ty(llvmContext),
                                   launch_grid_dim);
  context->gen_call(
      "launch_kernel_strm_sized",
      {entry, Builder->CreateBitCast(kernel_params_addr, ptr_t), strm,
       block_dim, grid_dim});
}
