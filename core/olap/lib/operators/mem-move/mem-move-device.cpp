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

#include "mem-move-device.hpp"

#include <gflags/gflags.h>
#include <lz4.h>

#include <atomic>
#include <codegen/jit/pipeline.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/timing.hpp>
#include <platform/util/tracing.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include <cufile.h>
#pragma clang diagnostic pop

#include "compression.hpp"
#include "lib/util/catalog.hpp"
#include "olap/plugins/binary-block-nvme-plugin.hpp"

DECLARE_string(compression_type);
DECLARE_bool(decom_shared);
DECLARE_bool(decom_register);
DECLARE_bool(use_metadata_in_block);
DECLARE_int32(tile_size);

#undef BATCHING
//#define BATCHING

buff_pair buff_pair::not_moved(proteus::managed_ptr buff) {
  return {std::move(buff), nullptr};
}

bool buff_pair::moved() const { return !old_buff; }

proteus::managed_ptr MemMoveDevice::MemMoveConf::force_push(
    const proteus::managed_ptr &src, size_t bytes, int target_device,
    uint64_t srcServer, cudaStream_t movestrm) {
  // FIXME: buffer manager should be able to provide blocks of arbitrary size
  assert(bytes <= BlockManager::block_size);
  auto buff = BlockManager::h_get_buffer(target_device);

  if (bytes > 0) {
    BlockManager::overwrite_bytes(buff.get(), src.get(), bytes, movestrm,
                                  false);
  }

  return buff;
}

proteus::managed_ptr MemMoveDevice::MemMoveConf::force_push_from_nvme(
    const proteus::managed_ptr &src, int target_device, cudaStream_t movestrm,
    workunit *wu) {
  const auto page_id = NvmePlugin::PageId_t::from_ptr(src.get());
  DCHECK_NE(nvme_plugin, nullptr);
  const auto page_io_info = nvme_plugin->getPageIoInfo(page_id);
  // --decom_shared + --use_metadata_in_block enables deferred in-query decompression:
  // MemMove only stages compressed bytes and CompressedBlockToTuples decodes.
  // If --use_metadata_in_block=false, MemMove keeps legacy decompression.
  const bool use_codec_metadata_in_block =
      FLAGS_use_metadata_in_block &&
      (FLAGS_compression_type == "RFOR" || FLAGS_compression_type == "FLBP");
  const bool use_late_decompression = FLAGS_decom_shared || FLAGS_decom_register;
  const bool defer_decompression_here =
      enable_deferred_rfor_decode && use_late_decompression &&
      use_codec_metadata_in_block;

  proteus::managed_ptr buff;
  if (do_transfer[wu->index_in_wu]) {
    /// "direct" load to data local to the target device
    buff = BlockManager::h_get_buffer(target_device);

  } else {
    DCHECK_LT(target_device, 0)
        << "force_push_from_nvme cannot have do_transfer[false] for a GPU. "
           "Staging for a GPU should move to CPU memory";
    /// "staging"
    /// load to memory local to nvme drive
    auto &nvme_cpu_aff =
        topology::getInstance().getCpuNumaNodes()[page_id.getCpuNumaAffinity()];
    buff = BlockManager::get_buffer_numa(nvme_cpu_aff);
  }

  //  auto buff_numa =
  //  topology::getInstance().getCpuNumaNodeAddressed(buff.get()); LOG(INFO) <<
  //  "moving col: " << std::to_string(wu->index_in_wu)
  //            << " with a thread on: " << affinity::get().id
  //            << " moving a buffer from nvme: "
  //            << std::to_string(page_id.getCpuNumaAffinity())
  //            << " to a buffer on " << buff_numa->id;

  // Deferred path: stage compressed bytes and decode in
  // CompressedBlockToTuples.
  if (page_io_info.is_compressed && defer_decompression_here) {
    DCHECK_LE(*page_io_info.size, BlockManager::block_size);
    DCHECK(*page_io_info.size % 4_K == 0);

    if (target_device >= 0) {
      // GPU target: direct cuFile read of compressed bytes
      event_range<range_log_op::CUFILE_READ> er{{}};
      ssize_t bytes_read =
          cuFileRead(page_io_info.cufile_handle, buff.get(), *page_io_info.size,
                     *page_io_info.offset, 0);
      CHECK_GT(bytes_read, 0) << "cuFileRead failed with: " << bytes_read;
    } else {
      // CPU target: IO uring read of compressed bytes, no decompression
      wu->complete += 1;
      proteus::storage::IoUringThreadUnsafe::CompletionCallBackSuccess cb =
          [wu] {
            std::atomic_fetch_add_explicit(&wu->complete, -1,
                                           std::memory_order_relaxed);
          };
      io_uring->read(page_io_info.fd, buff.get(), *page_io_info.size,
                     *page_io_info.offset, cb);
    }
    return buff;
  }

  if (page_io_info.is_compressed && target_device < 0) {
    // CPU target and compressed case
    // allocate temporary storage for IO target
    // decompress into the target buffer
    char *compressed_buff =
        static_cast<char *>(std::aligned_alloc(4096, *page_io_info.size));
    auto decompressed_buff = static_cast<char *>(buff.get());
    wu->complete += 1;

    // Get val_section_sizes for RFOR decompression
    auto val_section_sizes = nvme_plugin->getValSectionSizes(page_id);

    // decompressed buff will have a lifetime exceeding the callback, as the
    // catcher spins on wu->complete.
    // compressed buf ownership is moved into the lambda which frees it
    auto cb = [page_id, wu, compressed_buff, io_size = *page_io_info.size,
               max_decomp_chunk_size = page_io_info.decompressed_chunk_size,
               chunk_sizes = std::move(page_io_info.chunk_sizes),
               val_section_sizes = std::move(val_section_sizes),
               decompressed_buff = decompressed_buff] {
      auto decomp_span =
          std::span<char>(decompressed_buff, BlockManager::block_size);
      auto comp_span = std::span<char>(compressed_buff, io_size);


      int decomp_res;
      if (!val_section_sizes.empty()) {
       decomp_res = decompress_rfor_block(chunk_sizes, comp_span, decomp_span,
                                         max_decomp_chunk_size, val_section_sizes);
      } else {
        decomp_res = decompress_block(chunk_sizes, comp_span, decomp_span,
                                    max_decomp_chunk_size);
      }


      // auto decomp_res = decompress_block(chunk_sizes, comp_span, decomp_span,
      //                                    max_decomp_chunk_size);

      CHECK_GT(decomp_res, 0)
          << "failed to decompress block for page: " << page_id;
      std::free(compressed_buff);
      std::atomic_fetch_add_explicit(&wu->complete, -1,
                                     std::memory_order_relaxed);
      DCHECK_GE(wu->complete, 0);
    };
    auto read_size = *page_io_info.size;
    if (read_size % 512 != 0) {
      read_size += 512 - (read_size % 512);  // align to 512 bytes for O_DIRECT
    }
    DCHECK_GE(read_size, 0);
    DCHECK_EQ(read_size % 512, 0);
    DCHECK_GE(*page_io_info.offset, 0);
    DCHECK_EQ(*page_io_info.offset % 512, 0);

    io_uring->read(page_io_info.fd, compressed_buff, read_size,
                   *page_io_info.offset, std::move(cb));

  } else {
    DCHECK_LE(*page_io_info.size, BlockManager::block_size);
    DCHECK(*page_io_info.size % 4_K == 0);

    if (target_device >= 0) {
      // GPU case
      // cuFiles does not respect event synchronization. Hence, each WU has its
      // own stream
      //      static off_t buff_offset = 0;
      // required lifetimes of pointers passed to cuFileReadAsync is unknown
      //      CUfileError_t status = cuFileReadAsync(
      //          page_io_info.cufile_handle, buff.get(),
      //          const_cast<size_t *>(page_io_info.size),
      //          const_cast<off_t *>(page_io_info.offset), &buff_offset,
      //          &(wu->bytes_read[wu->index_in_wu]), wu->cufile_strm);
      //      CHECK_EQ(status.err, CU_FILE_SUCCESS)
      //          << "cuFileReadAsync failed with: "
      //          << cufileop_status_error(status.err);
      event_range<range_log_op::CUFILE_READ> er{{}};
      ssize_t bytes_read =
          cuFileRead(page_io_info.cufile_handle, buff.get(), *page_io_info.size,
                     *page_io_info.offset, 0);
      CHECK_GT(bytes_read, 0) << "cuFileRead failed with: " << bytes_read;
    } else {
      // CPU case
      wu->complete += 1;

      // TODO really should probably pass the result of IO to the callback
      proteus::storage::IoUringThreadUnsafe::CompletionCallBackSuccess cb =
          [wu] {
            std::atomic_fetch_add_explicit(&wu->complete, -1,
                                           std::memory_order_relaxed);
          };

      io_uring->read(page_io_info.fd, buff.get(), *page_io_info.size,
                     *page_io_info.offset, cb);
    }
  }
  return buff;
}

buff_pair MemMoveDevice::MemMoveConf::push(proteus::managed_ptr src,
                                           size_t bytes, int target_device,
                                           uint64_t srcServer, workunit *wu) {
  DCHECK_EQ(srcServer, 0);
  if (NvmePlugin::PageId_t::isPageIdPtr(src.get())) {
    auto buff = force_push_from_nvme(src, target_device, strm, wu);
    src.release();
    return buff_pair::not_moved(
        std::move(buff));  // src is not a pointer and does not need freeing
  } else {
    // regular memory to memory movement
    const auto *d = topology::getInstance().getGpuAddressed(src.get());
    int dev = d ? static_cast<int>(d->id) : -1;

    if (dev == target_device) {
      return buff_pair::not_moved(std::move(src));  // block in correct device
    }

    auto buff = force_push(src, bytes, target_device, srcServer, strm);
    return buff_pair{std::move(buff), std::move(src)};
  }
}


[[maybe_unused]]
static void *cudaMallocAligned(void** ptr, size_t size) {
   const size_t alignment = 4096;
   void* raw_ptr;
   cudaMalloc(&raw_ptr, size + alignment);
   size_t misalignment = (size_t)raw_ptr % alignment;
   if (misalignment == 0) {
    *ptr = raw_ptr;
    return raw_ptr;
   }
   size_t adjustment = alignment - misalignment;
   void* aligned_ptr = (void*)((char*)raw_ptr + adjustment);
   *ptr = aligned_ptr;
   return raw_ptr;
}

//static std::atomic_int n_blocks = 0;


std::vector<buff_pair> MemMoveDevice::MemMoveConf::batch_push_nvme_to_gpu(
    const std::vector<proteus::managed_ptr> &src, size_t bytes,
    int target_device, uint64_t srcServer, workunit *wu) {
  set_device_on_scope d(topology::getInstance().getGpus()[target_device]);
  std::vector<buff_pair> buff_pairs;
  buff_pairs.reserve(src.size());
  const bool use_codec_metadata_in_block =
      FLAGS_use_metadata_in_block &&
      (FLAGS_compression_type == "RFOR" || FLAGS_compression_type == "FLBP");
  const bool use_late_decompression = FLAGS_decom_shared || FLAGS_decom_register;
  const bool defer_decompression_here =
      enable_deferred_rfor_decode && use_late_decompression &&
      nvme_plugin != nullptr &&
      use_codec_metadata_in_block;

  // Deferred path: stage compressed bytes and decode in
  // CompressedBlockToTuples.
  if (defer_decompression_here) {

    for (uint8_t i = 0; i < src.size(); i++) {
      buff_pairs.emplace_back(
          buff_pair::not_moved(force_push_from_nvme(src[i], target_device, strm,
                                                    wu)));
    }
    return buff_pairs;
  }

  std::vector<std::vector<uint32_t>> block_chunk_sizes;
  std::vector<std::vector<uint32_t>> block_val_section_sizes;
  std::vector<std::span<char>> block_compressed_buffers;
  std::vector<std::span<char>> block_decompressed_buffers;

  static off_t buff_offset =
      0;  // offset for cuFileReadAsync to write into block

  for (uint8_t i = 0; i < src.size(); i++) {
    DCHECK(NvmePlugin::PageId_t::isPageIdPtr(src[i].get())) << std::to_string(i);
    if (!do_transfer[0]) {
      buff_pairs.emplace_back(
          buff_pair::not_moved(force_push_from_nvme(src[i], -1, strm, wu)));
      continue;
    }
    auto page_id = NvmePlugin::PageId_t::from_ptr(src[i].get());
    const auto page_io_info = nvme_plugin->getPageIoInfo(page_id);

    if (!page_io_info.is_compressed) {
      buff_pairs.emplace_back(buff_pair::not_moved(
          force_push_from_nvme(src[i], target_device, strm, wu)));
      continue;
    }

    // NVMe to GPU with compression
    // in this loop we do IO and set up buffers
    DCHECK(do_transfer[0]);
    DCHECK(page_io_info.is_compressed);
    auto decomp_buff = BlockManager::h_get_buffer(target_device);
    CHECK_LT(static_cast<size_t>(target_device), wu->compressed_buffers->size());
    CHECK_LT(i, wu->compressed_buffers->at(target_device).size())
        << "Missing compressed scratch buffer for compressed NVMe->GPU path. "
        << "compression_type=" << FLAGS_compression_type
        << " decom_shared=" << FLAGS_decom_shared
        << " decom_register=" << FLAGS_decom_register
        << " use_metadata_in_block=" << FLAGS_use_metadata_in_block;
    // Staging scratch for this (GPU, field) slot:
    // cuFile reads compressed bytes into comp_buff, and decompressor emits
    // uncompressed tuples into decomp_buff.
    void *comp_buff = wu->compressed_buffers->at(target_device)[i];
    DCHECK_EQ(((uintptr_t)comp_buff) % 4096, 0);

    auto decomp_span = std::span<char>(static_cast<char *>(decomp_buff.get()),
                                       BlockManager::block_size);
    auto comp_span =
        std::span<char>(static_cast<char *>(comp_buff), *page_io_info.size);
    block_compressed_buffers.emplace_back(comp_span);
    block_decompressed_buffers.emplace_back(decomp_span);
    block_chunk_sizes.emplace_back(page_io_info.chunk_sizes);

    auto val_section_sizes = nvme_plugin->getValSectionSizes(page_id);
    block_val_section_sizes.emplace_back(val_section_sizes);

    event_range<range_log_op::CUFILE_READ> er{{}};
    ssize_t bytes_read =
        cuFileRead(page_io_info.cufile_handle, comp_buff, *page_io_info.size,
                   *page_io_info.offset, 0);

    CHECK_GT(bytes_read, 0) << "cuFileRead failed with: " << bytes_read;
    buff_pairs.emplace_back(buff_pair::not_moved(std::move(decomp_buff)));
  }

  if (!block_chunk_sizes.empty()) {
    CHECK_LT(static_cast<size_t>(target_device), wu->decompressors->size())
        << "Missing GPU decompressor for compressed NVMe->GPU path. "
        << "compression_type=" << FLAGS_compression_type;
    int res = (*wu->decompressors)[target_device].batch_decompress_block_gpu(
        block_chunk_sizes, block_compressed_buffers, block_decompressed_buffers,
        wu->cufile_strm, block_val_section_sizes);
    CHECK_GE(res, 0) << "batch decompression failed";
  }

  return buff_pairs;
}

extern "C" {
void make_mem_move_device(char **src_ptrs, size_t *bytes, int target_device,
                          uint64_t srcServer, MemMoveDevice::MemMoveConf *mmc,
                          int num_buffers, pb *pair_buffs,
                          MemMoveDevice::workunit *wu) noexcept {
  //const static auto prt = profiling::ProfileRegionType("memmove:::make_mmd");
  //auto prof_reg = profiling::ProfileRegion(prt);
  wu->complete = 0;
  wu->index_in_wu = 0;

  if (NvmePlugin::PageId_t::isPageIdPtr(src_ptrs[0]) && target_device >= 0) {
    // for to GPU we do a batch push, this is for the compressed case where we
    // want to do batch decompression
    std::vector<proteus::managed_ptr> src_ptrs_vec;
    src_ptrs_vec.reserve(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      src_ptrs_vec.emplace_back(src_ptrs[i]);
    }
    auto moved_pair_buffs = mmc->batch_push_nvme_to_gpu(
        src_ptrs_vec, *bytes, target_device, srcServer, wu);
    for (int i = 0; i < num_buffers; i++) {
      auto &x = moved_pair_buffs[i];
      pair_buffs[i] = {x.new_buff.release(), x.old_buff.release()};
      src_ptrs_vec[i].release();
    }
  } else {
    for (int i = 0; i < num_buffers; i++) {
      auto x = mmc->push(proteus::managed_ptr{src_ptrs[i]}, bytes[i],
                         target_device, srcServer, wu);
      pair_buffs[i] = {x.new_buff.release(), x.old_buff.release()};
    }
  }

  // if we have an mmc->io_uring instance it means we have some CPU IO
  // so we submit the operations we just constructed
  if (mmc->io_uring) {
    mmc->io_uring->submit();
  }
}

MemMoveDevice::workunit *acquireWorkUnit(
    MemMoveDevice::MemMoveConf *mmc) noexcept {
  return mmc->acquire();
}

void propagateWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                       MemMoveDevice::workunit *buff, bool is_noop) noexcept {
  mmc->propagate(buff, is_noop);
}

void tran_empty(MemMoveDevice::MemMoveConf *mmc, int32_t *empty) noexcept {
  *empty = mmc->tran.empty();
}

void print_tran_empty(int32_t *empty) noexcept {
  //  std::cerr << "[DEBUG] tran empty: " << *empty << std::endl;
}

void print_wu_complete(MemMoveDevice::workunit *wu) noexcept {
  std::cerr << "[DEBUG] wu complete: " << wu->complete << std::endl;
}

void print_current_batch(int32_t batch) noexcept {
  std::cerr << "[DEBUG] current batch: " << batch << std::endl;
}

void print_current_block(int32_t currblock) noexcept {
  std::cerr << "[DEBUG] current block: " << currblock << std::endl;
}

void print_hey() noexcept {
  std::cerr << "[DEBUG] hey " << std::endl;
}

void print_hey1() noexcept {
  std::cerr << "[DEBUG] hey1 " << std::endl;
}

void print_hey2() noexcept {
  std::cerr << "[DEBUG] hey2 " << std::endl;
}

void print_hey3() noexcept {
  std::cerr << "[DEBUG] hey3 " << std::endl;
}

void print_hey4() noexcept {
  std::cerr << "[DEBUG] hey4 " << std::endl;
}

void print_hey5() noexcept {
  std::cerr << "[DEBUG] hey5 " << std::endl;
}

}

void MemMoveDevice::genReleaseOldBuffer(OlapParallelContext *context,
                                        llvm::Value *src) const {
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());
  context->gen_call(release_buffer,
                    {context->getBuilder()->CreateBitCast(src, charPtrType)});
}

#ifdef BATCHING

static void createStateVarPtr(StateVar& var, OlapParallelContext *context,
                              llvm::Type *base_type, llvm::Type *ptr_type,
                              llvm::Value *init_val) {
  var = context->appendStateVar(
      ptr_type,
      [=](llvm::Value* pip) -> llvm::Value* {
        // Be careful to use ParallelContext::allocateStateVar, but not
        // Context::allocateStateVar. Note that the
        // ParallelContext::allocateStateVar redirects calls to the {Cpu,
        // Gpu}PipelineGen, so it is safe to use it
        auto mem = context->allocateStateVar(base_type);
        context->getBuilder()->CreateStore(init_val, mem);
        return mem;
      },
      [=](llvm::Value* pip, llvm::Value* state_var) {
        context->deallocateStateVar(state_var);
      });
}
#endif

void MemMoveDevice::produce_(OlapParallelContext *context) {
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = llvm::Type::getInt32Ty(context->getLLVMContext());
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  auto oidType = pg->getOIDType()->getLLVMType(llvmContext);

  std::vector<llvm::Type *> tr_types;
  for (auto wantedField : wantedFields) {
    tr_types.push_back(wantedField->getLLVMType(llvmContext));
    tr_types.push_back(
        wantedField->getLLVMType(llvmContext));  // old buffer, to be released
  }
  tr_types.push_back(oidType);  // cnt
  tr_types.push_back(oidType);  // oid

  data_type = llvm::StructType::get(llvmContext, tr_types);

  RecordAttribute tupleCnt =
      RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?
  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType());

  // Generate catch code
  int p = context->appendParameter(llvm::PointerType::get(data_type, 0), true,
                                   true);
  context->setGlobalFunction();

  auto Builder = context->getBuilder();
  auto entryBB = Builder->GetInsertBlock();
  auto F = entryBB->getParent();

  auto mainBB = llvm::BasicBlock::Create(llvmContext, "main", F);

  auto endBB = llvm::BasicBlock::Create(llvmContext, "end", F);
  context->setEndingBlock(endBB);

  Builder->SetInsertPoint(entryBB);

  auto params = Builder->CreateLoad(
      context->getArgument(p)->getType()->getPointerElementType(),
      context->getArgument(p));

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    auto param = Builder->CreateExtractValue(params, 2 * i);

    auto src = Builder->CreateExtractValue(params, 2 * i + 1);

    genReleaseOldBuffer(context, src);

    variableBindings[*(wantedFields[i])] =
        context->toMem(param, context->createFalse());
  }

  auto cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size());

  variableBindings[tupleCnt] = context->toMem(cnt, context->createFalse());

  auto oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);

  variableBindings[tupleIdentifier] =
      context->toMem(oid, context->createFalse());

  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  Builder->SetInsertPoint(mainBB);

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  Builder->CreateBr(endBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(mainBB);

  Builder->SetInsertPoint(context->getEndingBlock());

  context->popPipeline();

  catch_pip = context->removeLatestPipeline();

  // push new pipeline for the throw part
  context->pushPipeline();

  device_id_var = context->appendStateVar(int32_type);
  memmvconf_var = context->appendStateVar(charPtrType);

#ifdef BATCHING

  LOG(INFO) << "batch_size: " << batch_size;

  /* Code for batching decompression */
  auto int32_ptr_type = llvm::Type::getInt32PtrTy(context->getLLVMContext());
  createStateVarPtr(batch_cnt_ptr, context, int32_type, int32_ptr_type,
                    context->createInt32(0));

//  createStateVarPtr(tran_empty_ptr, context, int32_type, int32_ptr_type,
//                    context->createInt32(-1));

  // Create the array type ([(batch_size * wantedFields.size()) x i8*])
  llvm::Type *int8_ptr_type = llvm::Type::getInt8PtrTy(context->getLLVMContext());
  llvm::Type *int8_ptr_array_type = llvm::ArrayType::get(int8_ptr_type, batch_size * wantedFields.size());
  llvm::PointerType *int8_ptr_array_ptr_type = llvm::PointerType::get(int8_ptr_array_type, 0);
  createStateVarPtr(mv_src_ptrs_arr_ptr, context, int8_ptr_array_type, int8_ptr_array_ptr_type,
                    llvm::Constant::getNullValue(int8_ptr_array_type));

  // Create the array type ([(batch_size * wantedFields.size()) x i64])
  llvm::Type *int64_type = llvm::Type::getInt64Ty(context->getLLVMContext());
  llvm::Type *int64_array_type = llvm::ArrayType::get(int64_type, batch_size * wantedFields.size());
  llvm::PointerType *int64_array_ptr_type = llvm::PointerType::get(int64_array_type, 0);
  createStateVarPtr(mv_sizes_bytes_arr_ptr, context, int64_array_type, int64_array_ptr_type,
                    llvm::Constant::getNullValue(int64_array_type));

  // Create the array type ([batch_size x i64])
  int64_array_type = llvm::ArrayType::get(int64_type, batch_size);
  int64_array_ptr_type = llvm::PointerType::get(int64_array_type, 0);
  createStateVarPtr(num_tuples_arr_ptr, context, int64_array_type, int64_array_ptr_type,
                    llvm::Constant::getNullValue(int64_array_type));
  createStateVarPtr(oid_arr_ptr, context, int64_array_type, int64_array_ptr_type,
                    llvm::Constant::getNullValue(int64_array_type));

  /* End of code for batching decompression */
#endif

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

ProteusValueMemory MemMoveDevice::getServerId(
    OlapParallelContext *context, const OperatorState &childState) const {
  return context->toMem(context->createInt64(0), context->createFalse());
}

void MemMoveDevice::consume(OlapParallelContext *context,
                            const OperatorState &childState) {
#ifdef BATCHING
  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();
  // TODO a bit of a hack to not manipulate tuple cnts for non-scan moves
  const bool non_scan_move =
      (wantedFields[0]->getRelationName().find("tmp") != std::string::npos) ||
      (wantedFields[0]->getRelationName().find("Pelago") != std::string::npos);

  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto workunit_type = llvm::StructType::get(
      llvmContext, std::vector<llvm::Type *>{charPtrType, charPtrType});

  // Find block size
  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper = childState[tupleCnt];
  ProteusValueMemory mem_srcServer = getServerId(context, childState);
  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto device_id = ((OlapParallelContext *)context)->getStateVar(device_id_var);

  // Begin inserting code
  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(
      mem_cntWrapper.mem->getType()->getPointerElementType(),
      mem_cntWrapper.mem);
  auto srcS = Builder->CreateLoad(
      mem_srcServer.mem->getType()->getPointerElementType(), mem_srcServer.mem);

  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};

  ProteusValueMemory mem_oidWrapper = childState[tupleIdentifier];
  llvm::Value *oid = Builder->CreateLoad(
      mem_oidWrapper.mem->getType()->getPointerElementType(),
      mem_oidWrapper.mem);

  llvm::Value *memmv =
      ((OlapParallelContext *)context)->getStateVar(memmvconf_var);

  auto *F = context->getGlobalFunction();
  auto int64_array_type =
      llvm::ArrayType::get(context->createSizeType(), wantedFields.size() * batch_size);
//  llvm::AllocaInst *mv_sizes_bytes = context->CreateEntryBlockAlloca(
//      F, "mv_sizes_bytes", int64_array_type, nullptr);

  auto mv_sizes_bytes_ptrs_val = ((OlapParallelContext *)context)->getStateVar(mv_sizes_bytes_arr_ptr);

  llvm::ArrayType *charptr_array_type = llvm::ArrayType::get(charPtrType, wantedFields.size() * batch_size);

  auto mv_src_ptrs_val = ((OlapParallelContext *)context)->getStateVar(mv_src_ptrs_arr_ptr);

  llvm::Value *is_noop = context->createTrue();

  // store size in bytes, and src_ptr into in the above declared arrays for each
  // field to be moved
  auto *page_id_mask = llvm::ConstantInt::get(context->createSizeType(),
                                              llvm::APInt(64, 1).shl(63));
  llvm::Value *any_ptr_is_page_id = context->createFalse();

  auto batch_cnt_ptr_val = ((OlapParallelContext *)context)->getStateVar(batch_cnt_ptr);

  auto batch_cnt_val = Builder->CreateLoad(
      batch_cnt_ptr_val->getType()->getPointerElementType(),
      batch_cnt_ptr_val, "batch_cnt");

  // block_index = (batch_cnt % batch_size) * wantedFields.size()
  auto batch_mod = Builder->CreateAnd(batch_cnt_val, context->createInt32(batch_size - 1), "batch_mod");

  auto block_index = Builder->CreateMul(
      batch_mod,
      context->createInt32(wantedFields.size()), "block_index");

  auto block_index_iter = block_index;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute block_attr(*(wantedFields[i]), true);

    ProteusValueMemory mem_valWrapper = childState[block_attr];

    auto mv_block_type = mem_valWrapper.mem->getType()
                             ->getPointerElementType()
                             ->getPointerElementType();

    llvm::Value *size = llvm::ConstantInt::get(
        llvmContext, llvm::APInt(64, context->getSizeOf(mv_block_type)));
    auto Nloc = Builder->CreateZExtOrBitCast(N, size->getType());
    size = Builder->CreateMul(size, Nloc);  /// now in bytes
    llvm::Value *size_as_int64 = Builder->CreateIntCast(
        size, llvm::Type::getInt64Ty(llvmContext), false);

    // same index into both sizes and ptrs arrays
    llvm::Value *gep_indices_batch[] = {
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 0),
        block_index_iter
    };

    llvm::Value *size_ptr =
        Builder->CreateGEP(int64_array_type, mv_sizes_bytes_ptrs_val, gep_indices_batch, "size_ptr");
    Builder->CreateStore(size_as_int64, size_ptr);

    auto mv_src_ptr = Builder->CreateBitCast(
        Builder->CreateLoad(
            mem_valWrapper.mem->getType()->getPointerElementType(),
            mem_valWrapper.mem),
        charPtrType, "mv_src_ptr");

    auto *src_ptr =
        Builder->CreateGEP(charptr_array_type, mv_src_ptrs_val, gep_indices_batch, "src_ptr");
    Builder->CreateStore(mv_src_ptr, src_ptr);

    auto *intPtr = Builder->CreatePtrToInt(
        mv_src_ptr, llvm::Type::getInt64Ty(context->getLLVMContext()));
    auto *and_res = Builder->CreateAnd(intPtr, page_id_mask);
    any_ptr_is_page_id = Builder->CreateOr(
        any_ptr_is_page_id, Builder->CreateICmpEQ(and_res, page_id_mask));

    block_index_iter = Builder->CreateAdd(block_index_iter, context->createInt32(1), "block_index");
  }

  {
    if (!non_scan_move) {
      // if we have anyPage Ids we need to update tupleCnt to store a tupleCnt
      // instead of blockCnt, which the NvmePlugin stores in `tupleCnt`
      RecordAttribute actualTupleCnt{wantedFields[0]->getRelationName(),
                                     "tupleCnt", pg->getOIDType()};
      if (childState.contains(actualTupleCnt)) {
        auto ifAnyPageId = context->gen_if({any_ptr_is_page_id});

        auto thenAnyIsPageId = std::move(ifAnyPageId)([&]() {  // NOLINT
          ProteusValueMemory mem_tupleCntWrapper = childState[actualTupleCnt];
          llvm::Value *mem_tupleCnt = Builder->CreateLoad(
              mem_tupleCntWrapper.mem->getType()->getPointerElementType(),
              mem_tupleCntWrapper.mem);
          //      context->log(mem_tupleCnt);
          Builder->CreateStore(mem_tupleCnt, mem_cntWrapper.mem);
        });
      }
    }
    // note, all of this is a bit hacky, we ignore size in the above and below
    // loops for page ids, but we could actually just store the size/fd/offset
    // in the Operator state
  }

  auto *num_tuples = Builder->CreateLoad(
      mem_cntWrapper.mem->getType()->getPointerElementType(),
      mem_cntWrapper.mem);

  llvm::Value *gep_indices_batch[] = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 0),
      batch_mod
  };

  int64_array_type =
      llvm::ArrayType::get(context->createSizeType(), batch_size);
  /* Insert into tuple_cnt_arr.  */
  auto *num_tuples_arr_ptr_val = ((OlapParallelContext *)context)->getStateVar(num_tuples_arr_ptr);
  auto *num_tuples_arr_ptr1 = Builder->CreateGEP(int64_array_type, num_tuples_arr_ptr_val, gep_indices_batch, "num_tuples_arr");
  // num_tuples_arr_ptr1 is of type [batch_size x i64], cast it to i64*
  num_tuples_arr_ptr1 = Builder->CreateBitCast(num_tuples_arr_ptr1, llvm::Type::getInt64PtrTy(llvmContext));
  Builder->CreateStore(num_tuples, num_tuples_arr_ptr1);

  /* Insert oid into oid_arr_ptr.  */
  auto *oid_arr_ptr_val = ((OlapParallelContext *)context)->getStateVar(oid_arr_ptr);
  auto *oid_arr_ptr1 = Builder->CreateGEP(int64_array_type, oid_arr_ptr_val, gep_indices_batch, "oid_arr");
  // oid_arr_ptr1 is of type [batch_size x i64], cast it to i64*
  oid_arr_ptr1 = Builder->CreateBitCast(oid_arr_ptr1, llvm::Type::getInt64PtrTy(llvmContext));
  Builder->CreateStore(oid, oid_arr_ptr1);

  /* Update batch_cnt.  */
  auto new_batch_cnt = Builder->CreateAdd(batch_cnt_val, context->createInt32(1), "new_batch_cnt");
  Builder->CreateStore(new_batch_cnt, batch_cnt_ptr_val);

  /* If block_count % batch_size == 0 continue, else return.  */
  batch_mod = Builder->CreateAnd(new_batch_cnt, context->createInt32(batch_size - 1), "batch_mod");
  auto to_continue = Builder->CreateICmpEQ(batch_mod, context->createInt32(0));
  auto continue_bb = llvm::BasicBlock::Create(context->getLLVMContext(), "continue", F);
  Builder->CreateCondBr(to_continue, continue_bb, context->getEndingBlock());
  Builder->SetInsertPoint(continue_bb);

  // create an array of `pb` to pass to make_mem_move_device as a return
  // argument
  auto mv_num_bufs = context->createInt32(wantedFields.size() * batch_size);
  auto *llvm_pb_type = context->toLLVM<std::remove_cv_t<pb>>();
  llvm::ArrayType *llvm_pb_array_type =
      llvm::ArrayType::get(llvm_pb_type, wantedFields.size() * batch_size);

  llvm::AllocaInst *pb_array =
      Builder->CreateAlloca(llvm_pb_array_type, nullptr, "pb_array");

  auto workunit_ptr8 = context->gen_call(acquireWorkUnit, {memmv});
  auto workunit_ptr = Builder->CreateBitCast(
      workunit_ptr8, llvm::PointerType::getUnqual(workunit_type));

  context->gen_call(make_mem_move_device,
                    {mv_src_ptrs_val, mv_sizes_bytes_ptrs_val, device_id, srcS, memmv,
                     mv_num_bufs, pb_array, workunit_ptr});

  std::vector<llvm::Value *> pushed[batch_size];
  int gep_idx = 0;
  for (size_t i = 0; i < batch_size; i++) {
    for (auto wantedField : wantedFields) {
      RecordAttribute block_attr(*wantedField, true);
      ProteusValueMemory mem_valWrapper = childState[block_attr];

      llvm::Value *gep_index_moved[] = {context->createInt32(0),
                                          context->createInt32(gep_idx),
                                          context->createInt32(0)};
      llvm::Value *gep_index_to_release[] = {context->createInt32(0),
                                              context->createInt32(gep_idx),
                                              context->createInt32(1)};

        llvm::Value *moved_ptr = Builder->CreateGEP(
            pb_array->getType()->getPointerElementType(), pb_array,
            gep_index_moved, "buffpair_structs_moved");
        llvm::Value *to_release_ptr = Builder->CreateGEP(
                pb_array->getType()->getPointerElementType(), pb_array,
                gep_index_to_release, "buffpair_structs_to_release");
        llvm::Value *moved =
                Builder->CreateLoad(charPtrType, moved_ptr, "load_moved");
        llvm::Value *to_release =
                Builder->CreateLoad(charPtrType, to_release_ptr, "load_to_release");

        pushed[i].push_back(Builder->CreateBitCast(
            moved, mem_valWrapper.mem->getType()->getPointerElementType()));
        pushed[i].push_back(Builder->CreateBitCast(
            to_release,
            mem_valWrapper.mem->getType()->getPointerElementType()));
        gep_idx++;
    }
    is_noop = context->createFalse();
  }

  // Get the number of tuples and the oid from state.
  std::vector<llvm::Value *> to_store(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    gep_indices_batch[1] = context->createInt32(i);

    num_tuples_arr_ptr1 = Builder->CreateGEP(int64_array_type, num_tuples_arr_ptr_val, gep_indices_batch, "num_tuples_arr");
    num_tuples = Builder->CreateLoad(num_tuples_arr_ptr1->getType()->getPointerElementType(), num_tuples_arr_ptr1, "num_tuples");

    oid_arr_ptr1 = Builder->CreateGEP(int64_array_type, oid_arr_ptr_val, gep_indices_batch, "oid_arr");
    oid = Builder->CreateLoad(oid_arr_ptr1->getType()->getPointerElementType(), oid_arr_ptr1, "oid");
    pushed[i].push_back(num_tuples);
    pushed[i].push_back(oid);

    llvm::Value *d = llvm::UndefValue::get(data_type);
    for (size_t p = 0; p < pushed[i].size(); ++p) {
      d = Builder->CreateInsertValue(d, pushed[i][p], p);
    }
    to_store[i] = d;
  }

  auto workunit_dat = Builder->CreateLoad(
      workunit_ptr->getType()->getPointerElementType(), workunit_ptr);
  auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
  d_ptr =
      Builder->CreateBitCast(d_ptr, llvm::PointerType::getUnqual(data_type));

  for (size_t i = 0; i < batch_size; ++i) {
    Builder->CreateStore(to_store[i], Builder->CreateGEP(data_type, d_ptr, context->createInt32(i)));
  }

  // finally propagate the workunit
  context->gen_call(propagateWorkUnit, {memmv, workunit_ptr8, is_noop});

#else



  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();
  // TODO a bit of a hack to not manipulate tuple cnts for non-scan moves
  const bool non_scan_move =
      (wantedFields[0]->getRelationName().find("tmp") != std::string::npos) ||
      (wantedFields[0]->getRelationName().find("Pelago") != std::string::npos);

  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto workunit_type = llvm::StructType::get(
      llvmContext, std::vector<llvm::Type *>{charPtrType, charPtrType});

  // Find block size
  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper = childState[tupleCnt];
  ProteusValueMemory mem_srcServer = getServerId(context, childState);
  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto device_id = ((OlapParallelContext *)context)->getStateVar(device_id_var);

  // Begin inserting code
  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(
      mem_cntWrapper.mem->getType()->getPointerElementType(),
      mem_cntWrapper.mem);
  auto srcS = Builder->CreateLoad(
      mem_srcServer.mem->getType()->getPointerElementType(), mem_srcServer.mem);

  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};

  ProteusValueMemory mem_oidWrapper = childState[tupleIdentifier];
  llvm::Value *oid = Builder->CreateLoad(
      mem_oidWrapper.mem->getType()->getPointerElementType(),
      mem_oidWrapper.mem);

  llvm::Value *memmv =
      ((OlapParallelContext *)context)->getStateVar(memmvconf_var);

  std::vector<llvm::Value *> pushed;
  llvm::Value *is_noop = context->createTrue();

  auto *F = context->getGlobalFunction();
  llvm::ArrayType *int64_array_type =
      llvm::ArrayType::get(context->createSizeType(), wantedFields.size());
  llvm::AllocaInst *mv_sizes_bytes = context->CreateEntryBlockAlloca(
      F, "mv_sizes_bytes", int64_array_type, nullptr);

  llvm::ArrayType *charptr_array_type =
      llvm::ArrayType::get(charPtrType, wantedFields.size());
  llvm::AllocaInst *mv_src_ptrs = context->CreateEntryBlockAlloca(
      F, "mv_src_ptrs", charptr_array_type, nullptr);

  // store size in bytes, and src_ptr into in the above declared arrays for each
  // field to be moved
  auto *page_id_mask = llvm::ConstantInt::get(context->createSizeType(),
                                              llvm::APInt(64, 1).shl(63));
  llvm::Value *any_ptr_is_page_id = context->createFalse();

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute block_attr(*(wantedFields[i]), true);

    ProteusValueMemory mem_valWrapper = childState[block_attr];

    auto mv_src_ptr = Builder->CreateBitCast(
        Builder->CreateLoad(
            mem_valWrapper.mem->getType()->getPointerElementType(),
            mem_valWrapper.mem),
        charPtrType);

    auto mv_block_type = mem_valWrapper.mem->getType()
                             ->getPointerElementType()
                             ->getPointerElementType();

    llvm::Value *size = llvm::ConstantInt::get(
        llvmContext, llvm::APInt(64, context->getSizeOf(mv_block_type)));
    auto Nloc = Builder->CreateZExtOrBitCast(N, size->getType());
    size = Builder->CreateMul(size, Nloc);  /// now in bytes
    llvm::Value *size_as_int64 = Builder->CreateIntCast(
        size, llvm::Type::getInt64Ty(llvmContext), false);

    // same index into both sizes and ptrs arrays
    llvm::Value *gepIndices[] = {
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 0),
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), i)};

    llvm::Value *size_ptr =
        Builder->CreateGEP(int64_array_type, mv_sizes_bytes, gepIndices);
    Builder->CreateStore(size_as_int64, size_ptr);

    auto *src_ptr =
        Builder->CreateGEP(charptr_array_type, mv_src_ptrs, gepIndices);

//    check if any src_ptr is a page_id
    auto *intPtr = Builder->CreatePtrToInt(
        mv_src_ptr, llvm::Type::getInt64Ty(context->getLLVMContext()));
    auto *and_res = Builder->CreateAnd(intPtr, page_id_mask);
    any_ptr_is_page_id = Builder->CreateOr(
        any_ptr_is_page_id, Builder->CreateICmpEQ(and_res, page_id_mask));
    Builder->CreateStore(mv_src_ptr, src_ptr);
  }

  {
    if (!non_scan_move) {
      // if we have anyPage Ids we need to update tupleCnt to store a tupleCnt
      // instead of blockCnt, which the NvmePlugin stores in `tupleCnt`
      RecordAttribute actualTupleCnt{wantedFields[0]->getRelationName(),
                                     "tupleCnt", pg->getOIDType()};
      if (childState.contains(actualTupleCnt)) {
        auto ifAnyPageId = context->gen_if({any_ptr_is_page_id});

        auto thenAnyIsPageId = std::move(ifAnyPageId)([&]() {  // NOLINT
          ProteusValueMemory mem_tupleCntWrapper = childState[actualTupleCnt];
          llvm::Value *mem_tupleCnt = Builder->CreateLoad(
              mem_tupleCntWrapper.mem->getType()->getPointerElementType(),
              mem_tupleCntWrapper.mem);
          //      context->log(mem_tupleCnt);
          Builder->CreateStore(mem_tupleCnt, mem_cntWrapper.mem);
        });
      }
    }
    // note, all of this is a bit hacky, we ignore size in the above and below
    // loops for page ids, but we could actually just store the size/fd/offset
    // in the Operator state
  }

  // create an array of `pb` to pass to make_mem_move_device as a return
  // argument
  auto mv_num_bufs = context->createInt32(wantedFields.size());
  auto *llvm_pb_type = context->toLLVM<std::remove_cv_t<pb>>();
    llvm::ArrayType *llvm_pb_array_type =
        llvm::ArrayType::get(llvm_pb_type, wantedFields.size());

  llvm::AllocaInst *pb_array =
      Builder->CreateAlloca(llvm_pb_array_type, nullptr, "pb_array");

  auto workunit_ptr8 = context->gen_call(acquireWorkUnit, {memmv});
  auto workunit_ptr = Builder->CreateBitCast(
      workunit_ptr8, llvm::PointerType::getUnqual(workunit_type));
  // TODO pass selectivities to make_mem_move_device
    context->gen_call(make_mem_move_device,
                      {mv_src_ptrs, mv_sizes_bytes, device_id, srcS, memmv,
                       mv_num_bufs, pb_array, workunit_ptr});

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute block_attr(*(wantedFields[i]), true);
    ProteusValueMemory mem_valWrapper = childState[block_attr];

    llvm::Value *gep_index_moved[] = {context->createInt32(0),
                                      context->createInt32(i),
                                      context->createInt32(0)};
    llvm::Value *gep_index_to_release[] = {context->createInt32(0),
                                           context->createInt32(i),
                                           context->createInt32(1)};
    llvm::Value *moved_ptr = Builder->CreateGEP(
        pb_array->getType()->getNonOpaquePointerElementType(), pb_array,
        gep_index_moved, "gep_buffpair_structs_moved");
    llvm::Value *to_release_ptr = Builder->CreateGEP(
        pb_array->getType()->getNonOpaquePointerElementType(), pb_array,
        gep_index_to_release, "gep_buffpair_structs_to_release");
    llvm::Value *moved =
        Builder->CreateLoad(charPtrType, moved_ptr, "load_moved");
    llvm::Value *to_release =
        Builder->CreateLoad(charPtrType, to_release_ptr, "load_to_release");

    pushed.push_back(Builder->CreateBitCast(
        moved, mem_valWrapper.mem->getType()->getPointerElementType()));
    pushed.push_back(Builder->CreateBitCast(
        to_release, mem_valWrapper.mem->getType()->getPointerElementType()));
    //    TODO handle not actually a noop for nvme
    is_noop = context->createFalse();
    //        Builder->CreateAnd(is_noop, Builder->CreateICmpEQ(moved,
    //        to_release));
  }

  // This is now definitely number of tuples
  auto *num_tuples = Builder->CreateLoad(
      mem_cntWrapper.mem->getType()->getPointerElementType(),
      mem_cntWrapper.mem);
  pushed.push_back(num_tuples);
  pushed.push_back(oid);

  llvm::Value *d = llvm::UndefValue::get(data_type);
  for (size_t i = 0; i < pushed.size(); ++i) {
    d = Builder->CreateInsertValue(d, pushed[i], i);
  }

  auto workunit_dat = Builder->CreateLoad(
      workunit_ptr->getType()->getPointerElementType(), workunit_ptr);
  auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
  d_ptr =
      Builder->CreateBitCast(d_ptr, llvm::PointerType::getUnqual(data_type));
  Builder->CreateStore(d, d_ptr);

  // finally propagate the workunit
  context->gen_call(propagateWorkUnit, {memmv, workunit_ptr8, is_noop});

#endif
}

MemMoveDevice::MemMoveConf *MemMoveDevice::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  auto mmc = new (pmmc) MemMoveConf;
  mmc->do_transfer = do_transfer;
  mmc->enable_deferred_rfor_decode = !to_cpu;
  return mmc;
}

void MemMoveDevice::destroyMoveConf(MemMoveDevice::MemMoveConf *mmc) const {
  mmc->~MemMoveConf();
  MemoryManager::freePinned(mmc);
}

void MemMoveDevice::open(Pipeline *pip) {
  auto *wu = (workunit *)MemoryManager::mallocPinned(sizeof(workunit) * slack);
  event_range<range_log_op::MEMMOVE_OPEN> er{id, catch_pip->getUUID(),
                                             pip->getGroup()};

  // nvtxRangePushA("memmove::open");
  cudaStream_t strm = nullptr;
  if (!to_cpu) {
    set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
    strm = createNonBlockingStream();
  } else {
    strm = createNonBlockingStream();
  }

  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

#ifdef BATCHING
  data_size *= batch_size;
#endif

  //  data_size = (batch_size * pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

  //  if (!to_cpu) {
  //    CUfileError_t status = cuFileStreamRegister(
  //        strm,
  //        CU_FILE_STREAM_PAGE_ALIGNED_INPUTS | CU_FILE_STREAM_FIXED_BUF_OFFSET
  //        |
  //            CU_FILE_STREAM_FIXED_FILE_OFFSET |
  //            CU_FILE_STREAM_FIXED_FILE_SIZE);
  //    CHECK_EQ(status.err, CU_FILE_SUCCESS)
  //        << "Failed to cuFileStreamRegister status: "
  //        << cufileop_status_error(status.err);
  //  }

  MemMoveConf *mmc = createMoveConf();

#ifndef NCUDA
  mmc->strm = strm;
#endif
  mmc->slack = slack;
  mmc->data_buffs = MemoryManager::mallocPinned(data_size * slack);
  auto *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  if (dynamic_cast<NvmePlugin *>(pg)) {
    mmc->nvme_plugin = dynamic_cast<NvmePlugin *>(pg);
    // if this a mem-move NVMe->CPU or nvme->GPU with staging in CPU, we need an
    // io_uring
    if (to_cpu | std::all_of(mmc->do_transfer.begin(), mmc->do_transfer.end(),
                             [](bool v) { return !v; })) {
      mmc->io_uring =
          std::make_unique<proteus::storage::IoUringThreadUnsafe>(32);
    }
  }
  // GPU batch decompression state (scratch buffers + decompressor instances)
  // is only needed when MemMove performs decompression itself.
  const bool use_codec_metadata_in_block =
      FLAGS_use_metadata_in_block &&
      (FLAGS_compression_type == "RFOR" || FLAGS_compression_type == "FLBP");
  const bool use_late_decompression = FLAGS_decom_shared || FLAGS_decom_register;
  const bool defer_decompression_here =
      mmc->enable_deferred_rfor_decode && use_late_decompression &&
      use_codec_metadata_in_block;
  const bool allocate_gpu_batch_decompression_state =
      !FLAGS_compression_type.empty() && FLAGS_compression_type != "NONE" &&
      !defer_decompression_here;
  char *data_buff = (char *)mmc->data_buffs;
  for (size_t i = 0; i < slack; ++i) {
    wu[i].data = ((void *)(data_buff + i * data_size));
    wu[i].complete = 0;
#ifdef BATCHING
    wu[i].bytes_read = static_cast<ssize_t *>(
        MemoryManager::mallocPinned(sizeof(ssize_t) * wantedFields.size() * batch_size));
#else
    wu[i].bytes_read = static_cast<ssize_t *>(
        MemoryManager::mallocPinned(sizeof(ssize_t) * wantedFields.size()));
#endif
    wu[i].index_in_wu = 0;
    if (!to_cpu) {
      set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
      wu[i].cufile_strm = createNonBlockingStream();
    } else {
      wu[i].cufile_strm = createNonBlockingStream();
    }
    mmc->idle.push(wu + i);

    if (!to_cpu) {
      wu[i].decompressors = new std::vector<GpuDecompressor>{};
      // Per-workunit compressed scratch, indexed as [gpu_idx][field_idx].
      // Each entry is a device allocation used as cuFile destination for
      // compressed bytes in batch_push_nvme_to_gpu().
      wu[i].compressed_buffers = new std::vector<std::vector<void*>>{};
      wu[i].decompressors->reserve(topology::getInstance().getGpuCount());
      wu[i].compressed_buffers->reserve(topology::getInstance().getGpuCount());
      for (int j = 0; j < topology::getInstance().getGpuCount(); j++) {
        set_device_on_scope d(topology::getInstance().getGpus()[j]);

#ifdef BATCHING
        wu[i].decompressors->emplace_back(16_K, wantedFields.size() * batch_size, 2_M / 16_K,
                                          j);
#else
        if (allocate_gpu_batch_decompression_state) {
          if (FLAGS_compression_type == "LZ4") {
            wu[i].decompressors->emplace_back(64_K, wantedFields.size(), 2_M / 64_K,
                                              CompressionAlgorithm::LZ4,
                                              j);
          } else if (FLAGS_compression_type == "CASCADED") {
            wu[i].decompressors->emplace_back(64_K, wantedFields.size(), 2_M / 64_K,
                                              CompressionAlgorithm::CASCADED,
                                              j);
          } else if (FLAGS_compression_type == "GDEFLATE") {
            wu[i].decompressors->emplace_back(64_K, wantedFields.size(), 2_M / 64_K,
                                              CompressionAlgorithm::GDEFLATE,
                                              j);
          } else if (FLAGS_compression_type == "RFOR") {
            LOG_ASSERT(FLAGS_tile_size == RFOR_TILE_SIZE_512 ||
                       FLAGS_tile_size == RFOR_TILE_SIZE_1024)
                << "Unsupported RFOR --tile_size=" << FLAGS_tile_size
                << " for MemMove decompression path";
            const size_t rfor_tile_size_bytes =
                static_cast<size_t>(FLAGS_tile_size) * sizeof(int);
            const size_t rfor_max_tiles_per_2m = 2_M / rfor_tile_size_bytes;

            wu[i].decompressors->emplace_back(
                rfor_tile_size_bytes,          // max_decomp_chunk_size
                wantedFields.size(),            // max_batch_block_count
                rfor_max_tiles_per_2m,          // max_chunks_per_block
                CompressionAlgorithm::RFOR,     // comp_algo
                j);                             // gpu_index_in_topo
          } else if (FLAGS_compression_type == "FLBP") {
            const size_t FLBP_VECTOR_BYTES = FLBP_VECTOR_SIZE * sizeof(int);
            const size_t FLBP_MAX_VECTORS_PER_2M = 2_M / FLBP_VECTOR_BYTES;

            wu[i].decompressors->emplace_back(
                FLBP_VECTOR_BYTES,             // max_decomp_chunk_size
                wantedFields.size(),           // max_batch_block_count
                FLBP_MAX_VECTORS_PER_2M,       // max_chunks_per_block
                CompressionAlgorithm::FLBP,    // comp_algo
                j);                            // gpu_index_in_topo
          } else {
            LOG_FIRST_N(WARNING, 5)
                << "Invalid compression_type value for GPU batch decompression: "
                << FLAGS_compression_type;
          }
        }

        wu[i].compressed_buffers->emplace_back();
        if (allocate_gpu_batch_decompression_state) {
          // Reserve one compressed staging buffer per moved field.
          // Size is fixed per slot and reused across batches for this workunit.
          for (int k = 0; k < wantedFields.size(); k++) {
            wu[i].compressed_buffers->back().push_back(nullptr);
            gpu_run(cudaMalloc(&(wu[i].compressed_buffers->back().back()), 4_M));
          }
        }
#endif
      }
      //      CUfileError_t status = cuFileStreamRegister(
      //          wu[i].cufile_strm, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS |
      //                                 CU_FILE_STREAM_FIXED_BUF_OFFSET |
      //                                 CU_FILE_STREAM_FIXED_FILE_OFFSET |
      //                                 CU_FILE_STREAM_FIXED_FILE_SIZE);
      //
      //      CHECK_EQ(status.err, CU_FILE_SUCCESS)
      //          << "Failed to cuFileStreamRegister status: "
      //          << cufileop_status_error(status.err);
    }
  }
  // nvtxRangePushA("memmove::open2");
  if (!to_cpu) {
    set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
    for (size_t i = 0; i < slack; ++i) {
      gpu_run(cudaEventCreateWithFlags(
          &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
    }
  } else {
    for (size_t i = 0; i < slack; ++i) {
      gpu_run(cudaEventCreateWithFlags(
          &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
    }
  }
  // nvtxRangePop();

  mmc->worker = ThreadPool::getInstance().enqueue(
      &MemMoveDevice::catcher, this, mmc, pip->getGroup(), exec_location{},
      pip->getSession());

  pip->setStateVar<int>(device_id_var, getTargetDevice());

  pip->setStateVar<void *>(memmvconf_var, mmc);
  // nvtxRangePop();
}

int MemMoveDevice::getTargetDevice() const {
  if (to_cpu) return -1;
  return topology::getInstance().getActiveGpu().id;
}

void MemMoveDevice::close(Pipeline *pip) {
  auto *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

  {
    event_range<range_log_op::MEMMOVE_CLOSE> er{id, catch_pip->getUUID(),
                                                pip->getGroup()};

  if (mmc->io_uring) {
     mmc->io_uring->flush();
  }
  mmc->tran.close();

    nvtxRangePop();
    mmc->worker.get();
  }

  event_range<range_log_op::MEMMOVE_CLOSE_CLEAN_UP> er{id, catch_pip->getUUID(),
                                                       pip->getGroup()};
  if (!to_cpu) {
    set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
    syncAndDestroyStream(mmc->strm);
  } else {
    syncAndDestroyStream(mmc->strm);
  }

  nvtxRangePushA("MemMoveDev_running2");
  nvtxRangePushA("MemMoveDev_running");

  nvtxRangePushA("MemMoveDev_release");
  workunit *start_wu = nullptr;
  for (size_t i = 0; i < slack; ++i) {
    workunit *wu = mmc->idle.pop_unsafe();
    if (!to_cpu) {
      set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
      syncAndDestroyStream(wu->cufile_strm);
    } else {
      syncAndDestroyStream(wu->cufile_strm);
    }
    if (!to_cpu) {
      //      CUfileError_t status = cuFileStreamDeregister(wu->cufile_strm);
      //      CHECK_EQ(status.err, CU_FILE_SUCCESS)
      //          << "failed to cuFileStreamDeregister status: "
      //          << cufileop_status_error(status.err);
      // Free [gpu_idx][field_idx] compressed staging allocations.
      for (size_t gpu_idx = 0; gpu_idx < wu->compressed_buffers->size();
           ++gpu_idx) {
        set_device_on_scope d(topology::getInstance().getGpus()[gpu_idx]);
        for (void *buff : wu->compressed_buffers->at(gpu_idx)) {
          gpu_run(cudaFree(buff));
        }
      }
      delete wu->decompressors;
      delete wu->compressed_buffers;
    }
    MemoryManager::freePinned(wu->bytes_read);
    if (!to_cpu) {
      set_device_on_scope d(topology::getInstance().getGpus()[getTargetDevice()]);
      gpu_run(cudaEventDestroy(wu->event));
    } else {
      gpu_run(cudaEventDestroy(wu->event));
    }
    if (i == 0 || wu < start_wu) start_wu = wu;
  }
  nvtxRangePop();
  nvtxRangePop();
  MemoryManager::freePinned(mmc->data_buffs);
  MemoryManager::freePinned(start_wu);

  mmc->idle.close();

  destroyMoveConf(mmc);
}

void MemMoveDevice::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                           bool is_noop) {
  if (!is_noop) {
    gpu_run(cudaEventRecord(buff->event, strm));
  }
  tran.push(buff);

  if (io_uring != nullptr) {
    // if idle is empty, then slack is fully utilized
    // This means we the catcher is either waiting on IO to be completed or
    // processing the workunit.
    while (idle.empty_unsafe()) {
      // poll and handle io_uring completions
      // Polling enters the kernel, reaps completions and executes the callbacks
      io_uring->poll();
//      DLOG_EVERY_N(INFO, 500000) << "polling.... ";
    }
  }
}

MemMoveDevice::workunit *MemMoveDevice::MemMoveConf::acquire() {
  // time_block t{"pop: "};
  // LOG(INFO) << "pop";
  MemMoveDevice::workunit *ret = nullptr;
#ifndef NDEBUG
  bool popres =
#endif
      idle.pop(ret);
  assert(popres);
  return ret;
}

bool MemMoveDevice::MemMoveConf::getPropagated(MemMoveDevice::workunit **ret) {
  if (!tran.pop(*ret)) {
    return false;
  }

  if (nvme_plugin != nullptr) {
    const static auto prt = profiling::ProfileRegionType("memmove:::get_prop");
    auto prof_reg = profiling::ProfileRegion(prt);
    event_range<range_log_op::MEMMOVE_GET_PROPOGATED> er{{}};
    // wait for GPU decompression if any
    gpu_run(cudaStreamSynchronize((*ret)->cufile_strm));
    //    cudaStreamSynchronize(strm);
    // wait for io_uring operations if any
    while ((*ret)->complete != 0) {
      std::this_thread::yield();
    }
    return true;
  }
  gpu_run(cudaEventSynchronize((*ret)->event));

  return true;
}

void MemMoveDevice::MemMoveConf::release(MemMoveDevice::workunit *buff) {
  // LOG(INFO) << "pushed";
  // time_block t{"pushed: "};
  idle.push(buff);
}

void MemMoveDevice::catcher(MemMoveConf *mmc, int group_id,
                            exec_location target_dev, const void *session) {
  set_exec_location_on_scope d(target_dev);
  std::this_thread::yield();

  nvtxRangePushA("memmove::catch");

  auto pip = catch_pip->getPipeline(group_id);

  nvtxRangePushA("memmove::catch_open");
  pip->open(session);
  nvtxRangePop();

  {
    do {
      MemMoveDevice::workunit *p = nullptr;
      if (!mmc->getPropagated(&p)) break;

#ifdef BATCHING
      const size_t rg_data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);
      void *data = p->data;
      for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < wantedFields.size(); ++i) {
          ((proteus::managed_ptr *)(data))[i * 2] =
              mmc->pull(std::move(((proteus::managed_ptr *)(data))[i * 2]));
        }
        {
          event_range<range_log_op::MEMMOVE_CONSUME> er{
              id, catch_pip->getUUID(), pip->getGroup()};
          nvtxRangePushA("memmove::catch_cons");
          pip->consume(data);
          nvtxRangePop();
        }

        data = (char *)data + rg_data_size;
      }

      mmc->release(p);
#else

      for (size_t i = 0; i < wantedFields.size(); ++i) {
        ((proteus::managed_ptr *)(p->data))[i * 2] =
            mmc->pull(std::move(((proteus::managed_ptr *)(p->data))[i * 2]));
      }
      {
        event_range<range_log_op::MEMMOVE_CONSUME> er{id, catch_pip->getUUID(),
                                                      pip->getGroup()};
        nvtxRangePushA("memmove::catch_cons");
        pip->consume(p->data);
        nvtxRangePop();
      }

      mmc->release(p);

#endif
    } while (true);
  }

  event_range<range_log_op::MEMMOVE_CLOSE> er{id, catch_pip->getUUID(),
                                              pip->getGroup()};
  nvtxRangePushA("memmove::catch_close");
  pip->close();
  nvtxRangePop();

  nvtxRangePop();
}
