/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that the
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
#include "gpu-decompress-blocks.hpp"

#include <algorithm>
#include <map>

#include <gflags/gflags.h>

#include <codegen/jit/pipeline.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/topology/topology.hpp>

#include "lib/util/catalog.hpp"
#include "lib/operators/mem-move/compression.hpp"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace llvm;

std::mutex GpuDecompressBlocks::live_instances_mtx_;
std::unordered_set<GpuDecompressBlocks *> GpuDecompressBlocks::live_instances_;

DECLARE_string(compression_type);
DECLARE_bool(precomputed_offsets);
namespace {
constexpr uint32_t kRforBlockThreads = 256;
constexpr uint32_t kRforItemsPerThread = 4;
constexpr uint32_t kRforTileElems = kRforBlockThreads * kRforItemsPerThread;
constexpr uint32_t kRforTileBytes = kRforTileElems * sizeof(int);
constexpr uint32_t kRforSharedScratchElems = 2 * kRforTileElems;
constexpr uint32_t kRforSharedOutElems = kRforTileElems;
constexpr uint32_t kRforBroadcastElems = 24;
constexpr uint32_t kRforSharedOutBase = kRforSharedScratchElems;
constexpr uint32_t kRforSharedBroadcastBase =
    kRforSharedOutBase + kRforSharedOutElems;
constexpr uint32_t kRforSharedTotalElems =
    kRforSharedBroadcastBase + kRforBroadcastElems;
constexpr uint32_t kRforMaxTilesPerBlock =
    BlockManager::block_size / kRforTileBytes;
constexpr uint32_t kRforMaxRowsPerBlock =
    BlockManager::block_size / sizeof(int);
constexpr uint64_t kRforMaxReasonableCompressedBytes = 64ULL * 1024ULL * 1024ULL;

}  // namespace

static std::string getRelNameFromWantedFields(
    const std::vector<expression_t> &wantedFields) {
  CHECK(!wantedFields.empty())
      << "GpuDecompressBlocks requires non-empty wantedFields";
  return wantedFields[0].getRegisteredRelName();
}

static std::string getRelNameFromState(
    const OperatorState &childState) {
  for (const auto &t : childState.getBindings()) {
    if (t.first.getAttrName() == "activeCnt") {
      return t.first.getRelationName();
    }
  }
  LOG(FATAL) << "GpuDecompressBlocks cannot determine relation name";
}

static CompressionAlgorithm getCompressionAlgorithmFromFlag() {
  if (FLAGS_compression_type == "RFOR") return CompressionAlgorithm::RFOR;
  if (FLAGS_compression_type == "LZ4") return CompressionAlgorithm::LZ4;
  if (FLAGS_compression_type == "CASCADED")
    return CompressionAlgorithm::CASCADED;
  if (FLAGS_compression_type == "GDEFLATE")
    return CompressionAlgorithm::GDEFLATE;
  if (FLAGS_compression_type == "FLBP") return CompressionAlgorithm::FLBP;
  LOG(FATAL) << "GpuDecompressBlocks does not support compression type: "
             << FLAGS_compression_type;
  __builtin_unreachable();
}

static void *allocateGpuRaw(size_t bytes) {
  if (bytes == 0) return nullptr;
  void *ptr = nullptr;
  set_device_on_scope d(topology::getInstance().getActiveGpu());
  gpu_run(cudaMalloc(&ptr, bytes));
  return ptr;
}

static void freeGpuRaw(void *ptr) {
  if (!ptr) return;
  set_device_on_scope d(topology::getInstance().getActiveGpu());
  gpu_run(cudaFree(ptr));
}

void GpuDecompressBlocks::releaseRuntimeBuffers() {
  for (size_t i = 0; i < decompressed_part_bases_.size(); ++i) {
    for (size_t part = 0; part < decompressed_part_bases_[i].size(); ++part) {
      auto *ptr = decompressed_part_bases_[i][part];
      if (!ptr) continue;
      freeGpuRaw(ptr);
      decompressed_part_bases_[i][part] = nullptr;
      if (i < decompressed_part_bytes_.size() &&
          part < decompressed_part_bytes_[i].size()) {
        decompressed_part_bytes_[i][part] = 0;
      }
    }
  }
  decompressed_part_bases_.clear();
  decompressed_part_bytes_.clear();
  part_block_counts_.clear();
  runtime_buffers_allocated_ = false;
  runtime_buffers_decompressed_ = false;
  nvme_pg_ = nullptr;
  rel_name_.clear();
}

GpuDecompressBlocks::~GpuDecompressBlocks() {
  releaseRuntimeBuffers();
  std::lock_guard<std::mutex> lock(live_instances_mtx_);
  live_instances_.erase(this);
}

void GpuDecompressBlocks::releaseAllRuntimeBuffers() {
  std::lock_guard<std::mutex> lock(live_instances_mtx_);
  for (auto *op : live_instances_) {
    if (!op) continue;
    op->releaseRuntimeBuffers();
  }
}

void GpuDecompressBlocks::ensureRuntimeBuffersAllocated(NvmePlugin *nvme_pg) {
  if (runtime_buffers_allocated_) {
    CHECK_EQ(nvme_pg_, nvme_pg);
    return;
  }

  nvme_pg_ = nvme_pg;
  rel_name_ = getRelNameFromWantedFields(wantedFields);

  const size_t part_count = nvme_pg_->getPartitionCount();
  part_block_counts_.resize(part_count, 0);
  for (size_t part_no = 0; part_no < part_count; ++part_no) {
    part_block_counts_[part_no] = nvme_pg_->getPartitionBlockCount(part_no);
  }

  decompressed_part_bases_.resize(wantedFields.size());
  decompressed_part_bytes_.resize(wantedFields.size());
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    const auto *block_ty =
        dynamic_cast<const BlockType *>(wantedFields[i].getExpressionType());
    if (!block_ty) continue;

    decompressed_part_bases_[i].resize(part_count, nullptr);
    decompressed_part_bytes_[i].resize(part_count, 0);
    for (size_t part_no = 0; part_no < part_count; ++part_no) {
      const size_t bytes = part_block_counts_[part_no] * BlockManager::block_size;
      if (bytes == 0) continue;
      decompressed_part_bases_[i][part_no] = allocateGpuRaw(bytes);
      decompressed_part_bytes_[i][part_no] = bytes;
    }
  }

  runtime_buffers_allocated_ = true;
  runtime_buffers_decompressed_ = false;
}

void GpuDecompressBlocks::decompressRuntimeBuffers() {
  CHECK(runtime_buffers_allocated_);
  CHECK(nvme_pg_);

  const auto comp_algo = getCompressionAlgorithmFromFlag();
  CHECK(comp_algo != CompressionAlgorithm::RFOR)
      << "GpuDecompressBlocks RFOR must run fused in consume() via "
         "LoadRBinPack_notemplate; open()-time RFOR materialization is disabled";

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    const auto *block_ty =
        dynamic_cast<const BlockType *>(wantedFields[i].getExpressionType());
    if (!block_ty) continue;

    constexpr size_t kMaxBlocksPerBatch = 256;

    size_t max_chunks_per_block = 1;
    int max_decomp_chunk_size = 0;

    if (comp_algo == CompressionAlgorithm::RFOR) {
      // RFOR chunks decompress to one fixed tile of ints.
      constexpr size_t kRforTileBytes = RFOR_TILE_SIZE * sizeof(int);
      max_chunks_per_block = BlockManager::block_size / kRforTileBytes;
      max_decomp_chunk_size = static_cast<int>(kRforTileBytes);
    } else {
      for (size_t part_no = 0; part_no < part_block_counts_.size(); ++part_no) {
        const uint8_t attr_u8 = static_cast<uint8_t>(i);
        const uint8_t part_u8 = static_cast<uint8_t>(part_no);
        for (size_t block_no = 0; block_no < part_block_counts_[part_no];
             ++block_no) {
          const uint32_t block_u32 = static_cast<uint32_t>(block_no);
          NvmePlugin::PageId_t page_id{/*cpu_numa_affinity=*/0, attr_u8, part_u8,
                                       block_u32};
          auto page_info = nvme_pg_->getPageIoInfo(page_id);
          CHECK(page_info.is_compressed)
              << "GpuDecompressBlocks expects compressed input blocks";
          max_chunks_per_block =
              std::max(max_chunks_per_block, page_info.chunk_sizes.size());
          max_decomp_chunk_size =
              std::max(max_decomp_chunk_size, page_info.decompressed_chunk_size);
        }
      }
    }

    if (max_decomp_chunk_size <= 0) {
      max_decomp_chunk_size = static_cast<int>(BlockManager::block_size);
    }

    auto decompressor = GpuDecompressor(
        static_cast<size_t>(max_decomp_chunk_size), kMaxBlocksPerBatch,
        max_chunks_per_block, comp_algo,
        topology::getInstance().getActiveGpu().index_in_topo);

    for (size_t part_no = 0; part_no < part_block_counts_.size(); ++part_no) {
      const size_t block_count = part_block_counts_[part_no];
      if (block_count == 0) continue;

      auto *decomp_part_base =
          static_cast<char *>(decompressed_part_bases_[i][part_no]);
      CHECK(decomp_part_base) << "Missing decompressed destination buffer";

      const uint8_t attr_u8 = static_cast<uint8_t>(i);
      const uint8_t part_u8 = static_cast<uint8_t>(part_no);
      for (size_t block_begin = 0; block_begin < block_count;
           block_begin += kMaxBlocksPerBatch) {
        const size_t batch_blocks =
            std::min(kMaxBlocksPerBatch, block_count - block_begin);

        std::vector<std::span<const uint32_t>> block_chunk_sizes;
        std::vector<std::span<char>> block_compressed_buffers;
        std::vector<std::span<char>> block_decompressed_buffers;
        std::vector<std::span<const uint32_t>> block_val_section_sizes;
        block_chunk_sizes.reserve(batch_blocks);
        block_compressed_buffers.reserve(batch_blocks);
        block_decompressed_buffers.reserve(batch_blocks);
        if (comp_algo == CompressionAlgorithm::RFOR) {
          block_val_section_sizes.reserve(batch_blocks);
        }

        for (size_t off = 0; off < batch_blocks; ++off) {
          const size_t block_no = block_begin + off;
          const uint32_t block_u32 = static_cast<uint32_t>(block_no);
          NvmePlugin::PageId_t page_id{/*cpu_numa_affinity=*/0, attr_u8,
                                       part_u8, block_u32};
          CHECK(nvme_pg_->isCompressed(page_id))
              << "GpuDecompressBlocks expects compressed input blocks";

          auto *compressed_ptr = static_cast<char *>(
              getNvmePreloadedCompressedBlockPtr(nvme_pg_, attr_u8, part_u8,
                                                 block_u32));
          auto *decompressed_ptr =
              decomp_part_base +
              block_no * static_cast<size_t>(BlockManager::block_size);

          const auto &chunk_sizes = nvme_pg_->getChunkSizesRef(page_id);
          block_chunk_sizes.emplace_back(chunk_sizes.data(), chunk_sizes.size());
          block_compressed_buffers.emplace_back(
              compressed_ptr, nvme_pg_->getBlockSizeBytes(page_id));
          block_decompressed_buffers.emplace_back(
              decompressed_ptr, static_cast<size_t>(BlockManager::block_size));
          if (comp_algo == CompressionAlgorithm::RFOR) {
            const auto &val_section_sizes = nvme_pg_->getValSectionSizesRef(page_id);
            block_val_section_sizes.emplace_back(val_section_sizes.data(),
                                                 val_section_sizes.size());
          }
        }

        int decomp_res = -1;
        if (comp_algo == CompressionAlgorithm::RFOR) {
          decomp_res = decompressor.batch_decompress_block_gpu(
              block_chunk_sizes, block_compressed_buffers,
              block_decompressed_buffers, /*stream=*/nullptr,
              block_val_section_sizes);
        } else {
          decomp_res = decompressor.batch_decompress_block_gpu(
              block_chunk_sizes, block_compressed_buffers,
              block_decompressed_buffers, /*stream=*/nullptr, {});
        }
        CHECK_GE(decomp_res, 0)
            << "GpuDecompressBlocks failed for relation " << rel_name_
            << " attr=" << i << " part=" << part_no
            << " block_begin=" << block_begin;
      }
    }
  }
}

void GpuDecompressBlocks::produce_(OlapParallelContext *context) {
  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });
  this->getChild()->produce(context);
}

void GpuDecompressBlocks::open(Pipeline *pip) {
  (void)pip;
  CHECK(runtime_buffers_allocated_)
      << "GpuDecompressBlocks runtime buffers must be allocated during prepare";
  const auto comp_algo = getCompressionAlgorithmFromFlag();
  // RFOR is fused with query execution in consume() via
  // LoadRBinPack_notemplate. Do not invoke the standalone RFOR batch
  // decompressor in open(), otherwise profiling still sees host pointer-table
  // setup and runRFORKernel launches.
  if (comp_algo == CompressionAlgorithm::RFOR) {
    runtime_buffers_decompressed_ = false;
    return;
  }
  if (!runtime_buffers_decompressed_) {
    decompressRuntimeBuffers();
    runtime_buffers_decompressed_ = true;
  }
}

void GpuDecompressBlocks::close(Pipeline *pip) {
  (void)pip;
}

void GpuDecompressBlocks::consume(OlapParallelContext *context,
                                  const OperatorState &childState) {
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  auto *F = Builder->GetInsertBlock()->getParent();
  auto *module = context->getModule();

  const auto relName = wantedFields.empty()
                           ? getRelNameFromState(childState)
                           : getRelNameFromWantedFields(wantedFields);
  const size_t block_field_count =
      std::count_if(wantedFields.begin(), wantedFields.end(),
                    [](const expression_t &field) {
                      return dynamic_cast<const BlockType *>(
                                 field.getExpressionType()) != nullptr;
                    });

  auto *pg = Catalog::getInstance().getPlugin(relName);
  auto *nvme_pg = dynamic_cast<NvmePlugin *>(pg);
  CHECK(nvme_pg) << "GpuDecompressBlocks requires NvmePlugin";
  ensureRuntimeBuffersAllocated(nvme_pg);

  RecordAttribute tupleIdentifier{relName, activeLoop, pg->getOIDType()};
  Value *oid = Builder->CreateLoad(
      childState[tupleIdentifier].mem->getType()->getNonOpaquePointerElementType(),
      childState[tupleIdentifier].mem, "block_oid");

  auto *oid_type = oid->getType();
  auto *nparts = ConstantInt::get(oid_type,
                                  static_cast<uint64_t>(nvme_pg->getPartitionCount()));
  Value *part_idx = Builder->CreateURem(oid, nparts, "part_idx");
  Value *block_idx = Builder->CreateUDiv(oid, nparts, "block_idx");
  const auto comp_algo = getCompressionAlgorithmFromFlag();
  const bool inline_rfor_decode = (comp_algo == CompressionAlgorithm::RFOR);
  if (inline_rfor_decode) {
    auto *pip = context->getCurrentPipeline();
    pip->setMaxWorkerSize(kRforBlockThreads, 1);
  }

  map<RecordAttribute, ProteusValueMemory> variableBindings =
      childState.getBindings();
  {
    RecordAttribute activeCnt{relName, "activeCnt", pg->getOIDType()};
    RecordAttribute tupleCnt{relName, "tupleCnt", pg->getOIDType()};
    if (variableBindings.find(tupleCnt) == variableBindings.end() &&
        variableBindings.find(activeCnt) != variableBindings.end()) {
      variableBindings[tupleCnt] = variableBindings[activeCnt];
    }
  }

  auto *i8_ptr_ty = Builder->getInt8PtrTy();
  auto *i8_ty = Builder->getInt8Ty();
  auto *i32_ty = Builder->getInt32Ty();
  auto *i64_ty = Builder->getInt64Ty();
  auto *i32_ptr_ty = PointerType::get(i32_ty, 0);
  auto *i32_ptr_ptr_ty = PointerType::get(i32_ptr_ty, 0);
  auto *void_ty = Type::getVoidTy(llvmContext);
  auto *zero_i32 = ConstantInt::get(i32_ty, 0);
  auto *zero_i64 = ConstantInt::get(i64_ty, 0);
  auto *one_i64 = ConstantInt::get(i64_ty, 1);
  auto *tile_elems_i64 = ConstantInt::get(i64_ty, kRforTileElems);
  auto *tile_bytes_i64 = ConstantInt::get(i64_ty, kRforTileBytes);
  auto *fallback_header_arr_ty = ArrayType::get(i32_ty, 8);
  auto *fallback_header_gv = module->getGlobalVariable("gdb_rfor_header_fallback");
  if (!fallback_header_gv) {
    fallback_header_gv = new GlobalVariable(
        *module, fallback_header_arr_ty, /*isConstant=*/true,
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(fallback_header_arr_ty),
        "gdb_rfor_header_fallback");
  }
  auto *fallback_header_ptr = Builder->CreateInBoundsGEP(
      fallback_header_arr_ty, fallback_header_gv, {zero_i32, zero_i32},
      "gdb_rfor_fallback_hdr_ptr");
  Value *fallback_header_i8 = Builder->CreateBitCast(
      fallback_header_ptr, i8_ptr_ty, "gdb_rfor_fallback_hdr_i8");

  Value *tuple_cnt_i64 = ConstantInt::get(i64_ty, 0);
  Value *tile_count_i64 = ConstantInt::get(i64_ty, 0);
  Value *tuple_cnt_i64_uniform = tuple_cnt_i64;

  if (inline_rfor_decode) {
    // In preload-to-GPU scan paths activeCnt carries rowgroup tuple count.
    RecordAttribute activeCnt{relName, "activeCnt", pg->getOIDType()};
    Value *tuple_cnt_raw = Builder->CreateLoad(
        childState[activeCnt].mem->getType()->getNonOpaquePointerElementType(),
        childState[activeCnt].mem, "tuple_cnt");
    tuple_cnt_i64 = Builder->CreateIntCast(tuple_cnt_raw, i64_ty, false);
    tile_count_i64 = Builder->CreateUDiv(
        Builder->CreateAdd(
            tuple_cnt_i64, ConstantInt::get(i64_ty, kRforTileElems - 1)),
        tile_elems_i64);
    tuple_cnt_i64_uniform = tuple_cnt_i64;
  }

  auto *part_idx_u8 = Builder->CreateTrunc(part_idx, Builder->getInt8Ty());
  auto *block_idx_u32 = Builder->CreateTrunc(block_idx, Builder->getInt32Ty());
  Value *part_idx_u8_uniform = part_idx_u8;
  Value *block_idx_u32_uniform = block_idx_u32;
  Value *tile_count_i64_uniform = tile_count_i64;
  Value *block_oid_i64 = Builder->CreateIntCast(oid, i64_ty, false);
  Value *block_oid_i64_uniform = block_oid_i64;

  FunctionCallee load_rfor_func;

  Value *shared_buffer_generic_ptr = nullptr;
  Value *shared_tile_out_ptr = nullptr;
  PointerType *rle_tile_metadata_ptr_ty = nullptr;
  ConstantInt *metadata_bytes_i64 = nullptr;
  Function *barrier = nullptr;
  Function *membar_cta = nullptr;
  Value *is_block_leader = nullptr;
  Value *broadcast_part_slot = nullptr;
  Value *broadcast_block_slot = nullptr;
  Value *broadcast_cnt_lo_slot = nullptr;
  Value *broadcast_cnt_hi_slot = nullptr;
  Value *broadcast_ptr_lo_slot = nullptr;
  Value *broadcast_ptr_hi_slot = nullptr;
  Value *broadcast_decode_flag_slot = nullptr;
  Value *broadcast_chunk_off_lo_slot = nullptr;
  Value *broadcast_chunk_off_hi_slot = nullptr;
  Value *broadcast_val_sec_lo_slot = nullptr;
  Value *broadcast_val_sec_hi_slot = nullptr;
  Value *broadcast_oid_lo_slot = nullptr;
  Value *broadcast_oid_hi_slot = nullptr;
  Value *broadcast_uniform_decode_slot = nullptr;
  Value *broadcast_uniform_tilecnt_lo_slot = nullptr;
  Value *broadcast_uniform_tilecnt_hi_slot = nullptr;
  Value *broadcast_safe_cnt_lo_slot = nullptr;
  Value *broadcast_safe_cnt_hi_slot = nullptr;
  auto store_tile_broadcast_i32 = [&](Value *value, Value *slot) {
    auto *st = Builder->CreateStore(value, slot);
    st->setVolatile(true);
    return st;
  };
  auto load_tile_broadcast_i32 = [&](Value *slot) {
    auto *ld = Builder->CreateLoad(i32_ty, slot);
    ld->setVolatile(true);
    return ld;
  };
  if (inline_rfor_decode) {
    auto *RLETileMetadata_ty =
        StructType::get(llvmContext, {i32_ty, i32_ty, i32_ty, i32_ty, i32_ty,
                                      i32_ty});
    auto *RLETileMetadata_ptr_ty = PointerType::get(RLETileMetadata_ty, 0);
    rle_tile_metadata_ptr_ty = RLETileMetadata_ptr_ty;
    metadata_bytes_i64 =
        ConstantInt::get(i64_ty, context->getSizeOf(RLETileMetadata_ty));
    auto *LoadRBinPack_ty =
        FunctionType::get(void_ty,
                          {RLETileMetadata_ptr_ty, i32_ptr_ty, i32_ptr_ty,
                           i32_ptr_ty, i32_ptr_ty},
                          false);
    load_rfor_func =
        module->getOrInsertFunction("LoadRBinPack_notemplate", LoadRBinPack_ty);

    auto *i32_scratch_array_ty = ArrayType::get(i32_ty, kRforSharedTotalElems);
    auto *i32_scratch_array_ptr_ty = PointerType::get(i32_scratch_array_ty, 0);
    auto *scratch = new GlobalVariable(
        *module, i32_scratch_array_ty,
        /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(i32_scratch_array_ty), "gdb_rfor_shared_scratch",
        nullptr, GlobalValue::NotThreadLocal, 3);
    auto *scratch_arr_generic =
        Builder->CreateAddrSpaceCast(scratch, i32_scratch_array_ptr_ty);
    shared_buffer_generic_ptr = Builder->CreateInBoundsGEP(
        scratch->getValueType(), scratch_arr_generic,
        {ConstantInt::get(i32_ty, 0), ConstantInt::get(i32_ty, 0)},
        "gdb_rfor_shared");
    shared_tile_out_ptr = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_generic_ptr,
        ConstantInt::get(i32_ty, kRforSharedOutBase), "gdb_rfor_shared_out");

    barrier = Intrinsic::getDeclaration(module, Intrinsic::nvvm_barrier0);
    membar_cta = Intrinsic::getDeclaration(module, Intrinsic::nvvm_membar_cta);
    is_block_leader = Builder->CreateICmpEQ(context->threadIdInBlock(),
                                            ConstantInt::get(i64_ty, 0));

    auto broadcast_slot = [&](uint32_t idx) -> Value * {
      return Builder->CreateInBoundsGEP(
          i32_ty, shared_buffer_generic_ptr,
          ConstantInt::get(i32_ty, kRforSharedBroadcastBase + idx));
    };
    broadcast_part_slot = broadcast_slot(0);
    broadcast_block_slot = broadcast_slot(1);
    broadcast_cnt_lo_slot = broadcast_slot(2);
    broadcast_cnt_hi_slot = broadcast_slot(3);
    broadcast_ptr_lo_slot = broadcast_slot(4);
    broadcast_ptr_hi_slot = broadcast_slot(5);
    broadcast_decode_flag_slot = broadcast_slot(8);
    broadcast_chunk_off_lo_slot = broadcast_slot(9);
    broadcast_chunk_off_hi_slot = broadcast_slot(10);
    broadcast_val_sec_lo_slot = broadcast_slot(11);
    broadcast_val_sec_hi_slot = broadcast_slot(12);
    broadcast_oid_lo_slot = broadcast_slot(13);
    broadcast_oid_hi_slot = broadcast_slot(14);
    // Dedicated block-uniform loop-control slots; keep separate from per-tile
    // chunk-offset slots to avoid aliasing/clobber during codegen.
    broadcast_uniform_decode_slot = broadcast_slot(15);
    broadcast_uniform_tilecnt_lo_slot = broadcast_slot(16);
    broadcast_uniform_tilecnt_hi_slot = broadcast_slot(17);
    broadcast_safe_cnt_lo_slot = broadcast_slot(18);
    broadcast_safe_cnt_hi_slot = broadcast_slot(19);

    context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
        [&]() {
          Builder->CreateStore(Builder->CreateZExt(part_idx_u8, i32_ty),
                               broadcast_part_slot);
          Builder->CreateStore(block_idx_u32, broadcast_block_slot);

          Value *cnt_lo = Builder->CreateTrunc(tuple_cnt_i64, i32_ty);
          Value *cnt_hi = Builder->CreateTrunc(
              Builder->CreateLShr(tuple_cnt_i64, ConstantInt::get(i64_ty, 32)),
              i32_ty);
          Builder->CreateStore(cnt_lo, broadcast_cnt_lo_slot);
          Builder->CreateStore(cnt_hi, broadcast_cnt_hi_slot);

          Value *oid_lo = Builder->CreateTrunc(block_oid_i64, i32_ty);
          Value *oid_hi = Builder->CreateTrunc(
              Builder->CreateLShr(block_oid_i64, ConstantInt::get(i64_ty, 32)),
              i32_ty);
          Builder->CreateStore(oid_lo, broadcast_oid_lo_slot);
          Builder->CreateStore(oid_hi, broadcast_oid_hi_slot);
        });
    Builder->CreateCall(barrier);

    part_idx_u8_uniform = Builder->CreateTrunc(
        Builder->CreateLoad(i32_ty, broadcast_part_slot), Builder->getInt8Ty());
    block_idx_u32_uniform = Builder->CreateLoad(i32_ty, broadcast_block_slot);
    Value *tuple_cnt_lo = Builder->CreateZExt(
        Builder->CreateLoad(i32_ty, broadcast_cnt_lo_slot), i64_ty);
    Value *tuple_cnt_hi = Builder->CreateZExt(
        Builder->CreateLoad(i32_ty, broadcast_cnt_hi_slot), i64_ty);
    tuple_cnt_i64_uniform = Builder->CreateOr(
        tuple_cnt_lo, Builder->CreateShl(tuple_cnt_hi, ConstantInt::get(i64_ty, 32)));
    tile_count_i64_uniform = Builder->CreateUDiv(
        Builder->CreateAdd(tuple_cnt_i64_uniform,
                           ConstantInt::get(i64_ty, kRforTileElems - 1)),
        tile_elems_i64);
    Value *oid_lo = Builder->CreateZExt(
        Builder->CreateLoad(i32_ty, broadcast_oid_lo_slot), i64_ty);
    Value *oid_hi = Builder->CreateZExt(
        Builder->CreateLoad(i32_ty, broadcast_oid_hi_slot), i64_ty);
    block_oid_i64_uniform = Builder->CreateOr(
        oid_lo, Builder->CreateShl(oid_hi, ConstantInt::get(i64_ty, 32)));

  }

  Value *safe_tuple_cnt_i64 = tuple_cnt_i64_uniform;
  auto *max_tiles_per_block_i64 =
      ConstantInt::get(i64_ty, kRforMaxTilesPerBlock);
  auto *max_rows_per_block_i64 =
      ConstantInt::get(i64_ty, kRforMaxRowsPerBlock);
  auto *header_fixed_bytes = ConstantInt::get(i64_ty, 3 * sizeof(uint32_t));
  auto *u32_bytes = ConstantInt::get(i64_ty, sizeof(uint32_t));
  auto *max_reasonable_compressed_bytes_i64 =
      ConstantInt::get(i64_ty, kRforMaxReasonableCompressedBytes);

  Value *part_idx_for_output = inline_rfor_decode
                                   ? Builder->CreateIntCast(
                                         part_idx_u8_uniform, part_idx->getType(), false)
                                   : part_idx;
  Value *block_idx_for_output = inline_rfor_decode
                                    ? Builder->CreateIntCast(
                                          block_idx_u32_uniform, block_idx->getType(), false)
                                    : block_idx;
  Value *part_block_count_for_output =
      ConstantInt::get(block_idx_for_output->getType(), 0);
  for (size_t part_no = 0; part_no < part_block_counts_.size(); ++part_no) {
    auto *is_part = Builder->CreateICmpEQ(
        part_idx_for_output,
        ConstantInt::get(part_idx_for_output->getType(), part_no));
    auto *part_block_count_const = ConstantInt::get(
        block_idx_for_output->getType(), part_block_counts_[part_no]);
    part_block_count_for_output = Builder->CreateSelect(
        is_part, part_block_count_const, part_block_count_for_output);
  }
  Value *part_has_blocks_for_output = Builder->CreateICmpUGT(
      part_block_count_for_output,
      ConstantInt::get(block_idx_for_output->getType(), 0));
  Value *block_idx_in_range_for_output = Builder->CreateICmpULT(
      block_idx_for_output, part_block_count_for_output);
  Value *block_oid_valid_for_output = Builder->CreateAnd(
      part_has_blocks_for_output, block_idx_in_range_for_output);
  Value *safe_block_idx_for_output = Builder->CreateSelect(
      block_oid_valid_for_output, block_idx_for_output,
      ConstantInt::get(block_idx_for_output->getType(), 0));

  uint8_t block_attr_no = 0;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    const auto &field = wantedFields[i];
    const auto *block_ty =
        dynamic_cast<const BlockType *>(field.getExpressionType());
    if (!block_ty) {
      continue;
    }

    RecordAttribute attr{field.getRegisteredAs()};
    RecordAttribute block_attr =
        dynamic_cast<const BlockType *>(attr.getOriginalType())
            ? attr
            : RecordAttribute(attr, true);
    const bool use_precomputed_tile_offsets_requested =
        inline_rfor_decode && FLAGS_precomputed_offsets &&
        nvme_pg->scanEmitsCompressedBlocks() &&
        nvme_pg->getPreloadedCompressedChunkOffsetsOidMapPtr(
            block_attr_no) &&
        nvme_pg->getPreloadedCompressedValSectionSizesOidMapPtr(
            block_attr_no);
    const bool use_precomputed_tile_offsets =
        use_precomputed_tile_offsets_requested;
    Value *chunk_offsets_oid_map = nullptr;
    Value *val_section_sizes_oid_map = nullptr;
    Value *block_sizes_oid_map = ConstantPointerNull::get(i32_ptr_ty);
    if (use_precomputed_tile_offsets) {
      chunk_offsets_oid_map = Builder->CreateIntToPtr(
          context->createInt64(reinterpret_cast<uintptr_t>(
              nvme_pg->getPreloadedCompressedChunkOffsetsOidMapPtr(
                  block_attr_no))),
          i32_ptr_ptr_ty);
      val_section_sizes_oid_map = Builder->CreateIntToPtr(
          context->createInt64(reinterpret_cast<uintptr_t>(
              nvme_pg->getPreloadedCompressedValSectionSizesOidMapPtr(
                  block_attr_no))),
          i32_ptr_ptr_ty);
    }
    if (inline_rfor_decode) {
      block_sizes_oid_map = Builder->CreateIntToPtr(
          context->createInt64(reinterpret_cast<uintptr_t>(
              nvme_pg->getPreloadedCompressedBlockSizesOidMapPtr(block_attr_no))),
          i32_ptr_ty);
    }

    // Scan stores packed block pointers as T* (T = nested scalar type), even
    // when the logical attr type is Block(T). Keep the same ABI here so
    // BlockToTuples sees identical child-state types.
    Type *ptr_t = PointerType::get(block_ty->getNestedType().getLLVMType(
                                       llvmContext),
                                   0);

    Value *part_base_i8 =
        ConstantPointerNull::get(cast<PointerType>(i8_ptr_ty));
    CHECK_LT(i, decompressed_part_bases_.size());
    for (size_t part_no = 0; part_no < part_block_counts_.size(); ++part_no) {
      void *part_base_host = nullptr;
      CHECK_LT(part_no, decompressed_part_bases_[i].size());
      if (part_block_counts_[part_no] > 0) {
        part_base_host = decompressed_part_bases_[i][part_no];
      }

      auto *is_part = Builder->CreateICmpEQ(
          part_idx_for_output,
          ConstantInt::get(part_idx_for_output->getType(), part_no));
      auto *part_base_const = Builder->CreateIntToPtr(
          context->createInt64(reinterpret_cast<uintptr_t>(part_base_host)),
          i8_ptr_ty);
      part_base_i8 =
          Builder->CreateSelect(is_part, part_base_const, part_base_i8);
    }

    Value *part_base_nonnull = Builder->CreateICmpNE(
        part_base_i8, ConstantPointerNull::get(cast<PointerType>(i8_ptr_ty)));
    Value *block_oid_valid_for_attr = Builder->CreateAnd(
        block_oid_valid_for_output, part_base_nonnull);
    Value *decomp_base_i8 = Builder->CreateSelect(part_base_nonnull, part_base_i8,
                                                  fallback_header_i8);
    auto *decomp_block_bytes = ConstantInt::get(
        safe_block_idx_for_output->getType(), BlockManager::block_size);
    auto *decomp_byte_off =
        Builder->CreateMul(safe_block_idx_for_output, decomp_block_bytes,
                           "decomp_off");
    Value *decomp_ptr = Builder->CreateGEP(i8_ty, decomp_base_i8, decomp_byte_off,
                                           "decomp_ptr_i8");

    if (inline_rfor_decode) {
      Value *compressed_ptr_raw_local = Builder->CreateLoad(
          childState[attr].mem->getType()->getNonOpaquePointerElementType(),
          childState[attr].mem, "compressed_block_ptr");
      Value *compressed_ptr_i64_local = Builder->CreatePtrToInt(
          Builder->CreateBitCast(compressed_ptr_raw_local, i8_ptr_ty), i64_ty);
      context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
          [&]() {
            Value *ptr_lo = Builder->CreateTrunc(compressed_ptr_i64_local, i32_ty);
            Value *ptr_hi = Builder->CreateTrunc(
                Builder->CreateLShr(compressed_ptr_i64_local,
                                    ConstantInt::get(i64_ty, 32)),
                i32_ty);
            store_tile_broadcast_i32(ptr_lo, broadcast_ptr_lo_slot);
            store_tile_broadcast_i32(ptr_hi, broadcast_ptr_hi_slot);
          });
      Builder->CreateCall(membar_cta);
      Builder->CreateCall(barrier);
      Value *compressed_ptr_lo = Builder->CreateZExt(
          load_tile_broadcast_i32(broadcast_ptr_lo_slot), i64_ty);
      Value *compressed_ptr_hi = Builder->CreateZExt(
          load_tile_broadcast_i32(broadcast_ptr_hi_slot), i64_ty);
      Value *compressed_ptr_i64 = Builder->CreateOr(
          compressed_ptr_lo,
          Builder->CreateShl(compressed_ptr_hi, ConstantInt::get(i64_ty, 32)));
      Value *compressed_ptr_i8 =
          Builder->CreateIntToPtr(compressed_ptr_i64, i8_ptr_ty);
      Value *compressed_ptr_nonnull = Builder->CreateICmpNE(
          compressed_ptr_i8,
          ConstantPointerNull::get(cast<PointerType>(i8_ptr_ty)));
      Value *compressed_ptr_for_parse = Builder->CreateSelect(
          Builder->CreateAnd(block_oid_valid_for_attr, compressed_ptr_nonnull),
          compressed_ptr_i8, fallback_header_i8);
      auto comp_block_bytes_var_name =
          std::string("gdb.rfor.comp_bytes.") + attr.getAttrName();
      auto *comp_block_bytes_ptr =
          context->CreateEntryBlockAlloca(comp_block_bytes_var_name, i64_ty);
      Builder->CreateStore(zero_i64, comp_block_bytes_ptr);
      Value *block_sizes_map_nonnull = Builder->CreateICmpNE(
          block_sizes_oid_map, ConstantPointerNull::get(i32_ptr_ty));
      Value *can_read_block_size = Builder->CreateAnd(block_oid_valid_for_attr,
                                                      block_sizes_map_nonnull);
      context->gen_if(
          {ProteusValue{can_read_block_size, context->createFalse()}})([&]() {
        Value *block_size_ptr = Builder->CreateInBoundsGEP(
            i32_ty, block_sizes_oid_map, block_oid_i64_uniform);
        Value *block_size_i64 =
            Builder->CreateZExt(Builder->CreateLoad(i32_ty, block_size_ptr), i64_ty);
        Builder->CreateStore(block_size_i64, comp_block_bytes_ptr);
      });
      Value *compressed_block_bytes_i64 =
          Builder->CreateLoad(i64_ty, comp_block_bytes_ptr);

      Value *header_i32 =
          Builder->CreateBitCast(compressed_ptr_for_parse, i32_ptr_ty);
      Value *magic_i32 = Builder->CreateLoad(i32_ty, header_i32);
      Value *magic_ok = Builder->CreateICmpEQ(
          magic_i32, ConstantInt::get(i32_ty, 0x50424D44));
      Value *num_chunks_i32 = Builder->CreateLoad(
          i32_ty, Builder->CreateInBoundsGEP(i32_ty, header_i32,
                                             ConstantInt::get(i32_ty, 2)));
      Value *num_chunks_i64 = Builder->CreateZExt(num_chunks_i32, i64_ty);
      Value *rows_from_chunks_i64 = Builder->CreateMul(num_chunks_i64, tile_elems_i64);
      Value *rows_fit_block_i64 = Builder->CreateSelect(
          Builder->CreateICmpULT(rows_from_chunks_i64, max_rows_per_block_i64),
          rows_from_chunks_i64, max_rows_per_block_i64);

      Value *chunk_index_ptr = Builder->CreateInBoundsGEP(
          i32_ty, header_i32, ConstantInt::get(i32_ty, 3));
      Value *val_section_sizes_ptr = Builder->CreateInBoundsGEP(
          i32_ty, chunk_index_ptr, num_chunks_i32);

      Value *num_chunks_bytes =
          Builder->CreateMul(num_chunks_i64, u32_bytes);
      Value *tile_meta_offset = Builder->CreateAdd(
          header_fixed_bytes, Builder->CreateAdd(num_chunks_bytes, num_chunks_bytes));
      Value *header_total_bytes = Builder->CreateAdd(
          tile_meta_offset, Builder->CreateMul(num_chunks_i64, metadata_bytes_i64));
      Value *compressed_block_size_ok = Builder->CreateAnd(
          Builder->CreateICmpUGT(compressed_block_bytes_i64, zero_i64),
          Builder->CreateICmpULE(compressed_block_bytes_i64,
                                 max_reasonable_compressed_bytes_i64));
      Value *header_in_bounds = Builder->CreateICmpULE(header_total_bytes,
                                                       compressed_block_bytes_i64);
      Value *chunk_region_bytes_i64 = Builder->CreateSelect(
          header_in_bounds,
          Builder->CreateSub(compressed_block_bytes_i64, header_total_bytes),
          zero_i64);
      Value *tile_meta_base_i8 = Builder->CreateGEP(
          i8_ty, compressed_ptr_for_parse, tile_meta_offset);
      Value *tile_meta_ptr_base =
          Builder->CreateBitCast(tile_meta_base_i8, rle_tile_metadata_ptr_ty);
      Value *chunk_data_base = Builder->CreateGEP(
          i8_ty, compressed_ptr_for_parse, header_total_bytes);

      Value *has_chunks =
          Builder->CreateICmpUGT(num_chunks_i64, ConstantInt::get(i64_ty, 0));
      Value *has_tiles =
          Builder->CreateICmpUGT(tile_count_i64_uniform, ConstantInt::get(i64_ty, 0));
      Value *fits_output_block =
          Builder->CreateICmpULE(num_chunks_i64, max_tiles_per_block_i64);
      Value *block_dim_ok = Builder->CreateICmpEQ(
          context->blockDim(), ConstantInt::get(i64_ty, kRforBlockThreads));
      // Keep header validation CTT-like and avoid full chunk-sum pre-pass.
      Value *can_decode_header = Builder->CreateAnd(
          block_oid_valid_for_attr,
          Builder->CreateAnd(
              compressed_ptr_nonnull,
              Builder->CreateAnd(
                  compressed_block_size_ok,
                  Builder->CreateAnd(
                      fits_output_block,
                      Builder->CreateAnd(
                          magic_ok,
                          Builder->CreateAnd(has_chunks, header_in_bounds))))));
      Value *safe_rows_for_field = Builder->CreateSelect(
          Builder->CreateAnd(can_decode_header, block_dim_ok), rows_fit_block_i64,
          ConstantInt::get(i64_ty, 0));
      safe_tuple_cnt_i64 = Builder->CreateSelect(
          Builder->CreateICmpULT(safe_tuple_cnt_i64, safe_rows_for_field),
          safe_tuple_cnt_i64, safe_rows_for_field);
      Value *can_decode = Builder->CreateAnd(
          can_decode_header, Builder->CreateAnd(has_tiles, block_dim_ok));
      Value *tile_count_by_input = Builder->CreateSelect(
          Builder->CreateICmpULT(tile_count_i64_uniform, num_chunks_i64),
          tile_count_i64_uniform, num_chunks_i64);
      Value *effective_tile_count = Builder->CreateSelect(
          Builder->CreateICmpULT(tile_count_by_input, max_tiles_per_block_i64),
          tile_count_by_input, max_tiles_per_block_i64);
      Value *can_decode_block = can_decode;
      Value *effective_tile_count_block = Builder->CreateSelect(
          can_decode_block, effective_tile_count, zero_i64);
      // Force loop-control values to be block-uniform before entering the
      // tile loop. A non-uniform loop bound can deadlock at per-tile barriers.
      context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
          [&]() {
            Value *decode_i32 = Builder->CreateZExt(can_decode_block, i32_ty);
            Value *tile_cnt_lo =
                Builder->CreateTrunc(effective_tile_count_block, i32_ty);
            Value *tile_cnt_hi = Builder->CreateTrunc(
                Builder->CreateLShr(effective_tile_count_block,
                                    ConstantInt::get(i64_ty, 32)),
                i32_ty);
            auto *st_decode =
                Builder->CreateStore(decode_i32, broadcast_uniform_decode_slot);
            auto *st_tile_lo =
                Builder->CreateStore(tile_cnt_lo, broadcast_uniform_tilecnt_lo_slot);
            auto *st_tile_hi =
                Builder->CreateStore(tile_cnt_hi, broadcast_uniform_tilecnt_hi_slot);
            st_decode->setVolatile(true);
            st_tile_lo->setVolatile(true);
            st_tile_hi->setVolatile(true);
          });
      Builder->CreateCall(membar_cta);
      Builder->CreateCall(barrier);
      auto *ld_decode =
          Builder->CreateLoad(i32_ty, broadcast_uniform_decode_slot);
      auto *ld_tile_lo =
          Builder->CreateLoad(i32_ty, broadcast_uniform_tilecnt_lo_slot);
      auto *ld_tile_hi =
          Builder->CreateLoad(i32_ty, broadcast_uniform_tilecnt_hi_slot);
      ld_decode->setVolatile(true);
      ld_tile_lo->setVolatile(true);
      ld_tile_hi->setVolatile(true);
      Value *decode_i32_uniform_raw = ld_decode;
      Value *tile_cnt_lo_i32_uniform_raw = ld_tile_lo;
      Value *tile_cnt_hi_i32_uniform_raw = ld_tile_hi;
      can_decode_block = Builder->CreateICmpNE(
          decode_i32_uniform_raw,
          ConstantInt::get(i32_ty, 0));
      Value *tile_cnt_lo_uniform = Builder->CreateZExt(
          tile_cnt_lo_i32_uniform_raw, i64_ty);
      Value *tile_cnt_hi_uniform = Builder->CreateZExt(
          tile_cnt_hi_i32_uniform_raw, i64_ty);
      Value *effective_tile_count_uniform = Builder->CreateOr(
          tile_cnt_lo_uniform,
          Builder->CreateShl(tile_cnt_hi_uniform, ConstantInt::get(i64_ty, 32)));
      effective_tile_count_block = effective_tile_count_uniform;
      can_decode_block = Builder->CreateAnd(
          can_decode_block,
          Builder->CreateICmpUGT(effective_tile_count_block,
                                 ConstantInt::get(i64_ty, 0)));

      auto tile_var_name =
          std::string("gdb.rfor.tile_idx.") + attr.getAttrName();
      auto *tile_idx_ptr = context->CreateEntryBlockAlloca(tile_var_name, i64_ty);
      // Unlike CompressedBlockToTuples, this operator must materialize the
      // full decompressed block before handing it to downstream BlockToTuples.
      // Decode all tiles locally in this CTA.
      Builder->CreateStore(zero_i64, tile_idx_ptr);

      context->gen_while([&]() {
        Value *tile_idx = Builder->CreateLoad(i64_ty, tile_idx_ptr);
        Value *has_more_tiles =
            Builder->CreateICmpULT(tile_idx, effective_tile_count_block);
        return ProteusValue{
            Builder->CreateAnd(can_decode_block, has_more_tiles),
            context->createFalse()};
      })([&](BranchInst * /*loop_cond*/) {
          Value *tile_idx = Builder->CreateLoad(i64_ty, tile_idx_ptr);
          Value *tile_idx_i32 = Builder->CreateTrunc(tile_idx, i32_ty);

	          Value *tile_out_off = Builder->CreateMul(tile_idx, tile_bytes_i64);
	          Value *tile_out_i8 =
	              Builder->CreateInBoundsGEP(i8_ty, decomp_ptr, tile_out_off);
          Value *tile_out_i32 = Builder->CreateBitCast(tile_out_i8, i32_ptr_ty);

	          if (use_precomputed_tile_offsets) {
	            context->gen_if(
	                {ProteusValue{is_block_leader, context->createFalse()}})([&]() {
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_decode_flag_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_chunk_off_lo_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_chunk_off_hi_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_val_sec_lo_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_val_sec_hi_slot);

	              Value *tile_in_range =
	                  Builder->CreateICmpULT(tile_idx, num_chunks_i64);
	              context->gen_if(
	                  {ProteusValue{tile_in_range, context->createFalse()}})([&]() {
	                Value *chunk_offsets_ptr_ptr = Builder->CreateInBoundsGEP(
	                    i32_ptr_ty, chunk_offsets_oid_map, block_oid_i64_uniform);
	                Value *val_section_sizes_ptr_ptr = Builder->CreateInBoundsGEP(
	                    i32_ptr_ty, val_section_sizes_oid_map, block_oid_i64_uniform);
	                Value *chunk_offsets_ptr =
	                    Builder->CreateLoad(i32_ptr_ty, chunk_offsets_ptr_ptr);
	                Value *val_section_sizes_data_ptr =
	                    Builder->CreateLoad(i32_ptr_ty, val_section_sizes_ptr_ptr);
	                Value *maps_ok = Builder->CreateAnd(
	                    Builder->CreateICmpNE(
	                        chunk_offsets_ptr,
	                        ConstantPointerNull::get(i32_ptr_ty)),
	                    Builder->CreateICmpNE(
	                        val_section_sizes_data_ptr,
	                        ConstantPointerNull::get(i32_ptr_ty)));
	                context->gen_if({ProteusValue{maps_ok, context->createFalse()}})(
	                    [&]() {
		                      Value *chunk_offset_ptr = Builder->CreateInBoundsGEP(
		                          i32_ty, chunk_offsets_ptr, tile_idx_i32);
		                      Value *val_section_size_ptr = Builder->CreateInBoundsGEP(
		                          i32_ty, val_section_sizes_data_ptr, tile_idx_i32);
		                      Value *chunk_off_i64 = Builder->CreateZExt(
		                          Builder->CreateLoad(i32_ty, chunk_offset_ptr), i64_ty);
		                      Value *val_sec_i64 = Builder->CreateZExt(
		                          Builder->CreateLoad(i32_ty, val_section_size_ptr), i64_ty);
			                      Value *next_tile_idx =
			                          Builder->CreateAdd(tile_idx, one_i64);
			                      Value *has_next_chunk = Builder->CreateICmpULT(
			                          next_tile_idx, num_chunks_i64);
			                      Value *next_tile_idx_i32 =
			                          Builder->CreateTrunc(next_tile_idx, i32_ty);
			                      Value *safe_next_tile_idx_i32 = Builder->CreateSelect(
			                          has_next_chunk, next_tile_idx_i32, tile_idx_i32);
			                      Value *next_chunk_offset_ptr = Builder->CreateInBoundsGEP(
			                          i32_ty, chunk_offsets_ptr, safe_next_tile_idx_i32);
			                      Value *next_chunk_offset_i64 = Builder->CreateSelect(
			                          has_next_chunk,
			                          Builder->CreateZExt(
			                              Builder->CreateLoad(
			                                  i32_ty, next_chunk_offset_ptr),
			                              i64_ty),
			                          chunk_region_bytes_i64);
			                      Value *chunk_size_i64 =
			                          Builder->CreateSub(next_chunk_offset_i64,
			                                             chunk_off_i64);
			                      Value *chunk_end_i64 =
			                          Builder->CreateAdd(chunk_off_i64, chunk_size_i64);
		                      Value *chunk_no_wrap = Builder->CreateICmpUGE(
		                          chunk_end_i64, chunk_off_i64);
		                      Value *chunk_in_bounds = Builder->CreateAnd(
		                          chunk_no_wrap,
		                          Builder->CreateAnd(
		                          Builder->CreateICmpULE(
		                              chunk_off_i64, chunk_region_bytes_i64),
		                          Builder->CreateICmpULE(
		                              chunk_end_i64, chunk_region_bytes_i64)));
		                      Value *tile_layout_ok = Builder->CreateICmpUGE(
		                          chunk_size_i64, metadata_bytes_i64);
		                      Value *val_sec_ok = Builder->CreateAnd(
		                          Builder->CreateICmpUGE(val_sec_i64, metadata_bytes_i64),
		                          Builder->CreateICmpULE(val_sec_i64, chunk_size_i64));
		                      Value *tile_prefix_safe = Builder->CreateAnd(
		                          tile_layout_ok,
		                          Builder->CreateAnd(chunk_in_bounds, val_sec_ok));
		                      context->gen_if(
		                          {ProteusValue{tile_prefix_safe, context->createFalse()}})(
		                          [&]() {
			                      Value *tile_metadata_ptr = Builder->CreateInBoundsGEP(
			                          rle_tile_metadata_ptr_ty
			                              ->getNonOpaquePointerElementType(),
			                          tile_meta_ptr_base, tile_idx);
				                      Value *val_ref = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
				                                  ->getNonOpaquePointerElementType(),
				                              tile_metadata_ptr, 0));
				                      Value *val_bw = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
				                                  ->getNonOpaquePointerElementType(),
				                              tile_metadata_ptr, 1));
				                      Value *val_count = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
			                                  ->getNonOpaquePointerElementType(),
			                              tile_metadata_ptr, 2));
				                      Value *rl_ref = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
				                                  ->getNonOpaquePointerElementType(),
				                              tile_metadata_ptr, 3));
				                      Value *rl_bw = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
				                                  ->getNonOpaquePointerElementType(),
				                              tile_metadata_ptr, 4));
				                      Value *rl_count = Builder->CreateLoad(
				                          i32_ty,
				                          Builder->CreateStructGEP(
				                              rle_tile_metadata_ptr_ty
				                                  ->getNonOpaquePointerElementType(),
				                              tile_metadata_ptr, 5));
				                      Value *ref_ok = Builder->CreateICmpULE(
				                          rl_ref, ConstantInt::get(i32_ty, kRforTileElems));
				                      Value *bitwidth_ok = Builder->CreateAnd(
				                          Builder->CreateICmpULE(val_bw, ConstantInt::get(i32_ty, 32)),
				                          Builder->CreateICmpULE(rl_bw, ConstantInt::get(i32_ty, 32)));
			                      Value *count_ok = Builder->CreateAnd(
			                          Builder->CreateICmpULE(
			                              val_count, ConstantInt::get(i32_ty, kRforTileElems)),
			                          Builder->CreateICmpULE(
			                              rl_count, ConstantInt::get(i32_ty, kRforTileElems)));
			                      Value *count_match_ok =
			                          Builder->CreateICmpEQ(val_count, rl_count);
			                      Value *val_count_i64 = Builder->CreateZExt(val_count, i64_ty);
			                      Value *rl_count_i64 = Builder->CreateZExt(rl_count, i64_ty);
			                      Value *val_bw_i64 = Builder->CreateZExt(val_bw, i64_ty);
			                      Value *rl_bw_i64 = Builder->CreateZExt(rl_bw, i64_ty);
			                      Value *val_words_i64 = Builder->CreateUDiv(
			                          Builder->CreateAdd(
			                              Builder->CreateMul(val_count_i64, val_bw_i64),
			                              ConstantInt::get(i64_ty, 31)),
			                          ConstantInt::get(i64_ty, 32));
			                      Value *rl_words_i64 = Builder->CreateUDiv(
			                          Builder->CreateAdd(
			                              Builder->CreateMul(rl_count_i64, rl_bw_i64),
			                              ConstantInt::get(i64_ty, 31)),
			                          ConstantInt::get(i64_ty, 32));
			                      Value *val_words_bytes_i64 =
			                          Builder->CreateMul(val_words_i64, u32_bytes);
			                      Value *rl_words_bytes_i64 =
			                          Builder->CreateMul(rl_words_i64, u32_bytes);
			                      Value *val_words_fit = Builder->CreateICmpULE(
			                          Builder->CreateAdd(metadata_bytes_i64, val_words_bytes_i64),
			                          val_sec_i64);
			                      Value *rl_words_fit = Builder->CreateICmpULE(
			                          Builder->CreateAdd(val_sec_i64, rl_words_bytes_i64),
			                          chunk_size_i64);
					                      Value *tile_safe = Builder->CreateAnd(
					                          Builder->CreateAnd(
					                              Builder->CreateAnd(ref_ok, bitwidth_ok),
					                              count_ok),
					                          Builder->CreateAnd(
					                              count_match_ok,
					                              Builder->CreateAnd(val_words_fit, rl_words_fit)));
			                      context->gen_if(
			                          {ProteusValue{tile_safe, context->createFalse()}})([&]() {
		                      Value *chunk_off_lo =
		                          Builder->CreateTrunc(chunk_off_i64, i32_ty);
		                      Value *val_sec_lo = Builder->CreateTrunc(val_sec_i64, i32_ty);
	                      store_tile_broadcast_i32(ConstantInt::get(i32_ty, 1),
	                                               broadcast_decode_flag_slot);
	                      store_tile_broadcast_i32(chunk_off_lo,
	                                               broadcast_chunk_off_lo_slot);
		                      store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
		                                               broadcast_chunk_off_hi_slot);
		                      store_tile_broadcast_i32(val_sec_lo,
		                                               broadcast_val_sec_lo_slot);
		                      store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
		                                               broadcast_val_sec_hi_slot);
		                      });
		                          });
		                    });
		              });
		            });
	          } else {
	            // Metadata prefix/validation is computed once per block on the leader
	            // and broadcast so decode-call control flow stays block-uniform.
	            context->gen_if(
	                {ProteusValue{is_block_leader, context->createFalse()}})([&]() {
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_decode_flag_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_chunk_off_lo_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_chunk_off_hi_slot);
	              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
	                                       broadcast_val_sec_lo_slot);
		              store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
		                                       broadcast_val_sec_hi_slot);

		              Value *val_sec_ptr = Builder->CreateInBoundsGEP(
		                  i32_ty, val_section_sizes_ptr, tile_idx_i32);
		              auto chunk_off_var_name =
		                  std::string("gdb.rfor.tile_chunk_off.") + attr.getAttrName();
		              auto chunk_size_var_name =
		                  std::string("gdb.rfor.tile_chunk_size.") + attr.getAttrName();
		              Value *tile_chunk_off_ptr =
		                  context->CreateEntryBlockAlloca(chunk_off_var_name, i64_ty);
		              Value *tile_chunk_size_ptr =
		                  context->CreateEntryBlockAlloca(chunk_size_var_name, i64_ty);
		              Builder->CreateStore(zero_i64, tile_chunk_off_ptr);
		              Builder->CreateStore(zero_i64, tile_chunk_size_ptr);

		              Value *first_chunk_index_i32 =
		                  Builder->CreateLoad(i32_ty, chunk_index_ptr);
		              Value *pbmd_stores_offsets = Builder->CreateICmpEQ(
		                  first_chunk_index_i32, ConstantInt::get(i32_ty, 0));
		              context->gen_if(
		                  {ProteusValue{pbmd_stores_offsets, context->createFalse()}})([&]() {
		                Value *chunk_offset_ptr =
		                    Builder->CreateInBoundsGEP(i32_ty, chunk_index_ptr, tile_idx_i32);
		                Value *chunk_off_i64 = Builder->CreateZExt(
		                    Builder->CreateLoad(i32_ty, chunk_offset_ptr), i64_ty);
		                Value *next_tile_idx = Builder->CreateAdd(tile_idx, one_i64);
		                Value *has_next_chunk =
		                    Builder->CreateICmpULT(next_tile_idx, num_chunks_i64);
		                Value *next_tile_idx_i32 =
		                    Builder->CreateTrunc(next_tile_idx, i32_ty);
		                Value *safe_next_tile_idx_i32 = Builder->CreateSelect(
		                    has_next_chunk, next_tile_idx_i32, tile_idx_i32);
		                Value *next_chunk_offset_ptr = Builder->CreateInBoundsGEP(
		                    i32_ty, chunk_index_ptr, safe_next_tile_idx_i32);
		                Value *next_chunk_offset_i64 = Builder->CreateSelect(
		                    has_next_chunk,
		                    Builder->CreateZExt(
		                        Builder->CreateLoad(i32_ty, next_chunk_offset_ptr), i64_ty),
		                    chunk_region_bytes_i64);
		                Builder->CreateStore(chunk_off_i64, tile_chunk_off_ptr);
		                Builder->CreateStore(
		                    Builder->CreateSub(next_chunk_offset_i64, chunk_off_i64),
		                    tile_chunk_size_ptr);
		              });

		              context->gen_if(
		                  {ProteusValue{Builder->CreateNot(pbmd_stores_offsets),
		                                context->createFalse()}})([&]() {
		                auto idx_var_name =
		                    std::string("gdb.rfor.chunk_idx.") + attr.getAttrName();
		                auto off_var_name =
		                    std::string("gdb.rfor.chunk_off.") + attr.getAttrName();
		                Value *chunk_idx_ptr =
		                    context->CreateEntryBlockAlloca(idx_var_name, i64_ty);
		                Value *chunk_off_ptr =
		                    context->CreateEntryBlockAlloca(off_var_name, i64_ty);
		                Builder->CreateStore(zero_i64, chunk_idx_ptr);
		                Builder->CreateStore(zero_i64, chunk_off_ptr);

		                context->gen_while([&]() {
		                  Value *idx = Builder->CreateLoad(i64_ty, chunk_idx_ptr);
		                  return ProteusValue{
		                      Builder->CreateICmpULT(idx, tile_idx),
		                      context->createFalse()};
		                })([&](BranchInst * /*prefix_cond*/) {
		                  Value *idx = Builder->CreateLoad(i64_ty, chunk_idx_ptr);
		                  Value *idx_i32 = Builder->CreateTrunc(idx, i32_ty);
		                  Value *prefix_chunk_size_ptr =
		                      Builder->CreateInBoundsGEP(i32_ty, chunk_index_ptr, idx_i32);
		                  Value *prefix_chunk_size_i64 = Builder->CreateZExt(
		                      Builder->CreateLoad(i32_ty, prefix_chunk_size_ptr), i64_ty);
		                  Value *off = Builder->CreateLoad(i64_ty, chunk_off_ptr);
		                  Builder->CreateStore(
		                      Builder->CreateAdd(off, prefix_chunk_size_i64),
		                      chunk_off_ptr);
		                  Builder->CreateStore(
		                      Builder->CreateAdd(idx, one_i64), chunk_idx_ptr);
		                });

		                Value *chunk_off_i64 = Builder->CreateLoad(i64_ty, chunk_off_ptr);
		                Value *chunk_size_ptr =
		                    Builder->CreateInBoundsGEP(i32_ty, chunk_index_ptr, tile_idx_i32);
		                Builder->CreateStore(chunk_off_i64, tile_chunk_off_ptr);
		                Builder->CreateStore(
		                    Builder->CreateZExt(
		                        Builder->CreateLoad(i32_ty, chunk_size_ptr), i64_ty),
		                    tile_chunk_size_ptr);
		              });

		              Value *chunk_off_i64 = Builder->CreateLoad(i64_ty, tile_chunk_off_ptr);
		              Value *chunk_size_i64 =
		                  Builder->CreateLoad(i64_ty, tile_chunk_size_ptr);
			              Value *val_sec_i64 = Builder->CreateZExt(
			                  Builder->CreateLoad(i32_ty, val_sec_ptr), i64_ty);

			              Value *val_sec_ok = Builder->CreateAnd(
			                  Builder->CreateICmpUGE(val_sec_i64, metadata_bytes_i64),
			                  Builder->CreateICmpULE(val_sec_i64, chunk_size_i64));
			              Value *tile_layout_ok =
			                  Builder->CreateICmpUGE(chunk_size_i64, metadata_bytes_i64);
              Value *chunk_end_i64 =
                  Builder->CreateAdd(chunk_off_i64, chunk_size_i64);
              Value *chunk_no_wrap =
                  Builder->CreateICmpUGE(chunk_end_i64, chunk_off_i64);
              Value *chunk_in_bounds = Builder->CreateAnd(
                  chunk_no_wrap,
                  Builder->CreateAnd(
                      Builder->CreateICmpULE(chunk_off_i64, chunk_region_bytes_i64),
                      Builder->CreateICmpULE(chunk_end_i64, chunk_region_bytes_i64)));
			              Value *tile_prefix_safe = Builder->CreateAnd(
			                  tile_layout_ok, Builder->CreateAnd(chunk_in_bounds, val_sec_ok));

			              context->gen_if(
			                  {ProteusValue{tile_prefix_safe, context->createFalse()}})([&]() {
                    Value *tile_metadata_ptr = Builder->CreateInBoundsGEP(
                        rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
                        tile_meta_ptr_base, tile_idx);
	                    Value *val_bw = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 1));
	                    Value *val_ref = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 0));
	                    Value *val_count = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 2));
	                    Value *rl_bw = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 4));
	                    Value *rl_ref = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 3));
	                    Value *rl_count = Builder->CreateLoad(
	                        i32_ty,
	                        Builder->CreateStructGEP(
	                            rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
	                            tile_metadata_ptr, 5));
	                    Value *ref_ok = Builder->CreateICmpULE(
	                        rl_ref, ConstantInt::get(i32_ty, kRforTileElems));

	                    Value *bitwidth_ok = Builder->CreateAnd(
	                        Builder->CreateICmpULE(val_bw, ConstantInt::get(i32_ty, 32)),
                        Builder->CreateICmpULE(rl_bw, ConstantInt::get(i32_ty, 32)));
                    Value *count_ok = Builder->CreateAnd(
                        Builder->CreateICmpULE(
                            val_count, ConstantInt::get(i32_ty, kRforTileElems)),
                        Builder->CreateICmpULE(
                            rl_count, ConstantInt::get(i32_ty, kRforTileElems)));
                    Value *count_match_ok = Builder->CreateICmpEQ(val_count, rl_count);
                    Value *val_count_i64 = Builder->CreateZExt(val_count, i64_ty);
                    Value *rl_count_i64 = Builder->CreateZExt(rl_count, i64_ty);
                    Value *val_bw_i64 = Builder->CreateZExt(val_bw, i64_ty);
                    Value *rl_bw_i64 = Builder->CreateZExt(rl_bw, i64_ty);
                    Value *val_words_i64 = Builder->CreateUDiv(
                        Builder->CreateAdd(
                            Builder->CreateMul(val_count_i64, val_bw_i64),
                            ConstantInt::get(i64_ty, 31)),
                        ConstantInt::get(i64_ty, 32));
                    Value *rl_words_i64 = Builder->CreateUDiv(
                        Builder->CreateAdd(
                            Builder->CreateMul(rl_count_i64, rl_bw_i64),
                            ConstantInt::get(i64_ty, 31)),
                        ConstantInt::get(i64_ty, 32));
                    Value *val_words_bytes_i64 =
                        Builder->CreateMul(val_words_i64, u32_bytes);
                    Value *rl_words_bytes_i64 =
                        Builder->CreateMul(rl_words_i64, u32_bytes);
                    Value *val_words_fit = Builder->CreateICmpULE(
                        Builder->CreateAdd(metadata_bytes_i64, val_words_bytes_i64),
                        val_sec_i64);
                    Value *rl_words_fit = Builder->CreateICmpULE(
                        Builder->CreateAdd(val_sec_i64, rl_words_bytes_i64),
                        chunk_size_i64);
	                    Value *tile_safe = Builder->CreateAnd(
	                        Builder->CreateAnd(
	                            Builder->CreateAnd(ref_ok, bitwidth_ok), count_ok),
	                        Builder->CreateAnd(
	                            count_match_ok,
	                            Builder->CreateAnd(val_words_fit, rl_words_fit)));

		              context->gen_if({ProteusValue{tile_safe, context->createFalse()}})(
		                  [&]() {
		                    Value *chunk_off_lo = Builder->CreateTrunc(chunk_off_i64, i32_ty);
	                    Value *val_sec_lo = Builder->CreateTrunc(val_sec_i64, i32_ty);
	                    store_tile_broadcast_i32(ConstantInt::get(i32_ty, 1),
	                                             broadcast_decode_flag_slot);
	                    store_tile_broadcast_i32(chunk_off_lo,
	                                             broadcast_chunk_off_lo_slot);
		                    store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
		                                             broadcast_chunk_off_hi_slot);
		                    store_tile_broadcast_i32(val_sec_lo,
		                                             broadcast_val_sec_lo_slot);
		                    store_tile_broadcast_i32(ConstantInt::get(i32_ty, 0),
		                                             broadcast_val_sec_hi_slot);
		                  });
                  });
		            });
		          }
	          Builder->CreateCall(membar_cta);
	          Builder->CreateCall(barrier);

	          Value *tile_can_decode = Builder->CreateICmpNE(
	              load_tile_broadcast_i32(broadcast_decode_flag_slot),
	              ConstantInt::get(i32_ty, 0));
          context->gen_if({ProteusValue{tile_can_decode, context->createFalse()}})(
              [&]() {
                Value *chunk_off_lo = Builder->CreateZExt(
                    load_tile_broadcast_i32(broadcast_chunk_off_lo_slot), i64_ty);
	                Value *chunk_off_i64 = chunk_off_lo;
	                Value *val_sec_lo = Builder->CreateZExt(
	                    load_tile_broadcast_i32(broadcast_val_sec_lo_slot), i64_ty);
	                Value *val_sec_i64 = val_sec_lo;

			                Value *chunk_base =
			                    Builder->CreateInBoundsGEP(i8_ty, chunk_data_base, chunk_off_i64);
			                Value *decode_metadata_ptr = Builder->CreateInBoundsGEP(
			                    rle_tile_metadata_ptr_ty->getNonOpaquePointerElementType(),
			                    tile_meta_ptr_base, tile_idx);
			                Value *val_data_i8 =
			                    Builder->CreateInBoundsGEP(i8_ty, chunk_base, metadata_bytes_i64);
	                Value *val_data_ptr = Builder->CreateBitCast(val_data_i8, i32_ptr_ty);
		                Value *rl_data_i8 =
		                    Builder->CreateInBoundsGEP(i8_ty, chunk_base, val_sec_i64);
		                Value *rl_data_ptr = Builder->CreateBitCast(rl_data_i8, i32_ptr_ty);
		                std::vector<Value *> decode_args{
		                    decode_metadata_ptr, val_data_ptr, rl_data_ptr,
		                    shared_buffer_generic_ptr, shared_tile_out_ptr};
		                Builder->CreateCall(load_rfor_func, decode_args);
                // Match CompressedBlockToTuples at the decode boundary:
                // decode into shared memory first, then materialize the tile
                // into the persistent global output block.
                Builder->CreateCall(barrier);
	                for (uint32_t item = 0; item < kRforItemsPerThread; ++item) {
	                  Value *elem_idx = Builder->CreateAdd(
	                      context->threadIdInBlock(),
                      ConstantInt::get(i64_ty,
                                       static_cast<uint64_t>(item) *
                                           kRforBlockThreads));
                  Value *shared_elem_ptr = Builder->CreateInBoundsGEP(
                      i32_ty, shared_tile_out_ptr, elem_idx);
                  Value *global_elem_ptr =
                      Builder->CreateInBoundsGEP(i32_ty, tile_out_i32, elem_idx);
	                  Value *elem_val = Builder->CreateLoad(i32_ty, shared_elem_ptr);
	                  Builder->CreateStore(elem_val, global_elem_ptr);
	                }
              });
          Builder->CreateCall(barrier);

          Value *next_tile = Builder->CreateAdd(tile_idx, one_i64);
          Builder->CreateStore(next_tile, tile_idx_ptr);
          // Keep CTA progression aligned across tile iterations.
          Builder->CreateCall(barrier);
        });
    }

    std::string currBufVar = std::string("decomp.") + attr.getAttrName() + "_ptr";
    AllocaInst *mem_currResult =
        context->CreateEntryBlockAlloca(F, currBufVar, ptr_t);
    auto value_as_ptr = Builder->CreateBitCast(decomp_ptr, ptr_t);
    Builder->CreateStore(value_as_ptr, mem_currResult);

    ProteusValueMemory mem_valWrapper{mem_currResult, context->createFalse()};
    variableBindings[block_attr] = mem_valWrapper;
    ++block_attr_no;
  }

  if (inline_rfor_decode) {
    context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
        [&]() {
          Value *safe_cnt_lo = Builder->CreateTrunc(safe_tuple_cnt_i64, i32_ty);
          Value *safe_cnt_hi = Builder->CreateTrunc(
              Builder->CreateLShr(safe_tuple_cnt_i64,
                                  ConstantInt::get(i64_ty, 32)),
              i32_ty);
          store_tile_broadcast_i32(safe_cnt_lo, broadcast_safe_cnt_lo_slot);
          store_tile_broadcast_i32(safe_cnt_hi, broadcast_safe_cnt_hi_slot);
        });
    Builder->CreateCall(membar_cta);
    Builder->CreateCall(barrier);
    Value *safe_cnt_lo_uniform = Builder->CreateZExt(
        load_tile_broadcast_i32(broadcast_safe_cnt_lo_slot), i64_ty);
    Value *safe_cnt_hi_uniform = Builder->CreateZExt(
        load_tile_broadcast_i32(broadcast_safe_cnt_hi_slot), i64_ty);
    safe_tuple_cnt_i64 = Builder->CreateOr(
        safe_cnt_lo_uniform,
        Builder->CreateShl(safe_cnt_hi_uniform, ConstantInt::get(i64_ty, 32)));

    auto *safe_tuple_cnt_cast = Builder->CreateIntCast(
        safe_tuple_cnt_i64, pg->getOIDType()->getLLVMType(llvmContext), false);
    auto *safe_cnt_mem = context->CreateEntryBlockAlloca(
        F, relName + ".activeCnt_safe_ptr", safe_tuple_cnt_cast->getType());
    Builder->CreateStore(safe_tuple_cnt_cast, safe_cnt_mem);

    RecordAttribute activeCnt{relName, "activeCnt", pg->getOIDType()};
    RecordAttribute tupleCnt{relName, "tupleCnt", pg->getOIDType()};
    ProteusValueMemory safe_cnt_mem_wrapper{safe_cnt_mem, context->createFalse()};
    variableBindings[activeCnt] = safe_cnt_mem_wrapper;
    variableBindings[tupleCnt] = safe_cnt_mem_wrapper;
  }



  // GpuDecompressBlocks materializes each output block cooperatively across the
  // whole CTA. Do not let BlockToTuples start reading global output buffers
  // until every lane finished writing the final attribute.
  if (inline_rfor_decode && barrier) {
    Builder->CreateCall(barrier);
  }

  OperatorState state{*this, variableBindings};
  this->getParent()->consume(context, state);
}
