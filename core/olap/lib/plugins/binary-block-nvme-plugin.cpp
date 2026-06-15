/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <rapidjson/document.h>

#include <magic_enum.hpp>
#include <gflags/gflags.h>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <regex>
#include <variant>


#include "lib/operators/operators.hpp"
#include "olap/util/parallel-context.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
using namespace llvm;

DECLARE_bool(decom_op);
DECLARE_bool(in_gpu);
DECLARE_bool(use_metadata_in_block);
DECLARE_int32(tile_size);

extern "C" {
void *getNvmePageIdPtr(NvmePlugin *pg, uint8_t attribute_no,
                       uint8_t partition_no, uint32_t block_no) noexcept {
  DCHECK_LT(attribute_no, pg->m_attribute_metadata.size());
  DCHECK_LT(partition_no, pg->m_attribute_metadata[attribute_no].size());
//  DCHECK_LT(partition_no,
//            pg->m_attribute_metadata[attribute_no][partition_no].num_blocks);
  const uint8_t cpu_numa_affinity =
      pg->m_attribute_metadata[attribute_no][partition_no].numa_node;
  uint64_t page_id =
      (uint64_t)(cpu_numa_affinity | NvmePlugin::PageId_t::page_id_bit_in_numa)
          << 56 |
      (uint64_t)attribute_no << 48 | (uint64_t)partition_no << 40 |
      (uint64_t)block_no;
  return reinterpret_cast<void *>(page_id);
}

void *getNvmePreloadedCompressedBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return pg->getPreloadedCompressedBlockPtr(attribute_no, partition_no,
                                            block_no);
}

void *getNvmePreloadedDecompressedBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return pg->getPreloadedDecompressedBlockPtr(attribute_no, partition_no,
                                              block_no);
}

void *getNvmePreloadedScanBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return pg->getPreloadedScanBlockPtr(attribute_no, partition_no, block_no);
}

void *getNvmePreloadedCompressedChunkOffsetsPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return pg->getPreloadedCompressedChunkOffsetsPtr(attribute_no, partition_no,
                                                   block_no);
}

void *getNvmePreloadedCompressedValSectionSizesPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return pg->getPreloadedCompressedValSectionSizesPtr(attribute_no, partition_no,
                                                      block_no);
}

void *getNvmePreloadedCompressedChunkAuxSizesPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept {
  return getNvmePreloadedCompressedValSectionSizesPtr(pg, attribute_no,
                                                      partition_no, block_no);
}

size_t getNvmePreloadedCompressedBlockSizeByOid(
    NvmePlugin *pg, uint8_t attribute_no, uint64_t oid) noexcept {
  CHECK(pg) << "NvmePlugin pointer is null";
  CHECK(pg->scanEmitsCompressedBlocks())
      << "Compressed block sizes requested while scan does not emit compressed "
         "blocks";
  CHECK_LT(attribute_no, pg->m_attribute_metadata.size());

  const auto part_count = pg->getPartitionCount();
  CHECK_GT(part_count, 0) << "NvmePlugin has no partitions";

  const size_t part_no = oid % part_count;
  const size_t block_no = oid / part_count;
  CHECK_LT(part_no, pg->m_attribute_metadata[attribute_no].size());

  const auto &part_md = pg->m_attribute_metadata[attribute_no][part_no];
  CHECK_LT(block_no, part_md.block_sizes.size())
      << "Requested block index out of range for attr=" << static_cast<int>(attribute_no)
      << " part=" << part_no << " oid=" << oid;
  return part_md.block_sizes[block_no];
}
}

std::ostream &operator<<(std::ostream &out,
                         const NvmePlugin::PageId_t &page_id) {
  return out << "PageId_t{cpu_numa_affinity="
             << static_cast<int>(page_id.cpu_numa_affinity)
             << ", attribute_no=" << static_cast<int>(page_id.attribute_no)
             << ", partition_no=" << static_cast<int>(page_id.partition_no)
             << ", block_no=" << page_id.block_no << "}";
}

template <class T>
std::vector<RecordAttribute *> getAttributeVector(
    const std::vector<std::pair<RecordAttribute *, T>> &whichFields) {
  std::vector<RecordAttribute *> attrs;
  attrs.reserve(whichFields.size());
  for (const auto &val : whichFields) {
    attrs.emplace_back(new RecordAttribute{*val.first});
  }
  return attrs;
}

template <class T>
RecordType recordTypeFromDataVector(
    const std::vector<std::pair<RecordAttribute *, T>> &whichFields) {
  return getAttributeVector(whichFields);
}

NvmePlugin::NvmePlugin(
    OlapParallelContext *context,
    const std::vector<
        std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
        &whichFields)
    : NvmePlugin(context, whichFields, false, false) {}

NvmePlugin::NvmePlugin(
    OlapParallelContext *context,
    const std::vector<
        std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
        &whichFields,
    bool preload_to_gpu, bool emit_compressed_blocks)
    : BinaryBlockPlugin(context, whichFields.front().first->getRelationName(),
                        recordTypeFromDataVector(whichFields),
                        getAttributeVector(whichFields), false),
      preload_to_gpu_(preload_to_gpu),
      emit_compressed_blocks_(emit_compressed_blocks) {
  Nparts = whichFields.front().second.size();
  for (const auto &field : whichFields) {
    CHECK_EQ(field.second.size(), Nparts)
        << "All attributes must have the same number of partitions. This field "
           "attribute: "
        << field.first->getAttrName();
  }

  // TODO fix m_attribute_metadata move / copy constructors
  m_attribute_metadata.reserve(whichFields.size());
  for (auto &field : whichFields) {
    size_t blocks_for_field = 0;
    m_attribute_metadata.push_back({});
    m_attribute_metadata.back().reserve(Nparts);
    for (auto &attr_part_meta : field.second) {
      m_attribute_metadata.back().emplace_back(attr_part_meta);
      blocks_for_field += m_attribute_metadata.back().back().num_blocks;
      //      LOG(INFO) << field.first->getAttrName() << " has "
      //                << m_attribute_metadata.back().back().num_blocks
      //                << " blocks in partition "
      //                << m_attribute_metadata.back().back().data_file_path;
    }
    LOG(INFO) << field.first->getAttrName() << " has " << blocks_for_field
              << " total blocks";
    //  TODO store dict path in metadata
    //  this is ugly
    const auto data_path =
        m_attribute_metadata.back()[0].data_file_path.string();
    std::string dict_path;
    auto &attr = field.first;
    if (attr->getOriginalType()->getTypeID() == DSTRING) {
      std::regex pattern("(.*)(_[0-9]+_[0-9]+)");
      std::smatch matches;
      // fetch the dictionary
      if (std::regex_search(data_path, matches, pattern)) {
        dict_path = matches[1];
      } else {
        LOG(FATAL) << "bad regex";
      }
      void *dict = StorageManager::getInstance().getDictionaryOf(dict_path);
      ((DStringType *)(attr->getOriginalType()))->setDictionary(dict);
    }
  }

  for (int i = 0; i < Nparts; i++) {
    auto num_blocks_in_part = m_attribute_metadata.front()[i].num_blocks;
    part_sizes.emplace_back(num_blocks_in_part);
    for (auto &attr_meta : m_attribute_metadata) {
      CHECK_EQ(attr_meta[i].num_blocks, num_blocks_in_part)
          << "expect the same partition number of each attribute to have the "
             "same number of blocks";
    }
  }

  if (preload_to_gpu_) {
    preloadBlocksToGpu();
    LOG(INFO) << "NvmePlugin preload initialized for " << fnamePrefix
              << " (scan emits "
              << (emit_compressed_blocks_.load(std::memory_order_acquire)
                      ? "compressed"
                      : "decompressed")
              << " blocks)";
  }
}

NvmePlugin::~NvmePlugin() {}

NvmePlugin::PageIOInfo NvmePlugin::getPageIoInfo(
    const PageId_t &page_id) const {
  if (page_id.getAttributeNo() >= m_attribute_metadata.size() ||
      page_id.getPartitionNo() >=
          m_attribute_metadata[page_id.getAttributeNo()].size()) {
    LOG(WARNING) << "PageId_t references non-existent attribute or partition";
    throw std::out_of_range(
        "PageId_t references non-existent attribute or partition");
  }

  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  if (page_id.getBlockNo() >= partMetaData.num_blocks) {
    LOG(WARNING) << "PageId_t references non-existent block";
    throw std::out_of_range("PageId_t references non-existent block");
  }

  int fd = partMetaData.fd;
  DCHECK_GE(fd, 0) << "File descriptor is not valid";
#ifndef NDEBUG
  auto res = fcntl(fd, F_GETFD);
  PCHECK(res != -1) << "File descriptor is not valid for part no: "
                    << page_id.getPartitionNo()
                    << ", block no: " << page_id.getBlockNo()
                    << partMetaData.data_file_path;
#endif
  const off_t *offset = &partMetaData.block_offsets[page_id.getBlockNo()];
  const size_t *size = &partMetaData.block_sizes[page_id.getBlockNo()];
  if (partMetaData.data_format ==
      NvmePlugin::AttributePartMetaData::DataFormat_t::COMPRESSED) {
    auto chunk_sizes = partMetaData.chunk_sizes[page_id.getBlockNo()];
    return {
        fd,   partMetaData.cufile_handle,          offset, size, chunk_sizes,
        true, partMetaData.decompressed_chunk_size};
  }

  return {fd, partMetaData.cufile_handle, offset, size, {}, false, 0};
}

std::pair<llvm::Value *, llvm::Value *> NvmePlugin::getPartitionSizes(
    OlapParallelContext *context, llvm::Value *session_ptr) const {
  uint64_t max_blocks_in_partition = 0;
  // this is assuming that all attrs have the same number of blocks
  for (int i = 0; i < Nparts; i++) {
    auto num_blocks_in_part = m_attribute_metadata.front()[i].num_blocks;
    max_blocks_in_partition =
        std::max(max_blocks_in_partition, num_blocks_in_part);
  }

  return {context->CastPtrToLlvmPtr(
              llvm::PointerType::getUnqual(
                  llvm::ArrayType::get(context->createSizeType(), Nparts)),
              part_sizes.data()),
          context->createSizeT(max_blocks_in_partition)};
}

void NvmePlugin::freePartitionSizes(OlapParallelContext *context,
                                    llvm::Value *v) const {}

void NvmePlugin::generate(const ::Operator &producer,
                          OlapParallelContext *context) {
  return scan(producer, context);
}

/**
 * Updates the loop control variables to point to the next data entry.
 * This method inserts code into the increment block (IncBB) during the scan
 * process. It handles incrementing the partition index or resetting it and
 * advancing block_idx if necessary,
 *
 */
void NvmePlugin::nextEntry(OlapParallelContext *context) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();
  Function *F = Builder->GetInsertBlock()->getParent();

  // Necessary because it's blockCtr that affects the scan loop
  auto mem_blockCtr = NamedValuesBinaryCol.at(blockCtrVar);

  // Necessary because it's the itemCtr that affects the scan loop
  auto part_idx_ptr = NamedValuesBinaryCol.at("part_idx_ptr");

  // Necessary because it's the itemCtr that affects the scan loop
  auto block_idx_ptr = NamedValuesBinaryCol.at("block_idx_ptr");

  // Increment and store back
  BasicBlock *wrapBB = BasicBlock::Create(llvmContext, "incWrap", F);
  BasicBlock *stepBB = BasicBlock::Create(llvmContext, "incStep", F);
  BasicBlock *afterBB = BasicBlock::Create(llvmContext, "incAfter", F);

  Value *part_idx =
      Builder->CreateLoad(part_idx_ptr->getType()->getPointerElementType(),
                          part_idx_ptr, "part_idx");

  auto *size_type = (IntegerType *)part_idx->getType();

  Value *part_N = ConstantInt::get(size_type, Nparts - 1);

  Value *cond = Builder->CreateICmpULT(part_idx, part_N);
  Builder->CreateCondBr(cond, stepBB, wrapBB);

  {
    // IfThen
    // increment part_idx (which partition we are accessing)
    Builder->SetInsertPoint(stepBB);

    Builder->CreateStore(
        Builder->CreateAdd(part_idx, ConstantInt::get(size_type, 1)),
        part_idx_ptr);

    Builder->CreateBr(afterBB);
  }

  {
    // IfElse
    // wrap around back to partition 0 and advance block_idx by 1
    Builder->SetInsertPoint(wrapBB);

    Builder->CreateStore(ConstantInt::get(size_type, 0), part_idx_ptr);

    Value *block_idx =
        Builder->CreateLoad(block_idx_ptr->getType()->getPointerElementType(),
                            block_idx_ptr, "block_idx");
    Builder->CreateStore(
        Builder->CreateAdd(block_idx, ConstantInt::get(size_type, 1)),
        block_idx_ptr);

    Builder->CreateBr(afterBB);
  }

  {
    // IfAfter
    // In both cases update mem_blockCtr which is used for the OID
    Builder->SetInsertPoint(afterBB);

    // itemCtr = block_idx_ptr * Nparts + part_idx_ptr * blockSize
    Value *itemCtr = Builder->CreateAdd(
        Builder->CreateMul(
            Builder->CreateLoad(
                block_idx_ptr->getType()->getPointerElementType(),
                block_idx_ptr),
            ConstantInt::get(size_type, Nparts)),
        Builder->CreateLoad(part_idx_ptr->getType()->getPointerElementType(),
                            part_idx_ptr));

    Builder->CreateStore(itemCtr, mem_blockCtr);
  }
}

void *NvmePlugin::getPreloadedCompressedBlockPtr(uint8_t attribute_no,
                                                 uint8_t partition_no,
                                                 uint32_t block_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_compressed_block_ptrs_.size());
  CHECK_LT(partition_no, preloaded_compressed_block_ptrs_[attribute_no].size());
  CHECK_LT(block_no,
           preloaded_compressed_block_ptrs_[attribute_no][partition_no].size());
  auto *ptr =
      preloaded_compressed_block_ptrs_[attribute_no][partition_no][block_no];
  CHECK_NE(ptr, nullptr)
      << "Compressed block was not preloaded: attr=" << +attribute_no
      << " part=" << +partition_no << " block=" << block_no;
  return ptr;
}

void *NvmePlugin::getPreloadedDecompressedBlockPtr(uint8_t attribute_no,
                                                   uint8_t partition_no,
                                                   uint32_t block_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_decompressed_block_ptrs_.size());
  CHECK_LT(partition_no,
           preloaded_decompressed_block_ptrs_[attribute_no].size());
  CHECK_LT(block_no, preloaded_decompressed_block_ptrs_[attribute_no]
                       [partition_no]
                           .size());
  auto *ptr =
      preloaded_decompressed_block_ptrs_[attribute_no][partition_no][block_no];
  CHECK_NE(ptr, nullptr)
      << "Decompressed block was not preloaded: attr=" << +attribute_no
      << " part=" << +partition_no << " block=" << block_no;
  return ptr;
}

void NvmePlugin::setScanEmitCompressedBlocks(bool emit) {
  CHECK(preload_to_gpu_) << "Changing scan pointer mode requires preload_to_gpu";
  std::lock_guard<std::mutex> lock(predecompress_mu_);
  if (!emit) {
    CHECK(preloaded_decompressed_ready_.load(std::memory_order_acquire))
        << "Cannot switch scan to decompressed pointers before they are ready";
  }
  emit_compressed_blocks_.store(emit, std::memory_order_release);
  LOG(INFO) << "NvmePlugin scan mode for " << fnamePrefix << " -> "
            << (emit ? "compressed" : "decompressed");
}

void NvmePlugin::schedulePreDecompressOnNextScan() {
  CHECK(preload_to_gpu_)
      << "Scheduling pre-decompression requires preload_to_gpu";
  std::lock_guard<std::mutex> lock(predecompress_mu_);
  predecompress_epoch_.fetch_add(1, std::memory_order_acq_rel);
  preloaded_decompressed_ready_.store(false, std::memory_order_release);
  // Keep scan in decompressed-pointer mode for --pre_decom semantics.
  // The execute epoch above guarantees decompression is paid once per execute
  // before scan pointers are consumed.
  emit_compressed_blocks_.store(false, std::memory_order_release);
  LOG(INFO) << "NvmePlugin pre-decompress scheduled for " << fnamePrefix
            << " (scan stays in decompressed-pointer mode)";
}

void *NvmePlugin::getPreloadedScanBlockPtr(uint8_t attribute_no,
                                           uint8_t partition_no,
                                           uint32_t block_no) const {
  bool emit_compressed =
      emit_compressed_blocks_.load(std::memory_order_acquire);

  // If scan mode requests decompressed pointers, ensure they are ready.
  if (!emit_compressed &&
      !preloaded_decompressed_ready_.load(std::memory_order_acquire)) {
    auto *self = const_cast<NvmePlugin *>(this);
    self->ensurePreloadedDecompressedBlocks();
  }

  if (emit_compressed) {
    return getPreloadedCompressedBlockPtr(attribute_no, partition_no, block_no);
  }
  return getPreloadedDecompressedBlockPtr(attribute_no, partition_no, block_no);
}

void *NvmePlugin::getPreloadedCompressedChunkOffsetsPtr(
    uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_compressed_chunk_offsets_ptrs_.size());
  CHECK_LT(partition_no,
           preloaded_compressed_chunk_offsets_ptrs_[attribute_no].size());
  CHECK_LT(block_no,
           preloaded_compressed_chunk_offsets_ptrs_[attribute_no][partition_no]
               .size());
  return preloaded_compressed_chunk_offsets_ptrs_[attribute_no][partition_no]
                                              [block_no];
}

void *NvmePlugin::getPreloadedCompressedValSectionSizesPtr(
    uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_compressed_val_section_sizes_ptrs_.size());
  CHECK_LT(partition_no,
           preloaded_compressed_val_section_sizes_ptrs_[attribute_no].size());
  CHECK_LT(
      block_no,
      preloaded_compressed_val_section_sizes_ptrs_[attribute_no][partition_no]
          .size());
  return preloaded_compressed_val_section_sizes_ptrs_[attribute_no][partition_no]
                                                 [block_no];
}

size_t NvmePlugin::getPartitionBlockCount(size_t partition_no) const {
  CHECK_LT(partition_no, part_sizes.size());
  return part_sizes[partition_no];
}

void *NvmePlugin::getPreloadedDecompressedPartBasePtr(
    size_t attribute_no, size_t partition_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_decompressed_block_ptrs_.size());
  CHECK_LT(partition_no, preloaded_decompressed_block_ptrs_[attribute_no].size());

  const auto &part_blocks =
      preloaded_decompressed_block_ptrs_[attribute_no][partition_no];
  if (part_blocks.empty()) {
    return nullptr;
  }

  auto *ptr = part_blocks[0];
  CHECK_NE(ptr, nullptr)
      << "Decompressed part base was not preloaded: attr=" << attribute_no
      << " part=" << partition_no;
  return ptr;
}

void *NvmePlugin::getPreloadedCompressedChunkOffsetsOidMapPtr(
    uint8_t attribute_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_compressed_chunk_offsets_oid_maps_.size());
  const auto &map_file = preloaded_compressed_chunk_offsets_oid_maps_[attribute_no];
  if (!map_file) return nullptr;
  return const_cast<void *>(map_file->getData());
}

void *NvmePlugin::getPreloadedCompressedValSectionSizesOidMapPtr(
    uint8_t attribute_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no,
           preloaded_compressed_val_section_sizes_oid_maps_.size());
  const auto &map_file =
      preloaded_compressed_val_section_sizes_oid_maps_[attribute_no];
  if (!map_file) return nullptr;
  return const_cast<void *>(map_file->getData());
}

void *NvmePlugin::getPreloadedCompressedBlockSizesOidMapPtr(
    uint8_t attribute_no) const {
  CHECK(preload_to_gpu_) << "Requested preloaded pointer without preload mode";
  CHECK_LT(attribute_no, preloaded_compressed_block_sizes_oid_maps_.size());
  const auto &map_file = preloaded_compressed_block_sizes_oid_maps_[attribute_no];
  if (!map_file) return nullptr;
  return const_cast<void *>(map_file->getData());
}

void NvmePlugin::preloadBlocksToGpu() {
  const int gpu_index = topology::getInstance().getActiveGpu().index_in_topo;
  set_device_on_scope device_scope(
      topology::getInstance().getGpus().at(gpu_index));

  const size_t attr_count = m_attribute_metadata.size();
  preloaded_compressed_part_files_.clear();
  preloaded_decompressed_part_files_.clear();
  preloaded_compressed_chunk_offsets_part_files_.clear();
  preloaded_compressed_val_section_sizes_part_files_.clear();
  preloaded_compressed_chunk_offsets_oid_maps_.clear();
  preloaded_compressed_val_section_sizes_oid_maps_.clear();
  preloaded_compressed_block_sizes_oid_maps_.clear();
  preloaded_compressed_block_ptrs_.clear();
  preloaded_decompressed_block_ptrs_.clear();
  preloaded_compressed_chunk_offsets_ptrs_.clear();
  preloaded_compressed_val_section_sizes_ptrs_.clear();
  preloaded_decompressed_ready_.store(false, std::memory_order_release);
  gpu_preloaded_compressed_bytes_.store(0, std::memory_order_release);
  gpu_preloaded_decompressed_bytes_.store(0, std::memory_order_release);
  gpu_preloaded_aux_bytes_.store(0, std::memory_order_release);
  preloaded_compressed_part_files_.resize(attr_count);
  preloaded_decompressed_part_files_.resize(attr_count);
  preloaded_compressed_chunk_offsets_part_files_.resize(attr_count);
  preloaded_compressed_val_section_sizes_part_files_.resize(attr_count);
  preloaded_compressed_chunk_offsets_oid_maps_.resize(attr_count);
  preloaded_compressed_val_section_sizes_oid_maps_.resize(attr_count);
  preloaded_compressed_block_sizes_oid_maps_.resize(attr_count);
  preloaded_compressed_block_ptrs_.resize(attr_count);
  preloaded_decompressed_block_ptrs_.resize(attr_count);
  preloaded_compressed_chunk_offsets_ptrs_.resize(attr_count);
  preloaded_compressed_val_section_sizes_ptrs_.resize(attr_count);

  for (size_t attr_no = 0; attr_no < attr_count; ++attr_no) {
    CHECK_LT(attr_no, wantedFields.size());
    const auto &attr_md = m_attribute_metadata[attr_no];
    const size_t part_count = attr_md.size();

    preloaded_compressed_part_files_[attr_no].resize(part_count);
    preloaded_decompressed_part_files_[attr_no].resize(part_count);
    preloaded_compressed_chunk_offsets_part_files_[attr_no].resize(part_count);
    preloaded_compressed_val_section_sizes_part_files_[attr_no].resize(
        part_count);
    preloaded_compressed_block_ptrs_[attr_no].resize(part_count);
    preloaded_decompressed_block_ptrs_[attr_no].resize(part_count);
    preloaded_compressed_chunk_offsets_ptrs_[attr_no].resize(part_count);
    preloaded_compressed_val_section_sizes_ptrs_[attr_no].resize(part_count);

    for (size_t part_no = 0; part_no < part_count; ++part_no) {
      const auto &part_meta = attr_md[part_no];
      const size_t block_count = part_meta.num_blocks;
      preloaded_compressed_block_ptrs_[attr_no][part_no].resize(block_count,
                                                                 nullptr);
      preloaded_decompressed_block_ptrs_[attr_no][part_no].resize(block_count,
                                                                   nullptr);
      preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no].resize(
          block_count, nullptr);
      preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no].resize(
          block_count, nullptr);

      if (part_meta.data_format == AttributePartMetaData::DataFormat_t::UNCOMPRESSED) {
        CHECK(!emit_compressed_blocks_.load(std::memory_order_acquire))
            << "Cannot emit compressed blocks for an uncompressed part: "
            << part_meta.data_file_path;
        preloaded_decompressed_part_files_[attr_no][part_no] =
            std::make_unique<mmap_file>(part_meta.data_file_path.string(),
                                        data_loc::GPU_RESIDENT);
        gpu_preloaded_decompressed_bytes_.fetch_add(
            preloaded_decompressed_part_files_[attr_no][part_no]->getFileSize(),
            std::memory_order_acq_rel);
        auto *base = static_cast<char *>(const_cast<void *>(
            preloaded_decompressed_part_files_[attr_no][part_no]->getData()));
        for (size_t block_no = 0; block_no < block_count; ++block_no) {
          preloaded_decompressed_block_ptrs_[attr_no][part_no][block_no] =
              base + part_meta.block_offsets[block_no];
        }
        continue;
      }

      // Keep compressed partitions GPU-resident. Decompression policy is
      // chosen by operators (CompressedBlockToTuples* / PreDecompress), not by
      // plugin.
      preloaded_compressed_part_files_[attr_no][part_no] =
          std::make_unique<mmap_file>(part_meta.data_file_path.string(),
                                      data_loc::GPU_RESIDENT);
      gpu_preloaded_compressed_bytes_.fetch_add(
          preloaded_compressed_part_files_[attr_no][part_no]->getFileSize(),
          std::memory_order_acq_rel);
      auto *compressed_base = static_cast<char *>(const_cast<void *>(
          preloaded_compressed_part_files_[attr_no][part_no]->getData()));
      for (size_t block_no = 0; block_no < block_count; ++block_no) {
        preloaded_compressed_block_ptrs_[attr_no][part_no][block_no] =
            compressed_base + part_meta.block_offsets[block_no];
      }

      size_t total_chunks = 0;
      for (size_t block_no = 0; block_no < block_count; ++block_no) {
        CHECK_LT(block_no, part_meta.chunk_sizes.size())
            << "Missing chunk_sizes for compressed block " << block_no
            << " in " << part_meta.data_file_path;
        total_chunks += part_meta.chunk_sizes[block_no].size();
      }

      if (total_chunks > 0) {
        if (part_meta.pbmd_chunk_offsets) {
          const bool has_header_val_section_sizes =
              part_meta.compression_algorithm == "RFOR" ||
              part_meta.compression_algorithm == "FLBP";
          for (size_t block_no = 0; block_no < block_count; ++block_no) {
            const auto chunk_count = part_meta.chunk_sizes[block_no].size();
            if (chunk_count == 0) {
              preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no]
                                                      [block_no] = nullptr;
              preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no]
                                                         [block_no] = nullptr;
              continue;
            }

            auto *block_base_i32 = reinterpret_cast<uint32_t *>(
                compressed_base + part_meta.block_offsets[block_no]);
            auto *chunk_offsets_ptr = block_base_i32 + 3;
            auto *val_section_sizes_ptr =
                has_header_val_section_sizes ? (chunk_offsets_ptr + chunk_count)
                                             : nullptr;
            preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no][block_no] =
                chunk_offsets_ptr;
            preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no]
                                                       [block_no] =
                val_section_sizes_ptr;
          }
        } else {
          std::vector<uint32_t> h_chunk_offsets(total_chunks);
          std::vector<uint32_t> h_val_section_sizes(total_chunks);
          std::vector<size_t> block_chunk_base(block_count, 0);

          size_t chunk_cursor = 0;
          for (size_t block_no = 0; block_no < block_count; ++block_no) {
            const auto &chunk_sizes = part_meta.chunk_sizes[block_no];
            CHECK_LT(block_no, part_meta.val_section_sizes.size())
                << "Missing val_section_sizes for compressed block "
                << block_no << " in " << part_meta.data_file_path;
            const auto &val_section_sizes = part_meta.val_section_sizes[block_no];
            CHECK_EQ(chunk_sizes.size(), val_section_sizes.size())
                << "chunk_sizes/val_section_sizes mismatch for block "
                << block_no << " in " << part_meta.data_file_path;

            block_chunk_base[block_no] = chunk_cursor;
            uint32_t running_offset = 0;
            for (size_t tile_no = 0; tile_no < chunk_sizes.size(); ++tile_no) {
              h_chunk_offsets[chunk_cursor] = running_offset;
              h_val_section_sizes[chunk_cursor] = val_section_sizes[tile_no];
              running_offset += chunk_sizes[tile_no];
              ++chunk_cursor;
            }
          }

          void *d_chunk_offsets =
              MemoryManager::mallocGpu(total_chunks * sizeof(uint32_t));
          void *d_val_section_sizes =
              MemoryManager::mallocGpu(total_chunks * sizeof(uint32_t));
          gpu_run(cudaMemcpy(d_chunk_offsets, h_chunk_offsets.data(),
                             total_chunks * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
          gpu_run(cudaMemcpy(d_val_section_sizes, h_val_section_sizes.data(),
                             total_chunks * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

          preloaded_compressed_chunk_offsets_part_files_[attr_no][part_no] =
              std::make_unique<mmap_file>(d_chunk_offsets,
                                          total_chunks * sizeof(uint32_t));
          preloaded_compressed_val_section_sizes_part_files_[attr_no][part_no] =
              std::make_unique<mmap_file>(d_val_section_sizes,
                                          total_chunks * sizeof(uint32_t));
          gpu_preloaded_aux_bytes_.fetch_add(
              total_chunks * sizeof(uint32_t) * 2, std::memory_order_acq_rel);

          auto *chunk_offsets_base =
              static_cast<uint32_t *>(d_chunk_offsets);
          auto *val_section_sizes_base =
              static_cast<uint32_t *>(d_val_section_sizes);
          for (size_t block_no = 0; block_no < block_count; ++block_no) {
            const auto chunk_count = part_meta.chunk_sizes[block_no].size();
            if (chunk_count == 0) {
              preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no]
                                                      [block_no] = nullptr;
              preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no]
                                                         [block_no] = nullptr;
              continue;
            }
            const auto base_idx = block_chunk_base[block_no];
            preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no][block_no] =
                chunk_offsets_base + base_idx;
            preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no]
                                                       [block_no] =
                val_section_sizes_base + base_idx;
          }
        }
      }
    }

    if (emit_compressed_blocks_.load(std::memory_order_acquire)) {
      const size_t max_blocks_in_partition =
          part_sizes.empty()
              ? 0
              : *std::max_element(part_sizes.begin(), part_sizes.end());
      const size_t oid_capacity = max_blocks_in_partition * Nparts;

      if (oid_capacity > 0) {
        std::vector<void *> h_chunk_offsets_map(oid_capacity, nullptr);
        std::vector<void *> h_val_section_sizes_map(oid_capacity, nullptr);
        std::vector<uint32_t> h_block_sizes_map(oid_capacity, 0);

        for (size_t part_no = 0; part_no < part_count; ++part_no) {
          const auto block_count =
              preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no].size();
          for (size_t block_no = 0; block_no < block_count; ++block_no) {
            const size_t oid = block_no * Nparts + part_no;
            CHECK_LT(oid, oid_capacity);
            h_chunk_offsets_map[oid] =
                preloaded_compressed_chunk_offsets_ptrs_[attr_no][part_no]
                                                       [block_no];
            h_val_section_sizes_map[oid] =
                preloaded_compressed_val_section_sizes_ptrs_[attr_no][part_no]
                                                           [block_no];
            CHECK_LT(attr_no, m_attribute_metadata.size());
            CHECK_LT(part_no, m_attribute_metadata[attr_no].size());
            CHECK_LT(block_no,
                     m_attribute_metadata[attr_no][part_no].block_sizes.size());
            h_block_sizes_map[oid] = static_cast<uint32_t>(
                m_attribute_metadata[attr_no][part_no].block_sizes[block_no]);
          }
        }

        void *d_chunk_offsets_map =
            MemoryManager::mallocGpu(oid_capacity * sizeof(void *));
        void *d_val_section_sizes_map =
            MemoryManager::mallocGpu(oid_capacity * sizeof(void *));
        void *d_block_sizes_map =
            MemoryManager::mallocGpu(oid_capacity * sizeof(uint32_t));
        gpu_run(cudaMemcpy(d_chunk_offsets_map, h_chunk_offsets_map.data(),
                           oid_capacity * sizeof(void *),
                           cudaMemcpyHostToDevice));
        gpu_run(cudaMemcpy(d_val_section_sizes_map,
                           h_val_section_sizes_map.data(),
                           oid_capacity * sizeof(void *),
                           cudaMemcpyHostToDevice));
        gpu_run(cudaMemcpy(d_block_sizes_map, h_block_sizes_map.data(),
                           oid_capacity * sizeof(uint32_t),
                           cudaMemcpyHostToDevice));

        preloaded_compressed_chunk_offsets_oid_maps_[attr_no] =
            std::make_unique<mmap_file>(d_chunk_offsets_map,
                                        oid_capacity * sizeof(void *));
        preloaded_compressed_val_section_sizes_oid_maps_[attr_no] =
            std::make_unique<mmap_file>(d_val_section_sizes_map,
                                        oid_capacity * sizeof(void *));
        preloaded_compressed_block_sizes_oid_maps_[attr_no] =
            std::make_unique<mmap_file>(d_block_sizes_map,
                                        oid_capacity * sizeof(uint32_t));
        gpu_preloaded_aux_bytes_.fetch_add(
            oid_capacity * (2 * sizeof(void *) + sizeof(uint32_t)),
            std::memory_order_acq_rel);
      }
    }
  }
  // Plugin preload never performs decompression; decompressed buffers are
  // produced lazily by PreDecompress/ensurePreloadedDecompressedBlocks().
  preloaded_decompressed_ready_.store(false, std::memory_order_release);
}

void NvmePlugin::ensurePreloadedDecompressedBlocks() {
  CHECK(preload_to_gpu_) << "Decompression requires preload_to_gpu";
  std::lock_guard<std::mutex> lock(predecompress_mu_);
  const uint64_t epoch = predecompress_epoch_.load(std::memory_order_acquire);
  if (preloaded_decompressed_ready_.load(std::memory_order_acquire) &&
      decompressed_epoch_.load(std::memory_order_acquire) == epoch) {
    return;
  }
  runNvmePreDecompressKernel(this);
  decompressed_epoch_.store(epoch, std::memory_order_release);
}

/**
 * Iterates over all partitions of the binary input a round-robin manner.
 * Each iteration sets the OperatorState that will be used by producer
 * @param producer The operator that is producing tuples. (scan.hpp)
 */
void NvmePlugin::scan(const ::Operator &producer,
                      OlapParallelContext *context) {
  LLVMContext &llvmContext = context->getLLVMContext();

  context->setGlobalFunction(true);

  Function *F = context->getGlobalFunction();
  IRBuilder<> *Builder = context->getBuilder();

  // Prepare
  IntegerType *size_type = context->createSizeType();

  // Container for the variable bindings
  map<RecordAttribute, ProteusValueMemory> variableBindings;

  // Get the ENTRY BLOCK
  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  llvm::Value *session = getSession(context);

  auto partsizes = getPartitionSizes(context, session);
  Value *part_sizes_ptr = partsizes.first;
  part_sizes_ptr->setName("part_sizes_ptr");
  Value *maxPackCnt = partsizes.second;  /// size of largest partition in blocks
  maxPackCnt->setName("maxPackCnt");

  // TODO this is now sort of the metadatas responsbility
  // but it also means we restrict ourselves to using the largest field size
  // in the whole table for the block size
  size_t max_field_size = 0;
  for (const auto &f : wantedFields) {
    size_t field_size = context->getSizeOf(f->getLLVMType(llvmContext));
    max_field_size = std::max(field_size, max_field_size);
  }

  ConstantInt *zero_idx = ConstantInt::get(size_type, 0);

  // index of the partition we are currently scanning
  AllocaInst *part_idx_ptr =
      context->CreateEntryBlockAlloca(F, "part_idx_ptr", size_type);
  Builder->CreateStore(zero_idx, part_idx_ptr);
  NamedValuesBinaryCol["part_idx_ptr"] = part_idx_ptr;

  // index of the block we are currently scanning in the partition
  AllocaInst *block_idx_ptr =
      context->CreateEntryBlockAlloca(F, "block_idx_ptr", size_type);
  Builder->CreateStore(zero_idx, block_idx_ptr);
  NamedValuesBinaryCol["block_idx_ptr"] = block_idx_ptr;

  AllocaInst *mem_blockCtr =
      context->CreateEntryBlockAlloca(F, blockCtrVar, size_type);
  Builder->CreateStore(zero_idx, mem_blockCtr);
  NamedValuesBinaryCol[blockCtrVar] = mem_blockCtr;

  auto blockSize =
      ConstantInt::get(size_type, BlockManager::block_size / max_field_size);

  BasicBlock *CondBB = BasicBlock::Create(llvmContext, "scanCond", F);

  // Make the new basic block for the loop header (BODY), inserting after
  // current block.
  BasicBlock *LoopBB = BasicBlock::Create(llvmContext, "scanBody", F);
  BasicBlock *MainBB = BasicBlock::Create(llvmContext, "scanMain", F);

  // Make the new basic block for the increment, inserting after current
  // block.
  BasicBlock *IncBB = BasicBlock::Create(llvmContext, "scanInc", F);

  // Create the "AFTER LOOP" block and insert it.
  BasicBlock *AfterBB = BasicBlock::Create(llvmContext, "scanEnd", F);
  context->setEndingBlock(AfterBB);

  {
    Builder->SetInsertPoint(CondBB);

    // /**
    //  * Equivalent:
    //  * while(block_idx < max(partsize))
    //  */

    Value *block_idx =
        Builder->CreateLoad(block_idx_ptr->getType()->getPointerElementType(),
                            block_idx_ptr, "block_idx");

    Value *cond = Builder->CreateICmpULT(block_idx, maxPackCnt);

    // Insert the conditional branch into the end of CondBB.
    Builder->CreateCondBr(cond, LoopBB, AfterBB);
  }

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  Value *part_idx =
      Builder->CreateLoad(part_idx_ptr->getType()->getPointerElementType(),
                          part_idx_ptr, "part_idx");
  // This load may be redundant
  Value *block_idx =
      Builder->CreateLoad(block_idx_ptr->getType()->getPointerElementType(),
                          block_idx_ptr, "block_idx");

  auto partBlockCntPtr = Builder->CreateInBoundsGEP(
      part_sizes_ptr->getType()->getNonOpaquePointerElementType(),
      part_sizes_ptr, std::vector<Value *>{context->createInt64(0), part_idx});
  Value *partBlockCnt = Builder->CreateLoad(
      partBlockCntPtr->getType()->getPointerElementType(), partBlockCntPtr);
  partBlockCnt->setName("partBlockCnt");

  Value *part_unfinished = Builder->CreateICmpULT(block_idx, partBlockCnt);

  // If we are already at the end of this block jump to IncBB
  Builder->CreateCondBr(part_unfinished, MainBB, IncBB);

  Builder->SetInsertPoint(MainBB);

  // Get the 'oid' of each record and pass it along.
  // More general/lazy plugins will only perform this action,
  // instead of eagerly 'converting' fields
  // FIXME This action corresponds to materializing the oid. Do we want this?
  RecordAttribute tupleIdentifier{
      fnamePrefix, activeLoop,
      this->getOIDType()};  // FIXME: OID type for blocks ?

  // OID is block number (across all partitions)
  // We should probably materialize the OID in unpack as tuple number
  // Not sure if he _have_ to do that
  ProteusValueMemory mem_posWrapper{mem_blockCtr, context->createFalse()};
  variableBindings[tupleIdentifier] = mem_posWrapper;

  // Actual Work (Loop through attributes etc.)
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute attr(*(wantedFields[i]));
    RecordAttribute block_attr(attr, true);

    Type *ptr_t =
        PointerType::get(attr.getLLVMType(context->getLLVMContext()), 0);

    // _should_ be fine, but really we should check
    Value *block_idx_32bit =
        Builder->CreateTruncOrBitCast(block_idx, Builder->getInt32Ty());
    Value *this_ptr = context->getBuilder()->CreateIntToPtr(
        context->createInt64((uintptr_t)this),
        Type::getInt8PtrTy(context->getLLVMContext()));

    Value *block_ptr_or_page_id = nullptr;
    if (preload_to_gpu_) {
      block_ptr_or_page_id = context->gen_call(
          getNvmePreloadedScanBlockPtr,
          {this_ptr, context->createInt8(i), part_idx, block_idx_32bit});
    } else {
      block_ptr_or_page_id = context->gen_call(
          getNvmePageIdPtr,
          {this_ptr, context->createInt8(i), part_idx, block_idx_32bit});
    }

    string bufVarStr = string(bufVar);
    string currBufVar = bufVarStr + "." + attr.getAttrName() + "_ptr";

    AllocaInst *mem_currResult =
        context->CreateEntryBlockAlloca(F, currBufVar, ptr_t);
    auto value_as_ptr = Builder->CreateBitCast(block_ptr_or_page_id, ptr_t);
    Builder->CreateStore(value_as_ptr, mem_currResult);

    ProteusValueMemory mem_valWrapper{mem_currResult, context->createFalse()};
    variableBindings[block_attr] = mem_valWrapper;
  }

  // This alloca stores the element count for the packed output row.
  // For page-id scan path (no preload), activeCnt is "1 block" and memmove
  // converts it to tuple-count later.
  // For preload_to_gpu_ path (direct block pointers), BlockToTuples consumes
  // activeCnt directly as tuple-count, so we set it to rowgroup tuple-count.
  AllocaInst *blockCnt_ptr =
      context->CreateEntryBlockAlloca(F, "blockCnt", partBlockCnt->getType());
  Builder->CreateStore(ConstantInt::get(size_type, 1), blockCnt_ptr);

  /**
   * activeCnt is a special attrName.
   * Here it is used for blockCnt, in mem-move when we convert a rowgroup of
   * PageId_t to real in-memory blocks we need to set activeCnt to the tupleCnt
   * of the rowgroup
   * @see BlockToTuples::consume
   */
  RecordAttribute blockCnt =
      RecordAttribute(fnamePrefix, "activeCnt", this->getOIDType());

  ProteusValueMemory mem_blockCntWrapper;
  mem_blockCntWrapper.mem = blockCnt_ptr;
  mem_blockCntWrapper.isNull = context->createFalse();
  variableBindings[blockCnt] = mem_blockCntWrapper;

  AllocaInst *tupleCnt_ptr =
      context->CreateEntryBlockAlloca(F, "tupleCnt", partBlockCnt->getType());
  Builder->CreateStore(ConstantInt::get(size_type, 1), tupleCnt_ptr);

  RecordAttribute tupleCnt{fnamePrefix, "tupleCnt", this->getOIDType()};
  Value *this_ptr = context->getBuilder()->CreateIntToPtr(
      context->createInt64((uintptr_t)this),
      Type::getInt8PtrTy(context->getLLVMContext()));

  Value *rowgroup_tuple_cnt =
      context->gen_call(&::getRowGroupTupleCount, {part_idx, block_idx, this_ptr});
  Builder->CreateStore(rowgroup_tuple_cnt, tupleCnt_ptr);
  if (preload_to_gpu_) {
    Builder->CreateStore(rowgroup_tuple_cnt, blockCnt_ptr);
  }
  ProteusValueMemory mem_tupleCntWrapper;
  mem_tupleCntWrapper.mem = tupleCnt_ptr;
  mem_tupleCntWrapper.isNull = context->createFalse();
  variableBindings[tupleCnt] = mem_tupleCntWrapper;

  // // Start insertion in IncBB.
  Builder->SetInsertPoint(IncBB);
  nextEntry(context);

  Builder->CreateBr(CondBB);

  Builder->SetInsertPoint(MainBB);

  OperatorState state{producer, variableBindings};
  producer.getParent()->consume(context, state);

  // Insert an explicit fall through from the current (body) block to IncBB.
  Builder->CreateBr(IncBB);

  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  Builder->CreateBr(CondBB);
  // Builder->CreateRetVoid();

  //  Finish up with end (the AfterLoop)
  //  Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(context->getEndingBlock());
  // Builder->SetInsertPoint(AfterBB);

  releaseSession(context, session);
}

NvmePlugin::AttributePartMetaData::AttributePartMetaData(
    const std::filesystem::path &md_path) {
  tile_size = 0;
  decompressed_chunk_size = 0;
  max_compressed_block_size = 0;

  CHECK(std::filesystem::exists(md_path))
      << "AttributePartMetaData file does not exist: " << md_path;
  auto json_metadata = mmap_file{md_path, PAGEABLE};

  const char *buf_json =
      reinterpret_cast<const char *>(json_metadata.getData());

  rapidjson::Document document;
  auto &parsed = document.Parse(buf_json, json_metadata.getFileSize());
  if (parsed.HasParseError()) {
    auto ok = (rapidjson::ParseResult)parsed;
    auto *err = "Error parsing AttributePartMetaData";
    LOG(FATAL) << "Error parsing: " << md_path << " . JSON parse error: "
               << RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()) << " ("
               << ok.Offset() << ")";
  }

  CHECK(document.IsObject());

  CHECK(document.HasMember("data_file"));
  CHECK(document["data_file"].IsString());
  data_file_path = document["data_file"].GetString();
  if (data_file_path.is_relative()) {
    data_file_path = md_path.parent_path() / data_file_path;
  }
  CHECK(std::filesystem::exists(data_file_path))
      << "data_file does not exist: " << data_file_path;

  CHECK(document.HasMember("block_sizes"));
  CHECK(document["block_sizes"].IsArray());
  auto block_sizes_json_array = document["block_sizes"].GetArray();
  block_sizes.reserve(block_sizes_json_array.Size());
  for (auto &v : block_sizes_json_array) {
    CHECK(v.IsUint());
    block_sizes.push_back(v.GetUint());
    // FIXME: chang adm-partition to but the 4k aligned with padding sizes in
    // the metadata write now it write data with padding to be 4k aligned, but
    // does not include the padding in the block size
    if (block_sizes.back() % 4_K != 0) {
      block_sizes.back() += 4_K - (block_sizes.back() % 4_K);
    }
    CHECK(block_sizes.back() % 4_K == 0);
  }

  CHECK(document.HasMember("block_offsets"));
  CHECK(document["block_offsets"].IsArray());
  auto block_offsets_json_array = document["block_offsets"].GetArray();
  block_offsets.reserve(block_offsets_json_array.Size());
  for (auto &v : block_offsets_json_array) {
    CHECK(v.IsUint64()) << " expected Uint64, is: "
                        << magic_enum::enum_name(v.GetType());
    block_offsets.push_back(v.GetUint64());
  }

  CHECK(document.HasMember("value_counts"));
  CHECK(document["value_counts"].IsArray());
  auto value_counts_json_array = document["value_counts"].GetArray();
  value_counts.reserve(value_counts_json_array.Size());
  for (auto &v : value_counts_json_array) {
    CHECK(v.IsUint());
    value_counts.push_back(v.GetUint());
  }

  CHECK_EQ(block_offsets.size(), block_sizes.size());
  CHECK_EQ(value_counts.size(), block_sizes.size());

  CHECK(document.HasMember("num_blocks"));
  CHECK(document["num_blocks"].IsUint64());

  num_blocks = document["num_blocks"].GetUint64();
  CHECK_EQ(block_sizes.size(), num_blocks);

  CHECK(document.HasMember("data_format"));
  CHECK(document["data_format"].IsString());
  auto maybe_data_format =
      magic_enum::enum_cast<DataFormat_t>(document["data_format"].GetString());
  CHECK(maybe_data_format.has_value());
  data_format = maybe_data_format.value();
  compression_algorithm.clear();
  if (document.HasMember("compression_algorithm") &&
      document["compression_algorithm"].IsString()) {
    compression_algorithm = document["compression_algorithm"].GetString();
  }
  pbmd_chunk_offsets = document.HasMember("pbmd_chunk_offsets") &&
                       document["pbmd_chunk_offsets"].IsBool() &&
                       document["pbmd_chunk_offsets"].GetBool();

  if (data_format == DataFormat_t::COMPRESSED) {
    CHECK(document.HasMember("chunk_sizes"));
    CHECK(document["chunk_sizes"].IsArray());
    auto chunk_sizes_json_array = document["chunk_sizes"].GetArray();
    chunk_sizes.reserve(chunk_sizes_json_array.Size());
    for (const auto &block_chunk_sizes_json : chunk_sizes_json_array) {
      CHECK(block_chunk_sizes_json.IsArray());
      std::vector<uint32_t> chunk_sizes_in_block;
      chunk_sizes_in_block.reserve(block_chunk_sizes_json.Size());
      for (const auto &chunk_size_json : block_chunk_sizes_json.GetArray()) {
        CHECK(chunk_size_json.IsUint());
        chunk_sizes_in_block.emplace_back(chunk_size_json.GetUint());
      }
      chunk_sizes.push_back(chunk_sizes_in_block);
    }

    if (document.HasMember("val_section_sizes")) {
      CHECK(document["val_section_sizes"].IsArray());
      auto val_section_sizes_json_array = document["val_section_sizes"].GetArray();
      val_section_sizes.reserve(val_section_sizes_json_array.Size());

      for (const auto &block_val_section_sizes_json : val_section_sizes_json_array) {
        CHECK(block_val_section_sizes_json.IsArray());
        std::vector<uint32_t> val_section_sizes_in_block;
        val_section_sizes_in_block.reserve(block_val_section_sizes_json.Size());

        for (const auto &val_section_size_json : block_val_section_sizes_json.GetArray()) {
          CHECK(val_section_size_json.IsUint());
          val_section_sizes_in_block.emplace_back(val_section_size_json.GetUint());
        }
        val_section_sizes.push_back(val_section_sizes_in_block);
      }
    }

    // It is possible for a partition to be empty
    if (block_sizes.size() > 0) {
      max_compressed_block_size =
          *std::max_element(block_sizes.begin(), block_sizes.end());
    } else {
      max_compressed_block_size = 0;
    }

    CHECK(document.HasMember("decompressed_chunk_size"));
    CHECK(document["decompressed_chunk_size"].IsInt());
    decompressed_chunk_size = document["decompressed_chunk_size"].GetInt();

    const bool is_rfor = compression_algorithm == "RFOR" ||
                         compression_algorithm == "rfor";
    const bool is_flbp = compression_algorithm == "FLBP" ||
                         compression_algorithm == "flbp";
    if (is_rfor) {
      if (document.HasMember("tile_size")) {
        CHECK(document["tile_size"].IsInt());
        tile_size = document["tile_size"].GetInt();
      }

      CHECK(FLAGS_tile_size == 512 || FLAGS_tile_size == 1024)
          << "--tile_size must be 512 or 1024 (got " << FLAGS_tile_size << ")";
      const int expected_tile_size = FLAGS_tile_size;
      const int expected_chunk_bytes =
          static_cast<int>(expected_tile_size * sizeof(int));

      if (tile_size > 0) {
        CHECK_EQ(tile_size, expected_tile_size)
            << "RFOR dataset tile_size mismatch in " << md_path
            << ". dataset tile_size=" << tile_size
            << ", runtime expected tile_size=" << expected_tile_size;
      }

      if (FLAGS_in_gpu) {
        CHECK_EQ(decompressed_chunk_size, expected_chunk_bytes)
            << "RFOR decompressed_chunk_size mismatch in " << md_path
            << ". dataset decompressed_chunk_size=" << decompressed_chunk_size
            << ", runtime expected=" << expected_chunk_bytes
            << " (tile_size=" << expected_tile_size << " ints).";
      } else if (decompressed_chunk_size != expected_chunk_bytes) {
        LOG(WARNING) << "RFOR decompressed_chunk_size (" << decompressed_chunk_size
                     << ") in " << md_path
                     << " differs from in-GPU runtime contract ("
                     << expected_chunk_bytes << ").";
      }
    } else if (is_flbp) {
      if (document.HasMember("vector_size")) {
        CHECK(document["vector_size"].IsInt());
        tile_size = document["vector_size"].GetInt();
      }

      constexpr int expected_vector_size = 1024;
      constexpr int expected_chunk_bytes =
          expected_vector_size * static_cast<int>(sizeof(int));
      if (tile_size > 0) {
        CHECK_EQ(tile_size, expected_vector_size)
            << "FLBP dataset vector_size mismatch in " << md_path
            << ". dataset vector_size=" << tile_size
            << ", runtime expected vector_size=" << expected_vector_size;
      }

      if (FLAGS_in_gpu) {
        CHECK_EQ(decompressed_chunk_size, expected_chunk_bytes)
            << "FLBP decompressed_chunk_size mismatch in " << md_path
            << ". dataset decompressed_chunk_size=" << decompressed_chunk_size
            << ", runtime expected=" << expected_chunk_bytes
            << " (vector_size=" << expected_vector_size << " ints).";
      } else if (decompressed_chunk_size != expected_chunk_bytes) {
        LOG(WARNING) << "FLBP decompressed_chunk_size (" << decompressed_chunk_size
                     << ") in " << md_path
                     << " differs from in-GPU runtime contract ("
                     << expected_chunk_bytes << ").";
      }
    }
  }
  numa_node = fileNameToNumaNodeIndex(data_file_path);
  DLOG(INFO) << "opening data file: " << data_file_path << " for " << md_path;
  fd = open(data_file_path.c_str(), O_DIRECT);
  PCHECK(fd > 0);
#ifndef NDEBUGF
  auto res = fcntl(fd, F_GETFD);
  PCHECK(res != -1) << "File descriptor is not valid";
#endif

  if (fileIsOnNvmeDrive(data_file_path)) {
    CUfileDescr_t cf_descr;
    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t status = cuFileHandleRegister(&cufile_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
      LOG(ERROR)
          << "failed to cuFileHandleRegister for file: " << data_file_path
          << " status: " << cufileop_status_error(status.err)
          << "\n This will only be fatal if cufile APIs are used on this file";
    }
  } else {
    LOG(WARNING) << "file is not on a NVMe drive, it cannot be used with "
                    "cufile APIs: "
                 << data_file_path;
  }

  if (FLAGS_use_metadata_in_block && data_format == DataFormat_t::COMPRESSED &&
      !block_offsets.empty()) {
    constexpr uint32_t kMetadataMagic = 0x50424D44;  // "PBMD"
    uint32_t magic = 0;
    int check_fd = open(data_file_path.c_str(), O_RDONLY);
    PCHECK(check_fd > 0) << "failed to open data file for magic check: "
                         << data_file_path;
    auto read_bytes = pread(check_fd, &magic, sizeof(magic),
                            static_cast<off_t>(block_offsets[0]));
    PCHECK(read_bytes == static_cast<ssize_t>(sizeof(magic)))
        << "failed to read metadata magic from: " << data_file_path;
    close(check_fd);
    CHECK_EQ(magic, kMetadataMagic)
        << "metadata magic mismatch for: " << data_file_path
        << " (did you generate with --metadata_in_block=true?)";
  }
}

std::vector<uint32_t> NvmePlugin::getValSectionSizes(const PageId_t &page_id) const {

  if (page_id.getAttributeNo() >= m_attribute_metadata.size() ||
      page_id.getPartitionNo() >= m_attribute_metadata[page_id.getAttributeNo()].size()) {
    std::cout << "ERROR: PageId_t references non-existent attribute or partition" << std::endl;
    return {};
      }

  const auto &partMetaData = m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];

  if (page_id.getBlockNo() >= partMetaData.num_blocks) {
    std::cout << "ERROR: PageId_t references non-existent block" << std::endl;
    return {};
  }

  if (partMetaData.data_format != AttributePartMetaData::DataFormat_t::COMPRESSED) {
    std::cout << "Page is not compressed, no val_section_sizes available" << std::endl;
    return {};
  }

  if (page_id.getBlockNo() >= partMetaData.val_section_sizes.size()) {
    // std::cout << "ERROR: Block index " << page_id.getBlockNo()
    //           << " exceeds val_section_sizes array size " << partMetaData.val_section_sizes.size() << std::endl;
    return {};
  }

  const auto& result = partMetaData.val_section_sizes[page_id.getBlockNo()];

  return result;
}

std::vector<uint32_t> NvmePlugin::getChunkAuxSizes(
    const PageId_t &page_id) const {
  return getValSectionSizes(page_id);
}

const std::vector<uint32_t> &NvmePlugin::getChunkSizesRef(
    const PageId_t &page_id) const {
  static const std::vector<uint32_t> kEmpty;
  if (page_id.getAttributeNo() >= m_attribute_metadata.size() ||
      page_id.getPartitionNo() >=
          m_attribute_metadata[page_id.getAttributeNo()].size()) {
    LOG(WARNING) << "PageId_t references non-existent attribute or partition";
    return kEmpty;
  }

  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  if (page_id.getBlockNo() >= partMetaData.num_blocks) {
    LOG(WARNING) << "PageId_t references non-existent block";
    return kEmpty;
  }

  if (partMetaData.data_format != AttributePartMetaData::DataFormat_t::COMPRESSED) {
    return kEmpty;
  }
  return partMetaData.chunk_sizes[page_id.getBlockNo()];
}

const std::vector<uint32_t> &NvmePlugin::getValSectionSizesRef(
    const PageId_t &page_id) const {
  static const std::vector<uint32_t> kEmpty;
  if (page_id.getAttributeNo() >= m_attribute_metadata.size() ||
      page_id.getPartitionNo() >=
          m_attribute_metadata[page_id.getAttributeNo()].size()) {
    LOG(WARNING) << "PageId_t references non-existent attribute or partition";
    return kEmpty;
  }

  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  if (page_id.getBlockNo() >= partMetaData.num_blocks) {
    LOG(WARNING) << "PageId_t references non-existent block";
    return kEmpty;
  }

  if (partMetaData.data_format != AttributePartMetaData::DataFormat_t::COMPRESSED ||
      page_id.getBlockNo() >= partMetaData.val_section_sizes.size()) {
    return kEmpty;
  }
  return partMetaData.val_section_sizes[page_id.getBlockNo()];
}

const std::vector<uint32_t> &NvmePlugin::getChunkAuxSizesRef(
    const PageId_t &page_id) const {
  return getValSectionSizesRef(page_id);
}

void *NvmePlugin::getPreloadedCompressedChunkAuxSizesOidMapPtr(
    uint8_t attribute_no) const {
  return getPreloadedCompressedValSectionSizesOidMapPtr(attribute_no);
}

size_t NvmePlugin::getBlockSizeBytes(const PageId_t &page_id) const {
  CHECK_LT(page_id.getAttributeNo(), m_attribute_metadata.size());
  CHECK_LT(page_id.getPartitionNo(),
           m_attribute_metadata[page_id.getAttributeNo()].size());
  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  CHECK_LT(page_id.getBlockNo(), partMetaData.block_sizes.size());
  return partMetaData.block_sizes[page_id.getBlockNo()];
}

int NvmePlugin::getDecompressedChunkSize(const PageId_t &page_id) const {
  CHECK_LT(page_id.getAttributeNo(), m_attribute_metadata.size());
  CHECK_LT(page_id.getPartitionNo(),
           m_attribute_metadata[page_id.getAttributeNo()].size());
  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  if (partMetaData.data_format != AttributePartMetaData::DataFormat_t::COMPRESSED) {
    return 0;
  }
  return partMetaData.decompressed_chunk_size;
}

bool NvmePlugin::isCompressed(const PageId_t &page_id) const {
  CHECK_LT(page_id.getAttributeNo(), m_attribute_metadata.size());
  CHECK_LT(page_id.getPartitionNo(),
           m_attribute_metadata[page_id.getAttributeNo()].size());
  const auto &partMetaData =
      m_attribute_metadata[page_id.getAttributeNo()][page_id.getPartitionNo()];
  return partMetaData.data_format == AttributePartMetaData::DataFormat_t::COMPRESSED;
}

NvmePlugin::AttributePartMetaData::~AttributePartMetaData() {
  if (fileIsOnNvmeDrive(data_file_path)) {
    cuFileHandleDeregister(cufile_handle);
  }
  close(fd);
}

uint64_t NvmePlugin::getRowGroupTupleCount(uint64_t partIdx,
                                           uint64_t blockIdx) {
#ifndef NDEBUG
  auto values_in_first_col =
      m_attribute_metadata[0][partIdx].value_counts[blockIdx];
  for (const auto &attr_meta : m_attribute_metadata) {
    CHECK_EQ(values_in_first_col, attr_meta[partIdx].value_counts[blockIdx]);
  }
#endif
  return m_attribute_metadata[0][partIdx].value_counts[blockIdx];
}

uint64_t getRowGroupTupleCount(uint64_t partIdx, uint64_t blockIdx,
                               NvmePlugin *pg) {
  return pg->getRowGroupTupleCount(partIdx, blockIdx);
}
