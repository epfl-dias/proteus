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

#ifndef PROTEUS_BLOCK_NVME_PLUGIN_HPP
#define PROTEUS_BLOCK_NVME_PLUGIN_HPP

#include <cufile_181/cufile.h>
#include <rapidjson/document.h>

#include <atomic>
#include <olap/plugins/binary-block-plugin.hpp>
#include <olap/values/types.hpp>
#include <platform/util/linux-exec.hpp>
#include <storage/mmap-file.hpp>
#include <memory>
#include <mutex>
#include <regex>
#include <variant>

class NvmePlugin;
void runNvmePreDecompressKernel(NvmePlugin *pg);
extern "C" void *getNvmePageIdPtr(NvmePlugin *pg, uint8_t attribute_no,
                                  uint8_t partition_no,
                                  uint32_t block_no) noexcept;
extern "C" void *getNvmePreloadedCompressedBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
extern "C" void *getNvmePreloadedDecompressedBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
extern "C" void *getNvmePreloadedScanBlockPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
extern "C" void *getNvmePreloadedCompressedChunkOffsetsPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
extern "C" void *getNvmePreloadedCompressedValSectionSizesPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
// Codec-agnostic alias for getNvmePreloadedCompressedValSectionSizesPtr.
extern "C" void *getNvmePreloadedCompressedChunkAuxSizesPtr(
    NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
    uint32_t block_no) noexcept;
extern "C" size_t getNvmePreloadedCompressedBlockSizeByOid(
    NvmePlugin *pg, uint8_t attribute_no, uint64_t oid) noexcept;

class NvmePlugin : public BinaryBlockPlugin {
 public:
  static constexpr auto type = "nvme-block";

  struct GpuMemoryFootprint {
    size_t compressed_dataset_bytes{0};
    size_t aux_bytes{0};

    [[nodiscard]] size_t total_bytes() const {
      return compressed_dataset_bytes + aux_bytes;
    }
  };

  struct PageId_t {
    static constexpr uint8_t cpu_numa_mask = 0b01111111;
    static constexpr uint8_t page_id_bit_in_numa = 0b10000000;
    static_assert((cpu_numa_mask | page_id_bit_in_numa) == 0b11111111,
                  "cpu_numa_mask and page_id_mask overlap");

    static constexpr uint64_t page_id_mask = 1ULL << 63;

    PageId_t(uint8_t cpu_numa_affinity, uint8_t attribute_no,
             uint8_t partition_no, uint32_t block_no) noexcept
        : cpu_numa_affinity(cpu_numa_affinity | page_id_bit_in_numa),
          attribute_no(attribute_no),
          partition_no(partition_no),
          _(0),
          block_no(block_no) {}

    PageId_t(const PageId_t &other) noexcept
        : cpu_numa_affinity(other.cpu_numa_affinity),
          attribute_no(other.attribute_no),
          partition_no(other.partition_no),
          _(0),
          block_no(other.block_no) {}

    static PageId_t from_ptr(void *ptr) noexcept {
      auto page_id = reinterpret_cast<uint64_t>(ptr);
      return PageId_t{static_cast<uint8_t>((page_id >> 56)),
                      static_cast<uint8_t>(page_id >> 48),
                      static_cast<uint8_t>(page_id >> 40),
                      static_cast<uint32_t>(page_id)};
    }
    static bool isPageIdPtr(void *ptr) noexcept {
      return reinterpret_cast<uint64_t>(ptr) & page_id_mask;
    }
    static bool isPageIdPtr(uintptr_t ptr) noexcept {
      return reinterpret_cast<uint64_t>(ptr) & page_id_mask;
    }

    [[nodiscard]] inline uint8_t getCpuNumaAffinity() const {
      return cpu_numa_affinity & cpu_numa_mask;
    }
    [[nodiscard]] inline uint8_t getAttributeNo() const { return attribute_no; }
    [[nodiscard]] inline uint8_t getPartitionNo() const { return partition_no; }
    [[nodiscard]] inline uint32_t getBlockNo() const { return block_no; }

   private:
    const uint8_t
        cpu_numa_affinity;  /// this is cpu_numa node in the lower bits and the
                            /// first bit indicates that this is whole struct is
                            /// a page id when stored as a ptr type
    const uint8_t attribute_no;
    const uint8_t partition_no;
    __attribute__((unused)) const uint8_t _;  /// padding
    const uint32_t block_no;

    friend std::ostream &operator<<(std::ostream &out, const PageId_t &page_id);
  };
  static_assert(sizeof(PageId_t) == sizeof(void *));

  struct AttributePartMetaData {
    enum DataFormat_t { COMPRESSED, UNCOMPRESSED };

    explicit AttributePartMetaData(const std::filesystem::path &md_path);
    ~AttributePartMetaData();
    uint64_t num_blocks;
    int fd;
    CUfileHandle_t cufile_handle;
    std::filesystem::path data_file_path;
    std::vector<off_t> block_offsets;    /// File offset in bytes of each block
    std::vector<uint64_t> block_sizes;   /// on disk size of each block in bytes
    std::vector<uint32_t> value_counts;  /// count of values in each block
    std::vector<std::vector<uint32_t>>
        chunk_sizes;  /// count of each compressed chunk in each block. Unset
                      /// for uncompressed Attribute parts
    // Legacy RFOR name kept for compatibility. Semantically this stores
    // per-chunk auxiliary section sizes (codec-specific meaning).
    std::vector<std::vector<uint32_t>> val_section_sizes;
    std::string compression_algorithm;
    int tile_size = 0;              /// optional: RFOR integers per tile
    int decompressed_chunk_size = 0;    /// the size of each chunk after
                                    /// decompression. The last chunk in a block
                                    /// may be smaller.
    bool pbmd_chunk_offsets = false;  /// true when metadata-in-block headers
                                      /// store chunk_offsets[] instead of
                                      /// chunk_sizes[] in the first array
    int max_compressed_block_size = 0;  /// the maximum compressed block size. Note
                                    /// calculated and not stored in metadata
    DataFormat_t data_format;
    int numa_node;
  };

  NvmePlugin(
      OlapParallelContext *context,
      const std::vector<std::pair<RecordAttribute *,
                                  std::vector<std::filesystem::path>>> &fields);
  NvmePlugin(
      OlapParallelContext *context,
      const std::vector<std::pair<RecordAttribute *,
                                  std::vector<std::filesystem::path>>> &fields,
      bool preload_to_gpu, bool emit_compressed_blocks);

  void generate(const Operator &producer,
                OlapParallelContext *context) override;

  ~NvmePlugin() override;

  struct PageIOInfo {
    const int fd;
    const CUfileHandle_t cufile_handle;
    const off_t *offset;  /// ptr because cuFileAsync expects a ptr and gives
                          /// no documentation on required lifetime. Non-owning
                          /// pointer. Life time is the same as the plugin
    const size_t *size;   /// same as above
    const std::vector<uint32_t> chunk_sizes;
    const bool is_compressed;
    const int decompressed_chunk_size;
  };
  /**
   * Retrieves the IO information for a given page.
   * @param page_id The PageId_t of the page for which to retrieve the IO
   * information.
   * @return A tuple containing the file descriptor, offset, and size in bytes
   * of the page.
   */
  [[nodiscard]] PageIOInfo getPageIoInfo(const PageId_t &page_id) const;

  /**
   * Legacy accessor name: get per-chunk auxiliary section sizes from JSON
   * metadata. For RFOR this corresponds to val_section_sizes.
   */
  [[nodiscard]] std::vector<uint32_t> getValSectionSizes(const PageId_t &page_id) const;
  // Codec-agnostic aliases for the same metadata.
  [[nodiscard]] std::vector<uint32_t> getChunkAuxSizes(
      const PageId_t &page_id) const;
  [[nodiscard]] const std::vector<uint32_t> &getChunkSizesRef(
      const PageId_t &page_id) const;
  [[nodiscard]] const std::vector<uint32_t> &getValSectionSizesRef(
      const PageId_t &page_id) const;
  [[nodiscard]] const std::vector<uint32_t> &getChunkAuxSizesRef(
      const PageId_t &page_id) const;
  [[nodiscard]] size_t getBlockSizeBytes(const PageId_t &page_id) const;
  [[nodiscard]] int getDecompressedChunkSize(const PageId_t &page_id) const;
  [[nodiscard]] bool isCompressed(const PageId_t &page_id) const;

  // In preload mode scan emits plugin-owned GPU pointers (compressed or
  // decompressed), not BlockManager buffers. Unpack operators must not recycle
  // them via release_buffers().
  [[nodiscard]] bool unpackShouldReleaseInputBuffers() const {
    return !preload_to_gpu_;
  }

  // Ensure preloaded decompressed blocks exist (used with emit_compressed_blocks_).
  void ensurePreloadedDecompressedBlocks();
  // Mark plugin state so the next scan access under --pre_decom must
  // (re)decompress before exposing scan pointers.
  void schedulePreDecompressOnNextScan();
  [[nodiscard]] bool scanEmitsCompressedBlocks() const {
    return preload_to_gpu_ &&
           emit_compressed_blocks_.load(std::memory_order_acquire);
  }
  void setScanEmitCompressedBlocks(bool emit);
  [[nodiscard]] bool emitsCompressedBlocksForScan() const {
    return emit_compressed_blocks_.load(std::memory_order_acquire);
  }
  [[nodiscard]] void *getPreloadedScanBlockPtr(
      uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const;
  [[nodiscard]] size_t getPartitionBlockCount(size_t partition_no) const;
  [[nodiscard]] void *getPreloadedDecompressedPartBasePtr(
      size_t attribute_no, size_t partition_no) const;
  [[nodiscard]] void *getPreloadedCompressedChunkOffsetsOidMapPtr(
      uint8_t attribute_no) const;
  [[nodiscard]] void *getPreloadedCompressedValSectionSizesOidMapPtr(
      uint8_t attribute_no) const;
  [[nodiscard]] void *getPreloadedCompressedChunkAuxSizesOidMapPtr(
      uint8_t attribute_no) const;
  [[nodiscard]] void *getPreloadedCompressedBlockSizesOidMapPtr(
      uint8_t attribute_no) const;

  [[nodiscard]] GpuMemoryFootprint getGpuMemoryFootprint() const {
    return {
        gpu_preloaded_compressed_bytes_.load(std::memory_order_acquire),
        gpu_preloaded_aux_bytes_.load(std::memory_order_acquire)};
  }

 protected:
  llvm::Value *getDataPointersForFile(OlapParallelContext *context, size_t i,
                                      llvm::Value *session_ptr) const override {
    LOG(FATAL) << "not implemented for NvmePlugin";
  }

  void freeDataPointersForFile(OlapParallelContext *context, size_t i,
                               llvm::Value *v) const override {
    LOG(FATAL) << "not implemented for NvmePlugin";
  }

  /**
   * @return a ptr to an array of length Nparts storing the number of blocks in
   * each partition, and the size in blocks of the largest partition
   */
  std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      OlapParallelContext *context, llvm::Value *session_ptr) const override;

  void freePartitionSizes(OlapParallelContext *context,
                          llvm::Value *v) const override;

 private:
  static constexpr auto blockCtrVar = "blockCtr";
  uint64_t getRowGroupTupleCount(uint64_t partIdx, uint64_t blockIdx);

  // Generates a for loop that emits a PageId_t for each block in each iteration
  void scan(const Operator &producer, OlapParallelContext *context);

  std::vector<std::vector<AttributePartMetaData>> m_attribute_metadata;
  std::vector<uint64_t> part_sizes;  /// in blocks
  bool preload_to_gpu_{false};
  std::atomic<bool> emit_compressed_blocks_{false};
  std::atomic<bool> preloaded_decompressed_ready_{false};
  std::atomic<uint64_t> predecompress_epoch_{0};
  std::atomic<uint64_t> decompressed_epoch_{0};
  std::atomic<size_t> gpu_preloaded_compressed_bytes_{0};
  std::atomic<size_t> gpu_preloaded_decompressed_bytes_{0};
  std::atomic<size_t> gpu_preloaded_aux_bytes_{0};
  mutable std::mutex predecompress_mu_;

  // [attribute][partition]
  std::vector<std::vector<std::unique_ptr<mmap_file>>>
      preloaded_compressed_part_files_;
  std::vector<std::vector<std::unique_ptr<mmap_file>>>
      preloaded_decompressed_part_files_;
  std::vector<std::vector<std::unique_ptr<mmap_file>>>
      preloaded_compressed_chunk_offsets_part_files_;
  std::vector<std::vector<std::unique_ptr<mmap_file>>>
      preloaded_compressed_val_section_sizes_part_files_;
  std::vector<std::unique_ptr<mmap_file>>
      preloaded_compressed_chunk_offsets_oid_maps_;
  std::vector<std::unique_ptr<mmap_file>>
      preloaded_compressed_val_section_sizes_oid_maps_;
  std::vector<std::unique_ptr<mmap_file>>
      preloaded_compressed_block_sizes_oid_maps_;

  // [attribute][partition][block]
  std::vector<std::vector<std::vector<void *>>>
      preloaded_compressed_block_ptrs_;
  std::vector<std::vector<std::vector<void *>>>
      preloaded_decompressed_block_ptrs_;
  std::vector<std::vector<std::vector<void *>>>
      preloaded_compressed_chunk_offsets_ptrs_;
  std::vector<std::vector<std::vector<void *>>>
      preloaded_compressed_val_section_sizes_ptrs_;

  void preloadBlocksToGpu();
  [[nodiscard]] void *getPreloadedCompressedBlockPtr(
      uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const;
  [[nodiscard]] void *getPreloadedDecompressedBlockPtr(
      uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const;
  [[nodiscard]] void *getPreloadedCompressedChunkOffsetsPtr(
      uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const;
  [[nodiscard]] void *getPreloadedCompressedValSectionSizesPtr(
      uint8_t attribute_no, uint8_t partition_no, uint32_t block_no) const;
  void nextEntry(OlapParallelContext *context);

  static std::string fileToDevPath(const std::filesystem::path &path) {
    CHECK(std::filesystem::exists(path));

    std::string command = "df " + path.string() + " --output=source";
    auto [dfOutput, returnCode] = execCommand(command);
    CHECK_EQ(returnCode, 0)
        << "Failed to df file: " << path.string() << " output: " << dfOutput
        << "  Are you sure it exists?";

    // df output looks like this:
    //  nicholso@diascld36:~$ df /scratch --output=source
    //  Filesystem
    //  /dev/sdb1
    // we care about the second line, which gets us the device this file is on
    auto ssDfOutput = std::stringstream{dfOutput};
    auto splitDfOutput = std::vector<std::string>{};
    for (std::string line; std::getline(ssDfOutput, line, '\n');)
      splitDfOutput.push_back(line);
    assert(splitDfOutput.size() == 2);

    std::string partition = splitDfOutput.at(1);
    // partitions is something like "/dev/nvme0n1p1"
    // we want to drop this final p1, as we don't care about partitions, we need
    // the dev path with namespace
    std::regex nvme_regex("/dev/nvme[0-9]+n[0-9]+p[0-9]");
    std::regex sata_regex("/dev/sda[0-9]");
    std::smatch device_path;
    std::string dev_path_str;
    if (std::regex_search(partition, device_path, nvme_regex)) {
      // if there is a match, there should only be 1
      CHECK_EQ(device_path.size(), 1) << "Device regex seriously broken";

      std::string dev_path_str_with_p = device_path[0];
      //    TODO this breaks portability, lopping off the final p1, see
      //    topology.hpp for details
      dev_path_str =
          dev_path_str_with_p.substr(0, dev_path_str_with_p.size() - 2);
    } else if (std::regex_search(partition, device_path, sata_regex)) {
      LOG(WARNING) << "File is not on a NVMe drive, assuming NUMA 0 affinity: "
                   << path;
      CHECK_EQ(device_path.size(), 1) << "Device regex seriously broken";
      dev_path_str = device_path[0];
    } else {
      LOG(FATAL) << "Failed to match device path in df output: " << partition;
    }
    return dev_path_str;
  }

  static bool fileIsOnNvmeDrive(const std::filesystem::path &path) {
    const auto dev_path_str = fileToDevPath(path);
    if (dev_path_str.find("nvme") != std::string::npos) {
      return true;
    } else {
      return false;
    }
  }

  static int fileNameToNumaNodeIndex(const std::filesystem::path &path) {
    const auto &topo = topology::getInstance();
    const std::string dev_path_str = fileToDevPath(path);
    if (dev_path_str.find("nvme") == std::string::npos) {
      LOG(WARNING) << "File is not on a NVMe device, assuming NUMA 0 affinity: "
                   << path;
      return 0;
    }
    std::reference_wrapper<const topology::numanode> drive =
        topo.devPathToNvme(dev_path_str);

    return topo.devPathToNvme(dev_path_str).getLocalCPUNumaNode().index_in_topo;
  }

  friend uint64_t getRowGroupTupleCount(uint64_t partIdx, uint64_t blockIdx,
                                        NvmePlugin *pg);

  friend void *getNvmePageIdPtr(NvmePlugin *pg, uint8_t attribute_no,
                                uint8_t partition_no,
                                uint32_t block_no) noexcept;
  friend void *getNvmePreloadedCompressedBlockPtr(
      NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
      uint32_t block_no) noexcept;
  friend size_t getNvmePreloadedCompressedBlockSizeByOid(
      NvmePlugin *pg, uint8_t attribute_no, uint64_t oid) noexcept;
  friend void *getNvmePreloadedDecompressedBlockPtr(
      NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
      uint32_t block_no) noexcept;
  friend void *getNvmePreloadedScanBlockPtr(
      NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
      uint32_t block_no) noexcept;
  friend void *getNvmePreloadedCompressedChunkOffsetsPtr(
      NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
      uint32_t block_no) noexcept;
  friend void *getNvmePreloadedCompressedValSectionSizesPtr(
      NvmePlugin *pg, uint8_t attribute_no, uint8_t partition_no,
      uint32_t block_no) noexcept;
  friend void runNvmePreDecompressKernel(NvmePlugin *pg);
};

uint64_t getRowGroupTupleCount(uint64_t partIdx, uint64_t blockIdx,
                               NvmePlugin *pg);

std::ostream &operator<<(std::ostream &out,
                         const NvmePlugin::PageId_t &page_id);

#endif /* PROTEUS_BLOCK_NVME_PLUGIN_HPP */
