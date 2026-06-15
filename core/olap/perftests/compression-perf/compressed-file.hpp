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

#ifndef PROTEUS_COMPRESSEDFILE_HPP
#define PROTEUS_COMPRESSEDFILE_HPP

#include <filesystem>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/util/glog.hpp>
#include <span>
#include <storage/mmap-file.hpp>

/**
 * Utility class to load a file based on its metadata.json file and provide
 * an interface to get each compressed block of the file
 */
class CompressedFile {
 public:
  /**
   * non-owning of the data for a block
   * Lifetime of the CompressedBlock is encapsulated by its CompressedFile
   */
  struct CompressedBlock {
    CompressedBlock(const std::span<const char>& block,
                    const std::vector<uint32_t>& chunk_sizes,
                    int decompressed_chunk_size)
        : block(block),
          chunk_sizes(chunk_sizes),
          decompressed_chunk_size(decompressed_chunk_size) {}
    const std::span<const char> block;
    const std::vector<uint32_t>
        chunk_sizes;  /// size of the compressed chunks in this block
    const int decompressed_chunk_size;  /// size of the decompressed chunks,
                                        /// e.g. 16KiB
  };

  /**
   * Load a compressed file based on its metadata.json file
   * @param md_path path to the metadata.json file
   * @param loc location to load the file. e.g use data_loca::PINNED to load to
   * the currently affinitized CPU node
   */
  explicit CompressedFile(const std::string& md_path, data_loc loc)
      : m_file_md(md_path),
        m_data(m_file_md.data_file_path, loc),
        m_data_span(static_cast<const char*>(m_data.getData()),
                    m_data.getFileSize()) {
    CHECK_EQ(m_file_md.data_format,
             NvmePlugin::AttributePartMetaData::COMPRESSED)
        << "Metadata does not describe a compressed file";
    LOG(INFO) << "File: " << md_path << " has " << m_file_md.num_blocks
              << " blocks";
  }
  [[nodiscard]] std::vector<CompressedBlock> getBlocks() const {
    std::vector<CompressedBlock> blocks;
    blocks.reserve(m_file_md.num_blocks);
    for (size_t i = 0; i < m_file_md.num_blocks; i++) {
      blocks.emplace_back(m_data_span.subspan(m_file_md.block_offsets[i],
                                              m_file_md.block_sizes[i]),
                          m_file_md.chunk_sizes[i],
                          m_file_md.decompressed_chunk_size);
    }
    return blocks;
  }

  NvmePlugin::AttributePartMetaData m_file_md;
  mmap_file m_data;
  std::span<const char> m_data_span;
};

#endif  // PROTEUS_COMPRESSEDFILE_HPP
