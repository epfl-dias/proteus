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

#ifndef PROTEUS_NVME_PLUGIN_TEST_UTILS_HPP
#define PROTEUS_NVME_PLUGIN_TEST_UTILS_HPP

#include <filesystem>
#include <platform/memory/block-manager.hpp>
/**
 * Check that the contents of blocks are the same as the contents of the input
 * file. The last block may be smaller than BlockManager::block_size. Only the
 * first input_file.size() bytes are compared.
 * @param blocks proteus::managed_ptr to CPU accessible buffers assumed to be
 * BlockManager::block_size in size
 * @param input_file path to the file to compare against
 */
void validateLoadedBlocks(const std::vector<proteus::managed_ptr>& blocks,
                          const std::filesystem::path& input_file);

#endif  // PROTEUS_NVME_PLUGIN_TEST_UTILS_HPP
