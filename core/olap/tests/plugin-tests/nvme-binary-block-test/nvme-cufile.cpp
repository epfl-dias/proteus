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

#include <cuda_runtime.h>
// Nvidia wrote a bad doxygen string, but that is not our problem
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include <cufile.h>
#pragma clang diagnostic pop
#include <gtest/gtest.h>

#include <filesystem>
#include <platform/memory/allocator.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/util/glog.hpp>

#include "olap/plugins/binary-block-nvme-plugin.hpp"
#include "utils.hpp"

#if defined(CU_FILE_STREAM_PAGE_ALIGNED_INPUTS) && \
    defined(CU_FILE_STREAM_FIXED_BUF_OFFSET) &&     \
    defined(CU_FILE_STREAM_FIXED_FILE_OFFSET) &&    \
    defined(CU_FILE_STREAM_FIXED_FILE_SIZE)
#define PROTEUS_HAS_CUFILE_STREAM_API 1
#else
#define PROTEUS_HAS_CUFILE_STREAM_API 0
#endif

#if !PROTEUS_HAS_CUFILE_STREAM_API
TEST(NvmeCuFile, streamAsyncApiUnavailable) {
  GTEST_SKIP() << "cuFile stream async API is not available in this GDS header";
}
#else
class StreamReadFileFixture
    : public testing::TestWithParam<std::pair<std::filesystem::path, size_t>> {
 protected:
  const NvmePlugin::AttributePartMetaData m_file_md;
  cudaStream_t m_strm;
  const size_t m_sync_every;

 public:
  StreamReadFileFixture()
      : m_file_md{GetParam().first},
        m_strm{},
        m_sync_every{GetParam().second} {}
  void SetUp() override {
    auto& topo = topology::getInstance();
    if (topo.getGpus().size() < 1) {
      LOG(WARNING) << "No GPUS, skipping test;";
      GTEST_SKIP();
    }

    set_exec_location_on_scope d{topo.getGpus().at(0)};
    std::this_thread::yield();
    CUfileError_t status;

    m_strm = createNonBlockingStream();
    // Tell CuFile that the input buffer is page aligned, and the offsets and
    // file size are know at submission time (now) and do not depend on pending
    // operations in the stream
    status = cuFileStreamRegister(m_strm, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS |
                                              CU_FILE_STREAM_FIXED_BUF_OFFSET |
                                              CU_FILE_STREAM_FIXED_FILE_OFFSET |
                                              CU_FILE_STREAM_FIXED_FILE_SIZE);
    ASSERT_EQ(status.err, CU_FILE_SUCCESS)
        << "failed to cuFileStreamRegister status: "
        << cufileop_status_error(status.err);
  }
  void TearDown() override {
    CUfileError_t status;
    status = cuFileStreamDeregister(m_strm);
    ASSERT_EQ(status.err, CU_FILE_SUCCESS)
        << "failed to cuFileStreamDeregister status: "
        << cufileop_status_error(status.err);
    syncAndDestroyStream(m_strm);
  }
};

/**
 * @brief Test loading an entire file to the GPU asynchronously using cuFile
 * File load is validated by copying the data to CPU memory and comparing it to
 * data loaded by the CPU. The data is copied back after all IO is complete (by
 * synchronizing the stream after IO and before the memcpy)
 * @note this test requires at least as much memory allocated to 2MiB buffers on
 * the GPU as the on-disk file size or it will deadlock waiting for buffers
 * @note this test will synchronize the stream after every m_sync_every blocks
 */
TEST_P(StreamReadFileFixture, streamReadFileStreamSynchronize) {
  std::vector<proteus::managed_ptr> gpu_buffers;
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    gpu_buffers.emplace_back(BlockManager::h_get_buffer(0));
  }

  /// the bytes_read_p parameter need to point to memory accessible by the GPU
  /// @see
  /// https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilereadasync
  std::vector<ssize_t, proteus::memory::PinnedMemoryAllocator<ssize_t>>
      bytes_read_vec;
  bytes_read_vec.resize(m_file_md.num_blocks);

  CUfileError_t status;
  off_t buff_offset = 0;
  EXPECT_NE(m_file_md.cufile_handle, nullptr);
  for (int i = 0; i < m_file_md.num_blocks; i++) {
    size_t block_size = static_cast<size_t>(m_file_md.block_sizes[i]);
    status = cuFileReadAsync(m_file_md.cufile_handle, gpu_buffers[i].get(),
                             (size_t*)(&block_size),
                             (off_t*)(&m_file_md.block_offsets[i]),
                             &buff_offset, &bytes_read_vec[i], m_strm);
    ASSERT_EQ(status.err, CU_FILE_SUCCESS)
        << "Failed to cuFileReadAsync " << cufileop_status_error(status.err)
        << " for block " << i << " reading: " << m_file_md.block_sizes[i];
    if (i % m_sync_every == 0) {
      gpu_run(cuStreamSynchronize(m_strm));
    }
  }

  // all IO should now be complete
  gpu_run(cuStreamSynchronize(m_strm));

  for (int i = 0; i < m_file_md.num_blocks; i++) {
    ASSERT_EQ(bytes_read_vec[i], m_file_md.block_sizes[i])
        << "cuFileRead failed to read the expected number of bytes for block "
        << i;
  }

  std::vector<proteus::managed_ptr> cpu_buffers;
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    cpu_buffers.emplace_back(BlockManager::get_buffer());
    BlockManager::overwrite_bytes(cpu_buffers[i].get(), gpu_buffers[i].get(),
                                  m_file_md.block_sizes[i], m_strm, false);
  }
  gpu_run(cuStreamSynchronize(m_strm));
  validateLoadedBlocks(cpu_buffers, m_file_md.data_file_path);
  for (auto& gpu_buff : gpu_buffers) {
    BlockManager::release_buffer(gpu_buff.release());
  }
  for (auto& cpu_buff : cpu_buffers) {
    BlockManager::release_buffer(cpu_buff.release());
  }
}

/**
 * @brief Test loading an entire file to the GPU asynchronously using cuFile
 * File load is validated by copying the data to CPU memory and comparing it to
 * data loaded by the CPU. This variation interleaves IO and the GPU-to-CPU
 * memcpy on the same stream.
 * @note this test requires at least as much memory allocated to 2MiB buffers on
 * the GPU as the on-disk file size or it will deadlock waiting for buffers
 * @note this test will synchronize the stream after every m_sync_every blocks
 */
TEST_P(StreamReadFileFixture, streamReadFileStreamSynchronizeInterleaveMemcpy) {
  std::vector<proteus::managed_ptr> gpu_buffers;
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    gpu_buffers.emplace_back(BlockManager::h_get_buffer(0));
  }

  std::vector<ssize_t, proteus::memory::PinnedMemoryAllocator<ssize_t>>
      bytes_read_vec;
  bytes_read_vec.resize(m_file_md.num_blocks);

  std::vector<proteus::managed_ptr> cpu_buffers;  // for validation
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    cpu_buffers.emplace_back(BlockManager::get_buffer());
  }

  CUfileError_t status;
  off_t buff_offset = 0;
  EXPECT_NE(m_file_md.cufile_handle, nullptr);
  for (int i = 0; i < m_file_md.num_blocks; i++) {
    size_t block_size = static_cast<size_t>(m_file_md.block_sizes[i]);
    status = cuFileReadAsync(m_file_md.cufile_handle, gpu_buffers[i].get(),
                             (size_t*)(&block_size),
                             (off_t*)(&m_file_md.block_offsets[i]),
                             &buff_offset, &bytes_read_vec[i], m_strm);
    ASSERT_EQ(status.err, CU_FILE_SUCCESS)
        << "Failed to cuFileReadAsync " << cufileop_status_error(status.err)
        << " for block " << i << " reading: " << m_file_md.block_sizes[i];
    BlockManager::overwrite_bytes(cpu_buffers[i].get(), gpu_buffers[i].get(),
                                  m_file_md.block_sizes[i], m_strm, false);
    if (i % m_sync_every == 0) {
      gpu_run(cuStreamSynchronize(m_strm));
    }
  }

  // all IO should now be complete
  gpu_run(cuStreamSynchronize(m_strm));

  for (int i = 0; i < m_file_md.num_blocks; i++) {
    ASSERT_EQ(bytes_read_vec[i], m_file_md.block_sizes[i])
        << "cuFileRead failed to read the expected number of bytes for block "
        << i;
  }

  validateLoadedBlocks(cpu_buffers, m_file_md.data_file_path);
  for (auto& gpu_buff : gpu_buffers) {
    BlockManager::release_buffer(gpu_buff.release());
  }
  for (auto& cpu_buff : cpu_buffers) {
    BlockManager::release_buffer(cpu_buff.release());
  }
}

/**
 * @brief Test loading an entire file to the GPU asynchronously using cuFile.
 * File load is validated by copying the data to CPU memory and comparing it to
 * data loaded by the CPU. The data copy to the CPU for each block is on a
 * distinct stream from IO. every m_sync_every IOs, we eventSynchronize the
 * stream to ensure all IOs up to that point are complete. Then we copy the the
 * data read via cuFile to the GPU since the last synchronization to the CPU in
 * reverse order of completion to validate that the previous IOs are actually
 * complete when the event is Synchronized.
 * @note this test requires at least as much memory allocated to 2MiB buffers on
 * the GPU as the on-disk file size or it will deadlock waiting for buffers
 * @note this test will use events for synchronization after every m_sync_every
 * blocks. It is ambiguous if cuFile works with cudaEventSynchronize
 */
TEST_P(StreamReadFileFixture, streamReadFileEventSynchronize) {
  std::vector<proteus::managed_ptr> gpu_buffers;
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    gpu_buffers.emplace_back(BlockManager::h_get_buffer(0));
  }
  std::vector<ssize_t, proteus::memory::PinnedMemoryAllocator<ssize_t>>
      bytes_read_vec;
  bytes_read_vec.resize(m_file_md.num_blocks);

  // allocate for the case where we need to synchronize after each event, if
  // m_sync_every is > 1 then we won't synchronize on all of these events
  std::vector<cudaEvent_t> events{m_file_md.num_blocks};
  for (size_t i = 0; i < m_file_md.num_blocks; ++i) {
  }
  cudaEvent_t event = nullptr;
  gpu_run(cudaEventCreateWithFlags(
      &event, cudaEventDisableTiming | cudaEventBlockingSync));

  std::vector<proteus::managed_ptr> cpu_buffers;  // for validation
  for (size_t i = 0; i < m_file_md.num_blocks; i++) {
    cpu_buffers.emplace_back(BlockManager::get_buffer());
  }

  cudaStream_t memcpy_strm = createNonBlockingStream();

  CUfileError_t status;
  off_t buff_offset = 0;
  EXPECT_NE(m_file_md.cufile_handle, nullptr);
  int last_event_sync = -1;
  for (int i = 0; i < m_file_md.num_blocks; i++) {
    size_t block_size = static_cast<size_t>(m_file_md.block_sizes[i]);
    status = cuFileReadAsync(m_file_md.cufile_handle, gpu_buffers[i].get(),
                             (size_t*)(&block_size),
                             (off_t*)(&m_file_md.block_offsets[i]),
                             &buff_offset, &bytes_read_vec[i], m_strm);
    ASSERT_EQ(status.err, CU_FILE_SUCCESS)
        << "Failed to cuFileReadAsync " << cufileop_status_error(status.err)
        << " for block " << i << " reading: " << m_file_md.block_sizes[i];
    gpu_run(cudaEventRecord(event, m_strm));
    if (i % m_sync_every == 0) {
      // wait for all IO up until this point
      gpu_run(cudaEventSynchronize(event));
      // copy to the CPU in reverse order of completion to validate that
      // cudaEventSynchronize works with cuFile. If it does not, then the IO for
      // the most recently "loaded" GPU blocks may not actually be complete.
      // use a different stream to copy to the CPU to avoid blocking the IO and
      // to increase chances of synchronization bugs within cuFile
      for (int j = i; j > last_event_sync; j--) {
        ASSERT_EQ(bytes_read_vec[j], m_file_md.block_sizes[j])
            << "cuFileRead failed to read the expected number of bytes after "
               "cudaEventSynchronize for block "
            << j;
        BlockManager::overwrite_bytes(
            cpu_buffers[j].get(), gpu_buffers[j].get(),
            m_file_md.block_sizes[j], memcpy_strm, false);
      }
      last_event_sync = i;
    }
  }
  gpu_run(cuStreamSynchronize(m_strm));
  for (int i = 0; i < m_file_md.num_blocks; i++) {
    ASSERT_EQ(bytes_read_vec[i], m_file_md.block_sizes[i])
        << "cuFileRead failed to read the expected number of bytes for block "
        << i;
  }

  syncAndDestroyStream(memcpy_strm);

  validateLoadedBlocks(cpu_buffers, m_file_md.data_file_path);
  for (auto& gpu_buff : gpu_buffers) {
    BlockManager::release_buffer(gpu_buff.release());
  }
  for (auto& cpu_buff : cpu_buffers) {
    BlockManager::release_buffer(cpu_buff.release());
  }
}

// TODO: hardcoded to paths on a specific server for data generated with
// adm-partition-ssb
INSTANTIATE_TEST_SUITE_P(
    sf1000_lo_linenumber_0_10, StreamReadFileFixture,
    testing::Values(
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       1),
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       2),
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       4),
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       8),
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       16),
        std::make_pair("/nvme0/nicholso/data/sbm1000_0_10/"
                       "lineorder.csv.lo_linenumber_0_10.metadata.json",
                       32)));
#endif
