/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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
#include <platform/memory/block-manager.hpp>
#include <platform/util/glog.hpp>
#include <storage/test/test-utils.hpp>

proteus::managed_ptr loadFirstBlockofFile(const std::string& file_name) {
  auto block = BlockManager::get_buffer();
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "Failed to open file for reading: " << file_name.c_str();
    LOG(ERROR) << "Errno: " << errno;
    EXPECT_NE(fd, -1) << "failed to open file for reading: "
                      << file_name.c_str() << " " << strerror(errno);
  }
  auto bytes_read = read(fd, block.get(), 2_M);
  EXPECT_EQ(bytes_read, 2_M)
      << "Failed to read 2MiB from file: " << file_name.c_str() << " "
      << strerror(errno);
  close(fd);

  return block;
}

TEST(cuFile, syncReadSingleBlock) {
  LOG(INFO) << "running cuFile smoke test";
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() < 1) {
    LOG(WARNING) << "No GPUS, skipping test;";
    GTEST_SKIP();
  }
  set_exec_location_on_scope d{topo.getGpus().at(0)};

  const size_t read_size = 2_M;
  CUfileError_t status;

  const std::string file_name = "inputs/ssbm100/customer.csv.c_custkey";
  EXPECT_TRUE(std::filesystem::exists(file_name))
      << "File not found: " << file_name;

  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  ASSERT_NE(fd, -1) << "Failed to open file for reading: " << file_name.c_str()
                    << " " << strerror(errno);

  // register a file handle
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
      << "cuFileHandleRegister fd " << fd << " status "
      << cufileop_status_error(status.err);

  // allocate memory on the gpu to load the 2MiB to
  auto device_target_io_mem = BlockManager::h_get_buffer(0);

  // cuFileBufRegister should be called for device buffers targeted by cufile
  //    but this is a performance consideration not correctness
  //    status = cuFileBufRegister(device_target_io_mem, read_size, 0);
  //  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
  //      << "cuFileHandleRegister fd " << fd << " status "
  //      << cufileop_status_error(status.err);

  // finally, do our read
  // last two arguments are file offset, and buffer offset, both 0 here
  // because we just want to read the first 2MiB of the file
  const ssize_t bytes_read =
      cuFileRead(cf_handle, device_target_io_mem.get(), read_size, 0, 0);
  std::string error_message = "Failed to cuFileRead: ";
  if (bytes_read == -1) {
    error_message += std::string(strerror(errno));
  }
  if (bytes_read < -1) {
    error_message +=
        std::string(cufileop_status_error(CUfileOpError(-bytes_read)));
  }

  ASSERT_EQ(bytes_read, read_size)
      << "cuFileRead failed with: " << error_message;

  // if we pinned, release the GPU memory pinning
  //    LOG(INFO) << "Unpinning cuFile buffer." << std::endl;
  //    status = cuFileBufDeregister(device_target_io_mem);
  //  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
  //      << "cuFileHandleRegister fd " << fd << " status "
  //      << cufileop_status_error(status.err);

  auto copied_to_cpu_buffer = BlockManager::get_buffer();
  gpu_run(cudaMemcpy(copied_to_cpu_buffer.get(), device_target_io_mem.get(),
                     read_size, cudaMemcpyKind::cudaMemcpyDefault));

  auto cpu_loaded = loadFirstBlockofFile(file_name);
  EXPECT_EQ(memcmp(copied_to_cpu_buffer.get(), cpu_loaded.get(), read_size), 0)
      << "CPU loaded file and GPU loaded file bytes differ";

  BlockManager::release_buffer(std::move(device_target_io_mem));
  BlockManager::release_buffer(std::move(cpu_loaded));
  BlockManager::release_buffer(std::move(copied_to_cpu_buffer));
  close(fd);
  cuFileHandleDeregister(cf_handle);
}

TEST(cuFile, streamReadSingleBlock) {
#if !defined(CU_FILE_STREAM_PAGE_ALIGNED_INPUTS) || \
    !defined(CU_FILE_STREAM_FIXED_BUF_OFFSET) ||   \
    !defined(CU_FILE_STREAM_FIXED_FILE_OFFSET) ||  \
    !defined(CU_FILE_STREAM_FIXED_FILE_SIZE)
  GTEST_SKIP() << "cuFile stream async API is not available in this GDS header";
#else
  auto& topo = topology::getInstance();
  if (topo.getGpus().size() < 1) {
    LOG(WARNING) << "No GPUS, skipping test;";
    GTEST_SKIP();
  }

  set_exec_location_on_scope d{topo.getGpus().at(0)};
  std::this_thread::yield();

  size_t read_size = 2_M;
  CUfileError_t status;

  const std::string file_name = "inputs/ssbm100/customer.csv.c_custkey";
  EXPECT_TRUE(std::filesystem::exists(file_name))
      << "File not found: " << file_name;

  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  ASSERT_NE(fd, -1) << "Failed to open file for reading: " << file_name.c_str()
                    << " " << strerror(errno);

  // register a file handle
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
      << "cuFileHandleRegister fd " << fd << " status "
      << cufileop_status_error(status.err);

  // allocate memory on the gpu to load the 2MiB to
  auto device_target_io_mem = BlockManager::h_get_buffer(0);
  ssize_t bytes_read = 0;
  auto strm = createNonBlockingStream();
  // Tell CuFile that the input buffer is page aligned, and the offsets and file
  // size are know at submission time (now) and do not depend on pending
  // operations in the stream
  status = cuFileStreamRegister(strm, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS |
                                          CU_FILE_STREAM_FIXED_BUF_OFFSET |
                                          CU_FILE_STREAM_FIXED_FILE_OFFSET |
                                          CU_FILE_STREAM_FIXED_FILE_SIZE);
  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
      << "failed to cuFileStreamRegister status: "
      << cufileop_status_error(status.err);
  off_t offset = 0;
  status = cuFileReadAsync(cf_handle, device_target_io_mem.get(), &read_size,
                           &offset, &offset, &bytes_read, strm);
  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
      << "Failed to cuFileReadAsync" << cufileop_status_error(status.err);

  auto copied_to_cpu_buffer = BlockManager::get_buffer();
  gpu_run(cudaMemcpyAsync(copied_to_cpu_buffer.get(),
                          device_target_io_mem.get(), read_size,
                          cudaMemcpyKind::cudaMemcpyDefault, strm));

  gpu_run(cuStreamSynchronize(strm));
  status = cuFileStreamDeregister(strm);
  ASSERT_EQ(status.err, CU_FILE_SUCCESS)
      << "failed to cuFileStreamDeregister status: "
      << cufileop_status_error(status.err);
  syncAndDestroyStream(strm);

  std::string error_message = "Failed to cuFileRead: ";
  if (bytes_read == -1) {
    error_message += std::string(strerror(errno));
  }
  if (bytes_read < -1) {
    error_message +=
        std::string(cufileop_status_error(CUfileOpError(-bytes_read)));
  }

  ASSERT_EQ(bytes_read, read_size)
      << "cuFileRead failed with: " << error_message;

  auto cpu_loaded = loadFirstBlockofFile(file_name);
  ASSERT_EQ(memcmp(copied_to_cpu_buffer.get(), cpu_loaded.get(), read_size), 0)
      << "CPU loaded file and GPU loaded file bytes differ";

  BlockManager::release_buffer(std::move(copied_to_cpu_buffer));
  BlockManager::release_buffer(std::move(device_target_io_mem));
  BlockManager::release_buffer(std::move(cpu_loaded));
  close(fd);
  cuFileHandleDeregister(cf_handle);
#endif
}
