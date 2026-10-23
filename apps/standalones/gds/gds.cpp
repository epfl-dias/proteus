/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include <cufile_181/cufile.h>
#pragma clang diagnostic pop

#include <sys/mman.h>
#include <unistd.h>

#include <cli-flags.hpp>
#include <platform/common/error-handling.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/threadvector.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>

__host__ __device__ inline void gpuAssert(CUfileError_t code, const char *file,
                                          int line) {
  if (code.err != CU_FILE_SUCCESS) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
    if (IS_CUFILE_ERR(code.err)) {
#pragma clang diagnostic pop
#ifndef __CUDA_ARCH__
      gpuAssert(cufileop_status_error((CUfileOpError)(code.err)), file, line);
#else
      printf("GPUassert: %s %s %d\n", "error", file, line);
      exit(code);
#endif
    } else if (IS_CUDA_ERR(code)) {
      gpu_run(CU_FILE_CUDA_ERR(code));
    } else {
#ifndef __CUDA_ARCH__
      gpuAssert("Unknown cufile error type (errno?)", file, line);
#else
      printf("GPUassert: %s %s %d\n", "error", file, line);
      exit(code);
#endif
    }
  }
}

int main(int argc, char *argv[]) {
  // For NVMeOF:
  //   https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/managing_storage_devices/overview-of-nvme-over-fabric-devicesmanaging-storage-devices#setting-up-nvme-rdma-target-using-nvmetcli_nvme-over-fabrics-using-rdma
  auto ctx = proteus::from_cli::olap("GDS testbed", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getGpus()[1]);

  //  gpu_run(cuFileDriverOpen());
  //  {
  //    CUfileDrvProps_t props{};
  //    gpu_run(cuFileDriverGetProperties(&props));
  //    LOG(INFO) << "NVFS version: " << props.nvfs.major_version << "."
  //              << props.nvfs.minor_version;
  //  }

  int fd = linux_run(open("/scratch2/data/ssbm100/lineorder.csv.lo_tax",
                          O_DIRECT | O_LARGEFILE | O_RDONLY));

  {
    auto page_size = sysconf(_SC_PAGESIZE);
    auto s = size_t{2} * 1024 * 1024 * 1024;
    auto *ptr =
        static_cast<uint8_t *>(mmap(nullptr, s, PROT_READ, MAP_PRIVATE, fd, 0));
    assert(ptr != MAP_FAILED);
    auto page_cnt = (s + page_size - 1) / page_size;
    unsigned char inmem[page_cnt];
    linux_run(mincore(ptr, s, inmem));
    size_t cnt = 0;
    for (size_t i = 0; i < page_cnt; ++i) {
      if ((inmem[i] & 1) == 0) {
        ++cnt;
      }
    }
    LOG(ERROR) << "Not in-memory: " << cnt << "/" << page_cnt;
    size_t sum = 0;
    for (size_t i = 0; i < s; ++i) {
      sum += ptr[i];
    }
    LOG(INFO) << sum;
    linux_run(mincore(ptr, s, inmem));
    cnt = 0;
    for (size_t i = 0; i < page_cnt; ++i) {
      if ((inmem[i] & 1) == 0) {
        ++cnt;
      }
    }
    LOG(ERROR) << "Not in-memory: " << cnt << "/" << page_cnt;
  }

  //
  //
  //  {
  //    CUfileDescr_t cf_descr{};
  //    cf_descr.handle.fd = fd;
  //    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  //    CUfileHandle_t cf_handle;
  //
  //    gpu_run(cuFileHandleRegister(&cf_handle, &cf_descr));
  //
  //    auto buff_size = size_t{4} * 1024 * 1024 * 1024;
  //    auto devPtr_base = MemoryManager::mallocGpu(buff_size);
  //    LOG(INFO) << "Socket: "
  //              << topology::getInstance()
  //                     .getGpuAddressed(devPtr_base)
  //                     ->getLocalCPUNumaNode()
  //                     .id;
  //
  //    gpu_run(cuFileBufRegister(devPtr_base, buff_size, 0));
  //
  //    // fill a pattern
  //    gpu_run(cudaMemset((void *)devPtr_base, 0xab, buff_size));
  //
  ////    std::vector<
  //    for (size_t j = 0 ; j < 10 ; ++j){
  //      time_block t{"T: "};
  //      threadvector vs{};
  //      vs.emplace_back([&](){
  //        for (size_t i = 0; i < 10; ++i) {
  //          auto s = size_t{2} * 1024 * 1024 * 1024;
  //          // perform write operation directly from GPU mem to file
  //          auto ret = cuFileRead(cf_handle, devPtr_base, s, 0, 0);
  //          assert(ret == s);
  //        }
  //      });
  //      for (auto &v: vs) v.wait();
  //    }
  //
  //    // ftruncate for resizes
  //
  //    // release the GPU memory pinning
  //    gpu_run(cuFileBufDeregister(devPtr_base));
  //    MemoryManager::freeGpu(devPtr_base);
  //
  //    // deregister the handle from cuFile
  //    (void)cuFileHandleDeregister(cf_handle);
  //  }
  //
  linux_run(close(fd));
  //
  //  ((void)cuFileDriverClose());

  return 0;
}
