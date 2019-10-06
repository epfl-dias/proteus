/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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
#include "mmap-file.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>

#include "common/gpu/gpu-common.hpp"
#include "memory/memory-manager.hpp"
#include "topology/topology.hpp"
#include "util/timing.hpp"

// Disable by default, as, active, it does not guarantee NUMA locality!
bool allow_readwrite = false;

size_t getFileSize(const char *filename) {
  struct stat st;
  stat(filename, &st);
  return st.st_size;
}

mmap_file::mmap_file(std::string name, data_loc loc)
    : mmap_file(name, loc, ::getFileSize(name.c_str()), 0) {}

mmap_file::mmap_file(std::string name, data_loc loc, size_t bytes,
                     size_t offset = 0)
    : loc(loc), filesize(bytes) {
  time_block t("Topen (" + name + ", " + std::to_string(offset) + ":" +
               std::to_string(offset + filesize) + "): ");
  readonly = false;

  size_t real_filesize = ::getFileSize(name.c_str());
  assert(offset + filesize <= real_filesize);

  fd = -1;
  if (allow_readwrite) fd = open(name.c_str(), O_RDWR, 0);

  if (fd == -1) {
    fd = open(name.c_str(), O_RDONLY, 0);
    readonly = true;
    if (fd != -1) LOG(INFO) << "Opening file " << name << " as read-only";
  }

  if (fd == -1) {
    std::string msg("[Storage: ] Failed to open input file " + name);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  // Execute mmap
  {
    time_block t("Tmmap: ");
    data =
        mmap(nullptr, filesize, PROT_READ | (readonly ? 0 : PROT_WRITE),
             (readonly ? MAP_PRIVATE : MAP_SHARED) | MAP_POPULATE, fd, offset);
    assert(data != MAP_FAILED);
  }

  // gpu_run(cudaHostRegister(data, filesize, 0));
  if (loc == PINNED) {
    if (readonly) {
      void *data2;
      {
        time_block t("Talloc-readonly: ");

        data2 = MemoryManager::mallocPinned(filesize);
      }
      memcpy(data2, data, filesize);
      munmap(data, filesize);
      close(fd);
      data = data2;
      gpu_data = data;
    } else {
      time_block t("Talloc: ");
      gpu_data = NUMAPinnedMemAllocator::reg(data, filesize);
    }
  } else {
    gpu_data = data;
  }

  if (loc == GPU_RESIDENT) {
    std::cout << "Dataset on device: "
              << topology::getInstance().getActiveGpu().id << std::endl;
    gpu_data = MemoryManager::mallocGpu(filesize);
    gpu_run(cudaMemcpy(gpu_data, data, filesize, cudaMemcpyDefault));
    munmap(data, filesize);
    close(fd);
  }
}

mmap_file::~mmap_file() {
  if (loc == GPU_RESIDENT) gpu_run(cudaFree(gpu_data));

  // gpu_run(cudaHostUnregister(data));
  // if (loc == PINNED)       gpu_run(cudaFreeHost(data));
  if (loc == PINNED) {
    if (readonly) {
      MemoryManager::freePinned(data);
    } else {
      NUMAPinnedMemAllocator::unreg(gpu_data);
      munmap(data, filesize);
      close(fd);
    }
  }

  if (loc == PAGEABLE) {
    munmap(data, filesize);
    close(fd);
  }
}

const void *mmap_file::getData() const { return gpu_data; }

size_t mmap_file::getFileSize() const { return filesize; }
