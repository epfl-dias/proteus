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
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <platform/common/error-handling.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/storage/mmap-file.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>

// Disable by default, as, active, it does not guarantee NUMA locality!
static bool allow_readwrite = false;

size_t getFileSize(const char *filename) {
  struct stat st {};
  linux_run(stat(filename, &st));
  return st.st_size;
}

mmap_file::mmap_file(std::string name, data_loc loc)
    : mmap_file(name, loc, ::getFileSize(name.c_str()), 0) {}

mmap_file::mmap_file(std::string name, data_loc loc, size_t bytes,
                     size_t offset = 0)
    : loc(loc) {
  size_t filesize = bytes;

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
    data =
        mmap(nullptr, filesize, PROT_READ | (readonly ? 0 : PROT_WRITE),
             (readonly ? MAP_PRIVATE : MAP_SHARED) | MAP_POPULATE, fd, offset);
    assert(data != MAP_FAILED);
  }

  void *gpu_data2;

  // gpu_run(cudaHostRegister(data, filesize, 0));
  if (loc == PINNED) {
    if (readonly) {
      void *data2 = MemoryManager::mallocPinned(filesize);
      memcpy(data2, data, filesize);
      munmap(data, filesize);
      close(fd);
      data = data2;
      gpu_data2 = data;
    } else {
      gpu_data2 = NUMAPinnedMemAllocator::reg(data, filesize);
    }
  } else {
    gpu_data2 = data;
  }

  if (loc == GPU_RESIDENT) {
    gpu_data2 = MemoryManager::mallocGpu(filesize);
    gpu_run(cudaMemcpy(gpu_data2, data, filesize, cudaMemcpyDefault));
    munmap(data, filesize);
    close(fd);
  }
  gpu_data = std::span<std::byte>((std::byte *)gpu_data2, filesize);
}

mmap_file::mmap_file(mmap_file &&other) noexcept
    : fd(other.fd),
      data(other.data),
      gpu_data(other.gpu_data),
      loc(other.loc),
      readonly(other.readonly) {
  other.gpu_data = std::span<std::byte>{};
  other.data = nullptr;
}

mmap_file &mmap_file::operator=(mmap_file &&other) noexcept {
  fd = other.fd;
  data = other.data;
  gpu_data = other.gpu_data;
  loc = other.loc;
  readonly = other.readonly;
  other.gpu_data = std::span<std::byte>{};
  other.data = nullptr;
  return *this;
}

mmap_file::~mmap_file() {
  if (gpu_data.empty() && !data) return;
  if (loc == GPU_RESIDENT) gpu_run(cudaFree(gpu_data.data()));

  // gpu_run(cudaHostUnregister(data));
  // if (loc == PINNED)       gpu_run(cudaFreeHost(data));
  if (loc == PINNED) {
    if (readonly) {
      MemoryManager::freePinned(data);
    } else {
      NUMAPinnedMemAllocator::unreg(gpu_data.data());
      munmap(data, gpu_data.size());
      close(fd);
    }
  }

  if (loc == PAGEABLE) {
    munmap(data, gpu_data.size());
    close(fd);
  }
}

const std::span<std::byte> &mmap_file::asSpan() const { return gpu_data; }
std::span<std::byte> &mmap_file::asSpan() { return gpu_data; }

const void *mmap_file::getData() const { return gpu_data.data(); }

size_t mmap_file::getFileSize() const { return gpu_data.size(); }
