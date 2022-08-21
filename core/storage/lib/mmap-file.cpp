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
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <storage/mmap-file.hpp>

// Disable by default, as, active, it does not guarantee NUMA locality!
static bool allow_readwrite = false;

size_t getFileSize(const char *filename) {
  struct stat st {};
  try {
    linux_run(stat(filename, &st));
  } catch (const std::runtime_error &e) {
    std::filesystem::filesystem_error error{
        e.what(), filename, std::error_code{1, std::system_category()}};
    throw error;
  }
  return st.st_size;
}

mmap_file::mmap_file(std::string name, data_loc loc)
    : mmap_file(name, loc, ::getFileSize(name.c_str()), 0) {}

class mmap_failed : public proteus::internal_error {
 public:
  using proteus::internal_error::internal_error;
};

static auto mmap_checked(void *addr, size_t len, int prot, int flags, int fd,
                         size_t offset) {
  auto ret = mmap(addr, len, prot, flags, fd, offset);

  if (ret == MAP_FAILED) {
    auto err = errno;
    throw mmap_failed{"mmap failed. errno: " + std::to_string(err) +
                      ". Message:  " + strerror(err)};
  }

  return ret;
};

mmap_file::mmap_file(std::string name, data_loc loc, size_t bytes,
                     size_t offset = 0)
    : loc(loc) {
  size_t filesize = bytes;

  //  time_block t("Topen (" + name + ", " + std::to_string(offset) + ":" +
  //               std::to_string(offset + filesize) + "): ");
  readonly = false;

  size_t real_filesize = ::getFileSize(name.c_str());
  assert(offset + filesize <= real_filesize);

  fd = -1;
  if (allow_readwrite) fd = open(name.c_str(), O_RDWR, 0);

  if ((fd == -1) && (loc != VIRTUAL)) {
    fd = open(name.c_str(), O_RDONLY, 0);
    readonly = true;
    if (fd != -1) LOG(INFO) << "Opening file " << name << " as read-only";
  }

  if ((fd == -1) && (loc != VIRTUAL)) {
    std::string msg("[Storage: ] Failed to open input file " + name);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  // Execute mmap
  /**
   * When we load virtual, we don't actually load any data now. We reserve
   * memory in the virtual address space but don't have any memory backing it
   * yet. If anyone accesses it they will segfault.
   */
  if (loc == VIRTUAL) {
    assert(fd == -1);
    //      LOG(INFO)<< "offset is: " << offset;
    data = mmap_checked(nullptr, filesize, PROT_NONE,
                        MAP_ANONYMOUS | MAP_PRIVATE, fd, offset);
  } else if (loc != PINNED || !readonly) {
    data = mmap_checked(
        nullptr, filesize, PROT_READ | (readonly ? 0 : PROT_WRITE),
        (readonly ? MAP_PRIVATE : MAP_SHARED) | MAP_POPULATE, fd, offset);
  } else {
    data = nullptr;
  }

  void *gpu_data2;

  // gpu_run(cudaHostRegister(data, filesize, 0));
  if (loc == PINNED) {
    if (readonly) {
      void *data2 = MemoryManager::mallocPinned(filesize);

      auto readall = [](int pfd, void *p, size_t size, size_t offs) {
        auto buff = static_cast<char *>(p);
        while (size > 0) {
          auto rdsize = pread(pfd, buff, size, offs);
          if (rdsize < 0) linux_run(rdsize);
          LOG_IF(FATAL, rdsize == 0) << "Unexpected EOF";
          buff += rdsize;
          offs += rdsize;
          size -= rdsize;
        }
      };

      readall(fd, data2, filesize, offset);
      //      memcpy(data2, data, filesize);
      //      munmap(data, filesize);
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
  actual_data = std::span<std::byte>((std::byte *)gpu_data2, filesize);
}

mmap_file::mmap_file(mmap_file &&other) noexcept
    : fd(other.fd), data(other.data), loc(other.loc), readonly(other.readonly) {
  actual_data = other.actual_data;
  other.actual_data = std::span<std::byte>{};
  other.data = nullptr;
}

mmap_file &mmap_file::operator=(mmap_file &&other) noexcept {
  fd = other.fd;
  data = other.data;
  actual_data = other.actual_data;
  loc = other.loc;
  readonly = other.readonly;
  other.actual_data = std::span<std::byte>{};
  other.data = nullptr;
  return *this;
}

mmap_file::mmap_file(void *ptr, size_t bytes) : loc(MANAGEDMEMORY), data(ptr) {
  actual_data = std::span<std::byte>((std::byte *)ptr, bytes);
}

mmap_file::~mmap_file() {
  if (actual_data.empty() && !data) return;
  if (loc == GPU_RESIDENT) MemoryManager::freeGpu(actual_data.data());

  // gpu_run(cudaHostUnregister(data));
  // if (loc == PINNED)       gpu_run(cudaFreeHost(data));
  if (loc == PINNED) {
    if (readonly) {
      MemoryManager::freePinned(data);
    } else {
      NUMAPinnedMemAllocator::unreg(actual_data.data());
      munmap(data, actual_data.size());
      close(fd);
    }
  }

  if (loc == PAGEABLE) {
    munmap(data, actual_data.size());
    close(fd);
  }

  if (loc == MANAGEDMEMORY) {
    if (topology::getInstance().getGpuAddressed(data)) {
      MemoryManager::freeGpu(data);
    } else {
      MemoryManager::freePinned(data);
    }
  }
}

const std::span<std::byte> &mem_region::asSpan() const { return actual_data; }
std::span<std::byte> &mem_region::asSpan() { return actual_data; }

const void *mem_region::getData() const { return actual_data.data(); }

size_t mem_region::getFileSize() const { return actual_data.size(); }

remote_mem_region::remote_mem_region(void *data, size_t size, size_t srv_id,
                                     std::function<void()> release)
    : srv_id(srv_id), release(std::move(release)) {
  actual_data = std::span<std::byte>((std::byte *)data, size);
}

[[nodiscard]] bool remote_mem_region::isServerLocalRegion() const {
  return srv_id == InfiniBandManager::server_id();
}

remote_mem_region::~remote_mem_region() { release(); }
