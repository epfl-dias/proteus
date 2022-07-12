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

#ifndef MMAP_FILE_HPP_
#define MMAP_FILE_HPP_

#include <span>
#include <string>

enum data_loc {
  GPU_RESIDENT,
  PINNED,
  PAGEABLE,
  ALLSOCKETS,
  ALLGPUS,
  EVERYWHERE,
  DISTRIBUTED,
  FROM_REGISTRY,
  VIRTUAL,
  MANAGEDMEMORY
};

struct mmap_file {
 private:
  int fd;

  void *data;

  std::span<std::byte> gpu_data;

  data_loc loc;

  bool readonly;

 public:
  mmap_file(std::string name, data_loc loc);
  mmap_file(std::string name, data_loc loc, size_t bytes, size_t offset);
  mmap_file(void *ptr, size_t bytes);
  mmap_file(const mmap_file &) = delete;
  mmap_file &operator=(const mmap_file &) = delete;
  mmap_file(mmap_file &&) noexcept;
  mmap_file &operator=(mmap_file &&) noexcept;
  ~mmap_file();

  [[nodiscard]] const std::span<std::byte> &asSpan() const;
  [[nodiscard]] std::span<std::byte> &asSpan();

  [[nodiscard]] const void *getData() const;
  [[nodiscard]] size_t getFileSize() const;

  [[nodiscard]] static mmap_file from(std::string blob);
};

size_t getFileSize(const char *filename);

#endif /* MMAP_FILE_HPP_ */
