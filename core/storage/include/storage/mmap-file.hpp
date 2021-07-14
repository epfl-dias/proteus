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

class mem_region {
 protected:
  std::span<std::byte> actual_data;

 public:
  virtual ~mem_region() = default;

  [[nodiscard]] virtual const std::span<std::byte> &asSpan() const;
  [[nodiscard]] virtual std::span<std::byte> &asSpan();

  [[nodiscard]] virtual const void *getData() const;
  [[nodiscard]] virtual size_t getFileSize() const;

  [[nodiscard]] virtual bool isServerLocalRegion() const { return true; }
};

class mmap_file : public mem_region {
 private:
  int fd;

  void *data;

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
  ~mmap_file() override;

  [[nodiscard]] static mmap_file from(std::string blob);
};

class remote_mem_region : public mem_region {
  size_t srv_id;
  std::function<void()> release;

 public:
  remote_mem_region(void *data, size_t size, size_t srv_id,
                    std::function<void()> release);
  remote_mem_region(const remote_mem_region &) = delete;
  remote_mem_region &operator=(const remote_mem_region &) = delete;
  remote_mem_region(remote_mem_region &&) noexcept = default;
  remote_mem_region &operator=(remote_mem_region &&) noexcept = default;
  ~remote_mem_region() override;

  [[nodiscard]] bool isServerLocalRegion() const override;
  [[nodiscard]] size_t getServerId() const { return srv_id; }
};

size_t getFileSize(const char *filename);

#endif /* MMAP_FILE_HPP_ */
