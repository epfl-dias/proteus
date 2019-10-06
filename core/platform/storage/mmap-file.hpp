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

#include <string>

enum data_loc {
  GPU_RESIDENT,
  PINNED,
  PAGEABLE,
  ALLSOCKETS,
  ALLGPUS,
  EVERYWHERE,
};

struct mmap_file {
 private:
  int fd;

  size_t filesize;
  void *data;
  void *gpu_data;
  data_loc loc;

  bool readonly;

 public:
  mmap_file(std::string name, data_loc loc = GPU_RESIDENT);
  mmap_file(std::string name, data_loc loc, size_t bytes, size_t offset);
  ~mmap_file();

  const void *getData() const;
  size_t getFileSize() const;
};

size_t getFileSize(const char *filename);

#endif /* MMAP_FILE_HPP_ */
