/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include "olap/plan/query-result.hpp"

#include <fcntl.h>
#include <sys/mman.h>

#include <cassert>
#include <iostream>
#include <platform/common/error-handling.hpp>
#include <thread>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
// TODO: remove as soon as the default GCC moves filesystem out of experimental
//  GCC 8.3 has made the transition, but the default GCC in Ubuntu 18.04 is 7.4
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

using namespace std::chrono_literals;

QueryResult::QueryResult(const std::string &outputfile)
    : outputfile(outputfile) {
  // if the output path is relative we assume it is in shared memory
  // this is the default case.
  std::filesystem::path output_path{outputfile};
  // mmap shared memory
  if (output_path.is_relative()) {
    auto p = "/dev/shm" / std::filesystem::path{outputfile};
    assert(std::filesystem::exists(p));

    int fd = linux_run(shm_open(outputfile.c_str(), O_RDONLY, S_IRWXU));

    if (std::filesystem::is_directory(p)) {
      fsize = 0;
    } else {
      fsize = std::filesystem::file_size(p);
    }

    if (fsize) {
      resultBuf = (char *)mmap(nullptr, fsize, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE, fd, 0);
      if (resultBuf == MAP_FAILED) {
        auto msg =
            std::string{"Opening result file failed ("} + strerror(errno) + ")";
        LOG(ERROR) << msg;
        throw std::runtime_error{msg};
      }
      assert(resultBuf != MAP_FAILED);
    } else {
      resultBuf = static_cast<char *>(MAP_FAILED);
      assert(resultBuf && "Destructor assumes MAP_FAILED != 0");
    }
    close(fd);  // close the descriptor produced by the shm_open
  } else {
    // mmap regular file
    if (std::filesystem::is_directory(output_path)) {
      fsize = 0;
    } else {
      fsize = std::filesystem::file_size(output_path);
    }
    int fd = open(output_path.c_str(), O_RDONLY, 0);
    resultBuf = (char *)mmap(nullptr, fsize, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE, fd, 0);
    close(fd);
  }
}

QueryResult::~QueryResult() {
  if (!resultBuf) return;  // object has been moved

  if (fsize) munmap(resultBuf, fsize);

  std::filesystem::path output_path{outputfile};

  // We can now unlink the file from the filesystem if it is in shared memory
  if (std::filesystem::is_directory(output_path) ||
      !output_path.is_relative()) {
    LOG(INFO) << "Not unlinking " << outputfile;
  } else {
    auto x = shm_unlink(outputfile.c_str());
    if (x < 0 &&
        errno == EISDIR) {  // std::filesystem may not know the file type
      LOG(INFO) << "Not unlinking dir " << outputfile;
    } else {
      linux_run(x);
    }
  }
}

std::ostream &operator<<(std::ostream &out, const QueryResult &qr) {
  if (qr.fsize) out.write(qr.resultBuf, sizeof(char) * qr.fsize);
  return out;
}

QueryResult::QueryResult(QueryResult &&o)
    : fsize(o.fsize), resultBuf(o.resultBuf), outputfile(o.outputfile) {
  o.resultBuf = nullptr;
}
