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

#include "plan/query-result.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <iostream>

#include "common/error-handling.hpp"

QueryResult::QueryResult(const std::string &q) : q(q) {
  int fd = shm_open(q.c_str(), O_RDONLY, S_IRWXU);

  struct stat statbuf;
  fstat(fd, &statbuf);

  fsize = statbuf.st_size;
  resultBuf =
      (char *)mmap(nullptr, fsize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  assert(resultBuf != MAP_FAILED);
}

QueryResult::~QueryResult() {
  if (!resultBuf) return;  // object has been moved

  munmap(resultBuf, fsize);
  // we can now free the pointer, mmap will keep the file open
  shm_unlink(q.c_str());
}

std::ostream &operator<<(std::ostream &out, const QueryResult &qr) {
  out.write(qr.resultBuf, sizeof(char) * qr.fsize);
  return out;
}

QueryResult::QueryResult(QueryResult &&o)
    : fsize(o.fsize), resultBuf(o.resultBuf), q(o.q) {
  o.resultBuf = nullptr;
}
