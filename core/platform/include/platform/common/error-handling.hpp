/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef ERROR_HANDLING_HPP_
#define ERROR_HANDLING_HPP_

#include <glog/logging.h>

#include <platform/common/common.hpp>

namespace errorhanding {
[[noreturn]] inline void failedLinuxRun(const char *str, const char *file,
                                        int line) {
  auto msg = std::string{str} + " failed (" + strerror(errno) + ")";
  google::LogMessage(file, line, google::GLOG_ERROR).stream() << msg;
  throw std::runtime_error{msg};
}

inline int assertLinuxRun(int x, const char *str, const char *file, int line) {
  if (unlikely(x < 0)) failedLinuxRun(str, file, line);
  return x;
}

template <typename T>
inline T *assertLinuxRun(T *x, const char *str, const char *file, int line) {
  if (unlikely(x == nullptr)) failedLinuxRun(str, file, line);
  return x;
}
}  // namespace errorhanding

#define linux_run(x) (errorhanding::assertLinuxRun(x, #x, __FILE__, __LINE__))

#endif /* ERROR_HANDLING_HPP_ */
