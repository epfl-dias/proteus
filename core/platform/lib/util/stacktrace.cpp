/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <execinfo.h>

#include <ostream>
#include <platform/util/stacktrace.hpp>

namespace proteus {
stacktrace::stacktrace() : backtrace_size(backtrace(trace, backtrace_limit)) {}
}  // namespace proteus

std::ostream &operator<<(std::ostream &out, const proteus::stacktrace &strace) {
  char **trace = backtrace_symbols(strace.trace, strace.backtrace_size);
  out << "trace: ";
  for (size_t i = 0; i < strace.backtrace_size; ++i) {
    out << "\n\t" << trace[i];
  }
  return out;
}
