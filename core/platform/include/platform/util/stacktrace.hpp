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

#ifndef PROTEUS_STACKTRACE_HPP
#define PROTEUS_STACKTRACE_HPP

namespace proteus {
class stacktrace;
}

std::ostream &operator<<(std::ostream &out, const proteus::stacktrace &trace);

namespace proteus {

class stacktrace {
  static constexpr size_t backtrace_limit = 32;
  void *trace[backtrace_limit]{};
  size_t backtrace_size;

 public:
  stacktrace();

  friend std::ostream & ::operator<<(std::ostream &out,
                                     const stacktrace &trace);
};

}  // namespace proteus

#endif /* PROTEUS_STACKTRACE_HPP */
