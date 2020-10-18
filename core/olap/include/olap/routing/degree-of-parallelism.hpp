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

#ifndef DEGREE_OF_PARALLELISM_HPP_
#define DEGREE_OF_PARALLELISM_HPP_

#include <cstddef>

class DegreeOfParallelism {
 private:
  size_t dop;

  friend class Router;

 public:
  explicit DegreeOfParallelism(size_t dop) : dop(dop) {}

  operator size_t() const { return dop; }

  inline bool operator==(const DegreeOfParallelism &o) const {
    return dop == o.dop;
  }
};

#endif /* DEGREE_OF_PARALLELISM_HPP_ */
