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

#ifndef GPU_HASH_JOIN_CHAINED_HPP_
#define GPU_HASH_JOIN_CHAINED_HPP_

#include "operators/hash-join-chained.hpp"

class GpuHashJoinChained : public HashJoinChained {
 public:
  // inherit constructor
  using HashJoinChained::HashJoinChained;

  void open_build(Pipeline *pip);
  void close_probe(Pipeline *pip);
  void close_build(Pipeline *pip);
};

#endif /* GPU_HASH_JOIN_CHAINED_HPP_ */
