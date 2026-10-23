/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_PARALLEL_CONTEXT_HPP_
#define PROTEUS_PARALLEL_CONTEXT_HPP_

#include <codegen/context/parallel-context.hpp>

class OlapParallelContext : public ParallelContext {
 public:
  explicit OlapParallelContext(const std::string &moduleName,
                               bool gpuRoot = false);

 private:
  friend class DeviceCross;
  friend class CpuToGpu;
  friend class GpuToCpu;
};

#endif /* PROTEUS_PARALLEL_CONTEXT_HPP_ */
