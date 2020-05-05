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

#include "common/common.hpp"

#include <common/olap-common.hpp>
#include <util/jit/pipeline.hpp>

namespace proteus {

class olap::impl {
  platform p;

 public:
  impl(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
       size_t log_buffers)
      : p(gpu_mem_pool_percentage, cpu_mem_pool_percentage, log_buffers) {
    LOG(INFO) << "Initializing codegen...";

    PipelineGen::init();
  }

  ~impl() = default;
};

olap::olap(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
           size_t log_buffers)
    : p_impl(std::make_unique<olap::impl>(
          gpu_mem_pool_percentage, cpu_mem_pool_percentage, log_buffers)) {}

olap::~olap() = default;

}  // namespace proteus
